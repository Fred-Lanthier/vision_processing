#!/usr/bin/env python3
"""Via-point detour router (PP pipeline).

Sits between the FM planner and the trajectory executor:

    casf_generative_node_PC_pp -> /planner/raw_nominal_trajectory
        -> THIS NODE -> /planner/nominal_trajectory -> trajectory_executor

Responsibility split (see thesis §architecture): the FM policy owns the task
skill, the Bernstein CBF owns 100 Hz safety, and NEITHER can decide the
homotopy class of the path around a large/non-convex obstacle -- a reactive
safety filter provably has stuck equilibria there (runPP1002.3ag: the escape
climbed a 1 m wall to the kinematic ceiling and parked). This node owns that
single discrete decision and nothing else: when the FM trajectory's TCP path
is blocked by the obstacle cloud, it enumerates the ways around from the
cloud's actual extents (around-left / around-right / over-the-top), scores
them by reachability + leg clearance + detour length, COMMITS to the winner
(sticky across 3 Hz replans, no side-flipping), and splices the blocked
segment of the JOINT trajectory through the via point (task-space
interpolation + fast_ik). When nothing is blocked -- the common case -- the
trajectory passes through untouched, so the executed motion stays 100%
FM + CBF. Deterministic: no sampling anywhere; same scene -> same detour.
"""
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import rospy
import xacro
from sensor_msgs.msg import JointState, PointCloud2
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker

from vision_processing import fast_ik_module


# ---------------------------------------------------------------------------
# Pure geometry helpers (unit-testable without ROS)
# ---------------------------------------------------------------------------

def rpy_to_mat(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), \
        np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def axis_angle_mat(axis, angle):
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12)) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


def quat_to_mat(q):
    x, y, z, w = q / max(np.linalg.norm(q), 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp_mat(R0, R1, f):
    q0, q1 = mat_to_quat(R0), mat_to_quat(R1)
    if q0 @ q1 < 0.0:
        q1 = -q1
    d = np.clip(q0 @ q1, -1.0, 1.0)
    if d > 0.9995:
        q = q0 + f * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - f) * th) * q0 + np.sin(f * th) * q1) / np.sin(th)
    return quat_to_mat(q)


def min_clearance(points, cloud):
    """Min distance of each point [N,3] to the cloud [M,3] -> [N]."""
    d = np.linalg.norm(points[:, None, :] - cloud[None, :, :], axis=-1)
    return d.min(axis=1)


def segment_samples(p0, p1, step):
    n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / max(step, 1e-4))) + 1)
    f = np.linspace(0.0, 1.0, n)[:, None]
    return p0[None, :] * (1.0 - f) + p1[None, :] * f


def find_blocked_span(P, cloud, margin, step):
    """First/last blocked original-waypoint segment along the TCP polyline
    P [N,3]. Returns (a, rejoin, tail_clear) or None. rejoin is the first
    waypoint at/after the last blocked segment whose own clearance is >=
    margin; tail_clear=False means no such waypoint exists -- the FM's
    horizon ENDS inside the obstacle region (rejoin = last waypoint,
    blocked), so a detour cannot demand a clear final leg."""
    blocked = []
    for i in range(len(P) - 1):
        s = segment_samples(P[i], P[i + 1], step)
        if min_clearance(s, cloud).min() < margin:
            blocked.append(i)
    if not blocked:
        return None
    a = blocked[0]
    b = blocked[-1]
    wp_clear = min_clearance(P, cloud)
    for j in range(b + 1, len(P)):
        if wp_clear[j] >= margin:
            return a, j, True
    return a, len(P) - 1, False


def via_candidates(p_entry, p_rejoin, cloud, near_radius, via_margin, z_min):
    """Enumerate the deterministic routes around the blocking cloud region:
    around-left / around-right of the entry->rejoin direction, and over the
    top. Each route is a CHAIN of up to two via points at the near/far edges
    of the obstacle's along-path extent (visibility-graph style) -- a single
    midpoint via would cut the corners: its straight legs graze the
    obstacle's edges. Returns [(name, [xyz, ...])].

    The 'near' set = cloud points within near_radius of the BLOCKED PATH
    polyline (not a ball around its midpoint): a second obstacle standing
    off to the side (e.g. the static sphere) must not stretch the perceived
    extents of the one actually blocking the path."""
    path = segment_samples(p_entry, p_rejoin, 0.03)          # [S,3]
    d = np.linalg.norm(cloud[:, None, :] - path[None, :, :],
                       axis=-1).min(axis=1)                  # [M]
    near = cloud[d < near_radius]
    if near.shape[0] == 0:
        return []
    m = 0.5 * (p_entry + p_rejoin)
    u = p_rejoin - p_entry
    u[2] = 0.0
    un = np.linalg.norm(u)
    if un < 1e-6:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = u / un
    lat = np.array([-u[1], u[0], 0.0])
    s = (near - m[None, :]) @ lat        # lateral coords of the obstacle
    t = (near - m[None, :]) @ u          # along-path coords
    # Along-path extent of the obstacle, padded so the corner clearance at
    # the via points is ~via_margin as well.
    t_lo, t_hi = float(t.min()) - 0.5 * via_margin, \
        float(t.max()) + 0.5 * via_margin
    z_pass = max(0.5 * (p_entry[2] + p_rejoin[2]), z_min)

    def chain(base_fn):
        pts = [base_fn(t_lo), base_fn(t_hi)]
        if np.linalg.norm(pts[1] - pts[0]) < 0.04:
            pts = [base_fn(0.5 * (t_lo + t_hi))]
        return pts

    cands = []
    for name, off in (("around_left", float(s.max()) + via_margin),
                      ("around_right", float(s.min()) - via_margin)):
        def mk(tv, off=off):
            c = m + off * lat + tv * u
            c[2] = z_pass
            return c
        cands.append((name, chain(mk)))
    z_top = float(near[:, 2].max()) + via_margin

    def mk_top(tv):
        c = m + tv * u
        c[2] = z_top
        return c
    cands.append(("over_top", chain(mk_top)))
    return cands


def route_clearance(waypoints, cloud, step):
    """Min clearance over the whole polyline entry -> vias -> rejoin."""
    cl = np.inf
    for p0, p1 in zip(waypoints[:-1], waypoints[1:]):
        cl = min(cl, leg_clearance(p0, p1, cloud, step))
    return cl


def leg_clearance(p0, p1, cloud, step):
    return float(min_clearance(segment_samples(p0, p1, step), cloud).min())


# ---------------------------------------------------------------------------
# URDF chain FK (numpy; same URDF the IK solver loads)
# ---------------------------------------------------------------------------

class ChainFK:
    """Fixed+revolute chain base->tip parsed from a URDF string. FK matches
    pinocchio on the same file (verified against fast_ik IK round-trip)."""

    def __init__(self, urdf_xml, tip_link, base_link="panda_link0"):
        root = ET.fromstring(urdf_xml)
        joints = {}
        for j in root.iter("joint"):
            # skip <joint> tags inside <transmission>/<gazebo> blocks
            if j.find("child") is None or j.find("parent") is None:
                continue
            child = j.find("child").attrib["link"]
            origin = j.find("origin")
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            if origin is not None:
                xyz = np.array([float(v) for v in origin.attrib.get(
                    "xyz", "0 0 0").split()])
                rpy = np.array([float(v) for v in origin.attrib.get(
                    "rpy", "0 0 0").split()])
            axis = j.find("axis")
            ax = (np.array([float(v) for v in axis.attrib["xyz"].split()])
                  if axis is not None else np.array([0.0, 0.0, 1.0]))
            joints[child] = (j.find("parent").attrib["link"],
                             j.attrib.get("type", "fixed"),
                             j.attrib.get("name", ""), xyz, rpy, ax)
        limits = {}
        for j in root.iter("joint"):
            lim = j.find("limit")
            if lim is not None and "lower" in lim.attrib:
                limits[j.attrib.get("name", "")] = (
                    float(lim.attrib["lower"]), float(lim.attrib["upper"]))
        chain = []
        link = tip_link
        while link != base_link:
            if link not in joints:
                raise ValueError("no joint chain %s -> %s (stuck at %s)"
                                 % (base_link, tip_link, link))
            parent, jtype, name, xyz, rpy, ax = joints[link]
            chain.append((jtype, name, xyz, rpy, ax))
            link = parent
        chain.reverse()
        self.chain = chain
        self.revolute_names = [c[1] for c in chain
                               if c[0] in ("revolute", "continuous")]
        self.limits = np.array([limits.get(n, (-1e9, 1e9))
                                for n in self.revolute_names])  # [7,2]

    def jac_pos(self, q_by_name, names, eps=1e-4):
        """Positional Jacobian [3,7] by finite differences (numpy FK)."""
        p0 = self.fk(q_by_name)[:3, 3]
        J = np.zeros((3, len(names)))
        for i, n in enumerate(names):
            qp = dict(q_by_name)
            qp[n] = qp[n] + eps
            J[:, i] = (self.fk(qp)[:3, 3] - p0) / eps
        return J

    def fk(self, q_by_name):
        T = np.eye(4)
        for jtype, name, xyz, rpy, ax in self.chain:
            A = np.eye(4)
            A[:3, :3] = rpy_to_mat(rpy)
            A[:3, 3] = xyz
            T = T @ A
            if jtype in ("revolute", "continuous"):
                R = np.eye(4)
                R[:3, :3] = axis_angle_mat(ax, q_by_name[name])
                T = T @ R
        return T


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ViapointDetourNode:
    def __init__(self):
        rospy.init_node("viapoint_detour")
        gp = rospy.get_param
        self.enabled = bool(gp("~enabled", True))
        self.in_topic = gp("~input_topic", "/planner/raw_nominal_trajectory")
        self.out_topic = gp("~output_topic", "/planner/nominal_trajectory")
        self.obs_topic = gp("~obs_topic", "/perception/cleaned_obstacles")
        self.tcp_link = gp("~tcp_link", "panda_TCP")
        urdf_path = gp("~urdf_path", "")
        # TCP-point clearance for "blocked": generous vs the CBF's whole-body
        # d_safe because this is a point check -- the hand needs ~a gripper
        # half-width of corridor.
        self.block_margin = float(gp("~block_margin", 0.10))
        self.via_margin = float(gp("~via_margin", 0.15))
        self.leg_margin = float(gp("~leg_margin", 0.09))
        self.sample_step = float(gp("~sample_step", 0.02))
        # Max distance from the BLOCKED PATH for a cloud point to count as
        # part of the blocking obstacle (path-corridor, not a midpoint ball:
        # keeps the off-path sphere from stretching the extents).
        self.near_radius = float(gp("~near_radius", 0.30))
        self.reach_min = float(gp("~reach_min", 0.25))
        self.reach_max = float(gp("~reach_max", 0.72))
        self.z_min = float(gp("~z_min", 0.08))
        self.z_max = float(gp("~z_max", 0.85))
        self.pass_radius = float(gp("~pass_radius", 0.07))
        self.commit_clear_time = float(gp("~commit_clear_time", 2.0))
        # Stall re-vote: the route validity check is a TCP-point corridor,
        # blind to whole-body feasibility (elbow vs CBF, joint limits). If
        # the TCP gets no closer to the current via by stall_min_progress
        # over stall_timeout seconds while a route is committed, the route
        # is declared stalled: tabu'd for stall_tabu_ttl and re-voted, so
        # the next-best side gets a turn. 0 = off.
        self.stall_timeout = float(gp("~stall_timeout", 4.0))
        self.stall_min_progress = float(gp("~stall_min_progress", 0.02))
        self.stall_tabu_ttl = float(gp("~stall_tabu_ttl", 12.0))
        # Reactive-first activation gate: 0 = route the FIRST blocked plan
        # (plan-first, aggressive). > 0 = while plans are blocked, keep
        # passing through and let the reactive stack (CBF slide +
        # q_dot_clear + escape nudge) work; only commit a route once the TCP
        # has moved less than activation_min_progress for this many seconds
        # (= genuinely stuck in the local minimum). A low prism the CBF can
        # graze over never engages the router; a tall wall pins the TCP and
        # engages it after the timeout.
        self.activation_stuck_time = float(gp("~activation_stuck_time", 0.0))
        self.activation_min_progress = float(gp(
            "~activation_min_progress", 0.03))
        self._act_anchor = None   # TCP anchor for stuck detection
        self._act_t0 = None
        self.joint_speed = float(gp("~retime_joint_speed", 0.3))
        self.min_dt = float(gp("~retime_min_dt", 0.05))
        self.ik_pos_tol = float(gp("~ik_pos_tol", 0.03))
        # Kinematic-quality gate on the IK'd detour (checked at splice time,
        # every check_stride-th waypoint): reject the route -- and tabu it,
        # so the vote tries the other side -- if any solution comes within
        # limit_margin of a joint limit or the positional Jacobian's min
        # singular value drops below sing_min_sigma (outstretched / wrist
        # singularity). Straight task-space legs + slerped orientation can
        # otherwise drag the arm through singular regions the geometric
        # clearance vote cannot see.
        self.sing_min_sigma = float(gp("~sing_min_sigma", 0.04))
        self.limit_margin_rad = float(gp("~limit_margin_rad", 0.05))
        self.quality_check_stride = max(1, int(gp("~quality_check_stride",
                                                  2)))
        self.publish_viz = bool(gp("~publish_viz", True))

        urdf_xml = xacro.process_file(urdf_path).toxml()
        fd, self._urdf_tmp = tempfile.mkstemp(suffix=".urdf")
        with os.fdopen(fd, "w") as f:
            f.write(urdf_xml)
        self.ik = fast_ik_module.FastIK(self._urdf_tmp, self.tcp_link)
        self.nq = int(self.ik.get_nq())
        self.fkc = ChainFK(urdf_xml, self.tcp_link)

        self.cloud = None
        self.q_now = None
        self._vias = None         # committed via chain [list of world [3]]
        self._via_name = ""
        self._last_blocked_t = 0.0
        self._route_tabu = {}     # route name -> tabu expiry (wall time)
        self._stall_t0 = None     # progress-window start
        self._stall_best = None   # best (smallest) TCP->via distance seen

        self.pub = rospy.Publisher(self.out_topic, JointTrajectory,
                                   queue_size=1)
        self.viz_pub = (rospy.Publisher("/viz/viapoint_detour", Marker,
                                        queue_size=2)
                        if self.publish_viz else None)
        rospy.Subscriber(self.obs_topic, PointCloud2, self._obs_cb,
                         queue_size=1)
        rospy.Subscriber("/joint_states", JointState, self._js_cb,
                         queue_size=1)
        rospy.Subscriber(self.in_topic, JointTrajectory, self._traj_cb,
                         queue_size=1)
        rospy.loginfo("viapoint_detour: %s -> %s (enabled=%s, block_margin="
                      "%.2f, via_margin=%.2f)", self.in_topic, self.out_topic,
                      self.enabled, self.block_margin, self.via_margin)

    # ------------------------------------------------------------------ io
    def _obs_cb(self, msg):
        try:
            pts = np.frombuffer(msg.data, dtype=np.float32).reshape(
                -1, msg.point_step // 4)[:, :3]
            self.cloud = pts.astype(np.float64).copy()
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "viapoint_detour: bad cloud: %s", exc)

    def _js_cb(self, msg):
        if len(msg.position) >= 7:
            self.q_now = np.asarray(msg.position[:7], dtype=np.float64)

    # ------------------------------------------------------------------ fk
    def _fk_traj(self, joint_names, Q):
        out_p = np.zeros((len(Q), 3))
        out_R = [None] * len(Q)
        for i, q in enumerate(Q):
            by = dict(zip(joint_names, q))
            T = self.fkc.fk(by)
            out_p[i] = T[:3, 3]
            out_R[i] = T[:3, :3]
        return out_p, out_R

    # ---------------------------------------------------------------- core
    def _traj_cb(self, msg):
        if not self.enabled or self.cloud is None or len(msg.points) < 2:
            self.pub.publish(msg)
            return
        try:
            out = self._route(msg)
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0, "viapoint_detour: routing failed (%s), passing through",
                exc)
            out = msg
        self.pub.publish(out)

    def _route(self, msg):
        cloud = self.cloud
        names = list(msg.joint_names[:7])
        Q = np.array([p.positions[:7] for p in msg.points])
        P, R = self._fk_traj(names, Q)
        span = find_blocked_span(P, cloud, self.block_margin,
                                 self.sample_step)
        now = rospy.get_time()
        if span is None:
            if (self._vias is not None
                    and now - self._last_blocked_t > self.commit_clear_time):
                rospy.loginfo("viapoint_detour: path clear, releasing "
                              "committed route (%s)", self._via_name)
                self._vias = None
            self._act_anchor = None  # next block episode starts fresh
            self._viz(None)
            return msg
        self._last_blocked_t = now
        a, rejoin, tail_clear = span
        p_entry, p_rejoin = P[a], P[rejoin]

        # Reactive-first gate: with no committed route, hand blocked plans
        # to the reactive stack unchanged until the robot is actually stuck
        # (TCP barely moving for activation_stuck_time while blocked).
        if (self._vias is None and self.activation_stuck_time > 0.0):
            if self.q_now is None:
                return msg
            tcp = self.fkc.fk(dict(zip(names, self.q_now)))[:3, 3]
            if (self._act_anchor is None
                    or np.linalg.norm(tcp - self._act_anchor)
                    >= self.activation_min_progress):
                self._act_anchor = tcp
                self._act_t0 = now
                return msg
            if now - self._act_t0 < self.activation_stuck_time:
                rospy.loginfo_throttle(
                    2.0, "viapoint_detour: blocked, reactive stack has "
                    "%.1f s left before router engages",
                    self.activation_stuck_time - (now - self._act_t0))
                return msg
            rospy.logwarn(
                "viapoint_detour: stuck for %.1f s (TCP moved < %.2f m) "
                "-> engaging router", self.activation_stuck_time,
                self.activation_min_progress)
        self._act_anchor = None

        # Passed-via check: pop chain heads the TCP has reached; drop the
        # route once every via is passed (any residual block re-votes).
        if self._vias is not None and self.q_now is not None:
            tcp = self.fkc.fk(dict(zip(names, self.q_now)))[:3, 3]
            while (self._vias
                   and np.linalg.norm(tcp - self._vias[0]) < self.pass_radius):
                rospy.loginfo("viapoint_detour: via of route %s passed",
                              self._via_name)
                self._vias = self._vias[1:] or None
                self._stall_t0 = None  # fresh window for the next via
            # Stall re-vote: no TCP progress toward the current via for
            # stall_timeout -> the route is whole-body infeasible even
            # though its TCP corridor is clear; tabu it and re-vote.
            if self._vias is not None and self.stall_timeout > 0.0:
                d = float(np.linalg.norm(tcp - self._vias[0]))
                if (self._stall_t0 is None
                        or d < self._stall_best - self.stall_min_progress):
                    self._stall_t0 = now
                    self._stall_best = d
                elif now - self._stall_t0 > self.stall_timeout:
                    rospy.logwarn(
                        "viapoint_detour: route %s stalled (%.3f m from via, "
                        "no progress for %.1f s) -> tabu %.0f s, re-voting",
                        self._via_name, d, self.stall_timeout,
                        self.stall_tabu_ttl)
                    self._route_tabu[self._via_name] = (
                        now + self.stall_tabu_ttl)
                    self._vias = None
                    self._stall_t0 = None

        # Sticky commit: keep the chosen route while it stays usable. With a
        # blocked horizon (tail_clear False) only the legs up to the last
        # via are required to be clear.
        vias = None
        if self._vias is not None:
            poly = [p_entry] + list(self._vias) + (
                [p_rejoin] if tail_clear else [])
            if route_clearance(poly, cloud,
                               self.sample_step) >= self.leg_margin:
                vias = self._vias
            else:
                rospy.logwarn("viapoint_detour: committed route (%s) no "
                              "longer clear, re-voting", self._via_name)
                self._vias = None
        if vias is None:
            vias = self._vote(p_entry, p_rejoin, cloud, tail_clear)
            if vias is None:
                self._viz(None)
                return msg

        out = self._splice(msg, names, Q, R, a, rejoin, vias, tail_clear)
        if out is None:
            return msg
        self._viz(vias)
        return out

    def _vote(self, p_entry, p_rejoin, cloud, tail_clear):
        """tail_clear False = the FM horizon ENDS inside the obstacle
        (runPP1004: every 16-waypoint plan terminated in the wall, so the
        via->rejoin leg could never validate and the router passed through
        forever). Then only the legs up to the last via must be clear; the
        distance last-via -> horizon-end stays in the cost as a heuristic so
        the side nearer the goal still wins."""
        cands = via_candidates(p_entry, p_rejoin, cloud, self.near_radius,
                               self.via_margin, self.z_min)
        now = rospy.get_time()
        self._route_tabu = {n: t for n, t in self._route_tabu.items()
                            if t > now}
        if self._route_tabu and all(n in self._route_tabu
                                    for n, _ in cands):
            # Every candidate stalled recently: forget and retry rather
            # than deadlock with no route at all.
            rospy.logwarn("viapoint_detour: all routes tabu'd, clearing")
            self._route_tabu = {}
        best, best_cost, best_name = None, np.inf, ""
        report = []
        for name, chain in cands:
            if name in self._route_tabu:
                report.append("%s: tabu (stalled)" % name)
                continue
            reach_ok = all(
                self.reach_min <= float(np.linalg.norm(c[:2]))
                <= self.reach_max and self.z_min <= c[2] <= self.z_max
                for c in chain)
            if not reach_ok:
                report.append("%s: unreachable" % name)
                continue
            poly = [p_entry] + list(chain) + (
                [p_rejoin] if tail_clear else [])
            cl = route_clearance(poly, cloud, self.sample_step)
            if cl < self.leg_margin:
                report.append("%s: blocked (clear=%.3f)" % (name, cl))
                continue
            cost = float(sum(np.linalg.norm(p1 - p0) for p0, p1
                             in zip(poly[:-1], poly[1:])))
            if not tail_clear:
                cost += float(np.linalg.norm(p_rejoin - chain[-1]))
            report.append("%s: OK cost=%.2f clear=%.3f" % (name, cost, cl))
            if cost < best_cost:
                best, best_cost, best_name = list(chain), cost, name
        if best is None:
            rospy.logwarn_throttle(
                2.0, "viapoint_detour: blocked but no valid route [%s]; "
                "passing through", "; ".join(report))
            return None
        rospy.loginfo("viapoint_detour: committing route %s (%d vias, "
                      "tail_clear=%s) [%s]", best_name, len(best),
                      tail_clear, "; ".join(report))
        self._vias = best
        self._via_name = best_name
        self._stall_t0 = None  # fresh progress window for the new route
        return best

    def _splice(self, msg, names, Q, R, a, rejoin, vias, tail_clear):
        # Task-space detour polyline entry -> vias (-> rejoin when the
        # horizon extends past the obstacle), orientation slerped between
        # the bounding FM waypoints. With a blocked horizon the trajectory
        # ENDS at the last via: the blocked tail is dropped, and the FM's
        # next replans (from along the detour) extend the horizon past the
        # obstacle until the normal rejoin case takes over.
        P_a = self.fkc.fk(dict(zip(names, Q[a])))[:3, 3]
        P_r = self.fkc.fk(dict(zip(names, Q[rejoin])))[:3, 3]
        step = max(self.sample_step, 0.03)
        poly = [P_a] + list(vias) + ([P_r] if tail_clear else [])
        segs = [segment_samples(p0, p1, step)
                for p0, p1 in zip(poly[:-1], poly[1:])]
        pts = np.vstack([segs[0]] + [s[1:] for s in segs[1:]])
        arc = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        arc = arc / max(arc[-1], 1e-9)
        poses = []
        for p, f in zip(pts, arc):
            T = np.eye(4)
            T[:3, :3] = slerp_mat(R[a], R[rejoin], float(f))
            T[:3, 3] = p
            poses.append(T)
        q_init = np.concatenate([Q[a], np.zeros(self.nq - 7)])
        q_det = np.asarray(self.ik.solve_batch(poses, q_init))[:, :7]
        # IK sanity + kinematic quality. Any failure TABUS the route (not
        # just pass-through): a route that IK cannot track well this cycle
        # will not track it next cycle either, and retrying forever is the
        # same trap as the stall.
        lim = self.fkc.limits
        for i, (qd, T) in enumerate(zip(q_det, poses)):
            by = dict(zip(names, qd))
            err = np.linalg.norm(self.fkc.fk(by)[:3, 3] - T[:3, 3])
            if err > self.ik_pos_tol:
                self._reject_route("IK miss %.3f m" % err)
                return None
            if i % self.quality_check_stride:
                continue
            near_lim = np.minimum(qd - lim[:, 0], lim[:, 1] - qd).min()
            if near_lim < self.limit_margin_rad:
                self._reject_route("joint limit (margin %.3f rad, joint %d)"
                                   % (near_lim, int(np.argmin(np.minimum(
                                       qd - lim[:, 0], lim[:, 1] - qd))) + 1))
                return None
            if self.sing_min_sigma > 0.0:
                sigma = np.linalg.svd(
                    self.fkc.jac_pos(by, names), compute_uv=False)[-1]
                if sigma < self.sing_min_sigma:
                    self._reject_route("near-singular (sigma_min %.3f)"
                                       % sigma)
                    return None
        if tail_clear:
            Q_out = np.vstack([Q[:a + 1], q_det[1:-1], Q[rejoin:]])
        else:
            Q_out = np.vstack([Q[:a + 1], q_det[1:]])
        out = JointTrajectory()
        out.header = msg.header
        out.joint_names = msg.joint_names
        t = 0.0
        for i, q in enumerate(Q_out):
            if i > 0:
                dq = float(np.abs(q - Q_out[i - 1]).max())
                t += max(dq / max(self.joint_speed, 1e-3), self.min_dt)
            pt = JointTrajectoryPoint()
            pt.positions = list(q)
            pt.time_from_start = rospy.Duration(t)
            out.points.append(pt)
        return out

    def _reject_route(self, reason):
        """Tabu the committed route for kinematic infeasibility and clear
        the commit, so the next cycle votes an alternative."""
        rospy.logwarn(
            "viapoint_detour: route %s rejected: %s -> tabu %.0f s, "
            "re-voting", self._via_name, reason, self.stall_tabu_ttl)
        self._route_tabu[self._via_name] = (
            rospy.get_time() + self.stall_tabu_ttl)
        self._vias = None
        self._stall_t0 = None

    # ----------------------------------------------------------------- viz
    def _viz(self, vias):
        if self.viz_pub is None:
            return
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = rospy.Time.now()
        m.ns = "viapoint"
        m.id = 0
        if not vias:
            m.action = Marker.DELETE
        else:
            m.action = Marker.ADD
            m.type = Marker.SPHERE_LIST
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.05
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 0.9
            for v in vias:
                m.points.append(_pt(v))
        self.viz_pub.publish(m)


def _pt(v):
    from geometry_msgs.msg import Point
    p = Point()
    p.x, p.y, p.z = float(v[0]), float(v[1]), float(v[2])
    return p


if __name__ == "__main__":
    try:
        ViapointDetourNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
