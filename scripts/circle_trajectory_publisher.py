#!/usr/bin/env python3
"""
Publishes a circular Cartesian trajectory to franka_gazebo's
position_joint_trajectory_controller.

Steps:
  1. Read the current EE pose from TF (panda_link0 → panda_link8).
  2. Generate N waypoints on a circle of given radius around that pose,
     keeping orientation fixed.
  3. Solve IK for each waypoint with fast_ik_module (warm-started from
     the previous solution so joint continuity is guaranteed).
  4. Publish the resulting JointTrajectory once (latched).

Topic out: /position_joint_trajectory_controller/command
"""
import os
import tempfile
import time
import numpy as np
import rospy
import rospkg
import xacro
import tf
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from vision_processing import fast_ik_module

# ── Joint names for the franka panda (no gripper) ─────────────────────────────
JOINT_NAMES = [f'panda_joint{i}' for i in range(1, 8)]

# ── Home configuration — must match initial_joint_positions in the launch file ─
HOME_Q = np.array([-0.000059, -0.125928,  0.000117,
                   -2.193312, -0.000251,  2.064780, 0.785511])

# ── Circle parameters ──────────────────────────────────────────────────────────
RADIUS     = 0.08   # circle radius [m]
N_POINTS   = 36     # waypoints per revolution (one every 10°)
PERIOD     = 12.0   # seconds per revolution


def get_bool_param(name, default=False):
    value = rospy.get_param(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 'yes', 'on')
    return bool(value)


def make_waypoint_markers(positions, frame_id='panda_link0'):
    """Build RViz markers for the Cartesian circle path and its waypoints."""
    markers = MarkerArray()

    clear = Marker()
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)

    line = Marker()
    line.header.stamp = rospy.Time.now()
    line.header.frame_id = frame_id
    line.ns = 'circle_trajectory_path'
    line.id = 0
    line.type = Marker.LINE_STRIP
    line.action = Marker.ADD
    line.scale.x = 0.006
    line.color.r = 0.0
    line.color.g = 0.35
    line.color.b = 1.0
    line.color.a = 1.0

    points = Marker()
    points.header = line.header
    points.ns = 'circle_trajectory_waypoints'
    points.id = 1
    points.type = Marker.SPHERE_LIST
    points.action = Marker.ADD
    points.scale.x = 0.018
    points.scale.y = 0.018
    points.scale.z = 0.018
    points.color.r = 1.0
    points.color.g = 0.65
    points.color.b = 0.0
    points.color.a = 1.0

    start = Marker()
    start.header = line.header
    start.ns = 'circle_trajectory_start'
    start.id = 2
    start.type = Marker.SPHERE
    start.action = Marker.ADD
    start.scale.x = 0.035
    start.scale.y = 0.035
    start.scale.z = 0.035
    start.color.r = 0.0
    start.color.g = 1.0
    start.color.b = 0.0
    start.color.a = 1.0

    for xyz in positions:
        p = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
        line.points.append(p)
        points.points.append(p)

    if positions:
        start.pose.position = Point(
            x=float(positions[0][0]),
            y=float(positions[0][1]),
            z=float(positions[0][2]),
        )

    markers.markers.extend([line, points, start])
    return markers


class ExecutionMonitor:
    """Logs tracking progress while the trajectory controller executes."""

    def __init__(self, traj, rate_hz):
        self.traj = traj
        self.q_current = None
        self.start_time = traj.header.stamp
        self.duration = traj.points[-1].time_from_start.to_sec() if traj.points else 0.0
        self.done = False

        rospy.Subscriber('/joint_states', JointState, self._joint_cb, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate_hz), self._log_status)

    def _joint_cb(self, msg):
        pos = {n: p for n, p in zip(msg.name, msg.position)}
        q = [pos.get(j) for j in JOINT_NAMES]
        if None not in q:
            self.q_current = np.array(q, dtype=np.float64)

    def _interpolate(self, t):
        points = self.traj.points
        if t <= points[0].time_from_start.to_sec():
            return np.array(points[0].positions[:7], dtype=np.float64)
        if t >= points[-1].time_from_start.to_sec():
            return np.array(points[-1].positions[:7], dtype=np.float64)

        for i in range(len(points) - 1):
            t0 = points[i].time_from_start.to_sec()
            t1 = points[i + 1].time_from_start.to_sec()
            if t0 <= t <= t1:
                q0 = np.array(points[i].positions[:7], dtype=np.float64)
                q1 = np.array(points[i + 1].positions[:7], dtype=np.float64)
                alpha = (t - t0) / max(t1 - t0, 1e-9)
                return q0 + alpha * (q1 - q0)

        return np.array(points[-1].positions[:7], dtype=np.float64)

    def _log_status(self, event):
        if self.done or self.q_current is None or not self.traj.points:
            return

        t0 = time.perf_counter()
        elapsed = (rospy.Time.now() - self.start_time).to_sec()

        if elapsed < 0.0:
            rospy.loginfo(
                "EXEC MONITOR | starts in %.3fs | monitor_compute=%.3f ms",
                -elapsed,
                (time.perf_counter() - t0) * 1000.0,
            )
            return

        t_eval = min(elapsed, self.duration)
        q_des = self._interpolate(t_eval)
        q_err = q_des - self.q_current
        err_norm = float(np.linalg.norm(q_err))
        err_max = float(np.max(np.abs(q_err)))
        progress = 100.0 * t_eval / max(self.duration, 1e-9)
        compute_ms = (time.perf_counter() - t0) * 1000.0

        rospy.loginfo(
            "EXEC MONITOR | t=%.2f/%.2fs progress=%.1f%% | "
            "joint_err_norm=%.4f rad max_abs=%.4f rad | monitor_compute=%.3f ms",
            t_eval,
            self.duration,
            progress,
            err_norm,
            err_max,
            compute_ms,
        )

        if elapsed >= self.duration:
            self.done = True
            self.timer.shutdown()


class WaypointStreamer:
    """Publishes rolling short-horizon JointTrajectory commands."""

    def __init__(self, traj, pub, segment_duration, start_delay, log_hz=5.0,
                 horizon_points=4, command_lead_time=0.08):
        self.traj = traj
        self.pub = pub
        self.segment_duration = float(segment_duration)
        self.points = traj.points
        self.idx = 0
        self.done = False
        self.q_current = None
        self.horizon_points = max(int(horizon_points), 2)
        self.command_lead_time = float(command_lead_time)
        self.last_log_time = 0.0
        self.log_period = 1.0 / max(float(log_hz), 1e-9)
        self.start_time = rospy.Time.now() + rospy.Duration(start_delay)
        rospy.Subscriber('/joint_states', JointState, self._joint_cb, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(start_delay), self._start, oneshot=True)

    def _joint_cb(self, msg):
        pos = {n: p for n, p in zip(msg.name, msg.position)}
        q = [pos.get(j) for j in JOINT_NAMES]
        if None not in q:
            self.q_current = np.array(q, dtype=np.float64)

    def _make_segment(self):
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now() + rospy.Duration(self.command_lead_time)
        msg.joint_names = JOINT_NAMES

        q_start = self.q_current
        if q_start is None:
            q_start = np.array(self.points[self.idx].positions[:7], dtype=np.float64)

        p0 = JointTrajectoryPoint()
        p0.positions = q_start.tolist()
        p0.velocities = [0.0] * 7
        p0.time_from_start = rospy.Duration(0.0)
        msg.points.append(p0)

        max_idx = min(self.idx + self.horizon_points, len(self.points) - 1)
        for j, src_idx in enumerate(range(self.idx + 1, max_idx + 1), start=1):
            pt = JointTrajectoryPoint()
            pt.positions = list(self.points[src_idx].positions[:7])
            # Leave velocities empty in rolling mode: the controller interpolates
            # between a current-state anchor and the next lookahead waypoints.
            pt.time_from_start = rospy.Duration(j * self.segment_duration)
            msg.points.append(pt)

        return msg

    def _start(self, event):
        rospy.loginfo(
            "STREAM EXECUTION | starting rolling-horizon waypoint mode | "
            "segments=%d segment_duration=%.3fs horizon_points=%d lead_time=%.3fs",
            max(len(self.points) - 1, 0),
            self.segment_duration,
            self.horizon_points,
            self.command_lead_time,
        )
        self.timer = rospy.Timer(rospy.Duration(self.segment_duration), self._step)
        self._step(None)

    def _step(self, event):
        if self.done or self.idx >= len(self.points) - 1:
            self.done = True
            self.timer.shutdown()
            rospy.loginfo("STREAM EXECUTION | completed %d segments", self.idx)
            return

        t0 = time.perf_counter()
        msg = self._make_segment()
        self.pub.publish(msg)
        compute_ms = (time.perf_counter() - t0) * 1000.0

        now = rospy.get_time()
        if now - self.last_log_time >= self.log_period:
            self.last_log_time = now
            rospy.loginfo(
                "STREAM EXECUTION | segment=%d/%d | points=%d | command_compute=%.3f ms",
                self.idx + 1,
                len(self.points) - 1,
                len(msg.points),
                compute_ms,
            )

        self.idx += 1


def build_ik_solver():
    """Build FastIK from the franka_description panda URDF (no hand/gripper)."""
    pkg = rospkg.RosPack().get_path('franka_description')
    xacro_path = os.path.join(pkg, 'robots', 'panda', 'panda.urdf.xacro')
    doc = xacro.process_file(xacro_path,
                             mappings={'hand': 'false', 'gazebo': 'false'})
    with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
        f.write(doc.toxml())
        urdf_path = f.name
    # panda_link8 is the fixed flange frame — the natural IK target without a gripper
    return fast_ik_module.FastIK(urdf_path, 'panda_link8')


def get_ee_pose_from_tf(listener, base_frame='panda_link0', ee_frame='panda_link8'):
    """Return (position [3], R [3×3]) of the EE in the robot base frame."""
    while not rospy.is_shutdown():
        try:
            trans, rot = listener.lookupTransform(base_frame, ee_frame, rospy.Time(0))
            pos = np.array(trans)
            R   = Rotation.from_quat(rot).as_matrix()
            return pos, R
        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException):
            rospy.sleep(0.1)


def build_trajectory(ik_solver, center_pos, R_fixed, log_each_ik=False,
                     smooth_velocities=False):
    """
    Generate circle waypoints in Cartesian space and solve IK for each.
    The orientation R_fixed is held constant throughout.
    """
    msg = JointTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.joint_names  = JOINT_NAMES

    dt    = PERIOD / N_POINTS
    q_prev = HOME_Q.copy()
    n_ok   = 0
    ik_times_ms = []
    waypoint_positions = []
    t_build_start = time.perf_counter()

    for i in range(N_POINTS + 1):          # +1 to close the loop
        theta = 2.0 * np.pi * i / N_POINTS

        # Circle in the XY plane (horizontal); change to XZ for a vertical circle
        pos = center_pos + np.array([RADIUS * np.cos(theta),
                                     RADIUS * np.sin(theta),
                                     0.0])

        T_target = np.eye(4)
        T_target[:3, :3] = R_fixed
        T_target[:3,  3] = pos

        t_ik_start = time.perf_counter()
        q = np.array(ik_solver.solve_single_ik(T_target, q_prev))[:7]
        ik_ms = (time.perf_counter() - t_ik_start) * 1000.0
        ik_times_ms.append(ik_ms)
        if log_each_ik:
            rospy.loginfo(
                "IK WAYPOINT | idx=%02d theta=%.1f deg | solve_single_ik=%.3f ms",
                i,
                np.degrees(theta),
                ik_ms,
            )

        if not np.all(np.isfinite(q)):
            rospy.logwarn("IK failed at waypoint %d (theta=%.1f°) — skipping",
                          i, np.degrees(theta))
            continue

        # Unwrap to avoid 2π jumps
        diff = q - q_prev
        diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
        q    = q_prev + diff
        q_prev = q.copy()

        pt = JointTrajectoryPoint()
        pt.positions       = q.tolist()
        pt.time_from_start = rospy.Duration(i * dt)
        msg.points.append(pt)
        waypoint_positions.append(pos.copy())
        n_ok += 1

    if smooth_velocities and len(msg.points) >= 2:
        q_mat = np.array([p.positions[:7] for p in msg.points], dtype=np.float64)
        times = np.array([p.time_from_start.to_sec() for p in msg.points], dtype=np.float64)
        for i, point in enumerate(msg.points):
            if i == 0:
                vel = (q_mat[1] - q_mat[0]) / max(times[1] - times[0], 1e-9)
            elif i == len(msg.points) - 1:
                vel = (q_mat[-1] - q_mat[-2]) / max(times[-1] - times[-2], 1e-9)
            else:
                vel = (q_mat[i + 1] - q_mat[i - 1]) / max(times[i + 1] - times[i - 1], 1e-9)
            point.velocities = vel.tolist()
    else:
        for point in msg.points:
            point.velocities = [0.0] * 7

    build_ms = (time.perf_counter() - t_build_start) * 1000.0
    if ik_times_ms:
        ik_np = np.array(ik_times_ms, dtype=np.float64)
        rospy.loginfo(
            "Trajectory built: %d/%d waypoints, %.1fs duration | total=%.2f ms | "
            "IK avg=%.2f ms min=%.2f ms max=%.2f ms | smooth_velocities=%s",
            n_ok, N_POINTS + 1, PERIOD, build_ms,
            float(np.mean(ik_np)), float(np.min(ik_np)), float(np.max(ik_np)),
            smooth_velocities,
        )
        rospy.loginfo(
            "IK TIMING | solver=vision_processing.fast_ik_module.FastIK | "
            "calls=%d total=%.2f ms avg=%.3f ms median=%.3f ms p95=%.3f ms max=%.3f ms",
            len(ik_np),
            float(np.sum(ik_np)),
            float(np.mean(ik_np)),
            float(np.median(ik_np)),
            float(np.percentile(ik_np, 95)),
            float(np.max(ik_np)),
        )
    else:
        rospy.loginfo(
            "Trajectory built: %d/%d waypoints, %.1fs duration | total=%.2f ms | no IK timings",
            n_ok, N_POINTS + 1, PERIOD, build_ms,
        )
    return msg, waypoint_positions


if __name__ == '__main__':
    rospy.init_node('circle_trajectory_publisher')
    start_delay = float(rospy.get_param('~start_delay', 0.25))
    monitor_hz = float(rospy.get_param('~monitor_hz', 1.0))
    log_each_ik = get_bool_param('~log_ik_each_waypoint', False)
    smooth_velocities = get_bool_param('~smooth_velocities', False)
    publish_waypoint_markers = get_bool_param('~publish_waypoint_markers', True)
    execution_mode = str(rospy.get_param('~execution_mode', 'full')).strip().lower()
    planner_topic = rospy.get_param('~planner_topic', '/planner/nominal_trajectory')
    stream_segment_duration = float(rospy.get_param(
        '~stream_segment_duration', PERIOD / N_POINTS))
    stream_horizon_points = int(rospy.get_param('~stream_horizon_points', 4))
    stream_command_lead_time = float(rospy.get_param('~stream_command_lead_time', 0.08))

    # ── 1. Build IK solver ────────────────────────────────────────────────────
    rospy.loginfo("Loading franka IK solver ...")
    t0 = time.perf_counter()
    ik_solver = build_ik_solver()
    rospy.loginfo("IK solver loaded in %.2f ms", (time.perf_counter() - t0) * 1000.0)

    # ── 2. Wait for the controller to be ready ────────────────────────────────
    rospy.loginfo("Waiting for /joint_states ...")
    t0 = time.perf_counter()
    rospy.wait_for_message('/joint_states', JointState)
    rospy.loginfo("Received first /joint_states after %.2f ms", (time.perf_counter() - t0) * 1000.0)
    rospy.sleep(2.0)   # let controller finish initialising

    # ── 3. Get current EE pose from TF ───────────────────────────────────────
    listener = tf.TransformListener()
    rospy.sleep(1.0)   # let TF buffer fill
    t0 = time.perf_counter()
    center_pos, R_fixed = get_ee_pose_from_tf(listener)
    rospy.loginfo("TF lookup panda_link0 -> panda_link8 completed in %.2f ms",
                  (time.perf_counter() - t0) * 1000.0)
    rospy.loginfo("Circle center (panda_link0 frame): x=%.3f y=%.3f z=%.3f",
                  *center_pos)

    # ── 4. Build and publish the circle trajectory ────────────────────────────
    command_topic = planner_topic if execution_mode == 'planner' else \
        '/position_joint_trajectory_controller/command'
    pub = rospy.Publisher(
        command_topic,
        JointTrajectory, queue_size=1, latch=(execution_mode in ('full', 'planner'))
    )
    marker_pub = rospy.Publisher(
        '/circle_trajectory/waypoint_markers',
        MarkerArray, queue_size=1, latch=True
    )

    t0 = time.perf_counter()
    traj, waypoint_positions = build_trajectory(
        ik_solver,
        center_pos,
        R_fixed,
        log_each_ik=log_each_ik,
        smooth_velocities=smooth_velocities,
    )
    rospy.loginfo("build_trajectory() wall time: %.2f ms",
                  (time.perf_counter() - t0) * 1000.0)

    if publish_waypoint_markers:
        marker_pub.publish(make_waypoint_markers(waypoint_positions))
        rospy.loginfo(
            "Published %d Cartesian waypoint markers on /circle_trajectory/waypoint_markers",
            len(waypoint_positions),
        )

    if execution_mode == 'planner':
        traj.header.stamp = rospy.Time.now() + rospy.Duration(start_delay)
        rospy.loginfo(
            "Execution mode: planner | publishing full trajectory to %s for external executor",
            planner_topic,
        )
        t0 = time.perf_counter()
        pub.publish(traj)
        rospy.loginfo("Published planner JointTrajectory with %d points in %.2f ms",
                      len(traj.points), (time.perf_counter() - t0) * 1000.0)
        monitor = ExecutionMonitor(traj, monitor_hz)
    elif execution_mode == 'stream':
        traj.header.stamp = rospy.Time.now() + rospy.Duration(start_delay)
        rospy.loginfo(
            "Execution mode: stream | one short command per waypoint | start_delay=%.3fs",
            start_delay,
        )
        streamer = WaypointStreamer(
            traj,
            pub,
            segment_duration=stream_segment_duration,
            start_delay=start_delay,
            log_hz=monitor_hz,
            horizon_points=stream_horizon_points,
            command_lead_time=stream_command_lead_time,
        )
        monitor = ExecutionMonitor(traj, monitor_hz)
    else:
        traj.header.stamp = rospy.Time.now() + rospy.Duration(start_delay)
        rospy.loginfo("Trajectory start scheduled %.3fs in the future at %.3f",
                      start_delay, traj.header.stamp.to_sec())

        t0 = time.perf_counter()
        pub.publish(traj)
        rospy.loginfo("Published JointTrajectory with %d points in %.2f ms",
                      len(traj.points), (time.perf_counter() - t0) * 1000.0)
        rospy.loginfo("Circle trajectory sent — robot should complete one revolution in %.0fs.",
                      PERIOD)
        monitor = ExecutionMonitor(traj, monitor_hz)

    rospy.spin()
