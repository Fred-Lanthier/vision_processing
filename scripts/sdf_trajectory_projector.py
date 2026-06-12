#!/usr/bin/env python3
"""
Project FM joint trajectories onto the robot SDF safe region.

This node sits between the planner and trajectory_executor:
  /planner/raw_nominal_trajectory -> /planner/nominal_trajectory

Architecture: the ROS spin thread is never blocked. The nominal trajectory is
forwarded immediately on each callback so the executor always has fresh data.
The SDF projection runs in a background worker thread and overwrites the nominal
with the projected version only if the result is still fresh enough to be useful.
The online CBF remains the last safety guard.
"""
import os
import sys
import time
import tempfile
import threading

import numpy as np
import rospy
import rospkg
import torch
import xacro
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState, PointCloud2
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

rospack = rospkg.RosPack()
pkg_path = rospack.get_path("vision_processing")
sys.path.insert(0, pkg_path)

# NOTE: URDFLayer/RDF_Weights/BernsteinCore (and their heavy deps such as
# pytorch_kinematics) are imported lazily inside _init_bernstein(), which only
# runs when ~enabled is true. With ~enabled=false this node is a pure
# pass-through (/planner/raw_nominal_trajectory -> /planner/nominal_trajectory)
# and must start even if those optional deps are unavailable.


JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]


def _bool_param(name, default=False):
    value = rospy.get_param(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class SDFTrajectoryProjector:
    def __init__(self):
        rospy.init_node("sdf_trajectory_projector")

        self.device = torch.device(
            rospy.get_param(
                "~device",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        )
        self.enabled = _bool_param("~enabled", True)
        self.input_topic = rospy.get_param(
            "~input_topic", "/planner/raw_nominal_trajectory")
        self.output_topic = rospy.get_param(
            "~output_topic", "/planner/nominal_trajectory")
        self.obs_topic = rospy.get_param("~obs_topic", "/perception/cleaned_obstacles")
        self.d_safe = float(rospy.get_param("~d_safe", 0.015))
        self.alpha = float(rospy.get_param("~barrier_alpha", 0.001))
        self.projection_margin = float(rospy.get_param("~projection_margin", 0.001))
        self.projection_iters = max(1, int(rospy.get_param("~projection_iters", 1)))
        self.projection_relaxation = float(rospy.get_param("~projection_relaxation", 0.8))
        self.smooth_relaxation = float(rospy.get_param("~smooth_relaxation", 0.10))
        self.tracking_relaxation = float(rospy.get_param("~tracking_relaxation", 0.02))
        self.max_joint_step = float(rospy.get_param("~max_joint_step_per_iter", 0.06))
        self.max_joint_velocity = float(rospy.get_param("~max_joint_velocity", 0.7))
        self.skip_first_waypoint = _bool_param("~skip_first_waypoint", True)
        self.line_search_steps = max(1, int(rospy.get_param("~line_search_steps", 5)))
        self.monotonic_tolerance = float(rospy.get_param(
            "~monotonic_tolerance", 1e-4))
        self.obstacle_input_max_points = int(rospy.get_param(
            "~obstacle_input_max_points", 5000))
        self.projection_max_points = max(1, int(rospy.get_param(
            "~projection_max_points", 32)))
        self.prefilter_radius = float(rospy.get_param(
            "~prefilter_radius", 0.30))
        self.prefilter_max_points = int(rospy.get_param(
            "~prefilter_max_points", 128))
        self.check_waypoints = max(1, int(rospy.get_param(
            "~check_waypoints", 6)))
        self.active_waypoints = max(1, int(rospy.get_param(
            "~active_waypoints", 2)))
        self.sdf_candidate_margin = float(rospy.get_param(
            "~sdf_candidate_margin", 0.08))
        self.sdf_chunk_size = max(1, int(rospy.get_param("~sdf_chunk_size", 1024)))
        # Max age for a projected result: discard if the trajectory it was computed
        # from is older than this many milliseconds (avoids retrograde jumps when
        # projection is slower than the planner period).
        self.projection_max_age_ms = float(rospy.get_param(
            "~projection_max_age_ms", 350.0))
        self.log_timing = _bool_param("~log_timing", True)
        self.publish_viz = _bool_param("~publish_viz", True)

        self.eye4 = torch.eye(4, device=self.device).unsqueeze(0)
        self.q_current = None
        self.qdot_current = None   # low-pass filtered joint velocity (1, 7)
        self._q_stamp = None       # perf_counter timestamp of last joint update
        self.obs_points = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.obs_stamp = rospy.Time(0)
        # Shared state lock (obs_points, q_current, qdot_current)
        self.lock = threading.Lock()

        # Worker thread state
        self._pending_msg = None
        self._pending_time = 0.0
        self._pending_seq = 0
        self._input_seq = 0
        self._pending_lock = threading.Lock()
        self._work_event = threading.Event()

        if self.enabled:
            self._init_bernstein()
        else:
            self.robot_layer = None
            self.bernstein_core = None

        self.pub = rospy.Publisher(self.output_topic, JointTrajectory, queue_size=1)
        self.viz_pub = rospy.Publisher(
            "/viz/projected_trajectory_3d", MarkerArray, queue_size=1)
        rospy.Subscriber(self.input_topic, JointTrajectory, self._trajectory_cb,
                         queue_size=1)
        rospy.Subscriber(self.obs_topic, PointCloud2, self._obstacle_cb, queue_size=1)
        rospy.Subscriber("/joint_states", JointState, self._joint_cb, queue_size=1)

        rospy.loginfo(
            "SDF trajectory projector ready: %s -> %s enabled=%s "
            "d_safe=%.3fm margin=%.3fm iters=%d max_points=%d "
            "prefilter=%.2fm/%d check_wp=%d active_wp=%d skip_first=%s "
            "max_age_ms=%.0f obs=%s",
            self.input_topic,
            self.output_topic,
            self.enabled,
            self.d_safe,
            self.projection_margin,
            self.projection_iters,
            self.projection_max_points,
            self.prefilter_radius,
            self.prefilter_max_points,
            self.check_waypoints,
            self.active_waypoints,
            self.skip_first_waypoint,
            self.projection_max_age_ms,
            self.obs_topic,
        )

    def _init_bernstein(self):
        # Imported here (not at module scope) so the disabled pass-through mode
        # has no hard dependency on pytorch_kinematics / RDF / Bernstein.
        from third_party.RDF.urdf_layer import URDFLayer
        from third_party.SDF_Bernstein_Basis.src.rdf_weights import RDF_Weights
        from third_party.SDF_Bernstein_Basis.bernstein_core import BernsteinCore

        urdf_path_raw = os.path.join(pkg_path, "urdf", "panda_camera.xacro")
        if urdf_path_raw.endswith(".xacro"):
            doc = xacro.process_file(urdf_path_raw)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".urdf", delete=False
            ) as f:
                f.write(doc.toxml())
                urdf_path = f.name
        else:
            urdf_path = urdf_path_raw

        self.robot_layer = URDFLayer(
            urdf_path=urdf_path,
            device=self.device,
            package_dir=pkg_path,
            voxel_dir=os.path.join(pkg_path, "third_party", "RDF",
                                   "panda_layer", "meshes", "voxel_128"),
        )

        self.link_names = [
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_link7",
            "panda_hand",
            "fork_tip",
        ]
        weight_handler = RDF_Weights(device=self.device, dtype=torch.float32)
        weight_handler.init_robot_folder(
            os.path.join(pkg_path, "third_party", "SDF_Bernstein_Basis",
                         "panda_test"),
            robot_name="panda",
        )
        weight_handler.add_models(self.link_names, robot_name="panda")
        self.bernstein_core = BernsteinCore(
            weight_handler, self.robot_layer, self.device, self.link_names)

    def _joint_cb(self, msg):
        pos = {n: p for n, p in zip(msg.name, msg.position)}
        q = [pos.get(j) for j in JOINT_NAMES]
        if None in q:
            return
        q_new = torch.tensor(q, dtype=torch.float32, device=self.device).view(1, 7)
        now = time.perf_counter()
        with self.lock:
            if self.q_current is not None and self._q_stamp is not None:
                dt = now - self._q_stamp
                if 0.005 < dt < 0.1:  # only on 5–100 ms intervals to avoid noise
                    qdot_raw = (q_new - self.q_current) / dt
                    if self.qdot_current is None:
                        self.qdot_current = qdot_raw
                    else:
                        self.qdot_current = 0.8 * self.qdot_current + 0.2 * qdot_raw
            self.q_current = q_new
            self._q_stamp = now

    def _obstacle_cb(self, msg):
        pts = [
            (p[0], p[1], p[2])
            for p in pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)
        ]
        if not pts:
            arr = np.empty((0, 3), dtype=np.float32)
        else:
            arr = np.asarray(pts, dtype=np.float32)
            if (
                self.obstacle_input_max_points > 0
                and arr.shape[0] > self.obstacle_input_max_points
            ):
                idx = np.linspace(
                    0, arr.shape[0] - 1, self.obstacle_input_max_points,
                    dtype=np.int64)
                arr = arr[idx]

        obs = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        with self.lock:
            self.obs_points = obs
            self.obs_stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0 else rospy.Time.now()

    # ------------------------------------------------------------------
    # Trajectory callback — must never block the ROS spin thread
    # ------------------------------------------------------------------

    def _trajectory_cb(self, msg):
        """Publish raw trajectory immediately, then queue projection work."""
        if not msg.points:
            return
        self.pub.publish(msg)
        if not self.enabled:
            return
        received_time = time.perf_counter()
        with self._pending_lock:
            self._input_seq += 1
            self._pending_msg = msg
            self._pending_time = received_time
            self._pending_seq = self._input_seq
        self._work_event.set()

    # ------------------------------------------------------------------
    # Background projection worker
    # ------------------------------------------------------------------

    def _worker_loop(self):
        """Background thread: project trajectories without blocking ROS callbacks."""
        while not rospy.is_shutdown():
            self._work_event.wait(timeout=1.0)
            self._work_event.clear()

            with self._pending_lock:
                msg = self._pending_msg
                received_time = self._pending_time
                seq = self._pending_seq
                self._pending_msg = None

            if msg is not None:
                self._process_trajectory(msg, received_time, seq)

    def _process_trajectory(self, msg, received_time, seq):
        """Project trajectory and always publish the best result available."""
        t0 = time.perf_counter()
        q_nom = torch.tensor(
            [p.positions[:7] for p in msg.points],
            dtype=torch.float32,
            device=self.device,
        )
        times = [p.time_from_start.to_sec() for p in msg.points]

        with self.lock:
            obs = self.obs_points.detach().clone()

        q_out = q_nom
        projected = False
        raw_min_h = float("inf")
        final_min_h = float("inf")
        obs_prefiltered_n = 0
        obs_sel_n = 0
        eval_n = 0

        if self.enabled and obs.shape[0] > 0:
            try:
                with torch.no_grad():
                    eval_idx = self._waypoint_indices(q_nom.shape[0])
                    q_eval = q_nom[eval_idx]
                    obs_prefiltered = self._prefilter_obstacles_by_link_centers(
                        q_eval, obs)
                    obs_prefiltered_n = obs_prefiltered.shape[0]

                if obs_prefiltered_n > 0:
                    with torch.no_grad():
                        obs_sel = self._select_obstacles_for_trajectory(
                            q_eval, obs_prefiltered)
                        obs_sel_n = obs_sel.shape[0]

                    if obs_sel_n > 0:
                        with torch.no_grad():
                            h_raw = self._barrier_h_no_grad(q_eval, obs_sel)
                            raw_min_h = float(h_raw.min().item())
                        eval_n = eval_idx.numel()

                        if raw_min_h <= self.projection_margin:
                            q_proj, final_h = self._project(
                                q_nom, obs_sel, times, eval_idx)
                            final_min_h = float(final_h.min().item())
                            improved = (
                                final_min_h >= raw_min_h - self.monotonic_tolerance)
                            if improved:
                                q_out = q_proj
                                projected = True
                            else:
                                rospy.logwarn_throttle(
                                    1.0,
                                    "SDF projection found no improvement "
                                    "(h_raw=%.4f h_proj=%.4f); publishing nominal.",
                                    raw_min_h, final_min_h,
                                )
                                final_min_h = raw_min_h
                        else:
                            final_min_h = raw_min_h
            except Exception as exc:
                rospy.logwarn_throttle(
                    1.0,
                    "SDF trajectory projection failed; publishing nominal: %s",
                    exc,
                )

        # Pin waypoint 0 to the current robot state so the executor can anchor
        # at "start" (waypoint 0 = where the robot is now) without a step-back.
        # Then re-enforce velocity limits from the new waypoint 0 so that
        # obstacle-projected waypoints don't create a velocity spike at the seam.
        with self.lock:
            q_now = self.q_current
            qdot_now = self.qdot_current
        if q_now is not None and q_out.shape[0] > 0:
            q_out = q_out.detach().clone()
            q_out[0] = q_now.squeeze(0)
            q_out = self._enforce_segment_velocity(q_out, times)

        # Gently blend waypoints 1–2 toward the robot's current velocity direction
        # to smooth the velocity kink that occurs when a new trajectory arrives.
        # This only adjusts direction (20 % / 5 % weights), so obstacle avoidance
        # from the projection is preserved.
        if qdot_now is not None and q_now is not None and q_out.shape[0] > 2 and len(times) > 2:
            q0 = q_now.squeeze(0)
            qdot = qdot_now.squeeze(0)
            dt_1 = max(times[1] - times[0], 1e-3)
            dt_2 = max(times[2] - times[0], 1e-3)
            q_out[1] = 0.80 * q_out[1] + 0.20 * (q0 + qdot * dt_1)
            q_out[2] = 0.95 * q_out[2] + 0.05 * (q0 + qdot * dt_2)
            q_out = self._enforce_segment_velocity(q_out, times)

        age_ms = (time.perf_counter() - received_time) * 1000.0
        with self._pending_lock:
            latest = seq == self._input_seq
        fresh = age_ms <= self.projection_max_age_ms
        out_msg = self._make_output_msg(msg, q_out)
        if projected and fresh and latest:
            self.pub.publish(out_msg)
            if self.publish_viz:
                self._publish_viz(q_out)
        elif projected and (not fresh or not latest):
            rospy.logwarn_throttle(
                1.0,
                "SDF trajectory projection discarded result: age=%.0f ms "
                "max_age=%.0f ms latest=%s h_raw=%.4f h_proj=%.4f",
                age_ms,
                self.projection_max_age_ms,
                latest,
                raw_min_h,
                final_min_h,
            )

        if self.log_timing:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            max_delta = float(torch.norm(q_out - q_nom, dim=1).max().item())
            success = final_min_h >= self.projection_margin
            improved = projected
            rospy.loginfo(
                "⏱️ [TIMING] SDF trajectory projection: %.2f ms (age %.0f ms) | "
                "projected=%s fresh=%s latest=%s points=%d check=%d obs_pref=%d obs=%d "
                "h_raw=%.4f h_proj=%.4f success=%s improved=%s "
                "max_delta=%.4frad",
                elapsed_ms,
                age_ms,
                projected,
                fresh,
                latest,
                q_nom.shape[0],
                eval_n,
                obs_prefiltered_n,
                obs_sel_n,
                raw_min_h,
                final_min_h,
                success,
                improved,
                max_delta,
            )

    # ------------------------------------------------------------------
    # Output message helpers
    # ------------------------------------------------------------------

    def _make_output_msg(self, in_msg, q_out):
        q_np = q_out.detach().cpu().numpy()
        times = [p.time_from_start.to_sec() for p in in_msg.points]
        if not times:
            times = [0.0] * q_np.shape[0]

        msg = JointTrajectory()
        msg.header = in_msg.header
        if msg.header.stamp.to_sec() <= 0.0:
            msg.header.stamp = rospy.Time.now()
        msg.joint_names = list(in_msg.joint_names) if in_msg.joint_names else JOINT_NAMES

        qdot = self._finite_difference_velocities(q_np, times)
        for i in range(q_np.shape[0]):
            pt = JointTrajectoryPoint()
            pt.positions = q_np[i].tolist()
            pt.velocities = qdot[i].tolist()
            pt.time_from_start = rospy.Duration(times[i])
            msg.points.append(pt)
        return msg

    def _finite_difference_velocities(self, q_np, times):
        h = q_np.shape[0]
        qdot = np.zeros_like(q_np)
        if h < 2:
            return qdot
        for i in range(h):
            if i == 0:
                dt = max(times[1] - times[0], 1e-3)
                vel = (q_np[1] - q_np[0]) / dt
            elif i == h - 1:
                dt = max(times[-1] - times[-2], 1e-3)
                vel = (q_np[-1] - q_np[-2]) / dt
            else:
                dt = max(times[i + 1] - times[i - 1], 1e-3)
                vel = (q_np[i + 1] - q_np[i - 1]) / dt
            qdot[i] = np.clip(vel, -self.max_joint_velocity,
                              self.max_joint_velocity)
        return qdot

    # ------------------------------------------------------------------
    # SDF projection logic
    # ------------------------------------------------------------------

    def _waypoint_indices(self, n_points):
        if n_points <= 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        start = 1 if self.skip_first_waypoint and n_points > 1 else 0
        n_eval = n_points - start
        if n_eval <= self.check_waypoints:
            return torch.arange(start, n_points, device=self.device)
        idx = torch.linspace(
            start,
            n_points - 1,
            steps=min(self.check_waypoints, n_eval),
            device=self.device,
        ).round().long()
        return torch.unique(idx, sorted=True)

    def _trajectory_link_centers(self, q):
        q_eval = q
        if q_eval.shape[-1] == 7:
            pad = torch.zeros((q_eval.shape[0], 2), device=self.device)
            q_eval = torch.cat([q_eval, pad], dim=-1)
        poses = self.robot_layer._native_forward_kinematics(q_eval)
        centers = []
        for link_name in self.link_names:
            T_link = poses.get(link_name, None)
            if T_link is not None:
                centers.append(T_link[:, :3, 3])
        if not centers:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)
        return torch.stack(centers, dim=1).reshape(-1, 3)

    def _prefilter_obstacles_by_link_centers(self, q, obs):
        if obs.shape[0] == 0:
            return obs

        centers = self._trajectory_link_centers(q)
        if centers.shape[0] == 0:
            return obs

        center_dist = torch.cdist(
            obs.unsqueeze(0), centers.unsqueeze(0)).squeeze(0).min(dim=1).values

        if self.prefilter_radius > 0.0:
            keep = center_dist <= self.prefilter_radius
            if not torch.any(keep):
                return torch.empty((0, 3), dtype=torch.float32, device=self.device)
            obs = obs[keep]
            center_dist = center_dist[keep]

        if (
            self.prefilter_max_points > 0
            and obs.shape[0] > self.prefilter_max_points
        ):
            _, idx = torch.topk(
                center_dist,
                k=self.prefilter_max_points,
                largest=False,
            )
            obs = obs[idx]
        return obs

    def _select_obstacles_for_trajectory(self, q, obs):
        if obs.shape[0] == 0:
            return obs

        scores = []
        pose = self.eye4.expand(q.shape[0], 4, 4)
        for start in range(0, int(obs.shape[0]), self.sdf_chunk_size):
            pts = obs[start:start + self.sdf_chunk_size]
            sdf = self.bernstein_core.get_whole_body_sdf_batch(
                pts, pose, q, return_per_link=False)
            scores.append(sdf.min(dim=0).values.detach())
        score = torch.cat(scores, dim=0)

        near_threshold = self.d_safe + self.sdf_candidate_margin
        near_mask = score <= near_threshold
        if torch.any(near_mask):
            candidate_idx = torch.nonzero(near_mask, as_tuple=False).flatten()
            candidate_score = score[candidate_idx]
        else:
            candidate_idx = torch.arange(obs.shape[0], device=self.device)
            candidate_score = score

        k = min(self.projection_max_points, int(candidate_idx.numel()))
        _, local_idx = torch.topk(candidate_score, k=k, largest=False)
        return obs[candidate_idx[local_idx]]

    def _barrier_h_no_grad(self, q, obs):
        pose = self.eye4.expand(q.shape[0], 4, 4)
        _, sdf_per_link = self.bernstein_core.get_whole_body_sdf_batch(
            obs, pose, q, return_per_link=True)
        sdf_min = sdf_per_link.min(dim=-1, keepdim=True).values
        shifted = sdf_per_link - sdf_min
        exp_terms = torch.exp(-shifted / max(self.alpha, 1e-6))
        h_per_link = (
            sdf_min.squeeze(-1)
            - self.alpha * torch.log(exp_terms.sum(dim=-1))
            - self.d_safe
        )
        return h_per_link.min(dim=1).values

    def _barrier_h_and_grad(self, q, obs):
        q_var = q.detach().clone().requires_grad_(True)
        h = self._barrier_h_no_grad(q_var, obs)
        grad = torch.autograd.grad(
            outputs=h,
            inputs=q_var,
            grad_outputs=torch.ones_like(h),
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]
        return h.detach(), grad.detach()

    def _project(self, q_nom, obs, times, eval_idx):
        q = q_nom.detach().clone()
        target_h = self.projection_margin
        with torch.no_grad():
            h_eval_cached = self._barrier_h_no_grad(q[eval_idx], obs).detach()
        best_h = h_eval_cached.clone()
        best_q = q.clone()
        best_min_h = float(best_h.min().item())

        for _ in range(self.projection_iters):
            # Reuse the SDF values computed at the end of the previous iteration
            # instead of recomputing them — saves one full SDF call per iteration.
            h_eval = h_eval_cached
            current_min_h = float(h_eval.min().item())
            violation_eval = (target_h - h_eval).clamp(min=0.0)
            active_local = torch.nonzero(
                violation_eval > 0.0, as_tuple=False).flatten()
            if active_local.numel() > self.active_waypoints:
                _, keep_local = torch.topk(
                    violation_eval[active_local],
                    k=self.active_waypoints,
                    largest=True,
                )
                active_local = active_local[keep_local]

            if active_local.numel() > 0:
                active_idx = eval_idx[active_local]
                h, grad = self._barrier_h_and_grad(q[active_idx], obs)
                violation = (target_h - h).clamp(min=0.0)
                active = violation > 0.0
            else:
                active_idx = None
                active = torch.empty((0,), dtype=torch.bool, device=self.device)

            if active_idx is not None and torch.any(active):
                denom = (grad ** 2).sum(dim=1, keepdim=True) + 1e-6
                step = (
                    self.projection_relaxation
                    * violation.view(-1, 1)
                    * grad
                    / denom
                )
                if self.max_joint_step > 0.0:
                    step_norm = torch.norm(step, dim=1, keepdim=True)
                    scale = torch.clamp(
                        self.max_joint_step / (step_norm + 1e-6),
                        max=1.0,
                    )
                    step = step * scale
                accepted = False
                step_scale = 1.0
                base_q = q.detach().clone()

                for _line in range(self.line_search_steps):
                    q_candidate = base_q.clone()
                    q_active = torch.where(
                        active.view(-1, 1),
                        base_q[active_idx] + step_scale * step,
                        base_q[active_idx],
                    )
                    q_candidate[active_idx] = q_active
                    q_candidate[0] = q_nom[0]

                    with torch.no_grad():
                        h_step = self._barrier_h_no_grad(
                            q_candidate[eval_idx], obs).detach()

                    q_regularized, h_regularized = self._regularize_candidate(
                        q_candidate, q_nom, times, eval_idx, obs, h_step)
                    candidate_min_h = float(h_regularized.min().item())

                    if candidate_min_h >= current_min_h - self.monotonic_tolerance:
                        q = q_regularized.detach().clone()
                        h_eval_cached = h_regularized.detach().clone()
                        accepted = True
                        if candidate_min_h > best_min_h:
                            best_min_h = candidate_min_h
                            best_h = h_regularized.detach().clone()
                            best_q = q.detach().clone()
                        break
                    step_scale *= 0.5

                if not accepted:
                    break
            else:
                break

            if best_min_h >= target_h - 1e-5:
                break

        return best_q.detach(), best_h.detach()

    def _regularize_candidate(self, q_candidate, q_nom, times, eval_idx, obs, h_step):
        """Apply shape regularizers only when they do not undo SDF progress."""
        q_best = q_candidate.detach().clone()
        h_best = h_step.detach().clone()
        best_min_h = float(h_best.min().item())

        q_reg = q_best.clone()
        if q_reg.shape[0] > 2 and self.smooth_relaxation > 0.0:
            q_smooth = q_reg.clone()
            lap = 0.5 * (q_reg[:-2] + q_reg[2:]) - q_reg[1:-1]
            q_smooth[1:-1] = q_reg[1:-1] + self.smooth_relaxation * lap
            q_reg = q_smooth

        if self.tracking_relaxation > 0.0:
            q_reg = q_reg + self.tracking_relaxation * (q_nom - q_reg)

        q_reg = self._enforce_segment_velocity(q_reg, times)
        q_reg[0] = q_nom[0]

        with torch.no_grad():
            h_reg = self._barrier_h_no_grad(q_reg[eval_idx], obs).detach()
        reg_min_h = float(h_reg.min().item())
        if reg_min_h >= best_min_h - self.monotonic_tolerance:
            return q_reg.detach(), h_reg.detach()
        return q_best.detach(), h_best.detach()

    def _enforce_segment_velocity(self, q, times):
        if self.max_joint_velocity <= 0.0 or q.shape[0] < 2:
            return q
        q_limited = q.clone()
        for i in range(1, q_limited.shape[0]):
            dt = max(times[i] - times[i - 1], 1e-3)
            max_delta = self.max_joint_velocity * dt
            delta = q_limited[i] - q_limited[i - 1]
            q_limited[i] = q_limited[i - 1] + torch.clamp(
                delta, min=-max_delta, max=max_delta)
        return q_limited

    def _get_fork_tip_pose(self, q7):
        q_eval = q7
        if q_eval.dim() == 1:
            q_eval = q_eval.unsqueeze(0)
        if q_eval.shape[-1] == 7:
            pad = torch.zeros((q_eval.shape[0], 2), device=self.device)
            q_eval = torch.cat([q_eval, pad], dim=-1)
        poses = self.robot_layer._native_forward_kinematics(q_eval)
        return poses.get("fork_tip", None)

    def _publish_viz(self, q):
        T = self._get_fork_tip_pose(q.detach())
        if T is None:
            return
        pts = T[:, :3, 3].detach().cpu().numpy()
        stamp = rospy.Time.now()

        markers = MarkerArray()
        line = Marker()
        line.header.frame_id = "world"
        line.header.stamp = stamp
        line.ns = "projected_trajectory"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.012
        line.color.a = 1.0
        line.color.r = 0.0
        line.color.g = 0.8
        line.color.b = 1.0
        for p in pts:
            line.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
        markers.markers.append(line)
        self.viz_pub.publish(markers)

    def run(self):
        worker = threading.Thread(target=self._worker_loop, daemon=True)
        worker.start()
        rospy.spin()


if __name__ == "__main__":
    SDFTrajectoryProjector().run()
