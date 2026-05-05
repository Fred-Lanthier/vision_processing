"""
PersistentTargetModel v5 — ICP-based tracking through occlusion
================================================================

Changes from v4:
  - Per-object debug logging via `name` parameter
  - ICP fail counter: after N consecutive failures, reset to CAPTURE
  - Identity ICP init (no centroid bias)
  - Timing instrumentation on every substep
"""

import torch
import numpy as np
import open3d as o3d
import time
from enum import Enum

try:
    import rospy
    HAS_ROSPY = True
except ImportError:
    HAS_ROSPY = False

def _log(msg):
    if HAS_ROSPY:
        rospy.loginfo(msg)
    else:
        print(msg)


MAX_FUSION_AGE_SEC = 10.0


class PersistentTargetModel:

    class State(Enum):
        CAPTURE  = "capture"
        LOCKED   = "locked"
        OCCLUDED = "occluded"

    def __init__(
        self,
        name="unnamed",            # <-- NEW: identifies this model in logs
        voxel_size=0.003,
        max_points=30000,
        device='cuda',
        depth_valid_threshold=0.3,
        depth_variance_max=0.05,
        min_reference_size=800,
        stable_frames_needed=10,
        icp_max_dist_factor=5.0,
        icp_min_fitness=0.2,
        icp_max_translation=0.10,
        icp_max_iterations=10,
        allow_fusion=True,
        icp_fail_reset=5,          # <-- NEW: reset to CAPTURE after this many consecutive ICP failures
    ):
        self.name = name
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.device = device
        self.depth_valid_threshold = depth_valid_threshold
        self.depth_variance_max = depth_variance_max
        self.min_reference_size = min_reference_size
        self.stable_frames_needed = stable_frames_needed
        self.allow_fusion = allow_fusion
        self.icp_fail_reset = icp_fail_reset

        self.icp_max_dist = voxel_size * icp_max_dist_factor
        self.icp_min_fitness = icp_min_fitness
        self.icp_max_translation = icp_max_translation
        self.icp_max_iterations = icp_max_iterations

        self.state = self.State.CAPTURE
        self._capture_points = None
        self._capture_voxels = None
        self._stable_count = 0
        self._last_size = 0

        self.reference_cloud = None
        self.reference_cloud_gpu = None
        self._point_timestamps = None

        self.just_jumped = False
        self._fusion_counter = 0
        self._update_count = 0
        self._icp_fail_count = 0

        # Timing storage (last values, not averaged — keeps it simple)
        self._timings = {}

        _log(f"[{self.name}] Init: voxel={voxel_size} max_pts={max_points} "
             f"icp_max_dist={self.icp_max_dist:.4f} icp_min_fitness={icp_min_fitness} "
             f"depth_var_max={depth_variance_max} fusion={'ON' if allow_fusion else 'OFF'} "
             f"icp_fail_reset={icp_fail_reset}")

    # ==================================================================
    #  TIMING HELPERS
    # ==================================================================

    def _tick(self, label):
        self._timings[label] = time.perf_counter()

    def _tock(self, label):
        if label in self._timings:
            self._timings[label] = time.perf_counter() - self._timings[label]

    def _ms(self, label):
        """Return last timing in ms, or -1 if not measured."""
        v = self._timings.get(label, -0.001)
        # If it's still a start timestamp (not a duration), return -1
        if v > 1e6:
            return -1.0
        return v * 1000.0

    # ==================================================================
    #  PUBLIC API
    # ==================================================================

    def update(self, mask, depth, T_world_cam, fx, fy, cx, cy):
        self._tick('total')
        self.just_jumped = False
        self._update_count += 1

        if mask is None or np.sum(mask) < 20:
            if self.state != self.State.CAPTURE:
                self.state = self.State.OCCLUDED
                # No mask = tracker lost the object. Do NOT increment icp_fail_count.
                # The reference stays frozen, waiting for re-detection.
            self._tock('total')
            self._log_debug(mask, None, None, False, 0.0, 0.0)
            return self._result(self.state.value, 0.0)
        
        # Mask is back → reset fail counter since tracker is working again
        if self.state == self.State.OCCLUDED:
            self._icp_fail_count = 0

        self._tick('assess_depth')
        valid_ratio, variance = self._assess_depth(mask, depth)
        self._tock('assess_depth')
        depth_ok = (valid_ratio >= self.depth_valid_threshold
                    and variance < self.depth_variance_max)

        if self.state == self.State.CAPTURE:
            r = self._handle_capture(mask, depth, T_world_cam,
                                     fx, fy, cx, cy, depth_ok, valid_ratio)
        elif self.state == self.State.LOCKED:
            r = self._handle_locked(mask, depth, T_world_cam,
                                    fx, fy, cx, cy, depth_ok, valid_ratio)
        else:
            r = self._handle_occluded(mask, depth, T_world_cam,
                                      fx, fy, cx, cy, depth_ok, valid_ratio)

        self._tock('total')
        self._log_debug(mask, valid_ratio, variance, depth_ok, valid_ratio, 0.0)
        return r

    def _log_debug(self, mask, valid_ratio, variance, depth_ok, vr, _):
        """Log every 30 frames — one compact line per object."""
        if self._update_count % 30 != 0:
            return

        mask_px = int(np.sum(mask)) if mask is not None else 0
        parts = [
            f"[{self.name}]",
            f"state={self.state.value}",
            f"pts={self.count()}",
            f"mask={mask_px}px",
        ]

        if valid_ratio is not None:
            parts.append(f"depth_ok={depth_ok}(ratio={valid_ratio:.2f},var={variance:.4f})")

        parts.append(f"icp_fails={self._icp_fail_count}")

        # Timings
        timing_parts = []
        for label in ['total', 'assess_depth', 'unproject', 'icp', 'fuse_capture', 'raycast']:
            ms = self._ms(label)
            if ms >= 0:
                timing_parts.append(f"{label}={ms:.1f}ms")

        if timing_parts:
            parts.append("| " + " ".join(timing_parts))

        _log(" ".join(parts))

    def get_points(self):
        if self.state == self.State.CAPTURE:
            return self._capture_points
        return self.reference_cloud_gpu

    def get_centroid_np(self):
        pts = self.get_points()
        if pts is None or len(pts) == 0:
            return None
        return pts.mean(dim=0).cpu().numpy().astype(np.float32)

    def count(self):
        pts = self.get_points()
        return 0 if pts is None else len(pts)

    def clear(self):
        self.state = self.State.CAPTURE
        self._capture_points = None
        self._capture_voxels = None
        self._stable_count = 0
        self._last_size = 0
        self.reference_cloud = None
        self.reference_cloud_gpu = None
        self._point_timestamps = None
        self._icp_fail_count = 0

    def get_state(self):
        return self.state.value

    # ==================================================================
    #  STATE HANDLERS
    # ==================================================================

    def _handle_capture(self, mask, depth, T, fx, fy, cx, cy, depth_ok, vr):
        if not depth_ok:
            return self._result('capture', vr)

        self._tick('unproject')
        pts = self._unproject(mask, depth, T, fx, fy, cx, cy)
        self._tock('unproject')

        if pts is None:
            return self._result('capture', vr)

        self._tick('fuse_capture')
        self._fuse_capture(pts)
        self._tock('fuse_capture')

        self._check_capture_complete()
        return self._result(self.state.value, vr)

    def _handle_locked(self, mask, depth, T, fx, fy, cx, cy, depth_ok, vr):
        if depth_ok:
            self._tick('unproject')
            partial = self._unproject(mask, depth, T, fx, fy, cx, cy)
            self._tock('unproject')
            if partial is not None and len(partial) > 15:
                self._tick('icp')
                self._icp_align(partial)
                self._tock('icp')
        else:
            self._tick('raycast')
            self._raycast_update(mask, T, fx, fy, cx, cy)
            self._tock('raycast')
        return self._result('locked', vr)

    def _handle_occluded(self, mask, depth, T, fx, fy, cx, cy, depth_ok, vr):
        if depth_ok:
            self._tick('unproject')
            partial = self._unproject(mask, depth, T, fx, fy, cx, cy)
            self._tock('unproject')
            if partial is not None and len(partial) > 15:
                self._tick('icp')
                self._icp_align(partial)
                self._tock('icp')
                self.state = self.State.LOCKED
                return self._result('locked', vr)
        return self._result('occluded', vr)

    # ==================================================================
    #  CAPTURE: VOXEL FUSION + FREEZE
    # ==================================================================

    def _fuse_capture(self, new_pts_np):
        new_pts = torch.from_numpy(new_pts_np).to(self.device)
        new_vox = torch.floor(new_pts / self.voxel_size).long()

        if self._capture_points is not None:
            combined_pts = torch.cat([self._capture_points, new_pts], dim=0)
            combined_vox = torch.cat([self._capture_voxels, new_vox], dim=0)
        else:
            combined_pts = new_pts
            combined_vox = new_vox

        unique_vox, inv = combined_vox.unique(dim=0, return_inverse=True)
        idx = torch.arange(len(inv), device=self.device)
        keep = torch.empty(len(unique_vox), dtype=torch.long, device=self.device)
        keep.scatter_(0, inv, idx)

        self._capture_points = combined_pts[keep]
        self._capture_voxels = unique_vox

        if len(self._capture_points) > self.max_points:
            perm = torch.randperm(len(self._capture_points),
                                  device=self.device)[:self.max_points]
            self._capture_points = self._capture_points[perm]
            self._capture_voxels = self._capture_voxels[perm]

    def _check_capture_complete(self):
        size = len(self._capture_points) if self._capture_points is not None else 0
        if size >= self.min_reference_size:
            if abs(size - self._last_size) < size * 0.05:
                self._stable_count += 1
            else:
                self._stable_count = 0
        else:
            self._stable_count = 0

        if self._update_count % 15 == 0:
            _log(f"[{self.name} CAPTURE] pts={size}/{self.min_reference_size} "
                 f"stable={self._stable_count}/{self.stable_frames_needed}")

        self._last_size = size

        if self._stable_count >= self.stable_frames_needed:
            self._freeze_reference()
            self.state = self.State.LOCKED

    def _freeze_reference(self):
        pts = self._capture_points

        if self.allow_fusion:
            max_ref = 500
        else:
            max_ref = min(len(pts), self.max_points)

        if len(pts) > max_ref:
            perm = torch.randperm(len(pts), device=self.device)[:max_ref]
            pts = pts[perm]

        self.reference_cloud = pts.cpu().numpy().astype(np.float32)
        self.reference_cloud_gpu = pts.clone()

        if self.allow_fusion:
            self._point_timestamps = np.full(len(self.reference_cloud), time.time(), dtype=np.float64)

        self._capture_points = None
        self._capture_voxels = None
        self._stable_count = 0
        self._icp_fail_count = 0
        _log(f"[{self.name}] Reference FROZEN: {len(self.reference_cloud)} pts")

    # ==================================================================
    #  ICP ALIGNMENT
    # ==================================================================

    def _icp_align(self, partial_pts_world):
        if self.reference_cloud is None or len(partial_pts_world) < 15:
            return

        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(self.reference_cloud.astype(np.float64))
        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(partial_pts_world.astype(np.float64))

        src_down = src.voxel_down_sample(self.voxel_size)
        tgt_down = tgt.voxel_down_sample(self.voxel_size)

        radius_normal = self.voxel_size * 4
        src_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        tgt_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        # Compute FPFH features
        radius_feature = self.voxel_size * 6
        src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            src_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=50))
        tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            tgt_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=50))

        src_centroid = np.mean(np.asarray(src_down.points), axis=0)
        tgt_centroid = np.mean(np.asarray(tgt_down.points), axis=0)
        gap = np.linalg.norm(src_centroid - tgt_centroid)
        icp_dist = self.icp_max_dist
        n_src = len(src_down.points)
        n_tgt = len(tgt_down.points)

        if self._update_count % 5 == 0:
            _log(f"[{self.name} PRE-ICP] src={n_src} tgt={n_tgt} gap={gap:.4f}m")

        # ── COARSE: FGR when clouds are far apart ──
        if gap > icp_dist:
            # Only compute features when actually needed
            radius_feature = self.voxel_size * 6
            src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                src_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=50))
            tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                tgt_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=50))

            fgr_result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                src_down, tgt_down, src_fpfh, tgt_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=icp_dist * 3.0
                ),
            )
            init_T = fgr_result.transformation
            if self._update_count % 10 == 0:
                _log(f"[{self.name}] FGR coarse: fitness={fgr_result.fitness:.3f} "
                    f"gap was {gap:.4f}m")
        else:
            init_T = np.eye(4)

        # ── FINE: ICP refinement ──
        loss = o3d.pipelines.registration.TukeyLoss(k=icp_dist)
        result = o3d.pipelines.registration.registration_icp(
            source=src_down, target=tgt_down,
            max_correspondence_distance=icp_dist,
            init=init_T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.icp_max_iterations
            ),
        )

        translation = np.linalg.norm(result.transformation[:3, 3])
        R_mat = result.transformation[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_mat) - 1.0) / 2.0, -1.0, 1.0))
        rmse = result.inlier_rmse

        # Reject: low fitness
        if result.fitness < self.icp_min_fitness:
            self._icp_fail_count += 1
            if self._update_count % 5 == 0:
                _log(f"[{self.name} ICP] FAIL fitness={result.fitness:.3f} rmse={rmse:.4f} "
                    f"fails={self._icp_fail_count}/{self.icp_fail_reset}")
            if self._icp_fail_count >= self.icp_fail_reset:
                _log(f"[{self.name}] ICP failed {self._icp_fail_count}x → RESET to CAPTURE")
                self.state = self.State.CAPTURE
                self._capture_points = None
                self._capture_voxels = None
                self._stable_count = 0
                self._last_size = 0
                self._icp_fail_count = 0
            return

        # Reject: physically impossible
        max_rot = 0.05 if not self.allow_fusion else 0.5
        max_rmse = self.voxel_size * 2.0
        if angle > max_rot or rmse > max_rmse:
            if self._update_count % 10 == 0:
                _log(f"[{self.name} ICP] REJECT trans={translation:.4f}m "
                    f"rot={np.degrees(angle):.1f}deg rmse={rmse:.4f}")
            return

        # Success
        self._icp_fail_count = 0
        T = result.transformation.astype(np.float32)
        self.reference_cloud = self._apply_rigid(self.reference_cloud, T)
        self.reference_cloud_gpu = torch.from_numpy(self.reference_cloud).to(self.device)

        if self._update_count % 30 == 0:
            _log(f"[{self.name} ICP] OK fitness={result.fitness:.3f} rmse={rmse:.4f} "
                f"trans={translation:.4f}m rot={np.degrees(angle):.1f}deg")

        if self.allow_fusion:
            self._fusion_counter += 1
            if self._fusion_counter % 3 == 0:
                self._fuse_locked(partial_pts_world)

    def _fuse_locked(self, partial_pts_world):
        """Fuse new observation into reference. Only called when allow_fusion=True."""
        now = time.time()

        if self._point_timestamps is not None:
            young = (now - self._point_timestamps) < MAX_FUSION_AGE_SEC
            if not np.all(young):
                self.reference_cloud = self.reference_cloud[young]
                self._point_timestamps = self._point_timestamps[young]
                self.reference_cloud_gpu = torch.from_numpy(
                    self.reference_cloud
                ).to(self.device)

        new_pts_gpu = torch.from_numpy(partial_pts_world).to(self.device)
        combined = torch.cat([self.reference_cloud_gpu, new_pts_gpu], dim=0)

        voxels = torch.floor(combined / self.voxel_size).long()
        unique_vox, inv = voxels.unique(dim=0, return_inverse=True)
        idx = torch.arange(len(inv), device=self.device)
        keep = torch.empty(len(unique_vox), dtype=torch.long, device=self.device)
        keep.scatter_(0, inv, idx)
        combined = combined[keep]

        n_old = len(self.reference_cloud)
        old_ts = self._point_timestamps if self._point_timestamps is not None else np.full(n_old, now)
        new_ts = np.full(len(partial_pts_world), now, dtype=np.float64)
        all_ts = np.concatenate([old_ts, new_ts])[keep.cpu().numpy()]

        if len(combined) > self.max_points:
            perm = torch.randperm(len(combined), device=self.device)[:self.max_points]
            combined = combined[perm]
            all_ts = all_ts[perm.cpu().numpy()]

        self.reference_cloud_gpu = combined
        self.reference_cloud = combined.cpu().numpy()
        self._point_timestamps = all_ts

    @staticmethod
    def _apply_rigid(points, T):
        ones = np.ones((len(points), 1), dtype=np.float32)
        pts_h = np.hstack([points, ones])
        return (T @ pts_h.T).T[:, :3].astype(np.float32)

    # ==================================================================
    #  RAYCAST FALLBACK
    # ==================================================================

    def _raycast_update(self, mask, T_world_cam, fx, fy, cx, cy):
        if self.reference_cloud is None or len(self.reference_cloud) == 0:
            return
        vs, us = np.where(mask)
        if len(vs) < 10:
            return

        u_c, v_c = us.mean(), vs.mean()
        d_cam = np.array([(u_c - cx) / fx, (v_c - cy) / fy, 1.0], dtype=np.float64)
        d_cam /= np.linalg.norm(d_cam)

        R = T_world_cam[:3, :3]
        t = T_world_cam[:3, 3]
        d_world = (R @ d_cam).astype(np.float32)
        d_world /= np.linalg.norm(d_world)
        o_world = t.astype(np.float32)

        o_t = torch.from_numpy(o_world).to(self.device)
        d_t = torch.from_numpy(d_world).to(self.device)
        v = self.reference_cloud_gpu - o_t
        proj_len = (v * d_t).sum(dim=1)
        in_front = proj_len > 0
        if not in_front.any():
            return

        perp = v - proj_len.unsqueeze(1) * d_t
        perp_dist = perp.norm(dim=1)
        perp_dist[~in_front] = float('inf')

        closest_idx = perp_dist.argmin()
        hit = o_t + proj_len[closest_idx] * d_t
        centroid = self.reference_cloud_gpu.mean(dim=0)
        shift = hit - centroid

        mag = shift.norm().item()
        if 0.005 < mag < 0.10:
            safe_shift = torch.zeros(3, device=self.device)
            safe_shift[0] = shift[0]
            safe_shift[1] = shift[1]
            self.reference_cloud_gpu += safe_shift
            self.reference_cloud = self.reference_cloud_gpu.cpu().numpy()

    # ==================================================================
    #  DEPTH / UNPROJECT / RESULT
    # ==================================================================

    def _assess_depth(self, mask, depth):
        pixels = depth[mask]
        total = len(pixels)
        if total == 0:
            return 0.0, 999.0
        valid = (pixels > 0.01) & (pixels < 5.0) & np.isfinite(pixels)
        valid_count = np.sum(valid)
        ratio = valid_count / total
        if valid_count < 10:
            return ratio, 999.0
        variance = np.std(pixels[valid])
        return float(ratio), float(variance)

    def _unproject(self, mask, depth, T_world_cam, fx, fy, cx, cy):
        z = depth.copy()
        mask = mask.astype(bool)
        valid = mask & (z > 0.01) & (z < 5.0) & np.isfinite(z)
        if np.sum(valid) < 10:
            return None

        v, u = np.where(valid)

        if len(v) > 100:
            median_z = np.median(z[valid])
            stride = max(1, int(0.3 / max(median_z, 0.05)))
            stride = max(stride, 1 + len(v) // 3000)
            if stride > 1:
                v, u = v[::stride], u[::stride]

        zv = z[v, u]
        pts_cam = np.stack([
            (u - cx) * zv / fx,
            (v - cy) * zv / fy,
            zv
        ], axis=-1).astype(np.float32)
        ones = np.ones((len(pts_cam), 1), dtype=np.float32)
        pts_world = (T_world_cam @ np.hstack([pts_cam, ones]).T).T[:, :3]
        return pts_world.astype(np.float32)

    def _result(self, mode, depth_valid_ratio):
        return {
            'points':             self.get_points(),
            'centroid':           self.get_centroid_np(),
            'mode':               mode,
            'state':              self.state.value,
            'depth_valid_ratio':  depth_valid_ratio,
            'model_size':         self.count(),
            'jumped':             self.just_jumped,
        }