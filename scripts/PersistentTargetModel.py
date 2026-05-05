"""
PersistentTargetModel v2 — ICP-based tracking through occlusion
================================================================

State machine with three states:

  CAPTURE   →  Accumulating a reference cloud from good depth frames.
               Once the cloud is large enough and stable, freeze it.

  LOCKED    →  Reference cloud is frozen. Each frame, ICP aligns the
               reference to the current partial observation. Even if
               the fork occludes 80% of the food, the 20% visible is
               enough to keep the full reference cloud positioned.

  OCCLUDED  →  No mask at all (fork fully blocks the view). The
               reference cloud stays frozen at its last known position
               in world frame. Since the food is static, this is safe.

Transitions:
  CAPTURE  → LOCKED    : reference is big enough and stable
  LOCKED   → OCCLUDED  : mask disappears
  OCCLUDED → LOCKED    : mask reappears with good depth → ICP re-aligns

ICP math (simple version):
  Given reference cloud R (Nx3) and partial observation P (Mx3), both
  in world frame, ICP finds the rigid transform T ∈ SE(3) that minimizes:

      T* = argmin_T  Σᵢ ‖T · rᵢ - nn(T · rᵢ, P)‖²

  where nn(x, P) is the nearest neighbor of x in P. Points in R that
  have no neighbor within max_correspondence_distance are ignored —
  this is what makes ICP robust to partial views.

  We then apply T* to ALL points in R, not just the matched ones.
  The unmatched points follow along rigidly (correct for a rigid object).
"""

import torch
import numpy as np
import open3d as o3d
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


class PersistentTargetModel:

    class State(Enum):
        CAPTURE  = "capture"
        LOCKED   = "locked"
        OCCLUDED = "occluded"

    def __init__(
        self,
        voxel_size=0.003,
        max_points=30000,
        device='cuda',
        # Depth quality thresholds
        depth_valid_threshold=0.3,
        depth_variance_max=0.05,
        # Capture → Locked transition
        min_reference_size=800,
        stable_frames_needed=10,
        # ICP parameters
        icp_max_dist_factor=5.0,    # max_correspondence_distance = voxel_size * this
        icp_min_fitness=0.2,        # reject ICP if fewer than 20% of points match
        icp_max_translation=0.10,   # reject ICP if it wants to move > 5cm
        icp_max_iterations=10,
    ):
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.device = device
        self.depth_valid_threshold = depth_valid_threshold
        self.depth_variance_max = depth_variance_max
        self.min_reference_size = min_reference_size
        self.stable_frames_needed = stable_frames_needed

        # ICP config
        self.icp_max_dist = voxel_size * icp_max_dist_factor
        self.icp_min_fitness = icp_min_fitness
        self.icp_max_translation = icp_max_translation
        self.icp_max_iterations = icp_max_iterations

        # ── State ──
        self.state = self.State.CAPTURE

        # ── Capture phase (growing voxel model) ──
        self._capture_points = None   # (M, 3) GPU tensor
        self._capture_voxels = None   # (M, 3) int64 GPU tensor
        self._stable_count = 0
        self._last_size = 0

        # ── Locked/Occluded phase (frozen reference) ──
        self.reference_cloud = None       # (N, 3) numpy float32, world frame
        self.reference_cloud_gpu = None   # (N, 3) GPU tensor

        self.just_jumped = False

    # ==================================================================
    #  PUBLIC API
    # ==================================================================

    def update(self, mask, depth, T_world_cam, fx, fy, cx, cy):
        """
        Call every frame. Returns a dict with tracking info.

        Args:
            mask:        (H, W) bool numpy array from Cutie, or None
            depth:       (H, W) float32 numpy in meters
            T_world_cam: (4, 4) camera-to-world transform from FK
            fx, fy, cx, cy: camera intrinsics (floats)
        """
        self.just_jumped = False

        # ── No mask → either still capturing (do nothing) or go OCCLUDED ──
        if mask is None or np.sum(mask) < 20:
            if self.state != self.State.CAPTURE:
                self.state = self.State.OCCLUDED
            return self._result(self.state.value, 0.0)

        # ── Assess depth quality under the mask ──
        valid_ratio, variance = self._assess_depth(mask, depth)
        depth_ok = (valid_ratio >= self.depth_valid_threshold
                    and variance < self.depth_variance_max)

        # ── Dispatch to state handler ──
        if self.state == self.State.CAPTURE:
            return self._handle_capture(mask, depth, T_world_cam,
                                        fx, fy, cx, cy, depth_ok, valid_ratio)

        elif self.state == self.State.LOCKED:
            return self._handle_locked(mask, depth, T_world_cam,
                                       fx, fy, cx, cy, depth_ok, valid_ratio)

        else:  # OCCLUDED
            return self._handle_occluded(mask, depth, T_world_cam,
                                         fx, fy, cx, cy, depth_ok, valid_ratio)

    def get_points(self):
        """Return the current best point cloud as a GPU tensor."""
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
        """Full reset — go back to CAPTURE."""
        self.state = self.State.CAPTURE
        self._capture_points = None
        self._capture_voxels = None
        self._stable_count = 0
        self._last_size = 0
        self.reference_cloud = None
        self.reference_cloud_gpu = None

    def get_state(self):
        return self.state.value

    # ==================================================================
    #  STATE HANDLERS
    # ==================================================================

    def _handle_capture(self, mask, depth, T, fx, fy, cx, cy, depth_ok, vr):
        """
        CAPTURE state: accumulate points into a voxel grid.
        Once the model is large and stable → freeze as reference → LOCKED.
        """
        if not depth_ok:
            return self._result('capture', vr)

        pts = self._unproject(mask, depth, T, fx, fy, cx, cy)
        if pts is None:
            return self._result('capture', vr)

        self._fuse_capture(pts)
        self._check_capture_complete()

        state_name = self.state.value  # might have just switched to LOCKED
        return self._result(state_name, vr)

    def _handle_locked(self, mask, depth, T, fx, fy, cx, cy, depth_ok, vr):
        """
        LOCKED state: reference cloud exists. Use ICP to track position
        from partial views, or raycast when depth is unreliable.
        """
        if depth_ok:
            partial = self._unproject(mask, depth, T, fx, fy, cx, cy)
            if partial is not None and len(partial) > 50:
                self._icp_align(partial)
        else:
            # Depth is bad (too close) but mask exists → raycast
            self._raycast_update(mask, T, fx, fy, cx, cy)

        return self._result('locked', vr)

    def _handle_occluded(self, mask, depth, T, fx, fy, cx, cy, depth_ok, vr):
        """
        OCCLUDED state: mask just reappeared. If depth is good, ICP
        re-aligns the reference → go back to LOCKED.
        """
        if depth_ok:
            partial = self._unproject(mask, depth, T, fx, fy, cx, cy)
            if partial is not None and len(partial) > 50:
                self._icp_align(partial)
                self.state = self.State.LOCKED
                return self._result('locked', vr)

        return self._result('occluded', vr)

    # ==================================================================
    #  CAPTURE: VOXEL FUSION + FREEZE
    # ==================================================================

    def _fuse_capture(self, new_pts_np):
        """Replace the capture grid with the freshest observation."""
        new_pts = torch.from_numpy(new_pts_np).to(self.device)
        new_vox = torch.floor(new_pts / self.voxel_size).long()

        unique_vox, inv = new_vox.unique(dim=0, return_inverse=True)
        idx = torch.arange(len(inv), device=self.device)
        keep = torch.empty(len(unique_vox), dtype=torch.long, device=self.device)
        keep.scatter_(0, inv, idx)
        
        # ── LE CORRECTIF ──
        # On écrase au lieu d'accumuler. Fini les fantômes !
        self._capture_points = new_pts[keep]
        self._capture_voxels = unique_vox

        # Cap memory
        if len(self._capture_points) > self.max_points:
            perm = torch.randperm(len(self._capture_points),
                                  device=self.device)[:self.max_points]
            self._capture_points = self._capture_points[perm]
            self._capture_voxels = self._capture_voxels[perm]

    def _check_capture_complete(self):
        """Transition CAPTURE → LOCKED when the model is large and stable."""
        size = len(self._capture_points) if self._capture_points is not None else 0

        if size >= self.min_reference_size:
            # "Stable" = size changed by less than 5% since last frame
            if abs(size - self._last_size) < size * 0.05:
                self._stable_count += 1
            else:
                self._stable_count = 0
        else:
            self._stable_count = 0

        self._last_size = size

        if self._stable_count >= self.stable_frames_needed:
            self._freeze_reference()
            self.state = self.State.LOCKED

    def _freeze_reference(self):
        pts = self._capture_points
        # Downsample to keep ICP fast (~5ms instead of 200ms)
        max_ref = 500
        if len(pts) > max_ref:
            perm = torch.randperm(len(pts), device=self.device)[:max_ref]
            pts = pts[perm]
        self.reference_cloud = pts.cpu().numpy().astype(np.float32)
        self.reference_cloud_gpu = pts.clone()
        self._capture_points = None
        self._capture_voxels = None
        self._stable_count = 0
        _log(f"[TargetModel] Reference FROZEN: {len(self.reference_cloud)} pts")

    # ==================================================================
    #  ICP ALIGNMENT
    # ==================================================================

    def _icp_align(self, partial_pts_world):
        """
        Align the frozen reference cloud to a new partial observation
        using Robust Point-to-Plane ICP with Voxel Downsampling.
        Dynamically fuses new points to grow the model and bounding box.
        """
        if self.reference_cloud is None or len(partial_pts_world) < 30:
            return

        # 1. Création des nuages Open3D
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(self.reference_cloud.astype(np.float64))

        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(partial_pts_world.astype(np.float64))

        # 2. Voxel Downsampling
        voxel_size_icp = 0.004 
        src_down = src.voxel_down_sample(voxel_size_icp)
        tgt_down = tgt.voxel_down_sample(voxel_size_icp)

        # 3. Calcul des normales
        radius_normal = voxel_size_icp * 4
        src_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        tgt_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        # 4. Fonction de perte Robuste (Tukey)
        loss = o3d.pipelines.registration.TukeyLoss(k=self.icp_max_dist)

        # 5. ICP Point-To-Plane Robuste
        result = o3d.pipelines.registration.registration_icp(
            source=src_down,
            target=tgt_down,
            max_correspondence_distance=self.icp_max_dist,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.icp_max_iterations
            ),
        )

        # ── Safety checks before applying the transform ──
        if result.fitness < self.icp_min_fitness:
            return

        T = result.transformation.astype(np.float32)
        translation = np.linalg.norm(T[:3, 3])

        if translation < self.icp_max_translation:
            # Apply rigid transform to existing cloud
            self.reference_cloud = self._apply_rigid(self.reference_cloud, T)
            self.reference_cloud_gpu = torch.from_numpy(
                self.reference_cloud
            ).to(self.device)

            # --- MISSING FUSION LOGIC (This makes the Bounding Box Dynamic!) ---
            if getattr(self, '_fusion_counter', 0) % 3 == 0:
                new_pts_gpu = torch.from_numpy(partial_pts_world).to(self.device)
                combined = torch.cat([self.reference_cloud_gpu, new_pts_gpu], dim=0)

                voxels = torch.floor(combined / self.voxel_size).long()
                unique_vox, inv = voxels.unique(dim=0, return_inverse=True)
                idx = torch.arange(len(inv), device=self.device)
                keep = torch.empty(len(unique_vox), dtype=torch.long, device=self.device)
                keep.scatter_(0, inv, idx)
                
                self.reference_cloud_gpu = combined[keep]

                if len(self.reference_cloud_gpu) > self.max_points:
                    perm = torch.randperm(len(self.reference_cloud_gpu),
                                          device=self.device)[:self.max_points]
                    self.reference_cloud_gpu = self.reference_cloud_gpu[perm]

                self.reference_cloud = self.reference_cloud_gpu.cpu().numpy()

            self._fusion_counter = getattr(self, '_fusion_counter', 0) + 1
            # -------------------------------------------------------------------

        else:
            _log(f"[TargetModel] Large ICP shift ({translation:.3f}m) — re-entering CAPTURE")
            self.just_jumped = True
            self.state = self.State.CAPTURE
            
            self._capture_points = torch.from_numpy(partial_pts_world).to(self.device)
            self._capture_voxels = torch.floor(
                self._capture_points / self.voxel_size
            ).long()
            self._stable_count = 0
            self._last_size = len(partial_pts_world)
            self.reference_cloud = None
            self.reference_cloud_gpu = None

    @staticmethod
    def _apply_rigid(points, T):
        """Apply a 4x4 rigid transform to an Nx3 numpy array."""
        ones = np.ones((len(points), 1), dtype=np.float32)
        pts_h = np.hstack([points, ones])
        return (T @ pts_h.T).T[:, :3].astype(np.float32)

    # ==================================================================
    #  RAYCAST FALLBACK (depth is bad, mask still exists)
    # ==================================================================

    def _raycast_update(self, mask, T_world_cam, fx, fy, cx, cy):
        """
        When depth is unreliable (camera too close) but the mask is
        still visible, cast a ray through the mask centroid and nudge
        the reference cloud in X-Y only.
        """
        if self.reference_cloud is None or len(self.reference_cloud) == 0:
            return

        vs, us = np.where(mask)
        if len(vs) < 10:
            return

        # Ray from camera through mask centroid
        u_c, v_c = us.mean(), vs.mean()
        d_cam = np.array([(u_c - cx) / fx, (v_c - cy) / fy, 1.0], dtype=np.float64)
        d_cam /= np.linalg.norm(d_cam)

        R = T_world_cam[:3, :3]
        t = T_world_cam[:3, 3]
        d_world = (R @ d_cam).astype(np.float32)
        d_world /= np.linalg.norm(d_world)
        o_world = t.astype(np.float32)

        # Find closest reference point to the ray
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
            # X-Y only shift (prevent Z-pulling)
            safe_shift = torch.zeros(3, device=self.device)
            safe_shift[0] = shift[0]
            safe_shift[1] = shift[1]
            self.reference_cloud_gpu += safe_shift
            self.reference_cloud = self.reference_cloud_gpu.cpu().numpy()

    # ==================================================================
    #  DEPTH ASSESSMENT
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

    # ==================================================================
    #  UNPROJECTION
    # ==================================================================

    def _unproject(self, mask, depth, T_world_cam, fx, fy, cx, cy):
        """Mask + depth → Nx3 numpy float32 in world frame."""
        z = depth.copy()
        mask = mask.astype(bool)
        valid = mask & (z > 0.01) & (z < 5.0) & np.isfinite(z)
        if np.sum(valid) < 10:
            return None
            
        v, u = np.where(valid)
        
        # --- STRIDE FIX FOR DENSE CLOUDS (Like Bowls) ---
        if len(v) > 3000:
            step = 3
            v, u = v[::step], u[::step]
            
        zv = z[v, u]
        # ------------------------------------------------
        
        pts_cam = np.stack([
            (u - cx) * zv / fx,
            (v - cy) * zv / fy,
            zv
        ], axis=-1).astype(np.float32)
        ones = np.ones((len(pts_cam), 1), dtype=np.float32)
        pts_world = (T_world_cam @ np.hstack([pts_cam, ones]).T).T[:, :3]
        return pts_world.astype(np.float32)

    # ==================================================================
    #  RESULT BUILDER
    # ==================================================================

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