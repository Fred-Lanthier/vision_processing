"""
ContainerTracker — One-shot container bbox from SAM3
=====================================================

Detects the container (bowl/plate/tupperware) once via SAM3,
computes its 3D bounding box from mask + depth, adds a margin,
and uses that box to filter environment points forever after.

No Cutie, no tracking, no VRAM cost after init.
"""

import torch
import numpy as np

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


class ContainerTracker:

    def __init__(self, device='cuda', margin=0.10):
        self.device = device
        self.margin = margin

        # 3D bounding box (world frame)
        self.bbox_min = None         # (3,) numpy float32
        self.bbox_max = None         # (3,) numpy float32
        self.bbox_min_gpu = None     # (3,) GPU tensor
        self.bbox_max_gpu = None     # (3,) GPU tensor

        self.locked = False          # True once bbox is computed

    # ==================================================================
    #  INIT FROM SAM3 — call once
    # ==================================================================

    def init_from_mask(self, mask, depth, T_world_cam, fx, fy, cx, cy):
        """
        Compute the container's 3D bounding box from a single SAM3 mask.

        Args:
            mask:        (H, W) bool numpy array
            depth:       (H, W) float32 numpy in meters
            T_world_cam: (4, 4) camera-to-world transform
            fx, fy, cx, cy: camera intrinsics (floats)
        """
        z = depth.copy() if depth.dtype == np.float32 else depth / 1000.0
        valid = mask & (z > 0.01) & (z < 2.0) & np.isfinite(z)
        if np.sum(valid) < 50:
            _log("[Container] Not enough valid depth points in mask, skipping.")
            return False

        v, u = np.where(valid)
        zv = z[valid]
        pts_cam = np.stack([
            (u - cx) * zv / fx,
            (v - cy) * zv / fy,
            zv
        ], axis=-1).astype(np.float32)

        ones = np.ones((len(pts_cam), 1), dtype=np.float32)
        pts_world = (T_world_cam @ np.hstack([pts_cam, ones]).T).T[:, :3]

        self.bbox_min = (pts_world.min(axis=0) - self.margin).astype(np.float32)
        self.bbox_max = (pts_world.max(axis=0) + self.margin).astype(np.float32)

        self.bbox_min_gpu = torch.from_numpy(self.bbox_min).to(self.device)
        self.bbox_max_gpu = torch.from_numpy(self.bbox_max).to(self.device)

        self.locked = True
        _log(f"[Container] FROZEN bbox: min={self.bbox_min}, max={self.bbox_max}")
        return True

    # ==================================================================
    #  POINT FILTERING — call from the obstacle thread
    # ==================================================================

    def filter_points_gpu(self, points_gpu):
        """
        Keep only points inside the container bbox + margin.

        Args:
            points_gpu: (N, 3) GPU tensor in world frame
        Returns:
            (M, 3) GPU tensor — the filtered subset
        """
        if self.bbox_min_gpu is None:
            return points_gpu

        inside = (
            (points_gpu >= self.bbox_min_gpu) &
            (points_gpu <= self.bbox_max_gpu)
        ).all(dim=1)

        return points_gpu[inside]

    def has_bbox(self):
        return self.locked

    def is_frozen(self):
        return self.locked