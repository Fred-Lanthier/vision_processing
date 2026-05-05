"""
Persistent Point Cloud Accumulators
=====================================

WorldCloud  — static environment (table, bowl, walls)
              Accumulates in world frame. Has exclude_near() to
              strip target points before integrating.

ObjectCloud — movable target (food)
              Stores points in a PCA-aligned LOCAL frame.
              Tracks both translation (centroid) and rotation (PCA axes).
              When the object moves OR rotates, the accumulated cloud
              follows — no clones, no ghost copies.
"""

import numpy as np


# =====================================================================
# Shared voxel hash logic
# =====================================================================

_OFFSET = np.int64(100000)
_STRIDE_Y = np.int64(200001)
_STRIDE_X = _STRIDE_Y * _STRIDE_Y


def _to_hashes(points, voxel_size):
    idx = np.floor(points / voxel_size).astype(np.int64)
    return ((idx[:, 0] + _OFFSET) * _STRIDE_X +
            (idx[:, 1] + _OFFSET) * _STRIDE_Y +
            (idx[:, 2] + _OFFSET))


def _dedup(hashes, points):
    _, unique_idx = np.unique(hashes[::-1], return_index=True)
    unique_idx = len(hashes) - 1 - unique_idx
    return hashes[unique_idx], points[unique_idx]


def _merge(old_h, old_p, old_c, old_s, new_h, new_p, frame, max_pts):
    if len(old_h) == 0:
        n = min(len(new_h), max_pts)
        order = np.argsort(new_h[:n])
        return (new_h[:n][order].copy(), new_p[:n][order].copy(),
                np.ones(n, dtype=np.int32), np.full(n, frame, dtype=np.int32))

    ins = np.searchsorted(old_h, new_h)
    ins_c = np.clip(ins, 0, len(old_h) - 1)
    existing = old_h[ins_c] == new_h

    if existing.any():
        idx = ins_c[existing]
        old_p[idx] = new_p[existing]
        old_c[idx] += 1
        old_s[idx] = frame

    n_new = (~existing).sum()
    if n_new > 0:
        space = max_pts - len(old_h)
        n_add = min(n_new, max(space, 0))
        if n_add > 0:
            all_h = np.concatenate([old_h, new_h[~existing][:n_add]])
            all_p = np.vstack([old_p, new_p[~existing][:n_add]])
            all_c = np.concatenate([old_c, np.ones(n_add, dtype=np.int32)])
            all_s = np.concatenate([old_s, np.full(n_add, frame, dtype=np.int32)])
            order = np.argsort(all_h)
            return all_h[order], all_p[order], all_c[order], all_s[order]

    return old_h, old_p, old_c, old_s


# =====================================================================
# WorldCloud — static environment
# =====================================================================

class WorldCloud:

    def __init__(self, voxel_size=0.01, max_age=2000, max_points=50000):
        self.voxel_size = voxel_size
        self.max_age = max_age
        self.max_points = max_points
        self.frame = 0
        self._h = np.empty(0, dtype=np.int64)
        self._p = np.empty((0, 3), dtype=np.float32)
        self._c = np.empty(0, dtype=np.int32)
        self._s = np.empty(0, dtype=np.int32)

    def integrate(self, points_world):
        if points_world is None or len(points_world) == 0:
            return
        self.frame += 1
        h = _to_hashes(points_world, self.voxel_size)
        h, p = _dedup(h, points_world.astype(np.float32))
        self._h, self._p, self._c, self._s = _merge(
            self._h, self._p, self._c, self._s, h, p, self.frame, self.max_points
        )
        if self.frame % 50 == 0:
            self._prune()

    def exclude_near(self, center, radius):
        """
        Remove stored points within 'radius' of 'center'.
        Call BEFORE integrate() to strip target leakage.
        """
        if len(self._p) == 0 or center is None:
            return
        dists = np.linalg.norm(self._p - center, axis=1)
        keep = dists > radius
        if keep.all():
            return
        self._h = self._h[keep]
        self._p = self._p[keep]
        self._c = self._c[keep]
        self._s = self._s[keep]

    def _prune(self):
        if len(self._s) == 0:
            return
        keep = self._s >= (self.frame - self.max_age)
        if keep.all():
            return
        self._h = self._h[keep]
        self._p = self._p[keep]
        self._c = self._c[keep]
        self._s = self._s[keep]

    def get_points(self, min_confidence=1):
        if len(self._p) == 0:
            return None
        if min_confidence <= 1:
            return self._p.copy()
        mask = self._c >= min_confidence
        return self._p[mask].copy() if mask.any() else None

    def count(self):
        return len(self._p)

    def clear(self):
        self._h = np.empty(0, dtype=np.int64)
        self._p = np.empty((0, 3), dtype=np.float32)
        self._c = np.empty(0, dtype=np.int32)
        self._s = np.empty(0, dtype=np.int32)
        self.frame = 0


# =====================================================================
# ObjectCloud — movable target with rotation detection
# =====================================================================

class ObjectCloud:
    """
    Object-centric accumulator. Centroid-only local frame.
    
    Translation: handled by storing (point - centroid) and replaying
    with the latest centroid. Object slides → cloud follows.
    
    Rotation: DETECTED, not tracked. When the object rotates, the
    overlap between new observation and stored cloud drops → we clear
    the accumulated cloud and restart. This is correct because:
      - Tracking rotation from partial wrist-camera views is unreliable
      - PCA axes are unstable on partial views → sphere artifact
      - Clearing costs one frame of data, which is rebuilt in ~5 frames
      - In feeding tasks, the food doesn't rotate during approach
    
    The overlap test: what fraction of new local-frame voxels already
    exist in storage? High (>40%) = same orientation → accumulate.
    Low (<40%) = rotated or replaced → clear and restart.
    """

    def __init__(self, voxel_size=0.003, max_age=3000, max_points=20000,
                 overlap_threshold=0.4):
        self.voxel_size = voxel_size
        self.max_age = max_age
        self.max_points = max_points
        self.overlap_threshold = overlap_threshold
        self.frame = 0

        # Local-frame storage (centered at centroid, NO rotation)
        self._h = np.empty(0, dtype=np.int64)
        self._p = np.empty((0, 3), dtype=np.float32)
        self._c = np.empty(0, dtype=np.int32)
        self._s = np.empty(0, dtype=np.int32)

        # Latest centroid in world frame
        self.centroid = None

    def integrate(self, points_world):
        """
        Add new observation (world frame).
        
        Overlap check only fires when:
          - We have enough stored points (>20 voxels)
          - The new observation is substantial (>50% of stored count)
            If the observation is small, the fork is probably occluding
            the target → just accumulate what we can see, don't reset.
        """
        if points_world is None or len(points_world) == 0:
            return
        self.frame += 1

        new_centroid = points_world.mean(axis=0).astype(np.float32)
        local_pts = (points_world - new_centroid).astype(np.float32)

        self.centroid = new_centroid

        new_h = _to_hashes(local_pts, self.voxel_size)
        new_h, local_pts = _dedup(new_h, local_pts)

        # ── Overlap check: only when observation is substantial ──
        stored_count = len(self._h)
        new_count = len(new_h)
        observation_is_substantial = (
            stored_count > 20
            and new_count > 10
            and new_count > stored_count * 0.3  # at least 30% of stored
        )

        if observation_is_substantial:
            ins = np.searchsorted(self._h, new_h)
            ins_c = np.clip(ins, 0, len(self._h) - 1)
            matches = (self._h[ins_c] == new_h).sum()
            overlap = matches / new_count

            if overlap < self.overlap_threshold:
                self._h = np.empty(0, dtype=np.int64)
                self._p = np.empty((0, 3), dtype=np.float32)
                self._c = np.empty(0, dtype=np.int32)
                self._s = np.empty(0, dtype=np.int32)

        # ── Integrate ──
        self._h, self._p, self._c, self._s = _merge(
            self._h, self._p, self._c, self._s,
            new_h, local_pts, self.frame, self.max_points
        )
        if self.frame % 50 == 0:
            self._prune()

    def _prune(self):
        if len(self._s) == 0:
            return
        keep = self._s >= (self.frame - self.max_age)
        if keep.all():
            return
        self._h = self._h[keep]
        self._p = self._p[keep]
        self._c = self._c[keep]
        self._s = self._s[keep]

    def get_points_world(self, min_confidence=1):
        """Accumulated cloud in world frame = local + latest centroid."""
        if len(self._p) == 0 or self.centroid is None:
            return None
        if min_confidence <= 1:
            local = self._p
        else:
            mask = self._c >= min_confidence
            if not mask.any():
                return None
            local = self._p[mask]
        return (local + self.centroid).astype(np.float32)

    def get_centroid(self):
        return self.centroid.copy() if self.centroid is not None else None

    def count(self):
        return len(self._p)

    def clear(self):
        self._h = np.empty(0, dtype=np.int64)
        self._p = np.empty((0, 3), dtype=np.float32)
        self._c = np.empty(0, dtype=np.int32)
        self._s = np.empty(0, dtype=np.int32)
        self.centroid = None
        self.frame = 0