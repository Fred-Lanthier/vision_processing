"""
ForkFilter — constant camera-frame fork occlusion filter.

The fork and the wrist camera are both rigidly attached to panda_TCP, so the
fork occupies a FIXED position in camera_wrist_optical_frame regardless of arm
configuration.  We exploit this to filter fork points with:

  Strategy C (primary)  — zero fork pixels in the raw depth image before any
                          3D projection, so fork points never become obstacles.
  Strategy B (backup)   — 3D distance filter on points still in camera frame,
                          catching pixels that leaked through the depth-edge
                          boundary of the 2D mask. Two flavours:
                            • capsule  — cheap analytic cylinder, loose
                            • sdf      — distance to the TRUE fork surface
                                         (fork_tip.stl), tight 5 mm shell.
                          The SDF flavour hugs the real tine geometry, so it
                          removes near-fork leaks without erasing real obstacles
                          a centimetre away (the bowl rim during the grasp),
                          which the 3 cm capsule cannot do.

All geometry is computed ONCE at startup from the URDF joint chain (the fork is
rigid in the camera frame), so the filter is a constant KD-tree query at runtime
— no per-frame FK.
"""

import os

import numpy as np
import cv2
from scipy.spatial.transform import Rotation


def _make_T(xyz, rpy_xyz):
    """Build a 4x4 homogeneous transform from xyz + intrinsic XYZ Euler angles."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler('xyz', rpy_xyz).as_matrix()
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


class ForkFilter:
    """
    Tunable parameters (exposed as ROS params in point_cloud_projector_node):

        fork_capsule_length   [m]     length of the prong capsule         (default 0.15)
        fork_capsule_radius   [m]     capsule bounding radius              (default 0.06)
        fork_prong_axis             unit vector in fork_tip LOCAL frame
                                    along which the prongs extend.
                                    Tune with RViz until the green marker
                                    covers the fork.                      (default [1,0,0])
        fork_mask_dilation_px [px]   dilation applied to the 2D mask
                                    to absorb depth-edge noise            (default 10)
    """

    def __init__(self,
                 capsule_length=0.15,
                 capsule_radius=0.03,
                 prong_axis_local=None,
                 mask_dilation_px=5,
                 sdf_filter=False,
                 sdf_margin=0.005,
                 fork_mesh_path=None,
                 sdf_surface_samples=4000):

        self.capsule_radius  = float(capsule_radius)
        self.mask_dilation_px = int(mask_dilation_px)
        self.sdf_margin       = float(sdf_margin)
        self.sdf_active       = False        # set True iff the KD-tree built ok
        self._fork_kdt        = None

        # [0,0,-1] is the fork_tip local axis that keeps both capsule endpoints
        # inside the image frame and extends in the physically correct direction
        # (away from camera, toward the food).  Tune via fork_prong_axis ROS param.
        axis = np.asarray(prong_axis_local if prong_axis_local is not None
                          else [0.0, 0.0, -1.0], dtype=np.float64)
        axis /= np.linalg.norm(axis)

        # ── Constant joint chain from panda_camera.xacro ─────────────────────
        # panda_TCP  →  fork_tip
        T_tcp_fork = _make_T(
            xyz=[-0.0055, 0.0, 0.1296],
            rpy_xyz=[0.0, -3.6215581978882336, 0.0],
        )
        # panda_TCP  →  camera_wrist_link
        T_tcp_cam_wrist = _make_T(
            xyz=[-0.052, 0.035, -0.045],
            rpy_xyz=[0.0, -np.pi / 2, 0.0],
        )
        # camera_wrist_link  →  camera_wrist_optical_frame
        T_cam_wrist_optical = _make_T(
            xyz=[0.0, 0.0, 0.0],
            rpy_xyz=[-np.pi / 2, 0.0, -np.pi / 2],
        )

        T_tcp_camera         = T_tcp_cam_wrist @ T_cam_wrist_optical
        T_fork_in_camera     = np.linalg.inv(T_tcp_camera) @ T_tcp_fork

        # Capsule endpoints in camera_wrist_optical_frame (constant forever)
        P1 = T_fork_in_camera[:3, 3]
        prong_dir_cam = T_fork_in_camera[:3, :3] @ axis
        P2 = P1 + float(capsule_length) * prong_dir_cam

        self.P1           = P1
        self.P2           = P2
        self._seg         = P2 - P1
        self._seg_len_sq  = float(self._seg @ self._seg) + 1e-8
        self.pixel_mask   = None   # populated by compute_pixel_mask()

        # Optional SDF backup: a KD-tree of the true fork surface in camera frame.
        if sdf_filter:
            self._build_fork_surface(fork_mesh_path, sdf_surface_samples,
                                     T_fork_in_camera)

    # ── SDF backup setup (constant, built once) ───────────────────────────────

    def _build_fork_surface(self, mesh_path, n_samples, T_fork_in_camera):
        """Sample fork_tip.stl and store its surface in the camera frame.

        The fork is rigid in camera_wrist_optical_frame, so the surface points
        are constant: the runtime filter is a single KD-tree query for the
        distance from each candidate point to the true fork surface. On any
        failure we log and leave sdf_active False so the caller falls back to
        the capsule.
        """
        try:
            import trimesh
            from scipy.spatial import cKDTree
        except Exception as e:                      # pragma: no cover
            print(f"[fork_filter] SDF backup disabled (deps missing: {e!r}); "
                  "using capsule.")
            return
        if not mesh_path or not os.path.isfile(mesh_path):
            print(f"[fork_filter] SDF backup disabled (mesh not found: "
                  f"{mesh_path}); using capsule.")
            return
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            pts, _ = trimesh.sample.sample_surface(mesh, int(n_samples))
            pts = np.asarray(pts, np.float64) * 0.001   # fork_tip.stl scale (mm→m)
            # Visual-origin offset: fork_tip.stl mesh frame → fork_tip LINK frame
            # (panda_camera.xacro <visual><origin> of link "fork_tip").
            T_link_visual = _make_T(
                xyz=[-0.033, -0.02, 0.0171378],
                rpy_xyz=[0.0, np.deg2rad(27.5), 0.0],
            )
            T = T_fork_in_camera @ T_link_visual        # mesh → camera frame
            pts_cam = (T[:3, :3] @ pts.T).T + T[:3, 3]
            self._fork_kdt = cKDTree(pts_cam)
            self.sdf_active = True
            print(f"[fork_filter] SDF backup active: {len(pts_cam)} fork-surface "
                  f"samples, {self.sdf_margin * 1e3:.0f} mm shell.")
        except Exception as e:                          # pragma: no cover
            print(f"[fork_filter] SDF backup disabled (build failed: {e!r}); "
                  "using capsule.")

    # ── Strategy C: 2D depth mask ─────────────────────────────────────────────

    def compute_pixel_mask(self, fx, fy, cx, cy, H, W):
        """Return a bool (H×W) mask of pixels that contain the fork.

        Samples 60 cross-sections along P1→P2, projects each centre to (u,v)
        with a circle of the correctly projected radius, unions the circles,
        then dilates by mask_dilation_px for depth-noise margin.
        Stores the result in self.pixel_mask and also returns it.
        """
        canvas = np.zeros((H, W), dtype=np.uint8)
        for i in range(61):
            t  = i / 60.0
            pt = self.P1 + t * self._seg
            Z  = pt[2]
            if Z <= 0.02:
                continue
            u     = int(fx * pt[0] / Z + cx)
            v     = int(fy * pt[1] / Z + cy)
            r_px  = max(4, int(fx * self.capsule_radius / Z))
            if 0 <= u < W and 0 <= v < H:
                cv2.circle(canvas, (u, v), r_px, 255, -1)

        if self.mask_dilation_px > 0:
            k      = self.mask_dilation_px
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (2 * k + 1, 2 * k + 1))
            canvas = cv2.dilate(canvas, kernel)

        self.pixel_mask = canvas.astype(bool)
        return self.pixel_mask

    # ── Strategy B: 3D camera-frame capsule filter ────────────────────────────

    def filter_camera_frame_points(self, pts):
        """Remove points lying on/near the fork, in camera_wrist_optical_frame.

        Uses the true-surface SDF shell when available (sdf_active), otherwise
        the analytic capsule. pts : (N, 3) array; returns (M, 3), M ≤ N.
        """
        if pts is None or len(pts) == 0:
            return pts
        if self.sdf_active:
            # Distance from each candidate point to the true fork surface.
            dist, _ = self._fork_kdt.query(np.asarray(pts), k=1)
            return pts[dist > self.sdf_margin]
        # Capsule fallback: distance to the P1-P2 segment.
        w       = pts - self.P1                                    # (N, 3)
        t       = np.clip(w @ self._seg / self._seg_len_sq, 0.0, 1.0)  # (N,)
        closest = self.P1 + t[:, None] * self._seg                # (N, 3)
        dist    = np.linalg.norm(pts - closest, axis=1)           # (N,)
        return pts[dist > self.capsule_radius]

    # ── Debug helpers ─────────────────────────────────────────────────────────

    def capsule_endpoints_camera_frame(self):
        """Return (P1, P2) for RViz marker publishing."""
        return self.P1.copy(), self.P2.copy()
