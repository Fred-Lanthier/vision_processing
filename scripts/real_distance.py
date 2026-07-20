"""Offline mesh clearance for validating the learned CBF barrier.

The pick-and-place controller protects links 2--7, the hand, and both fingers.
This module recomputes the geometric clearance of those same meshes against the
full obstacle cloud.  Crucially, FK consumes all nine measured Panda joints:
the seven arm joints and both prismatic finger joints.  Padding a seven-joint
trajectory with zeros makes an open gripper look closed and produces a false
clearance drop when the fingers move, so that input is rejected here.

Method (no rtree/embree needed): each protected link mesh is sampled once on its
surface in local coordinates; the samples are transformed to the base frame by
the link's forward-kinematics pose at every trajectory step (batched on GPU);
the minimum distance to the obstacle cloud is read off a KD-tree. This is the
surface-to-surface clearance, accurate to the surface-sampling resolution.

The legacy feeding geometry remains available explicitly through
``robot_variant="feeding"``.
"""

import os
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

PANDA_JOINTS = [
    'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
    'panda_joint5', 'panda_joint6', 'panda_joint7',
    'panda_finger_joint1', 'panda_finger_joint2',
]

# Must match ~cbf_link_names in the corresponding launch.
ROBOT_VARIANTS = {
    'pickplace': {
        'urdf': 'urdf/panda_pickplace.xacro',
        'protected_links': [
            'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5',
            'panda_link6', 'panda_link7', 'panda_hand',
            'panda_rightfinger', 'panda_leftfinger',
        ],
    },
    'feeding': {
        'urdf': 'urdf/panda_camera.xacro',
        'protected_links': [
            'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7',
            'panda_hand', 'fork_tip',
        ],
    },
}


def _pkg_path():
    try:
        import rospkg
        return rospkg.RosPack().get_path('vision_processing')
    except Exception:
        # scripts/ lives directly under the package root
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RealClearance:
    """Exact min distance between the protected robot meshes and a point cloud."""

    def __init__(self, samples_per_link=1500, device=None, pkg_path=None,
                 robot_variant='pickplace', sample_seed=0):
        import torch
        import trimesh
        import xacro

        pkg_path = pkg_path or _pkg_path()
        if robot_variant not in ROBOT_VARIANTS:
            raise ValueError(
                f"unknown robot_variant {robot_variant!r}; expected one of "
                f"{sorted(ROBOT_VARIANTS)}")
        config = ROBOT_VARIANTS[robot_variant]
        self.robot_variant = robot_variant
        self.protected_links = list(config['protected_links'])
        sys.path.insert(0, pkg_path)
        from third_party.RDF.urdf_layer import URDFLayer

        self.torch = torch
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        doc = xacro.process_file(os.path.join(pkg_path, config['urdf']))
        temp_urdf = tempfile.NamedTemporaryFile(
            mode='w', suffix='.urdf', delete=False)
        try:
            temp_urdf.write(doc.toxml())
            temp_urdf.close()
            # voxel_dir=None on purpose: use the actual visual meshes rather
            # than the coarse voxelised meshes used by the learned SDF loader.
            self.layer = URDFLayer(
                urdf_path=temp_urdf.name, device=self.device,
                package_dir=pkg_path, voxel_dir=None)
        finally:
            temp_urdf.close()
            if os.path.exists(temp_urdf.name):
                os.unlink(temp_urdf.name)

        self.joint_names = list(self.layer.joint_names)
        if self.joint_names != PANDA_JOINTS:
            raise RuntimeError(
                "offline FK joint order does not match the bag loader: "
                f"URDF={self.joint_names}, expected={PANDA_JOINTS}")

        # Sample each protected mesh once, in local (scaled, metres) coordinates.
        # mesh_idx indexes into the per-mesh transform list returned by
        # get_transformations_each_link (same order as meshes_info).
        self.mesh_idx = []
        self.local_pts = []
        for i, info in enumerate(self.layer.meshes_info):
            if info['link_name'] not in self.protected_links:
                continue
            mesh = trimesh.load(info['mesh_path'], force='mesh')
            # A deterministic seed keeps repeated thesis-figure generation
            # from changing the reported minimum by sampling noise.
            pts, _ = trimesh.sample.sample_surface(
                mesh, samples_per_link, seed=sample_seed + i)
            pts = np.asarray(pts, np.float32) * np.asarray(info['scale'], np.float32)
            self.mesh_idx.append(i)
            self.local_pts.append(pts)

        self.found_links = sorted({self.layer.meshes_info[i]['link_name']
                                   for i in self.mesh_idx})
        # Per-surface-point link name (length P, same concatenation order as
        # surface_points), so a min-distance point can be attributed to a link.
        self._pt_link = (np.concatenate([
            np.full(len(self.local_pts[j]),
                    self.layer.meshes_info[mi]['link_name'], dtype=object)
            for j, mi in enumerate(self.mesh_idx)])
            if self.mesh_idx else np.empty(0, dtype=object))
        missing = [l for l in self.protected_links if l not in self.found_links]
        if missing:
            print(f"[real_distance] WARNING: no resolvable mesh for {missing}; "
                  "those links are excluded from the real clearance.")
        if not self.mesh_idx:
            raise RuntimeError("No protected-link meshes could be loaded.")

    def surface_points(self, q_rows):
        """[T,9] measured joints -> [T,P,3] robot surface points.

        The last two columns are the live prismatic finger positions.  A
        seven-column input is rejected instead of being silently padded closed.
        """
        torch = self.torch
        q_array = np.asarray(q_rows, np.float32)
        if q_array.ndim == 1:
            q_array = q_array[None]
        if q_array.ndim != 2 or q_array.shape[1] != len(self.joint_names):
            raise ValueError(
                f"q_rows must have shape [T,{len(self.joint_names)}] in order "
                f"{self.joint_names}; got {q_array.shape}. Both measured "
                "finger joints are required for physical clearance.")
        if not np.isfinite(q_array).all():
            raise ValueError("q_rows contains non-finite joint positions")
        q = torch.as_tensor(q_array, device=self.device)
        if q.ndim == 1:
            q = q[None]
        pose = torch.eye(4, device=self.device).expand(q.shape[0], 4, 4)
        trans = self.layer.get_transformations_each_link(pose, q)  # list [B,4,4]
        parts = []
        for j, mi in enumerate(self.mesh_idx):
            T = trans[mi]                                   # [B,4,4]
            lp = torch.as_tensor(self.local_pts[j], device=self.device)  # [S,3]
            R, t = T[:, :3, :3], T[:, :3, 3]
            parts.append(torch.einsum('bij,sj->bsi', R, lp) + t[:, None, :])
        return torch.cat(parts, dim=1).detach().cpu().numpy()    # [B,P,3]

    def clearance(self, q_rows, obs_clouds, obs_index, batch=256,
                  return_details=False):
        """Minimum surface-to-cloud distance per trajectory step.

        q_rows:     [T,9] measured joint positions, including both fingers
        obs_clouds: list of [Ni,3] point clouds (base frame)
        obs_index:  [T] index into obs_clouds for each step
        returns:    [T] distances in metres (NaN where no usable cloud)

        If ``return_details`` is True, also returns a dict attributing each
        step's minimum to its source:
            'link'     [T] object  — protected link name owning the closest
                                     robot surface point (None where no cloud)
            'robot_pt' [T,3] float — that closest robot surface point (base frame)
            'obs_pt'   [T,3] float — the nearest obstacle point it matched to
        This tells a real wall approach (obs_pt on the obstacle, robot_pt on an
        arm link) apart from a self/swept-point artifact (link == 'fork_tip' and
        obs_pt sitting on the fork body).
        """
        trees = [cKDTree(c) if (c is not None and len(c) >= 1) else None
                 for c in obs_clouds]
        T = len(q_rows)
        d = np.full(T, np.nan)
        if return_details:
            links = np.full(T, None, dtype=object)
            robot_pt = np.full((T, 3), np.nan)
            obs_pt = np.full((T, 3), np.nan)
        for start in range(0, T, batch):
            sl = slice(start, min(start + batch, T))
            pts = self.surface_points(q_rows[sl])           # [b,P,3]
            for k in range(pts.shape[0]):
                gi = start + k
                ci = obs_index[gi]
                tree = trees[ci]
                if tree is None:
                    continue
                dist, oidx = tree.query(pts[k], k=1)        # [P], [P]
                pmin = int(np.argmin(dist))
                d[gi] = float(dist[pmin])
                if return_details:
                    links[gi] = self._pt_link[pmin]
                    robot_pt[gi] = pts[k][pmin]
                    obs_pt[gi] = obs_clouds[ci][oidx[pmin]]
        if return_details:
            return d, {"link": links, "robot_pt": robot_pt, "obs_pt": obs_pt}
        return d
