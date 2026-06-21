"""Offline ground-truth clearance for validating the soft barrier h.

The controller (cbf_safety_node_Bernstein) runs on a learned Bernstein-basis
*soft-min* SDF, evaluated over a pruned/selected obstacle subset, and only for
six protected links. This module recomputes, offline, the TRUE geometric
clearance of those same six links against the FULL obstacle cloud, so the logged
h(t) can be plotted against reality in h_evolution.

Method (no rtree/embree needed): each protected link mesh is sampled once on its
surface in local coordinates; the samples are transformed to the base frame by
the link's forward-kinematics pose at every trajectory step (batched on GPU);
the minimum distance to the obstacle cloud is read off a KD-tree. This is the
surface-to-surface clearance, accurate to the surface-sampling resolution.

The six protected links match cbf_safety_node_Bernstein.py exactly; keep them in
sync if the node's protected set changes.
"""

import os
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

# Must match `link_names` in cbf_safety_node_Bernstein.py.
PROTECTED_LINKS = ['panda_link4', 'panda_link5', 'panda_link6',
                   'panda_link7', 'panda_hand', 'fork_tip']


def _pkg_path():
    try:
        import rospkg
        return rospkg.RosPack().get_path('vision_processing')
    except Exception:
        # scripts/ lives directly under the package root
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RealClearance:
    """Exact min distance between the protected robot meshes and a point cloud."""

    def __init__(self, samples_per_link=1500, device=None, pkg_path=None):
        import torch
        import trimesh
        import xacro

        pkg_path = pkg_path or _pkg_path()
        sys.path.insert(0, pkg_path)
        from third_party.RDF.urdf_layer import URDFLayer

        doc = xacro.process_file(pkg_path + '/urdf/panda_camera.xacro')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(doc.toxml())
            urdf_path = f.name

        self.torch = torch
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # voxel_dir=None on purpose: we want the TRUE robot geometry (franka
        # visual meshes + the real fork_tip.stl), not the coarse voxelised
        # meshes the loader would otherwise substitute by basename.
        self.layer = URDFLayer(
            urdf_path=urdf_path, device=self.device, package_dir=pkg_path,
            voxel_dir=None)

        # Sample each protected mesh once, in local (scaled, metres) coordinates.
        # mesh_idx indexes into the per-mesh transform list returned by
        # get_transformations_each_link (same order as meshes_info).
        self.mesh_idx = []
        self.local_pts = []
        for i, info in enumerate(self.layer.meshes_info):
            if info['link_name'] not in PROTECTED_LINKS:
                continue
            mesh = trimesh.load(info['mesh_path'], force='mesh')
            pts, _ = trimesh.sample.sample_surface(mesh, samples_per_link)
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
        missing = [l for l in PROTECTED_LINKS if l not in self.found_links]
        if missing:
            print(f"[real_distance] WARNING: no resolvable mesh for {missing}; "
                  "those links are excluded from the real clearance.")
        if not self.mesh_idx:
            raise RuntimeError("No protected-link meshes could be loaded.")

    def surface_points(self, q_rows):
        """[T,7] joint angles -> [T,P,3] base-frame robot surface points."""
        torch = self.torch
        q = torch.as_tensor(np.asarray(q_rows, np.float32), device=self.device)
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

        q_rows:     [T,7] joint angles, one per output sample
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
