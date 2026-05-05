"""
GPU-Only Real-Time SDF — No scipy, No CPU round-trip.

Two backends:
  A) Grid SDF via CuPy EDT (GPU) or SciPy EDT (CPU)
     - Builds a full 3D distance field
     - Query via grid_sample (trilinear, differentiable)
  
  B) Direct nearest-neighbor query via cdist
     - No grid — stores obstacle points, queries via cdist
     - Best for small number of probe points (< 200)

Both compute the OmniGuide collision energy (Eq. 15):
    L_C(x) = -log(SDF(x))    for 0 < SDF(x) <= d
"""

import torch
import torch.nn.functional as F
import numpy as np
import time

# ── GPU EDT (optional — only if cupy is available) ──
try:
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as cupy_edt
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# ── CPU EDT (always available) ──
from scipy.ndimage import distance_transform_edt as scipy_edt


def gpu_edt(occupancy_torch: torch.BoolTensor) -> torch.FloatTensor:
    """EDT on GPU via CuPy. ~2-5ms for 130x100x80."""
    assert HAS_CUPY, "CuPy not available — use cpu_edt instead"

    torch.cuda.current_stream().synchronize()
    occ_cupy = cp.from_dlpack(torch.utils.dlpack.to_dlpack(occupancy_torch))
    sdf_cupy = cupy_edt(~occ_cupy).astype(cp.float32)
    cp.cuda.get_current_stream().synchronize()
    sdf_torch = torch.from_dlpack(sdf_cupy.toDlpack())
    return sdf_torch


def cpu_edt(occupancy_torch: torch.BoolTensor) -> torch.FloatTensor:
    """EDT on CPU via SciPy. ~10-20ms for 100x100x80."""
    occ_np = occupancy_torch.cpu().numpy()
    sdf_np = scipy_edt(~occ_np).astype(np.float32)
    return torch.from_numpy(sdf_np)


class VoxelSDF_GPU:
    """
    Name kept for backward compatibility, but now supports device='cpu'.
    """

    def __init__(self, voxel_size: float, bounds: tuple, device="cuda"):
        self.voxel_size = voxel_size
        self.device = device

        self.mins = torch.tensor([b[0] for b in bounds], dtype=torch.float32, device=device)
        self.maxs = torch.tensor([b[1] for b in bounds], dtype=torch.float32, device=device)
        self.grid_shape = ((self.maxs - self.mins) / voxel_size).ceil().long()

        self.sdf_grid = None
        self.obstacle_pts = None
        self._last_update_ms = 0.0

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------
    def update(self, points):
        t0 = time.perf_counter()

        if isinstance(points, np.ndarray):
            pts = torch.from_numpy(points.astype(np.float32)).to(self.device)
        else:
            pts = points.to(self.device).float()

        self.obstacle_pts = pts

        D, H, W = self.grid_shape.tolist()
        indices = ((pts - self.mins) / self.voxel_size).long()
        valid = (
            (indices[:, 0] >= 0) & (indices[:, 0] < D) &
            (indices[:, 1] >= 0) & (indices[:, 1] < H) &
            (indices[:, 2] >= 0) & (indices[:, 2] < W)
        )
        indices = indices[valid]

        occupancy = torch.zeros(D, H, W, dtype=torch.bool, device=self.device)
        occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = True

        # Pick EDT backend based on device
        if self.device == 'cpu' or str(self.device) == 'cpu':
            sdf_voxels = cpu_edt(occupancy)
        elif HAS_CUPY:
            sdf_voxels = gpu_edt(occupancy)
        else:
            # Fallback: move to CPU, run scipy, move back
            sdf_voxels = cpu_edt(occupancy).to(self.device)

        self.sdf_grid = (sdf_voxels * self.voxel_size).unsqueeze(0).unsqueeze(0)

        self._last_update_ms = (time.perf_counter() - t0) * 1000
        return self._last_update_ms

    # ------------------------------------------------------------------
    # GRID QUERY
    # ------------------------------------------------------------------
    def _world_to_grid(self, positions: torch.Tensor) -> torch.Tensor:
        normalized = (positions - self.mins) / (self.maxs - self.mins)
        grid_coords = 2.0 * normalized - 1.0
        grid_coords = grid_coords.flip(-1)
        return grid_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    def query(self, positions: torch.Tensor) -> torch.Tensor:
        assert self.sdf_grid is not None, "Call update() first"
        grid_coords = self._world_to_grid(positions)
        values = F.grid_sample(
            self.sdf_grid, grid_coords,
            mode='bilinear', padding_mode='border', align_corners=True
        )
        return values.squeeze()

    def collision_energy_and_gradient(
        self, positions: torch.Tensor, barrier_d: float = 0.15
    ) -> tuple:
        assert self.sdf_grid is not None, "Call update() first"
        pos = positions.detach().clone().requires_grad_(True)
        sdf_vals = self.query(pos)

        in_barrier = (sdf_vals > 0) & (sdf_vals <= barrier_d)
        safe_sdf = torch.clamp(sdf_vals, min=1e-6)
        energy_per_point = -torch.log(safe_sdf) * in_barrier.float()
        total_energy = energy_per_point.sum()

        total_energy.backward()
        gradient = pos.grad.clone()
        return total_energy.item(), gradient

    # ------------------------------------------------------------------
    # DIRECT QUERY
    # ------------------------------------------------------------------
    def collision_energy_and_gradient_direct(
        self, positions: torch.Tensor, barrier_d: float = 0.15
    ) -> tuple:
        assert self.obstacle_pts is not None and len(self.obstacle_pts) > 0
        pos = positions.detach().clone().requires_grad_(True)

        dists = torch.cdist(pos.unsqueeze(0), self.obstacle_pts.unsqueeze(0)).squeeze(0)
        min_dists, _ = dists.min(dim=1)

        in_barrier = (min_dists > 0) & (min_dists <= barrier_d)
        safe_dist = torch.clamp(min_dists, min=1e-6)
        energy_per_point = -torch.log(safe_dist) * in_barrier.float()
        total_energy = energy_per_point.sum()

        total_energy.backward()
        gradient = pos.grad.clone()
        return total_energy.item(), gradient

    # ------------------------------------------------------------------
    # SEMANTIC TARGET
    # ------------------------------------------------------------------
    @staticmethod
    def semantic_energy_and_gradient(
        ee_position: torch.Tensor,
        target_centroid: torch.Tensor,
        sigma_s: float = 0.05
    ) -> tuple:
        diff = ee_position - target_centroid
        energy = (diff * diff).sum() / (2 * sigma_s ** 2)
        gradient = diff / (sigma_s ** 2)
        return energy.item(), gradient

    # ------------------------------------------------------------------
    # RVIZ VISUALIZATION
    # ------------------------------------------------------------------
    @staticmethod
    def _sdf_colormap(vals, max_dist):
        rgb = np.zeros((len(vals), 3), dtype=np.uint8)

        on_surface = vals < 1e-4
        rgb[on_surface] = [255, 255, 255]

        barrier = ~on_surface
        if np.any(barrier):
            t = np.clip(vals[barrier] / max_dist, 0, 1)
            rgb[barrier, 0] = ((1.0 - t) * 255).astype(np.uint8)
            rgb[barrier, 1] = (t * 255).astype(np.uint8)
            rgb[barrier, 2] = np.zeros_like(t, dtype=np.uint8)

        return rgb

    def get_visualization_points(
        self,
        max_dist: float = 0.20,
        stride: int = 2
    ) -> tuple:
        if self.sdf_grid is None:
            return None, None, None

        sdf_3d = self.sdf_grid.squeeze()
        sdf_sub = sdf_3d[::stride, ::stride, ::stride]

        mask = sdf_sub <= max_dist
        if mask.sum() == 0:
            return None, None, None

        indices = mask.nonzero(as_tuple=False)
        mins_cpu = self.mins.cpu() if self.mins.is_cuda else self.mins
        world_pts = indices.float().cpu() * stride * self.voxel_size + mins_cpu
        world_pts = world_pts.numpy().astype(np.float32)

        vals = sdf_sub[mask].cpu().numpy() if sdf_sub.is_cuda else sdf_sub[mask].numpy()
        rgb = self._sdf_colormap(vals, max_dist)

        return world_pts, rgb, vals

    def get_slice_points(
        self,
        axis: str = "z",
        value: float = 0.1,
        max_dist: float = 0.20
    ) -> tuple:
        if self.sdf_grid is None:
            return None, None, None

        sdf_3d = self.sdf_grid.squeeze()
        D, H, W = sdf_3d.shape
        mins_cpu = self.mins.cpu().numpy() if self.mins.is_cuda else self.mins.numpy()

        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        voxel_idx = int(round((value - mins_cpu[axis_idx]) / self.voxel_size))
        voxel_idx = max(0, min(voxel_idx, sdf_3d.shape[axis_idx] - 1))

        if axis_idx == 0:
            sdf_slice = sdf_3d[voxel_idx, :, :]
            other_axes = (1, 2)
        elif axis_idx == 1:
            sdf_slice = sdf_3d[:, voxel_idx, :]
            other_axes = (0, 2)
        else:
            sdf_slice = sdf_3d[:, :, voxel_idx]
            other_axes = (0, 1)

        mask = sdf_slice <= max_dist
        if mask.sum() == 0:
            return None, None, None

        indices_2d = mask.nonzero(as_tuple=False)

        M = len(indices_2d)
        world_pts = np.zeros((M, 3), dtype=np.float32)
        world_pts[:, axis_idx] = value
        world_pts[:, other_axes[0]] = (
            indices_2d[:, 0].float().cpu().numpy() * self.voxel_size
            + mins_cpu[other_axes[0]]
        )
        world_pts[:, other_axes[1]] = (
            indices_2d[:, 1].float().cpu().numpy() * self.voxel_size
            + mins_cpu[other_axes[1]]
        )

        vals = sdf_slice[mask].cpu().numpy() if sdf_slice.is_cuda else sdf_slice[mask].numpy()
        rgb = self._sdf_colormap(vals, max_dist)

        return world_pts, rgb, vals

    @staticmethod
    def make_rviz_cloud_msg(points_xyz, points_rgb, frame_id="world", stamp=None):
        import rospy
        from sensor_msgs.msg import PointCloud2, PointField

        N = len(points_xyz)
        if N == 0:
            return None

        r = points_rgb[:, 0].astype(np.uint32)
        g = points_rgb[:, 1].astype(np.uint32)
        b = points_rgb[:, 2].astype(np.uint32)
        rgb_int = (r << 16) | (g << 8) | b
        rgb_packed = rgb_int.view(np.float32)

        cloud_data = np.zeros(N, dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)
        ])
        cloud_data['x'] = points_xyz[:, 0]
        cloud_data['y'] = points_xyz[:, 1]
        cloud_data['z'] = points_xyz[:, 2]
        cloud_data['rgb'] = rgb_packed

        msg = PointCloud2()
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp or rospy.Time.now()
        msg.height = 1
        msg.width = N
        msg.fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * N
        msg.data = cloud_data.tobytes()
        msg.is_dense = True

        return msg