"""
GPU Real-Time SDF from Point Cloud — Reproducing OmniGuide's approach.

Pipeline:
  1. Point cloud → binary occupancy grid  (GPU, torch)
  2. Occupancy grid → SDF via EDT          (CPU, scipy — ~5ms for 128³)
  3. SDF tensor on GPU                     (torch.nn.functional.grid_sample)
  4. Query any (x,y,z) → SDF value         (trilinear interpolation, differentiable)
  5. Gradients via autograd                 (free, no finite differences needed)

The energy from the paper (Eq. 15):
    L_C(x) = -log SDF_O(x)

The gradient (Eq. 16):
    -∇_x L_C(x) = (1 / SDF(x)) * (x - p*) / ||x - p*||

But since grid_sample is differentiable, we can just do:
    loss = -torch.log(sdf_value)
    loss.backward()   # gives us ∇_x L_C automatically
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
import time


class VoxelSDF:
    """
    Discrete SDF on a voxel grid, queryable on GPU via trilinear interpolation.
    
    Usage:
        sdf = VoxelSDF(voxel_size=0.01, bounds=((-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0)))
        sdf.update(point_cloud)           # (N, 3) numpy or torch
        values = sdf.query(positions)      # (M, 3) torch tensor → (M,) SDF values
        energy, grad = sdf.collision_energy_and_gradient(positions, barrier_d=0.15)
    """

    def __init__(self, voxel_size: float, bounds: tuple, device="cuda"):
        """
        Args:
            voxel_size: size of each voxel in meters (e.g. 0.01 = 1cm)
            bounds: ((xmin, xmax), (ymin, ymax), (zmin, zmax)) in meters
            device: 'cuda' or 'cpu'
        """
        self.voxel_size = voxel_size
        self.device = device

        # Store grid dimensions
        self.mins = np.array([b[0] for b in bounds], dtype=np.float32)
        self.maxs = np.array([b[1] for b in bounds], dtype=np.float32)
        self.grid_shape = np.ceil((self.maxs - self.mins) / voxel_size).astype(int)

        # Precompute for coordinate conversion
        self.mins_t = torch.from_numpy(self.mins).to(device)
        self.maxs_t = torch.from_numpy(self.maxs).to(device)

        # SDF grid (will be set by update())
        self.sdf_grid = None  # shape: (1, 1, D, H, W) for grid_sample

    def update(self, points: np.ndarray):
        """
        Rebuild the SDF from a new point cloud. This is the main "real-time" call.
        
        Args:
            points: (N, 3) point cloud in world coordinates
        """
        t0 = time.perf_counter()

        # --- Step 1: Voxelize (numpy, fast) ---
        indices = ((points - self.mins) / self.voxel_size).astype(int)

        # Keep only points inside the grid
        D, H, W = self.grid_shape
        valid = (
            (indices[:, 0] >= 0) & (indices[:, 0] < D) &
            (indices[:, 1] >= 0) & (indices[:, 1] < H) &
            (indices[:, 2] >= 0) & (indices[:, 2] < W)
        )
        indices = indices[valid]

        # Binary occupancy
        occupancy = np.zeros((D, H, W), dtype=bool)
        occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = True

        t1 = time.perf_counter()

        # --- Step 2: EDT on free space ---
        # distance_transform_edt gives distance from each 0-voxel to nearest 1-voxel
        # We want: for free voxels, distance to nearest obstacle
        # So we compute EDT on ~occupancy (free space = True → distance to obstacle = False)
        # Wait — EDT computes distance from 0-valued voxels. 
        # If we pass `occupancy` directly: 0 = free, 1 = occupied
        # EDT on (1 - occupancy): 0 = occupied, 1 = free → gives free voxels their distance to occupied
        sdf_np = distance_transform_edt(~occupancy).astype(np.float32) * self.voxel_size

        t2 = time.perf_counter()

        # --- Step 3: Transfer to GPU as 5D tensor for grid_sample ---
        # grid_sample expects (N, C, D, H, W) input
        self.sdf_grid = torch.from_numpy(sdf_np).to(self.device).unsqueeze(0).unsqueeze(0)

        t3 = time.perf_counter()

        print(f"SDF update: voxelize {(t1-t0)*1000:.1f}ms | "
              f"EDT {(t2-t1)*1000:.1f}ms | "
              f"GPU transfer {(t3-t2)*1000:.1f}ms | "
              f"grid {D}x{H}x{W} = {D*H*W/1e6:.1f}M voxels")

    def _world_to_grid(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Convert world coordinates to grid_sample coordinates in [-1, 1].
        
        grid_sample expects coordinates in [-1, 1] where:
            -1 corresponds to the first voxel, +1 to the last voxel.
        
        Args:
            positions: (M, 3) world coordinates
        Returns:
            (1, 1, 1, M, 3) grid coordinates for grid_sample (note: order is x,y,z → W,H,D)
        """
        # Normalize to [0, 1]
        normalized = (positions - self.mins_t) / (self.maxs_t - self.mins_t)
        # Map to [-1, 1]
        grid_coords = 2.0 * normalized - 1.0

        # IMPORTANT: grid_sample expects (x, y, z) = (W, H, D) ordering
        # Our SDF grid is stored as (D, H, W), so we need to flip the coordinates
        grid_coords = grid_coords.flip(-1)

        # Reshape to (1, 1, 1, M, 3) — batch=1, out_D=1, out_H=1, out_W=M
        return grid_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    def query(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Query SDF values at arbitrary positions via trilinear interpolation.
        
        Args:
            positions: (M, 3) world coordinates, requires_grad=True for gradients
        Returns:
            (M,) SDF values in meters
        """
        grid_coords = self._world_to_grid(positions)

        # Trilinear interpolation — this is differentiable!
        values = F.grid_sample(
            self.sdf_grid,           # (1, 1, D, H, W)
            grid_coords,             # (1, 1, 1, M, 3)
            mode='bilinear',         # bilinear in 3D = trilinear
            padding_mode='border',   # clamp to edge values outside grid
            align_corners=True
        )

        return values.squeeze()  # (M,)

    def collision_energy_and_gradient(
        self,
        positions: torch.Tensor,
        barrier_d: float = 0.15
    ) -> tuple:
        """
        Compute the collision energy and its gradient from OmniGuide Eq. 15.
        
        L_C(x) = -log(SDF(x))    for 0 < SDF(x) <= d
        
        Args:
            positions: (M, 3) query points (e.g. robot probe points)
            barrier_d: max distance for the barrier (d in the paper)
        Returns:
            energy: scalar, sum of collision energies
            gradient: (M, 3) gradient ∇_x L_C for each point
        """
        # Enable gradient tracking
        pos = positions.detach().clone().requires_grad_(True)

        sdf_vals = self.query(pos)

        # Mask: only apply energy inside the barrier zone (0 < sdf <= d)
        in_barrier = (sdf_vals > 0) & (sdf_vals <= barrier_d)

        # Clamp to avoid log(0)
        safe_sdf = torch.clamp(sdf_vals, min=1e-6)

        # Energy: -log(SDF(x)) — only for points in barrier zone
        energy_per_point = -torch.log(safe_sdf) * in_barrier.float()
        total_energy = energy_per_point.sum()

        # Gradient via autograd (this is the beauty of using grid_sample!)
        total_energy.backward()
        gradient = pos.grad.clone()  # (M, 3)

        return total_energy.item(), gradient


# =============================================================================
# Demo: generate a synthetic scene and test the pipeline
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Create a synthetic obstacle point cloud ---
    # A box obstacle centered at (0.3, 0.0, 0.4)
    box_center = np.array([0.3, 0.0, 0.4])
    box_size = np.array([0.1, 0.15, 0.1])
    n_surface_pts = 5000

    # Sample points on box surface
    pts = []
    for axis in range(3):
        for sign in [-1, 1]:
            n = n_surface_pts // 6
            p = np.random.uniform(-0.5, 0.5, (n, 3)) * box_size
            p[:, axis] = sign * box_size[axis] / 2
            pts.append(p + box_center)

    # Add a table surface at z=0.05
    table_pts = np.random.uniform(-0.4, 0.4, (3000, 3))
    table_pts[:, 2] = 0.05
    pts.append(table_pts)

    obstacle_cloud = np.vstack(pts).astype(np.float32)
    print(f"Obstacle point cloud: {obstacle_cloud.shape[0]} points")

    # --- Build SDF ---
    sdf = VoxelSDF(
        voxel_size=0.01,  # 1cm resolution
        bounds=((-0.5, 0.8), (-0.5, 0.5), (0.0, 0.8)),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    sdf.update(obstacle_cloud)

    # --- Query some robot probe points ---
    # Simulate an end-effector trajectory approaching the box
    t_steps = torch.linspace(0, 1, 20)
    start = torch.tensor([0.0, 0.0, 0.5])
    end = torch.tensor([0.3, 0.0, 0.4])  # heading toward box center
    trajectory = start + t_steps.unsqueeze(1) * (end - start)
    trajectory = trajectory.to(sdf.device)

    # Query SDF along trajectory
    sdf_values = sdf.query(trajectory)
    print("\n--- SDF values along trajectory ---")
    for i in range(len(trajectory)):
        print(f"  step {i:2d}: pos=({trajectory[i,0]:.2f}, {trajectory[i,1]:.2f}, {trajectory[i,2]:.2f})"
              f"  SDF={sdf_values[i]:.4f}m")

    # --- Compute collision energy + gradient ---
    energy, grad = sdf.collision_energy_and_gradient(trajectory, barrier_d=0.15)
    print(f"\nTotal collision energy: {energy:.4f}")
    print(f"Gradient shape: {grad.shape}")
    print(f"Max gradient norm: {grad.norm(dim=1).max():.4f}")

    # --- Benchmark ---
    print("\n--- Timing benchmark (100 updates) ---")
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        sdf.update(obstacle_cloud)
        times.append(time.perf_counter() - t0)
    times = np.array(times[10:])  # skip warmup
    print(f"SDF update: {times.mean()*1000:.1f} ± {times.std()*1000:.1f} ms")

    print("\n--- Timing benchmark (1000 queries of 50 points) ---")
    query_pts = torch.rand(50, 3, device=sdf.device) * 0.5
    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        sdf.query(query_pts)
        times.append(time.perf_counter() - t0)
    times = np.array(times[10:])
    print(f"SDF query (50 pts): {times.mean()*1000:.2f} ± {times.std()*1000:.2f} ms")
