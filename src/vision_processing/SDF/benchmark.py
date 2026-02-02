import numpy as np
import time
import os

# Check if grid exists
grid_file = "sdf_grid.npz"

if not os.path.exists(grid_file):
    print(f"❌ Grid not found: {grid_file}")
    print("   Run precompute_sdf.py first.")
    exit(1)

# Load grid
print("Loading SDF grid...")
data = np.load(grid_file)
sdf_grid = data['sdf_grid'].astype(np.float64)
bounds_min = data['bounds_min']
bounds_max = data['bounds_max']
voxel_size = float(data['voxel_size'])
n_voxels = data['n_voxels']

print(f"Grid shape: {sdf_grid.shape}")
print(f"Voxel size: {voxel_size * 1000:.2f} mm")


def trilinear_interpolate(grid_coords):
    """Fast trilinear interpolation."""
    gc = np.clip(grid_coords, 0, np.array(n_voxels) - 1.001)
    i0 = np.floor(gc).astype(int)
    i1 = np.minimum(i0 + 1, np.array(n_voxels) - 1)
    t = gc - i0
    
    d000 = sdf_grid[i0[:, 0], i0[:, 1], i0[:, 2]]
    d001 = sdf_grid[i0[:, 0], i0[:, 1], i1[:, 2]]
    d010 = sdf_grid[i0[:, 0], i1[:, 1], i0[:, 2]]
    d011 = sdf_grid[i0[:, 0], i1[:, 1], i1[:, 2]]
    d100 = sdf_grid[i1[:, 0], i0[:, 1], i0[:, 2]]
    d101 = sdf_grid[i1[:, 0], i0[:, 1], i1[:, 2]]
    d110 = sdf_grid[i1[:, 0], i1[:, 1], i0[:, 2]]
    d111 = sdf_grid[i1[:, 0], i1[:, 1], i1[:, 2]]
    
    tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
    
    c00 = d000 * (1 - tx) + d100 * tx
    c01 = d001 * (1 - tx) + d101 * tx
    c10 = d010 * (1 - tx) + d110 * tx
    c11 = d011 * (1 - tx) + d111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    
    return c0 * (1 - tz) + c1 * tz


def query_with_gradient(points):
    """Full query: distance + gradient."""
    grid_coords = (points - bounds_min) / voxel_size - 0.5
    
    # Distance
    distances = trilinear_interpolate(grid_coords)
    
    # Gradient via finite differences
    eps = 0.5
    d_xp = trilinear_interpolate(grid_coords + np.array([eps, 0, 0]))
    d_xm = trilinear_interpolate(grid_coords + np.array([-eps, 0, 0]))
    d_yp = trilinear_interpolate(grid_coords + np.array([0, eps, 0]))
    d_ym = trilinear_interpolate(grid_coords + np.array([0, -eps, 0]))
    d_zp = trilinear_interpolate(grid_coords + np.array([0, 0, eps]))
    d_zm = trilinear_interpolate(grid_coords + np.array([0, 0, -eps]))
    
    grad = np.stack([
        (d_xp - d_xm) / (2 * eps),
        (d_yp - d_ym) / (2 * eps),
        (d_zp - d_zm) / (2 * eps)
    ], axis=1)
    
    norms = np.linalg.norm(grad, axis=1, keepdims=True)
    gradients = grad / np.maximum(norms, 1e-10)
    
    return distances, gradients


def generate_random_points(n):
    margin = 0.02
    return np.random.uniform(bounds_min - margin, bounds_max + margin, size=(n, 3))


# Warm-up
_ = query_with_gradient(generate_random_points(100))

# Benchmark
batch_sizes = [512, 1024, 4096, 8192, 16384]

print()
print("=" * 60)
print("SDF Query Benchmark (Grid-based with Trilinear Interpolation)")
print("=" * 60)

for n in batch_sizes:
    points = generate_random_points(n)
    
    n_runs = 50
    times = []
    
    for _ in range(n_runs):
        t0 = time.perf_counter()
        distances, gradients = query_with_gradient(points)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    per_point_us = (np.mean(times) / n) * 1e6
    
    print(f"Batch size: {n:>6} | {avg_ms:>7.3f} ± {std_ms:.3f} ms | {per_point_us:.3f} µs/point")

print()
print("=" * 60)
print("Your use case: 16 poses × 512 points = 8192 queries")
print("Budget at 10 Hz: 100 ms per cycle")
print("=" * 60)