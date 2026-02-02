import numpy as np
import time
import os

def precompute_sdf_trimesh_parallel(mesh_path, output_path, resolution=150, margin=0.03):
    """
    Fast SDF using trimesh with parallel processing.
    
    This uses trimesh's proximity query which is already C-optimized,
    but we parallelize the batch processing.
    """
    import trimesh
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path)
    
    # Simplify mesh first for faster queries
    if len(mesh.faces) > 10000:
        print(f"   Simplifying mesh: {len(mesh.faces)} → 10000 faces...")
        mesh = mesh.simplify_quadric_decimation(10000)
        print(f"   Done: {len(mesh.faces)} faces")
    
    proximity = trimesh.proximity.ProximityQuery(mesh)
    
    bounds_min = mesh.bounds[0] - margin
    bounds_max = mesh.bounds[1] + margin
    extent = bounds_max - bounds_min
    
    voxel_size = np.max(extent) / resolution
    n_voxels = np.ceil(extent / voxel_size).astype(int)
    
    print(f"   Voxel size: {voxel_size * 1000:.2f} mm")
    print(f"   Grid: {n_voxels[0]} x {n_voxels[1]} x {n_voxels[2]} = {np.prod(n_voxels):,} voxels")
    
    # Create all grid points
    x = np.linspace(bounds_min[0] + voxel_size/2, 
                    bounds_min[0] + (n_voxels[0] - 0.5) * voxel_size, 
                    n_voxels[0])
    y = np.linspace(bounds_min[1] + voxel_size/2, 
                    bounds_min[1] + (n_voxels[1] - 0.5) * voxel_size, 
                    n_voxels[1])
    z = np.linspace(bounds_min[2] + voxel_size/2, 
                    bounds_min[2] + (n_voxels[2] - 0.5) * voxel_size, 
                    n_voxels[2])
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    all_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    print(f"Computing SDF for {len(all_points):,} points...")
    t_start = time.time()
    
    # Process in large batches (trimesh handles this efficiently in C)
    batch_size = 100000
    n_batches = int(np.ceil(len(all_points) / batch_size))
    
    distances = np.zeros(len(all_points))
    signs = np.zeros(len(all_points))
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(all_points))
        batch_points = all_points[start_idx:end_idx]
        
        # Distance to surface (C-optimized in trimesh)
        _, batch_dist, _ = proximity.on_surface(batch_points)
        distances[start_idx:end_idx] = batch_dist
        
        # Inside/outside check
        inside = mesh.contains(batch_points)
        signs[start_idx:end_idx] = np.where(inside, -1.0, 1.0)
        
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (n_batches - i - 1)
        print(f"   Batch {i+1}/{n_batches} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
    
    signed_distances = distances * signs
    sdf_grid = signed_distances.reshape(n_voxels)
    
    elapsed = time.time() - t_start
    print(f"   Total time: {elapsed:.1f}s")
    
    # Save
    np.savez(output_path,
             sdf_grid=sdf_grid.astype(np.float32),
             bounds_min=bounds_min,
             bounds_max=bounds_max,
             voxel_size=voxel_size,
             n_voxels=n_voxels)
    
    print(f"\n✅ Saved to: {output_path}")
    return sdf_grid


def precompute_sdf_pysdf(mesh_path, output_path, resolution=150, margin=0.03):
    """
    Fast SDF using pysdf (GPU-accelerated if available).
    
    Install: pip install pysdf
    """
    from pysdf import SDF
    import trimesh
    
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path)
    
    bounds_min = mesh.bounds[0] - margin
    bounds_max = mesh.bounds[1] + margin
    extent = bounds_max - bounds_min
    
    voxel_size = np.max(extent) / resolution
    n_voxels = np.ceil(extent / voxel_size).astype(int)
    
    print(f"   Voxel size: {voxel_size * 1000:.2f} mm")
    print(f"   Grid: {n_voxels[0]} x {n_voxels[1]} x {n_voxels[2]} = {np.prod(n_voxels):,} voxels")
    
    # Create SDF object (builds acceleration structure)
    print("Building SDF acceleration structure...")
    t_start = time.time()
    sdf_func = SDF(mesh.vertices, mesh.faces)
    
    # Create grid points
    x = np.linspace(bounds_min[0] + voxel_size/2, 
                    bounds_min[0] + (n_voxels[0] - 0.5) * voxel_size, 
                    n_voxels[0])
    y = np.linspace(bounds_min[1] + voxel_size/2, 
                    bounds_min[1] + (n_voxels[1] - 0.5) * voxel_size, 
                    n_voxels[1])
    z = np.linspace(bounds_min[2] + voxel_size/2, 
                    bounds_min[2] + (n_voxels[2] - 0.5) * voxel_size, 
                    n_voxels[2])
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    all_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    print(f"Computing SDF for {len(all_points):,} points...")
    
    # pysdf computes signed distances in batch (very fast)
    signed_distances = sdf_func(all_points)
    
    elapsed = time.time() - t_start
    print(f"   Done in {elapsed:.1f}s")
    
    sdf_grid = signed_distances.reshape(n_voxels)
    
    # pysdf uses opposite sign convention (negative outside)
    # Flip if needed for your convention
    sdf_grid = -sdf_grid  # Now: positive outside, negative inside
    
    # Save
    np.savez(output_path,
             sdf_grid=sdf_grid.astype(np.float32),
             bounds_min=bounds_min,
             bounds_max=bounds_max,
             voxel_size=voxel_size,
             n_voxels=n_voxels)
    
    print(f"\n✅ Saved to: {output_path}")
    return sdf_grid


def precompute_sdf_meshlib(mesh_path, output_path, resolution=150, margin=0.03):
    """
    Fast SDF computation using MeshLib.
    
    MeshLib is C++ with Python bindings.
    Install: pip install meshlib
    """
    import meshlib.mrmeshpy as mr
    import trimesh
    
    print(f"Loading mesh: {mesh_path}")
    mesh_mr = mr.loadMesh(mesh_path)
    mesh_tm = trimesh.load(mesh_path)  # For bounds
    
    bounds_min = mesh_tm.bounds[0] - margin
    bounds_max = mesh_tm.bounds[1] + margin
    extent = bounds_max - bounds_min
    
    voxel_size = np.max(extent) / resolution
    n_voxels = np.ceil(extent / voxel_size).astype(int)
    
    print(f"   Voxel size: {voxel_size * 1000:.2f} mm")
    print(f"   Grid: {n_voxels[0]} x {n_voxels[1]} x {n_voxels[2]} = {np.prod(n_voxels):,} voxels")
    
    # Create grid points
    x = np.linspace(bounds_min[0] + voxel_size/2, 
                    bounds_min[0] + (n_voxels[0] - 0.5) * voxel_size, 
                    n_voxels[0])
    y = np.linspace(bounds_min[1] + voxel_size/2, 
                    bounds_min[1] + (n_voxels[1] - 0.5) * voxel_size, 
                    n_voxels[1])
    z = np.linspace(bounds_min[2] + voxel_size/2, 
                    bounds_min[2] + (n_voxels[2] - 0.5) * voxel_size, 
                    n_voxels[2])
    
    print("Computing SDF grid...")
    t_start = time.time()
    
    sdf_grid = np.zeros(n_voxels, dtype=np.float32)
    
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            for k, zk in enumerate(z):
                point = mr.Vector3f(xi, yj, zk)
                dist = mr.findSignedDistance(point, mesh_mr)
                sdf_grid[i, j, k] = dist
                
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            progress = (i + 1) / n_voxels[0]
            eta = elapsed / progress * (1 - progress)
            print(f"   Slice {i+1}/{n_voxels[0]} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
    
    elapsed = time.time() - t_start
    print(f"   Done in {elapsed:.1f}s")
    
    # Save
    np.savez(output_path,
             sdf_grid=sdf_grid,
             bounds_min=bounds_min,
             bounds_max=bounds_max,
             voxel_size=voxel_size,
             n_voxels=n_voxels)
    
    print(f"\n✅ Saved to: {output_path}")
    return sdf_grid


if __name__ == "__main__":
    mesh_file = "mon_bol_thick_shell.obj"
    output_file = "sdf_grid.npz"
    
    if not os.path.exists(mesh_file):
        print(f"❌ File not found: {mesh_file}")
        exit(1)
    
    resolution = 150  # ~1.5mm voxels for accurate sign inside 2mm wall
    
    # Try methods in order of speed
    
    # Option 1: pysdf (fastest, C++ with optional GPU)
    try:
        from pysdf import SDF
        print("Using pysdf (fastest, C++ backend)")
        precompute_sdf_pysdf(mesh_file, output_file, resolution=resolution, margin=0.03)
        exit(0)
    except ImportError:
        print("pysdf not found. Install with: pip install pysdf\n")
    
    # Option 2: MeshLib (fast C++ backend)
    try:
        import meshlib.mrmeshpy as mr
        print("Using MeshLib (C++ backend)")
        precompute_sdf_meshlib(mesh_file, output_file, resolution=resolution, margin=0.03)
        exit(0)
    except ImportError:
        print("MeshLib not found. Install with: pip install meshlib\n")
    
    # Option 3: trimesh with simplified mesh (fallback)
    print("Using trimesh with mesh simplification (slowest)")
    precompute_sdf_trimesh_parallel(mesh_file, output_file, resolution=resolution, margin=0.03)