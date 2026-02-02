# import numpy as np
# import pyvista as pv
# import trimesh
# from scipy.spatial import cKDTree
# import matplotlib.pyplot as plt


# class MeshSDF:
#     """
#     Computes signed distance and gradient from a watertight mesh.
    
#     How it works:
#     1. Build a KD-tree from mesh vertices for fast nearest-point lookup
#     2. For any query point, find the closest surface point
#     3. Distance = ||query - closest||
#     4. Gradient = (query - closest) / distance  (unit vector pointing away from surface)
#     5. Sign = negative if inside the mesh (using ray casting)
#     """
    
#     def __init__(self, mesh_path):
#         print(f"Loading mesh: {mesh_path}")
        
#         # Load with trimesh (good for inside/outside queries)
#         self.mesh = trimesh.load(mesh_path)
        
#         # Build KD-tree from mesh vertices
#         # For more accuracy, we could sample points ON the faces,
#         # but vertices are usually dense enough for smooth meshes
#         self.vertices = np.array(self.mesh.vertices)
#         self.proximity = trimesh.proximity.ProximityQuery(self.mesh)
        
#         print(f"   {len(self.vertices)} vertices indexed")
#         print(f"   Bounds: {self.mesh.bounds}")
    
#     def query(self, points):
#         """
#         Compute distance and gradient for multiple query points.
        
#         Args:
#             points: (N, 3) array of query positions
        
#         Returns:
#             distances: (N,) signed distances (negative = inside)
#             gradients: (N, 3) unit vectors pointing away from surface
#         """
#         points = np.atleast_2d(points)
#         n_points = len(points)
        
#         # Find nearest vertex for each query point
#         closest_points, distances, face_ids = self.proximity.on_surface(points)
        
#         # Gradient = direction from surface to query point
#         diff = points - closest_points
        
#         # Normalize to get unit vectors
#         # Handle edge case where point is exactly on surface
#         norms = np.linalg.norm(diff, axis=1, keepdims=True)
#         norms = np.maximum(norms, 1e-10)  # Avoid division by zero
#         gradients = diff / norms
        
#         # Determine sign: negative if inside the mesh
#         # trimesh.contains is robust for watertight meshes
#         inside = self.mesh.contains(points)
#         signs = np.where(inside, -1.0, 1.0)
        
#         signed_distances = distances * signs
        
#         # Inside points: gradient should point outward (to escape)
#         # We flip the gradient for inside points
#         gradients = gradients * signs[:, np.newaxis]
        
#         return signed_distances, gradients
    
#     def query_single(self, point):
#         """Convenience method for single point query."""
#         d, g = self.query(point.reshape(1, 3))
#         return d[0], g[0]


# def visualize_sdf_slice(sdf, slice_axis='z', slice_value=None, resolution=100, extent=0.05):
#     """
#     Visualize a 2D slice of the SDF field.
    
#     Args:
#         sdf: MeshSDF object
#         slice_axis: 'x', 'y', or 'z' - the axis perpendicular to the slice
#         slice_value: position along that axis (None = mesh center)
#         resolution: grid resolution
#         extent: how far beyond mesh bounds to show (meters)
#     """
#     # Get mesh bounds
#     bounds = sdf.mesh.bounds
#     center = sdf.mesh.centroid
    
#     if slice_value is None:
#         slice_value = center[{'x': 0, 'y': 1, 'z': 2}[slice_axis]]
    
#     # Create 2D grid for the slice
#     if slice_axis == 'z':
#         x = np.linspace(bounds[0, 0] - extent, bounds[1, 0] + extent, resolution)
#         y = np.linspace(bounds[0, 1] - extent, bounds[1, 1] + extent, resolution)
#         X, Y = np.meshgrid(x, y)
#         Z = np.full_like(X, slice_value)
#         points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
#         xlabel, ylabel = 'X (m)', 'Y (m)'
        
#     elif slice_axis == 'y':
#         x = np.linspace(bounds[0, 0] - extent, bounds[1, 0] + extent, resolution)
#         z = np.linspace(bounds[0, 2] - extent, bounds[1, 2] + extent, resolution)
#         X, Z = np.meshgrid(x, z)
#         Y = np.full_like(X, slice_value)
#         points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
#         xlabel, ylabel = 'X (m)', 'Z (m)'
        
#     else:  # x
#         y = np.linspace(bounds[0, 1] - extent, bounds[1, 1] + extent, resolution)
#         z = np.linspace(bounds[0, 2] - extent, bounds[1, 2] + extent, resolution)
#         Y, Z = np.meshgrid(y, z)
#         X = np.full_like(Y, slice_value)
#         points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
#         xlabel, ylabel = 'Y (m)', 'Z (m)'
    
#     # Query SDF
#     print(f"Querying {len(points)} points...")
#     distances, gradients = sdf.query(points)
    
#     # Reshape for plotting
#     D = distances.reshape(resolution, resolution)
    
#     # Get the 2D gradient components for arrows
#     if slice_axis == 'z':
#         Gx = gradients[:, 0].reshape(resolution, resolution)
#         Gy = gradients[:, 1].reshape(resolution, resolution)
#         grid_x, grid_y = X, Y
#     elif slice_axis == 'y':
#         Gx = gradients[:, 0].reshape(resolution, resolution)
#         Gy = gradients[:, 2].reshape(resolution, resolution)
#         grid_x, grid_y = X, Z
#     else:
#         Gx = gradients[:, 1].reshape(resolution, resolution)
#         Gy = gradients[:, 2].reshape(resolution, resolution)
#         grid_x, grid_y = Y, Z
    
#     # Plot
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
#     # Left: Distance field with contours
#     ax1 = axes[0]
    
#     # Use symmetric colormap centered at 0
#     max_abs = np.max(np.abs(D))
#     im = ax1.contourf(grid_x, grid_y, D, levels=50, cmap='RdBu', vmin=-max_abs, vmax=max_abs)
#     ax1.contour(grid_x, grid_y, D, levels=[0], colors='black', linewidths=2)  # Zero level = surface
    
#     plt.colorbar(im, ax=ax1, label='Signed Distance (m)')
#     ax1.set_xlabel(xlabel)
#     ax1.set_ylabel(ylabel)
#     ax1.set_title(f'SDF Slice ({slice_axis}={slice_value:.4f}m)\nBlue=Outside, Red=Inside, Black=Surface')
#     ax1.set_aspect('equal')
    
#     # Right: Gradient field (arrows)
#     ax2 = axes[1]
    
#     # Subsample for cleaner arrows
#     skip = max(1, resolution // 20)
#     ax2.contourf(grid_x, grid_y, D, levels=50, cmap='RdBu', vmin=-max_abs, vmax=max_abs, alpha=0.5)
#     ax2.contour(grid_x, grid_y, D, levels=[0], colors='black', linewidths=2)
    
#     # Draw gradient arrows
#     ax2.quiver(grid_x[::skip, ::skip], grid_y[::skip, ::skip],
#                Gx[::skip, ::skip], Gy[::skip, ::skip],
#                scale=30, width=0.003, color='darkblue', alpha=0.8)
    
#     ax2.set_xlabel(xlabel)
#     ax2.set_ylabel(ylabel)
#     ax2.set_title('Gradient Field\nArrows point away from surface')
#     ax2.set_aspect('equal')
    
#     plt.tight_layout()
#     plt.savefig('sdf_visualization.png', dpi=150)
#     print("Saved: sdf_visualization.png")
#     plt.show()


# def visualize_3d_with_samples(sdf, n_samples=500):
#     """
#     3D visualization: mesh + sample points colored by distance + gradient arrows.
#     """
#     bounds = sdf.mesh.bounds
#     margin = 0.03  # 3cm margin
    
#     # Random sample points around the mesh
#     points = np.random.uniform(
#         bounds[0] - margin,
#         bounds[1] + margin,
#         size=(n_samples, 3)
#     )
    
#     distances, gradients = sdf.query(points)
    
#     # PyVista visualization
#     p = pv.Plotter()
    
#     # Add mesh
#     pv_mesh = pv.read("mon_bol_thick_shell.obj")
#     p.add_mesh(pv_mesh, color='lightgray', opacity=0.5)
    
#     # Add sample points colored by distance
#     point_cloud = pv.PolyData(points)
#     point_cloud['distance'] = distances
#     p.add_mesh(point_cloud, scalars='distance', cmap='RdBu', 
#                point_size=8, render_points_as_spheres=True,
#                clim=[-0.02, 0.02])
    
#     # Add gradient arrows (subsample for clarity)
#     arrow_idx = np.random.choice(len(points), size=min(100, len(points)), replace=False)
#     arrow_points = points[arrow_idx]
#     arrow_dirs = gradients[arrow_idx]
    
#     arrows = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=0.01)
#     for pt, dir in zip(arrow_points, arrow_dirs):
#         arrow = pv.Arrow(start=pt, direction=dir, scale=0.005)
#         p.add_mesh(arrow, color='green')
    
#     p.add_title("3D SDF: Blue=Outside, Red=Inside, Arrows=Gradient")
#     p.show()


# # =============================================
# # MAIN
# # =============================================
# if __name__ == "__main__":
#     import os
    
#     mesh_file = "mon_bol_thick_shell.obj"
    
#     if not os.path.exists(mesh_file):
#         print(f"❌ File not found: {mesh_file}")
#         print("   Run create_thick_shell.py first.")
#         exit(1)
    
#     # Create SDF object
#     sdf = MeshSDF(mesh_file)
    
#     # Test single query
#     test_point = sdf.mesh.centroid + np.array([0.02, 0, 0])  # 2cm from center
#     d, g = sdf.query_single(test_point)
#     print(f"\nTest query at {test_point}:")
#     print(f"   Distance: {d*1000:.2f} mm")
#     print(f"   Gradient: {g}")
    
#     # Visualize 2D slice through the middle (horizontal cut)
#     print("\n--- 2D Slice Visualization ---")
#     visualize_sdf_slice(sdf, slice_axis='y', resolution=80, extent=0.03)
    
#     # Visualize 3D (optional, can be slow)
#     print("\n--- 3D Visualization ---")
#     visualize_3d_with_samples(sdf, n_samples=60)











import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


class MeshSDF:
    """
    Fast SDF queries using a precomputed voxel grid.
    
    Instead of checking distance to 87k triangles, we:
    1. Load a precomputed 3D grid of SDF values
    2. For any query point, find its cell in the grid
    3. Interpolate the 8 corner values (trilinear interpolation)
    
    This gives O(1) query time regardless of mesh complexity.
    """
    
    def __init__(self, grid_path):
        """
        Load precomputed SDF grid.
        
        Args:
            grid_path: Path to .npz file from precompute_sdf.py
        """
        print(f"Loading SDF grid: {grid_path}")
        
        data = np.load(grid_path)
        self.sdf_grid = data['sdf_grid'].astype(np.float64)
        self.bounds_min = data['bounds_min']
        self.bounds_max = data['bounds_max']
        self.voxel_size = float(data['voxel_size'])
        self.n_voxels = data['n_voxels']
        
        print(f"   Grid shape: {self.sdf_grid.shape}")
        print(f"   Voxel size: {self.voxel_size * 1000:.2f} mm")
        print(f"   Bounds: [{self.bounds_min}] to [{self.bounds_max}]")
    
    def query(self, points):
        """
        Compute distance and gradient for multiple query points.
        
        Uses trilinear interpolation for smooth distance values,
        and finite differences for smooth gradients.
        
        Args:
            points: (N, 3) array of query positions
        
        Returns:
            distances: (N,) signed distances
            gradients: (N, 3) unit vectors pointing away from surface
        """
        points = np.atleast_2d(points).astype(np.float64)
        
        # Convert world coordinates to grid coordinates (continuous)
        # grid_coords[i] = 0 means at the center of voxel 0
        # grid_coords[i] = 0.5 means halfway between voxel 0 and 1
        grid_coords = (points - self.bounds_min) / self.voxel_size - 0.5
        
        # Trilinear interpolation for distance
        distances = self._trilinear_interpolate(grid_coords)
        
        # Gradient via finite differences
        gradients = self._compute_gradient(grid_coords)
        
        return distances, gradients
    
    def _trilinear_interpolate(self, grid_coords):
        """
        Trilinear interpolation of SDF values.
        
        For a point P inside a voxel cell, we interpolate the 8 corner values.
        
             d001-----d101
             /|       /|
           d011-----d111
            | d000---|d100
            |/       |/
           d010-----d110
        
        The interpolation formula:
            d = weighted average of 8 corners
            weights = (1-tx)*(1-ty)*(1-tz), tx*(1-ty)*(1-tz), etc.
        
        where (tx, ty, tz) is the fractional position within the cell.
        """
        # Clamp to valid grid range
        gc = np.clip(grid_coords, 0, np.array(self.n_voxels) - 1.001)
        
        # Integer part = which cell
        i0 = np.floor(gc).astype(int)
        i1 = np.minimum(i0 + 1, np.array(self.n_voxels) - 1)
        
        # Fractional part = position within cell [0, 1]
        t = gc - i0
        
        # Get the 8 corner values for each query point
        # d[a][b][c] where a,b,c ∈ {0,1} indicates low/high in x,y,z
        d000 = self.sdf_grid[i0[:, 0], i0[:, 1], i0[:, 2]]
        d001 = self.sdf_grid[i0[:, 0], i0[:, 1], i1[:, 2]]
        d010 = self.sdf_grid[i0[:, 0], i1[:, 1], i0[:, 2]]
        d011 = self.sdf_grid[i0[:, 0], i1[:, 1], i1[:, 2]]
        d100 = self.sdf_grid[i1[:, 0], i0[:, 1], i0[:, 2]]
        d101 = self.sdf_grid[i1[:, 0], i0[:, 1], i1[:, 2]]
        d110 = self.sdf_grid[i1[:, 0], i1[:, 1], i0[:, 2]]
        d111 = self.sdf_grid[i1[:, 0], i1[:, 1], i1[:, 2]]
        
        # Interpolation weights
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
        
        # Trilinear interpolation formula
        # First interpolate along x (4 pairs)
        c00 = d000 * (1 - tx) + d100 * tx
        c01 = d001 * (1 - tx) + d101 * tx
        c10 = d010 * (1 - tx) + d110 * tx
        c11 = d011 * (1 - tx) + d111 * tx
        
        # Then along y (2 pairs)
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        
        # Finally along z
        distances = c0 * (1 - tz) + c1 * tz
        
        return distances
    
    def _compute_gradient(self, grid_coords):
        """
        Compute gradient using central finite differences.
        
        For each axis, we sample SDF at +/- epsilon and compute:
            ∂d/∂x ≈ (d(x+ε) - d(x-ε)) / (2ε)
        
        This gives smooth gradients (C0 continuous).
        """
        eps = 0.5  # Half a voxel in grid coordinates
        
        # Sample at offset positions
        d_xp = self._trilinear_interpolate(grid_coords + np.array([eps, 0, 0]))
        d_xm = self._trilinear_interpolate(grid_coords + np.array([-eps, 0, 0]))
        d_yp = self._trilinear_interpolate(grid_coords + np.array([0, eps, 0]))
        d_ym = self._trilinear_interpolate(grid_coords + np.array([0, -eps, 0]))
        d_zp = self._trilinear_interpolate(grid_coords + np.array([0, 0, eps]))
        d_zm = self._trilinear_interpolate(grid_coords + np.array([0, 0, -eps]))
        
        # Central differences (in grid coordinates)
        grad_grid = np.stack([
            (d_xp - d_xm) / (2 * eps),
            (d_yp - d_ym) / (2 * eps),
            (d_zp - d_zm) / (2 * eps)
        ], axis=1)
        
        # Convert to world coordinates (divide by voxel_size)
        # Actually the ratio cancels out when we normalize, but let's be precise
        grad_world = grad_grid / self.voxel_size
        
        # Normalize to unit vectors
        norms = np.linalg.norm(grad_world, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        gradients = grad_world / norms
        
        return gradients
    
    def query_single(self, point):
        """Convenience method for single point query."""
        d, g = self.query(point.reshape(1, 3))
        return d[0], g[0]


def visualize_sdf_slice(sdf, slice_axis='z', slice_value=None, resolution=100, extent=0.0):
    """
    Visualize a 2D slice of the SDF field.
    """
    bounds_min = sdf.bounds_min
    bounds_max = sdf.bounds_max
    center = (bounds_min + bounds_max) / 2
    
    if slice_value is None:
        slice_value = center[{'x': 0, 'y': 1, 'z': 2}[slice_axis]]
    
    # Create 2D grid for the slice
    if slice_axis == 'z':
        x = np.linspace(bounds_min[0] - extent, bounds_max[0] + extent, resolution)
        y = np.linspace(bounds_min[1] - extent, bounds_max[1] + extent, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, slice_value)
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        xlabel, ylabel = 'X (m)', 'Y (m)'
        
    elif slice_axis == 'y':
        x = np.linspace(bounds_min[0] - extent, bounds_max[0] + extent, resolution)
        z = np.linspace(bounds_min[2] - extent, bounds_max[2] + extent, resolution)
        X, Z = np.meshgrid(x, z)
        Y = np.full_like(X, slice_value)
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        xlabel, ylabel = 'X (m)', 'Z (m)'
        
    else:  # x
        y = np.linspace(bounds_min[1] - extent, bounds_max[1] + extent, resolution)
        z = np.linspace(bounds_min[2] - extent, bounds_max[2] + extent, resolution)
        Y, Z = np.meshgrid(y, z)
        X = np.full_like(Y, slice_value)
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        xlabel, ylabel = 'Y (m)', 'Z (m)'
    
    # Query SDF
    print(f"Querying {len(points)} points...")
    distances, gradients = sdf.query(points)
    
    # Reshape for plotting
    D = distances.reshape(resolution, resolution)
    
    # Get the 2D gradient components for arrows
    if slice_axis == 'z':
        Gx = gradients[:, 0].reshape(resolution, resolution)
        Gy = gradients[:, 1].reshape(resolution, resolution)
        grid_x, grid_y = X, Y
    elif slice_axis == 'y':
        Gx = gradients[:, 0].reshape(resolution, resolution)
        Gy = gradients[:, 2].reshape(resolution, resolution)
        grid_x, grid_y = X, Z
    else:
        Gx = gradients[:, 1].reshape(resolution, resolution)
        Gy = gradients[:, 2].reshape(resolution, resolution)
        grid_x, grid_y = Y, Z
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Distance field with contours
    ax1 = axes[0]
    max_abs = np.max(np.abs(D))
    im = ax1.contourf(grid_x, grid_y, D, levels=50, cmap='RdBu', vmin=-max_abs, vmax=max_abs)
    ax1.contour(grid_x, grid_y, D, levels=[0], colors='black', linewidths=2)
    
    plt.colorbar(im, ax=ax1, label='Signed Distance (m)')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(f'SDF Slice ({slice_axis}={slice_value:.4f}m)\nBlue=Outside, Red=Inside, Black=Surface')
    ax1.set_aspect('equal')
    
    # Right: Gradient field (arrows)
    ax2 = axes[1]
    skip = max(1, resolution // 20)
    ax2.contourf(grid_x, grid_y, D, levels=50, cmap='RdBu', vmin=-max_abs, vmax=max_abs, alpha=0.5)
    ax2.contour(grid_x, grid_y, D, levels=[0], colors='black', linewidths=2)
    
    ax2.quiver(grid_x[::skip, ::skip], grid_y[::skip, ::skip],
               Gx[::skip, ::skip], Gy[::skip, ::skip],
               scale=30, width=0.003, color='darkblue', alpha=0.8)
    
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_title('Gradient Field\nArrows point away from surface')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('sdf_visualization.png', dpi=150)
    print("Saved: sdf_visualization.png")
    plt.show()


def visualize_3d_with_samples(sdf, mesh_path, n_samples=500):
    """
    3D visualization: mesh + sample points colored by distance + gradient arrows.
    """
    bounds_min = sdf.bounds_min
    bounds_max = sdf.bounds_max
    
    # Random sample points around the mesh
    points = np.random.uniform(bounds_min, bounds_max, size=(n_samples, 3))
    
    distances, gradients = sdf.query(points)
    
    # PyVista visualization
    p = pv.Plotter()
    
    # Add mesh
    pv_mesh = pv.read(mesh_path)
    p.add_mesh(pv_mesh, color='lightgray', opacity=0.5)
    
    # Add sample points colored by distance
    point_cloud = pv.PolyData(points)
    point_cloud['distance'] = distances
    p.add_mesh(point_cloud, scalars='distance', cmap='RdBu', 
               point_size=8, render_points_as_spheres=True,
               clim=[-0.02, 0.02])
    
    # Add gradient arrows (subsample for clarity)
    arrow_idx = np.random.choice(len(points), size=min(100, len(points)), replace=False)
    for idx in arrow_idx:
        pt = points[idx]
        dir = gradients[idx]
        arrow = pv.Arrow(start=pt, direction=dir, scale=0.005)
        p.add_mesh(arrow, color='green')
    
    p.add_title("3D SDF: Blue=Outside, Red=Inside, Arrows=Gradient")
    p.show()


# =============================================
# MAIN
# =============================================
if __name__ == "__main__":
    import os
    
    grid_file = "sdf_grid.npz"
    mesh_file = "mon_bol_thick_shell.obj"
    
    if not os.path.exists(grid_file):
        print(f"❌ Grid not found: {grid_file}")
        print("   Run precompute_sdf.py first.")
        exit(1)
    
    # Create SDF object (fast queries)
    sdf = MeshSDF(grid_file)
    
    # Test single query
    center = (sdf.bounds_min + sdf.bounds_max) / 2
    test_point = center + np.array([0.02, 0, 0])
    d, g = sdf.query_single(test_point)
    print(f"\nTest query at {test_point}:")
    print(f"   Distance: {d*1000:.2f} mm")
    print(f"   Gradient: {g}")
    
    # Visualize 2D slice
    print("\n--- 2D Slice Visualization ---")
    visualize_sdf_slice(sdf, slice_axis='y', resolution=80)
    
    # Visualize 3D
    if os.path.exists(mesh_file):
        print("\n--- 3D Visualization ---")
        visualize_3d_with_samples(sdf, mesh_file, n_samples=300)

        