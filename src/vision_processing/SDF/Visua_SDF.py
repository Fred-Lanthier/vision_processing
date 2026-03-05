import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rospkg

def load_sdf_grid(grid_path):
    """Load precomputed SDF grid."""
    data = np.load(grid_path)
    return {
        'sdf_grid': data['sdf_grid'],
        'bounds_min': data['bounds_min'],
        'bounds_max': data['bounds_max'],
        'voxel_size': float(data['voxel_size']),
        'n_voxels': data['n_voxels']
    }


def trilinear_interpolate(points, sdf_data):
    """Trilinear interpolation of SDF values."""
    grid = sdf_data['sdf_grid']
    bounds_min = sdf_data['bounds_min']
    voxel_size = sdf_data['voxel_size']
    n_voxels = np.array(grid.shape)
    
    grid_coords = (points - bounds_min) / voxel_size - 0.5
    gc = np.clip(grid_coords, 0, n_voxels - 1.001)
    
    i0 = np.floor(gc).astype(int)
    i1 = np.minimum(i0 + 1, n_voxels - 1)
    t = gc - i0
    
    d000 = grid[i0[:, 0], i0[:, 1], i0[:, 2]]
    d001 = grid[i0[:, 0], i0[:, 1], i1[:, 2]]
    d010 = grid[i0[:, 0], i1[:, 1], i0[:, 2]]
    d011 = grid[i0[:, 0], i1[:, 1], i1[:, 2]]
    d100 = grid[i1[:, 0], i0[:, 1], i0[:, 2]]
    d101 = grid[i1[:, 0], i0[:, 1], i1[:, 2]]
    d110 = grid[i1[:, 0], i1[:, 1], i0[:, 2]]
    d111 = grid[i1[:, 0], i1[:, 1], i1[:, 2]]
    
    tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
    
    c00 = d000 * (1 - tx) + d100 * tx
    c01 = d001 * (1 - tx) + d101 * tx
    c10 = d010 * (1 - tx) + d110 * tx
    c11 = d011 * (1 - tx) + d111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    
    return c0 * (1 - tz) + c1 * tz


def visualize_sdf_slice_2d(sdf_data, slice_axis='y', slice_value=None, resolution=100,
                           save_path='sdf_slice_2d.png'):
    """
    2D visualization: slice through the SDF field.
    
    Color convention:
        - Red = Negative SDF = Inside object (collision)
        - Blue = Positive SDF = Outside object (safe)
        - Black line = SDF = 0 (surface)
    """
    grid = sdf_data['sdf_grid']
    bounds_min = sdf_data['bounds_min']
    bounds_max = sdf_data['bounds_max']
    
    center = (bounds_min + bounds_max) / 2
    
    if slice_value is None:
        slice_value = center[{'x': 0, 'y': 1, 'z': 2}[slice_axis]]
    
    if slice_axis == 'y':
        x = np.linspace(bounds_min[0], bounds_max[0], resolution)
        z = np.linspace(bounds_min[2], bounds_max[2], resolution)
        X, Z = np.meshgrid(x, z)
        Y = np.full_like(X, slice_value)
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        xlabel, ylabel = 'X (m)', 'Z (m)'
        grid_x, grid_y = X, Z
    elif slice_axis == 'z':
        x = np.linspace(bounds_min[0], bounds_max[0], resolution)
        y = np.linspace(bounds_min[1], bounds_max[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, slice_value)
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        xlabel, ylabel = 'X (m)', 'Y (m)'
        grid_x, grid_y = X, Y
    else:
        y = np.linspace(bounds_min[1], bounds_max[1], resolution)
        z = np.linspace(bounds_min[2], bounds_max[2], resolution)
        Y, Z = np.meshgrid(y, z)
        X = np.full_like(Y, slice_value)
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        xlabel, ylabel = 'Y (m)', 'Z (m)'
        grid_x, grid_y = Y, Z
    
    distances = trilinear_interpolate(points, sdf_data)
    D = distances.reshape(resolution, resolution)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    max_val = np.max(D)
    min_val = np.min(D)
    
    if min_val < 0 and max_val > 0:
        neg_ratio = abs(min_val) / (abs(min_val) + max_val)
        n_neg = max(1, int(256 * neg_ratio))
        n_pos = 256 - n_neg
        
        red_colors = np.ones((n_neg, 4))
        red_colors[:, 0] = 1.0
        red_colors[:, 1] = 0.0
        red_colors[:, 2] = 0.0
        
        blue_gradient = np.zeros((n_pos, 4))
        blue_gradient[:, 0] = np.linspace(1.0, 0.0, n_pos)
        blue_gradient[:, 1] = np.linspace(1.0, 0.0, n_pos)
        blue_gradient[:, 2] = 1.0
        blue_gradient[:, 3] = 1.0
        
        all_colors = np.vstack([red_colors, blue_gradient])
        custom_cmap = mcolors.ListedColormap(all_colors)
        
        im = ax.contourf(grid_x, grid_y, D, levels=50, cmap=custom_cmap,
                         vmin=min_val, vmax=max_val)
    else:
        max_abs = np.max(np.abs(D))
        im = ax.contourf(grid_x, grid_y, D, levels=50, cmap='RdBu',
                         vmin=-max_abs, vmax=max_abs)
    
    ax.contour(grid_x, grid_y, D, levels=[0], colors='black', linewidths=2)
    
    plt.colorbar(im, ax=ax, label='Signed Distance (m)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'SDF Slice ({slice_axis}={slice_value:.4f}m)\n'
                 f'Red = Inside (collision), Blue = Outside (safe), Black = Surface')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def visualize_3d_isosurfaces(sdf_data, mesh_path, distances_to_show=[0, 0.01, 0.02],
                             save_path='sdf_isosurfaces_3d.png'):
    """
    3D visualization: mesh + SDF isosurfaces.
    """
    grid = sdf_data['sdf_grid']
    bounds_min = sdf_data['bounds_min']
    bounds_max = sdf_data['bounds_max']
    
    x = np.linspace(bounds_min[0], bounds_max[0], grid.shape[0])
    y = np.linspace(bounds_min[1], bounds_max[1], grid.shape[1])
    z = np.linspace(bounds_min[2], bounds_max[2], grid.shape[2])
    
    pv_grid = pv.RectilinearGrid(x, y, z)
    pv_grid['sdf'] = grid.flatten(order='F')
    
    mesh = pv.read(mesh_path)
    
    p = pv.Plotter(off_screen=True, window_size=[1600, 1200])
    p.add_mesh(mesh, color='lightblue', opacity=0.3, label='Mesh')
    
    colors = ['green', 'yellow', 'orange', 'red']
    for i, d in enumerate(distances_to_show):
        try:
            iso = pv_grid.contour([d], scalars='sdf')
            if iso.n_points > 0:
                color = colors[i % len(colors)]
                p.add_mesh(iso, color=color, opacity=0.5, label=f'SDF = {d*1000:.0f}mm')
        except:
            print(f"Could not create isosurface at d={d}")
    
    p.add_legend()
    p.add_title("SDF Isosurfaces")
    p.screenshot(save_path)
    print(f"Saved: {save_path}")
    p.close()


def visualize_3d_with_samples(sdf_data, mesh_path, n_samples=1000,
                              save_path='sdf_samples_3d.png'):
    """
    3D visualization: mesh + random sample points colored by SDF value.
    """
    bounds_min = sdf_data['bounds_min']
    bounds_max = sdf_data['bounds_max']
    
    points = np.random.uniform(bounds_min, bounds_max, size=(n_samples, 3))
    distances = trilinear_interpolate(points, sdf_data)
    
    mesh = pv.read(mesh_path)
    max_abs = max(abs(distances.min()), abs(distances.max()), 0.03)
    
    p = pv.Plotter(off_screen=True, window_size=[1600, 1200])
    p.add_mesh(mesh, color='lightgray', opacity=0.4)
    
    point_cloud = pv.PolyData(points)
    point_cloud['SDF (m)'] = distances
    p.add_mesh(point_cloud, scalars='SDF (m)', cmap='RdBu',
               point_size=8, render_points_as_spheres=True,
               clim=[-max_abs, max_abs])
    
    p.add_title("SDF Field: Red = Inside (collision), Blue = Outside (safe)")
    p.screenshot(save_path)
    print(f"Saved: {save_path}")
    p.close()


def visualize_3d_slice(sdf_data, mesh_path, slice_axis='y', slice_value=None,
                       save_path='sdf_3d_slice.png'):
    """
    3D visualization: mesh with a colored slice plane showing SDF values.
    """
    grid = sdf_data['sdf_grid']
    bounds_min = sdf_data['bounds_min']
    bounds_max = sdf_data['bounds_max']
    
    center = (bounds_min + bounds_max) / 2
    if slice_value is None:
        slice_value = center[{'x': 0, 'y': 1, 'z': 2}[slice_axis]]
    
    x = np.linspace(bounds_min[0], bounds_max[0], grid.shape[0])
    y = np.linspace(bounds_min[1], bounds_max[1], grid.shape[1])
    z = np.linspace(bounds_min[2], bounds_max[2], grid.shape[2])
    
    pv_grid = pv.RectilinearGrid(x, y, z)
    pv_grid['sdf'] = grid.flatten(order='F')
    
    if slice_axis == 'x':
        normal = [1, 0, 0]
    elif slice_axis == 'y':
        normal = [0, 1, 0]
    else:
        normal = [0, 0, 1]
    
    origin = center.copy()
    origin[{'x': 0, 'y': 1, 'z': 2}[slice_axis]] = slice_value
    
    slice_mesh = pv_grid.slice(normal=normal, origin=origin)
    mesh = pv.read(mesh_path)
    max_abs = np.max(np.abs(grid))
    
    p = pv.Plotter(off_screen=True, window_size=[1600, 1200])
    p.add_mesh(mesh, color='lightgray', opacity=0.3)
    p.add_mesh(slice_mesh, scalars='sdf', cmap='RdBu',
               clim=[-max_abs, max_abs], show_scalar_bar=True)
    p.add_title(f"SDF Slice ({slice_axis}={slice_value:.3f}m)\n"
                f"Red = Inside (collision), Blue = Outside (safe)")
    p.screenshot(save_path)
    print(f"Saved: {save_path}")
    p.close()


# =============================================
# MAIN
# =============================================
if __name__ == "__main__":
    import os
    package_path = rospkg.RosPack().get_path('vision_processing')
    grid_file = os.path.join(package_path, "models", "sdf_field_No_outter.npz")
    mesh_file = "03_surface_aligned.obj"
    
    if not os.path.exists(grid_file):
        print(f"❌ Grid not found: {grid_file}")
        exit(1)
    
    if not os.path.exists(mesh_file):
        print(f"❌ Mesh not found: {mesh_file}")
        exit(1)
    
    print("Loading SDF grid...")
    sdf_data = load_sdf_grid(grid_file)
    print(f"   Grid shape: {sdf_data['sdf_grid'].shape}")
    print(f"   Voxel size: {sdf_data['voxel_size'] * 1000:.2f} mm")
    print(f"   SDF range: [{sdf_data['sdf_grid'].min():.4f}, {sdf_data['sdf_grid'].max():.4f}]")
    
    print("\n--- 2D Slice ---")
    visualize_sdf_slice_2d(sdf_data, slice_axis='y', resolution=500,
                           save_path='sdf_slice_2d.png')
    
    print("\n--- 3D: Sample Points ---")
    visualize_3d_with_samples(sdf_data, mesh_file, n_samples=500,
                              save_path='sdf_samples_3d.png')
    
    print("\n--- 3D: Slice Plane ---")
    visualize_3d_slice(sdf_data, mesh_file, slice_axis='y',
                       save_path='sdf_3d_slice.png')
    
    print("\n--- 3D: Isosurfaces ---")
    visualize_3d_isosurfaces(sdf_data, mesh_file,
                             distances_to_show=[0, 0.005, 0.01, 0.02],
                             save_path='sdf_isosurfaces_3d.png')
    
    print("\n✅ All figures saved.")