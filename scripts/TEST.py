"""
Precompute Representative Fork Safety Points
=============================================

Goal: From the fork STL mesh, extract two sets of key points
that capture the fork's collision envelope:

  - 5 points  (fast set)  → used during ODE integration (every step)
  - 50 points (full set)  → used for terminal safety projection (once)

Method:
  1. Load fork.STL
  2. Apply the visual origin transform from the XACRO
     so all points are in the fork_tip frame
  3. Compute convex hull (only outer-envelope vertices matter)
  4. FPS on hull vertices → 5 points  (max spatial coverage)
  5. FPS on ALL vertices  → 50 points (dense coverage)

Output: fork_safety_points.npz containing:
  - fork_points_fast  (5, 3)
  - fork_points_full  (50, 3)
  - fork_all_vertices (N, 3)   ← for visualization / debugging

Usage:
  python precompute_fork_safety_points.py --stl_path path/to/fork.STL --visualize
"""

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
import argparse
import os


# ================================================================
# 1. VISUAL ORIGIN TRANSFORM (from your panda_camera.xacro)
# ================================================================
# In the xacro, the fork visual has:
#   <origin xyz="-0.033 -0.02 0.0171378" rpy="0 ${27.5 * pi / 180} 0"/>
#
# This means: STL vertices are in the mesh's own frame.
# To bring them to fork_tip frame, we apply this transform.

def build_visual_origin_transform():
    """
    Build the 4x4 homogeneous transform from the fork STL local frame
    to the fork_tip URDF link frame.

    From xacro:  xyz = (-0.033, -0.02, 0.0171378)
                 rpy = (0, 27.5°, 0)  → rotation around Y axis
    """
    tx, ty, tz = -0.033, -0.02, 0.0171378
    angle_y = 27.5 * np.pi / 180.0  # 27.5 degrees in radians

    # Rotation matrix around Y axis
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    R = np.array([
        [ cy, 0, sy],
        [  0, 1,  0],
        [-sy, 0, cy]
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


# ================================================================
# 2. FARTHEST POINT SAMPLING (FPS)
# ================================================================
# We implement a simple greedy FPS here so we don't depend
# on fpsample (which needs specific array types).
# The idea: start from one point, then iteratively pick the point
# that is farthest from all already-selected points.

def farthest_point_sampling(points, n_samples):
    """
    Greedy Farthest Point Sampling.

    Args:
        points:    (N, 3) array of candidate points
        n_samples: how many to pick

    Returns:
        indices: (n_samples,) indices into `points`

    How it works:
        1. Start with the point farthest from the centroid
        2. At each step, compute dist from every unselected point
           to the nearest selected point
        3. Pick the unselected point with the largest such distance
        4. Repeat until we have n_samples

    This maximizes the minimum distance between selected points,
    giving the best spatial spread.
    """
    N = points.shape[0]
    n_samples = min(n_samples, N)

    # Start: pick the point farthest from centroid
    centroid = points.mean(axis=0)
    dists_to_centroid = np.linalg.norm(points - centroid, axis=1)
    first_idx = np.argmax(dists_to_centroid)

    selected = [first_idx]

    # min_dists[i] = distance from point i to the nearest selected point
    min_dists = np.linalg.norm(points - points[first_idx], axis=1)

    for _ in range(n_samples - 1):
        # Pick the point with the largest min distance
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)

        # Update min distances
        new_dists = np.linalg.norm(points - points[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    return np.array(selected)


# ================================================================
# 3. SYNTHETIC FORK (for testing without the real STL)
# ================================================================

def create_synthetic_fork():
    """
    Create a rough approximation of a fork geometry for testing.
    The real script uses the actual fork.STL.

    Geometry (in STL local frame, before visual origin transform):
      - Handle:  cylinder along Z, length ~10cm, radius ~5mm
      - Neck:    tapered section
      - Tines:   4 prongs, each ~3cm long, spread ~8mm apart

    Returns a trimesh object.
    """
    parts = []

    # Handle: cylinder along Z
    handle = trimesh.creation.cylinder(radius=0.005, height=0.10,
                                        sections=16)
    handle.apply_translation([0, 0, 0.05])  # center at z=0.05
    parts.append(handle)

    # Neck: slightly tapered
    neck = trimesh.creation.cylinder(radius=0.004, height=0.03,
                                      sections=16)
    neck.apply_translation([0, 0, 0.115])
    parts.append(neck)

    # Tines: 4 thin cylinders
    tine_offsets_x = [-0.006, -0.002, 0.002, 0.006]
    for tx in tine_offsets_x:
        tine = trimesh.creation.cylinder(radius=0.001, height=0.03,
                                          sections=8)
        tine.apply_translation([tx, 0, 0.145])
        parts.append(tine)

    # Merge all parts
    combined = trimesh.util.concatenate(parts)
    return combined


# ================================================================
# 4. MAIN PIPELINE
# ================================================================

def compute_safety_points(stl_path=None, n_fast=5, n_full=50,
                          visualize=False, output_path="fork_safety_points.npz"):
    """
    Main function: load fork → transform → hull → FPS → save.

    Args:
        stl_path:    path to fork.STL (None = use synthetic)
        n_fast:      number of points for ODE loop (default 5)
        n_full:      number of points for terminal check (default 50)
        visualize:   if True, show 3D plot
        output_path: where to save the .npz
    """

    # ---- Load mesh ----
    if stl_path and os.path.exists(stl_path):
        print(f"Loading STL: {stl_path}")
        mesh = trimesh.load(stl_path)
        # STL scale in xacro is 0.001 (mm → meters)
        mesh.apply_scale(0.001)
    else:
        if stl_path:
            print(f"STL not found at: {stl_path}")
        print("Using synthetic fork geometry for demonstration")
        mesh = create_synthetic_fork()

    print(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # ---- Apply visual origin transform (STL frame → fork_tip frame) ----
    T_visual = build_visual_origin_transform()
    mesh.apply_transform(T_visual)

    vertices = mesh.vertices.copy()  # (N, 3) in fork_tip frame
    print(f"  Bounding box in fork_tip frame:")
    print(f"    min: {vertices.min(axis=0)}")
    print(f"    max: {vertices.max(axis=0)}")

    # ---- Convex hull ----
    hull = ConvexHull(vertices)
    hull_vertex_indices = np.unique(hull.simplices.flatten())
    hull_vertices = vertices[hull_vertex_indices]
    print(f"  Convex hull: {len(hull_vertex_indices)} vertices "
          f"(out of {len(vertices)} total)")

    # ---- FPS on hull → fast set (5 points) ----
    fps_hull_idx = farthest_point_sampling(hull_vertices, n_fast)
    fork_points_fast = hull_vertices[fps_hull_idx]
    print(f"\n  Fast set ({n_fast} points) — for ODE integration:")
    for i, pt in enumerate(fork_points_fast):
        print(f"    [{i}] ({pt[0]:+.4f}, {pt[1]:+.4f}, {pt[2]:+.4f})")

    # ---- FPS on ALL vertices → full set (50 points) ----
    fps_all_idx = farthest_point_sampling(vertices, n_full)
    fork_points_full = vertices[fps_all_idx]
    print(f"\n  Full set ({n_full} points) — for terminal safety check:")
    span = fork_points_full.max(axis=0) - fork_points_full.min(axis=0)
    print(f"    Covers span: ({span[0]:.4f}, {span[1]:.4f}, {span[2]:.4f}) m")

    # ---- Save ----
    np.savez(output_path,
             fork_points_fast=fork_points_fast.astype(np.float32),
             fork_points_full=fork_points_full.astype(np.float32),
             fork_all_vertices=vertices.astype(np.float32),
             T_visual_origin=T_visual,
             n_fast=n_fast,
             n_full=n_full)
    print(f"\n  Saved to: {output_path}")

    # ---- Visualize ----
    if visualize:
        visualize_fork_points(vertices, hull_vertices,
                              fork_points_fast, fork_points_full)

    return fork_points_fast, fork_points_full


# ================================================================
# 5. VISUALIZATION
# ================================================================

def visualize_fork_points(all_vertices, hull_vertices,
                          fast_points, full_points):
    """
    3D visualization with PyVista showing:
      - All fork vertices (grey, small)       → the full geometry
      - Convex hull vertices (light blue)     → the envelope
      - Full 50 points (green, medium)        → terminal set
      - Fast 5 points (red, large)            → ODE set

    Each layer is a different color and size so you can see
    the hierarchy clearly.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not available — skipping visualization.")
        print("Install with: pip install pyvista")
        return

    pv.global_theme.background = 'white'

    plotter = pv.Plotter()
    plotter.set_background('white')

    # Layer 1: All vertices (grey, tiny)
    cloud_all = pv.PolyData(all_vertices)
    plotter.add_mesh(cloud_all, color='lightgrey', point_size=3,
                     render_points_as_spheres=True,
                     label=f'All vertices ({len(all_vertices)})')

    # Layer 2: Convex hull vertices (light blue)
    cloud_hull = pv.PolyData(hull_vertices)
    plotter.add_mesh(cloud_hull, color='cornflowerblue', point_size=6,
                     render_points_as_spheres=True,
                     label=f'Convex hull ({len(hull_vertices)})')

    # Layer 3: Full 50 points (green, medium)
    cloud_full = pv.PolyData(full_points)
    plotter.add_mesh(cloud_full, color='limegreen', point_size=12,
                     render_points_as_spheres=True,
                     label=f'Terminal set ({len(full_points)})')

    # Layer 4: Fast 5 points (red, large) with labels
    cloud_fast = pv.PolyData(fast_points)
    plotter.add_mesh(cloud_fast, color='red', point_size=20,
                     render_points_as_spheres=True,
                     label=f'ODE set ({len(fast_points)})')

    # Add index labels to the 5 fast points
    for i, pt in enumerate(fast_points):
        plotter.add_point_labels(
            pv.PolyData(pt.reshape(1, 3)),
            [f'  P{i}'],
            font_size=14,
            text_color='red',
            bold=True,
            shape=None
        )

    # Add coordinate axes for reference
    plotter.add_axes(line_width=2, labels_off=False)
    plotter.add_legend(bcolor='white', face='circle', size=(0.25, 0.20))

    plotter.add_title("Fork Safety Points\n"
                      "Grey=all | Blue=hull | Green=terminal(50) | Red=ODE(5)",
                      font_size=10)

    # Show the fork_tip origin
    origin = pv.PolyData(np.array([[0.0, 0.0, 0.0]]))
    plotter.add_mesh(origin, color='black', point_size=15,
                     render_points_as_spheres=True)
    plotter.add_point_labels(origin, ['  fork_tip origin'],
                             font_size=12, text_color='black', bold=True)

    plotter.show()


# ================================================================
# 6. HEADLESS VISUALIZATION (saves image instead of showing window)
# ================================================================

def visualize_fork_points_headless(all_vertices, hull_vertices,
                                   fast_points, full_points,
                                   output_image="fork_safety_points.png"):
    """
    Same as visualize_fork_points but renders to an image file
    instead of opening a window. Useful for remote / headless servers.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not available.")
        return

    pv.start_xvfb()  # virtual framebuffer for headless rendering

    plotter = pv.Plotter(off_screen=True, window_size=[1400, 900])
    plotter.set_background('white')

    # All vertices (grey)
    plotter.add_mesh(pv.PolyData(all_vertices), color='lightgrey',
                     point_size=3, render_points_as_spheres=True,
                     label=f'All vertices ({len(all_vertices)})')

    # Convex hull (blue)
    plotter.add_mesh(pv.PolyData(hull_vertices), color='cornflowerblue',
                     point_size=6, render_points_as_spheres=True,
                     label=f'Convex hull ({len(hull_vertices)})')

    # Terminal 50 (green)
    plotter.add_mesh(pv.PolyData(full_points), color='limegreen',
                     point_size=12, render_points_as_spheres=True,
                     label=f'Terminal set ({len(full_points)})')

    # ODE 5 (red)
    plotter.add_mesh(pv.PolyData(fast_points), color='red',
                     point_size=20, render_points_as_spheres=True,
                     label=f'ODE set ({len(fast_points)})')

    # Labels on the 5 fast points
    for i, pt in enumerate(fast_points):
        plotter.add_point_labels(
            pv.PolyData(pt.reshape(1, 3)),
            [f'  P{i}'], font_size=14, text_color='red',
            bold=True, shape=None
        )

    # Origin marker
    plotter.add_mesh(pv.PolyData(np.array([[0.0, 0.0, 0.0]])),
                     color='black', point_size=15,
                     render_points_as_spheres=True)
    plotter.add_point_labels(
        pv.PolyData(np.array([[0.0, 0.0, 0.0]])),
        ['  fork_tip origin'], font_size=12,
        text_color='black', bold=True
    )

    plotter.add_axes(line_width=2, labels_off=False)
    plotter.add_legend(bcolor='white', face='circle', size=(0.25, 0.20))
    plotter.add_title("Fork Safety Points\n"
                      "Grey=all | Blue=hull | Green=terminal(50) | Red=ODE(5)",
                      font_size=10)

    # Camera: look from the side so we see the fork's full length
    center = (all_vertices.max(axis=0) + all_vertices.min(axis=0)) / 2
    extent = np.linalg.norm(all_vertices.max(axis=0) - all_vertices.min(axis=0))
    cam_dist = extent * 2.5
    plotter.camera_position = [
        (center[0] + cam_dist * 0.5, center[1] - cam_dist * 0.8, center[2] + cam_dist * 0.4),
        tuple(center),
        (0, 0, 1)
    ]

    plotter.screenshot(output_image)
    print(f"  Visualization saved to: {output_image}")
    plotter.close()


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute representative fork safety points")
    parser.add_argument("--stl_path", type=str, default=None,
                        help="Path to fork.STL (uses synthetic if not provided)")
    parser.add_argument("--n_fast", type=int, default=5,
                        help="Number of points for ODE loop")
    parser.add_argument("--n_full", type=int, default=50,
                        help="Number of points for terminal safety check")
    parser.add_argument("--visualize", action="store_true",
                        help="Show interactive 3D visualization")
    parser.add_argument("--save_image", action="store_true",
                        help="Save visualization as PNG (headless)")
    parser.add_argument("--output", type=str,
                        default="fork_safety_points.npz",
                        help="Output .npz path")

    args = parser.parse_args()

    fast, full = compute_safety_points(
        stl_path=args.stl_path,
        n_fast=args.n_fast,
        n_full=args.n_full,
        visualize=args.visualize,
        output_path=args.output
    )

    # Headless image if requested
    if args.save_image:
        data = np.load(args.output)
        all_verts = data['fork_all_vertices']
        hull = ConvexHull(all_verts)
        hull_verts = all_verts[np.unique(hull.simplices.flatten())]
        visualize_fork_points_headless(all_verts, hull_verts, fast, full)