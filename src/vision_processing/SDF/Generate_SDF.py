import open3d as o3d
import pyvista as pv
import pymeshlab
import numpy as np
from pysdf import SDF
import trimesh
import matplotlib.pyplot as plt
import time
import tempfile
import os
import json

class GenerateSDF:
    """
    Pipeline to create a volumetric SDF from a depth map using 'Virtual Thickness'.
    
    Instead of creating a thick geometric shell, this computes the distance to the 
    thin surface and subtracts a radius.
    
    Math: 
        SDF(x) = |DistanceToSurface(x)| - Thickness_Radius
    
    Result:
        SDF < 0 : Inside the object (or within the thickness buffer)
        SDF = 0 : On the virtual boundary
        SDF > 0 : Outside (safe)
    """
    
    def __init__(self, depth_map=None, intrinsics=None):
        self.depth_map = depth_map
        self.intrinsics = intrinsics
        self.point_cloud_xyz = None
        
        if depth_map is not None:
            self.point_cloud_xyz = self._convert_depth_to_xyz()
        
        # Intermediate mesh (PyVista) - Single layer only
        self.surface_mesh = None
        
        # SDF grid and metadata
        self.sdf_grid = None
        self.bounds_min = None
        self.bounds_max = None
        self.voxel_size = None
        self.n_voxels = None
        self.virtual_radius = 0.0  # Stores the radius used for calculation
    
    @classmethod
    def from_file(cls, path):
        instance = cls()
        instance.load(path)
        return instance
    
    # =========================================================
    # PIPELINE
    # =========================================================
    
    def run_pipeline(self, thickness_mm=2.0, target_faces=10000, sdf_resolution=250,
                     save_meshes=True, output_dir="."):
        """
        Run the pipeline: Depth -> Surface -> SDF (Virtual Thickness)
        """
        print("=" * 60)
        print(f"SDF Generation (Virtual Thickness Mode: {thickness_mm}mm)")
        print("=" * 60)
        
        t_start = time.time()
        
        # Step 1: Poisson reconstruction
        self._generate_surface_reconstruction()
        if save_meshes:
            path = os.path.join(output_dir, "01_surface_raw.obj")
            self.surface_mesh.save(path)
            print(f"   💾 Saved: {path}")
        
        # Step 2: Polish (hole repair, smooth, decimate)
        self._polish_mesh(target_faces=target_faces)
        if save_meshes:
            path = os.path.join(output_dir, "02_surface_polished.obj")
            self.surface_mesh.save(path)
            print(f"   💾 Saved: {path}")
        
        # Step 3: PCA Align
        self._pca_align()
        if save_meshes:
            path = os.path.join(output_dir, "03_surface_aligned.obj")
            self.surface_mesh.save(path)
            print(f"   💾 Saved: {path}")
        
        # Step 4: Compute SDF (Virtual Thickness)
        # We want total thickness of X mm, so we expand X/2 mm in every direction
        radius_mm = thickness_mm / 2.0
        self._compute_sdf(resolution=sdf_resolution, virtual_radius_mm=radius_mm)
        
        elapsed = time.time() - t_start
        
        print("=" * 60)
        print(f"✅ Pipeline complete in {elapsed:.1f}s")
        print(f"   SDF grid shape: {self.sdf_grid.shape}")
        print("=" * 60)
    
    def cleanup(self):
        self.surface_mesh = None
    
    def _convert_depth_to_xyz(self):
        rows, cols = self.depth_map.shape
        v, u = np.meshgrid(range(rows), range(cols), indexing='ij')
        
        valid_mask = self.depth_map > 0
        z_valid = self.depth_map[valid_mask]
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        
        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        cx, cy = self.intrinsics['ppx'], self.intrinsics['ppy']
        scale = self.intrinsics.get('depth_scale', 0.001)
        
        if np.max(z_valid) > 10.0:
            z_valid = z_valid * scale

        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy
        z = z_valid

        xyz = np.vstack((x, y, z)).transpose()
        return xyz.astype(np.float64)

    def _generate_surface_reconstruction(self):
        print("\n[1/3] Surface Reconstruction (Poisson)")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud_xyz)
        
        # 1. OUTLIER REMOVAL: Keep this, it removes flying noise
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.2)
        
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=30)
        )
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=7, width=0, scale=1.1, linear_fit=False
        )

        # --- FIX 1: DENSITY REMOVAL ---
        # The sides have low density because the camera missed them. 
        # By removing low density, we delete the walls.
        # We lower the threshold drastically (or comment it out entirely).
        
        # OLD: vertices_to_remove = densities < np.quantile(densities, 0.02)
        # NEW: Lower threshold to keep more "interpolated" areas
        vertices_to_remove = densities < np.quantile(densities, 0.005) 
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # --- FIX 2: DISTANCE THRESHOLD ---
        # This deletes mesh parts that are too far from original depth points.
        # Since the camera missed the sides, the "reconstructed" side is far from any point.
        # We increase this tolerance so the sides are not deleted.
        
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        vertices = np.asarray(mesh.vertices)
        
        # OLD: dist_threshold = 0.006 (6mm) -> Too strict for missing sides
        # NEW: dist_threshold = 0.02 (20mm) -> Allows interpolation up to 2cm
        dist_threshold = 0.02 
        
        points_to_keep = []
        for i in range(len(vertices)):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(vertices[i], 1)
            dist = np.linalg.norm(vertices[i] - np.asarray(pcd.points)[idx[0]])
            points_to_keep.append(dist < dist_threshold)

        mesh.remove_vertices_by_mask(np.invert(points_to_keep))

        verts = np.asarray(mesh.vertices)
        faces = np.hstack([[3, *face] for face in np.asarray(mesh.triangles)])
        pv_mesh = pv.PolyData(verts, faces)
        
        # Keep only the main object (removes floating noise islands)
        pv_mesh = pv_mesh.extract_largest()
        
        # Smooth lightly
        pv_mesh = pv_mesh.smooth_taubin(n_iter=50, pass_band=0.001)
        
        self.surface_mesh = pv_mesh
        print(f"   Output: {pv_mesh.n_points} vertices, {pv_mesh.n_cells} faces")

    def _polish_mesh(self, target_faces=10000):
        print("\n[2/3] Mesh Polishing")
        
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
            temp_path = f.name
        self.surface_mesh.save(temp_path)
        
        try:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(temp_path)
            
            # Basic cleanup
            ms.apply_filter('meshing_remove_duplicate_faces')
            ms.apply_filter('meshing_repair_non_manifold_vertices')
            
            # Smooth
            try:
                ms.apply_filter('apply_coord_laplacian_smoothing_surface_preserving',
                                angledeg=270, iterations=20, selection=False)
            except:
                pass
            
            # Decimate if needed
            if ms.current_mesh().face_number() > target_faces:
                ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                                targetfacenum=target_faces,
                                preserveboundary=True,
                                preservenormal=True,
                                preservetopology=True,
                                qualitythr=0.5)
            
            ms.save_current_mesh(temp_path)
            self.surface_mesh = pv.read(temp_path)
            print(f"   Output: {self.surface_mesh.n_cells} faces")
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _pca_align(self):
        print("   Aligning mesh to Origin via PCA...")
        
        points = self.surface_mesh.points
        
        # 1. D'ABORD : Centrer au centre de gravité (Centroid)
        # C'est crucial pour que la rotation PCA se fasse "sur place"
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid

        # 2. Calcul de la rotation (PCA) sur les points centrés
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Trier du plus grand axe au plus petit (X=Long, Y=Moyen, Z=Court)
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sort_indices]

        # Garantir un système main-droite
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] = -eigenvectors[:, 2]

        R = eigenvectors

        # 3. Appliquer la rotation aux points DÉJÀ centrés
        # aligned_points sera maintenant centré en (0,0,0) et aligné
        aligned_points = centered_points @ R
        
        # 4. (Optionnel) "Poser" l'objet sur le sol (Z=0)
        # On regarde le point le plus bas en Z et on remonte tout le mesh
        min_z = np.min(aligned_points[:, 2])
        aligned_points[:, 2] -= min_z

        # 5. Mettre à jour le mesh
        self.surface_mesh.points = aligned_points
        
        print(f"   Mesh aligned and placed on floor (Z=0).")

    def _compute_sdf(self, resolution=150, margin=0.03, virtual_radius_mm=1.0):
        """
        Compute SDF: |Distance| - virtual_radius
        """
        print(f"\n[3/3] Computing SDF (resolution={resolution})")
        print(f"   Logic: |Distance_to_Surface| - {virtual_radius_mm}mm")
        
        self.virtual_radius = virtual_radius_mm / 1000.0  # Convert to meters
        
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
            temp_path = f.name
        self.surface_mesh.save(temp_path)
        
        try:
            mesh = trimesh.load(temp_path)
            
            self.bounds_min = mesh.bounds[0] - margin
            self.bounds_max = mesh.bounds[1] + margin
            extent = self.bounds_max - self.bounds_min
            
            self.voxel_size = float(np.max(extent) / resolution)
            self.n_voxels = np.ceil(extent / self.voxel_size).astype(int)
            
            print(f"   Voxel size: {self.voxel_size * 1000:.2f} mm")
            
            # Prepare Grid
            x = np.linspace(self.bounds_min[0] + self.voxel_size/2,
                            self.bounds_min[0] + (self.n_voxels[0] - 0.5) * self.voxel_size,
                            self.n_voxels[0])
            y = np.linspace(self.bounds_min[1] + self.voxel_size/2,
                            self.bounds_min[1] + (self.n_voxels[1] - 0.5) * self.voxel_size,
                            self.n_voxels[1])
            z = np.linspace(self.bounds_min[2] + self.voxel_size/2,
                            self.bounds_min[2] + (self.n_voxels[2] - 0.5) * self.voxel_size,
                            self.n_voxels[2])
            
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            all_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            
            # Compute Raw Distance
            t_start = time.time()
            sdf_func = SDF(mesh.vertices, mesh.faces)
            raw_signed_distances = sdf_func(all_points)
            
            # KEY CHANGE: Use Absolute distance - Radius
            # This makes it insensitive to normal flips or open meshes
            final_sdf = np.abs(raw_signed_distances) - self.virtual_radius
            
            elapsed = time.time() - t_start
            print(f"   Computed in {elapsed:.2f}s")
            
            self.sdf_grid = final_sdf.reshape(self.n_voxels).astype(np.float32)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # =========================================================
    # QUERY
    # =========================================================
    
    def query(self, points):
        if self.sdf_grid is None:
            raise RuntimeError("SDF not computed.")
        
        points = np.atleast_2d(points).astype(np.float64)
        grid_coords = (points - self.bounds_min) / self.voxel_size - 0.5
        
        distances = self._trilinear_interpolate(grid_coords)
        gradients = self._compute_gradient(grid_coords)
        
        return distances, gradients
    
    def query_single(self, point):
        d, g = self.query(point.reshape(1, 3))
        return float(d[0]), g[0]
    
    def _trilinear_interpolate(self, grid_coords):
        gc = np.clip(grid_coords, 0, np.array(self.n_voxels) - 1.001)
        i0 = np.floor(gc).astype(int)
        i1 = np.minimum(i0 + 1, np.array(self.n_voxels) - 1)
        t = gc - i0
        
        d000 = self.sdf_grid[i0[:, 0], i0[:, 1], i0[:, 2]]
        d001 = self.sdf_grid[i0[:, 0], i0[:, 1], i1[:, 2]]
        d010 = self.sdf_grid[i0[:, 0], i1[:, 1], i0[:, 2]]
        d011 = self.sdf_grid[i0[:, 0], i1[:, 1], i1[:, 2]]
        d100 = self.sdf_grid[i1[:, 0], i0[:, 1], i0[:, 2]]
        d101 = self.sdf_grid[i1[:, 0], i0[:, 1], i1[:, 2]]
        d110 = self.sdf_grid[i1[:, 0], i1[:, 1], i0[:, 2]]
        d111 = self.sdf_grid[i1[:, 0], i1[:, 1], i1[:, 2]]
        
        c00 = d000 * (1 - t[:, 0]) + d100 * t[:, 0]
        c01 = d001 * (1 - t[:, 0]) + d101 * t[:, 0]
        c10 = d010 * (1 - t[:, 0]) + d110 * t[:, 0]
        c11 = d011 * (1 - t[:, 0]) + d111 * t[:, 0]
        
        c0 = c00 * (1 - t[:, 1]) + c10 * t[:, 1]
        c1 = c01 * (1 - t[:, 1]) + c11 * t[:, 1]
        
        return c0 * (1 - t[:, 2]) + c1 * t[:, 2]
    
    def _compute_gradient(self, grid_coords):
        eps = 0.5
        d_xp = self._trilinear_interpolate(grid_coords + np.array([eps, 0, 0]))
        d_xm = self._trilinear_interpolate(grid_coords + np.array([-eps, 0, 0]))
        d_yp = self._trilinear_interpolate(grid_coords + np.array([0, eps, 0]))
        d_ym = self._trilinear_interpolate(grid_coords + np.array([0, -eps, 0]))
        d_zp = self._trilinear_interpolate(grid_coords + np.array([0, 0, eps]))
        d_zm = self._trilinear_interpolate(grid_coords + np.array([0, 0, -eps]))
        
        grad = np.stack([(d_xp - d_xm), (d_yp - d_ym), (d_zp - d_zm)], axis=1) / (2 * eps * self.voxel_size)
        norms = np.linalg.norm(grad, axis=1, keepdims=True)
        return grad / np.maximum(norms, 1e-10)
    
    # =========================================================
    # SAVE / LOAD
    # =========================================================
    
    def save(self, path):
        if self.sdf_grid is None: raise RuntimeError("SDF not computed.")
        np.savez(path, sdf_grid=self.sdf_grid, bounds_min=self.bounds_min,
                 bounds_max=self.bounds_max, voxel_size=self.voxel_size,
                 n_voxels=self.n_voxels, virtual_radius=self.virtual_radius)
        print(f"💾 Saved SDF to: {path}")
    
    def load(self, path):
        data = np.load(path)
        self.sdf_grid = data['sdf_grid']
        self.bounds_min = data['bounds_min']
        self.bounds_max = data['bounds_max']
        self.voxel_size = float(data['voxel_size'])
        self.n_voxels = data['n_voxels']
        self.virtual_radius = float(data['virtual_radius']) if 'virtual_radius' in data else 0.0

    # =========================================================
    # VISUALIZATION (Corrected for Virtual Thickness)
    # =========================================================
    
    def visualize_slice(self, slice_axis='z', slice_value=None, resolution=100, 
                        extent=0.0, save_path='sdf_slice.png'):
        """
        Visualize SDF slice.
        - Black Line: The collision boundary (SDF=0)
        - Dashed Line: The original surface (SDF = -radius)
        """
        if self.sdf_grid is None: raise RuntimeError("SDF not computed.")
        
        center = (self.bounds_min + self.bounds_max) / 2
        if slice_value is None:
            slice_value = center[{'x':0,'y':1,'z':2}[slice_axis]]
            
        # Generate grid for slice
        if slice_axis == 'z':
            a1 = np.linspace(self.bounds_min[0]-extent, self.bounds_max[0]+extent, resolution)
            a2 = np.linspace(self.bounds_min[1]-extent, self.bounds_max[1]+extent, resolution)
            A1, A2 = np.meshgrid(a1, a2)
            pts = np.stack([A1.ravel(), A2.ravel(), np.full_like(A1.ravel(), slice_value)], axis=1)
            xlabel, ylabel = 'X', 'Y'
        elif slice_axis == 'y':
            a1 = np.linspace(self.bounds_min[0]-extent, self.bounds_max[0]+extent, resolution)
            a2 = np.linspace(self.bounds_min[2]-extent, self.bounds_max[2]+extent, resolution)
            A1, A2 = np.meshgrid(a1, a2)
            pts = np.stack([A1.ravel(), np.full_like(A1.ravel(), slice_value), A2.ravel()], axis=1)
            xlabel, ylabel = 'X', 'Z'
        else:
            a1 = np.linspace(self.bounds_min[1]-extent, self.bounds_max[1]+extent, resolution)
            a2 = np.linspace(self.bounds_min[2]-extent, self.bounds_max[2]+extent, resolution)
            A1, A2 = np.meshgrid(a1, a2)
            pts = np.stack([np.full_like(A1.ravel(), slice_value), A1.ravel(), A2.ravel()], axis=1)
            xlabel, ylabel = 'Y', 'Z'
            
        d, _ = self.query(pts)
        D = d.reshape(resolution, resolution)
        
        plt.figure(figsize=(10, 8))
        
        # Color Map
        # Red = Inside (Collision), Blue = Outside (Safe)
        max_val = np.max(np.abs(D))
        plt.contourf(A1, A2, D, levels=50, cmap='RdBu', vmin=-max_val, vmax=max_val)
        plt.colorbar(label='Signed Distance (m)')
        
        # SOLID LINE: The Effective Collision Boundary (SDF = 0)
        # Anything crossing this line is a collision.
        plt.contour(A1, A2, D, levels=[0], colors='black', linewidths=2.5)
        
        # DASHED LINE: The Real Object Surface (SDF = -radius)
        # This shows where the actual mesh is deep inside the safety bubble.
        if self.virtual_radius > 0:
            plt.contour(A1, A2, D, levels=[-self.virtual_radius], 
                       colors='black', linestyles='dashed', linewidths=1)
            
        plt.title(f"SDF Slice ({slice_axis}={slice_value:.3f})\nThick Line=Safety Boundary | Dashed=Actual Surface")
        plt.xlabel(f"{xlabel} (m)")
        plt.ylabel(f"{ylabel} (m)")
        plt.axis('equal')
        plt.savefig(save_path)
        print(f"Saved visualization: {save_path}")
        plt.show()

    def visualize_zone_3d(self):
        """
        Visualizes the 3D 'No-Go Zone' (SDF <= 0) as a red translucent shell.
        """
        if self.sdf_grid is None: 
            raise RuntimeError("SDF not computed.")
        
        print("\nVisualizing 3D No-Go Zone...")
        
        # 1. Create a PyVista Grid matching our data
        grid = pv.ImageData()
        grid.dimensions = self.n_voxels
        grid.origin = self.bounds_min
        grid.spacing = (self.voxel_size, self.voxel_size, self.voxel_size)
        
        # 2. Fill it with our SDF data
        # 'F' order flattening matches the (nx, ny, nz) structure of our numpy array
        grid.point_data["SDF"] = self.sdf_grid.flatten(order="F") 
        
        p = pv.Plotter()
        
        # 3. Extract and Plot the "No Go Boundary" (SDF = 0)
        try:
            # Generate the 3D surface where SDF is exactly 0
            collision_shell = grid.contour(isosurfaces=[0.0])
            
            # Plot as a Red, translucent shell
            p.add_mesh(collision_shell, color="red", opacity=0.4, 
                       label="No-Go Zone (Virtual Wall)")
        except Exception as e:
            print(f"Could not generate No-Go shell: {e}")

        # 4. (Optional) Show the "Real" thin surface inside for comparison
        # The real surface is located at SDF = -radius
        if self.virtual_radius > 0:
            try:
                real_surface = grid.contour(isosurfaces=[-self.virtual_radius])
                p.add_mesh(real_surface, color="white", style='wireframe', 
                           label="Actual Object Surface")
            except:
                pass

        p.add_axes()
        p.add_legend()
        p.show()
        p.screenshot("sdf_3d.png")
# ===========================================
# MAIN EXECUTION
# ===========================================
if __name__ == "__main__":
    depth_path = "Images_Test/resultat_depth_mask.npy"
    intrinsics_path = "Images_Test/intrinsics.json"
    
    if os.path.exists(depth_path):
        depth_map = np.load(depth_path)
        with open(intrinsics_path, 'r') as f:
            intrinsics = json.load(f)
        
        sdf = GenerateSDF(depth_map, intrinsics)
        
        # NOTE: thickness_mm is now the TOTAL virtual thickness
        sdf.run_pipeline(thickness_mm=2.0, target_faces=10000, sdf_resolution=200)
        
        sdf.save("Images_Test/sdf_field.npz")
        sdf.visualize_zone_3d()
        sdf.visualize_slice(slice_axis='y', resolution=500)
        sdf.cleanup()
        
    elif os.path.exists("Images_Test/sdf_field.npz"):
        sdf = GenerateSDF.from_file("Images_Test/sdf_field.npz")
        sdf.visualize_slice(slice_axis='y', resolution=100)