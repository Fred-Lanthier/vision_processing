import open3d as o3d
import pyvista as pv
import numpy as np
import rospkg
import os
import json
import copy

def convert_depth_to_xyz(depth_map, intrinsics):
    """
    Convertit une depth map masquée (2D) en nuage de points (N, 3).
    """
    rows, cols = depth_map.shape
    v, u = np.meshgrid(range(rows), range(cols), indexing='ij')
    
    valid_mask = depth_map > 0
    z_valid = depth_map[valid_mask]
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['ppx'], intrinsics['ppy']
    scale = intrinsics.get('depth_scale', 0.001) 
    
    if np.max(z_valid) > 10.0: 
        z_valid = z_valid * scale

    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    z = z_valid

    xyz = np.vstack((x, y, z)).transpose()
    return xyz.astype(np.float64)

def generate_surface_reconstruction(point_cloud_xyz, output_name="surface_finale.obj"):
    """
    Génération robuste : Poisson (Lisse) + Coupe par distance (Ouverture).
    Plus besoin de boucher les trous après coup.
    """
    print(f"\n--- Démarrage Reconstruction (Méthode: Poisson Trimmed) ---")
    print(f"1. Préparation du nuage ({len(point_cloud_xyz)} points)...")
    
    # --- A. PRÉPARATION ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_xyz)
    
    # Nettoyage statistique (garde le signal fort)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.2)
    
    # Estimation des normales (CRUCIAL)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

    # --- B. GÉNÉRATION POISSON (Surface continue par définition) ---
    print("2. Génération de la surface mathématique (Poisson)...")
    # depth=9 ou 10 offre un bon détail. linear_fit=True garde les bords nets.
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=7, width=0, scale=1.1, linear_fit=False
    )
    
    # À ce stade, le mesh est une bulle fermée sans trous, mais avec un "couvercle".

    # --- C. DÉCOUPAGE CHIRURGICAL (La magie opère ici) ---
    print("3. Suppression des zones inventées (Couvercle)...")
    
    # 1. On supprime d'abord les zones de très faible densité (bruit lointain)
    vertices_to_remove = densities < np.quantile(densities, 0.02)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # 2. DÉCOUPAGE PAR DISTANCE (KD-Tree)
    # C'est ça qui remplace le "bouchage de trous". On part d'un objet plein
    # et on ne garde que ce qui est proche des points réels.
    
    # On crée un arbre de recherche sur les points ORIGINAUX
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    vertices = np.asarray(mesh.vertices)
    
    # --- PARAMÈTRE CRITIQUE : TOLÉRANCE ---
    # C'est la distance max autorisée entre la surface créée et un vrai point.
    # - 0.005 (5mm) : Suffisant pour garder les parois lisses interpolées.
    # - Si le bol a des trous dans les parois -> AUGMENTE (ex: 0.008)
    # - Si le couvercle ne part pas -> DIMINUE (ex: 0.003)
    dist_threshold = 0.006
    
    points_to_keep = []
    
    # Pour chaque sommet du maillage généré
    for i in range(len(vertices)):
        # On cherche le point réel le plus proche
        [k, idx, _] = pcd_tree.search_knn_vector_3d(vertices[i], 1)
        nearest_pt = np.asarray(pcd.points)[idx[0]]
        
        # Distance entre la surface générée et la réalité
        dist = np.linalg.norm(vertices[i] - nearest_pt)
        
        # Si c'est proche, on garde (C'est le mur du bol)
        # Si c'est loin, on jette (C'est le couvercle inventé)
        points_to_keep.append(dist < dist_threshold)
            
    # Application de la découpe
    mesh.remove_vertices_by_mask(np.invert(points_to_keep))

    # --- D. LISSAGE FINAL ---
    print("4. Finition (PyVista)...")
    
    # Conversion vers PyVista
    verts = np.asarray(mesh.vertices)
    faces = np.hstack([[3, *face] for face in np.asarray(mesh.triangles)])
    pv_mesh = pv.PolyData(verts, faces)
    
    # Nettoyage des petits îlots isolés qui auraient survécu
    pv_mesh = pv_mesh.extract_largest()
    
    # Lissage de Taubin (Rend la surface dérivable sans perdre le volume)
    smoothed_mesh = pv_mesh.smooth_taubin(n_iter=50, pass_band=0.001)
    
    # Calcul des normales finales
    smoothed_mesh = smoothed_mesh.compute_normals(auto_orient_normals=True)
    
    # Sauvegarde
    smoothed_mesh.save(output_name)
    print(f"✅ Terminé ! Surface propre générée : {output_name}")
    
    # Visualisation
    p = pv.Plotter()
    p.add_mesh(smoothed_mesh, color="lightblue", pbr=True, metallic=0.5)
    p.add_title("Surface Poisson 'Trimmed' (Sans Trous)")
    p.show()

# --- EXECUTION ---
if __name__ == "__main__":
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    
    base_dir = os.path.join(package_path, "src", "vision_processing", "SDF", "Images_Test")
    depth_path = os.path.join(base_dir, "resultat_depth_mask.npy") 
    intrinsics_path = os.path.join(base_dir, "intrinsics.json")
    
    # Chargement
    intrinsics = { "fx": 630.3, "fy": 630.3, "ppx": 320.0, "ppy": 240.0, "depth_scale": 0.001 }
    if os.path.exists(intrinsics_path):
        with open(intrinsics_path, 'r') as f:
            intrinsics = json.load(f)

    if os.path.exists(depth_path):
        depth_data = np.load(depth_path)
        points_3d = convert_depth_to_xyz(depth_data, intrinsics)

        if len(points_3d) > 0:
            generate_surface_reconstruction(points_3d, output_name="mon_bol_propre.obj")
        else:
            print("❌ Erreur : Nuage vide.")
    else:
        print(f"❌ Fichier introuvable : {depth_path}")