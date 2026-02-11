import numpy as np
import open3d as o3d
import os
import json

def _convert_depth_to_xyz(depth_map, intrinsics):
    rows, cols = depth_map.shape
    v, u = np.meshgrid(range(rows), range(cols), indexing='ij')
    
    valid_mask = depth_map > 0
    z_valid = depth_map[valid_mask]
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['ppx'], intrinsics['ppy']
    scale = intrinsics.get('depth_scale', 0.001)
    
    # Conversion simple si l'échelle n'est pas déjà appliquée
    # (Ajustez ce seuil selon vos données brutes, ici > 10.0 suppose des mm)
    if np.max(z_valid) > 10.0:
        z_valid = z_valid * scale

    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    z = z_valid

    xyz = np.vstack((x, y, z)).transpose()
    return xyz.astype(np.float64)

def load_and_clean_data(filepath, intrinsics_path):
    """Charge le fichier .npy et nettoie les valeurs invalides (NaN/Inf)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} est introuvable.")
    
    data = np.load(filepath)
    with open(intrinsics_path, 'r') as f:
        intrinsics = json.load(f)
    
    data = _convert_depth_to_xyz(data, intrinsics)
    
    # Si le fichier est une map de profondeur (H, W) ou (H, W, 3), l'aplatir
    if data.ndim > 2:
        data = data.reshape(-1, 3)
        
    # Vérifier que nous avons bien 3 colonnes (x, y, z)
    if data.shape[1] != 3:
        raise ValueError(f"Format de données incorrect: {data.shape}. Attendu (N, 3).")

    # Supprimer les NaNs et Infs
    data = data[np.isfinite(data).all(axis=1)]
    
    print(f"Données chargées : {len(data)} points valides.")
    return data

def remove_outliers(points, nb_neighbors=50, std_ratio=1.5):
    """
    Supprime les points aberrants (bruit de capteur) via Open3D.
    - nb_neighbors : Combien de voisins analyser pour chaque point.
    - std_ratio : Seuil de tolérance (plus bas = plus strict).
    """
    print(f"Début du débruitage (SOR)... (Entrée: {len(points)} points)")
    
    # 1. Conversion NumPy -> Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 2. Application du filtre statistique
    # remove_statistical_outlier retourne (nuage_propre, liste_indices)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    # 3. Sélection des inliers et retour en NumPy
    pcd_clean = pcd.select_by_index(ind)
    points_clean = np.asarray(pcd_clean.points)
    
    print(f"Débruitage terminé. Points restants : {len(points_clean)} (Supprimés : {len(points) - len(points_clean)})")
    return points_clean

def pca_align(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_indices]

    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] = -eigenvectors[:, 2]

    # FIX: Use eigenvectors directly, not transpose
    R = eigenvectors  # Each COLUMN is a new axis
    aligned_points = centered_points @ R  # Project onto new axes

    return aligned_points, R

def visual_validation(original_pts, aligned_pts):
    """Affiche les deux nuages pour comparaison."""
    
    # 1. Nuage original (Gris) - On peut l'afficher un peu décalé pour mieux voir
    pcd_orig = o3d.geometry.PointCloud()
    pcd_orig.points = o3d.utility.Vector3dVector(original_pts)
    pcd_orig.paint_uniform_color([0.6, 0.6, 0.6]) 
    
    # 2. Nuage aligné et débruité (Cyan)
    pcd_aligned = o3d.geometry.PointCloud()
    pcd_aligned.points = o3d.utility.Vector3dVector(aligned_pts)
    pcd_aligned.paint_uniform_color([0.0, 0.8, 0.8]) 

    # 3. Repère
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    print("\n--- COMMANDES VISUALISATION ---")
    print("Gris : Nuage Original (Bruité)")
    print("Cyan : Nuage Aligné (Débruité & Centré)")
    print("Axes : Rouge=X, Vert=Y, Bleu=Z")
    
    # On affiche tout
    pcd_orig = pcd_orig.voxel_down_sample(voxel_size=0.002)
    pcd_aligned = pcd_aligned.voxel_down_sample(voxel_size=0.002)

    o3d.visualization.draw_geometries([pcd_orig, pcd_aligned, coord_frame], 
                                      window_name="Validation PCA + Denoise")

if __name__ == "__main__":
    # Chemins
    FILE_PATH = "Images_Test/resultat_depth_mask.npy"
    INTRINSICS_PATH = "Images_Test/intrinsics.json"
    
    try:
        # 1. Chargement & Conversion
        raw_points = load_and_clean_data(FILE_PATH, INTRINSICS_PATH)

        # 2. DÉBRUITAGE (Nouvelle étape)
        # Ajustez std_ratio si vous voulez filtrer plus (1.0) ou moins (2.0)
        clean_points = remove_outliers(raw_points, nb_neighbors=50, std_ratio=1.5)

        # 3. Alignement PCA (sur les points propres uniquement)
        aligned_points, rotation_matrix = pca_align(clean_points)
        print(rotation_matrix @ rotation_matrix.T)
        # 4. Post-traitement : Poser sur Z=0
        min_z = np.min(aligned_points[:, 2])
        aligned_points[:, 2] -= min_z
        
        print(f"Alignement terminé. Matrice de rotation :\n{rotation_matrix}")

        # 5. Sauvegarde
        output_path = "Images_Test/nuage_oriente.npy"
        np.save(output_path, aligned_points)
        print(f"Sauvegardé sous : {output_path}")

        # 6. Validation (On compare le brut original vs le propre aligné)
        visual_validation(raw_points, aligned_points)

    except Exception as e:
        print(f"Erreur : {e}")