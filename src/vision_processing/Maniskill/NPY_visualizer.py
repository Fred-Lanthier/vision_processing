import os
import numpy as np
import rospkg
import open3d as o3d

def create_point_cloud_from_depth():
    # =========================================================
    # 1. RÉSOLUTION DU CHEMIN (Ton code)
    # =========================================================
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    path = os.path.join(package_path, 'datas', "Trajectories_preprocess_TEST", 
                        "Trajectory_46", "Merged_Fork_Trajectory_46", "Merged.npy")
    
    # Vérification de sécurité
    if not os.path.exists(path):
        print(f"Erreur : Le fichier est introuvable au chemin : {path}")
        return

    # =========================================================
    # 2. LECTURE DE LA CARTE DE PROFONDEUR
    # =========================================================
    # On charge le fichier .npy. SAPIEN sort généralement des profondeurs en mètres.
    depth_image = np.load(path)
    
    # On s'assure que c'est un tableau 2D (H, W). S'il y a un canal extra (H, W, 1), on l'écrase.
    depth_image = np.squeeze(depth_image) 
    print(max(depth_image.flatten()), min(depth_image.flatten()))  # Affiche les valeurs max et min pour vérifier les unités
    height, width = depth_image.shape
    print(f"Image de profondeur chargée : {width}x{height} pixels.")

    # =========================================================
    # 3. PARAMÈTRES INTRINSÈQUES DE LA CAMÉRA (D415 simulée)
    # =========================================================
    # Les valeurs calculées précédemment (FOV 42°, 640x480)
    fx = 625.22
    fy = 625.22
    cx = 320.0
    cy = 240.0

    # =========================================================
    # 4. VECTORISATION MATHÉMATIQUE (Le secret de la vitesse)
    # =========================================================
    # On crée une grille de coordonnées (u, v) pour chaque pixel
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # On filtre les pixels invalides (profondeur à 0 ou à l'infini)
    # Si ton environnement a un fond vide, la profondeur peut être très grande
    valid_mask = (depth_image > .0) & (depth_image < 8000.0)  # On coupe à 5 mètres max
    
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = depth_image[valid_mask]
    
    # Application des équations sténopé (pinhole)
    x_valid = (u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy
    
    # On empile X, Y, Z pour créer notre tableau de points (N, 3)
    # Note: Dans SAPIEN et OpenCV, l'axe Z pointe vers l'avant, Y vers le bas, X vers la droite.
    points_3d = np.vstack((x_valid, y_valid, z_valid)).T
    print(f"Nuage de points généré avec {points_3d.shape[0]} points valides.")

    # =========================================================
    # 5. VISUALISATION AVEC OPEN3D
    # =========================================================
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # Optionnel : Inverser les axes Y et Z pour un affichage plus intuitif dans la fenêtre
    # car la convention caméra a souvent la tête en bas par rapport au monde 3D
    transform = [[1, 0, 0, 0], 
                 [0, -1, 0, 0], 
                 [0, 0, -1, 0], 
                 [0, 0, 0, 1]]
    pcd.transform(transform)

    print("Ouverture de la fenêtre de visualisation...")
    o3d.visualization.draw_geometries([pcd], window_name="Nuage de Points - Camera EE")

if __name__ == "__main__":
    create_point_cloud_from_depth()