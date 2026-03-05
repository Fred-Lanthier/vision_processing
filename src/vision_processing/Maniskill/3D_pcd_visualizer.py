import os
import glob
import numpy as np
import rospkg
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

def get_trajectory_files(traj_id, pcd_type):
    """
    Récupère la liste triée des fichiers .npy pour un type donné.
    """
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    folder_name = f"Trajectory_{traj_id}"
    
    if pcd_type == "Merged":
        subfolder = f"Merged_pcd_{folder_name}"
        prefix = "Merged_"
    elif pcd_type == "Merged_urdf_fork":
        subfolder = f"Merged_urdf_fork_{folder_name}"
        prefix = "Merged_urdf_fork_"
    elif pcd_type == "Merged_Fork":
        subfolder = f"Merged_Fork_{folder_name}"
        prefix = "Merged_Fork_"
    else:
        raise ValueError("Type de PCD invalide.")

    base_path = os.path.join(package_path, 'datas', 'Trajectories_preprocess_TEST', folder_name, subfolder)
    
    if not os.path.exists(base_path):
        print(f"  ❌ Dossier introuvable : {base_path}")
        return [], base_path

    files = glob.glob(os.path.join(base_path, f"{prefix}*.npy"))
    files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    
    return files, base_path

def colorize_by_z(points):
    """ Colore les points selon leur hauteur (Z) """
    z_vals = points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    if z_max == z_min:
        return np.tile([0.8, 0.2, 0.2], (len(points), 1))
    z_norm = (z_vals - z_min) / (z_max - z_min)
    cmap = plt.get_cmap("viridis")
    return cmap(z_norm)[:, :3]

def record_trajectory_animations(traj_id=47, fps=10):
    """
    Génère et sauvegarde une vidéo MP4 pour chaque type de nuage de points.
    """
    pcd_types = ["Merged", "Merged_urdf_fork", "Merged_Fork"]
    
    # Résolution de la vidéo (doit correspondre à la fenêtre Open3D)
    width, height = 1024, 768

    print(f"🎬 Début de l'enregistrement pour la Trajectoire {traj_id}")
    print("="*60)

    for pcd_type in pcd_types:
        files, folder_path = get_trajectory_files(traj_id, pcd_type)
        
        if not files:
            continue
        
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('vision_processing')
        video_folder = os.path.join(package_path, 'src', 'vision_processing', f"Maniskill", "Datas")
        os.makedirs(video_folder, exist_ok=True)
        video_filename = os.path.join(video_folder, f"{pcd_type}_animation.mp4")
        print(f"🎥 Enregistrement en cours : {pcd_type} ({len(files)} frames)")
        print(f"   Destination : {video_filename}")

        # =========================================================
        # 1. INITIALISATION VIDEO WRITER (OpenCV)
        # =========================================================
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out_video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

        # =========================================================
        # 2. INITIALISATION OPEN3D
        # =========================================================
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Render - {pcd_type}", width=width, height=height, visible=True)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.15])
        opt.point_size = 4.0

        pcd = o3d.geometry.PointCloud()
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        
        first_points = np.load(files[0])
        if first_points.ndim == 2 and first_points.shape[0] == 3 and first_points.shape[1] != 3:
             first_points = first_points.T
             
        pcd.points = o3d.utility.Vector3dVector(first_points)
        pcd.colors = o3d.utility.Vector3dVector(colorize_by_z(first_points))

        vis.add_geometry(pcd)
        vis.add_geometry(world_frame)
        
        # 1. On force Open3D à calculer la vraie boîte englobante de CE nuage
        vis.poll_events()
        vis.update_renderer()
        vis.reset_view_point(True)

        # 2. On prend le contrôle de la caméra paramétriquement
        ctr = vis.get_view_control()
        
        if pcd_type == "Merged_Fork":
            # La fourchette est minuscule et centrée en 0,0,0
            ctr.set_lookat([0.0, 0.0, 0.0])
            ctr.set_front([-1.0, -1.0, 0.8]) # Angle de vue en diagonale plongeante
            ctr.set_up([0.0, 0.0, 1.0])
            ctr.set_zoom(0.7) # Zoom rapproché
        else:
            # Le robot entier est immense
            ctr.set_front([-1.0, -1.0, 0.5])
            ctr.set_up([0.0, 0.0, 1.0])
            ctr.set_zoom(0.8) # Zoom plus reculé pour voir tout le bras

        # 3. On applique la nouvelle caméra
        vis.poll_events()
        vis.update_renderer()

        # =========================================================
        # 3. BOUCLE D'ANIMATION ET CAPTURE
        # =========================================================
        for i, file_path in enumerate(files):
            points = np.load(file_path)
            
            if points.ndim == 2 and points.shape[0] == 3 and points.shape[1] != 3:
                points = points.T
            if points.ndim != 2 or points.shape[1] != 3:
                continue

            # Mise à jour de la géométrie
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colorize_by_z(points))
            vis.update_geometry(pcd)
            
            # Forcer le rendu de la frame
            vis.poll_events()
            vis.update_renderer()

            # Capturer l'image rendue par Open3D (Float buffer retourne des valeurs entre 0 et 1)
            img_float = vis.capture_screen_float_buffer(do_render=True)
            img_np = np.asarray(img_float)
            
            # Convertir Float [0, 1] RGB en Uint8 [0, 255] BGR (format OpenCV)
            img_bgr = (img_np * 255.0).astype(np.uint8)[:, :, ::-1]
            
            # Écrire la frame dans la vidéo
            out_video.write(img_bgr)

        # Nettoyage
        out_video.release()
        vis.destroy_window()
        print(f"   ✅ Vidéo {pcd_type} terminée !\n")

    print("🎉 Toutes les animations ont été générées avec succès.")

if __name__ == "__main__":
    # Change l'ID pour pointer vers la trajectoire que tu veux enregistrer
    record_trajectory_animations(traj_id=47, fps=10)