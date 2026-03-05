# Transform the .png files to .jpg files
# Segment the robot with SAM 2 and SAM 3
# Segment the food in the first image of ee_rgb
# Transform the static_npy with ee_npy (first depth file) to get the relation
# 
import cv2
import os
import rospkg
import glob
import re
import time
import json
import yaml
import torch
from PIL import Image
import numpy as np
import gc
from tqdm import tqdm
import fpsample
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
import open3d as o3d

# --- IMPORTS SAM 3 ---
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- IMPORTS SAM 2 ---
from sam2.build_sam import build_sam2_video_predictor

def get_t_ee_fork():
    """
    Retourne la matrice de transformation constante entre l'effecteur (TCP) 
    et la pointe de la fourchette (fork_tip).
    """
    T_ee_fork = np.eye(4)
    # Rotation: Pitch de -207.5 deg (180 + 27.5 compensé)
    rot_rel = R.from_euler('xyz', [0, -207.5, 0.0], degrees=True)
    T_ee_fork[:3, :3] = rot_rel.as_matrix()
    # Translation: xyz="-0.0055 -0.0 ${0.233-0.1034}"
    T_ee_fork[:3, 3] = [-0.0055, 0.0, 0.233-0.1034] 
    return T_ee_fork

def force_gpu_clear():
    """
    Force le nettoyage du Garbage Collector Python et du Cache CUDA.
    À appeler entre deux chargements de gros modèles.
    """
    print("🧹 Nettoyage du GPU en cours...")
    
    # 1. Force le Garbage Collector de Python à supprimer les objets non référencés
    gc.collect()
    
    # 2. Vide le cache de PyTorch (la mémoire réservée mais inutilisée)
    torch.cuda.empty_cache()
    
    # 3. (Optionnel) Reset du peak memory stats pour le monitoring
    torch.cuda.reset_peak_memory_stats()
    
    print("✨ GPU nettoyé.")

def get_json_paths_map():
    """
    Retourne une liste de tuples (source_path, destination_path) pour les fichiers JSON.
    Crée automatiquement les dossiers de destination dans Trajectories_preprocess_TEST.
    """
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    datas_path = os.path.join(package_path, 'datas')
    
    source_base = os.path.join(datas_path, 'Trajectories_record_TEST')
    source_folders = glob.glob(os.path.join(source_base, 'Trajectory*'))
    source_folders = sorted(source_folders, key=lambda x: int(x.split('_')[-1]))

    path_mapping = []

    for src_folder in source_folders:
        folder_name = os.path.basename(src_folder) # "Trajectory_20"
        traj_id = folder_name.split('_')[-1]       # "20"

        src_json_name = f'trajectory_{traj_id}.json' 
        src_json_path = os.path.join(src_folder, src_json_name)
        
        dest_folder = src_folder.replace('Trajectories_record_TEST', 'Trajectories_preprocess_TEST')
        os.makedirs(dest_folder, exist_ok=True)

        dest_json_path = os.path.join(dest_folder, src_json_name)
        path_mapping.append((src_json_path, dest_json_path))

    return path_mapping

def load_datas_path():
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    datas_path = os.path.join(package_path, 'datas')

    return datas_path

def load_png_datas_static():
    datas_path = load_datas_path()
    Trajectories_path = glob.glob(os.path.join(datas_path, 'Trajectories_record_TEST', 'Trajectory*'))
    Trajectories_path = sorted(Trajectories_path, key=lambda x: int(x.split('_')[-1]))
    
    static_rgb = []
    for i, trajectory_path in enumerate(Trajectories_path):
        idx = int(trajectory_path.split('_')[-1])
        new_static_rgb = glob.glob(os.path.join(trajectory_path, f'images_Trajectory_{idx}', 'static_rgb_step_*.png'))
        new_static_rgb = sorted(new_static_rgb, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
        for j, _ in enumerate(new_static_rgb):
            static_rgb.append(new_static_rgb[j])

    return static_rgb

def load_depth_datas_static():
    datas_path = load_datas_path()
    Trajectories_path = glob.glob(os.path.join(datas_path, 'Trajectories_record_TEST', 'Trajectory*'))
    Trajectories_path = sorted(Trajectories_path, key=lambda x: int(x.split('_')[-1]))
    
    # Load all the files
    static_depth = []
    for i, trajectory_path in enumerate(Trajectories_path):
        idx = int(trajectory_path.split('_')[-1])
        static_depth.append(glob.glob(os.path.join(trajectory_path, f'images_Trajectory_{idx}', 'static_depth_step_*.npy')))
    
    return static_depth

def png_to_jpg_static(png_path: str, jpg_path: str):
    """
    Convertit un PNG en JPG avec OpenCV.
    Crée automatiquement le dossier de destination s'il n'existe pas.
    """
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"Source introuvable : {png_path}")

    output_dir = os.path.dirname(jpg_path)
    os.makedirs(output_dir, exist_ok=True) 

    img = cv2.imread(png_path)
    if img is None:
        raise ValueError(f"Image corrompue ou illisible : {png_path}")
    
    cv2.imwrite(jpg_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

def segment_food_init_step():
    # --- CONFIGURATION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Utilisation du device : {device}")

    # Prompt pour SAM 3 (Anglais recommandé pour meilleure précision)
    TEXT_PROMPT = "dark object on a white surface" 
    
    # Chemins
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    datas_path = os.path.join(package_path, 'datas')
    
    # On récupère la liste des trajectoires dans RECORD (là où sont les PNG et NPY sources)
    traj_record_base = os.path.join(datas_path, 'Trajectories_record_TEST')
    traj_folders = glob.glob(os.path.join(traj_record_base, 'Trajectory*'))
    
    # Tri numérique
    traj_folders = sorted(traj_folders, key=lambda x: int(x.split('_')[-1]))

    # ====================================================
    # ÉTAPE 0 : Chargement du Modèle (Une seule fois)
    # ====================================================
    print("\n🧠 Chargement de SAM 3 (Segmentation Sémantique)...")
    sam3 = build_sam3_image_model()
    sam3_image_predictor = Sam3Processor(sam3, confidence_threshold=0.1)

    count_success = 0

    # ====================================================
    # BOUCLE SUR LES TRAJECTOIRES
    # ====================================================
    for src_folder in traj_folders:
        folder_name = os.path.basename(src_folder) # ex: "Trajectory_20"
        traj_id = folder_name.split('_')[-1]       # ex: "20"
        
        print(f"\n📂 Traitement : {folder_name} (Step 0001)")

        # 1. Construction des chemins SOURCE (Record)
        # On cherche spécifiquement le step 0001
        images_subfolder = os.path.join(src_folder, f'images_{folder_name}')
        
        rgb_path = os.path.join(images_subfolder, 'ee_rgb_step_0001.png')
        depth_path = os.path.join(images_subfolder, 'ee_depth_step_0001.npy')

        # Vérification existence fichiers
        print(f"  📁 Vérification fichiers : {rgb_path}")
        if not os.path.exists(rgb_path):
            print(f"  ⚠️ Manquant : {rgb_path}")
            continue
        if not os.path.exists(depth_path):
            print(f"  ⚠️ Manquant : {depth_path}")
            continue

        # 2. Construction chemin DESTINATION (Preprocess)
        dest_base = os.path.join(datas_path, 'Trajectories_preprocess_TEST', folder_name)
        output_dir = os.path.join(dest_base, f'filtered_pcd_{folder_name}')
        os.makedirs(output_dir, exist_ok=True)
        
        output_npy_path = os.path.join(output_dir, 'food_filtered.npy')

        # ====================================================
        # ÉTAPE 1 : Segmentation avec SAM 3
        # ====================================================
        try:
            image_pil = Image.open(rgb_path).convert("RGB")
        except Exception as e:
            print(f"  ❌ Erreur lecture image : {e}")
            continue

        # Inférence SAM 3
        inference_state = sam3_image_predictor.set_image(image_pil)
        output = sam3_image_predictor.set_text_prompt(state=inference_state, prompt=TEXT_PROMPT)
        
        # Récupération du masque
        # output["masks"] est généralement un tenseur [N, H, W]
        print(output["scores"])
        max_score_idx = torch.argmax(output["scores"])
        mask_tensor = output["masks"][max_score_idx]
        
        if mask_tensor is None:
            print(f"  ❌ '{TEXT_PROMPT}' non détecté par SAM 3.")
            continue

        # Conversion en Numpy Booléen (H, W)
        mask_np = mask_tensor.detach().cpu().numpy().squeeze()
        mask_np = mask_np > 0 # Force bool

        # ====================================================
        # ÉTAPE 2 : Projection 3D (Depth + Mask -> PCD)
        # ====================================================
        success = process_depth_to_food_cloud(depth_path, mask_np, output_npy_path)
        
        if success:
            print(f"  ✅ Sauvegardé : {output_npy_path}")
            count_success += 1

    print(f"\n🎉 Terminé ! {count_success}/{len(traj_folders)} trajectoires traitées.")


def process_depth_to_food_cloud(depth_file, mask_2d, save_path):
    """
    Génère le nuage de points du 'food' et le sauvegarde sous food_filtered.npy
    """
    try:
        depth_map = np.load(depth_file)
    except Exception as e:
        print(f"  ⚠️ Erreur chargement depth: {e}")
        return False

    # Vérification dimensions
    if len(depth_map.shape) == 3 and depth_map.shape[2] == 1:
        depth_map = np.squeeze(depth_map) # Assure 2D

    if depth_map.shape != mask_2d.shape:
        print(f"  ⚠️ Dimension mismatch (Resize masque...)")
        # Resize masque pour coller à la depth (Nearest neighbor pour garder bool)
        import cv2
        mask_2d = cv2.resize(mask_2d.astype(np.uint8), 
                             (depth_map.shape[1], depth_map.shape[0]), 
                             interpolation=cv2.INTER_NEAREST).astype(bool)

    # --- PARAMÈTRES INTRINSÈQUES ---
    # ⚠️ ATTENTION : Sont-ce les mêmes pour la caméra 'ee' (bras) et 'static'
    # Si non, remplace ces valeurs par celles de ta caméra au poignet.
    fx = 625.22  # Focal length x
    fy = 625.22  # Focal length y
    cx = 320.0    # Principal point x (image center)
    cy = 240.0    # Principal point y (image center)

    # Création grid
    h, w = depth_map.shape
    v_grid, u_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Filtre : Masque SAM (Vrai) ET Profondeur valide (>0)
    combined_mask = mask_2d & (depth_map > 0)
    
    if np.sum(combined_mask) == 0:
        print("  ⚠️ Masque vide ou pas de profondeur valide sur l'objet.")
        return False

    # Extraction 
    z_mm = depth_map[combined_mask]
    u = u_grid[combined_mask]
    v = v_grid[combined_mask]

    # Projection 3D
    z = z_mm / 1000.0
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Assemblage (N, 3)
    food_pcd = np.column_stack((x, y, z))

    # Sauvegarde
    np.save(save_path, food_pcd)
    return True

def segment_robot_and_create_pcd():
    # --- CONFIGURATION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Utilisation du device : {device}")

    package_path = rospkg.RosPack().get_path('vision_processing')
    
    # Chemins
    TRAJECTORIES_PATH = glob.glob(os.path.join(package_path, 'datas', 'Trajectories_preprocess_TEST', 'Trajectory*'))
    TRAJECTORIES_PATH_RAW = glob.glob(os.path.join(package_path, 'datas', 'Trajectories_record_TEST', 'Trajectory*'))
    
    # Tri numérique robuste
    TRAJECTORIES_PATH = sorted(TRAJECTORIES_PATH, key=lambda x: int(x.split('_')[-1]))
    TRAJECTORIES_PATH_RAW = sorted(TRAJECTORIES_PATH_RAW, key=lambda x: int(x.split('_')[-1]))

    SAM2_CHECKPOINT = os.path.expanduser("~/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt")
    SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml" 
    TEXT_PROMPT = "robot" 

    # ====================================================
    # ÉTAPE 0 : Chargement UNIQUE des modèles (Hors boucle)
    # ====================================================
    print("\n🧠 Chargement des modèles IA (SAM3 & SAM2)...")
    
    # SAM 3 Init
    sam3 = build_sam3_image_model() # Supposé charger sur device
    sam3_image_predictor = Sam3Processor(sam3)
    
    # SAM 2 Init
    sam2_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)

    # --- BOUCLE SUR LES TRAJECTOIRES ---
    for i, trajectory_path in enumerate(TRAJECTORIES_PATH):
        traj_idx = int(trajectory_path.split('_')[-1])
        print(f"\n📂 Traitement Trajectoire {traj_idx}...")

        # 1. Définition des dossiers
        # Attention : Vérifie ici si tes images sont directement dans Trajectory_X ou dans un sous-dossier images_Trajectory_X
        # Selon ton script précédent, elles sont probablement directement dans Trajectory_X si tu as utilisé preprocess
        IMAGES_FOLDER = os.path.join(trajectory_path, f'images_Trajectory_{traj_idx}')
        if not os.path.exists(IMAGES_FOLDER):
            # Fallback : peut-être qu'elles sont à la racine de la trajectoire dans preprocess ?
            IMAGES_FOLDER = trajectory_path
        
        # 2. Listing des images JPG (Nécessaire pour SAM3 et Frame 0)
        image_files = sorted(glob.glob(os.path.join(IMAGES_FOLDER, "*.jpg")))
        
        # 3. Listing des Depth NPY (Raw data)
        # On va chercher dans le dossier RECORD correspondant
        raw_traj_path = TRAJECTORIES_PATH_RAW[i]
        DEPTH_FOLDER = os.path.join(raw_traj_path, f'images_Trajectory_{traj_idx}')
        depth_files = sorted(glob.glob(os.path.join(DEPTH_FOLDER, 'static_depth_step_*.npy')), 
                             key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

        if not image_files:
            print(f"⚠️ Aucune image JPG trouvée dans {IMAGES_FOLDER}")
            continue
        if not depth_files:
            print(f"⚠️ Aucun fichier Depth trouvé dans {DEPTH_FOLDER}")
            continue

        # 4. Dossiers de sortie
        OUTPUT_ROBOT_PCD_DIR = os.path.join(trajectory_path, f'filtered_pcd_Trajectory_{traj_idx}')
        os.makedirs(OUTPUT_ROBOT_PCD_DIR, exist_ok=True)

        # ====================================================
        # ÉTAPE 1 : SAM 3 (Init sur Frame 0)
        # ====================================================
        print(f"  👁️  SAM 3 : Segmentation texte '{TEXT_PROMPT}' sur frame 0")
        frame_0_path = image_files[0]
        frame_0 = Image.open(frame_0_path).convert("RGB")
        
        inference_state_sam3 = sam3_image_predictor.set_image(frame_0)
        output_sam3 = sam3_image_predictor.set_text_prompt(state=inference_state_sam3, prompt=TEXT_PROMPT)
        
        # Récupération du masque (Check structure output SAM3 selon ta lib)
        initial_mask_tensor = output_sam3["masks"][0] 
        
        if initial_mask_tensor is None:
            print("❌ Echec SAM 3 : Robot non détecté.")
            continue

        # --- Traitement immédiat Frame 0 ---
        mask_0_np = initial_mask_tensor.detach().cpu().numpy().squeeze()
        # Conversion booléenne explicite
        mask_0_np = mask_0_np > 0 
        
        if len(depth_files) > 0:
            print(np.load(depth_files[0]).shape)
            process_depth_to_robot_cloud(depth_files[0], mask_0_np, 0, OUTPUT_ROBOT_PCD_DIR)
        
        # ====================================================
        # ÉTAPE 2 : SAM 2 (Tracking)
        # ====================================================
        print(f"  ⚡ SAM 2 : Propagation...")
        
        # Note: SAM2 s'attend souvent à ce que les images soient des fichiers JPG sur disque
        inference_state = sam2_predictor.init_state(video_path=IMAGES_FOLDER)
        
        # Reset de l'état mémoire de SAM2 pour cette nouvelle vidéo
        sam2_predictor.reset_state(inference_state)

        # Injection du masque initial SAM 3 dans SAM 2
        mask_input = torch.from_numpy(mask_0_np).bool().to(device)
        # Ajouter une dimension batch/channel si nécessaire pour SAM2 (souvent [1, H, W])
        # if mask_input.ndim == 2:
            # mask_input = mask_input.unsqueeze(0) 

        _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_mask(
            inference_state=inference_state, 
            frame_idx=0, 
            obj_id=1, 
            mask=mask_input
        )

        # --- Propagation ---
        count = 0
        # inference_mode désactive le calcul de gradient (économie VRAM)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            
            for frame_idx, obj_ids, video_res_masks in sam2_predictor.propagate_in_video(inference_state):
                
                # frame_idx sort parfois dans le désordre ou commence à 0, SAM2 gère ça.
                # video_res_masks est le masque prédit pour la frame courante
                
                # 1. Extraction Masque (Logits -> Bool)
                # video_res_masks est souvent [K, 1, H, W] ou [K, H, W]
                raw_mask = (video_res_masks[0] > 0.0).cpu().numpy() 
                mask_np = raw_mask.squeeze() # Devient (H, W)
                
                # 2. Correspondance avec le fichier Depth
                # Attention : SAM2 peut processer tout le dossier. On s'assure de ne pas dépasser.
                if frame_idx < len(depth_files):
                    depth_path = depth_files[frame_idx]
                    
                    process_depth_to_robot_cloud(
                        depth_file=depth_path, 
                        mask_2d=mask_np, 
                        frame_idx=frame_idx, 
                        output_dir=OUTPUT_ROBOT_PCD_DIR
                    )
                    count += 1
                
                if count % 20 == 0:
                    print(f"    -> Processed frame {frame_idx}")

        print(f"✅ Trajectoire {traj_idx} terminée : {count} nuages générés.")

    print("🎉 Traitement complet terminé.")


def process_depth_to_robot_cloud(depth_file, mask_2d, frame_idx, output_dir):
    """
    Sauvegarde le nuage de points correspondant aux pixels True du masque.
    """
    try:
        depth_map = np.load(depth_file)
    except Exception as e:
        print(f"⚠️ Erreur read {depth_file}: {e}")
        return

    # Resize de sécurité : Parfois SAM resize légèrement ou padding
    if len(depth_map.shape) == 3 and depth_map.shape[2] == 1:
        depth_map = np.squeeze(depth_map) # Assure 2D
    
    if depth_map.shape != mask_2d.shape:
        # On resize le masque pour coller à la depth (plus sûr que l'inverse)
        mask_2d = cv2.resize(mask_2d.astype(np.uint8), (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # Paramètres Intrinsèques (Tes valeurs)
    fx, fy = 625.22, 625.22
    cx, cy = 320.0, 240.0

    # Création grid
    h, w = depth_map.shape
    v_grid, u_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # --- FILTRE LOGIQUE ---
    # mask_2d = True là où est le robot.
    # On garde les pixels où (C'est le Robot) ET (La profonfeur est valide)
    combined_mask = mask_2d & (depth_map > 0)

    # NOTE : Si tu voulais SUPPRIMER le robot pour voir juste la nourriture :
    # combined_mask = (~mask_2d) & (depth_map > 0)
    
    if np.sum(combined_mask) == 0:
        return

    # Extraction vectorisée
    z_mm = depth_map[combined_mask]
    u = u_grid[combined_mask]
    v = v_grid[combined_mask]

    # Projection 3D
    z = z_mm / 1000.0
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    robot_pcd = np.column_stack((x, y, z))

    # Sauvegarde
    output_filename = f"robot_cloud_{frame_idx:04d}.npy"
    save_path = os.path.join(output_dir, output_filename)
    np.save(save_path, robot_pcd)


def load_matrix_from_json(json_data, key, step_index=0):
    """
    Récupère une matrice 4x4 de manière robuste en analysant la shape Numpy.
    Gère : Matrice unique (4,4), Liste de matrices (N,4,4), Matrice plate (16,), Liste de plates (N,16).
    """
    if key not in json_data:
        raise ValueError(f"Clé {key} introuvable dans le JSON.")
    
    data = json_data[key]
    
    # 1. On convertit directement en array pour voir la "vraie" forme
    arr = np.array(data)
    
    matrix = None

    # CAS A : C'est une matrice unique 4x4 (statique)
    # C'est ton cas pour T_static_s
    if arr.shape == (4, 4):
        matrix = arr

    # CAS B : C'est une liste de matrices 4x4 (dynamique) -> Forme (N, 4, 4)
    # C'est peut-être le cas pour T_ee_s si elle bouge
    elif arr.ndim == 3 and arr.shape[1:] == (4, 4):
        # On prend le step demandé (en s'assurant de ne pas dépasser)
        idx = min(step_index, arr.shape[0] - 1)
        matrix = arr[idx]

    # CAS C : Matrice unique aplatie (16 valeurs)
    elif arr.shape == (16,):
        matrix = arr.reshape(4, 4)

    # CAS D : Liste de matrices aplaties (N, 16)
    elif arr.ndim == 2 and arr.shape[1] == 16:
        idx = min(step_index, arr.shape[0] - 1)
        matrix = arr[idx].reshape(4, 4)

    else:
        raise ValueError(f"Forme de matrice inconnue pour {key} : {arr.shape}")

    return matrix

def apply_transform(points, transformation_matrix):
    """
    Applique une transformation rigide (4x4) à un nuage de points (N, 3).
    """
    # Si vide
    if points.size == 0:
        return points

    # --- SÉCURITÉ DIMENSIONS ---
    # On s'assure que points est bien (N, 3). 
    # Si c'est (3,), ça devient (1, 3).
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    # On s'assure que c'est bien des colonnes de 3 (x,y,z)
    if points.shape[1] != 3:
         print(f"⚠️ Attention: Nuage de points forme étrange {points.shape}, tentative de correction...")
         # Cas rare ou le tableau serait (3, N) -> on transpose
         if points.shape[0] == 3:
             points = points.T

    # Passage en homogène (N, 4)
    # np.hstack colle une colonne de 1 à droite
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack((points, ones))

    # Multiplication : (T @ P.T).T  <-- Plus robuste que T @ P
    # P_hom.T est (4, N)
    # T est (4, 4)
    # Result est (4, N)
    transformed_hom_T = transformation_matrix @ points_hom.T
    
    # On re-transpose pour avoir (N, 4)
    transformed_hom = transformed_hom_T.T

    # Retour (x, y, z)
    return transformed_hom[:, :3]

def pose_to_matrix(pos, quat):
    """ Convertit position + quaternion [x,y,z,w] en matrice 4x4 """
    mat = np.eye(4)
    quate = [quat[1], quat[2], quat[3], quat[0]] # Convert [x,y,z,w] -> [w,x,y,z] pour scipy
    mat[:3, :3] = R.from_quat(quate).as_matrix()
    mat[:3, 3] = pos
    return mat

from scipy.ndimage import gaussian_filter1d

# ============================================================
# SHARED HELPERS (put these before the 3 merge functions)
# ============================================================

def detect_grasp_index(json_data, food_centroid_world, T_ee_fork, warn_dist=0.05):
    """
    Detects the grasp index by finding the FIRST local minimum of the fork tip's Z position.
    
    This corresponds to the exact moment the fork finishes its downward plunge 
    into the food and is about to lift it up.
    
    Args:
        json_data:           The trajectory JSON (with 'states')
        food_centroid_world: (3,) centroid of food in world frame (for sanity check)
        T_ee_fork:           (4,4) constant transform from TCP/EE to fork tip
        warn_dist:           Minimum distance threshold for sanity check (meters)
    
    Returns:
        grasp_idx: int, the timestep index where grasp occurs
    """
    states = json_data['states']
    fork_positions = []

    for state in states:
        # 1. Calcul de la position du bout de la fourchette dans le monde
        # Note: state['end_effector...'] contient maintenant ton panda_TCP grâce à ta modification
        T_world_ee = pose_to_matrix(
            state['end_effector_position'],
            state['end_effector_orientation']
        )
        # On applique l'offset de la fourchette
        T_world_fork = T_world_ee @ T_ee_fork
        fork_tip_world = T_world_fork[:3, 3]
        fork_positions.append(fork_tip_world)

    # Conversion en array NumPy (N, 3)
    fork_tip_np = np.array(fork_positions)
    
    # 2. On isole l'axe Z
    z_pos = fork_tip_np[:, 2]
    
    # 3. Logique du premier minimum local
    local_minima = np.where((z_pos[1:-1] < z_pos[:-2]) & (z_pos[1:-1] < z_pos[2:]))[0] + 1
    
    if len(local_minima) > 0:
        grasp_idx = int(local_minima[0])  # On prend le tout premier creux
    else:
        # Sécurité : si aucun minimum local strict n'est détecté 
        # (ex: le robot descend et reste figé), on prend le point le plus bas
        grasp_idx = int(np.argmin(z_pos))
        print("  ⚠️ Aucun minimum local strict trouvé, utilisation du minimum absolu.")
    
    # 4. Sanity check : Vérifier la distance avec la nourriture
    min_z_fork_pos = fork_tip_np[grasp_idx]
    dist_at_grasp = np.linalg.norm(min_z_fork_pos - food_centroid_world)

    if dist_at_grasp > warn_dist:
        print(f"  ⚠️ Grasp suspiciously far from food: {dist_at_grasp:.4f}m (threshold: {warn_dist}m) at step {grasp_idx} (Z={z_pos[grasp_idx]:.4f}m)")
    else:
        print(f"  ✅ Grasp detected at step {grasp_idx} (Local min Z={z_pos[grasp_idx]:.4f}m), distance to food: {dist_at_grasp:.4f}m")

    return grasp_idx

def compute_food_at_step(food_points_world_init, step_idx, grasp_idx, json_data, T_ee_fork):
    """
    Computes the food point cloud position at a given step.
    
    Before grasp: food stays at its initial world position (static on the plate).
    After grasp:  food moves rigidly with the fork frame.
    
    The motion is computed in the fork frame, not the EE frame, because the fork 
    is what physically holds the food. (Mathematically equivalent since T_ee_fork is 
    constant, but semantically clearer and correct if T_ee_fork ever changes.)
    
    The transform logic:
        At grasp:   T_world_fork_grasp = T_world_ee_grasp @ T_ee_fork
        At step i:  T_world_fork_i     = T_world_ee_i     @ T_ee_fork
        
        Relative fork motion since grasp:
            T_motion = T_world_fork_i @ inv(T_world_fork_grasp)
        
        Food at step i:
            food_i = T_motion @ food_grasp
    
    Args:
        food_points_world_init: (N,3) food points in world frame (initial position)
        step_idx:               current timestep
        grasp_idx:              timestep when grasp occurs
        json_data:              trajectory JSON
        T_ee_fork:              (4,4) constant EE -> fork tip transform
    
    Returns:
        (N,3) food points in world frame at step_idx
    """
    # Before grasp: food is static
    if step_idx < grasp_idx or food_points_world_init.size == 0:
        return food_points_world_init.copy()
    
    # Compute fork pose at grasp
    grasp_state = json_data['states'][grasp_idx]
    T_world_ee_grasp = pose_to_matrix(
        grasp_state['end_effector_position'],
        grasp_state['end_effector_orientation']
    )
    T_world_fork_grasp = T_world_ee_grasp @ T_ee_fork
    
    # Compute fork pose at current step
    curr_state = json_data['states'][step_idx]
    T_world_ee_curr = pose_to_matrix(
        curr_state['end_effector_position'],
        curr_state['end_effector_orientation']
    )
    T_world_fork_curr = T_world_ee_curr @ T_ee_fork
    
    # Relative motion of fork since grasp
    T_motion = T_world_fork_curr @ np.linalg.inv(T_world_fork_grasp)
    
    # Apply to food (food moves with fork)
    return apply_transform(food_points_world_init, T_motion)


def load_food_world_init(food_pcd_path, json_data):
    """
    Loads the food point cloud from camera frame and transforms it to world frame
    using the first step's T_ee_s transform.
    
    Returns:
        food_points_world: (N,3) or (0,3) if no food found
        food_centroid:     (3,) centroid in world frame, or None
    """
    food_points_world = np.empty((0, 3))
    food_centroid = None
    
    if os.path.exists(food_pcd_path):
        try:
            food_points_cam = np.load(food_pcd_path)
            if food_points_cam.size > 0:
                T_ee_s_step0 = load_matrix_from_json(json_data, "T_ee_s", step_index=0)
                food_points_world = apply_transform(food_points_cam, T_ee_s_step0)
                food_centroid = food_points_world.mean(axis=0)
        except Exception as e:
            print(f"  ⚠️ Error loading food: {e}")
    
    return food_points_world, food_centroid


def downsample_fps(points, num_target=1024, h=5):
    """
    Farthest Point Sampling to reduce a point cloud to a fixed size.
    """
    if points.shape[0] > num_target:
        indices = fpsample.bucket_fps_kdline_sampling(
            points.astype(np.float32), num_target, h=h
        )
        return points[indices]
    return points


# ============================================================
# STEP 5 : merge_pcd_trajectory() — Segmented Robot + Food
# ============================================================

def merge_pcd_trajectory():
    """
    Merges segmented robot point cloud (from SAM) with food point cloud.
    Output is in WORLD frame (same convention as DP3).
    """
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    base_path = os.path.join(package_path, 'datas', 'Trajectories_preprocess_TEST')
    
    traj_folders = sorted(
        glob.glob(os.path.join(base_path, 'Trajectory*')),
        key=lambda x: int(x.split('_')[-1])
    )

    T_ee_fork = get_t_ee_fork()

    print(f"🔄 Step 5: Merging Segmented Robot + Food for {len(traj_folders)} trajectories...")

    for traj_folder in traj_folders:
        folder_name = os.path.basename(traj_folder)
        traj_id = folder_name.split('_')[-1]
        
        # --- Paths ---
        json_path = os.path.join(traj_folder, f'trajectory_{traj_id}.json')
        if not os.path.exists(json_path):
            json_path = os.path.join(traj_folder, f'trajectory{traj_id}.json')
        
        food_pcd_path = os.path.join(traj_folder, f'filtered_pcd_{folder_name}', 'food_filtered.npy')
        robot_pcd_folder = os.path.join(traj_folder, f'filtered_pcd_{folder_name}')
        output_dir = os.path.join(traj_folder, f'Merged_pcd_{folder_name}')
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n📂 Processing: {folder_name}")

        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        # --- Load food + detect grasp ---
        food_points_world_init, food_centroid = load_food_world_init(food_pcd_path, json_data)
        
        if food_centroid is not None:
            grasp_idx = detect_grasp_index(json_data, food_centroid, T_ee_fork)
        else:
            grasp_idx = len(json_data['states'])  # no food => never grasp
            print("  ⚠️ No food found, skipping grasp detection")

        # --- Robot loop ---
        robot_files = sorted(glob.glob(os.path.join(robot_pcd_folder, 'robot_cloud_*.npy')))
        if not robot_files:
            continue

        for robot_file in tqdm(robot_files, desc=f"  ↳ Merging {folder_name}", unit="step"):
            filename = os.path.basename(robot_file)
            step_str = os.path.splitext(filename)[0].split('_')[-1]
            try:
                step_idx = int(step_str) - 1
            except ValueError:
                continue

            try:
                robot_points_cam = np.load(robot_file)
                if robot_points_cam.size == 0:
                    continue

                # =======================================================
                # FILTRAGE DES OUTLIERS (Statistical Outlier Removal)
                # =======================================================
                pcd_robot = o3d.geometry.PointCloud()
                pcd_robot.points = o3d.utility.Vector3dVector(robot_points_cam)
                
                # nb_neighbors : Nombre de voisins à analyser (20 à 50 est un bon standard)
                # std_ratio : Plus ce chiffre est bas, plus le filtre est agressif (ex: 1.0 ou 2.0)
                cl, ind = pcd_robot.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)
                
                # On écrase l'ancien array avec les points filtrés (inliers)
                robot_points_cam = np.asarray(cl.points)
                
                # Sécurité au cas où le filtre efface tout
                if robot_points_cam.shape[0] == 0:
                    continue
                # =======================================================

                # Robot: camera frame -> world frame
                sapien2opencv = np.array([
                        [ 0.0,  0.0,  1.0,  0.0],
                        [-1.0,  0.0,  0.0,  0.0],
                        [ 0.0, -1.0,  0.0,  0.0],
                        [ 0.0,  0.0,  0.0,  1.0]
                    ])
                T_static_s = load_matrix_from_json(json_data, "T_static_s", step_index=step_idx)
                robot_points_world = apply_transform(robot_points_cam, T_static_s)

                # Food: static or moving with fork
                current_food = compute_food_at_step(
                    food_points_world_init, step_idx, grasp_idx, json_data, T_ee_fork
                )

                # Merge
                if current_food.shape[0] > 0:
                    merged = np.vstack((robot_points_world, current_food))
                else:
                    merged = robot_points_world

                # FPS downsample
                merged = downsample_fps(merged, num_target=1024)

                # Save (world frame — same as DP3)
                save_path = os.path.join(output_dir, f"Merged_{step_str}.npy")
                np.save(save_path, merged)

            except Exception as e:
                print(f"  ❌ Error on {filename}: {e}")
                continue

    print("\n✅ Step 5 done.")


# ============================================================
# STEP 6 : merge_pcd_trajectory_urdf() — URDF Robot + Food
# ============================================================

def merge_pcd_trajectory_urdf():
    """
    Merges URDF robot point cloud (Robot_point_cloud_*.npy from FK) 
    with food point cloud. Output is in WORLD frame (same convention as DP3).
    """
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    base_path_preprocess = os.path.join(package_path, 'datas', 'Trajectories_preprocess_TEST')
    base_path_record = os.path.join(package_path, 'datas', 'Trajectories_record_TEST')
    
    traj_folders_preprocess = sorted(
        glob.glob(os.path.join(base_path_preprocess, 'Trajectory*')),
        key=lambda x: int(x.split('_')[-1])
    )

    T_ee_fork = get_t_ee_fork()

    print(f"🔄 Step 6: Merging URDF Robot + Food for {len(traj_folders_preprocess)} trajectories...")

    for traj_folder_proc in traj_folders_preprocess:
        folder_name = os.path.basename(traj_folder_proc)
        traj_id = folder_name.split('_')[-1]
        
        # --- Paths ---
        json_path = os.path.join(traj_folder_proc, f'trajectory_{traj_id}.json')
        if not os.path.exists(json_path):
            json_path = os.path.join(traj_folder_proc, f'trajectory{traj_id}.json')
        
        food_pcd_path = os.path.join(traj_folder_proc, f'filtered_pcd_{folder_name}', 'food_filtered.npy')
        
        traj_folder_rec = os.path.join(base_path_record, folder_name)
        robot_urdf_folder = os.path.join(traj_folder_rec, f'images_{folder_name}')
        
        output_dir = os.path.join(traj_folder_proc, f'Merged_urdf_fork_{folder_name}')
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n📂 Processing URDF: {folder_name}")

        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        # --- Load food + detect grasp ---
        food_points_world_init, food_centroid = load_food_world_init(food_pcd_path, json_data)
        
        if food_centroid is not None:
            grasp_idx = detect_grasp_index(json_data, food_centroid, T_ee_fork)
        else:
            grasp_idx = len(json_data['states'])
            print("  ⚠️ No food found, skipping grasp detection")

        # --- URDF Robot loop ---
        urdf_files = sorted(glob.glob(os.path.join(robot_urdf_folder, 'Robot_point_cloud_*.npy')))
        if not urdf_files:
            continue

        for urdf_file in tqdm(urdf_files, desc=f"  ↳ Merging URDF {folder_name}", unit="step"):
            filename = os.path.basename(urdf_file)
            try:
                step_str = filename.split('.')[0].split('_')[-1]
                step_idx = int(step_str) - 1
            except ValueError:
                continue

            try:
                robot_points_world = np.load(urdf_file)
                if robot_points_world.ndim == 1:
                    robot_points_world = robot_points_world.reshape(1, -1)
                if robot_points_world.shape[1] != 3 and robot_points_world.shape[0] == 3:
                    robot_points_world = robot_points_world.T

                # Food: static or moving with fork
                current_food = compute_food_at_step(
                    food_points_world_init, step_idx, grasp_idx, json_data, T_ee_fork
                )

                # Merge
                if current_food.shape[0] > 0:
                    merged = np.vstack((robot_points_world, current_food))
                else:
                    merged = robot_points_world

                # FPS downsample
                merged = downsample_fps(merged, num_target=1024)

                # Save (world frame — same as DP3)
                save_path = os.path.join(output_dir, f"Merged_urdf_fork_{step_str}.npy")
                np.save(save_path, merged)

            except Exception as e:
                print(f"  ❌ Error {filename}: {e}")

    print("\n✅ Step 6 done.")


# ============================================================
# STEP 7 : merge_pcd_trajectory_fork() — Fork Only + Food
# ============================================================

def merge_pcd_trajectory_fork():
    """
    Merges fork-only point cloud (Fork_point_cloud_*.npy) with food.
    Output is TRANSLATION-CENTERED at fork tip (world orientation preserved).
    
    Why translation-only (not full SE(3)):
      - With only tool + target in the PCD (no robot links), full SE(3) 
        makes the observation FROZEN after grasp (everything is constant 
        in the tool's own frame).
      - Translation-only keeps world orientation visible, so the model 
        can distinguish "fork pointing down into plate" from "fork 
        lifting up" from "fork moving to user".
      - This makes the policy position-invariant but gravity-aware.
    """
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    base_path_preprocess = os.path.join(package_path, 'datas', 'Trajectories_preprocess_TEST')
    base_path_record = os.path.join(package_path, 'datas', 'Trajectories_record_TEST')
    
    traj_folders_preprocess = sorted(
        glob.glob(os.path.join(base_path_preprocess, 'Trajectory*')),
        key=lambda x: int(x.split('_')[-1])
    )

    T_ee_fork = get_t_ee_fork()

    print(f"🔄 Step 7: Merging Fork + Food for {len(traj_folders_preprocess)} trajectories...")

    for traj_folder_proc in traj_folders_preprocess:
        folder_name = os.path.basename(traj_folder_proc)
        traj_id = folder_name.split('_')[-1]
        
        # --- Paths ---
        json_path = os.path.join(traj_folder_proc, f'trajectory_{traj_id}.json')
        if not os.path.exists(json_path):
            json_path = os.path.join(traj_folder_proc, f'trajectory{traj_id}.json')
        
        food_pcd_path = os.path.join(traj_folder_proc, f'filtered_pcd_{folder_name}', 'food_filtered.npy')
        
        traj_folder_rec = os.path.join(base_path_record, folder_name)
        robot_urdf_folder = os.path.join(traj_folder_rec, f'images_{folder_name}')
        
        output_dir = os.path.join(traj_folder_proc, f'Merged_Fork_{folder_name}')
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n📂 Processing Fork: {folder_name}")

        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        # --- Load food + detect grasp ---
        food_points_world_init, food_centroid = load_food_world_init(food_pcd_path, json_data)
        
        if food_centroid is not None:
            grasp_idx = detect_grasp_index(json_data, food_centroid, T_ee_fork)
        else:
            grasp_idx = len(json_data['states'])
            print("  ⚠️ No food found, skipping grasp detection")

        # --- Fork loop ---
        fork_files = sorted(glob.glob(os.path.join(robot_urdf_folder, 'Fork_point_cloud_*.npy')))
        if not fork_files:
            continue

        for fork_file in tqdm(fork_files, desc=f"  ↳ Merging Fork {folder_name}", unit="step"):
            filename = os.path.basename(fork_file)
            try:
                step_str = filename.split('.')[0].split('_')[-1]
                step_idx = int(step_str) - 1
            except ValueError:
                continue

            try:
                fork_points_world = np.load(fork_file)
                if fork_points_world.ndim == 1:
                    fork_points_world = fork_points_world.reshape(1, -1)
                if fork_points_world.shape[1] != 3 and fork_points_world.shape[0] == 3:
                    fork_points_world = fork_points_world.T

                # Food: static or moving with fork
                current_food = compute_food_at_step(
                    food_points_world_init, step_idx, grasp_idx, json_data, T_ee_fork
                )

                # Merge in world frame first
                if current_food.shape[0] > 0:
                    merged_world = np.vstack((fork_points_world, current_food))
                else:
                    merged_world = fork_points_world

                # FPS downsample (in world frame, before centering)
                merged_world = downsample_fps(merged_world, num_target=1024)

                # Translation-only centering: subtract fork tip position
                # World orientation is PRESERVED (no rotation applied)
                curr_state = json_data['states'][step_idx]
                T_world_ee = pose_to_matrix(
                    curr_state['end_effector_position'],
                    curr_state['end_effector_orientation']
                )
                T_world_fork = T_world_ee @ T_ee_fork
                fork_tip_pos = T_world_fork[:3, 3]
                merged_centered = merged_world - fork_tip_pos

                # Save (translation-centered, world orientation)
                save_path = os.path.join(output_dir, f"Merged_Fork_{step_str}.npy")
                np.save(save_path, merged_centered)

            except Exception as e:
                print(f"  ❌ Error {filename}: {e}")

    print("\n✅ Step 7 done.")

def update_json_merged_pcd():
    """
    Étape 8 : Met à jour le JSON avec :
    1. Les chemins vers les nuages de points (Steps 5, 6, 7)
    2. La pose mondiale (position/orientation) de la fourchette pour chaque état.
    3. La matrice constante T_ee_fork en fin de fichier.
    """
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    base_path = os.path.join(package_path, 'datas', 'Trajectories_preprocess_TEST')
    
    traj_folders = sorted(glob.glob(os.path.join(base_path, 'Trajectory*')), 
                          key=lambda x: int(x.split('_')[-1]))

    T_ee_fork = get_t_ee_fork()

    print(f"🔄 Mise à jour des JSON pour {len(traj_folders)} trajectoires...")

    for traj_folder in traj_folders:
        folder_name = os.path.basename(traj_folder)
        traj_id = folder_name.split('_')[-1]
        
        json_path = os.path.join(traj_folder, f'trajectory_{traj_id}.json')
        if not os.path.exists(json_path):
            json_path = os.path.join(traj_folder, f'trajectory{traj_id}.json')
            if not os.path.exists(json_path): continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        if 'states' in data and isinstance(data['states'], list):
            for state in data['states']:
                step_num = int(state.get('step_number'))
                step_str = f"{step_num:04d}"
                
                # 1. Mise à jour des liens vers les nuages de points
                state['Merged_point_cloud'] = f"Merged_{step_str}.npy"
                state['Merged_urdf_fork_point_cloud'] = f"Merged_urdf_fork_{step_str}.npy"
                state['Merged_Fork_point_cloud'] = f"Merged_Fork_{step_str}.npy"

                # 2. Calcul de la pose mondiale du Fork Tip
                # On récupère la pose EE du JSON
                ee_pos = state['end_effector_position']
                ee_quat = state['end_effector_orientation']
                
                T_world_ee = pose_to_matrix(ee_pos, ee_quat)
                T_world_fork = T_world_ee @ T_ee_fork

                # Extraction position et quaternion [x, y, z, w]
                state['fork_tip_position'] = T_world_fork[:3, 3].tolist()
                state['fork_tip_orientation'] = R.from_matrix(T_world_fork[:3, :3]).as_quat().tolist()

        # 3. Ajout de la matrice constante T_ee_fork à la racine du JSON
        data['T_ee_fork'] = T_ee_fork.tolist()

        try:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"  ❌ Erreur écriture {json_path} : {e}")

    print("🎉 Mise à jour terminée.")

def main():

    print("="*80)
    print("Step 1 : Convertir les PNG static en JPG pour faire la segmentation et le tracking avec SAM 2")
    print("="*80)
    start_time = time.time()
    png_datas = load_png_datas_static()
    print(png_datas)
    for png_path in png_datas:
        directory_path = os.path.dirname(png_path)
        filename = os.path.basename(png_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        directory_path = directory_path.replace("Trajectories_record_TEST", "Trajectories_preprocess_TEST")
        
        try:
            # On extrait le numéro "11" depuis "static_rgb_step_11"
            step_number = int(name_no_ext.split('_')[-1])
            
            # On définit le nom CIBLE qu'on veut obtenir : "0011.jpg"
            target_name = f"{step_number:04d}.jpg"
            target_path = os.path.join(directory_path, target_name)
            
            # LE TEST ULTIME : Si le résultat existe déjà, on passe !
            if os.path.exists(target_path):
                print(f"Le fichier {target_name} existe déjà. Skip.")
                continue

            # Sinon, on lance la conversion
            print(f"Création de {target_name} à partir de {filename}...")
            jpg_path = os.path.join(directory_path, target_name)
            print(jpg_path)
            png_to_jpg_static(png_path, jpg_path)
        except Exception as e:
            print(f"Erreur sur {png_path}: {e}")
    end_time = time.time()
    print(f"Temps total pour transformer les PNG static en JPG : {end_time - start_time}")
    
    
    print("="*80)
    print("Step 2 : Ajouter les fichiers json dans les nouveaux dossiers")
    print("="*80)
    start_time = time.time()
    mappings = get_json_paths_map()
    for src, dest in mappings:
        print(f"Source: {os.path.basename(src)} -> Dest: {dest}")
        with open(src, 'r') as f:
            data = json.load(f)
        with open(dest, 'w') as f:
            json.dump(data, f)
        
    end_time = time.time()
    print(f"Temps total pour ajouter les fichiers json dans les nouveaux dossiers : {end_time - start_time}")
    
    print("="*80)
    print("Step 3 : Segmenter les images avec SAM 2 et creer les nuages de points du robot segmenté et de la nourriture segmentée")
    print("="*80)
    start_time = time.time()
    
    try:
        # Lancement de la fonction principale
        segment_robot_and_create_pcd()
        
    except KeyboardInterrupt:
        print("\n🛑 Interruption par l'utilisateur.")
        
    end_time = time.time()
    print(f"Temps total pour segmenter les nuages de points : {end_time - start_time}")
    

    print("="*80)
    print("Step 4 : Segmenter la nourriture dans l'assiette et créer le nuage de point correspondant.")
    print("="*80)
    force_gpu_clear()
    start_time = time.time()
    segment_food_init_step()
    end_time = time.time()
    print(f"Temps total pour segmenter la nourriture dans l'assiette et créer le nuage de point correspondant : {end_time - start_time}")
    
    print("="*80)
    print("Step 5 : Créer les nuages de points merge du robot segmenter et de la nourriture segmentée.")
    print("="*80)
    start_time = time.time()
    merge_pcd_trajectory()
    end_time = time.time()
    print(f"Temps total pour créer les nuages de points merge : {end_time - start_time}")
    
    force_gpu_clear()
    print("="*80)
    print("Step 6 : Créer les nuages de points merge du robot urdf avec fourchette et de la nourriture segmentée.")
    print("="*80)
    start_time = time.time()
    merge_pcd_trajectory_urdf()
    end_time = time.time()
    print(f"Temps total pour créer les nuages de points urdf merge : {end_time - start_time}")
    
    force_gpu_clear()

    print("="*80)
    print("Step 7 : Créer les nuages de points merge de la fourchette et de la nourriture segmentée.")
    print("="*80)
    start_time = time.time()
    merge_pcd_trajectory_fork()
    end_time = time.time()
    print(f"Temps total pour créer les nuages de points fourchette merge : {end_time - start_time}")

    force_gpu_clear()
    
    print("\n" + "="*80)
    print("Step 8 : Mise à jour des fichiers JSON (Link Merged PCD)")
    print("="*80)
    start_time = time.time()
    update_json_merged_pcd()
    end_time = time.time()
    print(f"Temps total pour mettre à jour les fichiers JSON : {end_time - start_time}")
    
    print("\n✅ PIPELINE COMPLET TERMINÉ AVEC SUCCÈS.")

if __name__ == "__main__":
    main()