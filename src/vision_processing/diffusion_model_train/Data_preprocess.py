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
    T_ee_fork[:3, 3] = [-0.0055, 0.0, 0.233 - 0.1034] 
    return T_ee_fork

def pose_to_matrix(pos, quat):
    """ Convertit position + quaternion [x,y,z,w] en matrice 4x4 """
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat(quat).as_matrix()
    mat[:3, 3] = pos
    return mat

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
    Crée automatiquement les dossiers de destination dans Trajectories_preprocess.
    """
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    datas_path = os.path.join(package_path, 'datas')
    
    source_base = os.path.join(datas_path, 'Trajectories_record')
    source_folders = glob.glob(os.path.join(source_base, 'Trajectory*'))
    source_folders = sorted(source_folders, key=lambda x: int(x.split('_')[-1]))

    path_mapping = []

    for src_folder in source_folders:
        folder_name = os.path.basename(src_folder) # "Trajectory_20"
        traj_id = folder_name.split('_')[-1]       # "20"

        src_json_name = f'trajectory_{traj_id}.json' 
        src_json_path = os.path.join(src_folder, src_json_name)
        
        dest_folder = src_folder.replace('Trajectories_record', 'Trajectories_preprocess')
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
    Trajectories_path = glob.glob(os.path.join(datas_path, 'Trajectories_record', 'Trajectory*'))
    Trajectories_path = sorted(Trajectories_path, key=lambda x: int(x.split('_')[-1]))
    
    static_rgb = []
    for i, trajectory_path in enumerate(Trajectories_path):
        new_static_rgb = glob.glob(os.path.join(trajectory_path, f'images_Trajectory_{i+1}', 'static_rgb_step_*.png'))
        new_static_rgb = sorted(new_static_rgb, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
        for j, _ in enumerate(new_static_rgb):
            static_rgb.append(new_static_rgb[j])

    return static_rgb

def load_depth_datas_static():
    datas_path = load_datas_path()
    Trajectories_path = glob.glob(os.path.join(datas_path, 'Trajectories_record', 'Trajectory*'))
    Trajectories_path = sorted(Trajectories_path, key=lambda x: int(x.split('_')[-1]))
    
    # Load all the files
    static_depth = []
    for i, trajectory_path in enumerate(Trajectories_path):
        static_depth.append(glob.glob(os.path.join(trajectory_path, f'images_Trajectory_{i+1}', 'static_depth_step_*.npy')))
    
    return ee_depth, static_depth

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
    TEXT_PROMPT = "dark object in trapezoidal shape with pink side" 
    
    # Chemins
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    datas_path = os.path.join(package_path, 'datas')
    
    # On récupère la liste des trajectoires dans RECORD (là où sont les PNG et NPY sources)
    traj_record_base = os.path.join(datas_path, 'Trajectories_record')
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
        dest_base = os.path.join(datas_path, 'Trajectories_preprocess', folder_name)
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
    if depth_map.shape != mask_2d.shape:
        print(f"  ⚠️ Dimension mismatch (Resize masque...)")
        # Resize masque pour coller à la depth (Nearest neighbor pour garder bool)
        import cv2
        mask_2d = cv2.resize(mask_2d.astype(np.uint8), 
                             (depth_map.shape[1], depth_map.shape[0]), 
                             interpolation=cv2.INTER_NEAREST).astype(bool)

    # --- PARAMÈTRES INTRINSÈQUES ---
    # ⚠️ ATTENTION : Sont-ce les mêmes pour la caméra 'ee' (bras) et 'static' ?
    # Si non, remplace ces valeurs par celles de ta caméra au poignet.
    fx = 607.18261719  # Focal length x
    fy = 606.91986084  # Focal length y
    cx = 320.85250854    # Principal point x (image center)
    cy = 243.40284729    # Principal point y (image center)

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
    TRAJECTORIES_PATH = glob.glob(os.path.join(package_path, 'datas', 'Trajectories_preprocess', 'Trajectory*'))
    TRAJECTORIES_PATH_RAW = glob.glob(os.path.join(package_path, 'datas', 'Trajectories_record', 'Trajectory*'))
    
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
        traj_idx = i + 1
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
    if depth_map.shape != mask_2d.shape:
        # On resize le masque pour coller à la depth (plus sûr que l'inverse)
        mask_2d = cv2.resize(mask_2d.astype(np.uint8), (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

    # Paramètres Intrinsèques (Tes valeurs)
    fx, fy = 616.1005249, 615.82617188
    cx, cy = 318.38803101, 249.23504639

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
    mat[:3, :3] = R.from_quat(quat).as_matrix()
    mat[:3, 3] = pos
    return mat

def merge_pcd_trajectory():
    # --- CONFIGURATION ---
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    base_path = os.path.join(package_path, 'datas', 'Trajectories_preprocess')
    
    traj_folders = sorted(glob.glob(os.path.join(base_path, 'Trajectory*')), 
                          key=lambda x: int(x.split('_')[-1]))

    print(f"🔄 Début de la fusion (Segmented Robot) pour {len(traj_folders)} trajectoires...")

    for traj_folder in traj_folders:
        folder_name = os.path.basename(traj_folder)
        traj_id = folder_name.split('_')[-1]
        
        json_path = os.path.join(traj_folder, f'trajectory_{traj_id}.json')
        if not os.path.exists(json_path): json_path = os.path.join(traj_folder, f'trajectory{traj_id}.json')
        
        food_pcd_path = os.path.join(traj_folder, f'filtered_pcd_{folder_name}', 'food_filtered.npy')
        robot_pcd_folder = os.path.join(traj_folder, f'filtered_pcd_{folder_name}')
        output_dir = os.path.join(traj_folder, f'Merged_pcd_{folder_name}')
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n📂 Traitement : {folder_name}")

        if not os.path.exists(json_path): continue
        with open(json_path, 'r') as f: json_data = json.load(f)

        # --- OPTION B PREPARATION : Trouver le Grasp (Smart Detection) ---
        states = json_data['states']
        positions = [s['end_effector_position'] for s in states]
        
        # Récupération directe car c'est déjà une liste de nombres dans le JSON
        orientations = [s['end_effector_orientation'] for s in states] 
        
        zs = np.array([p[2] for p in positions])
        
        # Conversion directe de la liste de listes en matrice NumPy (N, 4)
        qs = np.array(orientations)

        # 1. Trouver le point le plus bas absolu (fin probable du scoop)
        min_z_idx = np.argmin(zs)

        # 2. Remonter le temps pour trouver le début de la rotation (début du scoop)
        grasp_idx = min_z_idx
        
        # Seuil : Si l'angle entre deux pas change de plus de ~0.5 degré (0.01 rad),
        # on considère que c'est un mouvement volontaire du poignet.
        rotation_threshold = 0.01 

        # On remonte du point le plus bas (min_z) vers le début
        for i in range(min_z_idx, 0, -1):
            # Produit scalaire entre le quaternion i et i-1
            dot_product = np.dot(qs[i], qs[i-1])
            
            # Gestion du "Double Cover" (q et -q sont la même rotation) et bornes num
            dot_product = np.abs(dot_product)
            dot_product = np.clip(dot_product, -1.0, 1.0) 
            
            # Calcul de l'angle de rotation entre les deux pas
            angle_change = 2 * np.arccos(dot_product)

            # TANT QUE l'angle change beaucoup, on est dans le "scoop".
            # DÈS QUE l'angle devient stable (petit), on est revenu à la descente verticale.
            if angle_change < rotation_threshold:
                grasp_idx = i
                break
        
        # Sécurité : Si on remonte trop loin (ex: début du fichier), on garde le min_z
        if grasp_idx < 5: 
             grasp_idx = min_z_idx
             print(f"   ⚠️ Pas de rotation détectée avant le min Z, utilisation de min_z ({min_z_idx})")
        else:
             print(f"   Success: Grasp détecté à l'idx {grasp_idx} (Début rotation), Min Z était à {min_z_idx}")

        grasp_state = json_data['states'][grasp_idx]
        
        # Note: Assurez-vous que votre fonction pose_to_matrix accepte une liste pour l'orientation
        T_world_grasp = pose_to_matrix(
            grasp_state['end_effector_position'], 
            grasp_state['end_effector_orientation']
        )
        T_grasp_world = np.linalg.inv(T_world_grasp)
        # -----------------------------------------------

        # 2. FOOD (Initial Statique)
        food_points_world_init = np.empty((0, 3))
        if os.path.exists(food_pcd_path):
            try:
                food_points_cam = np.load(food_pcd_path)
                if food_points_cam.size > 0:
                    T_ee_s_step0 = load_matrix_from_json(json_data, "T_ee_s", step_index=0)
                    food_points_world_init = apply_transform(food_points_cam, T_ee_s_step0)
            except Exception as e:
                print(f"  ⚠️ Erreur Food: {e}")

        # 3. ROBOT Loop
        robot_files = sorted(glob.glob(os.path.join(robot_pcd_folder, 'robot_cloud_*.npy')))
        if not robot_files: continue

        for robot_file in tqdm(robot_files, desc=f"  ↳ Fusion {folder_name}", unit="step"):
            filename = os.path.basename(robot_file)
            step_str = os.path.splitext(filename)[0].split('_')[-1] 
            try: step_idx = int(step_str) - 1
            except: continue

            try:
                robot_points_cam = np.load(robot_file)
                if robot_points_cam.size == 0: continue
                    
                T_static_s = load_matrix_from_json(json_data, "T_static_s", step_index=step_idx)
                robot_points_world = apply_transform(robot_points_cam, T_static_s)

                # --- GESTION FOOD DYNAMIQUE (Option B) ---
                current_food = food_points_world_init.copy()
                
                ### --- DEBUT BLOC OPTION B ---
                # Si on commente ce bloc, la nourriture reste statique (Option A)
                if step_idx >= grasp_idx and current_food.size > 0:
                    # Pose actuelle du robot
                    curr_state = json_data['states'][step_idx]
                    T_world_curr = pose_to_matrix(
                        curr_state['end_effector_position'], 
                        curr_state['end_effector_orientation']
                    )
                    # Transformation composée : World -> Grasp (Local) -> World (Nouveau)
                    # Le mouvement relatif du robot est appliqué à la nourriture
                    T_motion = T_world_curr @ T_grasp_world
                    current_food = apply_transform(food_points_world_init, T_motion)
                ### --- FIN BLOC OPTION B ---

                # 4. MERGE
                if current_food.shape[0] > 0:
                    merged_points = np.vstack((robot_points_world, current_food))
                else:
                    merged_points = robot_points_world

                # --- AJOUT FPS ICI ---
                num_target_points = 1024 
                if merged_points.shape[0] > num_target_points:
                    # Utilisation de fpsample pour réduire à 1024 points immédiatement
                    indices = fpsample.bucket_fps_kdline_sampling(merged_points.astype(np.float32), num_target_points, h=5)
                    merged_points = merged_points[indices]
                # ----------------------

                # 5. SAVE
                save_name = f"Merged_{step_str}.npy"
                save_path = os.path.join(output_dir, save_name)
                np.save(save_path, merged_points)
                
            except Exception as e:
                print(f"  ❌ Erreur sur {filename}: {e}")
                continue

    print("\n✅ Fusion terminée.")

def merge_pcd_trajectory_urdf():
    # --- CONFIGURATION ---
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    base_path_preprocess = os.path.join(package_path, 'datas', 'Trajectories_preprocess')
    base_path_record = os.path.join(package_path, 'datas', 'Trajectories_record')
    
    traj_folders_preprocess = sorted(glob.glob(os.path.join(base_path_preprocess, 'Trajectory*')), 
                                     key=lambda x: int(x.split('_')[-1]))

    print(f"🔄 Début de la fusion URDF pour {len(traj_folders_preprocess)} trajectoires...")

    for traj_folder_proc in traj_folders_preprocess:
        folder_name = os.path.basename(traj_folder_proc)
        traj_id = folder_name.split('_')[-1]
        
        json_path = os.path.join(traj_folder_proc, f'trajectory_{traj_id}.json')
        if not os.path.exists(json_path): json_path = os.path.join(traj_folder_proc, f'trajectory{traj_id}.json')
        
        food_pcd_path = os.path.join(traj_folder_proc, f'filtered_pcd_{folder_name}', 'food_filtered.npy')
        
        traj_folder_rec = os.path.join(base_path_record, folder_name)
        robot_urdf_folder = os.path.join(traj_folder_rec, f'images_{folder_name}')
        
        output_dir = os.path.join(traj_folder_proc, f'Merged_urdf_fork_{folder_name}')
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n📂 Traitement URDF avec fork: {folder_name}")

        if not os.path.exists(json_path): continue
        with open(json_path, 'r') as f: json_data = json.load(f)

        # --- OPTION B PREPARATION : Trouver le Grasp ---
        positions = [s['end_effector_position'] for s in json_data['states']]
        zs = [p[2] for p in positions]
        grasp_idx = np.argmin(zs)
        
        grasp_state = json_data['states'][grasp_idx]
        T_world_grasp = pose_to_matrix(
            grasp_state['end_effector_position'], 
            grasp_state['end_effector_orientation']
        )
        T_grasp_world = np.linalg.inv(T_world_grasp)
        # -----------------------------------------------

        # 2. FOOD (Initial Statique)
        food_points_world_init = np.empty((0, 3))
        if os.path.exists(food_pcd_path):
            try:
                food_points_cam = np.load(food_pcd_path)
                if food_points_cam.size > 0:
                    T_ee_s_step0 = load_matrix_from_json(json_data, "T_ee_s", step_index=0)
                    food_points_world_init = apply_transform(food_points_cam, T_ee_s_step0)
            except Exception as e:
                print(f"  ⚠️ Erreur Food: {e}")

        # 3. ROBOT URDF Loop
        urdf_files = sorted(glob.glob(os.path.join(robot_urdf_folder, 'Robot_point_cloud_*.npy')))
        if not urdf_files: continue

        for urdf_file in tqdm(urdf_files, desc=f"  ↳ Fusion URDF {folder_name}", unit="step"):
            filename = os.path.basename(urdf_file)
            try: 
                step_str = filename.split('.')[0].split('_')[-1]
                step_idx = int(step_str) - 1 # Attention à l'indexation (si fichiers commencent à 1)
            except: continue

            try:
                robot_points_world = np.load(urdf_file)
                if robot_points_world.ndim == 1: robot_points_world = robot_points_world.reshape(1, -1)
                if robot_points_world.shape[1] != 3 and robot_points_world.shape[0] == 3: robot_points_world = robot_points_world.T
                
                # --- GESTION FOOD DYNAMIQUE (Option B) ---
                current_food = food_points_world_init.copy()
                
                ### --- DEBUT BLOC OPTION B ---
                # Si on commente ce bloc, la nourriture reste statique
                if step_idx >= grasp_idx and current_food.size > 0:
                    curr_state = json_data['states'][step_idx]
                    T_world_curr = pose_to_matrix(
                        curr_state['end_effector_position'], 
                        curr_state['end_effector_orientation']
                    )
                    T_motion = T_world_curr @ T_grasp_world
                    current_food = apply_transform(food_points_world_init, T_motion)
                ### --- FIN BLOC OPTION B ---
                
                if current_food.shape[0] > 0:
                    merged_points = np.vstack((robot_points_world, current_food))
                else:
                    merged_points = robot_points_world

                # --- AJOUT FPS ICI ---
                num_target_points = 1024 
                if merged_points.shape[0] > num_target_points:
                    # Utilisation de fpsample pour réduire à 1024 points immédiatement
                    indices = fpsample.bucket_fps_kdline_sampling(merged_points.astype(np.float32), num_target_points, h=5)
                    merged_points = merged_points[indices]
                # ----------------------

                save_name = f"Merged_urdf_fork_{step_str}.npy"
                save_path = os.path.join(output_dir, save_name)
                np.save(save_path, merged_points)

            except Exception as e:
                print(f"  ❌ Erreur {filename}: {e}")

    print("\n✅ Fusion URDF terminée.")

def merge_pcd_trajectory_fork():
    # --- CONFIGURATION ---
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    base_path_preprocess = os.path.join(package_path, 'datas', 'Trajectories_preprocess')
    base_path_record = os.path.join(package_path, 'datas', 'Trajectories_record')
    
    traj_folders_preprocess = sorted(glob.glob(os.path.join(base_path_preprocess, 'Trajectory*')), 
                                     key=lambda x: int(x.split('_')[-1]))

    print(f"🔄 Début de la fusion Fork pour {len(traj_folders_preprocess)} trajectoires...")

    for traj_folder_proc in traj_folders_preprocess:
        folder_name = os.path.basename(traj_folder_proc)
        traj_id = folder_name.split('_')[-1]
        
        json_path = os.path.join(traj_folder_proc, f'trajectory_{traj_id}.json')
        if not os.path.exists(json_path): json_path = os.path.join(traj_folder_proc, f'trajectory{traj_id}.json')
        
        food_pcd_path = os.path.join(traj_folder_proc, f'filtered_pcd_{folder_name}', 'food_filtered.npy')
        
        traj_folder_rec = os.path.join(base_path_record, folder_name)
        robot_urdf_folder = os.path.join(traj_folder_rec, f'images_{folder_name}')
        
        output_dir = os.path.join(traj_folder_proc, f'Merged_Fork_{folder_name}')
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n📂 Traitement avec Fork: {folder_name}")

        if not os.path.exists(json_path): continue
        with open(json_path, 'r') as f: json_data = json.load(f)

        # --- OPTION B PREPARATION : Trouver le Grasp ---
        positions = [s['end_effector_position'] for s in json_data['states']]
        zs = [p[2] for p in positions]
        grasp_idx = np.argmin(zs)
        
        grasp_state = json_data['states'][grasp_idx]
        T_world_grasp = pose_to_matrix(
            grasp_state['end_effector_position'], 
            grasp_state['end_effector_orientation']
        )
        T_grasp_world = np.linalg.inv(T_world_grasp)
        # -----------------------------------------------

        # 2. FOOD (Initial Statique)
        food_points_world_init = np.empty((0, 3))
        if os.path.exists(food_pcd_path):
            try:
                food_points_cam = np.load(food_pcd_path)
                if food_points_cam.size > 0:
                    T_ee_s_step0 = load_matrix_from_json(json_data, "T_ee_s", step_index=0)
                    food_points_world_init = apply_transform(food_points_cam, T_ee_s_step0)
            except Exception as e:
                print(f"  ⚠️ Erreur Food: {e}")

        # 3. ROBOT URDF Loop
        fork_files = sorted(glob.glob(os.path.join(robot_urdf_folder, 'Fork_point_cloud_*.npy')))
        if not fork_files: continue

        for fork_file in tqdm(fork_files, desc=f"  ↳ Fusion Fork {folder_name}", unit="step"):
            filename = os.path.basename(fork_file)
            try: 
                step_str = filename.split('.')[0].split('_')[-1]
                step_idx = int(step_str) - 1 # Attention à l'indexation (si fichiers commencent à 1)
            except: continue

            try:
                fork_points_world = np.load(fork_file)
                if fork_points_world.ndim == 1: fork_points_world = fork_points_world.reshape(1, -1)
                if fork_points_world.shape[1] != 3 and fork_points_world.shape[0] == 3: fork_points_world = fork_points_world.T
                
                # --- GESTION FOOD DYNAMIQUE (Option B) ---
                current_food = food_points_world_init.copy()
                
                ### --- DEBUT BLOC OPTION B ---
                # Si on commente ce bloc, la nourriture reste statique
                if step_idx >= grasp_idx and current_food.size > 0:
                    curr_state = json_data['states'][step_idx]
                    T_world_curr = pose_to_matrix(
                        curr_state['end_effector_position'], 
                        curr_state['end_effector_orientation']
                    )
                    T_motion = T_world_curr @ T_grasp_world
                    current_food = apply_transform(food_points_world_init, T_motion)
                ### --- FIN BLOC OPTION B ---
                
                if current_food.shape[0] > 0:
                    merged_points = np.vstack((fork_points_world, current_food))
                else:
                    merged_points = fork_points_world

                # --- AJOUT FPS ICI ---
                num_target_points = 1024 
                if merged_points.shape[0] > num_target_points:
                    # Utilisation de fpsample pour réduire à 1024 points immédiatement
                    indices = fpsample.bucket_fps_kdline_sampling(merged_points.astype(np.float32), num_target_points, h=5)
                    merged_points = merged_points[indices]
                # ----------------------

                save_name = f"Merged_Fork_{step_str}.npy"
                save_path = os.path.join(output_dir, save_name)
                np.save(save_path, merged_points)

            except Exception as e:
                print(f"  ❌ Erreur {filename}: {e}")

    print("\n✅ Fusion avec fourchette terminée.")

def update_json_merged_pcd():
    """
    Étape 8 : Met à jour le JSON avec :
    1. Les chemins vers les nuages de points (Steps 5, 6, 7)
    2. La pose mondiale (position/orientation) de la fourchette pour chaque état.
    3. La matrice constante T_ee_fork en fin de fichier.
    """
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    base_path = os.path.join(package_path, 'datas', 'Trajectories_preprocess')
    
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
        
        directory_path = directory_path.replace("Trajectories_record", "Trajectories_preprocess")
        
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
    
    # print("="*80)
    # print("Step 3 : Segmenter les images avec SAM 2 et creer les nuages de points du robot segmenté et de la nourriture segmentée")
    # print("="*80)
    # start_time = time.time()
    
    # try:
    #     # Lancement de la fonction principale
    #     segment_robot_and_create_pcd()
        
    # except KeyboardInterrupt:
    #     print("\n🛑 Interruption par l'utilisateur.")
        
    # end_time = time.time()
    # print(f"Temps total pour segmenter les nuages de points : {end_time - start_time}")
    

    print("="*80)
    print("Step 4 : Segmenter la nourriture dans l'assiette et créer le nuage de point correspondant.")
    print("="*80)
    force_gpu_clear()
    start_time = time.time()
    segment_food_init_step()
    end_time = time.time()
    print(f"Temps total pour segmenter la nourriture dans l'assiette et créer le nuage de point correspondant : {end_time - start_time}")
    
    # print("="*80)
    # print("Step 5 : Créer les nuages de points merge du robot segmenter et de la nourriture segmentée.")
    # print("="*80)
    # start_time = time.time()
    # merge_pcd_trajectory()
    # end_time = time.time()
    # print(f"Temps total pour créer les nuages de points merge : {end_time - start_time}")
    
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