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

# --- IMPORTS SAM 3 ---
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- IMPORTS SAM 2 ---
from sam2.build_sam import build_sam2_video_predictor

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

def segment_food():
    # TODO
    return None

def segment_robot_and_create_pcd():
    # --- CONFIGURATION ---
    package_path = rospkg.RosPack().get_path('vision_processing')
    TRAJECTORIES_PATH = glob.glob(os.path.join(package_path, 'datas', 'Trajectories_preprocess', 'Trajectory*'))
    TRAJECTORIES_PATH_RAW = glob.glob(os.path.join(package_path, 'datas', 'Trajectories_record', 'Trajectory*'))
    TRAJECTORIES_PATH = sorted(TRAJECTORIES_PATH, key=lambda x: int(x.split('_')[-1]))
    TRAJECTORIES_PATH_RAW = sorted(TRAJECTORIES_PATH_RAW, key=lambda x: int(x.split('_')[-1]))

    SAM2_CHECKPOINT = os.path.expanduser(f"~/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt")
    SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml" 
    TEXT_PROMPT = "robot" 

    for i, trajectory_path in enumerate(TRAJECTORIES_PATH):
        IMAGES_FOLDER = os.path.join(trajectory_path, f'images_Trajectory_{i+1}')
        DEPTH_NPY_FILES = glob.glob(os.path.join(TRAJECTORIES_PATH_RAW[i], f'images_Trajectory_{i+1}', 'static_depth_step_*.npy'))
        DEPTH_NPY_FILES = sorted(DEPTH_NPY_FILES, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

        OUTPUT_ROBOT_PCD_DIR = os.path.join(trajectory_path, 'filtered_pcd')
        os.makedirs(OUTPUT_ROBOT_PCD_DIR, exist_ok=True)

        OUTPUT_SEGMENTED_IMAGES_DIR = os.path.join(trajectory_path, 'segmented_images')
        os.makedirs(OUTPUT_SEGMENTED_IMAGES_DIR, exist_ok=True)

    
        # ====================================================
        # ÉTAPE 0 : Configurer les paths
        # ====================================================
        IMAGES_FOLDER = os.path.join(trajectory_path, f'images_Trajectory_{i+1}')
        DEPTH_NPY_FILES = glob.glob(os.path.join(TRAJECTORIES_PATH_RAW[i], f'images_Trajectory_{i+1}', 'static_depth_step_*.npy'))
        DEPTH_NPY_FILES = sorted(DEPTH_NPY_FILES, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

        OUTPUT_ROBOT_PCD_DIR = os.path.join(trajectory_path, 'filtered_pcd')
        os.makedirs(OUTPUT_ROBOT_PCD_DIR, exist_ok=True)

        OUTPUT_SEGMENTED_IMAGES_DIR = os.path.join(trajectory_path, 'segmented_images')
        os.makedirs(OUTPUT_SEGMENTED_IMAGES_DIR, exist_ok=True)
        
        # ====================================================
        # ÉTAPE 1 : SAM 3 (Init)
        # ====================================================
        print("\n🧠 [PHASE 1] SAM 3 : Initialisation sémantique...")
        sam3 = build_sam3_image_model()
        sam3_image_predictor = Sam3Processor(sam3)
        
        frame_0 = Image.open(frame_names[0]).convert("RGB")
        inference_state = sam3_image_predictor.set_image(frame_0)
        output = sam3_image_predictor.set_text_prompt(state=inference_state, prompt=TEXT_PROMPT)
        initial_mask_tensor = output["masks"][0]
        
        if initial_mask_tensor is None: return

        # --- TRAITEMENT FRAME 0 ---
        print("   ☁️  Calcul 3D Frame 0...")
        mask_0_np = initial_mask_tensor.detach().cpu().numpy().squeeze()
        if len(depth_files) > 0:
            process_depth_to_robot_cloud(depth_files[0], mask_0_np, 0, OUTPUT_ROBOT_PCD_DIR)
        
        # ====================================================
        # ÉTAPE 2 : SAM 2 (Tracking)
        # ====================================================
        print(f"\n⚡ [PHASE 2] SAM 2 : Tracking & Calcul 3D...")
        
        prepare_images_for_sam2(IMAGES_FOLDER)
        sam2_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
        inference_state = sam2_predictor.init_state(video_path=IMAGES_FOLDER)
        
        # Injection Masque SAM 3
        if isinstance(initial_mask_tensor, torch.Tensor):
            mask_np = initial_mask_tensor.detach().cpu().numpy()
        else:
            mask_np = initial_mask_tensor
            
        mask_np = np.squeeze(mask_np) 
        mask_input = torch.from_numpy(mask_np > 0).bool().to(device)
        
        sam2_predictor.add_new_mask(inference_state=inference_state, frame_idx=0, obj_id=1, mask=mask_input)

        # --- BOUCLE PRINCIPALE ---
        start_time = time.time()
        count = 0
        
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            
            for frame_idx, obj_ids, video_res_masks in sam2_predictor.propagate_in_video(inference_state):
                
                # 1. Récupérer le masque SAM (CPU, Numpy)
                raw_mask = (video_res_masks[0] > 0).cpu().numpy()
                mask_np = raw_mask.squeeze()
                if mask_np.ndim > 2: mask_np = mask_np[0]

                # 2. CALCUL 3D DU ROBOT
                # On vérifie que le fichier depth correspondant existe
                if frame_idx < len(depth_files):
                    process_depth_to_robot_cloud(
                        depth_file=depth_files[frame_idx], 
                        mask_2d=mask_np, 
                        frame_idx=frame_idx, 
                        output_dir=OUTPUT_ROBOT_PCD_DIR
                    )

                # 3. Visualisation (Optionnelle, pour debug)
                if count % 10 == 0:
                    print(f"   Frame {frame_idx}: 3D Cloud generated.")
                
                count += 1

        total_time = time.time() - start_time
        print(f"\n✅ Terminé ! {count} nuages de points générés en {total_time:.2f} secondes dans : {OUTPUT_ROBOT_PCD_DIR}")

def process_depth_to_robot_cloud(depth_file, mask_2d, frame_idx, output_dir):
    """
    Combine le masque 2D de SAM et la Depth Map pour créer un nuage de points 3D du robot.
    """
    # 1. Charger la Depth Map (H, W) en mm
    try:
        depth_map = np.load(depth_file)
    except Exception as e:
        print(f"⚠️ Erreur chargement depth {depth_file}: {e}")
        return

    # Vérification dimensions
    if depth_map.shape != mask_2d.shape:
        # print(f"⚠️ Dimension mismatch: Depth {depth_map.shape} vs Mask {mask_2d.shape}")
        return

    # 2. Intrinsèques Caméra (Vos valeurs)
    fx = 616.1005249
    fy = 615.82617188
    cx = 318.38803101
    cy = 249.23504639

    # 3. Création des coordonnées de pixels (u, v)
    h, w = depth_map.shape
    # 'ij' indexing: v=lignes (y), u=colonnes (x)
    v_grid, u_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # 4. Masque Combiné : Pixel doit être "Robot" ET avoir une "Profondeur Valide (>0)"
    combined_mask = mask_2d & (depth_map > 0)
    
    count_points = np.sum(combined_mask)
    if count_points == 0:
        return # Rien à calculer

    # 5. Extraction Optimisée (Vectorization)
    # On ne prend QUE les pixels qui nous intéressent. C'est très rapide.
    z_mm_valid = depth_map[combined_mask]
    u_valid = u_grid[combined_mask]
    v_valid = v_grid[combined_mask]

    # 6. Projection Inverse (2D -> 3D)
    z_meters = z_mm_valid / 1000.0
    x_meters = (u_valid - cx) * z_meters / fx
    y_meters = (v_valid - cy) * z_meters / fy

    # 7. Création du nuage (N, 3)
    robot_pcd = np.column_stack((x_meters, y_meters, z_meters))

    # 8. Ajout des point de la nourriture

    # 8. Sauvegarde
    output_filename = f"robot_cloud_{frame_idx:04d}.npy"
    save_path = os.path.join(output_dir, output_filename)
    np.save(save_path, robot_pcd)


def transform_static_ee_npy():
    # TODO
    return None

def main():

    print("="*80)
    print("Step 0 : Vérifier quelles trajectoires ont déjà été traitées")
    print("="*80)
    


    print("="*80)
    print("Step 1 : Convertir les PNG static en JPG pour faire la segmentation et le tracking avec SAM 2")
    print("="*80)
    # start_time = time.time()
    # png_datas = load_png_datas_static()
    
    # for png_path in png_datas:
    #     directory_path = os.path.dirname(png_path)
    #     filename = os.path.basename(png_path)
    #     name_no_ext = os.path.splitext(filename)[0]
        
    #     directory_path = directory_path.replace("Trajectories_record", "Trajectories_preprocess")
        
    #     try:
    #         # On extrait le numéro "11" depuis "static_rgb_step_11"
    #         step_number = int(name_no_ext.split('_')[-1])
            
    #         # On définit le nom CIBLE qu'on veut obtenir : "0011.jpg"
    #         target_name = f"{step_number:04d}.jpg"
    #         target_path = os.path.join(directory_path, target_name)
            
    #         # LE TEST ULTIME : Si le résultat existe déjà, on passe !
    #         if os.path.exists(target_path):
    #             # print(f"Le fichier {target_name} existe déjà. Skip.")
    #             continue

    #         # Sinon, on lance la conversion
    #         print(f"Création de {target_name} à partir de {filename}...")
    #         jpg_path = os.path.join(directory_path, target_name)
    #         print(jpg_path)
    #         png_to_jpg_static(png_path, jpg_path)
    #     except Exception as e:
    #         print(f"Erreur sur {png_path}: {e}")
    # end_time = time.time()
    # print(f"Temps total pour transformer les PNG static en JPG : {end_time - start_time}")
    
    
    print("="*80)
    print("Step 2 : Ajouter les fichiers json dans les nouveaux dossiers")
    print("="*80)
    # start_time = time.time()
    # mappings = get_json_paths_map()
    # for src, dest in mappings:
    #     print(f"Source: {os.path.basename(src)} -> Dest: {dest}")
    #     with open(src, 'r') as f:
    #         data = json.load(f)
    #     with open(dest, 'w') as f:
    #         json.dump(data, f)
        
    # end_time = time.time()
    # print(f"Temps total pour ajouter les fichiers json dans les nouveaux dossiers : {end_time - start_time}")
    
    print("="*80)
    print("Step 3 : Segmenter les images avec SAM 2 et creer les nuages de points du robot segmenté et de la nourriture segmentée")
    print("="*80)
    
    start_time = time.time()
    
    end_time = time.time()
    print(f"Temps total pour segmenter les nuages de points : {end_time - start_time}")
    
    print("="*80)
    print("Step 4 : Créer les nuages de points merge.")
    print("="*80)
    
    start_time = time.time()
    
    end_time = time.time()
    print(f"Temps total pour créer les nuages de points merge : {end_time - start_time}")
    
if __name__ == "__main__":
    main()