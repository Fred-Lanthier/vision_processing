import os
import torch
import numpy as np
import cv2
import glob
import gc
import sys
import rospkg
import re
from PIL import Image
import time

# --- IMPORTS SAM 3 ---
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- IMPORTS SAM 2 ---
from sam2.build_sam import build_sam2_video_predictor

# --- CONFIGURATION ---
package_path = rospkg.RosPack().get_path('vision_processing')
IMAGES_FOLDER = os.path.join(package_path, 'scripts', 'images_trajectory')
# Dossier contenant vos .npy de profondeur
DEPTH_NPY_FOLDER = os.path.join(package_path, 'scripts', 'Robot_depth_trajectory') 
TEXT_PROMPT = "robot" 
OUTPUT_DIR = os.path.join(package_path, 'scripts', 'results_hybrid_final')
OUTPUT_ROBOT_PCD_DIR = os.path.join(package_path, 'scripts', 'Robot_pcd_filtered')

SAM2_CHECKPOINT = os.path.expanduser(f"~/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml" 

def prepare_images_for_sam2(folder_path):
    """Convertit PNG -> JPG et renomme en 00001.jpg pour SAM 2"""
    # ... (Même code que précédemment) ...
    all_files = sorted(glob.glob(os.path.join(folder_path, "*")))
    pngs = [f for f in all_files if f.lower().endswith('.png')]
    cnt = 0
    for file_path in pngs:
        filename = os.path.basename(file_path)
        name_no_ext = os.path.splitext(filename)[0]
        match = re.search(r'(\d+)', name_no_ext)
        if match:
            number_str = match.group(1)
            new_filename = f"{int(number_str):05d}.jpg"
            new_path = os.path.join(folder_path, new_filename)
            if not os.path.exists(new_path):
                img = cv2.imread(file_path)
                if img is not None:
                    cv2.imwrite(new_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    cnt += 1
    final_jpgs = glob.glob(os.path.join(folder_path, "*.jpg"))
    if not final_jpgs: raise RuntimeError("❌ Pas d'images JPG valides !")
    print(f"ready: {len(final_jpgs)} images prêtes pour SAM 2.")

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

    # 8. Sauvegarde
    output_filename = f"robot_cloud_{frame_idx:04d}.npy"
    save_path = os.path.join(output_dir, output_filename)
    np.save(save_path, robot_pcd)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 [PHASE 0] Préparation des données...")
    if not os.path.exists(OUTPUT_ROBOT_PCD_DIR): os.makedirs(OUTPUT_ROBOT_PCD_DIR)
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # Listes des fichiers
    frame_names = sorted(glob.glob(os.path.join(IMAGES_FOLDER, "*.png")) + glob.glob(os.path.join(IMAGES_FOLDER, "*.jpg")))
    depth_files = sorted(glob.glob(os.path.join(DEPTH_NPY_FOLDER, "*.npy"))) # Vos fichiers depth
    
    if len(frame_names) == 0: return
    if len(depth_files) == 0: return
    
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

if __name__ == "__main__":
    main()