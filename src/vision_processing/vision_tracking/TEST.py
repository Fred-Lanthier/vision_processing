import os
import torch
import numpy as np
import cv2
import glob
import sys
import rospkg
import re
from PIL import Image
import time
import zipfile
import tempfile

print(f"Python version: {sys.version}")

# --- IMPORTS SAM 3 ---
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- IMPORTS SAM 2 ---
from sam2.build_sam import build_sam2_video_predictor

# --- CONFIGURATION ---
package_path = rospkg.RosPack().get_path('vision_processing')
base_path = os.path.join(package_path, 'src', "vision_processing", "vision_tracking")
zip_file = os.path.join(base_path, "Trajectories_record.zip")

traj_idx = 1
OUTPUT_DIR = os.path.join(base_path, f"output_trajectory_{traj_idx}")
TEXT_PROMPT = "dark object in trapezoidal shape with pink side" 

SAM2_CHECKPOINT = os.path.expanduser("~/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml" 

def extract_and_prepare_temp_images(zip_path, traj_idx, temp_dir):
    """
    Lit uniquement les PNG "ee_rgb_step_XXXX.png" depuis le ZIP, 
    les convertit en JPG et les place dans un dossier temporaire.
    """
    print("📦 Lecture des images depuis le ZIP vers la mémoire temporaire...")
    cnt = 0
    
    # Le préfixe (chemin interne dans le zip) pour trouver tes images
    prefix = f'Trajectories_record/Trajectory_{traj_idx}/images_Trajectory_{traj_idx}/'
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        # On filtre par le préfixe du dossier ET par le nom du fichier "ee_rgb_step_"
        png_files = []
        for f in z.namelist():
            if f.startswith(prefix) and f.lower().endswith('.png'):
                filename = os.path.basename(f)
                if filename.startswith('ee_rgb_step_'):
                    png_files.append(f)
                    
        png_files = sorted(png_files)
        
        if not png_files:
            raise RuntimeError(f"❌ Aucune image 'ee_rgb_step_' trouvée dans le zip avec le préfixe : {prefix}")

        for file_path in png_files:
            file_data = z.read(file_path)
            
            nparr = np.frombuffer(file_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                filename = os.path.basename(file_path)
                # On s'assure de bien extraire les chiffres du nom "ee_rgb_step_XXXX.png"
                match = re.search(r'ee_rgb_step_(\d+)', filename)
                if match:
                    number_str = match.group(1)
                    new_filename = f"{int(number_str):05d}.jpg"
                    new_path = os.path.join(temp_dir, new_filename)
                    
                    cv2.imwrite(new_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    cnt += 1
                    
    print(f"✅ {cnt} images prêtes en mémoire temporaire pour SAM 2.")
    return png_files

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 [PHASE 0] Préparation des données...")
    
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)

    with tempfile.TemporaryDirectory() as temp_image_dir:
        original_png_paths = extract_and_prepare_temp_images(zip_file, traj_idx, temp_image_dir)
        
        temp_jpgs = sorted(glob.glob(os.path.join(temp_image_dir, "*.jpg")))
        if not temp_jpgs:
            return

        # ====================================================
        # ÉTAPE 1 : SAM 3 (Init)
        # ====================================================
        print("\n🧠 [PHASE 1] SAM 3 : Initialisation sémantique...")
        sam3 = build_sam3_image_model()
        sam3_image_predictor = Sam3Processor(sam3)
        
        frame_0 = Image.open(temp_jpgs[0]).convert("RGB")
        inference_state = sam3_image_predictor.set_image(frame_0)
        output = sam3_image_predictor.set_text_prompt(state=inference_state, prompt=TEXT_PROMPT)
        initial_mask_tensor = output["masks"][0]
        
        if initial_mask_tensor is None: 
            print("❌ SAM 3 n'a trouvé aucun masque.")
            return

        print("   ☁️ Extraction du masque Frame 0...")
        
        # ====================================================
        # ÉTAPE 2 : SAM 2 (Tracking)
        # ====================================================
        print(f"\n⚡ [PHASE 2] SAM 2 : Tracking & Sauvegarde des masques...")
        
        sam2_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
        inference_state = sam2_predictor.init_state(video_path=temp_image_dir)
        
        if isinstance(initial_mask_tensor, torch.Tensor):
            mask_np = initial_mask_tensor.detach().cpu().numpy()
        else:
            mask_np = initial_mask_tensor
            
        mask_np = np.squeeze(mask_np) 
        mask_input = torch.from_numpy(mask_np > 0).bool().to(device)
        
        sam2_predictor.add_new_mask(inference_state=inference_state, frame_idx=0, obj_id=1, mask=mask_input)

        # --- Paramètres d'affichage du masque ---
        # OpenCV utilise le format BGR (Bleu, Vert, Rouge). 
        # [255, 0, 255] donne un Magenta/Rose bien visible.
        mask_color_bgr = np.array([255, 0, 255], dtype=np.uint8) 
        alpha = 0.5 # 50% de transparence pour le masque

        # --- BOUCLE PRINCIPALE ---
        start_time = time.time()
        count = 0
        
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            for frame_idx, obj_ids, video_res_masks in sam2_predictor.propagate_in_video(inference_state):
                
                raw_mask = (video_res_masks[0] > 0).cpu().numpy()
                mask_np = raw_mask.squeeze()
                if mask_np.ndim > 2: mask_np = mask_np[0]
                
                # 1. Charger l'image originale correspondante (depuis le dossier temporaire)
                img_path = os.path.join(temp_image_dir, f"{frame_idx:05d}.jpg")
                img = cv2.imread(img_path)

                if img is not None:
                    # 2. Créer une copie de l'image et y appliquer la couleur pure sur les pixels du masque
                    overlay = img.copy()
                    overlay[mask_np > 0] = mask_color_bgr
                    
                    # 3. Fusionner (Blend) l'image modifiée avec l'image originale pour l'effet de transparence
                    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                    
                    # 4. Sauvegarder le résultat
                    output_filename = os.path.join(OUTPUT_DIR, f"overlay_ee_rgb_step_{frame_idx:04d}.png")
                    cv2.imwrite(output_filename, blended)

                if count % 10 == 0:
                    print(f"   Frame {frame_idx}: Image superposée sauvegardée.")
                count += 1

        total_time = time.time() - start_time
        print(f"\n✅ Terminé ! {count} images générées en {total_time:.2f} secondes dans : {OUTPUT_DIR}")

    print("🧹 Nettoyage terminé : Les images temporaires ont été supprimées.")

if __name__ == "__main__":
    main()