import os
import torch
import numpy as np
import cv2
import glob
import gc # Garbage Collector pour la mémoire
import sys
import rospkg
import re

# --- IMPORTS SAM 3 (Pour l'intelligence Textuelle) ---
from sam3.model_builder import build_sam3_video_predictor

# --- IMPORTS SAM 2 (Pour la vitesse de Tracking) ---
# Basé sur votre exemple, mais version Vidéo
from sam2.build_sam import build_sam2_video_predictor

# --- CONFIGURATION ---
package_path = rospkg.RosPack().get_path('vision_processing')
IMAGES_FOLDER = os.path.join(package_path, 'scripts', 'images_trajectory')
TEXT_PROMPT = "robot" 
OUTPUT_DIR = os.path.join(package_path, 'scripts', 'results_hybrid_final')

# Chemins SAM 2 (Adaptez selon votre installation)
# Le modèle "Tiny" (hiera_t) est recommandé pour le temps réel sur 3090
SAM2_CHECKPOINT = os.path.expanduser(f"~/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml" 

def prepare_images_for_sam2(folder_path):
    """
    1. Convertit PNG -> JPG
    2. RENOMME les fichiers pour ne garder que les chiffres (Ex: 'img_0106.png' -> '00106.jpg')
    C'est obligatoire pour SAM 2.
    """
    print(f"🔧 Préparation des images dans {folder_path}...")
    
    # On liste tout
    all_files = sorted(glob.glob(os.path.join(folder_path, "*")))
    pngs = [f for f in all_files if f.lower().endswith('.png')]
    jpgs = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg'))]
    
    # Si on a déjà des JPGs qui ressemblent à des nombres ("00106.jpg"), c'est bon
    # Sinon, on doit nettoyer.
    
    cnt = 0
    for file_path in pngs:
        filename = os.path.basename(file_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # On cherche le nombre dans le nom (ex: "static_rgb_step_0106" -> "0106")
        match = re.search(r'(\d+)', name_no_ext)
        
        if match:
            number_str = match.group(1)
            # On crée un nom propre pour SAM 2 : "00106.jpg"
            new_filename = f"{int(number_str):05d}.jpg"
            new_path = os.path.join(folder_path, new_filename)
            
            # On ne convertit que si le JPG n'existe pas déjà
            if not os.path.exists(new_path):
                img = cv2.imread(file_path)
                if img is not None:
                    cv2.imwrite(new_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    cnt += 1
        else:
            print(f"⚠️ Impossible de trouver un numéro dans : {filename}")

    if cnt > 0:
        print(f"✅ {cnt} images converties et renommées (Format 00106.jpg).")
    else:
        print("ℹ️ Aucune nouvelle conversion nécessaire (ou fichiers introuvables).")

    # Vérification finale pour SAM 2
    final_jpgs = glob.glob(os.path.join(folder_path, "*.jpg"))
    if not final_jpgs:
        raise RuntimeError("❌ Pas d'images JPG valides trouvées après conversion !")
        
    print(f"ready: {len(final_jpgs)} images prêtes pour SAM 2.")

def clean_gpu():
    """Force le nettoyage de la VRAM"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device}")
    
    # Vérification des fichiers SAM 2
    if not os.path.exists(SAM2_CHECKPOINT):
        print(f"❌ ERREUR: Checkpoint SAM 2 introuvable : {SAM2_CHECKPOINT}")
        return

    frame_names = sorted(glob.glob(os.path.join(IMAGES_FOLDER, "*.png")) + glob.glob(os.path.join(IMAGES_FOLDER, "*.jpg")))
    if not frame_names: 
        print("❌ Aucune image trouvée.")
        return

    # ====================================================
    # ÉTAPE 1 : SAM 3 (Texte -> Masque Initial)
    # ====================================================
    print("\n🧠 [PHASE 1] SAM 3 : Initialisation sémantique...")
    
    sam3 = build_sam3_video_predictor()
    
    # Démarrage session SAM 3
    resp = sam3.handle_request(request={"type": "start_session", "resource_path": IMAGES_FOLDER})
    s3_id = resp["session_id"]
    
    # Prompt Textuel
    print(f"   🤖 Recherche du concept : '{TEXT_PROMPT}'...")
    sam3.handle_request(request={"type": "add_prompt", "session_id": s3_id, "frame_index": 0, "text": TEXT_PROMPT})
    
    # On récupère juste le masque de la frame 0
    # On demande un tracking très court (1 frame) juste pour extraire la donnée
    gen = sam3.handle_stream_request(request={
        "type": "propagate_in_video", 
        "session_id": s3_id, 
        "start_frame_idx": 0,
        "max_frame_num_to_track": 1 
    })
    
    initial_mask_tensor = None
    
    # Utilisation de bfloat16 pour accélérer
    with torch.autocast(device, dtype=torch.bfloat16):
        for res in gen:
            if res['frame_index'] == 0:
                # C'est ici qu'on capture le masque généré par SAM 3
                # Il est sous forme de Tensor CUDA
                initial_mask_tensor = res['outputs']['out_binary_masks']
                break
    
    if initial_mask_tensor is None:
        print("❌ SAM 3 n'a pas réussi à segmenter l'objet.")
        return

    print(f"✅ Masque capturé (Shape: {initial_mask_tensor.shape})")

    # ====================================================
    # ÉTAPE 2 : HANDOVER (Nettoyage Mémoire)
    # ====================================================
    print("\n🧹 [PHASE 2] Suppression de SAM 3 de la mémoire...")
    
    # On supprime tout ce qui concerne SAM 3
    del sam3
    del gen
    del resp
    clean_gpu()
    
    print("   VRAM libérée pour SAM 2.")

    # ====================================================
    # ÉTAPE 3 : SAM 2 (Masque -> Tracking Haute Vitesse)
    # ====================================================
    print(f"\n⚡ [PHASE 3] SAM 2 : Tracking Temps Réel ({SAM2_CONFIG})...")
    
    # On construit le prédicteur VIDÉO de SAM 2 (et non ImagePredictor)
    prepare_images_for_sam2(IMAGES_FOLDER)
    sam2_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    
    # Initialisation de l'état SAM 2
    inference_state = sam2_predictor.init_state(video_path=IMAGES_FOLDER)
    
    # --- L'INJECTION DU MASQUE ---
    print("💉 Injection du masque SAM 3 dans le moteur SAM 2...")
    
    # 1. On repasse tout en Numpy pour nettoyer proprement
    if isinstance(initial_mask_tensor, torch.Tensor):
        mask_np = initial_mask_tensor.detach().cpu().numpy()
    else:
        mask_np = initial_mask_tensor

    # 2. On s'assure d'avoir une matrice 2D (H, W) -> (480, 640)
    # SAM 2 déteste le format (1, 480, 640) pour add_new_mask
    mask_np = np.squeeze(mask_np) 
    
    # Sécurité : Si squeeze a tout enlevé (cas scalaire rare), on expand
    if mask_np.ndim == 0:
        raise ValueError("Le masque est vide ou invalide !")

    # 3. Conversion en Tensor PyTorch BOOLLÉEN sur GPU
    # C'est exactement ce que "assert mask.dim() == 2" attend
    mask_input = torch.from_numpy(mask_np > 0).bool().to(device)
    
    print(f"   Shape finale envoyée à SAM 2 : {mask_input.shape} (Type: {mask_input.dtype})")

    # 4. Appel à l'API
    _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        mask=mask_input # Tensor 2D sur GPU
    )

    # --- LE TRACKING ---
    print("🌊 Lancement du Tracking...")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    import time
    start_time = time.time()
    count = 0
    
    # Mode Inférence pure
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        
        # Propagation via SAM 2
        for frame_idx, obj_ids, video_res_masks in sam2_predictor.propagate_in_video(inference_state):
            
            # Pour la démo, on sauvegarde une image sur 5
            if count % 5 == 0:
                # ÉTAPE CORRIGÉE :
                # 1. Récupération (GPU -> CPU)
                # video_res_masks est souvent [K, 1, H, W] ou [K, H, W]
                raw_mask = (video_res_masks[0] > 0).cpu().numpy()
                
                # 2. Aplatissement (Squeeze)
                # Transforme (1, 480, 640) -> (480, 640)
                mask_np = raw_mask.squeeze()
                
                # Vérification de sécurité (au cas où squeeze enlèverait trop)
                if mask_np.ndim > 2:
                    mask_np = mask_np[0] # On force la 2D

                # I/O Disque
                img_path = frame_names[frame_idx]
                image = cv2.imread(img_path)
                
                if image is not None:
                    # Overlay Vert
                    overlay = image.copy()
                    
                    # Maintenant les dimensions matchent !
                    try:
                        overlay[mask_np] = [0, 255, 0]
                        output = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
                        
                        filename = f"track_{frame_idx:04d}.jpg"
                        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), output)
                    except IndexError as e:
                        print(f"⚠️ Skip frame {frame_idx}: Erreur de dimension masque/image ({mask_np.shape} vs {image.shape})")
            
            count += 1
            if count % 20 == 0:
                print(f"   Tracking frame {frame_idx}...")

    total_time = time.time() - start_time
    print(f"\n✅ Terminé !")
    print(f"🚀 {count} frames traitées en {total_time:.2f}s")
    print(f"📊 FPS Global (incluant I/O partiel) : {count/total_time:.2f}")

if __name__ == "__main__":
    main()