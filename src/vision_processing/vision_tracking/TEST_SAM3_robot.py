#! /usr/bin/env python3
import torch
import numpy as np
import os
import shutil
import rospkg
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize
from scipy.signal import convolve2d

# Imports SAM 3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class SegmentEndEffectorDebug:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⏳ Chargement du modèle SAM 3 sur {self.device}...")
        self.model = build_sam3_image_model()
        if hasattr(self.model, "to"): self.model.to(self.device)
        self.processor = Sam3Processor(self.model)
        self.output_dir = os.path.expanduser("~/Github/src/vision_processing/scripts")
        
        self.debug_dir = os.path.join(self.output_dir, "debug_tests")
        if os.path.exists(self.debug_dir): shutil.rmtree(self.debug_dir)
        os.makedirs(self.debug_dir, exist_ok=True)
        print(f"✅ Modèle chargé. Debug dans : {self.debug_dir}")

    def _get_overlay(self, image, mask, color_rgb, alpha_val=0.6):
        ov = Image.new("RGBA", image.size, color_rgb + (0,))
        m = np.squeeze(mask)
        if m.ndim > 2: m = m[0,:,:]
        # Assurer que m est booléen pour la conversion
        m = m > 0
        m_pil = Image.fromarray((m*255).astype(np.uint8), mode='L')
        if m_pil.size != image.size: m_pil = m_pil.resize(image.size, Image.NEAREST)
        ov.putalpha(m_pil.point(lambda x: int(255*alpha_val) if x>0 else 0))
        return Image.alpha_composite(image.convert("RGBA"), ov)

    def get_skeleton_endpoints(self, mask):
        print("🦴 Squelettisation...")
        mask = np.squeeze(mask)
        if mask.ndim > 2: mask = mask[0, :, :]
        skeleton = skeletonize(mask)
        skel_int = skeleton.astype(int)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbors = convolve2d(skel_int, kernel, mode='same')
        endpoints_mask = (skel_int == 1) & (neighbors == 1)
        y_ends, x_ends = np.where(endpoints_mask)
        return [[int(x), int(y)] for x, y in zip(x_ends, y_ends)]

    def process_end_effector(self, image_path, parent_prompt="robot"):
        if not os.path.exists(image_path): return
        print(f"📸 Image : {image_path}")
        image_full = Image.open(image_path).convert("RGB")
        W, H = image_full.size
        
        # 1. Segmentation Globale
        state_full = self.processor.set_image(image_full)
        out_p = self.processor.set_text_prompt(state=state_full, prompt=parent_prompt)
        m = out_p["masks"]
        if isinstance(m, torch.Tensor): m = m.detach().cpu().numpy()
        robot_mask = np.any(m > 0, axis=0) if m.ndim > 2 else m > 0
        if not np.any(robot_mask): print("❌ Robot introuvable."); return
        full_robot_area = np.sum(robot_mask)

        # 2. Extrémités
        endpoints = self.get_skeleton_endpoints(robot_mask)
        print(f"📍 {len(endpoints)} points à tester.")

        # 3. Boucle de tests
        best_mask_full = None
        min_area = float('inf')
        CROP_S = 160 
        half = CROP_S // 2

        for idx, pt in enumerate(endpoints):
            cx, cy = pt
            x1, y1 = max(0, cx-half), max(0, cy-half)
            x2, y2 = min(W, cx+half), min(H, cy+half)
            image_crop = image_full.crop((x1, y1, x2, y2))
            wc, hc = image_crop.size
            rcx, rcy = cx-x1, cy-y1

            print(f"--- Test P{idx} ---")
            state_crop = self.processor.set_image(image_crop)
            
            box_pos = [rcx-2, rcy-2, rcx+2, rcy+2]
            mar = 4
            neg_boxes = [[0,0,mar,mar], [wc-mar,0,wc,mar], [0,hc-mar,mar,hc], [wc-mar,hc-mar,wc,hc]]

            try:
                if hasattr(self.processor, "reset_all_prompts"): self.processor.reset_all_prompts(state_crop)
                self.processor.add_geometric_prompt(box=box_pos, label=True, state=state_crop)
                for nb in neg_boxes:
                    out_c = self.processor.add_geometric_prompt(box=nb, label=False, state=state_crop)
            except: continue

            mask_local = None
            ratio = 1.0
            is_inverted = False # Flag pour le nom de fichier

            if out_c.get("masks") is not None:
                mc = out_c["masks"]
                if isinstance(mc, torch.Tensor): mc = mc.detach().cpu().numpy()
                if isinstance(mc, list): mc = np.array(mc)
                
                # On prend d'abord le plus petit masque candidat
                if mc.ndim > 2: mask_local = mc[np.argmin([np.sum(s) for s in mc])]
                else: mask_local = mc
                
                # Nettoyage dimension (H, W) booléen
                mask_local = np.squeeze(mask_local)
                if mask_local.ndim > 2: mask_local = mask_local[0,:,:]
                mask_local = mask_local > 0 # Force bool

                if np.any(mask_local):
                    # ==================================================
                    # --- DÉTECTION ET CORRECTION D'INVERSION ---
                    # ==================================================
                    h_m, w_m = mask_local.shape
                    # Somme des pixels sur les 4 bords
                    border_pixels = (np.sum(mask_local[0, :]) + np.sum(mask_local[-1, :]) +
                                     np.sum(mask_local[:, 0]) + np.sum(mask_local[:, -1]))
                    perimeter = 2 * (h_m + w_m)
                    
                    # HEURISTIQUE : Si plus de 25% du bord est touché, c'est probablement le fond.
                    if border_pixels > (perimeter * 0.25):
                        print(f"   🔄 DÉTECTION FOND (Bordures touchées). Inversion du masque !")
                        mask_local = ~mask_local # Inversion booléenne (NOT)
                        is_inverted = True
                        
                        # Vérification post-inversion : si le résultat est vide ou minuscule, on abandonne ce point
                        if np.sum(mask_local) < 50: # Trop petit après inversion
                             print("   ⚠️ Masque inversé trop petit ou vide. Abandon.")
                             mask_local = np.zeros_like(mask_local)

                    # ==================================================

                    estimated_area = np.sum(mask_local)
                    ratio = estimated_area / full_robot_area
                else:
                    ratio = 0.0

            # Génération image debug
            debug_view = image_crop.convert("RGBA")
            draw_d = ImageDraw.Draw(debug_view)
            draw_d.rectangle(box_pos, outline="#00ff00", width=2)
            for nb in neg_boxes: draw_d.rectangle(nb, outline="#ff0000", width=2)
            
            if mask_local is not None and np.any(mask_local):
                # Si inversé, on affiche en JAUNE pour différencier, sinon ROUGE
                color = (255, 255, 0) if is_inverted else (255, 0, 0)
                debug_view = self._get_overlay(debug_view, mask_local, color, 0.5)
                
            status = "FAIL"
            if 0.005 < ratio < 0.35: status = "CANDIDAT"
            inv_tag = "_INVERTED" if is_inverted else ""
            
            debug_name = f"Test_P{idx}_Ratio_{ratio:.2f}_{status}{inv_tag}.png"
            debug_view.save(os.path.join(self.debug_dir, debug_name))
            print(f"   👉 Ratio: {ratio:.2f} (Inversé: {is_inverted}) -> Saved")
            
            # Sélection
            if status == "CANDIDAT":
                m_reproj = np.zeros((H, W), dtype=bool)
                # Resize si nécessaire avant reprojection
                if mask_local.shape != (hc, wc):
                     tmp = Image.fromarray(mask_local.astype(np.uint8)).resize((wc,hc), Image.NEAREST)
                     mask_local = np.array(tmp) > 0
                m_reproj[y1:y1+hc, x1:x1+wc] = mask_local
                current_area = np.sum(m_reproj)
                if current_area < min_area:
                    min_area = current_area
                    best_mask_full = m_reproj

        # Résultat Final
        final_ov = image_full.convert("RGBA")
        final_ov = self._get_overlay(final_ov, robot_mask, (0,0,255), 0.15)
        if best_mask_full is not None:
            final_ov = self._get_overlay(final_ov, best_mask_full, (0,255,0), 0.8) # Vert pour le résultat final
            print("\n✅ Effecteur isolé.")
        else:
            print("\n⚠️ Aucun effecteur détecté.")

        save_path = os.path.join(self.output_dir, "resultat_final_debugged.png")
        final_ov.save(save_path)
        print(f"🎉 Image finale : {save_path}")

if __name__ == "__main__":
    try:
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("vision_processing")
        image_path = os.path.join(package_path, "scripts", "images_trajectory", "00001.jpg")
    except:
        image_path = "images_trajectory/00001.jpg"
    
    seg = SegmentEndEffectorDebug()
    seg.process_end_effector(image_path)