import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
import cv2
import ast
from google import genai

# Imports natifs SAM 3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class FoodSegmenterNative:
    def __init__(self, google_api_key):
        # 1. Configuration Hardware
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 [SAM3] Initialisation sur {self.device}...")
        
        # 2. Chargement Modèle SAM 3
        self.model = build_sam3_image_model()
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        self.processor = Sam3Processor(self.model)
        print("✅ [SAM3] Modèle chargé sur GPU.")

        # 3. Configuration Gemini
        self.api_key = google_api_key
        if "AIza" not in self.api_key:
            print("⚠️ [GEMINI] Attention: Clé API invalide ou manquante.")
        self.client = genai.Client(api_key=self.api_key)
        print("✅ [GEMINI] Client prêt.")

    def get_food_list_from_gemini(self, image_pil):
        """ Envoie l'image PIL à Gemini pour obtenir la liste """
        print("🤖 [GEMINI] Analyse de l'image...")
        prompt_text = """
        Identify all the food items in this image. 
        Return strictly a Python list of strings containing the names of the foods.
        Example format: ['rice', 'broccoli', 'chicken']
        Do not write any code block markers (like ```python), just the list.
        Be concise. Use singular nouns (e.g., 'carrot' not 'carrots').
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[image_pil, prompt_text]
            )
            text = response.text.replace("```python", "").replace("```", "").strip()
            food_list = ast.literal_eval(text)
            return food_list if isinstance(food_list, list) else []
        except Exception as e:
            print(f"❌ Erreur Gemini: {e}")
            return []

    def process_image(self, image_path):
        """ 
        Fonction principale.
        1. Segmente TOUT.
        2. Sauvegarde les images de visualisation (Overlay + Rainbow).
        3. Renvoie les infos du "plus gros objet" pour le robot.
        """
        start_time = time.time()
        
        # Initialisation du résultat pour le serveur ROS
        result_data = {
            "success": False,
            "selected_food": "None",
            "total_pieces": 0,       # Nombre de pièces de l'aliment séléctionné
            "centroid": [0.0, 0.0],
            "selected_piece_area": 0,
            "computation_time": 0.0,
            "processed_image": None  # Image OpenCV finale
        }

        if not os.path.exists(image_path):
            print(f"❌ Erreur : Image introuvable {image_path}")
            return result_data

        # --- Préparation des Images ---
        image_pil = Image.open(image_path).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR) # Pour ROS
        
        # Images de visualisation (Sauvegardées sur disque)
        final_overlay_image = image_pil.convert("RGBA")
        rainbow_map_image = Image.new("RGB", image_pil.size, (0, 0, 0))
        
        # 1. Appel Gemini
        start_gemini = time.time()
        food_list = self.get_food_list_from_gemini(image_pil)
        print("="*60)
        print(f"🍽️  Menu détecté : {food_list}")
        print(f"⏱️  Temps Gemini: {time.time() - start_gemini:.2f}s")
        print("="*60)
        
        if not food_list:
            return result_data

        # 2. Init SAM 3
        inference_state = self.processor.set_image(image_pil)
        
        # Variables pour suivre le "Gagnant" (le plus gros aliment pour le robot)
        global_max_area = 0
        winner_food_name = ""
        winner_mask = None
        winner_count = 0
        
        # Couleurs
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / (len(food_list) + 1))[:3] for i in range(len(food_list))]

        # 3. Boucle de segmentation
        for i, food_name in enumerate(food_list):
            try:
                start = time.time()
                color_rgb = tuple(int(c * 255) for c in colors[i])
                output = self.processor.set_text_prompt(state=inference_state, prompt=food_name)
                masks = output["masks"]

                if masks is not None:
                    if isinstance(masks, torch.Tensor):
                        masks = masks.detach().cpu().numpy()
                    
                    if masks.size == 0 or not np.any(masks):
                        print(f"   ⚠️  '{food_name}': Rien trouvé.")
                        continue

                    # --- A. Comptage et Fusion ---
                    current_item_count = 0
                    current_item_max_area = 0
                    current_item_best_mask = None

                    # Cas Multimask (N, H, W)
                    if masks.ndim > 2:
                        current_item_count = len(masks)
                        # Fusionner pour l'affichage global
                        masks_reshaped = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
                        combined_mask = np.any(masks_reshaped > 0, axis=0)
                        
                        # Trouver le plus gros morceau de cet aliment spécifique
                        for m_idx in range(masks.shape[0]):
                            single_area = np.sum(masks[m_idx])
                            if single_area > current_item_max_area:
                                current_item_max_area = single_area
                                current_item_best_mask = masks[m_idx]

                    # Cas Monomask (H, W)
                    else:
                        current_item_count = 1
                        combined_mask = masks > 0
                        current_item_max_area = np.sum(masks)
                        current_item_best_mask = masks

                    print(f"   👉 '{food_name}' : {current_item_count} morceaux détectés.")
                    print(f"      ⏱️  Temps traitement: {time.time() - start:.2f}s")
                    # --- B. Mise à jour des Images Visuelles ---
                    # Overlay Transparent
                    final_overlay_image = self.overlay_mask_transparent(final_overlay_image, combined_mask, color_rgb)
                    # Rainbow Map (Solide)
                    self.paint_rainbow_mask(rainbow_map_image, combined_mask, color_rgb)

                    # --- C. Logique de sélection pour ROS ---
                    # Si cet aliment a un morceau plus gros que le "champion" actuel, il devient la cible
                    if current_item_max_area > global_max_area:
                        global_max_area = current_item_max_area
                        winner_food_name = food_name
                        winner_mask = current_item_best_mask
                        winner_count = current_item_count
                        
                        # On dessine aussi les contours sur l'image OpenCV (pour le retour ROS)
                        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                        mask_uint8 = combined_mask.astype(np.uint8)
                        if mask_uint8.ndim > 2: mask_uint8 = np.squeeze(mask_uint8)
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(image_cv, contours, -1, color_bgr, 2)

            except Exception as e:
                print(f"❌ Erreur sur {food_name}: {e}")

        # 4. Sauvegarde sur disque (Fonctionnalité demandée)
        output_dir = os.path.dirname(os.path.abspath(__file__))
        overlay_path = os.path.join(output_dir, "resultat_overlay.png")
        rainbow_path = os.path.join(output_dir, "resultat_rainbow_map.png")
        
        final_overlay_image.save(overlay_path)
        rainbow_map_image.save(rainbow_path)
        print("-" * 60)
        print(f"💾 Images sauvegardées :\n   1. {overlay_path}\n   2. {rainbow_path}")

        # 5. Calcul du Centroïde FINAL (Uniquement pour le gagnant)
        cx, cy = 0.0, 0.0
        if winner_mask is not None:
            try:
                winner_mask_uint8 = winner_mask.astype(np.uint8)
                winner_mask_uint8 = np.squeeze(winner_mask_uint8) # Correction dims
                
                M = cv2.moments(winner_mask_uint8)
                if M["m00"] != 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])
                    # Marquer le centre sur l'image ROS
                    cv2.circle(image_cv, (int(cx), int(cy)), 10, (255, 255, 255), -1)
                    cv2.circle(image_cv, (int(cx), int(cy)), 6, (0, 0, 255), -1)
            except Exception as e:
                print(f"⚠️ Erreur moments: {e}")

        # 6. Return final pour ROS
        result_data["success"] = True
        result_data["selected_food"] = winner_food_name
        result_data["total_pieces"] = winner_count
        result_data["centroid"] = [cx, cy]
        result_data["selected_piece_area"] = int(global_max_area)
        result_data["computation_time"] = time.time() - start_time
        result_data["processed_image"] = image_cv # BGR pour ROS

        return result_data

    # --- Helper Functions Graphiques ---
    def overlay_mask_transparent(self, image, mask, color_rgb, alpha_val=0.5):
        color_layer = Image.new("RGBA", image.size, color_rgb + (0,))
        mask_uint8 = (mask * 255).astype(np.uint8)
        try:
            mask_pil = Image.fromarray(mask_uint8)
        except:
            mask_uint8 = np.squeeze(mask_uint8)
            mask_pil = Image.fromarray(mask_uint8)

        if mask_pil.size != image.size:
            mask_pil = mask_pil.resize(image.size, resample=Image.NEAREST)
            
        mask_pil = mask_pil.point(lambda x: int(255 * alpha_val) if x > 0 else 0)
        color_layer.putalpha(mask_pil)
        return Image.alpha_composite(image, color_layer)

    def paint_rainbow_mask(self, target_image_rgb, mask, color_rgb):
        solid_color_layer = Image.new("RGB", target_image_rgb.size, color_rgb)
        mask_uint8 = (mask * 255).astype(np.uint8)
        try:
            mask_pil = Image.fromarray(mask_uint8) # Mode 'L' auto-détecté
        except:
             mask_uint8 = np.squeeze(mask_uint8)
             mask_pil = Image.fromarray(mask_uint8)

        if mask_pil.size != target_image_rgb.size:
            mask_pil = mask_pil.resize(target_image_rgb.size, resample=Image.NEAREST)

        target_image_rgb.paste(solid_color_layer, mask=mask_pil)


# --- TEST (Si lancé directement) ---
if __name__ == "__main__":
    API_KEY = "AIzaSyBiI8ij7bP_P0CUHkxv8W_cjMOCNa24-7I"
    IMAGE = "images/Brocoli.jpeg"
    
    if "AIza" in API_KEY:
        seg = FoodSegmenterNative(API_KEY)
        # Cela va générer les images ET renvoyer les données
        res = seg.process_image(IMAGE)
        
        print(f"\n🤖 [ROS DATA] Selected: {res['selected_food']} | Area: {res['selected_piece_area']} | Centroid: {res['centroid']}")
    else:
        print("❌ Configurez la clé API en bas du fichier.")