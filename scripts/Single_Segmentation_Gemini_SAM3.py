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

class FoodSegmenterIndividual:
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
        if not self.api_key or "AIza" not in self.api_key:
             print("⚠️ [GEMINI] Attention: Clé API invalide ou absente.")
        else:
             self.client = genai.Client(api_key=self.api_key)
             print("✅ [GEMINI] Client prêt.")

    def get_food_list_from_gemini(self, image_pil):
        print("🤖 [GEMINI] Analyse de l'image...")
        prompt_text = "Identify distinct food items. Return ONLY a Python list of strings (e.g., ['rice', 'chicken'])."
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
        start_time = time.time()
        
        # Initialisation du résultat
        result_data = {
            "success": False,
            "selected_food": "None",
            "total_pieces": 0,
            "centroid": [0.0, 0.0],
            "selected_piece_area": 0,
            "computation_time": 0.0,
            # NOUVEAUX CHAMPS POUR LES IMAGES NUMPY (OPENCV BGR)
            "selected_overlay_image": None,
            "selected_rainbow_image": None
        }

        if not os.path.exists(image_path):
            print(f"❌ Erreur : Image introuvable {image_path}")
            return result_data

        # Préparation des images de base
        image_pil_rgb = Image.open(image_path).convert("RGB")
        
        # 1. Appel Gemini
        food_list = self.get_food_list_from_gemini(image_pil_rgb)
        print(f"🍽️  Menu détecté : {food_list}")
        
        if not food_list:
            return result_data

        # 2. Init SAM 3
        print("🚀 [SAM3] Initialisation sur cuda...")
        inference_state = self.processor.set_image(image_pil_rgb)
        
        # Variables pour suivre le "Gagnant"
        global_max_area = 0
        winner_food_name = ""
        winner_mask = None      # Le masque booléen du gagnant
        winner_count = 0
        winner_color_rgb = (0,0,0) # La couleur du gagnant

        # Couleurs
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / (len(food_list) + 1))[:3] for i in range(len(food_list))]

        # 3. Boucle de segmentation
        for i, food_name in enumerate(food_list):
            try:
                color_rgb = tuple(int(c * 255) for c in colors[i])
                output = self.processor.set_text_prompt(state=inference_state, prompt=food_name)
                masks = output["masks"]

                if masks is not None:
                    if isinstance(masks, torch.Tensor):
                        masks = masks.detach().cpu().numpy()
                    
                    if masks.size == 0 or not np.any(masks):
                        continue

                    # --- A. Comptage et Fusion ---
                    current_item_count = 0
                    current_item_max_area = 0

                    # Fusionner les masques si nécessaire
                    if masks.ndim > 2:
                        current_item_count = len(masks)
                        masks_reshaped = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
                        combined_mask = np.any(masks_reshaped > 0, axis=0)
                        
                        # Trouver l'aire maximale parmi les morceaux
                        for m_idx in range(masks.shape[0]):
                            current_item_max_area = max(current_item_max_area, np.sum(masks[m_idx]))
                    else:
                        current_item_count = 1
                        combined_mask = masks > 0
                        current_item_max_area = np.sum(masks)

                    # --- B. Logique de sélection pour ROS ---
                    if current_item_max_area > global_max_area:
                        global_max_area = current_item_max_area
                        winner_food_name = food_name
                        winner_mask = combined_mask # On garde le masque combiné du gagnant
                        winner_count = current_item_count
                        winner_color_rgb = color_rgb

            except Exception as e:
                print(f"❌ Erreur sur {food_name}: {e}")

        # 4. Génération des images UNIQUEMENT pour le gagnant
        cx, cy = 0.0, 0.0
        winner_overlay_cv = None
        winner_rainbow_cv = None
        print("🤖 [GEMINI] Analyse de l'image...")
        if winner_mask is not None:
            try:
                # --- A. Calcul du Centroïde ---
                winner_mask_uint8 = winner_mask.astype(np.uint8)
                if winner_mask_uint8.ndim > 2: winner_mask_uint8 = np.squeeze(winner_mask_uint8)
                M = cv2.moments(winner_mask_uint8)
                if M["m00"] != 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])

                # --- B. Génération des Images PIL ---
                # 1. Base Overlay : Image originale en RGBA
                base_overlay_pil = image_pil_rgb.convert("RGBA")
                winner_overlay_pil = self.overlay_mask_transparent(base_overlay_pil, winner_mask, winner_color_rgb)

                # 2. Base Rainbow : Image noire en RGB
                base_rainbow_pil = Image.new("RGB", image_pil_rgb.size, (0, 0, 0))
                self.paint_rainbow_mask(base_rainbow_pil, winner_mask, winner_color_rgb)
                winner_rainbow_pil = base_rainbow_pil

                # --- C. Conversion PIL -> OpenCV (BGR) pour le serveur ---
                # Convertir RGBA -> BGR
                winner_overlay_cv = cv2.cvtColor(np.array(winner_overlay_pil), cv2.COLOR_RGBA2BGR)
                # Convertir RGB -> BGR
                winner_rainbow_cv = cv2.cvtColor(np.array(winner_rainbow_pil), cv2.COLOR_RGB2BGR)

            except Exception as e:
                print(f"⚠️ Erreur génération images gagnant: {e}")

        # 5. Return final
        result_data["success"] = True
        result_data["selected_food"] = winner_food_name
        result_data["total_pieces"] = winner_count
        result_data["centroid"] = [cx, cy]
        result_data["selected_piece_area"] = int(global_max_area)
        result_data["computation_time"] = time.time() - start_time
        # On renvoie les images numpy BGR
        result_data["selected_overlay_image"] = winner_overlay_cv
        result_data["selected_rainbow_image"] = winner_rainbow_cv

        return result_data

    # --- Helper Functions Graphiques (Inchangées) ---
    def overlay_mask_transparent(self, image, mask, color_rgb, alpha_val=0.5):
        color_layer = Image.new("RGBA", image.size, color_rgb + (0,))
        mask_uint8 = (mask * 255).astype(np.uint8)
        try: mask_pil = Image.fromarray(mask_uint8)
        except: mask_uint8 = np.squeeze(mask_uint8); mask_pil = Image.fromarray(mask_uint8)
        if mask_pil.size != image.size: mask_pil = mask_pil.resize(image.size, resample=Image.NEAREST)
        mask_pil = mask_pil.point(lambda x: int(255 * alpha_val) if x > 0 else 0)
        color_layer.putalpha(mask_pil)
        return Image.alpha_composite(image, color_layer)

    def paint_rainbow_mask(self, target_image_rgb, mask, color_rgb):
        solid_color_layer = Image.new("RGB", target_image_rgb.size, color_rgb)
        mask_uint8 = (mask * 255).astype(np.uint8)
        try: mask_pil = Image.fromarray(mask_uint8)
        except: mask_uint8 = np.squeeze(mask_uint8); mask_pil = Image.fromarray(mask_uint8)
        if mask_pil.size != target_image_rgb.size: mask_pil = mask_pil.resize(target_image_rgb.size, resample=Image.NEAREST)
        target_image_rgb.paste(solid_color_layer, mask=mask_pil)

if __name__ == "__main__":
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    segmenter = FoodSegmenterIndividual(GOOGLE_API_KEY)
    test_image_path = "images/Filou.jpeg"  # Remplacez par votre image de test
    result = segmenter.process_image(test_image_path)