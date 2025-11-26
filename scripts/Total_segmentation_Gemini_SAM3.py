import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
import ast
from google import genai
import argparse

# Imports natifs SAM 3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class FoodSegmenterTotal:
    def __init__(self, google_api_key):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⏳ Chargement du modèle SAM 3 (Native) sur {self.device}...")
        
        self.model = build_sam3_image_model()
        if hasattr(self.model, "to"):
            self.model.to(self.device)
            
        self.processor = Sam3Processor(self.model)
        print("✅ Modèle chargé avec succès.")

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
            
    def process_food_list(self, image_path):
        # --- Config des chemins de sortie ---
        output_dir = os.path.expanduser("~/Github/src/vision_processing/scripts")
        overlay_filename = os.path.join(output_dir, "resultat_overlay.png")
        rainbow_filename = os.path.join(output_dir, "resultat_rainbow_map.png")

        if not os.path.exists(image_path):
            print(f"❌ Erreur : L'image {image_path} n'existe pas.")
            return

        print(f"📸 Chargement de l'image : {image_path}")
        image_pil = Image.open(image_path).convert("RGB")
        
        # Images de sortie
        final_overlay_image = image_pil.convert("RGBA")
        rainbow_map_image = Image.new("RGB", image_pil.size, (0, 0, 0))
        
        # Initialisation SAM 3
        inference_state = self.processor.set_image(image_pil)
        start_gemini = time.time()      
        self.food_list = self.get_food_list_from_gemini(image_pil)
        # self.food_list = ['sausage slices', 'green beans', 'mashed potatoes']
        end_gemini = time.time()
        print(f"🤖 [GEMINI] Temps écoulé : {end_gemini - start_gemini:.2f} secondes")

        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / (len(self.food_list) + 1))[:3] for i in range(len(self.food_list))]

        # Liste pour stocker les résultats détaillés
        detection_results = []
        
        print(f"🍽️  Analyse de la liste : {self.food_list}")
        print("-" * 40)
        start_sam = time.time() 
        for i, food_name in enumerate(self.food_list):
            item_start_time = time.time()
            color_rgb = tuple(int(c * 255) for c in colors[i])

            print(f"   👉 Recherche de : '{food_name}' (Couleur ID: {color_rgb})")

            try:
                output = self.processor.set_text_prompt(state=inference_state, prompt=food_name)
                masks = output["masks"]
                
                if masks is not None:
                    # Conversion Tensor -> Numpy
                    if isinstance(masks, torch.Tensor):
                        masks = masks.detach().cpu().numpy()
                    
                    print(f"      📊 Shape des masques : {masks.shape}")
                    
                    # Gestion des dimensions (N, 1, H, W) -> (N, H, W)
                    if masks.ndim == 4 and masks.shape[1] == 1:
                        masks = masks.squeeze(1)
                    elif masks.ndim == 4:
                         # Cas rare (N, C, H, W) avec C > 1 ? On prend le premier canal ?
                         masks = masks[:, 0, :, :]
                    
                    # --- SECURITY CHECK / FALLBACK ---
                    if masks.size == 0 or not np.any(masks):
                        print(f"      ⚠️ Aucun pixel trouvé pour '{food_name}'")
                        
                        found_fallback = False
                        if " " in food_name:
                            print(f"      🔄 Tentative avec sous-parties du nom...")
                            for sub_word in food_name.split():
                                print(f"         👉 Essai sous-partie : '{sub_word}'")
                                try:
                                    output_sub = self.processor.set_text_prompt(state=inference_state, prompt=sub_word)
                                    masks_sub = output_sub["masks"]
                                    
                                    if masks_sub is not None:
                                        if isinstance(masks_sub, torch.Tensor):
                                            masks_sub = masks_sub.detach().cpu().numpy()
                                        
                                        if masks_sub.size > 0 and np.any(masks_sub):
                                            print(f"         ✅ Trouvé avec '{sub_word}'!")
                                            
                                            # Gestion des dimensions pour le fallback (N, 1, H, W) -> (N, H, W)
                                            if masks_sub.ndim == 4 and masks_sub.shape[1] == 1:
                                                masks_sub = masks_sub.squeeze(1)
                                            elif masks_sub.ndim == 4:
                                                masks_sub = masks_sub[:, 0, :, :]
                                                
                                            masks = masks_sub
                                            found_fallback = True
                                            break
                                except Exception as e_sub:
                                    print(f"         ❌ Erreur sur sous-partie '{sub_word}': {e_sub}")

                        if not found_fallback:
                            continue

                    # --- ANALYSE DES MASQUES ---
                    # Si le tableau est en 3D (N, H, W), len(masks) donne N.
                    # Si le tableau est en 2D (H, W), c'est qu'il y a 1 seul masque.
                    current_centroids = []
                    current_areas = []
                    
                    if masks.ndim > 2:
                        count = len(masks)
                        # Calcul des centroïdes pour chaque masque individuel
                        for m_idx in range(count):
                            single_mask = masks[m_idx]
                            if np.any(single_mask):
                                coords = np.argwhere(single_mask)
                                # argwhere retourne (row, col) -> (y, x)
                                # On veut souvent (x, y) pour l'affichage, mais gardons (row, col) ou précisons.
                                # Standard image coordinates: x (width), y (height).
                                # Numpy: axis 0 is y, axis 1 is x.
                                y_center, x_center = coords.mean(axis=0)
                                current_centroids.append([float(x_center), float(y_center)])
                                current_areas.append(int(np.sum(single_mask)))
                    else:
                        count = 1
                        if np.any(masks):
                            coords = np.argwhere(masks)
                            y_center, x_center = coords.mean(axis=0)
                            current_centroids.append([float(x_center), float(y_center)])
                            current_areas.append(int(np.sum(masks)))
                    
                    print(f"      🔢 Nombre de masques (len) : {count}")
                    print(f"      🎯 Centroïdes : {current_centroids}")

                    # --- Fusion pour l'affichage ---
                    if masks.ndim > 2:
                        masks_reshaped = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
                        combined_mask = np.any(masks_reshaped > 0, axis=0)
                    else:
                        combined_mask = masks > 0

                    # --- Application sur les images ---
                    # 1. Overlay Transparent
                    final_overlay_image = self.overlay_mask_transparent(final_overlay_image, combined_mask, color_rgb)
                    
                    # 2. Rainbow Map (Solide)
                    self.paint_rainbow_mask(rainbow_map_image, combined_mask, color_rgb)
                    
                    item_time = time.time() - item_start_time
                    print(f"      ✅ Masques appliqués.")
                    print(f"      ⏱️ Temps écoulé : {item_time:.2f} secondes")
                    print("-" * 20)
                    
                    # Ajout aux résultats
                    detection_results.append({
                        "name": food_name,
                        "count": count,
                        "centroids": current_centroids,
                        "areas": current_areas,
                        "computation_time": item_time
                    })

                else:
                    print(f"      ⚠️ Retour vide pour '{food_name}'")

            except Exception as e:
                print(f"      ❌ Erreur critique sur '{food_name}': {e}")

        # Sauvegarde locale (optionnel, mais utile pour debug)
        final_overlay_image.save(overlay_filename)
        rainbow_map_image.save(rainbow_filename)
        print(f"\n🎉 Images sauvegardées dans : {output_dir}")
        
        return {
            "success": True,
            "food_items": detection_results,
            "total_computation_time": time.time() - start_sam,
            "overlay_image": final_overlay_image,
            "rainbow_image": rainbow_map_image
        }

    def overlay_mask_transparent(self, image, mask, color_rgb, alpha_val=0.5):
        color_layer = Image.new("RGBA", image.size, color_rgb + (0,))
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        try:
            mask_pil = Image.fromarray(mask_uint8)
        except Exception:
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
            mask_pil = Image.fromarray(mask_uint8) 
        except Exception:
             mask_uint8 = np.squeeze(mask_uint8)
             mask_pil = Image.fromarray(mask_uint8)

        if mask_pil.size != target_image_rgb.size:
            mask_pil = mask_pil.resize(target_image_rgb.size, resample=Image.NEAREST)

        target_image_rgb.paste(solid_color_layer, mask=mask_pil)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation d'images avec SAM 3 et Gemini")
    parser.add_argument("--image", type=str, required=True, help="Chemin de l'image à segmenter")
    args = parser.parse_args()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    segmenter = FoodSegmenterTotal(google_api_key)
    segmenter.process_food_list(args.image)