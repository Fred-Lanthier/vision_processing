import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
import rospkg

# Imports natifs SAM 3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class FoodSegmenterNative:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⏳ Chargement du modèle SAM 3 (Native) sur {self.device}...")
        
        self.model = build_sam3_image_model()
        if hasattr(self.model, "to"):
            self.model.to(self.device)
            
        self.processor = Sam3Processor(self.model, confidence_threshold=0.25)
        print("✅ Modèle chargé avec succès.")

    def process_food_list(self, image_path, food_list):
        # --- Config des chemins de sortie ---
        output_dir = os.path.expanduser("~/Github/src/vision_processing/src/vision_processing/vision_segmentation/images")
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

        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / (len(food_list) + 1))[:3] for i in range(len(food_list))]

        print(f"🍽️  Analyse de la liste : {food_list}")
        print("-" * 40)

        for i, food_name in enumerate(food_list):
            start = time.time()
            color_rgb = tuple(int(c * 255) for c in colors[i])

            print(f"   👉 Recherche de : '{food_name}' (Couleur ID: {color_rgb})")

            try:
                output = self.processor.set_text_prompt(state=inference_state, prompt=food_name)
                print(output["scores"])
                masks = output["masks"]
                
                if masks is not None:
                    # Conversion Tensor -> Numpy
                    if isinstance(masks, torch.Tensor):
                        masks = masks.detach().cpu().numpy()
                    
                    if masks.size == 0 or not np.any(masks):
                        print(f"      ⚠️ Aucun pixel trouvé pour '{food_name}'")
                        continue

                    # --- COMPTAGE SUPER SIMPLE ---
                    # Si le tableau est en 3D (N, H, W), len(masks) donne N.
                    # Si le tableau est en 2D (H, W), c'est qu'il y a 1 seul masque.
                    if masks.ndim > 2:
                        count = len(masks) # C'est ça que tu voulais
                    else:
                        count = 1
                    
                    print(f"      🔢 Nombre de masques (len) : {count}")

                    # --- Fusion pour l'affichage ---
                    # On doit quand même combiner les masques pour l'affichage visuel
                    if masks.ndim > 2:
                        # On aplatit pour l'image (N, H, W) -> (H, W)
                        masks_reshaped = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
                        combined_mask = np.any(masks_reshaped > 0, axis=0)
                    else:
                        combined_mask = masks > 0

                    # --- Application sur les images ---
                    # 1. Overlay Transparent
                    final_overlay_image = self.overlay_mask_transparent(final_overlay_image, combined_mask, color_rgb)
                    
                    # 2. Rainbow Map (Solide)
                    self.paint_rainbow_mask(rainbow_map_image, combined_mask, color_rgb)
                    
                    print(f"      ✅ Masques appliqués.")
                    print(f"      ⏱️ Temps écoulé : {time.time() - start:.2f} secondes")
                    print("-" * 20)

                else:
                    print(f"      ⚠️ Retour vide pour '{food_name}'")

            except Exception as e:
                print(f"      ❌ Erreur critique sur '{food_name}': {e}")

        # Sauvegarde
        final_overlay_image.save(overlay_filename)
        rainbow_map_image.save(rainbow_filename)
        print(f"\n🎉 Images sauvegardées dans : {output_dir}")

    def overlay_mask_transparent(self, image, mask, color_rgb, alpha_val=0.5):
        color_layer = Image.new("RGBA", image.size, color_rgb + (0,))
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        try:
            mask_pil = Image.fromarray(mask_uint8, mode='L')
        except Exception:
            mask_uint8 = np.squeeze(mask_uint8)
            mask_pil = Image.fromarray(mask_uint8, mode='L')

        if mask_pil.size != image.size:
            mask_pil = mask_pil.resize(image.size, resample=Image.NEAREST)
            
        mask_pil = mask_pil.point(lambda x: int(255 * alpha_val) if x > 0 else 0)
        color_layer.putalpha(mask_pil)
        return Image.alpha_composite(image, color_layer)

    def paint_rainbow_mask(self, target_image_rgb, mask, color_rgb):
        solid_color_layer = Image.new("RGB", target_image_rgb.size, color_rgb)
        mask_uint8 = (mask * 255).astype(np.uint8)
        try:
            mask_pil = Image.fromarray(mask_uint8, mode='L') 
        except Exception:
             mask_uint8 = np.squeeze(mask_uint8)
             mask_pil = Image.fromarray(mask_uint8, mode='L')

        if mask_pil.size != target_image_rgb.size:
            mask_pil = mask_pil.resize(target_image_rgb.size, resample=Image.NEAREST)

        target_image_rgb.paste(solid_color_layer, mask=mask_pil)

if __name__ == "__main__":
    rospkg = rospkg.RosPack()
    pkg_path = rospkg.get_path("vision_processing")
    IMAGE_PATH = os.path.join(pkg_path, "resources/images_food/Penne.jpeg") 
    LISTE_ALIMENTS = ["sliced tomato", "single penne pasta"]
    
    segmenter = FoodSegmenterNative()
    segmenter.process_food_list(IMAGE_PATH, LISTE_ALIMENTS)