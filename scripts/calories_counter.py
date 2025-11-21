import torch
import numpy as np
from PIL import Image
import os
import cv2
import ast
import sys
import requests
import base64

# --- IA GOOGLE (Juste pour identifier le nom de l'aliment) ---
from google import genai

# --- IA META (SAM 3) ---
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- IA PROFONDEUR ---
from transformers import pipeline

# ==========================================
# 1. CLIENT FATSECRET (Gestion de l'API)
# ==========================================
class FatSecretClient:
    def __init__(self):
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.token = None
        
        if not self.client_id or not self.client_secret:
            print("⚠️ FatSecret credentials manquants. Les calories seront à 0.")

    def get_token(self):
        """Authentification OAuth 2.0 pour obtenir un token"""
        url = "https://oauth.fatsecret.com/connect/token"
        data = {'grant_type': 'client_credentials', 'scope': 'basic'}
        auth = (self.client_id, self.client_secret)
        try:
            response = requests.post(url, data=data, auth=auth)
            if response.status_code == 200:
                self.token = response.json()['access_token']
                return True
        except Exception as e:
            print(f"❌ Erreur Auth FatSecret: {e}")
        return False

    def search_food(self, query):
        """Cherche un aliment et renvoie ses macros (Kcal/100g)"""
        if not self.token and not self.get_token(): return None
        
        url = "https://platform.fatsecret.com/rest/server.api"
        headers = {'Authorization': f'Bearer {self.token}'}
        params = {
            'method': 'foods.search',
            'search_expression': query,
            'format': 'json',
            'max_results': 1
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            data = response.json()
            
            if 'foods' in data and 'food' in data['foods']:
                # On prend le premier résultat
                food_item = data['foods']['food'][0]
                description = food_item.get('food_description', '')
                
                # Le format est souvent: "Per 100g - Calories: 120kcal | Fat: 2g..."
                # On va essayer de parser ça grossièrement ou utiliser une densité standard
                # Note: L'API gratuite donne le résumé string. L'API payante donne les détails.
                # On va extraire "Calories: XXXkcal"
                import re
                cal_match = re.search(r'Calories:\s*(\d+)', description)
                kcal = int(cal_match.group(1)) if cal_match else 100
                
                # On extrait aussi la portion si possible, sinon on normalise à 100g
                # Pour simplifier ici, on retourne la valeur brute trouvée
                # ATTENTION: FatSecret retourne souvent par "Serving", pas par 100g.
                # Pour une vraie app, il faut appeler 'food.get.v2' pour avoir les détails par 100g.
                # Ici, on va faire une approximation : kcal renvoyé / 100g (densité standard)
                
                return kcal 
        except Exception as e:
            print(f"⚠️ Erreur recherche FatSecret '{query}': {e}")
        
        return 150 # Valeur par défaut moyenne

# ==========================================
# 2. SCANNER 3D (Main Class)
# ==========================================
class NutrientScannerV4:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 [INIT] Démarrage sur {self.device}...")

        # APIs
        self.google_key = os.getenv("GOOGLE_API_KEY")
        self.gemini_client = genai.Client(api_key=self.google_key)
        self.fatsecret = FatSecretClient()

        # SAM 3
        print("⏳ Chargement SAM 3...")
        self.sam_model = build_sam3_image_model()
        if hasattr(self.sam_model, "to"): self.sam_model.to(self.device)
        self.sam_processor = Sam3Processor(self.sam_model)

        # Depth Anything V2
        print("⏳ Chargement Depth V2...")
        self.depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=0 if self.device=="cuda" else -1)

        # --- CONFIGURATION PHYSIQUE (A MESURER CHEZ VOUS !) ---
        self.REAL_PLATE_DIAMETER_CM = 22.5  # Diamètre intérieur
        self.REAL_PLATE_DEPTH_CM = 1.8      # Profondeur (Bord haut -> Fond)
        
        print("✅ SYSTÈME V4 PRÊT.")

    def force_2d(self, mask):
        if isinstance(mask, torch.Tensor): mask = mask.detach().cpu().numpy()
        while mask.ndim > 2:
            if mask.shape[0] == 1: mask = mask.squeeze(0)
            elif mask.shape[-1] == 1: mask = mask.squeeze(-1)
            else: mask = np.any(mask, axis=0)
        return mask.astype(np.uint8)

    def analyze_plate(self, image_path):
        if not os.path.exists(image_path): return
        image_pil = Image.open(image_path).convert("RGB")
        
        # 1. Depth Map
        print("📐 Calcul de la profondeur...")
        depth_res = self.depth_pipe(image_pil)
        depth_map = np.array(depth_res["predicted_depth"])
        depth_map = cv2.resize(depth_map, image_pil.size, interpolation=cv2.INTER_CUBIC)
        
        # Normalisation 0-1
        d_min, d_max = depth_map.min(), depth_map.max()
        depth_map_norm = (depth_map - d_min) / (d_max - d_min + 1e-6)

        # 2. SAM & Calibration
        inference_state = self.sam_processor.set_image(image_pil)
        print("📏 Calibration sur l'assiette...")
        px_per_cm, z_scale_factor, plate_mask = self.calibrate_on_plate(inference_state, depth_map_norm)
        
        if px_per_cm == 0:
            print("⚠️ Calibration échouée. Mode dégradé.")
            px_per_cm = 40.0
            z_scale_factor = 10.0

        # 3. Gemini
        print("🤖 Gemini identifie les aliments...")
        food_names = self.get_food_names_only(image_pil)
        
        total_calories = 0
        print("-" * 60)
        
        for name in food_names:
            kcal_per_100g = self.fatsecret.search_food(name)
            phys_density = 0.85 
            
            try:
                output = self.sam_processor.set_text_prompt(state=inference_state, prompt=name)
                masks = output["masks"]
                if masks is None: continue
                
                mask_2d = self.force_2d(masks)
                if np.sum(mask_2d) == 0: continue

                # A. Aire
                area_px = np.sum(mask_2d > 0)
                area_cm2 = area_px / (px_per_cm ** 2)

                # B. Volume (CORRECTION ICI)
                # On cherche le sol LOCALEMENT autour de l'aliment
                # Création d'un anneau (Dilatation - Masque Original)
                kernel_size = int(20 * (px_per_cm / 40.0)) # Anneau proportionnel à la taille
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                dilated_mask = cv2.dilate(mask_2d, kernel, iterations=1)
                local_ring_mask = (dilated_mask > 0) & (mask_2d == 0)
                
                # On s'assure de rester dans l'assiette si possible
                if plate_mask is not None:
                    valid_ring = local_ring_mask & (plate_mask > 0)
                    if np.sum(valid_ring) > 50: # Si l'intersection est valide
                        local_ring_mask = valid_ring

                # Calcul de la référence du sol (Floor)
                if np.sum(local_ring_mask) > 0:
                    ring_depth_values = depth_map_norm[local_ring_mask]
                    # ASTUCE : On prend le 10e centile (les 10% les plus sombres/loins)
                    # Cela évite de prendre le haut d'un autre aliment qui toucherait celui-ci
                    plate_floor_val = np.percentile(ring_depth_values, 10)
                else:
                    # Fallback : valeur min globale du masque assiette
                    plate_floor_val = np.min(depth_map_norm[plate_mask > 0]) if plate_mask is not None else 0

                # Extraction des hauteurs
                food_vals = depth_map_norm[mask_2d > 0]
                
                # Hauteur relative brute
                heights_rel = food_vals - plate_floor_val
                
                # DEBUG : Afficher les valeurs pour comprendre
                avg_food_depth = np.mean(food_vals)
                diff_moyenne = avg_food_depth - plate_floor_val
                # print(f"   [DEBUG {name}] Sol: {plate_floor_val:.3f} | Food(Moy): {avg_food_depth:.3f} | Diff: {diff_moyenne:.3f}")

                # Filtrage des valeurs négatives (bruit)
                heights_rel = np.maximum(heights_rel, 0)
                
                # Intégration
                vol_index = np.sum(heights_rel)
                
                # Conversion finale CM3
                volume_cm3 = vol_index * (z_scale_factor / (px_per_cm ** 2))

                # C. Calories
                weight_g = volume_cm3 * phys_density
                calories = (weight_g / 100) * kcal_per_100g

                print(f"🍔 {name.upper()}")
                print(f"   Aire: {area_cm2:.1f} cm²")
                print(f"   Volume: {volume_cm3:.1f} cm³ (H.Moyenne: {(volume_cm3/area_cm2 if area_cm2>0 else 0):.2f} cm)")
                print(f"   Poids: {int(weight_g)}g | Cal: {int(calories)} kcal")
                
                total_calories += calories
                
            except Exception as e:
                print(f"❌ Erreur {name}: {e}")

        print("=" * 60)
        print(f"🔥 TOTAL REPAS: {int(total_calories)} kcal")

    def calibrate_on_plate(self, inference_state, depth_map_norm):
        """
        Utilise l'assiette pour calibrer X, Y (Diamètre) ET Z (Profondeur).
        """
        prompts = ["plate", "dish", "bowl"]
        
        for p in prompts:
            try:
                output = self.sam_processor.set_text_prompt(state=inference_state, prompt=p)
                masks = output["masks"]
                if masks is None: continue
                
                mask = self.force_2d(masks)
                if np.sum(mask) < 5000: continue # Trop petit

                # --- 1. CALIBRATION 2D (PIXELS PER CM) ---
                # On assume que le masque couvre l'intérieur de l'assiette
                area_px = np.sum(mask > 0)
                diameter_px = 2 * np.sqrt(area_px / np.pi)
                px_per_cm = diameter_px / self.REAL_PLATE_DIAMETER_CM
                print(f"   📏 Diamètre assiette: {diameter_px:.0f}px -> 1cm = {px_per_cm:.1f}px")

                # --- 2. CALIBRATION 3D (Z-SCALE) ---
                # On doit trouver la valeur de profondeur du "Bord" (Haut) et du "Fond" (Bas)
                
                # A. Le Fond : Le centre du masque
                # On érode fortement pour garder le coeur de l'assiette
                kernel = np.ones((20,20), np.uint8)
                center_mask = cv2.erode(mask, kernel, iterations=5)
                if np.sum(center_mask) == 0: center_mask = mask # Fallback
                
                # Valeur médiane du fond (généralement plus sombre/loin = valeur plus basse dans certaines maps, ou inverse)
                # Depth Anything: Blanc = Proche (Bord), Noir = Loin (Fond)
                # Donc Bord > Fond
                val_floor = np.median(depth_map_norm[center_mask > 0])
                
                # B. Le Bord : La périphérie du masque
                # Dilater - Eroder = Anneau du bord
                dilated = cv2.dilate(mask, kernel, iterations=2)
                rim_mask = (dilated > 0) & (mask == 0)
                
                if np.sum(rim_mask) > 0:
                    val_rim = np.median(depth_map_norm[rim_mask])
                else:
                    # Fallback: max value dans le masque global (supposant que le bord est le point le plus haut)
                    val_rim = np.max(depth_map_norm[mask > 0])

                delta_z_norm = abs(val_rim - val_floor)
                
                if delta_z_norm < 0.05:
                    print("   ⚠️ Contraste profondeur trop faible sur l'assiette. Z-Scale par défaut.")
                    z_scale_factor = 10.0 # Valeur arbitraire de secours
                else:
                    # Si la différence normée (0.2) correspond à 2.5cm
                    # Alors 1.0 (full range) correspond à 2.5 / 0.2 = 12.5cm
                    z_scale_factor = self.REAL_PLATE_DEPTH_CM / delta_z_norm
                    print(f"   📐 Profondeur assiette (Norm): {delta_z_norm:.3f} -> Z-Factor: {z_scale_factor:.1f}")

                return px_per_cm, z_scale_factor, mask

            except Exception as e:
                print(f"Err Calib: {e}")
                continue
        
        return 0, 0, None

    def get_food_names_only(self, image_pil):
        prompt = "Identify distinct food items. Return ONLY a Python list of strings. Ex: ['rice', 'steak']."
        try:
            res = self.gemini_client.models.generate_content(model="gemini-2.5-flash", contents=[image_pil, prompt])
            txt = res.text.strip().replace("```python", "").replace("```", "").strip()
            if "[" in txt: txt = txt[txt.find("["):txt.rfind("]")+1]
            return ast.literal_eval(txt)
        except: return []

if __name__ == "__main__":
    IMG = "images/Filou.jpeg"
    if os.path.exists(IMG):
        scanner = NutrientScannerV4()
        scanner.analyze_plate(IMG)
    else:
        print("Image introuvable.")