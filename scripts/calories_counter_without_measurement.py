import torch
import numpy as np
from PIL import Image
import os
import cv2
import ast
import sys
import requests
import base64
from dotenv import load_dotenv
import re

# --- IA GOOGLE ---
from google import genai

# --- IA META (SAM 3) ---
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- IA PROFONDEUR ---
from transformers import pipeline

load_dotenv()

# ==========================================
# 1. CLIENT FATSECRET (AVEC MACROS)
# ==========================================
class FatSecretClient:
    def __init__(self):
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.token = None
        
        if not self.client_id or not self.client_secret:
            print("⚠️ FatSecret credentials manquants. Les calories seront à 0.")

    def get_token(self):
        """Authentification OAuth 2.0"""
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
        """
        Cherche un aliment et renvoie un dictionnaire complet de macros pour 100g.
        Retourne: {'kcal': int, 'protein': float, 'carbs': float, 'fat': float}
        """
        if not self.token and not self.get_token(): 
            return self._default_macros()
        
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
                food_item = data['foods']['food'][0]
                description = food_item.get('food_description', '')
                
                # Format typique: "Per 100g - Calories: 120kcal | Fat: 2.50g | Carbs: 15.00g | Protein: 5.00g"
                # Extraction via Regex
                cal_match = re.search(r'Calories:\s*(\d+)', description)
                fat_match = re.search(r'Fat:\s*([\d\.]+)g', description)
                carb_match = re.search(r'Carbs:\s*([\d\.]+)g', description)
                prot_match = re.search(r'Protein:\s*([\d\.]+)g', description)
                
                macros = {
                    "kcal": int(cal_match.group(1)) if cal_match else 0,
                    "fat": float(fat_match.group(1)) if fat_match else 0.0,
                    "carbs": float(carb_match.group(1)) if carb_match else 0.0,
                    "protein": float(prot_match.group(1)) if prot_match else 0.0
                }
                
                print(f"      ✅ FatSecret '{query}': {macros['kcal']}kcal | P:{macros['protein']}g | G:{macros['carbs']}g | L:{macros['fat']}g")
                return macros
                
        except Exception as e:
            print(f"⚠️ Erreur recherche FatSecret '{query}': {e}")
        
        return self._default_macros()

    def _default_macros(self):
        return {"kcal": 100, "fat": 5.0, "carbs": 10.0, "protein": 5.0}

# ==========================================
# 2. SCANNER 3D (Modifié pour le bilan complet)
# ==========================================
class NutrientScannerV5:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 [INIT] Démarrage sur {self.device}...")

        self.google_key = os.getenv("GOOGLE_API_KEY")
        self.gemini_client = genai.Client(api_key=self.google_key)
        self.fatsecret = FatSecretClient()

        print("⏳ Chargement Modèles...")
        self.sam_model = build_sam3_image_model()
        if hasattr(self.sam_model, "to"): self.sam_model.to(self.device)
        self.sam_processor = Sam3Processor(self.sam_model)

        self.depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0 if self.device=="cuda" else -1)

        self.REF_REAL_DIAMETER_CM = 2.80 
        self.REF_NAME = "coin"

    def force_2d(self, mask):
        if isinstance(mask, torch.Tensor): mask = mask.detach().cpu().numpy()
        while mask.ndim > 2:
            if mask.shape[0] == 1: mask = mask.squeeze(0)
            elif mask.shape[-1] == 1: mask = mask.squeeze(-1)
            else: mask = np.any(mask, axis=0)
        return mask.astype(np.uint8)

    def analyze_plate(self, image_path):
        if not os.path.exists(image_path): return {"error": "Image introuvable"}
        image_pil = Image.open(image_path).convert("RGB")
        
        # 1. Depth Map
        print("📐 Calcul Profondeur...")
        depth_res = self.depth_pipe(image_pil)
        depth_map = cv2.resize(np.array(depth_res["predicted_depth"]), image_pil.size, interpolation=cv2.INTER_CUBIC)
        d_min, d_max = depth_map.min(), depth_map.max()
        depth_map_norm = (depth_map - d_min) / (d_max - d_min + 1e-6)

        # 2. Calibration
        inference_state = self.sam_processor.set_image(image_pil)
        print("📏 Calibration Echelle...")
        px_per_cm = self.calibrate_pixel_scale(inference_state)
        if px_per_cm == 0:
            print("⚠️ Calibration échouée. Mode dégradé (40px = 1cm).")
            px_per_cm = 40.0

        # 3. Analyse Contenant (Profondeur Z)
        print("🥣 Analyse du contenant...")
        container_info = self.auto_characterize_container(image_pil, inference_state, px_per_cm, depth_map_norm)
        plate_mask = container_info['mask']
        
        if plate_mask is not None:
            vals = depth_map_norm[plate_mask > 0]
            delta_z = np.max(vals) - np.min(vals)
            if delta_z < 0.05: delta_z = 0.1
            z_scale_factor = container_info['depth_cm'] / delta_z
        else:
            z_scale_factor = 10.0

        # 4. Identification
        print("🤖 Gemini identifie les aliments...")
        food_names = self.get_food_names(image_pil)
        
        # --- INITIALISATION DU DICTIONNAIRE FINAL ---
        plate_summary = {
            "total_calories": 0,
            "total_protein_g": 0.0,
            "total_carbs_g": 0.0,
            "total_fat_g": 0.0,
            "items": []
        }
        
        print("-" * 60)
        
        for name in food_names:
            print(f"   🔎 Traitement de : '{name}'...")
            
            # 1. Obtenir Macros (Dict complet maintenant)
            macros_100g = self.fatsecret.search_food(name)
            phys_density = 0.85 
            
            try:
                # Retry Logic SAM 3
                prompts_to_try = [name]
                simple = name.lower().replace("slices", "").replace("pieces", "").replace("cooked", "").strip()
                if simple != name.lower() and len(simple) > 2: prompts_to_try.append(simple)
                if len(name.split()) > 1: prompts_to_try.append(name.split()[-1])

                mask_2d = None
                used_prompt = ""

                for p in prompts_to_try:
                    out = self.sam_processor.set_text_prompt(state=inference_state, prompt=p)
                    if out["masks"] is not None:
                        temp = self.force_2d(out["masks"])
                        if np.sum(temp) > 100:
                            mask_2d = temp
                            used_prompt = p
                            break
                
                if mask_2d is None:
                    print(f"      ❌ Pas de masque pour '{name}'")
                    continue

                # Calcul Volume
                area_px = np.sum(mask_2d > 0)
                area_cm2 = area_px / (px_per_cm ** 2)

                kernel = np.ones((15,15), np.uint8)
                dilated = cv2.dilate(mask_2d, kernel, iterations=1)
                local_ring = (dilated > 0) & (mask_2d == 0)
                if plate_mask is not None: 
                    valid_ring = local_ring & (plate_mask > 0)
                    if np.sum(valid_ring) > 50: local_ring = valid_ring
                
                floor_val = np.percentile(depth_map_norm[local_ring], 10) if np.sum(local_ring) > 0 else np.min(depth_map_norm)
                
                heights = np.maximum(depth_map_norm[mask_2d > 0] - floor_val, 0)
                vol_index = np.sum(heights)
                volume_cm3 = vol_index * (z_scale_factor / (px_per_cm ** 2))

                # Calcul Poids
                weight_g = volume_cm3 * phys_density
                
                # Calcul Macros Totaux pour cet aliment
                item_cals = (weight_g / 100) * macros_100g['kcal']
                item_prot = (weight_g / 100) * macros_100g['protein']
                item_carb = (weight_g / 100) * macros_100g['carbs']
                item_fat  = (weight_g / 100) * macros_100g['fat']

                # Ajout au dictionnaire de l'aliment
                item_data = {
                    "name": name,
                    "detected_as": used_prompt,
                    "weight_g": int(weight_g),
                    "calories": int(item_cals),
                    "macros": {
                        "protein": round(item_prot, 1),
                        "carbs": round(item_carb, 1),
                        "fat": round(item_fat, 1)
                    },
                    "volume_cm3": round(volume_cm3, 1),
                    "area_cm2": round(area_cm2, 1)
                }
                
                # Mise à jour du bilan global
                plate_summary["items"].append(item_data)
                plate_summary["total_calories"] += item_cals
                plate_summary["total_protein_g"] += item_prot
                plate_summary["total_carbs_g"] += item_carb
                plate_summary["total_fat_g"] += item_fat

                print(f"🍔 {name.upper()} ({int(weight_g)}g)")
                print(f"   Cal: {int(item_cals)} | P: {item_prot:.1f}g | C: {item_carb:.1f}g | F: {item_fat:.1f}g")

            except Exception as e: 
                print(f"❌ Erreur critique {name}: {e}")

        # Arrondir les totaux finaux
        plate_summary["total_calories"] = int(plate_summary["total_calories"])
        plate_summary["total_protein_g"] = round(plate_summary["total_protein_g"], 1)
        plate_summary["total_carbs_g"] = round(plate_summary["total_carbs_g"], 1)
        plate_summary["total_fat_g"] = round(plate_summary["total_fat_g"], 1)

        print("=" * 60)
        print(f"🔥 BILAN: {plate_summary['total_calories']} kcal | Prot: {plate_summary['total_protein_g']}g")
        
        return plate_summary

    # ... (Méthodes calibrate_pixel_scale, auto_characterize_container, get_food_names inchangées) ...
    def calibrate_pixel_scale(self, inference_state):
        KNOWN_PLATE_DIAMETER_CM = 22.5
        prompts = ["plate", "dish", "bowl", "container"]
        for p in prompts:
            try:
                out = self.sam_processor.set_text_prompt(state=inference_state, prompt=p)
                mask = self.force_2d(out["masks"])
                if np.sum(mask) > 5000:
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(mask, cnts, -1, (1), thickness=cv2.FILLED)
                    diam_px = 2 * np.sqrt(np.sum(mask)/np.pi)
                    return diam_px / KNOWN_PLATE_DIAMETER_CM
            except: pass
        return 0

    def auto_characterize_container(self, image_pil, inference_state, px_per_cm, depth_map_norm):
        info = {'mask': None, 'depth_cm': 2.0, 'type': 'unknown'}
        prompts = ["plate", "bowl", "dish"]
        for p in prompts:
            try:
                out = self.sam_processor.set_text_prompt(state=inference_state, prompt=p)
                m = self.force_2d(out["masks"])
                if np.sum(m) > 5000:
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    filled = np.zeros_like(m)
                    cv2.drawContours(filled, cnts, -1, (1), thickness=cv2.FILLED)
                    info['mask'] = filled > 0
                    break
            except: continue
            
        if info['mask'] is None: return info

        area_px = np.sum(info['mask'])
        diam_cm = (2 * np.sqrt(area_px / np.pi)) / px_per_cm
        
        prompt = f"Look at the container. Diameter: {diam_cm:.1f}cm. Return JSON: {{'type': 'Bowl', 'depth_cm': 5.5}}. DO NOT use approx symbols."
        try:
            res = self.gemini_client.models.generate_content(model="gemini-2.5-flash", contents=[image_pil, prompt])
            txt = re.sub(r'[≈~`]', '', res.text.strip())
            match = re.search(r'\{.*\}', txt, re.DOTALL)
            if match:
                data = ast.literal_eval(match.group(0))
                info.update(data)
        except: pass
        return info

    def get_food_names(self, img):
        try:
            res = self.gemini_client.models.generate_content(model="gemini-2.5-flash", contents=[img, "List food items. Python list of strings only."])
            return ast.literal_eval(res.text.replace("```python", "").replace("```", "").strip())
        except: return []

if __name__ == "__main__":
    IMG = "images/Filou.jpeg" 
    scanner = NutrientScannerV5()
    scanner.analyze_plate(IMG)