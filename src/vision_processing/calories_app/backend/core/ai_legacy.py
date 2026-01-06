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

    def fit_plane_to_background(self, depth_map, background_mask):
        """
        Ajuste un plan (Ax + By + C = Z) aux pixels du fond (assiette vide/table)
        pour corriger l'inclinaison de la caméra (Tilt).
        Retourne la depth map 'aplatie' où le sol est à 0.
        """
        # On prend les coordonnées (y, x) des points du background
        y_coords, x_coords = np.where(background_mask > 0)
        z_vals = depth_map[background_mask > 0]
        
        if len(z_vals) < 100:
            return depth_map # Pas assez de points pour fitter

        # Construction de la matrice pour les moindres carrés: [x, y, 1] * [A, B, C]^T = z
        A_matrix = np.c_[x_coords, y_coords, np.ones(len(z_vals))]
        C_coeffs, _, _, _ = np.linalg.lstsq(A_matrix, z_vals, rcond=None)
        
        # On recrée le plan sur toute l'image
        h, w = depth_map.shape
        X_grid, Y_grid = np.meshgrid(np.arange(w), np.arange(h))
        plane_surface = C_coeffs[0] * X_grid + C_coeffs[1] * Y_grid + C_coeffs[2]
        
        # On soustrait le plan (Flattening)
        # On veut que les objets ressortent, donc Depth_Original - Plane
        # Note: Depth Anything: Proche=255/High, Loin=0/Low.
        # Donc Objet (Proche) > Fond (Loin).
        flattened = depth_map - plane_surface
        
        # On remet le 'sol' à 0 environ
        flattened = flattened - np.median(flattened[background_mask > 0])
        
        return np.maximum(flattened, 0)

    def analyze_plate(self, image_path):
        if not os.path.exists(image_path): return {"error": "Image introuvable"}
        image_pil = Image.open(image_path).convert("RGB")
        
        # 1. Depth Map (FULL IMAGE - NE PAS CROPPER)
        print("📐 Calcul Profondeur (Depth Anything V2)...")
        depth_res = self.depth_pipe(image_pil)
        depth_map = cv2.resize(np.array(depth_res["predicted_depth"]), image_pil.size, interpolation=cv2.INTER_CUBIC)
        
        # Normalisation Initiale
        d_min, d_max = depth_map.min(), depth_map.max()
        depth_map_norm = (depth_map - d_min) / (d_max - d_min + 1e-6)

        # 2. Calibration XY (Carte ou Assiette)
        inference_state = self.sam_processor.set_image(image_pil)
        print("📏 Calibration Echelle...")
        px_per_cm = self.calibrate_pixel_scale(inference_state)
        if px_per_cm == 0:
            print("⚠️ Calibration échouée. Mode dégradé (40px = 1cm).")
            px_per_cm = 40.0

        # 3. Segmentation Structurelle (Assiette vs Nourriture)
        print("🥣 Analyse Structurelle (Plane Fitting)...")
        
        # A. Masque de l'assiette globale
        plate_prompt_out = self.sam_processor.set_text_prompt(state=inference_state, prompt="plate")
        plate_mask = self.force_2d(plate_prompt_out["masks"])
        if np.sum(plate_mask) == 0:
             # Fallback: utiliser toute l'image comme 'table' si pas d'assiette
             plate_mask = np.ones_like(depth_map_norm, dtype=np.uint8)

        # B. Masque de TOUTE la nourriture (pour l'exclure du calcul du sol)
        food_prompt_out = self.sam_processor.set_text_prompt(state=inference_state, prompt="food item")
        all_food_mask = self.force_2d(food_prompt_out["masks"])
        
        # C. Définition du "Sol" (Background): C'est l'assiette MOINS la nourriture
        background_mask = (plate_mask > 0) & (all_food_mask == 0)
        
        # Sécurité: Si le masque est vide (assiette pleine à craquer), on prend le bord de l'assiette
        if np.sum(background_mask) < 1000:
            kernel = np.ones((20,20), np.uint8)
            eroded_plate = cv2.erode(plate_mask, kernel, iterations=3)
            background_mask = (plate_mask > 0) & (eroded_plate == 0) # Juste le bord (Rim)

        # 4. Correction du Tilt (Plane Flattening)
        flat_depth_map = self.fit_plane_to_background(depth_map_norm, background_mask)
        
        # 5. Calibration Z (Profondeur) via Assiette Standard
        # On suppose que la hauteur du "Bord" (Rim) par rapport au "Fond" (Center) est ~1.8cm
        # Si on a bien aplati l'image, le bord et le fond sont sur le plan 0, 
        # MAIS Depth Anything capture la concavité.
        # On va chercher le Z-Scale qui fait correspondre la dynamique de l'assiette à 1.8cm.
        
        # On regarde la variation Z sur l'assiette vide (si visible)
        z_vals_plate = flat_depth_map[plate_mask > 0]
        # On prend le 95e centile (hauts bords) - 5e centile (fond)
        z_range_norm = np.percentile(z_vals_plate, 95) - np.percentile(z_vals_plate, 5)
        
        REAL_PLATE_DEPTH_CM = 1.8 
        if z_range_norm > 0.02:
            z_scale_factor = REAL_PLATE_DEPTH_CM / z_range_norm
            print(f"   📐 Z-Calibration (Assiette): Range={z_range_norm:.3f} -> Factor={z_scale_factor:.1f}")
        else:
            # Fallback si l'assiette est plate ou invisible
            z_scale_factor = 15.0 # Valeur empirique moyenne
            print("   ⚠️ Z-Calibration fallback (Assiette plate/invisible).")

        # 6. Identification & Calcul Final
        print("🤖 Gemini identifie les aliments...")
        food_items_data = self.get_food_names(image_pil)
        
        plate_summary = {
            "total_calories": 0, "total_protein_g": 0.0, "total_carbs_g": 0.0, "total_fat_g": 0.0, "items": []
        }
        
        DENSITY_MAP = {"LOW": 0.35, "MEDIUM": 0.85, "HIGH": 1.05}

        print("-" * 60)
        for item in food_items_data:
            if isinstance(item, str): name, density_str = item, "MEDIUM"
            else: name, density_str = item.get("name", "Unknown"), item.get("density", "MEDIUM")

            print(f"   🔎 '{name}' ({density_str})...")
            macros_100g = self.fatsecret.search_food(name)
            phys_density = DENSITY_MAP.get(density_str.upper(), 0.85)
            
            try:
                # Segmentation spécifique
                prompts_to_try = [name, name.split()[0]]
                mask_2d = None
                for p in prompts_to_try:
                    out = self.sam_processor.set_text_prompt(state=inference_state, prompt=p)
                    if out["masks"] is not None:
                        temp = self.force_2d(out["masks"])
                        if np.sum(temp) > 100:
                            mask_2d = temp; break
                
                if mask_2d is None: continue

                # Calcul Volume sur la Depth Map APLATIE
                # Volume = Somme des hauteurs (Z) * Aire d'un pixel (XY)
                
                # Hauteur pixel = Valeur Z * Z_Scale
                # Aire pixel = (1/px_per_cm) * (1/px_per_cm)
                
                # On ne compte que la hauteur au dessus du "sol local" (qui est 0 grâce au flattening)
                # Mais par sécurité, on soustrait le min local pour enlever le bruit résiduel
                local_vals = flat_depth_map[mask_2d > 0]
                floor_bias = np.percentile(local_vals, 5) # Petit offset pour éviter le bruit de fond
                heights_cm = (local_vals - floor_bias) * z_scale_factor
                heights_cm = np.maximum(heights_cm, 0) # Pas de hauteur négative
                
                pixel_area_cm2 = 1.0 / (px_per_cm ** 2)
                
                # Volume = Somme(Hauteur_cm) * Aire_Pixel_cm2
                volume_cm3 = np.sum(heights_cm) * pixel_area_cm2
                area_cm2 = np.sum(mask_2d > 0) * pixel_area_cm2

                # Calculs nutritionnels
                weight_g = volume_cm3 * phys_density
                item_cals = (weight_g / 100) * macros_100g['kcal']
                
                # --- Update Summary ---
                plate_summary["items"].append({
                    "name": name, "weight_g": int(weight_g), "calories": int(item_cals),
                    "volume_cm3": round(volume_cm3, 1), "density_type": density_str,
                    "macros": macros_100g
                })
                plate_summary["total_calories"] += item_cals
                plate_summary["total_protein_g"] += (weight_g/100)*macros_100g['protein']
                plate_summary["total_carbs_g"] += (weight_g/100)*macros_100g['carbs']
                plate_summary["total_fat_g"] += (weight_g/100)*macros_100g['fat']

                print(f"🍔 {name.upper()}")
                print(f"   Vol: {volume_cm3:.1f} cm3 (H.Moy: {(np.mean(heights_cm) if len(heights_cm)>0 else 0):.1f}cm)")
                print(f"   Poids: {int(weight_g)}g | Cal: {int(item_cals)}")

            except Exception as e: print(f"Err {name}: {e}")

        # Final Rounding
        plate_summary["total_calories"] = int(plate_summary["total_calories"])
        plate_summary["total_protein_g"] = round(plate_summary["total_protein_g"], 1)
        plate_summary["total_carbs_g"] = round(plate_summary["total_carbs_g"], 1)
        plate_summary["total_fat_g"] = round(plate_summary["total_fat_g"], 1)

        print("=" * 60)
        print(f"🔥 BILAN: {plate_summary['total_calories']} kcal")
        return plate_summary

    # ... (Méthodes calibrate_pixel_scale, auto_characterize_container, get_food_names inchangées) ...
    def calibrate_pixel_scale(self, inference_state):
        """
        Tente de trouver une référence connue pour déterminer px_per_cm.
        Priorité 1: Carte standard (Crédit/ID) -> Format ISO ID-1 (8.56cm x 5.40cm = ~46.22 cm²)
        Priorité 2: Assiette standard (22.5cm)
        """
        
        # --- 1. CALIBRATION PAR CARTE (PRÉCIS) ---
        # Une carte est rectangulaire, mais SAM renvoie un masque. 
        # L'aire est constante peu importe l'orientation 2D.
        REAL_CARD_AREA_CM2 = 46.22
        card_prompts = ["credit card", "payment card", "debit card", "ID card", "membership card"]

        print("   💳 Recherche d'une carte de référence...")
        for p in card_prompts:
            try:
                out = self.sam_processor.set_text_prompt(state=inference_state, prompt=p)
                mask = self.force_2d(out["masks"])
                
                # Filtre : Une carte ne doit être ni minuscule ni géante (ex: 500px à 1/4 de l'image)
                # On met un seuil arbitraire bas pour capter même les petites cartes au fond
                if np.sum(mask) > 1000:
                    area_px = np.sum(mask)
                    
                    # Math: Scale^2 = Area_px / Area_real
                    # Scale (px/cm) = sqrt(Area_px / Area_real)
                    px_per_cm = np.sqrt(area_px / REAL_CARD_AREA_CM2)
                    
                    print(f"      ✅ CARTE détectée ('{p}').")
                    print(f"      Aire: {area_px}px -> Scale: {px_per_cm:.2f} px/cm")
                    return px_per_cm
            except Exception as e:
                # print(f"Err Card {p}: {e}")
                pass

        # --- 2. FALLBACK : CALIBRATION PAR ASSIETTE (APPROXIMATIF) ---
        print("   🍽️ Pas de carte. Recherche d'une assiette standard...")
        KNOWN_PLATE_DIAMETER_CM = 22.5
        plate_prompts = ["plate", "dish", "bowl", "container"]
        
        for p in plate_prompts:
            try:
                out = self.sam_processor.set_text_prompt(state=inference_state, prompt=p)
                mask = self.force_2d(out["masks"])
                if np.sum(mask) > 5000:
                    # On assume cercle pour l'assiette
                    diam_px = 2 * np.sqrt(np.sum(mask)/np.pi)
                    px_per_cm = diam_px / KNOWN_PLATE_DIAMETER_CM
                    print(f"      ✅ ASSIETTE détectée (Hypothèse {KNOWN_PLATE_DIAMETER_CM}cm). Scale: {px_per_cm:.2f} px/cm")
                    return px_per_cm
            except: pass
            
        print("   ⚠️ Aucune référence trouvée.")
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
        """
        Identifie les plats et estime leur densité pour le calcul de poids.
        Retourne une liste de dicts: [{'name': 'Pizza', 'density_type': 'MEDIUM'}, ...]
        """
        prompt = """
        You are an expert AI Nutritionist. Analyze the image to identify distinct DISHES for calorie tracking.
        
        STRICT RULES:
        1. THE "COMPOSITE DISH" RULE:
           - Group visible ingredients into their final culinary dish name.
           - Example: Dough + Tomato + Cheese -> Detect 'Cheese Pizza' (NOT 'dough', 'cheese').
           - Example: Pasta + Meat Sauce -> Detect 'Spaghetti Bolognese' (NOT 'pasta', 'sauce', 'meat').
           - Example: Burger with bun -> Detect 'Hamburger' (NOT 'bun', 'patty').

        2. THE "DENSITY" RULE:
           - Estimate the physical density of the food to help convert Volume to Weight.
           - 'LOW' (approx 0.3 g/cm3): Leafy salads, chips, popcorn, fluffy bread, meringue.
           - 'MEDIUM' (approx 0.85 g/cm3): Pasta, rice, pizza, cake, ice cream, fruits, mixed meals.
           - 'HIGH' (approx 1.05 g/cm3): Solid meat (steak/chicken breast), hard cheese, chocolate, dense mashed potatoes.

        OUTPUT FORMAT:
        Return ONLY a raw JSON list of objects. No markdown formatting.
        Example: [{"name": "Pepperoni Pizza", "density": "MEDIUM"}, {"name": "Caesar Salad", "density": "LOW"}]
        """
        try:
            res = self.gemini_client.models.generate_content(model="gemini-2.5-flash", contents=[img, prompt])
            txt = res.text.strip()
            # Nettoyage bourrin pour extraire le JSON
            if "```" in txt:
                txt = txt.split("```")[1].replace("json", "").replace("python", "").strip()
            
            # Parsing flexible
            import json
            try:
                return json.loads(txt)
            except:
                # Fallback si le JSON est malformé, on tente un eval python
                return ast.literal_eval(txt)
        except Exception as e:
            print(f"⚠️ Erreur Gemini: {e}")
            return []

if __name__ == "__main__":
    IMG = "images/Filou.jpeg" 
    scanner = NutrientScannerV5()
    scanner.analyze_plate(IMG)