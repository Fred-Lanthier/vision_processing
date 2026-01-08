import torch
import numpy as np
from PIL import Image
import os
import sys
import cv2
import requests
import ast
import re
import threading
import concurrent.futures
import time
from dotenv import load_dotenv

# --- SETUP PATHS ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'dust3r'))

# --- IMPORTS IA ---
from google import genai
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

try:
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import GlobalAligner
except ImportError:
    print("❌ DUSt3R manquant.")
    sys.exit(1)

load_dotenv()

# ==========================================
# 1. CLIENT API (Nutrition)
# ==========================================
class FatSecretClient:
    def __init__(self):
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.token = None

    def get_token(self):
        url = "https://oauth.fatsecret.com/connect/token"
        data = {'grant_type': 'client_credentials', 'scope': 'basic'}
        auth = (self.client_id, self.client_secret)
        try:
            response = requests.post(url, data=data, auth=auth)
            if response.status_code == 200:
                self.token = response.json()['access_token']
                return True
        except: pass
        return False

    def search_food(self, query):
        if not self.token and not self.get_token(): return self._default()
        url = "https://platform.fatsecret.com/rest/server.api"
        headers = {'Authorization': f'Bearer {self.token}'}
        params = {'method': 'foods.search', 'search_expression': query, 'format': 'json', 'max_results': 1}
        try:
            res = requests.get(url, headers=headers, params=params).json()
            if 'foods' in res and 'food' in res['foods']:
                desc = res['foods']['food'][0].get('food_description', '')
                cal = re.search(r'Calories:\s*(\d+)', desc)
                prot = re.search(r'Protein:\s*([\d\.]+)g', desc)
                carb = re.search(r'Carbs:\s*([\d\.]+)g', desc)
                fat = re.search(r'Fat:\s*([\d\.]+)g', desc)
                return {
                    "kcal": int(cal.group(1)) if cal else 0,
                    "protein": float(prot.group(1)) if prot else 0.0,
                    "carbs": float(carb.group(1)) if carb else 0.0,
                    "fat": float(fat.group(1)) if fat else 0.0
                }
        except: pass
        return self._default()

    def _default(self):
        return {"kcal": 150, "protein": 5.0, "carbs": 20.0, "fat": 5.0}

# ==========================================
# 2. MOTEUR MULTI-VUE (Le Cerveau)
# ==========================================
class NutrientScannerMultiView:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 [INIT PRO] Démarrage sur {self.device}...")

        # Lock pour SAM (non thread-safe)
        self.sam_lock = threading.RLock()

        # APIs
        self.google_key = os.getenv("GOOGLE_API_KEY")
        self.gemini_client = genai.Client(api_key=self.google_key)
        self.fatsecret = FatSecretClient()

        # IA 2D (Segmentation)
        print("⏳ Chargement SAM 3...")
        self.sam_model = build_sam3_image_model()
        if hasattr(self.sam_model, "to"): self.sam_model.to(self.device)
        self.sam_processor = Sam3Processor(self.sam_model)

        # IA 3D (Reconstruction)
        print("⏳ Chargement DUSt3R...")
        self.dust3r = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to(self.device)
        self.dust3r.eval()

        # Constantes Physiques
        self.CARD_WIDTH_CM = 8.56
        self.CARD_AREA_CM2 = 46.22

    def analyze_scene(self, image_paths):
        """
        Entrée: Liste de chemins d'images ['img1.jpg', 'img2.jpg', ...]
        Sortie: Bilan nutritionnel précis.
        Exécution PARALLÈLE :
        - Branche 1 (GPU): Masking + Reconstruction 3D (DUSt3R)
        - Branche 2 (API): Identification (Gemini) + Info Nutrition (FatSecret) + Calibration 2D
        """
        if len(image_paths) < 2:
            return {"error": "Il faut au moins 2 images pour la 3D précise."}

        start_time = time.time()
        print(f"🚀 Démarrage Analyse Parallèle ({len(image_paths)} images)...")

        # Chargement images une seule fois
        # On garde une copie pour l'affichage/debug et une pour le traitement
        ref_img_path = image_paths[0]
        
        # --- Lancement des Threads ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Note: SAM et DUSt3R utilisent le GPU. Pour éviter les conflits VRAM,
            # on va séquencer les tâches GPU dans un seul thread si nécessaire,
            # ou espérer que PyTorch gère bien le contexte (souvent risqué).
            # STRATÉGIE: 
            # Thread 1 (Lourd GPU): Masking (SAM) -> DUSt3R
            # Thread 2 (API/CPU): Gemini -> FatSecret -> Calcul Scale 2D (sur CPU/SAM léger)
            
            # Attention: SAM est un objet partagé. Il faut un Lock ou l'utiliser dans un seul thread.
            # On va faire le Masking AVANT le split pour être sûr, ou dédier le GPU au Thread 1.
            # Pour l'instant, on fait séquentiel pour la partie "Image Loading" pour éviter les bugs IO.
            
            future_ai = executor.submit(self._pipeline_ai_analysis, ref_img_path)
            future_3d = executor.submit(self._pipeline_3d_reconstruction, image_paths)
            
            # Attente des résultats
            try:
                ai_result = future_ai.result()
                scene_3d_result = future_3d.result()
            except Exception as e:
                print(f"❌ Erreur dans les threads: {e}")
                import traceback
                traceback.print_exc()
                return {"error": f"Erreur interne: {e}"}

        # --- FUSION DES RÉSULTATS ---
        print(f"⏱️  Fusion des résultats (Temps écoulé: {time.time() - start_time:.1f}s)...")
        
        # Récupération données 3D
        pts3d_map = scene_3d_result['pts3d_map'] # (H, W, 3) dans le repère de la caméra 1
        
        # Récupération données AI
        food_items = ai_result['food_items']
        plate_diameter_cm = ai_result['plate_diameter_cm']
        sam_state_ref = ai_result['sam_state'] # État SAM sur l'image de référence

        # --- CALIBRATION ÉCHELLE ---
        # On a le diamètre réel de l'assiette (via la carte en 2D).
        # On doit mesurer l'assiette dans le nuage 3D DUSt3R.
        print(f"📏 Calibration: Diamètre Assiette Réel calculé = {plate_diameter_cm:.1f} cm")
        
        scale_factor = self._compute_scale_factor(pts3d_map, sam_state_ref, plate_diameter_cm)
        
        if scale_factor <= 0:
            print("⚠️ Échec calibration 3D. Tentative fallback densité standard.")
            scale_factor = 0 # On devra gérer ça
            
        print(f"   ✅ Facteur d'échelle final: 1.0 unité 3D = {scale_factor:.2f} cm")

        # --- CALCUL FINAL VOLUMES ---
        summary = {"items": [], "total_calories": 0}
        DENSITY_MAP = {"LOW": 0.35, "MEDIUM": 0.85, "HIGH": 1.05}

        # Pour le masque 3D, on doit s'assurer que pts3d_map correspond à la taille de sam_state
        # DUSt3R sort du 512x512 (ou 224), SAM travaille sur l'originale.
        # On resize les masques SAM vers la shape de pts3d_map.
        tgt_h, tgt_w = pts3d_map.shape[:2]

        for item in food_items:
            name = item.get("name", "Food")
            density_str = item.get("density", "MEDIUM")
            print(f"   🔎 '{name}'...")

            # Masque via SAM (sur l'image de ref originale)
            mask_2d = self.get_sam_mask(sam_state_ref, name)
            if mask_2d is None: continue
            
            # Resize masque vers map 3D
            mask_resized = cv2.resize(mask_2d, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST)
            
            # Extraction Points 3D
            food_pts = pts3d_map[mask_resized > 0]
            if len(food_pts) < 50: continue

            if scale_factor > 0:
                # Volume = Aire_Base * Hauteur
                # 1. Aire Base (Projection sur plan table)
                # On projette sur le plan XY local (approx)
                # DUSt3R aligne souvent la vue 1 avec le monde.
                
                # Hauteur : Distance point au plan de l'assiette
                # On récupère le plan de l'assiette
                plate_mask = self.get_sam_mask(sam_state_ref, "plate")
                if plate_mask is not None:
                    plate_mask_res = cv2.resize(plate_mask, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST)
                    plate_pts = pts3d_map[plate_mask_res > 0]
                    # Fit plan
                    if len(plate_pts) > 100:
                        # RANSAC Plane fitting
                        from sklearn.linear_model import RANSACRegressor
                        reg = RANSACRegressor().fit(plate_pts[:, :2], plate_pts[:, 2])
                        z_ground = reg.predict(food_pts[:, :2])
                        heights = np.abs(food_pts[:, 2] - z_ground)
                    else:
                        heights = np.zeros(len(food_pts)) # Fallback
                else:
                    # Fallback min Z
                    heights = food_pts[:, 2] - np.min(food_pts[:, 2])

                avg_height_cm = np.mean(heights) * scale_factor
                
                # Aire Base en cm2
                # Approx: Aire masque pixels * (scale * pixel_size)^2 ??
                # Plus robuste: Convex Hull 2D des points projetés * scale^2
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(food_pts[:, :2])
                    area_unit2 = hull.volume # En 2D c'est l'aire
                    area_cm2 = area_unit2 * (scale_factor**2)
                except:
                    area_cm2 = 0

                vol_cm3 = area_cm2 * avg_height_cm
            else:
                vol_cm3 = 0 # Echec 3D

            # Fallback si 3D échoue mais qu'on a identifié
            if vol_cm3 == 0: 
                vol_cm3 = 150 # Valeur par défaut "portion moyenne"

            # Macros
            macros = self.fatsecret.search_food(name)
            phys_density = DENSITY_MAP.get(density_str.upper(), 0.85)
            weight_g = vol_cm3 * phys_density
            kcals = (weight_g/100)*macros['kcal']
            
            summary["items"].append({
                "name": name, "weight": int(weight_g), "kcal": int(kcals),
                "vol": round(vol_cm3, 1), "macros": macros
            })
            summary["total_calories"] += kcals
            print(f"     🍔 {int(weight_g)}g - {int(kcals)} kcal")

        print(f"🔥 TOTAL FINAL: {int(summary['total_calories'])} kcal")
        return summary

    def _pipeline_ai_analysis(self, ref_img_path):
        """
        Thread API : Identifie aliments, cherche calories, calcule taille assiette (2D).
        """
        print("   [AI] Début analyse image référence...")
        img_pil = Image.open(ref_img_path).convert("RGB")
        
        # 1. Gemini Identification
        food_items = self.get_food_names(img_pil)
        print(f"   [AI] Aliments trouvés: {[f['name'] for f in food_items]}")
        
        # 2. SAM pour Calibration 2D (Carte -> Assiette)
        # On initialise SAM ici pour ce thread
        # Note: Si SAM est sur GPU, attention aux conflits.
        # On suppose que l'inférence SAM est rapide et "interleaved".
        with self.sam_lock:
            sam_state = self.sam_processor.set_image(img_pil)
            
            plate_diameter_cm = 25.0 # Valeur par défaut standard
            
            # Recherche Carte Crédit
            card_mask = self.get_sam_mask(sam_state, ["credit card", "card", "debit card"])
            plate_mask = self.get_sam_mask(sam_state, ["plate", "dish"])
        
        if card_mask is not None and plate_mask is not None:
            # Calcul Diamètre Pixels
            # Carte: Diagonale connue 10.1cm (ou largeur 8.56cm)
            # On prend la bounding box orientée ou simple
            y_c, x_c = np.where(card_mask > 0)
            y_p, x_p = np.where(plate_mask > 0)
            
            if len(y_c) > 0 and len(y_p) > 0:
                # Taille Carte (Max Dist)
                # Approx rapide: Bounding Box diagonale
                h_c = np.max(y_c) - np.min(y_c)
                w_c = np.max(x_c) - np.min(x_c)
                diag_card_px = np.sqrt(h_c**2 + w_c**2)
                
                # Taille Assiette
                h_p = np.max(y_p) - np.min(y_p)
                w_p = np.max(x_p) - np.min(x_p)
                diag_plate_px = np.sqrt(h_p**2 + w_p**2) # Ou avg(h,w)
                
                # Règle de 3
                # 10.1 cm (diag carte) -> diag_card_px
                # ? cm -> diag_plate_px
                cm_per_px = 10.1 / diag_card_px
                plate_diameter_cm = diag_plate_px * cm_per_px
                print(f"   [AI] 💳 Carte détectée! Scale 2D: {cm_per_px:.4f} cm/px")
                print(f"   [AI] 🍽️ Diamètre Assiette estimé: {plate_diameter_cm:.1f} cm")
        else:
            print("   [AI] Pas de carte trouvée, utilisation diamètre standard (25cm).")

        return {
            "food_items": food_items,
            "plate_diameter_cm": plate_diameter_cm,
            "sam_state": sam_state
        }

    def _pipeline_3d_reconstruction(self, image_paths):
        """
        Thread GPU : Masking Background -> DUSt3R -> Global Alignment
        """
        print("   [3D] Début Pipeline 3D...")
        
        # 1. Masking (Black out background)
        # On doit charger chaque image, détecter "Food + Plate", et mettre le reste en noir.
        masked_images = []
        
        # On charge les images en PIL
        imgs_pil = load_images(image_paths, size=512)
        
        print("   [3D] Segmentation & Masquage du fond...")
        # Pour aller vite, on utilise le SAM déjà chargé dans self (attention conflit thread)
        # Idéalement on instancierait un petit modèle de segmentation ici ou on lock.
        # On va faire simple: On suppose que self.sam_processor est thread-safe pour l'inférence si séquentiel
        
        for i, img_dict in enumerate(imgs_pil):
            # img_dict['img'] est un tenseur (1, 3, H, W) normalisé
            # Il nous faut le PIL ou numpy uint8 pour SAM
            # On recharge brut pour SAM pour éviter artefacts de dé-normalisation
            orig_pil = Image.open(image_paths[i]).convert("RGB").resize(img_dict['img'].shape[-2:][::-1])
            
            # Segmenter Assiette + Bouffe
            # Prompt large
            with self.sam_lock:
                state = self.sam_processor.set_image(orig_pil)
                # On demande "Food" et "Plate"
                mask_food = self.get_sam_mask(state, ["food", "meal", "dish"])
                mask_plate = self.get_sam_mask(state, ["plate", "bowl"])
            
            final_mask = np.zeros((orig_pil.height, orig_pil.width), dtype=np.uint8)
            if mask_food is not None: final_mask = np.maximum(final_mask, mask_food)
            if mask_plate is not None: final_mask = np.maximum(final_mask, mask_plate)
            
            # Dilater un peu pour ne pas couper les bords
            kernel = np.ones((5,5), np.uint8)
            final_mask = cv2.dilate(final_mask, kernel, iterations=2)
            
            # Appliquer masque (Black out)
            img_np = np.array(orig_pil)
            img_np[final_mask == 0] = [0, 0, 0] # Noir
            
            # Convertir pour DUSt3R
            # DUSt3R attend une liste de dicts
            from dust3r.utils.image import ImgNorm
            masked_pil = Image.fromarray(img_np)
            imgs_pil[i]['img'] = ImgNorm(masked_pil)[None]
            
        # 2. DUSt3R Inference
        print("   [3D] Inférence DUSt3R (Multi-View)...")
        pairs = make_pairs(imgs_pil, scene_graph='complete')
        output = inference(pairs, self.dust3r, self.device, batch_size=1, verbose=False)
        
        # 3. Global Alignment
        print("   [3D] Alignement Global...")
        scene = GlobalAligner(output, device=self.device, mode='GlobalAligner')
        scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=0.01)
        
        # On retourne le nuage de points de la vue 1 (Ref)
        pts3d = scene.get_pts3d()[0].detach().cpu().numpy()
        
        return {"pts3d_map": pts3d}

    def _compute_scale_factor(self, pts3d_map, sam_state, real_diameter_cm):
        """
        Mesure l'assiette dans la 3D et calcule le ratio.
        """
        # Masque Assiette sur la vue 1
        mask = self.get_sam_mask(sam_state, "plate")
        if mask is None: return 0
        
        # Resize masque
        mask = cv2.resize(mask, (pts3d_map.shape[1], pts3d_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        pts = pts3d_map[mask > 0]
        
        if len(pts) < 100: return 0
        
        # Mesure diamètre 3D (Max distance paire points ? Trop lent)
        # PCA ou Bounding Box
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2) # On projette sur le plan principal
        pts_2d_proj = pca.fit_transform(pts)
        
        # Diamètre approx = (Max X - Min X + Max Y - Min Y) / 2
        min_vals = np.min(pts_2d_proj, axis=0)
        max_vals = np.max(pts_2d_proj, axis=0)
        dims = max_vals - min_vals
        diameter_3d = np.mean(dims) # Moyenne largeur/hauteur
        
        if diameter_3d < 1e-4: return 0
        
        return real_diameter_cm / diameter_3d



    def get_sam_mask(self, state, prompt):
        with self.sam_lock:
            out = self.sam_processor.set_text_prompt(state=state, prompt=prompt)
        if out["masks"] is None: return None
        return self._force_2d(out["masks"])

    def _force_2d(self, mask):
        if isinstance(mask, torch.Tensor): mask = mask.detach().cpu().numpy()
        while mask.ndim > 2:
            if mask.shape[0] == 1: mask = mask.squeeze(0)
            elif mask.shape[-1] == 1: mask = mask.squeeze(-1)
            else: mask = np.any(mask, axis=0)
        return mask.astype(np.uint8)

    def get_food_names(self, img):
        prompt = "List distinct DISHES as JSON [{'name': 'Pizza', 'density': 'MEDIUM'}]."
        try:
            res = self.gemini_client.models.generate_content(model="gemini-2.5-flash", contents=[img, prompt])
            return ast.literal_eval(res.text.strip().replace("```json", "").replace("```", ""))
        except: return []

if __name__ == "__main__":
    # Exemple d'usage
    # images = ["path/to/img1.jpg", "path/to/img2.jpg"]
    # scanner = NutrientScannerMultiView()
    # scanner.analyze_scene(images)
    pass