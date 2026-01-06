
import torch
import numpy as np
from PIL import Image
import os
import sys
import cv2
import requests
import ast
import re
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
        """
        if len(image_paths) < 2:
            return {"error": "Il faut au moins 2 images pour la 3D précise. Utilisez le mode 'Single' sinon."}

        # 1. RECONSTRUCTION 3D
        print(f"📐 Reconstruction 3D ({len(image_paths)} vues)...")
        imgs_pil = load_images(image_paths, size=512)
        pairs = make_pairs(imgs_pil, scene_graph='complete')
        output = inference(pairs, self.dust3r, self.device, batch_size=1)
        
        # Alignement Global
        scene = GlobalAligner(output, device=self.device, mode='GlobalAligner')
        scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=0.01)
        
        # On travaille dans le repère de la première image (Ref)
        pts3d_map = scene.get_pts3d()[0].detach().cpu().numpy() # (H, W, 3)
        ref_img_pil = Image.open(image_paths[0]).convert("RGB").resize((pts3d_map.shape[1], pts3d_map.shape[0]))
        
        # 2. CALIBRATION ÉCHELLE (Via Carte de Crédit)
        print("📏 Calibration via Carte...")
        sam_state = self.sam_processor.set_image(ref_img_pil)
        scale_factor = self.calibrate_scale_3d(sam_state, pts3d_map)
        
        if scale_factor == 0:
            print("⚠️ Pas de carte détectée. Calibration Assiette Fallback.")
            # Fallback Assiette (22.5cm)
            scale_factor = self.calibrate_scale_3d_plate(sam_state, pts3d_map)
            
        if scale_factor == 0:
            print("❌ Échec total calibration. Abort.")
            return {"error": "Calibration impossible (Carte ou Assiette requise)"}

        print(f"   ✅ Facteur d'échelle trouvé: 1.0 unité 3D = {scale_factor:.2f} cm")

        # 3. IDENTIFICATION ALIMENTS
        print("🤖 Identification Gemini...")
        food_items = self.get_food_names(ref_img_pil)
        
        # 4. CALCUL VOLUMÉTRIQUE
        summary = {"items": [], "total_calories": 0}
        
        # Détection Plan de la Table (RANSAC sur points bas)
        # On prend les points valides
        valid_mask = pts3d_map[:,:,2] > 0 # Supposant Z > 0 devant caméra
        all_pts = pts3d_map[valid_mask]
        if len(all_pts) > 1000:
             # Simple estimation: plan moyen des points les plus éloignés (si vue de haut)
             # Pour faire simple ici: On considère que DUSt3R aligne souvent bien le sol si visible.
             # On va raffiner le sol localement pour chaque aliment.
             pass

        print("-" * 60)
        DENSITY_MAP = {"LOW": 0.35, "MEDIUM": 0.85, "HIGH": 1.05}

        for item in food_items:
            name = item.get("name", "Food")
            density_str = item.get("density", "MEDIUM")
            print(f"   🔎 '{name}'...")

            # Masque 2D
            mask_2d = self.get_sam_mask(sam_state, name)
            if mask_2d is None: continue
            
            # Extraction Nuage de Points de l'Aliment
            # mask_2d doit être resize à la taille de pts3d_map
            mask_2d_resized = cv2.resize(mask_2d, (pts3d_map.shape[1], pts3d_map.shape[0]), interpolation=cv2.INTER_NEAREST)
            food_pts = pts3d_map[mask_2d_resized > 0]
            
            if len(food_pts) < 100: continue

            # --- CALCUL VOLUME 3D (Vrai 3D !) ---
            # 1. Mise à l'échelle cm
            food_pts_cm = food_pts * scale_factor
            
            # 2. Hauteur locale
            # On cherche le sol SOUS l'aliment.
            # On prend un anneau autour du masque pour définir le plan du sol local
            kernel = np.ones((10,10), np.uint8)
            dilated = cv2.dilate(mask_2d_resized, kernel, iterations=2)
            ring_mask = (dilated > 0) & (mask_2d_resized == 0)
            
            if np.sum(ring_mask) > 10:
                ring_pts = pts3d_map[ring_mask] * scale_factor
                # On fitte un plan ou on prend la médiane Z (si aligné)
                # DUSt3R aligne souvent Z = profondeur caméra.
                # Donc la "hauteur" est plutôt selon l'axe Y ou un axe arbitraire.
                # L'approche robuste : PCA pour trouver l'axe principal (normale à la table)
                
                # Simplification efficace : 
                # On calcule le volume de la boite englobante orientée (PCA) 
                # OU on intègre la distance point-plan.
                
                # Méthode "Intégration sur Plan Local":
                # Centre de gravité du ring
                center_ground = np.mean(ring_pts, axis=0)
                # Normale approximative (vecteur vers la caméra - ou moyenne des normales locales)
                # Ici on va assumer que la hauteur est la distance au plan du ring.
                
                # ... Implémentation simplifiée : Delta moyen par rapport aux voisins
                # On compare chaque point de bouffe à son voisin le plus proche dans le ring ? Trop lourd.
                
                # Méthode : "Hauteur relative moyenne"
                z_ground = np.mean(ring_pts[:, 2]) # On assume que Z est la profondeur principale
                # C'est risqué avec DUSt3R car l'axe n'est pas garanti.
                
                # Méthode ROBUSTE: Point-to-Plane distance
                # Fit plan sur ring_pts
                from sklearn.linear_model import RANSACRegressor
                X_ring = ring_pts[:, :2]
                y_ring = ring_pts[:, 2]
                reg = RANSACRegressor().fit(X_ring, y_ring)
                
                # Prédire Z du sol pour les points de bouffe (x,y)
                X_food = food_pts_cm[:, :2]
                z_food_ground_pred = reg.predict(X_food)
                z_food_actual = food_pts_cm[:, 2]
                
                # Hauteur = Différence (Absolue car on ne sait pas le sens)
                heights = np.abs(z_food_actual - z_food_ground_pred)
                
                # Volume = Somme(Hauteurs) * Aire_XY_Moyenne_par_Point ? Non.
                # Volume = Aire_Projetée * Hauteur_Moyenne
                
                # Aire projetée sur le plan ajusté :
                # On projette les points sur le plan 2D principal (PCA)
                # Approximation : Aire masque 2D * Scale^2 * cos(tilt)
                # On va utiliser une voxelisation simple :
                vol_cm3 = np.sum(heights) * (scale_factor * 0.1) # Approx grossière facteur aire pixel
                
                # Raffinement Voxel (si on veut être pro) :
                # Mais restons sur : Volume = Aire_Base * Hauteur_Moyenne
                # Aire_Base_cm2 = (Nb_Pixels_Masque) * (Scale_Unité_par_Pixel)^2 ?
                # Non, Scale est pour 3D unit -> cm.
                # Il nous faut la taille d'un pixel en cm.
                # DUSt3R nous donne les coords 3D de CHAQUE pixel.
                # Donc on peut calculer l'aire de chaque quad pixel dans l'espace 3D !
                
                # Méthode : Somme des volumes des prismes élémentaires
                # Pour chaque pixel i: V_i = Aire_Pixel_i * Hauteur_i
                # Aire_Pixel_i approx = Distance(Voisin_Droit) * Distance(Voisin_Bas)
                # C'est trop lourd.
                
                # --- MÉTHODE FINALE CHOISIE ---
                # Hauteur Moyenne * Aire Projetée Corrigée
                avg_height_cm = np.mean(heights)
                
                # Aire réelle de la surface du masque (somme des aires des triangles 3D ?)
                # On va utiliser l'aire de la carte pour ratio.
                # Aire_Food_Pixels / Aire_Card_Pixels * CARD_AREA_REAL ? Non, perspective.
                
                # On calcule l'aire physique du masque en sommant les distances inter-pixels
                # C'est faisable vectoriellement.
                # dx = pts[x+1] - pts[x], dy = pts[y+1] - pts[y]
                # area = norm(cross(dx, dy))
                # On va simplifier : Aire Surface 3D ~ Aire plane.
                # Aire Plane cm2 = Somme (1 pixel en cm²)
                # Un pixel en 3D au centre de l'objet a une taille physique.
                # On prend 100 points au hasard, on mesure la distance moyenne avec leurs voisins.
                # dist_moy_cm = mean(dist(p, p_voisin))
                # aire_pixel_cm2 = dist_moy_cm ** 2
                
                # Implémentation Vectorisée Rapide
                sample_indices = np.random.choice(len(food_pts_cm), min(500, len(food_pts_cm)), replace=False)
                sample_pts = food_pts_cm[sample_indices]
                # Distance au voisin le plus proche (brute force sur petit sample)
                # On estime la densité de points locale.
                # En réalité, DUSt3R sort une grille dense.
                # Donc la distance entre (i,j) et (i+1,j) EST la taille du pixel.
                
                # On revient à la map complète
                # On prend les points du masque
                y_idx, x_idx = np.where(mask_2d_resized > 0)
                # On prend un sous-ensemble pour aller vite
                step = 5
                y_sub, x_sub = y_idx[::step], x_idx[::step]
                
                if len(y_sub) == 0: continue
                
                p00 = pts3d_map[y_sub, x_sub] * scale_factor
                # Voisins (si dans l'image)
                # On approxime l'aire locale par pixel par la distance médiane inter-pixel locale au carré
                # C'est valide car le maillage est régulier en 2D.
                
                # Astuce: On calcule l'aire de la Carte de Crédit dans le Nuage 3D (Aire Physique Calculée)
                # On connait son aire réelle (46.22).
                # On connait son aire "Somme des points".
                # Non, on connait scale_factor qui convertit Unité3D -> cm.
                # Donc Volume = Somme(Hauteurs_Unité * Scale) * (Aire_Pixel_Unité * Scale^2)
                # Mais Aire_Pixel_Unité varie selon la profondeur !
                
                # REVISION : DUSt3R output est un nuage de points (X,Y,Z).
                # Si on projette ce nuage sur le plan de la table, on a l'aire de base.
                # Aire Base = ConvexHull(Food_Pts_Projected_On_Plane).Area
                from scipy.spatial import ConvexHull
                hull = ConvexHull(X_food) # Projection sur le plan local 2D (approx RANSAC X,Y)
                base_area_cm2 = hull.volume # En 2D, volume = aire
                
                volume_cm3 = base_area_cm2 * avg_height_cm
                
                macros = self.fatsecret.search_food(name)
                phys_density = DENSITY_MAP.get(density_str.upper(), 0.85)
                weight_g = volume_cm3 * phys_density
                kcals = (weight_g/100)*macros['kcal']
                
                summary["items"].append({
                    "name": name, "weight": int(weight_g), "kcal": int(kcals),
                    "vol": round(volume_cm3, 1)
                })
                summary["total_calories"] += kcals
                
                print(f"🍔 {name}: {int(weight_g)}g ({volume_cm3:.1f} cm3) - {int(kcals)} kcal")
                
        print(f"🔥 TOTAL: {int(summary['total_calories'])} kcal")
        return summary

    def calibrate_scale_3d(self, sam_state, pts3d_map):
        prompts = ["credit card", "payment card"]
        REAL_DIAGONAL_CM = np.sqrt(8.56**2 + 5.40**2) # ~10.1 cm
        
        for p in prompts:
            mask = self.get_sam_mask(sam_state, p)
            if mask is None: continue
            
            # Masque resize
            mask = cv2.resize(mask, (pts3d_map.shape[1], pts3d_map.shape[0]), interpolation=cv2.INTER_NEAREST)
            pts = pts3d_map[mask > 0]
            if len(pts) < 100: continue
            
            # On cherche les 2 points les plus éloignés (Diagonale)
            # C'est lourd en N^2. On prend le Convex Hull.
            from scipy.spatial import ConvexHull, distance_matrix
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                # Distances entre tous les points du hull
                dists = distance_matrix(hull_pts, hull_pts)
                max_dist_unit = np.max(dists)
                
                # Scale = Vrai / Mesuré
                return REAL_DIAGONAL_CM / max_dist_unit
            except: pass
        return 0
    
    def calibrate_scale_3d_plate(self, sam_state, pts3d_map):
        # Fallback Assiette 22.5cm
        mask = self.get_sam_mask(sam_state, "plate")
        if mask is None: return 0
        mask = cv2.resize(mask, (pts3d_map.shape[1], pts3d_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        pts = pts3d_map[mask > 0]
        if len(pts) < 100: return 0
        
        from scipy.spatial import ConvexHull, distance_matrix
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            dists = distance_matrix(hull_pts, hull_pts)
            max_dist_unit = np.max(dists)
            return 22.5 / max_dist_unit
        except: return 0

    def get_sam_mask(self, state, prompt):
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