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
try:
    from google import genai
except ImportError:
    genai = None
    print("⚠️ Google GenAI manquant.")

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("⚠️ SAM3 manquant. Le mode 'Segmentation' sera désactivé.")
    build_sam3_image_model = None
    Sam3Processor = None

try:
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
except ImportError as e:
    print(f"⚠️ DUSt3R manquant ({e}). Le mode '3D' sera désactivé.")
    inference = None
    AsymmetricCroCo3DStereo = None
    GlobalAlignerMode = None

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
        
        # On tente de forcer la récupération pour 100g
        search_query = f"{query} 100g"
        
        url = "https://platform.fatsecret.com/rest/server.api"
        headers = {'Authorization': f'Bearer {self.token}'}
        params = {'method': 'foods.search', 'search_expression': search_query, 'format': 'json', 'max_results': 1}
        try:
            res = requests.get(url, headers=headers, params=params).json()
            if 'foods' in res and 'food' in res['foods']:
                # FatSecret renvoie parfois une liste, parfois un dict si résultat unique
                food_data = res['foods']['food']
                if isinstance(food_data, list):
                    food_data = food_data[0]
                
                desc = food_data.get('food_description', '')
                print(f"   ℹ️ FatSecret ({search_query}): {desc}")
                
                cal = re.search(r'Calories:\s*(\d+)', desc)
                prot = re.search(r'Protein:\s*([\d\.]+)g', desc)
                carb = re.search(r'Carbs:\s*([\d\.]+)g', desc)
                fat = re.search(r'Fat:\s*([\d\.]+)g', desc)
                sugar = re.search(r'Sugar:\s*([\d\.]+)g', desc)
                fiber = re.search(r'Fiber:\s*([\d\.]+)g', desc)
                return {
                    "kcal": int(cal.group(1)) if cal else 0,
                    "protein": float(prot.group(1)) if prot else 0.0,
                    "carbs": float(carb.group(1)) if carb else 0.0,
                    "fat": float(fat.group(1)) if fat else 0.0,
                    "sugar": float(sugar.group(1)) if sugar else 0.0,
                    "fiber": float(fiber.group(1)) if fiber else 0.0
                }
        except Exception as e:
            print(f"FatSecret Error: {e}")
            pass
        return self._default()

    def _default(self):
        return {"kcal": 150, "protein": 5.0, "carbs": 20.0, "fat": 5.0, "sugar": 5.0, "fiber": 2.0}

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
        if build_sam3_image_model:
            self.sam_model = build_sam3_image_model()
            if hasattr(self.sam_model, "to"): self.sam_model.to(self.device)
            self.sam_processor = Sam3Processor(self.sam_model)
        else:
            self.sam_model = None
            self.sam_processor = None

        # IA 3D (Reconstruction)
        print("⏳ Chargement DUSt3R...")
        if AsymmetricCroCo3DStereo:
            self.dust3r = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to(self.device)
            self.dust3r.eval()
        else:
            self.dust3r = None

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
        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
        scene.compute_global_alignment(init='mst', niter=500, schedule='cosine', lr=0.01)
        
        # On travaille dans le repère global
        all_pts3d = scene.get_pts3d()
        ref_img_pil = Image.open(image_paths[0]).convert("RGB").resize((all_pts3d[0].shape[1], all_pts3d[0].shape[0]))
        
        # --- DEBUG EXPORT (Merged cloud) ---
        try:
            print("💾 Saving debug 3D plot (Merged)...")
            # Downsample merging for plot
            merged_pts = np.concatenate([p.detach().cpu().numpy().reshape(-1,3)[::10] for p in all_pts3d])
            # For colors, we'd need to resize all images, let's just use ref for simplicity or skip colors
            self.save_debug_plot(merged_pts.reshape(-1,1,3), np.zeros_like(merged_pts).reshape(-1,1,3), "debug_scene.png")
        except Exception as e:
            print(f"⚠️ Could not save 3D Plot: {e}")

        # 2. CALIBRATION ÉCHELLE (Via Carte de Crédit sur image 0)
        print("📏 Calibration via Carte...")
        sam_state_ref = self.sam_processor.set_image(ref_img_pil)
        scale_factor = self.calibrate_scale_3d(sam_state_ref, all_pts3d[0].detach().cpu().numpy())
        
        if scale_factor == 0:
            scale_factor = self.calibrate_scale_3d_plate(sam_state_ref, all_pts3d[0].detach().cpu().numpy())
            
        if scale_factor == 0:
            print("❌ Échec total calibration.")
            return {"error": "Calibration impossible"}

        print(f"   ✅ Facteur d'échelle trouvé: 1.0 unité 3D = {scale_factor:.2f} cm")

        # 3. IDENTIFICATION ALIMENTS
        print("🤖 Identification Gemini...")
        food_items = self.get_food_names(ref_img_pil)
        
        # 4. ACCUMULATION MULTI-VUES
        summary = {"items": [], "total_calories": 0}
        DENSITY_MAP = {"AIRY": 0.35, "PARTICULATE": 0.85, "SOLID": 1.05, "LOW": 0.35, "MEDIUM": 0.85, "HIGH": 1.05} # Fallback legacy

        for item in food_items:
            name = item.get("name", "Food")
            density_str = item.get("density", "SOLID") # Default to SOLID if unsure
            print(f"   🔎 Processing Multi-view for '{name}' ({density_str})...")

            all_item_pts = []
            
            # On boucle sur chaque vue pour récupérer les points de cet aliment
            for v_idx, img_path in enumerate(image_paths):
                try:
                    curr_pts = all_pts3d[v_idx].detach().cpu().numpy()
                    h, w = curr_pts.shape[:2]
                    
                    # On doit segmenter dans cette vue
                    v_img = Image.open(img_path).convert("RGB").resize((w, h))
                    v_sam_state = self.sam_processor.set_image(v_img)
                    v_mask = self.get_sam_mask(v_sam_state, name)
                    
                    if v_mask is not None:
                        # Resize mask si besoin
                        if v_mask.shape != (h, w):
                            v_mask = cv2.resize(v_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        pts = curr_pts[v_mask > 0]
                        if len(pts) > 0:
                            all_item_pts.append(pts)
                except:
                    continue

            if not all_item_pts:
                print(f"   ⚠️ No points found for {name} in any view.")
                continue

            # Fusionner tous les points de toutes les vues
            food_pts_raw = np.concatenate(all_item_pts)
            food_pts_cm = food_pts_raw * scale_factor

            # --- CALCUL VOLUME 3D (Convex Hull sur Nuage Fusionné) ---
            from scipy.spatial import ConvexHull
            is_liquid = any(l in name.lower() for l in ['water', 'juice', 'beer', 'soda', 'coke', 'wine', 'milk', 'drink', 'coffee', 'tea'])
            
            try:
                # Nettoyage Outliers
                mean = np.mean(food_pts_cm, axis=0)
                std = np.std(food_pts_cm, axis=0)
                mask_clean = np.all(np.abs(food_pts_cm - mean) < 3.0 * std, axis=1)
                clean_pts = food_pts_cm[mask_clean]
                
                if len(clean_pts) < 4:
                    volume_cm3 = 0
                else:
                    hull = ConvexHull(clean_pts)
                    volume_cm3 = hull.volume
                    dims = np.max(clean_pts, axis=0) - np.min(clean_pts, axis=0)
                    print(f"   📐 Geom 3D '{name}': Vol={volume_cm3:.1f} cm3 | BBox={dims[0]:.1f}x{dims[1]:.1f}x{dims[2]:.1f}cm | Pts={len(clean_pts)}")

            except Exception as e:
                print(f"   ⚠️ Hull 3D Error: {e}")
                volume_cm3 = 0
            
            macros = self.fatsecret.search_food(name)
            phys_density = DENSITY_MAP.get(density_str.upper(), 0.85)
            if is_liquid: phys_density = 1.0
            
            weight_g = volume_cm3 * phys_density
            kcals = (weight_g/100)*macros['kcal']
            
            summary["items"].append({
                "name": name, "weight": int(weight_g), "kcal": int(kcals),
                "vol": round(volume_cm3, 1),
                "macros": macros
            })
            summary["total_calories"] += kcals
            
            # Aggrégation Macros Globales
            summary.setdefault("total_protein", 0)
            summary.setdefault("total_carbs", 0)
            summary.setdefault("total_fat", 0)
            summary.setdefault("total_sugar", 0)
            summary.setdefault("total_fiber", 0)
            
            summary["total_protein"] += (weight_g/100)*macros['protein']
            summary["total_carbs"] += (weight_g/100)*macros['carbs']
            summary["total_fat"] += (weight_g/100)*macros['fat']
            summary["total_sugar"] += (weight_g/100)*macros.get('sugar', 0)
            summary["total_fiber"] += (weight_g/100)*macros.get('fiber', 0)

            print(f"🍔 {name}: {int(weight_g)}g ({volume_cm3:.1f} cm3) - {int(kcals)} kcal")
                
        print(f"🔥 TOTAL: {int(summary['total_calories'])} kcal")
        return summary

    def calibrate_scale_3d(self, sam_state, pts3d_map):
        prompts = ["credit card", "payment card", "debit card", "id card", "card", "rectangular card", "transit card", "opus card"]
        REAL_DIAGONAL_CM = np.sqrt(8.56**2 + 5.40**2) # ~10.1 cm
        
        for p in prompts:
            mask = self.get_sam_mask(sam_state, p)
            if mask is None: continue
            
            # Vérifier si le masque est assez grand (bruit)
            if np.sum(mask > 0) < 500: continue
            
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
        prompt = (
            "Analyze this image and list only the EDIBLE FOOD items present. "
            "Ignore plates, bowls, cutlery, cups, tables, and background objects. "
            "For each food item, estimate its physical structure/porosity type: "
            "'AIRY' (for leafy greens, popcorn, chips, puffed items with lots of air gaps), "
            "'PARTICULATE' (for piles of small items like rice, pasta, beans, berries, ground meat where there are small air gaps), "
            "'SOLID' (for single dense objects like an egg, steak, apple, bread slice, cheese block, mashed potatoes). "
            "Return a JSON list: [{'name': 'Grilled Chicken', 'density': 'SOLID'}, ...]. "
            "Do NOT include 'plate' or 'bowl'."
        )
        BLACKLIST = ["plate", "bowl", "dish", "spoon", "fork", "knife", "table", "tray", "cup", "glass", "napkin"]
        
        try:
            res = self.gemini_client.models.generate_content(model="gemini-2.5-flash", contents=[img, prompt])
            raw_items = ast.literal_eval(res.text.strip().replace("```json", "").replace("```", ""))
            
            # Filtrage robuste
            clean_items = []
            for item in raw_items:
                name = item.get("name", "").lower()
                if not any(b in name for b in BLACKLIST):
                    clean_items.append(item)
                else:
                    print(f"   ⚠️ Ignored non-food item: {name}")
            
            return clean_items
        except Exception as e: 
            print(f"Gemini Error: {e}")
            return []

    def save_debug_plot(self, pts3d, colors, filename):
        """
        Saves a 3D scatter plot of the point cloud as a PNG image.
        """
        import matplotlib.pyplot as plt
        
        # Downsample for visualization speed (take 1 point every 100)
        pts = pts3d.reshape(-1, 3)[::100]
        cols = colors.reshape(-1, 3)[::100] / 255.0
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter plot
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Reconstructed 3D Scene')
        
        # Set equal aspect ratio for realistic proportions
        # (Matplotlib 3D doesn't handle aspect ratio perfectly automatically)
        max_range = np.array([pts[:,0].max()-pts[:,0].min(), pts[:,1].max()-pts[:,1].min(), pts[:,2].max()-pts[:,2].min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(pts[:,0].max()+pts[:,0].min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(pts[:,1].max()+pts[:,1].min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(pts[:,2].max()+pts[:,2].min())
        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax.plot([xb], [yb], [zb], 'w')
           
        plt.savefig(filename)
        plt.close()
        print(f"   ✅ Saved {filename}")

if __name__ == "__main__":
    # Exemple d'usage
    # images = ["path/to/img1.jpg", "path/to/img2.jpg"]
    # scanner = NutrientScannerMultiView()
    # scanner.analyze_scene(images)
    pass