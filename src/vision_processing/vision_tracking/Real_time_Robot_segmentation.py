import os
import torch
import numpy as np
import cv2
import shutil
import gc
from PIL import Image
import rospkg 

# --- IMPORTS ---
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam2.build_sam import build_sam2_video_predictor

class SegmentRobotSAM2:
    def __init__(self, temp_work_dir="/tmp/robot_tracking_live"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing SegmentRobotSAM2 on {self.device}...")

        # --- CONFIGURATION CHEMINS ---
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('vision_processing')
        
        self.pcd_output_dir = os.path.join(package_path, "scripts", "Robot_pcd_filtered")
        if not os.path.exists(self.pcd_output_dir):
            os.makedirs(self.pcd_output_dir)

        self.sam2_checkpoint = os.path.expanduser("~/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt")
        self.sam2_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.text_prompt = "robot"

        # --- SESSION ---
        self.frame_count = 0
        self.session_frame_idx = 0
        self.MAX_MEMORY_FRAMES = 100
        self.last_valid_mask = None
        self.inference_state = None
        
        self.video_height = None
        self.video_width = None
        
        # --- OPTIMISATION CPU : CACHE POUR LA GÉOMÉTRIE ---
        self.cached_grids = {} # Stockera les meshgrids pré-calculés
        # Intrinsèques (Fixes pour une caméra donnée)
        self.fx = 616.1005249
        self.fy = 615.82617188
        self.cx = 318.38803101
        self.cy = 249.23504639
        
        # Temp folder
        self.work_dir = temp_work_dir
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
        os.makedirs(self.work_dir)

        # --- LOAD SAM 2 ---
        print(f"⚡ Loading SAM 2 ({self.sam2_config})...")
        self.sam2_predictor = build_sam2_video_predictor(self.sam2_config, self.sam2_checkpoint, device=self.device)

        # --- SAM 3 ---
        self.sam3_model = None
        self.sam3_processor = None
        self._load_sam3_and_warmup()

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

    def _load_sam3_and_warmup(self):
        if self.sam3_model is None or self.sam3_processor is None:
            print("🧠 Loading & Warming up SAM 3...")
            self.sam3_model = build_sam3_image_model()
            self.sam3_processor = Sam3Processor(self.sam3_model)
            try:
                dummy = Image.new('RGB', (640, 480), 'black')
                st = self.sam3_processor.set_image(dummy)
                _ = self.sam3_processor.set_text_prompt(state=st, prompt="test")
                print("✅ SAM 3 Ready.")
            except: pass
            # self._unload_sam3()

    def _unload_sam3(self):
        del self.sam3_processor
        del self.sam3_model
        self.sam3_processor = None
        self.sam3_model = None
        gc.collect()
        torch.cuda.empty_cache()

    def process_new_frame(self, cv_image, depth_array=None):
        # 1. Sauvegarde (Seulement Init et Reset)
        # On évite d'écrire sur le disque à chaque frame
        if self.frame_count == 0 or (self.frame_count % self.MAX_MEMORY_FRAMES == 0):
            # Logique pour nommer correctement en cas de reset
            if self.frame_count > 0:
                # Si reset, on nettoie d'abord et on écrit 00000.jpg
                for f in os.listdir(self.work_dir):
                    try: os.remove(os.path.join(self.work_dir, f))
                    except: pass
                filename = "00000.jpg"
            else:
                filename = f"{self.frame_count:05d}.jpg"
                
            file_path = os.path.join(self.work_dir, filename)
            cv2.imwrite(file_path, cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 2. Logique Reset
        is_resetting = False
        if self.frame_count > 0 and (self.frame_count % self.MAX_MEMORY_FRAMES == 0):
            # print(f"♻️  MEMORY RESET...")
            is_resetting = True
            self.session_frame_idx = 0

        # 3. Execution
        result = None
        
        # CAS A : Init / Reset / Crash Recovery
        if self.frame_count == 0 or is_resetting or self.inference_state is None:
            prior_mask = self.last_valid_mask if is_resetting else None
            
            result = self._initialize_tracking(cv_image, depth_array, prior_mask=prior_mask)
            
            # Fallback SAM 3 si le reset rapide échoue
            if not result["success"] and is_resetting:
                result = self._initialize_tracking(cv_image, depth_array, prior_mask=None)
                
            if not result["success"]: return result

        # CAS B : Tracking Normal
        else:
            result = self._update_tracking(cv_image, depth_array)

        if result["success"]:
            self.last_valid_mask = result["mask"]
        
        self.frame_count += 1
        self.session_frame_idx += 1
        return result

    def _initialize_tracking(self, cv_image, depth_array, prior_mask=None):
        mask_np = None

        if prior_mask is not None:
            mask_np = prior_mask
        else:
            print(f"   🏁 SAM 3: Recherche '{self.text_prompt}'...")
            if self.sam3_model is None: self._load_sam3_and_warmup()

            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            try:
                inf_state = self.sam3_processor.set_image(pil_image)
                out = self.sam3_processor.set_text_prompt(state=inf_state, prompt=self.text_prompt)
                initial_mask = out["masks"][0]
                if initial_mask is not None:
                    mask_np = initial_mask.detach().cpu().numpy().squeeze()
            except Exception as e:
                print(f"❌ SAM 3 Error: {e}")
            
            # self._unload_sam3()

        if mask_np is None: return {"success": False}

        try:
            self.inference_state = self.sam2_predictor.init_state(video_path=self.work_dir)
            if isinstance(self.inference_state["images"], torch.Tensor):
                self.inference_state["images"] = [t for t in self.inference_state["images"]]
            
            if len(self.inference_state["images"]) > 0:
                self.video_height, self.video_width = self.inference_state["images"][0].shape[-2:]

            mask_input = torch.from_numpy(mask_np).bool().to(self.device)
            if mask_input.ndim > 2: mask_input = mask_input.squeeze()

            self.sam2_predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=1,
                mask=mask_input
            )
        except Exception as e:
             return {"success": False}
        
        return self._generate_output(mask_np, depth_array)

    def _update_tracking(self, cv_image, depth_array):
        target_h, target_w = self.video_height, self.video_width
        
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h))
        
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.to(self.device)
        img_tensor = (img_tensor - self.mean) / self.std
        
        self.inference_state["images"].append(img_tensor)
        self.inference_state["num_frames"] = len(self.inference_state["images"])

        # Nettoyage RAM
        if len(self.inference_state["images"]) > 6:
             idx_to_clear = len(self.inference_state["images"]) - 7
             if self.inference_state["images"][idx_to_clear] is not None:
                 self.inference_state["images"][idx_to_clear] = None

        current_idx = self.session_frame_idx
        mask_np = None
        
        try:
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                for _, _, video_res_masks in self.sam2_predictor.propagate_in_video(
                    self.inference_state,
                    start_frame_idx=current_idx,
                    max_frame_num_to_track=1
                ):
                    raw_mask = (video_res_masks[0] > 0).cpu().numpy()
                    mask_np = raw_mask.squeeze()
                    if mask_np.ndim > 2: mask_np = mask_np[0]
                    break 
        except Exception as e:
            return {"success": False}
        
        return self._generate_output(mask_np, depth_array)

    def _generate_output(self, mask_np, depth_array):
        if mask_np is None: return {"success": False}

        if depth_array is not None:
             orig_h, orig_w = depth_array.shape
             if mask_np.shape != (orig_h, orig_w):
                 mask_np = cv2.resize(mask_np.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)

        ys, xs = np.where(mask_np)
        centroid = (int(np.mean(xs)), int(np.mean(ys))) if len(xs) > 0 else (0,0)

        pcd_path = ""
        if depth_array is not None:
            pcd_path = self._compute_cartesian_pcd(depth_array, mask_np)

        return {
            "success": True,
            "mask": mask_np,
            "centroid": centroid,
            "pcd_path": pcd_path
        }

    def _compute_cartesian_pcd(self, depth_map, mask_2d):
        """
        VERSION OPTIMISÉE : Utilise un cache pour les grilles U/V
        """
        combined_mask = mask_2d & (depth_map > 0)
        if np.sum(combined_mask) == 0: return ""

        h, w = depth_map.shape
        
        # --- CACHE CHECK ---
        # Si on n'a pas encore calculé les grilles pour cette résolution, on le fait
        if (h, w) not in self.cached_grids:
            # print(f"⚙️  Calcul et Cache des grilles 3D pour {w}x{h}")
            v_grid, u_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            # Pré-calcul partiel : (u - cx) / fx et (v - cy) / fy
            # Comme ça on n'a plus qu'à multiplier par Z ensuite
            x_factor = (u_grid - self.cx) / self.fx
            y_factor = (v_grid - self.cy) / self.fy
            
            self.cached_grids[(h, w)] = (x_factor, y_factor)

        # --- UTILISATION DU CACHE ---
        x_factor, y_factor = self.cached_grids[(h, w)]
        
        # Extraction Vectorisée (Seulement les points utiles)
        z_mm = depth_map[combined_mask]
        x_fac_val = x_factor[combined_mask]
        y_fac_val = y_factor[combined_mask]

        z_m = z_mm / 1000.0
        x_m = x_fac_val * z_m
        y_m = y_fac_val * z_m

        pcd = np.column_stack((x_m, y_m, z_m))

        filename = f"robot_cloud_latest.npy"
        save_path = os.path.join(self.pcd_output_dir, filename) 
        
        # Sauvegarde (C'est la seule partie qui prend du temps maintenant)
        np.save(save_path, pcd)
        
        return save_path