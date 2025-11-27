import os
import torch
import numpy as np
import cv2
import shutil
import gc
from PIL import Image

# --- IMPORTS ---
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam2.build_sam import build_sam2_video_predictor

class SegmentRobotSAM2:
    def __init__(self, temp_work_dir="/tmp/robot_tracking_live"):
        """
        Initialize models.
        SAM 2 is loaded immediately.
        SAM 3 is loaded only for the first frame then deleted to save VRAM.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing SegmentRobotSAM2 on {self.device}...")

        # --- CONFIGURATION ---
        # Adjust these paths to match your system
        self.sam2_checkpoint = os.path.expanduser("~/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt")
        self.sam2_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.text_prompt = "robot"

        # --- SESSION MANAGEMENT ---
        self.frame_count = 0
        self.is_tracking = False
        self.inference_state = None
        
        # Temp folder to mimic a "video" for SAM 2
        self.work_dir = temp_work_dir
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
        os.makedirs(self.work_dir)

        # --- LOAD SAM 2 (Persistent) ---
        print(f"⚡ Loading SAM 2 ({self.sam2_config})...")
        self.sam2_predictor = build_sam2_video_predictor(self.sam2_config, self.sam2_checkpoint, device=self.device)

        # We will load SAM 3 only when the first frame arrives
        self.sam3_model = None
        self.sam3_processor = None

    def _load_sam3(self):
        print("🧠 Loading SAM 3 for initialization...")
        self.sam3_model = build_sam3_image_model()
        self.sam3_processor = Sam3Processor(self.sam3_model)

    def _unload_sam3(self):
        print("🧹 Unloading SAM 3 to free VRAM...")
        del self.sam3_processor
        del self.sam3_model
        self.sam3_processor = None
        self.sam3_model = None
        gc.collect()
        torch.cuda.empty_cache()

    def process_new_frame(self, cv_image, depth_array=None):
        """
        Main entry point for real-time processing.
        cv_image: numpy array (BGR)
        depth_array: numpy array (H, W) in mm (Optional)
        """
        # 1. Save current frame to disk (SAM 2 Video API requires file paths)
        filename = f"{self.frame_count:05d}.jpg"
        file_path = os.path.join(self.work_dir, filename)
        cv2.imwrite(file_path, cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        result = {
            "mask": None,
            "centroid": None,
            "pcd_path": None,
            "frame_idx": self.frame_count
        }

        # --- LOGIC BRANCHING ---
        if self.frame_count == 0:
            result = self._initialize_tracking(file_path, cv_image, depth_array)
        else:
            result = self._update_tracking(file_path, depth_array)

        self.frame_count += 1
        return result

    def _initialize_tracking(self, file_path, cv_image, depth_array):
        print(f"🏁 Frame 0: Initializing tracking for '{self.text_prompt}'...")
        
        # 1. Run SAM 3 (Image Mode)
        self._load_sam3()
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        inf_state = self.sam3_processor.set_image(pil_image)
        out = self.sam3_processor.set_text_prompt(state=inf_state, prompt=self.text_prompt)
        initial_mask = out["masks"][0] # Tensor

        if initial_mask is None:
            print("❌ SAM 3 failed to find the robot.")
            return None

        # 2. Inject into SAM 2
        # Initialize state with ONLY the first frame
        self.inference_state = self.sam2_predictor.init_state(video_path=self.work_dir)
        
        # Convert mask for SAM 2
        mask_np = initial_mask.detach().cpu().numpy().squeeze()
        mask_input = torch.from_numpy(mask_np > 0).bool().to(self.device)
        
        self.sam2_predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=0,
            obj_id=1,
            mask=mask_input
        )

        # 3. Clean up
        self._unload_sam3()
        self.is_tracking = True

        # 4. Generate Output
        return self._generate_output(mask_np, depth_array, 0)

    def _update_tracking(self, new_file_path, depth_array):
        """
        Trick to stream data into SAM 2:
        We manually append the new image path to the internal state 
        and run propagation only on the new frame.
        """
        # 1. Update Internal State of SAM 2
        # self.inference_state is a dictionary containing 'images' list
        self.inference_state["images"].append(new_file_path)
        # Update metadata if necessary (usually num_frames)
        # Note: This relies on SAM 2 implementation details. 
        # If 'video_height'/'video_width' are stored, they remain constant.

        current_idx = self.frame_count
        
        # 2. Propagate ONLY on the new frame
        # We start at current_idx and track for 1 frame
        mask_np = None
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            for _, _, video_res_masks in self.sam2_predictor.propagate_in_video(
                self.inference_state,
                start_frame_idx=current_idx,
                max_frame_num_to_track=1
            ):
                # Extract Mask
                raw_mask = (video_res_masks[0] > 0).cpu().numpy()
                mask_np = raw_mask.squeeze()
                if mask_np.ndim > 2: mask_np = mask_np[0]
                break # We only expect one result here
        
        return self._generate_output(mask_np, depth_array, current_idx)

    def _generate_output(self, mask_np, depth_array, frame_idx):
        if mask_np is None: return {"success": False}

        # Calculate Centroid
        ys, xs = np.where(mask_np)
        if len(xs) > 0:
            centroid = (int(np.mean(xs)), int(np.mean(ys)))
        else:
            centroid = (0, 0)

        # Generate PCD if depth is present
        pcd_path = None
        if depth_array is not None:
            pcd_path = self._compute_cartesian_pcd(depth_array, mask_np, frame_idx)

        return {
            "success": True,
            "mask": mask_np,
            "centroid": centroid,
            "pcd_path": pcd_path
        }

    def _compute_cartesian_pcd(self, depth_map, mask_2d, frame_idx):
        # Camera Intrinsics
        fx = 616.1005249
        fy = 615.82617188
        cx = 318.38803101
        cy = 249.23504639

        # Filter valid pixels
        combined_mask = mask_2d & (depth_map > 0)
        if np.sum(combined_mask) == 0: return None

        # Vectorized Projection
        h, w = depth_map.shape
        v_grid, u_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        z_mm = depth_map[combined_mask]
        u = u_grid[combined_mask]
        v = v_grid[combined_mask]

        z_m = z_mm / 1000.0
        x_m = (u - cx) * z_m / fx
        y_m = (v - cy) * z_m / fy

        pcd = np.column_stack((x_m, y_m, z_m))

        # Save
        filename = f"robot_cloud_{frame_idx:04d}.npy"
        save_path = os.path.join("Robot_pcd_filtered", filename) # Saves relative to script exec
        # Ensure dir exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, pcd)
        
        return save_path