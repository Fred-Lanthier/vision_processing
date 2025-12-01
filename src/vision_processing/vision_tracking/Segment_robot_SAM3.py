import time
import random
import numpy as np
import re
import gc
from PIL import Image
# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from GroundingDINO_with_Segment_Anything_Utils import *
# SAM 3 imports
from sam3.model_builder import build_sam3_image_model, build_sam3_video_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor
import cv2
import os
import torch
import glob
import sys

# Try importing rospkg, but handle failure gracefully
try:
    import rospkg
    ROSPKG_AVAILABLE = True
except ImportError:
    ROSPKG_AVAILABLE = False
    print("⚠️ rospkg not found. Using fallback paths.")

class SegmentRobot:
    def __init__(self):
        """
        SAM 3 initialization
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detection_list = ['robot']
        
        # Build SAM 3 Video Model
        print(f"🚀 [SAM3] Initializing Video Model on {self.device}...")
        # Initialize predictor directly as per user feedback/fix
        self.video_predictor = build_sam3_video_predictor()
        print("✅ [SAM3] Video Model loaded.")
        
        # Tracking variables for optimization
        self.frame_count = 0
        
    def process_video_sequence(self, images_dir, output_dir, prompt="robot"):
        """
        Process a sequence of images as a video using SAM 3
        """
        # Get all images in directory, sorted
        image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")) + 
                             glob.glob(os.path.join(images_dir, "*.jpg")) +
                             glob.glob(os.path.join(images_dir, "*.jpeg")))
        
        if not image_files:
            print(f"❌ No images found in {images_dir}")
            return

        num_frames = len(image_files)
        print(f"🎬 Starting video session for {num_frames} frames...")
        
        start_total_time = time.time()
        
        # Start session
        try:
            self.inference_state = self.video_predictor.start_session(images_dir)
            session_id = self.inference_state['session_id']
            print(f"   ✅ Session started. ID: {session_id}")
        except Exception as e:
            print(f"❌ Error starting session: {e}")
            return

        # Add prompt to the first frame (frame_idx=0)
        print(f"   ➕ Adding text prompt '{prompt}' to first frame...")
        try:
            self.video_predictor.add_prompt(
                session_id=session_id, 
                frame_idx=0,
                text=prompt
            )
        except Exception as e:
             print(f"❌ Error adding prompt: {e}")
             return

        # Propagate through the video
        print("   🌊 Propagating mask...")
        
        try:
            # Propagate forward from frame 0
            video_segments = self.video_predictor.propagate_in_video(
                session_id=session_id,
                propagation_direction="forward",
                start_frame_idx=0,
                max_frame_num_to_track=num_frames
            )
        except Exception as e:
             print(f"❌ Error propagating: {e}")
             return
             
        # Process and save results
        print("   💾 Saving results...")
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        processed_count = 0
        
        # Measure inference FPS separately
        inference_start_time = time.time()
        inference_frames = 0
        pure_inference_time = 0
        
        # Enable Mixed Precision for speed
        use_half = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_half else torch.float16
        print(f"⚡ Mixed Precision Enabled: {amp_dtype}")

        # Buffer for results
        results_buffer = []

        step_start = time.time()
        
        # Wrap loop in autocast
        with torch.autocast(device_type=self.device, dtype=amp_dtype):
            for step_result in video_segments:
                # Accumulate pure inference time (time spent waiting for generator)
                pure_inference_time += (time.time() - step_start)
                inference_frames += 1
                
                # Extract data from dictionary
                out_frame_idx = step_result['frame_index']
                outputs = step_result['outputs']
                out_mask_logits = outputs['out_binary_masks']
                
                # Store minimal data in buffer
                # We need to move tensor to CPU/Numpy to save GPU memory if buffer gets large
                # But for 120 frames it's fine. 
                # Let's convert to numpy boolean immediately to save space.
                
                if isinstance(out_mask_logits, torch.Tensor):
                    masks = out_mask_logits.detach().cpu().numpy()
                else:
                    masks = out_mask_logits
                
                # Process mask to get single boolean mask
                best_mask = None
                if masks.ndim > 2:
                     best_mask = masks[0] 
                else:
                     best_mask = masks
                     
                if best_mask.ndim > 2:
                    best_mask = np.squeeze(best_mask)
                
                best_mask = best_mask > 0
                
                results_buffer.append({
                    'idx': out_frame_idx,
                    'mask': best_mask
                })
                
                processed_count += 1
                
                # Reset step start for next iteration
                step_start = time.time()

        total_inference_time = pure_inference_time
        inference_fps = inference_frames / total_inference_time if total_inference_time > 0 else 0
        
        print(f"\n✅ Inference complete!")
        print(f"   Frames processed: {processed_count}")
        print(f"   Inference Time: {total_inference_time:.3f}s")
        print(f"   Inference FPS: {inference_fps:.2f}")
        
        # --- Pass 2: Save Images ---
        print("\n💾 Saving results to disk...")
        save_start_time = time.time()
        
        for item in results_buffer:
            frame_idx = item['idx']
            mask = item['mask']
            
            # Handle frame index
            if isinstance(frame_idx, int):
                image_path = image_files[frame_idx]
            else:
                image_path = os.path.join(images_dir, str(frame_idx))
                
            image_name = os.path.basename(image_path)
            
            image = cv2.imread(image_path)
            if image is None: continue
            
            image[mask] = [0, 255, 0]
            output_filename = f"video_seg_{image_name}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, image)
            
        save_time = time.time() - save_start_time
        print(f"   Save Time: {save_time:.3f}s")
        
        total_time = time.time() - inference_start_time
        effective_fps = processed_count / total_time if total_time > 0 else 0
        
        print(f"\n✅ All done!")
        print(f"   Total time (Inference + I/O): {total_time:.3f}s")
        print(f"   Effective FPS: {effective_fps:.2f}")

def main():
    # Determine package path
    if ROSPKG_AVAILABLE:
        try:
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('vision_processing')
        except Exception as e:
            print(f"⚠️ Error using rospkg: {e}")
            package_path = "/home/flanthier/Github/src/vision_processing"
    else:
        package_path = "/home/flanthier/Github/src/vision_processing"
        
    print(f"📂 Package path: {package_path}")
    
    # Input directory: images_trajectory
    images_dir = os.path.join(package_path, 'scripts', 'images_trajectory')
    output_dir = os.path.join(package_path, 'scripts', 'results_trajectory')
    
    if not os.path.exists(images_dir):
         print(f"❌ Image directory not found: {images_dir}")
         # Fallback to images for testing if trajectory doesn't exist
         images_dir = os.path.join(package_path, 'scripts', 'images')
         print(f"⚠️ Falling back to: {images_dir}")
    
    # Initialize SegmentRobot
    segment_robot = SegmentRobot()
    
    # Run video processing
    segment_robot.process_video_sequence(images_dir, output_dir)

if __name__ == "__main__":
    main()