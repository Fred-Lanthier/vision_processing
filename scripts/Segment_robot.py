import time
import random
import numpy as np
import re
import gc
from PIL import Image
# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from GroundingDINO_with_Segment_Anything_Utils import *
# SAM 2 imports (changed from SAM 1)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import rospkg
import rospy
import os
import torch

class SegmentRobot:
    def __init__(self, sam2_checkpoint, sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml"):
        """
        SAM 2 initialization
        
        Args:
            sam2_checkpoint: path to SAM 2 .pt file (e.g., "sam2.1_hiera_large.pt")
            sam2_config: path to config yaml file
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detection_list = ['robot']
        
        # Build SAM 2 model
        self.sam2 = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
        self.sam_predictor = SAM2ImagePredictor(self.sam2)
        
        # Tracking variables for optimization
        self.robot_box = None
        self.frame_count = 0
        self.detection_interval = 30
        
        print("✅ SAM 2 ready for segmentation!")

    def grounding_dino_detect(self, image_path, detection_list):
        """Run GroundingDINO detection"""
        image = load_image(image_path)
        detection = detect(
                image=image,
                labels=detection_list,
                threshold=0.2,
                detector_id="IDEA-Research/grounding-dino-tiny"
            )
        return detection
    
    def detect_robot(self, image_np, force_detection=False):
        """
        Optimized robot detection with caching
        
        Args:
            image_np: numpy array of the image (RGB format)
            force_detection: if True, force GroundingDINO to run
        
        Returns:
            dict with mask, score, and computation times
        """
        start_time = time.time()
        
        # Only run GroundingDINO every N frames or if forced
        grounding_time = 0
        if self.frame_count % self.detection_interval == 0 or self.robot_box is None or force_detection:
            grounding_start = time.time()
            
            # Save temporary image for GroundingDINO
            temp_path = "/tmp/temp_detection.png"
            cv2.imwrite(temp_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            
            # Run detection
            detections = self.grounding_dino_detect(temp_path, self.detection_list)
            
            if detections:
                # Get best detection and cache the box
                best_detection = [max(detections, key=lambda x: x.score)]
                bbox = best_detection[0].box
                self.robot_box = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
                print(f"   🔄 Updated robot box at frame {self.frame_count}")
            elif self.robot_box is None:
                print("   ⚠️  No robot detected and no cached box!")
                return None
                
            grounding_time = time.time() - grounding_start

            print("Visualize GroundingDINO results for debugging:")
            self.visualize_grounding_dino_results(temp_path, best_detection)
        # SAM 2 prediction with center point for better accuracy
        sam_start = time.time()
        self.sam_predictor.set_image(image_np)
        
        input_box = np.array([self.robot_box])
        
        # Add center point as extra prompt (improves accuracy)
        center_x = (self.robot_box[0] + self.robot_box[2]) / 2
        center_y = (self.robot_box[1] + self.robot_box[3]) / 2
        input_point = np.array([[center_x, center_y]])
        input_label = np.array([1])  # 1 = foreground
        
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,  # Get 3 mask options
        )
        
        # Pick best mask from the 3 options
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        sam_time = time.time() - sam_start
        
        self.frame_count += 1
        
        result = {
            'mask': best_mask,
            'score': best_score,
            'computation_time': time.time() - start_time,
            'grounding_time': grounding_time,
            'sam_time': sam_time,
            'used_cached_box': grounding_time == 0
        }
        
        return result
    
    def visualize_grounding_dino_results(self, image_path: str, detections: list, 
                                       save_path: str = "grounding_dino_detections.png") -> None:
        """Visualize GroundingDINO detection results"""
        if not detections:
            print("   ⚠️  No detections to visualize")
            return
            
        print(f"🎨 Creating GroundingDINO visualization...")
        
        try:
            image = load_image(image_path)
            image_array = np.array(image)
            plot_detections(image_array, detections, save_path)
            print(f"   ✅ GroundingDINO visualization saved: {save_path}")
        except Exception as e:
            print(f"   ❌ Error creating GroundingDINO visualization: {e}")
        
def main():
    # Load rospack to get package paths
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')

    # SAM 2 model paths
    sam2_checkpoint = os.path.expanduser("~/segment-anything-2/checkpoints/sam2.1_hiera_base_plus.pt")
    
    # Choose config based on model size:
    # sam2_hiera_t.yaml = tiny
    # sam2_hiera_s.yaml = small  
    # sam2_hiera_b+.yaml = base plus
    # sam2_hiera_l.yaml = large
    sam2_config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    
    # Initialize SegmentRobot with SAM 2
    segment_robot = SegmentRobot(sam2_checkpoint, sam2_config)
    
    # List of image numbers to process
    # image_numbers = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    image_numbers = [1]    
    print("🎬 Processing multiple images with SAM 2...")
    
    for img_num in image_numbers:
        # Build the image filename with zero padding
        image_filename = f'static_rgb_step_{img_num:06d}.png'
        image_filename = 'Test_2.jpeg'
        image_path = os.path.join(package_path, 'scripts', 'images', image_filename)
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"   ⚠️  Skipping {image_filename} - file not found")
            continue
        
        print(f"\n📸 Processing {image_filename}...")
        
        # Load image
        image = load_image(image_path)
        image_np = np.array(image)
        
        # Detect robot
        robot_detection = segment_robot.detect_robot(image_np)
        if robot_detection is None:
            print(f"   ❌ No robot detected in {image_filename}")
            continue
        
        status = "CACHED BOX" if robot_detection['used_cached_box'] else "FULL DETECTION"
        print(f"   ✅ Detection done: {robot_detection['computation_time']:.3f}s ({status})")
        print(f"      GroundingDINO: {robot_detection['grounding_time']:.3f}s")
        print(f"      SAM 2: {robot_detection['sam_time']:.3f}s")
        
        # Visualize robot mask
        mask = robot_detection['mask']
        result_image = image_np.copy()
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        result_image[mask > 0] = [0, 255, 0]  # Green for robot
        
        # Save with same numbering
        output_filename = f'robot_segmentation_{img_num:06d}.png'
        output_path = os.path.join(package_path, 'scripts', output_filename)
        cv2.imwrite(output_path, result_image)
        print(f"   💾 Saved: {output_filename}")

if __name__ == "__main__":
    main()