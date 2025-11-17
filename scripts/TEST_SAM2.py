#!/home/flanthier/Documents/src/vision_processing/venv_py310/bin/python3
"""
Fixed Single Food Piece Detector
MODIFIED: Uses SAM 2 instead of SAM 1
"""

import time
import random
import numpy as np
import re
import gc
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from GroundingDINO_with_Segment_Anything_Utils import *
# SAM 2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import torch
import os

class SingleFoodDetector:
    def __init__(self, sam2_checkpoint, sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml", output_dir=None):
        """Initialize the single food detector with SAM 2"""
        print("🚀 Initializing Single Food Detector with SAM 2...")
        
        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.expanduser("~/Documents/src/vision_processing/scripts")
        else:
            self.output_dir = os.path.expanduser(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Output directory: {self.output_dir}")

        # Load LLaVA Next model
        print("📥 Loading LLaVA-NeXT model...")
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"✅ LLaVA loaded on {self.device}")

        # Load SAM 2 model
        print("🎯 Loading SAM 2 model...")
        self.sam2 = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
        self.sam_predictor = SAM2ImagePredictor(self.sam2)
        print("✅ SAM 2 ready for segmentation!")

    def _get_output_path(self, filename):
        """Helper to get full output path"""
        return os.path.join(self.output_dir, filename)
    
    def clear_gpu_memory(self):
        """Clear GPU memory to prevent OOM errors"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def describe_plate(self, image_path: str) -> list:
        print("🧠 Describing the plate with LLaVA...")
        llava_start_time = time.time()
        
        image = Image.open(image_path)
        
        prompt = """[INST] <image>
    Look at this image and identify all the different types of food you can see. 
    List each DISTINCT food type only once, using clear, simple names:

    - food type 1
    - food type 2
    - food type 3

    Be specific but concise (e.g., "chicken breast", "cherry tomatoes", "rice"). Only list each food category once. 
    Only list items that are clearly edible food, not stickers or labels or qr code. [/INST]"""
        
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3
            )
        
        llava_time = time.time() - llava_start_time
        print(f"   🕒 LlaVA description completed in {llava_time:.2f} seconds")

        response = self.processor.decode(output[0], skip_special_tokens=True)
        
        if '[/INST]' in response:
            model_response = response.split('[/INST]')[-1].strip()
        else:
            model_response = response.strip()
        
        # Extract bullet points from model response only
        lines = model_response.split('\n')
        food_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('*'):
                line_lower = line.lower().strip()
                if line_lower in ['- food type 1', '- food type 2', '- food type 3']:
                    continue
                food_lines.append(line)
        
        # Simple but robust parsing
        cleaned_items = []
        seen = set()
        
        for line in food_lines:
            item = line.lstrip('- *').strip()
            
            if ':' in item:
                parts = item.split(':', 1)
                if len(parts) == 2:
                    prefix = parts[0].strip().lower()
                    food_name = parts[1].strip()
                    
                    if any(template in prefix for template in ['food type', 'food item', 'item', 'type']):
                        cleaned_food = food_name
                    else:
                        cleaned_food = item
                else:
                    cleaned_food = item
            else:
                cleaned_food = item
            
            cleaned_food = re.sub(r'[^\w\s]', '', cleaned_food).strip()
            cleaned_food = re.sub(r'\s+', ' ', cleaned_food)
            
            if (len(cleaned_food) >= 3 and 
                cleaned_food.lower() not in seen and
                cleaned_food.lower() not in ['food', 'type', 'item']):
                
                seen.add(cleaned_food.lower())
                cleaned_items.append(cleaned_food)
        
        if not cleaned_items:
            cleaned_items = ["food"]

        s = self.simplify_food_names(cleaned_items)
        return s

    def simplify_food_names(self, food_items: list) -> list:
        food_simplification_rules = {
            'chicken': {
                'chicken cubes': 'chicken cubes',
                'chicken pieces': 'chicken pieces',
                'chicken chunks': 'chicken chunks',
                'chicken breast': 'chicken breast',
                'grilled chicken': 'chicken',
                'roasted chicken': 'chicken',
                'chicken thigh': 'chicken thigh',
            },
            'beef': {
                'beef cubes': 'beef cubes',
                'beef chunks': 'beef chunks', 
                'ground beef': 'ground beef',
                'beef steak': 'steak',
                'grilled steak': 'steak',
            },
            'carrots': {
                'carrot slices': 'carrot slices',
                'carrot sticks': 'carrot sticks',
                'baby carrots': 'baby carrots',
                'diced carrots': 'diced carrots',
                'roasted carrots': 'carrots',
            },
            'broccoli': {
                'broccoli florets': 'broccoli florets',
                'steamed broccoli': 'broccoli',
            },
            'potatoes': {
                'potato wedges': 'potato wedges',
                'potato cubes': 'potato cubes',
                'mashed potatoes': 'mashed potatoes',
                'baked potatoes': 'potatoes',
                'roasted potatoes': 'potatoes',
            },
            'rice': {
                'rice grains': 'rice',
                'white rice': 'rice',
                'brown rice': 'rice', 
                'rice pilaf': 'rice',
            },
        }
        
        cooking_descriptors_to_remove = [
            'grilled', 'roasted', 'baked', 'fried', 'steamed', 'sauteed',
            'fresh', 'organic', 'cooked', 'raw', 'for garnish'
        ]
        
        simplified_items = []
        seen = set()
        
        for food_item in food_items:
            food_lower = food_item.lower().strip()
            simplified_name = None
            
            for base_food, rules in food_simplification_rules.items():
                for pattern, simplified in rules.items():
                    if pattern in food_lower:
                        simplified_name = simplified
                        break
                if simplified_name:
                    break
            
            if not simplified_name:
                cleaned = food_lower
                for descriptor in cooking_descriptors_to_remove:
                    cleaned = cleaned.replace(descriptor, '').strip()
                cleaned = re.sub(r'\s+', ' ', cleaned)
                cleaned = re.sub(r'\b(and|with|on|for)\b', '', cleaned)
                cleaned = cleaned.strip()
                
                if cleaned != food_lower and cleaned:
                    simplified_name = cleaned
                else:
                    simplified_name = food_item
            
            if simplified_name and simplified_name.lower() not in seen:
                simplified_items.append(simplified_name)
                seen.add(simplified_name.lower())
        
        return simplified_items

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1.xyxy
        x2_min, y2_min, x2_max, y2_max = box2.xyxy
        
        intersection_x_min = max(x1_min, x2_min)
        intersection_y_min = max(y1_min, y2_min)
        intersection_x_max = min(x1_max, x2_max)
        intersection_y_max = min(y1_max, y2_max)
        
        if intersection_x_max <= intersection_x_min or intersection_y_max <= intersection_y_min:
            return 0.0
        
        intersection_area = (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min)
        
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def smart_duplicate_removal(self, detections: list,
                               confidence_threshold: float = 0.35,
                               same_class_iou: float = 0.65,
                               cross_class_iou: float = 0.4) -> list:
        """Smart duplicate removal that preserves actual food counts"""
        print(f"🔧 Smart filtering of {len(detections)} detections...")
        
        high_conf_detections = [d for d in detections if d.score >= confidence_threshold]
        print(f"   ✅ After confidence filter (≥{confidence_threshold}): {len(high_conf_detections)}")
        
        if not high_conf_detections:
            print("   ⚠️  No high-confidence detections, lowering threshold...")
            high_conf_detections = [d for d in detections if d.score >= 0.25]
        
        filtered_detections = []
        detections_sorted = sorted(high_conf_detections, key=lambda x: x.score, reverse=True)
        
        for detection in detections_sorted:
            is_duplicate = False
            
            for selected in filtered_detections:
                iou = self.calculate_iou(detection.box, selected.box)
                
                if detection.label == selected.label and iou > same_class_iou:
                    is_duplicate = True
                    break
                elif detection.label != selected.label and iou > cross_class_iou:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_detections.append(detection)
        
        print(f"   ✅ After duplicate removal: {len(filtered_detections)}")
        final_detections = sorted(filtered_detections, key=lambda x: x.score, reverse=True)
        return final_detections

    def localize_food(self, image_path: str, food_items: list) -> list:
        """Step 2: Use GroundingDINO to localize the food with smart filtering"""
        print("📍 Localizing food with GroundingDINO...")
        image = load_image(image_path)
        labels = [f"{food}." for food in food_items]
        
        print(f"   🔍 Searching for labels: {labels}")
        
        try:
            self.clear_gpu_memory()
            
            raw_detections = detect(
                image=image,
                labels=labels,
                threshold=0.2,
                detector_id="IDEA-Research/grounding-dino-tiny"
            )
            
            print(f"   📊 Raw detections found: {len(raw_detections)}")
            
            # Apply smart filtering
            clean_detections = self.smart_duplicate_removal(raw_detections)
            
            print(f"   ✅ Final localized regions: {len(clean_detections)}")
            
            # Print detection details for debugging
            if clean_detections:
                print("   📋 Final detection details:")
                for i, detection in enumerate(clean_detections):
                    print(f"      {i+1}. Label: '{detection.label}' | Confidence: {detection.score:.3f}")
            
            return clean_detections
            
        except Exception as e:
            print(f"   ❌ Error during localization: {e}")
            return []

    def visualize_grounding_dino_results(self, image_path: str, detections: list, 
                                       save_path: str = "grounding_dino_detections.png") -> None:
        """Visualize GroundingDINO detection results with bounding boxes and food labels"""
        if not detections:
            print("   ⚠️  No detections to visualize")
            return
            
        print(f"🎨 Creating GroundingDINO visualization...")
        
        # Use output directory
        full_save_path = self._get_output_path(save_path)
        
        try:
            image = load_image(image_path)
            image_array = np.array(image)
            
            try:
                plot_detections(image_array, detections, full_save_path)
                print(f"   ✅ GroundingDINO visualization saved: {full_save_path}")
            except:
                self._create_custom_detection_visualization(image_array, detections, full_save_path)
                print(f"   ✅ GroundingDINO visualization saved: {full_save_path}")
            
            food_groups = {}
            for detection in detections:
                food_type = detection.label.replace('.', '').strip()
                if food_type not in food_groups:
                    food_groups[food_type] = []
                food_groups[food_type].append(detection)
            
            print(f"   📋 GroundingDINO Results Summary:")
            for food_type, items in food_groups.items():
                print(f"      🥗 {food_type.title()}: {len(items)} regions (confidence: {[f'{item.score:.3f}' for item in items]})")
            
        except Exception as e:
            print(f"   ❌ Error creating GroundingDINO visualization: {e}")

    def _create_custom_detection_visualization(self, image_array: np.ndarray, detections: list, save_path: str):
        """Custom visualization method as fallback"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_array)
        
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta']
        food_colors = {}
        
        for i, detection in enumerate(detections):
            bbox = detection.box
            food_type = detection.label.replace('.', '').strip()
            confidence = detection.score
            
            if food_type not in food_colors:
                food_colors[food_type] = colors[len(food_colors) % len(colors)]
            
            color = food_colors[food_type]
            
            rect = patches.Rectangle(
                (bbox.xmin, bbox.ymin), 
                bbox.xmax - bbox.xmin, 
                bbox.ymax - bbox.ymin,
                linewidth=2, 
                edgecolor=color, 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            label = f"{food_type} ({confidence:.2f})"
            ax.text(bbox.xmin, bbox.ymin - 5, label, 
                   fontsize=10, color=color, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlim(0, image_array.shape[1])
        ax.set_ylim(image_array.shape[0], 0)
        ax.axis('off')
        plt.title('GroundingDINO Food Detection Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def select_random_food(self, food_items: list) -> str:
        """Step 3: Select one random food type"""
        selected_food = random.choice(food_items)
        print(f"🎲 Selected food type: '{selected_food}'")
        return selected_food
    
    def select_random_food_from_detections(self, detections: list) -> str:
        """Select one random food type from actual GroundingDINO detections"""
        if not detections:
            raise ValueError("No detections available to select from")
        
        detected_food_types = []
        for detection in detections:
            food_type = detection.label.strip(".").strip()
            if food_type not in detected_food_types:
                detected_food_types.append(food_type)
        
        selected_food = random.choice(detected_food_types)
        print(f"🎲 Selected food type from detections: '{selected_food}'")
        print(f"   📋 Available detected food types: {detected_food_types}")
        return selected_food
    
    def detect_overlaps(self, detections):
        """Enhanced overlap detection - finds smaller items inside larger ones"""
        if len(detections) < 2:
            return []
        
        print("   🔍 Detecting overlapping food items...")
        overlaps = []
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections):
                if i >= j:
                    continue
                    
                area1 = (det1.box.xmax - det1.box.xmin) * (det1.box.ymax - det1.box.ymin)
                area2 = (det2.box.xmax - det2.box.xmin) * (det2.box.ymax - det2.box.ymin)
                
                x_overlap = max(0, min(det1.box.xmax, det2.box.xmax) - max(det1.box.xmin, det2.box.xmin))
                y_overlap = max(0, min(det1.box.ymax, det2.box.ymax) - max(det1.box.ymin, det2.box.ymin))
                intersection = x_overlap * y_overlap
                
                if intersection == 0:
                    continue
                    
                smaller_area = min(area1, area2)
                overlap_ratio = intersection / smaller_area
                
                if overlap_ratio > 0.6:
                    if area1 < area2:
                        overlaps.append({'small': det1, 'large': det2, 'overlap_ratio': overlap_ratio})
                        print(f"      📦 Overlap detected: {det1.label.replace('.', '')} inside {det2.label.replace('.', '')} ({overlap_ratio:.2f})")
                    else:
                        overlaps.append({'small': det2, 'large': det1, 'overlap_ratio': overlap_ratio})
                        print(f"      📦 Overlap detected: {det2.label.replace('.', '')} inside {det1.label.replace('.', '')} ({overlap_ratio:.2f})")
        
        return overlaps

    def segment_food(self, image_path: str, detections: list, selected_food: str) -> dict:
        """Enhanced overlap-aware segmentation using SAM 2"""
        print(f"🎯 Enhanced overlap-aware segmentation with SAM 2 for: '{selected_food}'...")
        image = load_image(image_path)
        image_np = np.array(image)
        self.sam_predictor.set_image(image_np)

        overlaps = self.detect_overlaps(detections)
        
        print("🛡️ Building exclusion mask from overlapping items...")
        exclusion_mask = np.zeros(image_np.shape[:2], dtype=bool)
        
        small_items = [overlap['small'] for overlap in overlaps]
        processed_small_items = []
        
        for small_detection in small_items:
            small_food_type = small_detection.label.replace('.', '').strip()
            print(f"   🔧 Pre-processing overlapping item: {small_food_type}")
            
            try:
                bbox = small_detection.box
                points = self.extract_smart_points_for_food(bbox, small_food_type, image_np.shape)
                
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=np.array(points),
                    point_labels=np.array([1] * len(points)),
                    box=np.array([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]),
                    multimask_output=True
                )
                
                best_mask = masks[np.argmax(scores)]

                best_mask = self._clean_mask(best_mask)
                
                exclusion_mask = np.logical_or(exclusion_mask, best_mask)
                processed_small_items.append({
                    'detection': small_detection,
                    'mask': best_mask,
                    'food_type': small_food_type
                })
                print(f"   🛡️ Added {small_food_type} to exclusion mask")
                
            except Exception as e:
                print(f"   ⚠️ Failed to process overlapping item {small_food_type}: {e}")
        
        selected_detections = []
        for d in detections:
            detection_label = d.label.strip(".").strip().lower()
            selected_food_normalized = selected_food.strip().lower()
            
            if detection_label == selected_food_normalized:
                selected_detections.append(d)
                print(f"   ✅ Found matching detection: '{d.label}' (confidence: {d.score:.3f})")
        
        if not selected_detections:
            print(f"   ⚠️ No exact matches found for '{selected_food}', trying partial matches...")
            for d in detections:
                detection_label = d.label.strip(".").strip().lower()
                if selected_food.lower() in detection_label or detection_label in selected_food.lower():
                    selected_detections.append(d)
                    print(f"   ✅ Found partial match: '{d.label}' (confidence: {d.score:.3f})")
        
        if not selected_detections:
            print(f"   ❌ No detections found for '{selected_food}'. Available labels:")
            for d in detections:
                print(f"      - '{d.label.strip('.')}'")
            raise ValueError(f"No detections found for the selected food type: '{selected_food}'")

        detection = max(selected_detections, key=lambda x: x.score)
        bbox = detection.box
        print(f"   🎯 Using detection: '{detection.label}' (confidence: {detection.score:.3f})")

        for processed_item in processed_small_items:
            if processed_item['detection'] == detection:
                print(f"   🎯 Using pre-processed overlapping item: {selected_food}")
                ys, xs = np.where(processed_item['mask'])
                centroid = (int(np.mean(xs)), int(np.mean(ys)))
                
                selected_piece = {
                    "mask": processed_item['mask'],
                    "centroid": centroid,
                    "bbox": [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax],
                    "food_type": selected_food,
                    "score": detection.score
                }
                
                self._visualize_final_piece(image, selected_piece, "overlap_item")
                return selected_piece

        print(f"🔍 Segmenting main food item with SAM 2: {selected_food}")
        points = self.extract_smart_points_for_food(bbox, selected_food, image_np.shape)
        
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array([1] * len(points)),
            box=np.array([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]),
            multimask_output=True
        )
        
        best_mask = masks[np.argmax(scores)]
        
        # AJOUT: Clean the mask before applying exclusions
        print("🧹 Cleaning mask to remove noise...")
        best_mask = self._clean_mask(best_mask)
        
        is_large_item = any(overlap['large'] == detection for overlap in overlaps)
        if is_large_item and np.sum(exclusion_mask) > 0:
            print(f"   🔧 Removing overlapping items from {selected_food}")
            original_area = np.sum(best_mask)
            best_mask = np.logical_and(best_mask, ~exclusion_mask)
            final_area = np.sum(best_mask)
            print(f"   📊 Area reduced from {original_area} to {final_area} pixels ({final_area/original_area:.2%} remaining)")
        
        ys, xs = np.where(best_mask)
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError(f"No valid pixels found for {selected_food} after overlap removal")
            
        centroid = (int(np.mean(xs)), int(np.mean(ys)))
        
        selected_piece = {
            "mask": best_mask,
            "centroid": centroid,
            "bbox": [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax],
            "food_type": selected_food,
            "score": scores[np.argmax(scores)],
            "is_large_item": is_large_item,
            "exclusion_applied": is_large_item and np.sum(exclusion_mask) > 0
        }
        
        print(f"   ✅ Selected food piece with centroid: {centroid}")
        
        overlap_type = "large_item" if is_large_item else "standalone_item"
        self._visualize_final_piece(image, selected_piece, overlap_type)
        
        return selected_piece
    
    def _visualize_final_piece(self, image, selected_piece, item_type):
        """Helper method to visualize the final selected piece"""
        print("🎨 Visualizing the final selected food piece...")
        final_image = np.array(image.copy())

        if item_type == "overlap_item":
            color = [255, 255, 0]  # Yellow
        elif item_type == "large_item":
            color = [0, 255, 255]  # Cyan
        else:
            color = [255, 0, 0]    # Red

        # Convert mask to boolean type if needed
        mask = selected_piece["mask"]
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        final_image[mask] = color

        cx, cy = selected_piece["centroid"]
        cv2.circle(final_image, (cx, cy), radius=5, color=(0, 255, 0), thickness=-1)

        food_type = selected_piece["food_type"]
        cv2.putText(final_image, f"{food_type} ({item_type})", (cx + 10, cy - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save to output directory
        final_visualization_path = self._get_output_path("final_selected_food_piece.png")
        cv2.imwrite(final_visualization_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
        print(f"✅ Final selected food piece visualization saved to {final_visualization_path}")

    def extract_smart_points_for_food(self, bbox, food_type: str, image_shape: tuple) -> list:
        """Extract strategic points from bounding box based on food type characteristics"""
        height, width = image_shape[:2]
        cx = int((bbox.xmin + bbox.xmax) / 2)
        cy = int((bbox.ymin + bbox.ymax) / 2)
        w = bbox.xmax - bbox.xmin
        h = bbox.ymax - bbox.ymin
        
        points = []
        food_lower = food_type.lower()
        
        points.append([cx, cy])
        
        if any(food_word in food_lower for food_word in ['pizza', 'bread', 'slice', 'pancake', 'waffle']):
            offset_x, offset_y = w//6, h//6
            points.extend([
                [cx - offset_x, cy - offset_y],
                [cx + offset_x, cy + offset_y], 
                [cx - offset_x, cy + offset_y],
                [cx + offset_x, cy - offset_y]
            ])
        elif any(food_word in food_lower for food_word in ['round', 'ball', 'cherry', 'grape', 'olive', 'meatball']):
            offset = min(w, h) // 8
            points.extend([
                [cx - offset, cy],
                [cx + offset, cy],
                [cx, cy - offset],
                [cx, cy + offset]
            ])
        else:
            offset_x, offset_y = w//4, h//4
            points.extend([
                [cx - offset_x, cy],
                [cx + offset_x, cy],
                [cx, cy - offset_y],
                [cx, cy + offset_y]
            ])
        
        valid_points = []
        for x, y in points:
            x = max(0, min(width-1, x))
            y = max(0, min(height-1, y))
            
            margin = 5
            if (bbox.xmin + margin <= x <= bbox.xmax - margin and 
                bbox.ymin + margin <= y <= bbox.ymax - margin):
                valid_points.append([x, y])
        
        if not valid_points:
            valid_points = [[cx, cy]]
            
        return valid_points

    def _clean_mask(self, mask):
        """Clean up mask with morphological operations to remove noise"""
        # Convert to uint8 for OpenCV
        mask_uint8 = mask.astype(np.uint8)
        
        # Remove small noise (opening = erosion + dilation)
        kernel_small = np.ones((3, 3), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Fill small holes (closing = dilation + erosion)
        kernel_medium = np.ones((5, 5), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        
        # Optional: Fill larger holes using flood fill
        # This is useful if there are holes inside the food
        h, w = mask_uint8.shape
        mask_floodfill = mask_uint8.copy()
        cv2.floodFill(mask_floodfill, np.zeros((h+2, w+2), np.uint8), (0,0), 255)
        mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)
        mask_uint8 = mask_uint8 | mask_floodfill_inv
        
        # Convert back to boolean
        return mask_uint8.astype(bool)
    
    def compute_centroid(self, mask: np.ndarray) -> tuple:
        """Step 7: Compute the centroid of the selected mask"""
        print("📍 Computing centroid of the selected food piece...")
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("Mask is empty, cannot compute centroid.")
        centroid = (int(np.mean(xs)), int(np.mean(ys)))
        print(f"   ✅ Centroid: {centroid}")
        return centroid

    def visualize_result(self, image_path: str, result: dict, save_path: str = "single_food_result.png"):
        """Visualize the selected food piece on the original image"""
        print("🎨 Visualizing the selected food piece...")
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Convert mask to boolean type if needed
        mask = result["mask"]
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        overlay = np.zeros_like(image_np, dtype=np.uint8)
        overlay[mask] = [255, 0, 0]
        alpha = 0.5
        image_np = cv2.addWeighted(overlay, alpha, image_np, 1 - alpha, 0)

        cx, cy = result["centroid"]
        cv2.circle(image_np, (cx, cy), radius=5, color=(0, 255, 0), thickness=-1)

        food_type = result.get("food_type", "unknown")
        cv2.putText(image_np, f"{food_type}", (cx + 10, cy - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save to output directory
        full_save_path = self._get_output_path(save_path)
        cv2.imwrite(full_save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        print(f"✅ Visualization saved to {full_save_path}")

    def detect_single_food_piece(self, image_path: str):
        """Run the full pipeline to detect a single food piece"""
        start_time = time.time()
        
        food_items = self.describe_plate(image_path)
        detections = self.localize_food(image_path, food_items)
        
        self.visualize_grounding_dino_results(image_path, detections, "grounding_dino_single_detection.png")
        
        if not detections:
            print("❌ No detections found by GroundingDINO. Cannot proceed with segmentation.")
            return None

        selected_food = self.select_random_food_from_detections(detections)
        segmentation_result = self.segment_food(image_path, detections, selected_food)
        centroid = self.compute_centroid(segmentation_result["mask"])
        
        computation_time = time.time() - start_time
        print(f"⏱️ Computation time (excluding model loading and visualization): {computation_time:.2f} seconds")
        
        return {
            "selected_food": selected_food,
            "centroid": centroid,
            "mask": segmentation_result["mask"],
            "bbox": segmentation_result["bbox"],
            "food_type": segmentation_result.get("food_type", selected_food),
            "computation_time": computation_time
        }

    def detect_individual_food_pieces(self, image_path: str):
        """
        Automatic pipeline that adapts based on food type:
        - For grains (rice, quinoa, etc.): returns the entire portion
        - For discrete items (carrots, meat): segments individual pieces
        """
        print("🔍 Starting adaptive food detection pipeline with SAM 2...")
        start_time = time.time()

        step1_start = time.time()
        print("🔍 Step 1: Getting mask from single food detector...")
        food_items = self.describe_plate(image_path)
        detections = self.localize_food(image_path, food_items)
        print(f"   🕒 Step 1 completed in {time.time() - step1_start:.2f} seconds")

        if not detections:
            print("❌ No detections found by GroundingDINO. Cannot proceed with segmentation.")
            return None

        selected_food = self.select_random_food_from_detections(detections)
        segmentation_result = self.segment_food(image_path, detections, selected_food)
        food_type_mask = segmentation_result["mask"]

        selected_detection = None
        for d in detections:
            detection_label = d.label.strip(".").strip().lower()
            selected_food_normalized = selected_food.strip().lower()
            if detection_label == selected_food_normalized:
                selected_detection = d
                break

        if not selected_detection:
            for d in detections:
                detection_label = d.label.strip(".").strip().lower()
                if selected_food.lower() in detection_label or detection_label in selected_food.lower():
                    selected_detection = d
                    break

        if not selected_detection:
            print(f"❌ Could not find bounding box for selected food: {selected_food}")
            return None

        bbox = selected_detection.box
        print(f"🎯 Selected food bounding box: ({bbox.xmin:.0f}, {bbox.ymin:.0f}, {bbox.xmax:.0f}, {bbox.ymax:.0f})")

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        step1_vis = image_np.copy()
        step1_vis[food_type_mask] = [0, 255, 0]
        
        step1_path = self._get_output_path("step1_food_type_mask.png")
        cv2.imwrite(step1_path, cv2.cvtColor(step1_vis, cv2.COLOR_RGB2BGR))

        # DÉTECTION AUTOMATIQUE: Est-ce un aliment "en vrac" ou des pièces individuelles?
        selected_food_lower = selected_food.lower()
        
        # Liste des aliments "en vrac" qui ne doivent PAS être segmentés en pièces individuelles
        bulk_foods = ['rice', 'grain', 'grains', 'quinoa', 'couscous', 'lentil', 'lentils', 
                    'bean', 'beans', 'pea', 'peas', 'corn', 'seeds', 'pasta', 'noodles',
                    'salad', 'lettuce', 'spinach', 'mash', 'mashed', 'mashed potatoes', 'mesh', 'puree', 'sauce',
                    'soup', 'stew', 'porridge', 'oatmeal']
        
        is_bulk_food = any(bulk in selected_food_lower for bulk in bulk_foods)
        
        if is_bulk_food:
            # Pour les aliments en vrac: retourner directement la portion complète de l'étape 1
            print(f"🌾 Detected bulk/grain food: '{selected_food}' - returning entire portion")
            
            ys, xs = np.where(food_type_mask)
            if len(xs) == 0:
                print("❌ Empty mask")
                return None
                
            centroid = (int(np.mean(xs)), int(np.mean(ys)))
            
            computation_time = time.time() - start_time
            print(f"⏱️ Computation time: {computation_time:.2f} seconds")

            # Visualize
            visualization = image_np.copy()
            visualization[food_type_mask] = [255, 0, 0]
            cv2.circle(visualization, centroid, radius=8, color=(0, 255, 0), thickness=-1)
            cv2.putText(visualization, f"{selected_food} (entire portion)", 
                    (centroid[0] + 10, centroid[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            output_path = self._get_output_path("selected_food_piece_from_bbox_intersection.png")
            cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

            print(f"✅ Entire food portion segmented with centroid: {centroid}")
            print(f"   📊 Total area: {np.sum(food_type_mask)} pixels")

            return {
                "selected_food": selected_food,
                "total_pieces": 1,  # Une seule portion
                "selected_piece": {
                    'mask': food_type_mask,
                    'area': np.sum(food_type_mask)
                },
                "centroid": centroid,
                "bbox_region": (int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)),
                "area_reduction": 0,
                "visualization_path": output_path,
                "step1_visualization": step1_path,
                "is_bulk_food": True,
                "computation_time": computation_time
            }
        
        # Pour les autres aliments: continuer avec la segmentation des pièces individuelles
        print(f"🥕 Detected discrete food items: '{selected_food}' - segmenting individual pieces")
        
        step2_start = time.time()
        print("🔍 Step 2: Running SAM 2 automatic mask generation within bounding box...")

        padding = 20
        x1 = max(0, int(bbox.xmin - padding))
        y1 = max(0, int(bbox.ymin - padding))
        x2 = min(image_np.shape[1], int(bbox.xmax + padding))
        y2 = min(image_np.shape[0], int(bbox.ymax + padding))

        cropped_image = image_np[y1:y2, x1:x2]
        print(f"🔧 Cropped image size: {cropped_image.shape[1]}x{cropped_image.shape[0]} pixels")

        # Paramètres pour pièces individuelles
        sam_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2,
            points_per_side=16,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=200
        )

        complete_masks = sam_generator.generate(cropped_image)
        print(f"✅ Generated {len(complete_masks)} masks within bounding box region")
        print(f"   🕒 Step 2 completed in {time.time() - step2_start:.2f} seconds")

        # Adjust masks back to full image coordinates
        adjusted_masks = []
        for mask in complete_masks:
            full_mask = np.zeros(image_np.shape[:2], dtype=bool)
            full_mask[y1:y2, x1:x2] = mask['segmentation']
            adjusted_mask = mask.copy()
            adjusted_mask['segmentation'] = full_mask
            adjusted_masks.append(adjusted_mask)

        complete_masks = adjusted_masks

        step2_vis = image_np.copy()
        cv2.rectangle(step2_vis, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.putText(step2_vis, f"Segmentation Region", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        colors = np.random.randint(0, 255, size=(len(complete_masks), 3))
        for i, mask in enumerate(complete_masks):
            color = colors[i]
            step2_vis[mask['segmentation']] = color
        
        step2_path = self._get_output_path("step2_bbox_segmentation.png")
        cv2.imwrite(step2_path, cv2.cvtColor(step2_vis, cv2.COLOR_RGB2BGR))

        step3_start = time.time()
        print("🔍 Step 3: Finding intersections and counting food pieces...")
        intersecting_masks = []

        for mask in complete_masks:
            segmentation = mask['segmentation']
            intersection = np.logical_and(segmentation, food_type_mask)

            if np.sum(intersection) > 100:
                intersecting_masks.append({
                    'mask': intersection,
                    'area': np.sum(intersection)
                })

        print(f"✅ Found {len(intersecting_masks)} food pieces in the selected food type region")
        print(f"   🕒 Step 3 completed in {time.time() - step3_start:.2f} seconds")

        step3_vis = image_np.copy()
        intersection_colors = np.random.randint(100, 255, size=(len(intersecting_masks), 3))
        for i, mask_data in enumerate(intersecting_masks):
            color = intersection_colors[i]
            step3_vis[mask_data['mask']] = color
        
        step3_path = self._get_output_path("step3_intersection_masks.png")
        cv2.imwrite(step3_path, cv2.cvtColor(step3_vis, cv2.COLOR_RGB2BGR))

        if intersecting_masks:
            selected_piece = max(intersecting_masks, key=lambda x: x['area'])
            
            print(f"🎯 Selected largest food piece with area: {selected_piece['area']} pixels")

            ys, xs = np.where(selected_piece['mask'])
            if len(xs) > 0:
                centroid = (int(np.mean(xs)), int(np.mean(ys)))

                computation_time = time.time() - start_time
                print(f"⏱️ Computation time (excluding model loading and visualization): {computation_time:.2f} seconds")

                visualization = image_np.copy()
                visualization[selected_piece['mask']] = [255, 0, 0]
                cv2.circle(visualization, centroid, radius=5, color=(0, 255, 0), thickness=-1)
                cv2.putText(visualization, f"{selected_food}", (centroid[0] + 10, centroid[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                output_path = self._get_output_path("selected_food_piece_from_bbox_intersection.png")
                cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

                print(f"✅ Selected largest food piece with centroid: {centroid}")
                print(f"🎯 Optimization: Processed {(cropped_image.size / image_np.size) * 100:.1f}% of original image area")

                return {
                    "selected_food": selected_food,
                    "total_pieces": len(intersecting_masks),
                    "selected_piece": selected_piece,
                    "centroid": centroid,
                    "bbox_region": (x1, y1, x2, y2),
                    "area_reduction": 1 - (cropped_image.size / image_np.size),
                    "visualization_path": output_path,
                    "step1_visualization": step1_path,
                    "step2_visualization": step2_path, 
                    "step3_visualization": step3_path,
                    "is_bulk_food": False,
                    "computation_time": computation_time
                }
        else:
            computation_time = time.time() - start_time
            print(f"⏱️ Computation time (excluding model loading): {computation_time:.2f} seconds")
            print("❌ No food pieces found in intersection")
            return None

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Single Food Detector with SAM 2")
    parser.add_argument("--image", type=str, default="images/snack-plate-meal-1_orig.jpg", help="Path to the input image")
    parser.add_argument("--model", type=str, default="large", choices=["tiny", "small", "base_plus", "large"], help="SAM 2 model size")
    parser.add_argument("--output-dir", type=str, default="~/Documents/src/vision_processing/scripts", help="Output directory for results")

    args = parser.parse_args()
    image_path = args.image
    
    # SAM 2 model paths
    sam2_checkpoint = os.path.expanduser(f"~/segment-anything-2/checkpoints/sam2.1_hiera_{args.model}.pt")
    
    # Config mapping
    config_map = {
        "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "large": "configs/sam2.1/sam2.1_hiera_l.yaml"
    }
    sam2_config = config_map[args.model]

    print("🌟 SINGLE FOOD DETECTOR WITH SAM 2")
    print("Features:")
    print("• ✅ SAM 2 for faster and better segmentation")
    print("• ✅ Automatic detection: bulk foods vs individual pieces")
    print("• ✅ Results saved to specified output directory")
    print("• ✅ Fixed LLaVA food parsing")
    print("• ✅ Smart duplicate removal with IoU filtering")  
    print("• ✅ Confidence-based filtering")
    print("• ✅ GroundingDINO visualization")
    print("• ✅ GPU memory management")
    print("• ✅ Bounding box segmentation optimization")
    print("="*80)

    detector = SingleFoodDetector(sam2_checkpoint, sam2_config, output_dir=args.output_dir)

    result = detector.detect_individual_food_pieces(image_path)
    if result:
        print("\n🎉 FOOD DETECTION RESULT (SAM 2):")
        print(f"   🍴 Selected Food Type: {result['selected_food']}")
        
        # Check if it's a bulk food or discrete items
        if result.get('is_bulk_food', False):
            print(f"   🌾 Food Type: Bulk/Grain (entire portion)")
        else:
            print(f"   🥕 Food Type: Discrete items")
            
        print(f"   📊 Total Pieces Found: {result['total_pieces']}")
        print(f"   📍 Selected Piece Centroid: {result['centroid']}")
        
        if 'area_reduction' in result and result['area_reduction'] > 0:
            print(f"   🎯 Area Reduction: {result['area_reduction']*100:.1f}% less processing")
            
        if 'bbox_region' in result:
            print(f"   📦 Bounding Box Region: {result['bbox_region']}")
            
        print(f"   ⏱️ Computation Time: {result['computation_time']:.2f} seconds")
        print(f"   📁 All results saved to: {detector.output_dir}")
        
        # Only show step visualizations if they exist
        if 'step1_visualization' in result:
            print(f"   🎨 Step 1 - Food Type Mask: {result['step1_visualization']}")
        if 'step2_visualization' in result:
            print(f"   🎨 Step 2 - BBox Segmentation: {result['step2_visualization']}")
        if 'step3_visualization' in result:
            print(f"   🎨 Step 3 - Intersection Masks: {result['step3_visualization']}")
            
        print(f"   🎨 Final Result Visualization: {result['visualization_path']}")
    else:
        print("\n❌ FOOD DETECTION FAILED:")
        print("   No detections found or no food pieces could be segmented.")