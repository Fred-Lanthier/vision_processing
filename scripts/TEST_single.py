#!/usr/bin/env python3
"""
Fixed Single Food Piece Detector
MODIFIED: Segments only within the wanted bounding box instead of entire picture
"""

import time
import random
import numpy as np
import re
import gc
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from GroundingDINO_with_Segment_Anything_Utils import *
from segment_anything import sam_model_registry, SamPredictor
import cv2

class SingleFoodDetector:
    def __init__(self, model_type, model_name):
        """Initialize the single food detector"""
        print("🚀 Initializing Single Food Detector...")

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

        # Load SAM model
        print("🎯 Loading SAM model...")
        self.sam = sam_model_registry[model_type](checkpoint=model_name)
        self.sam.to(self.device)
        self.sam_predictor = SamPredictor(self.sam)
        print("✅ SAM ready for segmentation!")

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
        
        try:
            image = load_image(image_path)
            image_array = np.array(image)
            
            # Use the existing plot_detections utility function if available
            try:
                plot_detections(image_array, detections, save_path)
                print(f"   ✅ GroundingDINO visualization saved: {save_path}")
            except:
                # Fallback: Create our own visualization
                self._create_custom_detection_visualization(image_array, detections, save_path)
                print(f"   ✅ GroundingDINO visualization saved: {save_path}")
            
            # Print summary of detected foods
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
        
        # Color map for different food types
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta']
        food_colors = {}
        
        for i, detection in enumerate(detections):
            bbox = detection.box
            food_type = detection.label.replace('.', '').strip()
            confidence = detection.score
            
            # Assign color
            if food_type not in food_colors:
                food_colors[food_type] = colors[len(food_colors) % len(colors)]
            
            color = food_colors[food_type]
            
            # Draw bounding box
            rect = patches.Rectangle(
                (bbox.xmin, bbox.ymin), 
                bbox.xmax - bbox.xmin, 
                bbox.ymax - bbox.ymin,
                linewidth=2, 
                edgecolor=color, 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
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
        # selected_food = "carrot slices"
        print(f"🎲 Selected food type: '{selected_food}'")
        return selected_food
    
    def select_random_food_from_detections(self, detections: list) -> str:
        """Select one random food type from actual GroundingDINO detections"""
        if not detections:
            raise ValueError("No detections available to select from")
        
        # Extract unique food types from detections
        detected_food_types = []
        for detection in detections:
            food_type = detection.label.strip(".").strip()
            if food_type not in detected_food_types:
                detected_food_types.append(food_type)
        
        selected_food = random.choice(detected_food_types)
        # selected_food = "carrot slices"
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
                    
                # Calculate areas
                area1 = (det1.box.xmax - det1.box.xmin) * (det1.box.ymax - det1.box.ymin)
                area2 = (det2.box.xmax - det2.box.xmin) * (det2.box.ymax - det2.box.ymin)
                
                # Calculate intersection
                x_overlap = max(0, min(det1.box.xmax, det2.box.xmax) - max(det1.box.xmin, det2.box.xmin))
                y_overlap = max(0, min(det1.box.ymax, det2.box.ymax) - max(det1.box.ymin, det2.box.ymin))
                intersection = x_overlap * y_overlap
                
                if intersection == 0:
                    continue
                    
                # Check if smaller item is mostly inside larger item
                smaller_area = min(area1, area2)
                overlap_ratio = intersection / smaller_area
                
                if overlap_ratio > 0.6:  # 60% overlap threshold
                    if area1 < area2:
                        overlaps.append({'small': det1, 'large': det2, 'overlap_ratio': overlap_ratio})
                        print(f"      📦 Overlap detected: {det1.label.replace('.', '')} inside {det2.label.replace('.', '')} ({overlap_ratio:.2f})")
                    else:
                        overlaps.append({'small': det2, 'large': det1, 'overlap_ratio': overlap_ratio})
                        print(f"      📦 Overlap detected: {det2.label.replace('.', '')} inside {det1.label.replace('.', '')} ({overlap_ratio:.2f})")
        
        return overlaps

    def segment_food(self, image_path: str, detections: list, selected_food: str) -> dict:
        """Enhanced overlap-aware segmentation that properly handles all overlaps"""
        print(f"🎯 Enhanced overlap-aware segmentation for: '{selected_food}'...")
        image = load_image(image_path)
        image_np = np.array(image)
        self.sam_predictor.set_image(image_np)

        # Step 1: Detect ALL overlaps in the scene (regardless of selected food)
        overlaps = self.detect_overlaps(detections)
        
        # Step 2: Build complete exclusion mask from ALL small overlapping items
        print("🛡️ Building exclusion mask from overlapping items...")
        exclusion_mask = np.zeros(image_np.shape[:2], dtype=bool)
        
        # Process ALL small overlapping items first (regardless of selection)
        small_items = [overlap['small'] for overlap in overlaps]
        processed_small_items = []
        
        for small_detection in small_items:
            small_food_type = small_detection.label.replace('.', '').strip()
            print(f"   🔧 Pre-processing overlapping item: {small_food_type}")
            
            try:
                # Segment the small item
                bbox = small_detection.box
                points = self.extract_smart_points_for_food(bbox, small_food_type, image_np.shape)
                
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=np.array(points),
                    point_labels=np.array([1] * len(points)),
                    box=np.array([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]),
                    multimask_output=True
                )
                
                best_mask = masks[np.argmax(scores)]
                
                # Add to exclusion mask
                exclusion_mask = np.logical_or(exclusion_mask, best_mask)
                processed_small_items.append({
                    'detection': small_detection,
                    'mask': best_mask,
                    'food_type': small_food_type
                })
                print(f"   🛡️ Added {small_food_type} to exclusion mask")
                
            except Exception as e:
                print(f"   ⚠️ Failed to process overlapping item {small_food_type}: {e}")
        
        # Step 3: Find the selected food detection
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

        # Use the highest confidence detection
        detection = max(selected_detections, key=lambda x: x.score)
        bbox = detection.box
        print(f"   🎯 Using detection: '{detection.label}' (confidence: {detection.score:.3f})")

        # Step 4: Check if this is a small overlapping item that was already processed
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

        # Step 5: Segment the selected food (if not already processed as overlapping item)
        print(f"🔍 Segmenting main food item: {selected_food}")
        points = self.extract_smart_points_for_food(bbox, selected_food, image_np.shape)
        
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array([1] * len(points)),
            box=np.array([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]),
            multimask_output=True
        )
        
        best_mask = masks[np.argmax(scores)]
        
        # Step 6: KEY FIX - Check if this is a large item and apply exclusion mask
        is_large_item = any(overlap['large'] == detection for overlap in overlaps)
        if is_large_item and np.sum(exclusion_mask) > 0:
            print(f"   🔧 Removing overlapping items from {selected_food}")
            original_area = np.sum(best_mask)
            best_mask = np.logical_and(best_mask, ~exclusion_mask)
            final_area = np.sum(best_mask)
            print(f"   📊 Area reduced from {original_area} to {final_area} pixels ({final_area/original_area:.2%} remaining)")
        
        # Step 7: Create final result
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
        
        # Visualize result
        overlap_type = "large_item" if is_large_item else "standalone_item"
        self._visualize_final_piece(image, selected_piece, overlap_type)
        
        return selected_piece
    
    def _visualize_final_piece(self, image, selected_piece, item_type):
        """Helper method to visualize the final selected piece"""
        print("🎨 Visualizing the final selected food piece...")
        final_image = np.array(image.copy())

        # Color coding based on item type
        if item_type == "overlap_item":
            color = [255, 255, 0]  # Yellow for overlapping items
        elif item_type == "large_item":
            color = [0, 255, 255]  # Cyan for large items with exclusion applied
        else:
            color = [255, 0, 0]    # Red for standalone items

        # Highlight the selected food piece mask
        final_image[selected_piece["mask"]] = color

        # Draw the centroid
        cx, cy = selected_piece["centroid"]
        cv2.circle(final_image, (cx, cy), radius=5, color=(0, 255, 0), thickness=-1)

        # Add text label with type info
        food_type = selected_piece["food_type"]
        cv2.putText(final_image, f"{food_type} ({item_type})", (cx + 10, cy - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save the visualization
        final_visualization_path = "final_selected_food_piece.png"
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
        
        # Always include center point
        points.append([cx, cy])
        
        # Food-specific point strategies
        if any(food_word in food_lower for food_word in ['pizza', 'bread', 'slice', 'pancake', 'waffle']):
            # For flat foods: focus on interior
            offset_x, offset_y = w//6, h//6
            points.extend([
                [cx - offset_x, cy - offset_y],
                [cx + offset_x, cy + offset_y], 
                [cx - offset_x, cy + offset_y],
                [cx + offset_x, cy - offset_y]
            ])
        elif any(food_word in food_lower for food_word in ['round', 'ball', 'cherry', 'grape', 'olive', 'meatball']):
            # For round/small foods
            offset = min(w, h) // 8
            points.extend([
                [cx - offset, cy],
                [cx + offset, cy],
                [cx, cy - offset],
                [cx, cy + offset]
            ])
        else:
            # General strategy
            offset_x, offset_y = w//4, h//4
            points.extend([
                [cx - offset_x, cy],
                [cx + offset_x, cy],
                [cx, cy - offset_y],
                [cx, cy + offset_y]
            ])
        
        # Ensure all points are within bounds
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

        # Overlay the mask
        mask = result["mask"]
        overlay = np.zeros_like(image_np, dtype=np.uint8)
        overlay[mask > 0] = [255, 0, 0]  # Red color for the mask
        alpha = 0.5  # Transparency level
        image_np = cv2.addWeighted(overlay, alpha, image_np, 1 - alpha, 0)

        # Draw the centroid
        cx, cy = result["centroid"]
        cv2.circle(image_np, (cx, cy), radius=5, color=(0, 255, 0), thickness=-1)  # Green dot for the centroid

        # Add text label
        food_type = result.get("food_type", "unknown")
        cv2.putText(image_np, f"{food_type}", (cx + 10, cy - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save the visualization
        cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        print(f"✅ Visualization saved to {save_path}")

    def detect_single_food_piece(self, image_path: str):
        """Run the full pipeline to detect a single food piece"""
        start_time = time.time()
        
        food_items = self.describe_plate(image_path)
        detections = self.localize_food(image_path, food_items)
        
        # Visualize GroundingDINO results
        self.visualize_grounding_dino_results(image_path, detections, "grounding_dino_single_detection.png")
        
        # Check if we have any detections
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
        MODIFIED: Complete pipeline to detect individual food pieces using BOUNDING BOX segmentation
        Instead of segmenting the entire picture, we now segment only the wanted bounding box
        """
        print("🔍 Starting individual food pieces detection pipeline...")
        start_time = time.time()

        # Step 1: Get the mask from single_food_detector pipeline
        step1_start = time.time()
        print("🔍 Step 1: Getting mask from single food detector...")
        food_items = self.describe_plate(image_path)
        detections = self.localize_food(image_path, food_items)
        print(f"   🕒 Step 1 completed in {time.time() - step1_start:.2f} seconds")

        # Save GroundingDINO results without printing
        # self.visualize_grounding_dino_results(image_path, detections, "grounding_dino_individual_detection.png")

        # Check if we have any detections
        if not detections:
            print("❌ No detections found by GroundingDINO. Cannot proceed with segmentation.")
            return None

        selected_food = self.select_random_food_from_detections(detections)
        segmentation_result = self.segment_food(image_path, detections, selected_food)
        food_type_mask = segmentation_result["mask"]

        # Get the bounding box for the selected food
        selected_detection = None
        for d in detections:
            detection_label = d.label.strip(".").strip().lower()
            selected_food_normalized = selected_food.strip().lower()
            if detection_label == selected_food_normalized:
                selected_detection = d
                break

        if not selected_detection:
            # Try partial match
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

        # Visualize Step 1: Food type mask
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)  # Ensure image_np is defined
        step1_vis = image_np.copy()
        step1_vis[food_type_mask] = [0, 255, 0]  # Green color for food type mask
        cv2.imwrite("step1_food_type_mask.png", cv2.cvtColor(step1_vis, cv2.COLOR_RGB2BGR))

        # MODIFIED Step 2: Get segmentation ONLY within the bounding box instead of entire image
        step2_start = time.time()
        print("🔍 Step 2: Running segmentation within bounding box only...")
        from segment_anything import SamAutomaticMaskGenerator

        # Load the full image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # MODIFICATION: Crop image to bounding box region with some padding
        padding = 20  # Add some padding around the bounding box
        x1 = max(0, int(bbox.xmin - padding))
        y1 = max(0, int(bbox.ymin - padding))
        x2 = min(image_np.shape[1], int(bbox.xmax + padding))
        y2 = min(image_np.shape[0], int(bbox.ymax + padding))

        # Crop the image to the bounding box region
        cropped_image = image_np[y1:y2, x1:x2]
        print(f"🔧 Cropped image size: {cropped_image.shape[1]}x{cropped_image.shape[0]} pixels")

        # Run SAM generator on the CROPPED image instead of full image
        sam_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=200
        )

        complete_masks = sam_generator.generate(cropped_image)
        print(f"✅ Generated {len(complete_masks)} masks within bounding box region")
        print(f"   🕒 Step 2 completed in {time.time() - step2_start:.2f} seconds")

        # MODIFICATION: Adjust mask coordinates back to full image space
        adjusted_masks = []
        for mask in complete_masks:
            # Create a full-size mask
            full_mask = np.zeros(image_np.shape[:2], dtype=bool)

            # Place the cropped mask in the correct position within the full image
            full_mask[y1:y2, x1:x2] = mask['segmentation']

            # Update the mask data
            adjusted_mask = mask.copy()
            adjusted_mask['segmentation'] = full_mask
            adjusted_masks.append(adjusted_mask)

        complete_masks = adjusted_masks

        # Visualize Step 2: Bounding box segmentation (instead of complete segmentation)
        step2_vis = image_np.copy()

        # Draw bounding box outline
        cv2.rectangle(step2_vis, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.putText(step2_vis, f"Segmentation Region", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Color the masks within the bounding box
        colors = np.random.randint(0, 255, size=(len(complete_masks), 3))
        for i, mask in enumerate(complete_masks):
            color = colors[i]
            step2_vis[mask['segmentation']] = color

        cv2.imwrite("step2_bbox_segmentation.png", cv2.cvtColor(step2_vis, cv2.COLOR_RGB2BGR))

        # Step 3: Find intersections and count food pieces (same as before)
        step3_start = time.time()
        print("🔍 Step 3: Finding intersections and counting food pieces...")
        intersecting_masks = []

        for mask in complete_masks:
            segmentation = mask['segmentation']
            intersection = np.logical_and(segmentation, food_type_mask)

            # Check if there's significant intersection
            if np.sum(intersection) > 100:  # Minimum threshold for valid intersection
                intersecting_masks.append({
                    'mask': intersection,
                    'area': np.sum(intersection)
                })

        print(f"✅ Found {len(intersecting_masks)} food pieces in the selected food type region")
        print(f"   🕒 Step 3 completed in {time.time() - step3_start:.2f} seconds")

        # Visualize Step 3: Intersection masks
        step3_vis = image_np.copy()
        intersection_colors = np.random.randint(100, 255, size=(len(intersecting_masks), 3))
        for i, mask_data in enumerate(intersecting_masks):
            color = intersection_colors[i]
            step3_vis[mask_data['mask']] = color
        cv2.imwrite("step3_intersection_masks.png", cv2.cvtColor(step3_vis, cv2.COLOR_RGB2BGR))

        # Step 4: Select the food piece with the largest mask area
        if intersecting_masks:
            # Sort by area and select the largest
            selected_piece = max(intersecting_masks, key=lambda x: x['area'])
            
            print(f"🎯 Selected largest food piece with area: {selected_piece['area']} pixels")

            # Compute centroid of selected piece
            ys, xs = np.where(selected_piece['mask'])
            if len(xs) > 0:
                centroid = (int(np.mean(xs)), int(np.mean(ys)))

                computation_time = time.time() - start_time
                print(f"⏱️ Computation time (excluding model loading and visualization): {computation_time:.2f} seconds")

                # Visualize the selected piece (not included in timing)
                visualization = image_np.copy()
                visualization[selected_piece['mask']] = [255, 0, 0]  # Red color
                cv2.circle(visualization, centroid, radius=5, color=(0, 255, 0), thickness=-1)

                # Add text label
                cv2.putText(visualization, f"{selected_food}", (centroid[0] + 10, centroid[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                output_path = "selected_food_piece_from_bbox_intersection.png"
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
                    "step1_visualization": "step1_food_type_mask.png",
                    "step2_visualization": "step2_bbox_segmentation.png", 
                    "step3_visualization": "step3_intersection_masks.png",
                    "computation_time": computation_time
                }
        else:
            computation_time = time.time() - start_time
            print(f"⏱️ Computation time (excluding model loading): {computation_time:.2f} seconds")
            print("❌ No food pieces found in intersection")
            return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MODIFIED Single Food Detector with Bounding Box Segmentation")
    parser.add_argument("--image", type=str, default="images/snack-plate-meal-1_orig.jpg", help="Path to the input image")
    parser.add_argument("--model", type=str, default="l", choices=["h", "l", "b"], help="SAM model to use (h=huge, l=large, b=base)")

    args = parser.parse_args()
    image_path = args.image
    model_type = "vit_" + args.model
    model_name = "sam_vit_" + args.model + ".pth"

    print("🌟 MODIFIED SINGLE FOOD DETECTOR WITH BOUNDING BOX SEGMENTATION")
    print("Features:")
    print("• ✅ Fixed LLaVA food parsing (no more 'food type 1')")
    print("• ✅ Smart duplicate removal with IoU filtering")  
    print("• ✅ Confidence-based filtering")
    print("• ✅ GroundingDINO visualization")
    print("• ✅ GPU memory management")
    print("• ✅ MODIFIED: Bounding box segmentation instead of full image")
    print("• ✅ Performance optimization with area reduction")
    print("="*80)

    detector = SingleFoodDetector(model_type, model_name)

    # Use the individual pieces detection method (MODIFIED with bbox segmentation)
    result = detector.detect_individual_food_pieces(image_path)
    if result:
        print("\n🎉 BOUNDING BOX INDIVIDUAL FOOD PIECES DETECTION RESULT:")
        print(f"   🍴 Selected Food Type: {result['selected_food']}")
        print(f"   📊 Total Pieces Found: {result['total_pieces']}")
        print(f"   📍 Selected Piece Centroid: {result['centroid']}")
        print(f"   🎯 Area Reduction: {result['area_reduction']*100:.1f}% less processing")
        print(f"   📦 Bounding Box Region: {result['bbox_region']}")
        print(f"   ⏱️ Computation Time: {result['computation_time']:.2f} seconds")
        print(f"   🎨 GroundingDINO Visualization: grounding_dino_individual_detection.png")
        if 'step1_visualization' in result:
            print(f"   🎨 Step 1 - Food Type Mask: {result['step1_visualization']}")
            print(f"   🎨 Step 2 - BBox Segmentation: {result['step2_visualization']}")
            print(f"   🎨 Step 3 - Intersection Masks: {result['step3_visualization']}")
        print(f"   🎨 Final Result Visualization: {result['visualization_path']}")
    else:
        print("\n❌ BOUNDING BOX INDIVIDUAL FOOD PIECES DETECTION FAILED:")
        print("   No detections found by GroundingDINO or no food pieces could be segmented.")