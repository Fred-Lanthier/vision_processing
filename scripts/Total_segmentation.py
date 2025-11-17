#!/usr/bin/env python3
"""
Complete Food Plate Semantic Segmentation with Adaptive Hierarchical Detection using SAM 2
"""

import time
import numpy as np
import re
import gc
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from GroundingDINO_with_Segment_Anything_Utils import *
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

class CompleteFoodDetector:
    def __init__(self, sam2_checkpoint, sam2_config, centroid_distance_threshold=8, output_dir="~/Documents/src/vision_processing/scripts"):
        print("🚀 Initializing Complete Food Plate Detector with SAM 2 and Adaptive Search...")
        self.centroid_distance_threshold = centroid_distance_threshold
        self.output_dir = os.path.expanduser(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define granular foods that should not be segmented into individual pieces
        self.granular_foods = [
            'rice', 'quinoa', 'couscous', 'grains', 'seeds', 
            'corn', 'peas', 'beans', 'lentils', 'chickpeas',
            'pasta', 'noodles', 'macaroni', 'spaghetti'
        ]

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

        # Load SAM 2 model
        print("🎯 Loading SAM 2 model...")
        self.sam2 = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2)
        
        self.sam2_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=500
        )
        print("✅ SAM 2 ready for segmentation!")

    def clear_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def describe_plate(self, image_path: str) -> list:
        print("🧠 Describing the plate with LLaVA...")
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
        
        simplified_items = self.simplify_food_names(cleaned_items)
        return simplified_items

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

    def generate_search_variants(self, food_item: str) -> list:
        words = food_item.lower().strip().split()
        filler_words = ['for', 'with', 'and', 'on', 'the', 'a', 'an']
        meaningful_words = [w for w in words if w not in filler_words]
        
        special_cases = {
            'chicken cubes': ['chicken cubes', 'chicken pieces', 'chicken', 'cubes'],
            'chicken pieces': ['chicken pieces', 'chicken cubes', 'chicken', 'pieces'],
            'beef cubes': ['beef cubes', 'beef pieces', 'beef', 'cubes'],
            'carrot slices': ['carrot slices', 'carrots', 'carrot', 'slices'],
            'potato wedges': ['potato wedges', 'potatoes', 'potato', 'wedges'],
            'broccoli florets': ['broccoli florets', 'broccoli', 'florets'],
        }
        
        food_key = food_item.lower().strip()
        if food_key in special_cases:
            return special_cases[food_key]
        
        variants = []
        if len(meaningful_words) >= 2:
            variants.append(' '.join(meaningful_words))
            variants.append(meaningful_words[0])
            if meaningful_words[-1] != meaningful_words[0]:
                variants.append(meaningful_words[-1])
            if len(meaningful_words) > 2:
                variants.append(' '.join(meaningful_words[:2]))
        elif len(meaningful_words) == 1:
            variants.append(meaningful_words[0])
        else:
            variants.append(food_item.lower().strip())
        
        unique_variants = []
        seen = set()
        for variant in variants:
            if variant not in seen:
                unique_variants.append(variant)
                seen.add(variant)
        
        return unique_variants

    def localize_all_food(self, image_path: str, food_items: list) -> list:
        print("📍 Adaptive food localization with hierarchical search...")
        image = load_image(image_path)
    
        # OPTIMIZATION 1: Batch all primary searches first (single call)
        primary_labels = [f"{food_item}." for food_item in food_items]
        
        try:
            # Single detection call with higher threshold for speed
            all_detections = detect(
                image=image,
                labels=primary_labels,
                threshold=0.35,  # INCREASED from 0.2 - major speedup!
                detector_id="IDEA-Research/grounding-dino-tiny"
            )
            
            if all_detections:
                # Clean once after getting results
                clean_detections = self.smart_duplicate_removal(all_detections)
                high_conf = [d for d in clean_detections if d.score >= 0.4]
                
                if high_conf:
                    return self.remove_global_duplicates(high_conf)
        except Exception as e:
            print(f"Batch detection failed: {e}")
        
        # OPTIMIZATION 2: Fallback with reduced variants (only if needed)
        print("   🔄 Fallback: trying simplified variants...")
        fallback_detections = []
        
        for food_item in food_items[:3]:  # Limit to first 3 food types max
            simplified_variants = [food_item.split()[0]]  # Just first word
            
            for variant in simplified_variants:
                try:
                    labels = [f"{variant}."]
                    detections = detect(
                        image=image,
                        labels=labels,
                        threshold=0.35,  # Still higher than original
                        detector_id="IDEA-Research/grounding-dino-tiny"
                    )
                    
                    if detections:
                        for d in detections:
                            d.original_food_item = food_item
                            d.successful_variant = variant
                        fallback_detections.extend(detections)
                        break  # Success - move to next food type
                        
                except Exception:
                    continue
        
        return self.remove_global_duplicates(fallback_detections) if fallback_detections else []

    def remove_global_duplicates(self, detections: list) -> list:
        if not detections:
            return []
        
        detection_info = []
        for i, detection in enumerate(detections):
            bbox = detection.box
            xmin, ymin, xmax, ymax = self.get_bbox_coordinates(bbox)
            
            centroid_x = (xmin + xmax) / 2
            centroid_y = (ymin + ymax) / 2
            area = (xmax - xmin) * (ymax - ymin)
            
            food_type = getattr(detection, 'original_food_item', detection.label.replace('.', '').strip())
            
            detection_info.append({
                'detection': detection,
                'centroid': (centroid_x, centroid_y),
                'area': area,
                'food_type': food_type.lower(),
                'confidence': detection.score,
                'bbox': (xmin, ymin, xmax, ymax),
                'index': i
            })
        
        detection_info.sort(key=lambda x: x['confidence'], reverse=True)
        
        final_detections = []
        removed_indices = set()
        
        for i, current in enumerate(detection_info):
            if current['index'] in removed_indices:
                continue
            
            current_centroid = current['centroid']
            current_bbox = current['bbox']
            current_food = current['food_type']
            is_duplicate = False
            
            for selected in final_detections:
                selected_centroid = selected['centroid']
                selected_bbox = selected['bbox']
                selected_food = selected['food_type']
                
                if current_food == selected_food:
                    centroid_distance = np.sqrt(
                        (current_centroid[0] - selected_centroid[0])**2 + 
                        (current_centroid[1] - selected_centroid[1])**2
                    )
                    
                    image_diagonal = np.sqrt(
                        (current_bbox[2] - current_bbox[0])**2 + 
                        (current_bbox[3] - current_bbox[1])**2
                    )
                    relative_distance = centroid_distance / max(image_diagonal, 1)
                    
                    if relative_distance < 0.15:
                        is_duplicate = True
                        break
                
                iou = self.calculate_iou_from_coords(current_bbox, selected_bbox)
                
                if iou > 0.7:
                    is_duplicate = True
                    break
                
                if current_food == selected_food and iou > 0.4:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_detections.append(current)
            else:
                removed_indices.add(current['index'])
        
        clean_detection_objects = [info['detection'] for info in final_detections]
        return clean_detection_objects

    def calculate_iou_from_coords(self, bbox1: tuple, bbox2: tuple) -> float:
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
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

    def smart_duplicate_removal(self, detections: list, confidence_threshold: float = 0.25, 
                               same_class_iou: float = 0.65, cross_class_iou: float = 0.4) -> list:
        if not detections:
            return []
        
        high_conf_detections = [d for d in detections if d.score >= confidence_threshold]
        
        if not high_conf_detections:
            high_conf_detections = [d for d in detections if d.score >= 0.15]
        
        if not high_conf_detections:
            high_conf_detections = detections
        
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
        
        return sorted(filtered_detections, key=lambda x: x.score, reverse=True)

    def calculate_iou(self, box1, box2):
        try:
            x1_min, y1_min, x1_max, y1_max = box1.xmin, box1.ymin, box1.xmax, box1.ymax
            x2_min, y2_min, x2_max, y2_max = box2.xmin, box2.ymin, box2.xmax, box2.ymax
        except AttributeError:
            try:
                x1_min, y1_min, x1_max, y1_max = box1.xyxy
                x2_min, y2_min, x2_max, y2_max = box2.xyxy
            except:
                raise ValueError("Unable to access bounding box coordinates")
        
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

    def get_bbox_coordinates(self, bbox):
        try:
            if hasattr(bbox, 'xmin'):
                return bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
            elif hasattr(bbox, 'xyxy'):
                return bbox.xyxy
            elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                return bbox
            else:
                raise ValueError(f"Unknown bbox format: {type(bbox)}")
        except Exception as e:
            raise e

    def extract_smart_points_for_food(self, bbox, food_type: str, image_shape: tuple) -> list:
        height, width = image_shape[:2]
        
        xmin, ymin, xmax, ymax = self.get_bbox_coordinates(bbox)
        
        # Convert to int for calculations
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        
        cx = int((xmin + xmax) / 2)
        cy = int((ymin + ymax) / 2)
        w = xmax - xmin
        h = ymax - ymin
        
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
            x = max(0, min(width-1, int(x)))
            y = max(0, min(height-1, int(y)))
            
            margin = 5
            if (xmin + margin <= x <= xmax - margin and 
                ymin + margin <= y <= ymax - margin):
                valid_points.append([x, y])
        
        if not valid_points:
            valid_points = [[cx, cy]]
            
        return valid_points

    def detect_bbox_overlaps(self, detections):
        bbox_areas = []
        for i, detection in enumerate(detections):
            bbox = detection.box
            xmin, ymin, xmax, ymax = self.get_bbox_coordinates(bbox)
            area = (xmax - xmin) * (ymax - ymin)
            
            food_type = getattr(detection, 'original_food_item', detection.label.replace('.', '').strip())
            
            bbox_areas.append({
                'index': i,
                'detection': detection,
                'area': area,
                'food_type': food_type
            })
        
        sorted_bboxes = sorted(bbox_areas, key=lambda x: x['area'])
        
        hierarchy = {}
        
        for i, small_bbox in enumerate(sorted_bboxes):
            small_idx = small_bbox['index']
            hierarchy[small_idx] = []
            
            for j, large_bbox in enumerate(sorted_bboxes[i+1:], i+1):
                large_idx = large_bbox['index']
                
                iou = self.calculate_iou(small_bbox['detection'].box, large_bbox['detection'].box)
                
                if iou > 0.3:
                    hierarchy[small_idx].append(large_idx)
        
        return sorted_bboxes, hierarchy

    def segment_food_type_clean(self, image_np: np.ndarray, detection) -> np.ndarray:
        food_type = getattr(detection, 'original_food_item', detection.label.replace('.', '').strip())
        bbox = detection.box
        
        xmin, ymin, xmax, ymax = self.get_bbox_coordinates(bbox)
        
        points = self.extract_smart_points_for_food(bbox, food_type, image_np.shape)
        
        try:
            # Convert to numpy arrays with correct dtype for SAM 2
            point_coords = np.array(points, dtype=np.float32)
            point_labels = np.array([1] * len(points), dtype=np.int32)
            box_coords = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            
            # SAM 2 predict method
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_coords,
                multimask_output=True
            )
            
            # Select best mask and ensure it's boolean
            best_mask = masks[np.argmax(scores)]
            best_mask = best_mask.astype(bool)
            return best_mask
            
        except Exception as e:
            print(f"❌ Error in segment_food_type_clean for {food_type}: {e}")
            print(f"   bbox coords: [{xmin}, {ymin}, {xmax}, {ymax}]")
            print(f"   points: {points}")
            import traceback
            traceback.print_exc()
            raise e

    def remove_close_piece_centroids(self, results: dict, pixel_distance_threshold: int = 8) -> dict:
        if not results:
            return results
        
        all_pieces = []
        for bbox_id, result in results.items():
            food_type = result['food_type'].lower().strip()
            bbox_index = result['bbox_index']
            
            for piece_idx, piece in enumerate(result['pieces']):
                piece_info = {
                    'bbox_id': bbox_id,
                    'bbox_index': bbox_index,
                    'piece_index': piece_idx,
                    'food_type': food_type,
                    'centroid': piece['centroid'],
                    'area': piece['area'],
                    'sam_score': piece.get('sam_score', 0.0),
                    'mask': piece['mask'],
                    'piece_data': piece,
                    'bbox_confidence': result['detection_info']['confidence']
                }
                all_pieces.append(piece_info)
        
        food_groups = {}
        for piece in all_pieces:
            food_type = piece['food_type']
            if food_type not in food_groups:
                food_groups[food_type] = []
            food_groups[food_type].append(piece)
        
        cleaned_pieces = []
        
        for food_type, pieces in food_groups.items():
            if len(pieces) <= 1:
                cleaned_pieces.extend(pieces)
                continue
            
            pieces.sort(key=lambda x: (x['sam_score'], x['area'], x['bbox_confidence']), reverse=True)
            
            kept_pieces = []
            
            for current_piece in pieces:
                current_centroid = current_piece['centroid']
                is_duplicate = False
                
                for kept_piece in kept_pieces:
                    kept_centroid = kept_piece['centroid']
                    
                    euclidean_distance = np.sqrt(
                        (current_centroid[0] - kept_centroid[0])**2 + 
                        (current_centroid[1] - kept_centroid[1])**2
                    )
                    
                    if euclidean_distance <= pixel_distance_threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    kept_pieces.append(current_piece)
            
            cleaned_pieces.extend(kept_pieces)
        
        cleaned_results = {}
        
        pieces_by_bbox = {}
        for piece in cleaned_pieces:
            bbox_id = piece['bbox_id']
            if bbox_id not in pieces_by_bbox:
                pieces_by_bbox[bbox_id] = []
            pieces_by_bbox[bbox_id].append(piece)
        
        for bbox_id, result in results.items():
            if bbox_id in pieces_by_bbox:
                bbox_pieces = pieces_by_bbox[bbox_id]
                
                cleaned_piece_list = []
                for piece_info in bbox_pieces:
                    cleaned_piece_list.append(piece_info['piece_data'])
                
                updated_result = result.copy()
                updated_result['pieces'] = cleaned_piece_list
                updated_result['total_pieces'] = len(cleaned_piece_list)
                updated_result['total_area'] = sum(piece['area'] for piece in cleaned_piece_list)
                
                cleaned_results[bbox_id] = updated_result
        
        return cleaned_results

    def detect_all_food_pieces(self, image_path: str):
        print("🍽️ Starting COMPLETE food plate semantic segmentation with SAM 2 and ADAPTIVE DETECTION...")
        start_time = time.time()

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        self.sam2_predictor.set_image(image_np)

        # Step 1: Describe the plate
        step1_start = time.time()
        print("\n🧠 STEP 1: Analyzing plate contents...")
        food_items = self.describe_plate(image_path)
        print(f"   🕒 Step 1 completed in {time.time() - step1_start:.2f} seconds")

        # Step 2: ADAPTIVE localization with hierarchical fallback
        step2_start = time.time()
        print("\n📍 STEP 2: Adaptive food localization with hierarchical search...")
        detections = self.localize_all_food(image_path, food_items)
        print(f"   Found {len(detections)} detections")
        for i, det in enumerate(detections):
            food_type = getattr(det, 'original_food_item', det.label.replace('.', '').strip())
            print(f"      Detection {i}: {food_type} (confidence: {det.score:.3f})")
        print(f"   🕒 Step 2 completed in {time.time() - step2_start:.2f} seconds")

        if not detections:
            print("❌ No detections found even with adaptive search. Cannot proceed.")
            return None

        # Step 2.5: Detect overlaps and create hierarchical processing order
        step2_5_start = time.time()
        print("\n🔍 STEP 2.5: Analyzing bounding box hierarchy...")
        sorted_bboxes, hierarchy = self.detect_bbox_overlaps(detections)
        print(f"   🕒 Step 2.5 completed in {time.time() - step2_5_start:.2f} seconds")

        # Step 3: Create semantic masks with SAM 2 and proper hierarchical processing (smallest first)
        step3_start = time.time()
        print("\n🎯 STEP 3: Creating semantic masks with SAM 2 step-by-step hierarchical processing...")
        print(f"   Processing {len(sorted_bboxes)} bounding boxes...")
        bbox_masks = {}
        bbox_info = {}
        reserved_regions = np.zeros(image_np.shape[:2], dtype=int)

        for i, bbox_data in enumerate(sorted_bboxes):
            detection = bbox_data['detection']
            bbox_index = int(bbox_data['index'])  # Convert to int
            food_type = bbox_data['food_type']
            area = bbox_data['area']
            bbox_id = f"bbox_{bbox_index}_{food_type}"

            print(f"   📍 Processing bbox {i+1}/{len(sorted_bboxes)}: {food_type}...")

            try:
                raw_mask = self.segment_food_type_clean(image_np, detection)

                overlap_with_reserved = np.logical_and(raw_mask, reserved_regions > 0).astype(bool)
                overlap_area = np.sum(overlap_with_reserved)

                excluded_items = []
                if overlap_area > 0:
                    overlapping_bbox_indices = np.unique(reserved_regions[overlap_with_reserved])
                    overlapping_bbox_indices = overlapping_bbox_indices[overlapping_bbox_indices > 0]

                    for overlapping_idx in overlapping_bbox_indices:
                        overlapping_idx = int(overlapping_idx)  # Convert to int for indexing
                        overlapping_detection = detections[overlapping_idx]
                        overlapping_food_type = getattr(overlapping_detection, 'original_food_item', 
                                                    overlapping_detection.label.replace('.', '').strip())
                        excluded_items.append(overlapping_food_type)

                    final_mask = np.logical_and(raw_mask, reserved_regions == 0)
                else:
                    final_mask = raw_mask
                
                # Convert mask to boolean type for indexing
                final_mask = final_mask.astype(bool)

                reserved_regions[final_mask] = bbox_index

                bbox_masks[bbox_id] = final_mask

                xmin, ymin, xmax, ymax = self.get_bbox_coordinates(detection.box)

                bbox_info[bbox_id] = {
                    'detection': detection,
                    'mask': final_mask,
                    'bbox': [xmin, ymin, xmax, ymax],
                    'confidence': detection.score,
                    'food_type': food_type,
                    'bbox_index': bbox_index,
                    'area': area,
                    'excluded_items': excluded_items,
                    'is_container': len(excluded_items) > 0,
                    'reserved_area': np.sum(final_mask),
                    'successful_variant': getattr(detection, 'successful_variant', food_type)
                }
                print(f"      ✅ Success! Mask area: {np.sum(final_mask)} pixels")

            except Exception as e:
                print(f"      ❌ Failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"   Successfully processed {len(bbox_masks)}/{len(sorted_bboxes)} bounding boxes")
        print(f"   🕒 Step 3 completed in {time.time() - step3_start:.2f} seconds")

        # Step 4: Run complete automatic segmentation with SAM 2
        step4_start = time.time()
        print("\n🧩 STEP 4: Running complete automatic segmentation with SAM 2...")
        complete_masks = self.sam2_generator.generate(image_np)
        print(f"   🕒 Step 4 completed in {time.time() - step4_start:.2f} seconds")

        # Step 5: Find intersections and count individual pieces for each bounding box
        step5_start = time.time()
        print("\n🔗 STEP 5: Finding intersections and counting individual pieces for each bounding box...")
        results = {}

        for bbox_id, bbox_mask in bbox_masks.items():
            bbox_data = bbox_info[bbox_id]
            food_type = bbox_data['food_type']
            bbox_index = bbox_data['bbox_index']
            
            # Check if food is granular (should not be segmented into pieces)
            is_granular = any(granular in food_type.lower() for granular in self.granular_foods)
            
            if is_granular:
                # For granular foods, treat the entire mask as one piece
                print(f"   🌾 {food_type.title()} is granular - treating as 1 bulk region (not segmenting)")
                ys, xs = np.where(bbox_mask)
                if len(xs) > 0:
                    centroid = (int(np.mean(xs)), int(np.mean(ys)))
                    intersecting_pieces = [{
                        'mask': bbox_mask,
                        'area': np.sum(bbox_mask),
                        'centroid': centroid,
                        'sam_score': 1.0
                    }]
                else:
                    intersecting_pieces = []
            else:
                # For non-granular foods, segment into individual pieces
                intersecting_pieces = []

                for mask in complete_masks:
                    segmentation = mask['segmentation']
                    intersection = np.logical_and(segmentation, bbox_mask)

                    intersection_area = np.sum(intersection)
                    if intersection_area > 100:
                        ys, xs = np.where(intersection)
                        if len(xs) > 0:
                            centroid = (int(np.mean(xs)), int(np.mean(ys)))

                            intersecting_pieces.append({
                                'mask': intersection,
                                'area': intersection_area,
                                'centroid': centroid,
                                'sam_score': mask.get('stability_score', 0.0)
                            })

                intersecting_pieces.sort(key=lambda x: x['area'], reverse=True)

            results[bbox_id] = {
                'bbox_id': bbox_id,
                'food_type': food_type,
                'bbox_index': bbox_index,
                'total_pieces': len(intersecting_pieces),
                'pieces': intersecting_pieces,
                'bbox_mask': bbox_mask,
                'detection_info': bbox_data,
                'total_area': np.sum(bbox_mask),
                'is_granular': is_granular
            }
        print(f"   🕒 Step 5 completed in {time.time() - step5_start:.2f} seconds")

        # Step 7: Piece-level centroid duplicate removal
        step7_start = time.time()
        print(f"\n🧹 STEP 7: Removing duplicate individual food pieces with close centroids...")
        cleaned_results = self.remove_close_piece_centroids(results, pixel_distance_threshold=8)
        print(f"   🕒 Step 7 completed in {time.time() - step7_start:.2f} seconds")

        # Step 7.5: Calculate pieces per food type
        step7_5_start = time.time()
        print(f"\n📊 STEP 7.5: Calculating pieces per food type...")
        pieces_per_food_type = self.calculate_pieces_per_food_type(cleaned_results)
        print(f"   🕒 Step 7.5 completed in {time.time() - step7_5_start:.2f} seconds")

        final_pieces = sum(result['total_pieces'] for result in cleaned_results.values())
        final_bboxes = len(cleaned_results)

        computation_time = time.time() - start_time

        # Step 8: Create comprehensive visualizations with cleaned results
        step8_start = time.time()
        print(f"\n🎨 STEP 8: Creating visualizations with cleaned results...")
        self.visualize_complete_results(image_path, cleaned_results, detections)
        print(f"   🕒 Step 8 completed in {time.time() - step8_start:.2f} seconds")

        print(f"\n🎉 COMPLETE ADAPTIVE SEMANTIC SEGMENTATION RESULTS:")

        # Display results by bounding box
        for bbox_id, result in cleaned_results.items():
            pieces_count = result['total_pieces']
            food_type = result['food_type']
            bbox_index = result['bbox_index']
            confidence = result['detection_info']['confidence']
            is_granular = result.get('is_granular', False)
            
            if is_granular:
                print(f"   📦 BBox {bbox_index} ({food_type.title()}): 1 bulk region 🌾")
            else:
                print(f"   📦 BBox {bbox_index} ({food_type.title()}): {pieces_count} pieces")
            print(f"      📊 Detection confidence: {confidence:.3f}")

        # Display results by food type
        print(f"\n🍽️ PIECES COUNT PER FOOD TYPE:")
        
        # Separate granular and non-granular foods
        granular_items = {}
        non_granular_items = {}
        
        for food_type, count in sorted(pieces_per_food_type.items()):
            is_granular = any(granular in food_type.lower() for granular in self.granular_foods)
            if is_granular:
                granular_items[food_type] = count
            else:
                non_granular_items[food_type] = count
        
        # Display non-granular foods (with piece count)
        for food_type, count in sorted(non_granular_items.items()):
            print(f"   🥗 {food_type.title()}: {count} pieces")
        
        # Display granular foods (bulk regions)
        if granular_items:
            print(f"\n   🌾 Bulk/Granular foods (not segmented into pieces):")
            for food_type, count in sorted(granular_items.items()):
                print(f"   🌾 {food_type.title()}: 1 bulk region")

        print(f"\n   📦 TOTAL BOUNDING BOXES: {final_bboxes}")
        print(f"   🍽️ TOTAL FOOD PIECES DETECTED: {final_pieces}")
        print(f"   🕒 Total computation time: {computation_time:.2f} seconds")

        return {
            'results': cleaned_results,
            'pieces_per_food_type': pieces_per_food_type,
            'total_pieces': final_pieces,
            'total_bboxes': final_bboxes,
            'computation_time': computation_time,
            'detections': detections
        }

    def calculate_pieces_per_food_type(self, results: dict) -> dict:
        """
        Calculate the total number of pieces for each food type.
        
        Args:
            results: Dictionary with bbox_id as keys and result data as values
            
        Returns:
            Dictionary with food_type as keys and total piece count as values
        """
        pieces_per_food_type = {}
        
        for bbox_id, result in results.items():
            food_type = result['food_type'].lower().strip()
            piece_count = result['total_pieces']
            
            if food_type in pieces_per_food_type:
                pieces_per_food_type[food_type] += piece_count
            else:
                pieces_per_food_type[food_type] = piece_count
        
        return pieces_per_food_type

    def visualize_complete_results(self, image_path: str, results: dict, detections: list):
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        all_food_types = []
        for result in results.values():
            all_food_types.append(result['food_type'])
        unique_food_types = sorted(list(set(all_food_types)))
        
        colors_palette = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], 
            [0, 255, 255], [255, 128, 0], [128, 0, 255], [255, 192, 203], [0, 128, 0],
        ]
        
        food_type_colors = {}
        for i, food_type in enumerate(unique_food_types):
            food_type_colors[food_type] = colors_palette[i % len(colors_palette)]
        
        # Create visualization
        pieces_vis = image_np.copy()
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['bbox_index'])
        
        for bbox_id, result in sorted_results:
            food_type = result['food_type']
            base_color = np.array(food_type_colors[food_type])
            
            for piece_idx, piece in enumerate(result['pieces']):
                if len(result['pieces']) > 1:
                    brightness_factor = 0.7 + 0.3 * (piece_idx / max(1, len(result['pieces']) - 1))
                    piece_color = (base_color * brightness_factor).astype(np.uint8)
                else:
                    piece_color = base_color.astype(np.uint8)
                
                pieces_vis[piece['mask']] = piece_color
                cv2.circle(pieces_vis, piece['centroid'], radius=3, color=(255, 255, 255), thickness=-1)
        
        output_path = os.path.join(self.output_dir, "individual_pieces.png")
        cv2.imwrite(output_path, cv2.cvtColor(pieces_vis, cv2.COLOR_RGB2BGR))
        print(f"   💾 Saved visualization to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Complete Food Plate Segmentation with SAM 2")
    parser.add_argument("--image", type=str, default="images/snack-plate-meal-1_orig.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default="large", choices=["tiny", "small", "base_plus", "large"], help="SAM 2 model size")
    parser.add_argument("--output-dir", type=str, default="~/Documents/src/vision_processing/scripts", help="Output directory")

    args = parser.parse_args()
    image_path = args.image
    
    sam2_checkpoint = os.path.expanduser(f"~/segment-anything-2/checkpoints/sam2.1_hiera_{args.model}.pt")
    
    config_map = {
        "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "large": "configs/sam2.1/sam2.1_hiera_l.yaml"
    }
    sam2_config = config_map[args.model]

    print("🌟 COMPLETE FOOD PLATE DETECTOR WITH SAM 2")
    print("Features:")
    print("• ✅ SAM 2 for faster and better segmentation")
    print("• ✅ Automatic bulk food detection (no individual pieces)")
    print("• ✅ Fast contour-based piece detection (instead of slow mask generator)")
    print("• ✅ Optimized batch food localization")
    print("• ✅ Mask cleaning with morphological operations")
    print("="*80)

    detector = CompleteFoodDetector(sam2_checkpoint, sam2_config, output_dir=args.output_dir)
    
    result = detector.detect_all_food_pieces(image_path)
    
    if result:
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"   📦 Total Bounding Boxes: {result['total_bboxes']}")
        print(f"   🥗 Total Individual Pieces: {result['total_pieces']}")
        
        # Separate granular and non-granular foods
        granular_items = {}
        non_granular_items = {}
        
        for food_type, count in sorted(result['pieces_per_food_type'].items()):
            is_granular = any(granular in food_type.lower() for granular in detector.granular_foods)
            if is_granular:
                granular_items[food_type] = count
            else:
                non_granular_items[food_type] = count
        
        # Display non-granular foods
        if non_granular_items:
            print(f"\n   🍽️ Individual pieces per food type:")
            for food_type, count in sorted(non_granular_items.items()):
                print(f"      • {food_type.title()}: {count} pieces")
        
        # Display granular foods
        if granular_items:
            print(f"\n   🌾 Bulk/Granular foods (not segmented):")
            for food_type, count in sorted(granular_items.items()):
                print(f"      • {food_type.title()}: 1 bulk region")
        
        print(f"\n   ⏱️ Processing Time: {result['computation_time']:.2f} seconds")
    else:
        print("\n❌ ANALYSIS FAILED:")
        print("   No food could be detected or segmented.")