"""
SAM-3 based food detection and segmentation service with adaptive thresholding
"""

import os
import torch
import numpy as np
import random
import base64
import tempfile
import time
from datetime import datetime
from PIL import Image
from scipy.ndimage import zoom
import cv2
import matplotlib.pyplot as plt

# Import SAM-3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Initialize device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load SAM-3 model globally to avoid reloading
print("Loading SAM-3 model...")
sam3_model = build_sam3_image_model()
sam3_processor = Sam3Processor(sam3_model, confidence_threshold=0.1)
print(f"SAM-3 model loaded successfully!")

def compute_adaptive_threshold(scores, base_threshold=0.1):
    """
    Adaptive confidence threshold for filtering detections.
    Only meaningful with 2+ detections.
    
    Args:
        scores: List or array of detection confidence scores
        base_threshold: Minimum threshold to enforce (default 0.1)
    
    Returns:
        Adaptive threshold value
    """
    scores = np.array(scores, dtype=float)

    # No detections or single detection → use base threshold
    if len(scores) <= 1:
        return base_threshold

    # CASE 2+: Unified clustering logic
    sorted_scores = np.sort(scores)[::-1]
    gaps = np.diff(sorted_scores)
    gaps = np.abs(gaps)  # Take absolute values for gap magnitude
    largest_gap_idx = np.argmax(gaps)
    largest_gap = gaps[largest_gap_idx]

    cluster_size = largest_gap_idx + 1
    cluster = sorted_scores[:cluster_size]
    cluster_mean = cluster.mean()
    cluster_std = cluster.std()

    # DEBUG: Print analysis
    print(f"\n=== ADAPTIVE THRESHOLD DEBUG ===")
    print(f"Total detections: {len(scores)}")
    print(f"Sorted scores (top 10): {sorted_scores[:10]}")
    print(f"Gaps (top 10): {gaps[:10]}")
    print(f"Largest gap: {largest_gap:.4f} at index {largest_gap_idx}")
    print(f"Cluster size: {cluster_size} (scores up to index {largest_gap_idx})")
    print(f"Cluster mean: {cluster_mean:.4f}, std: {cluster_std:.4f}")
    print(f"Cluster scores: {cluster}")

    # RULE A: Tight high-confidence cluster
    if cluster_mean > 0.5 and cluster_std < 0.10:
        threshold = max(cluster[-1], base_threshold)
        print(f"RULE A applied: tight high-confidence cluster → threshold = {threshold:.4f}")
        return threshold

    # RULE B: Significant confidence gap
    if largest_gap > 0.1:
        threshold = max(cluster[-1], base_threshold)
        print(f"RULE B applied: significant gap > 0.10 → threshold = {threshold:.4f}")
        return threshold

    # RULE C: General case — statistical fallback
    global_mean = sorted_scores.mean()
    global_std = sorted_scores.std()

    if global_std < 0.05:
        threshold = max(sorted_scores[-1], base_threshold)
        print(f"RULE C (low std): global_std = {global_std:.4f} < 0.05 → threshold = {threshold:.4f}")
        return threshold

    threshold = max(global_mean, base_threshold)
    print(f"RULE C (statistical): global_mean = {global_mean:.4f}, global_std = {global_std:.4f} → threshold = {threshold:.4f}")
    print(f"=== END DEBUG ===\n")
    return threshold

def detect_and_segment_food_item(image_base64, food_item, save_outputs=True, keep_largest_only=False, save_visualization=False):
    """
    Detect and segment a specific food item using SAM-3
    
    Args:
        image_base64: Base64 encoded image
        food_item: Name of food item to detect and segment
        save_outputs: Whether to save detection outputs (JSON) to disk
        keep_largest_only: For granular foods, keep only the largest mask (useful for scoop actions)
        save_visualization: Whether to generate and save visualization figures (set True for local testing)
        
    Returns:
        Dictionary with detection results and segmentation masks, or error info
    """
    try:
        # Generate timestamp for unique file names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        
        # Save to temporary file and load with PIL
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file.write(image_data)
            temp_path = temp_file.name
        
        image = Image.open(temp_path).convert("RGB")
        image_np = np.array(image)
        
        # SAM-3 detection and segmentation for food item
        print(f"\nDetecting {food_item} in image...")
        food_inference_state = sam3_processor.set_image(image)
        
        # Run SAM-3 detection and segmentation
        food_output = sam3_processor.set_text_prompt(state=food_inference_state, prompt=food_item)
        
        # Extract results
        food_masks = food_output["masks"]
        food_boxes = food_output["boxes"]
        food_scores = food_output["scores"]
        
        print(f"SAM-3 detection results: {len(food_masks)} instances found")
        print(f"Scores: {food_scores.tolist()}")
        
        # Check if any detections found
        if len(food_masks) == 0:
            print(f"No {food_item} detected in image")
            os.unlink(temp_path)
            return {
                "success": False,
                "error": f"No {food_item} detected in image",
                "detections": [],
                "masks": []
            }
        
        # Convert to numpy arrays first
        matching_boxes = food_boxes.cpu().numpy()
        matching_masks = food_masks.cpu().numpy()
        matching_scores = food_scores.cpu().numpy()
        
        # Apply adaptive confidence threshold to filter detections
        adaptive_threshold = compute_adaptive_threshold(matching_scores, base_threshold=0.1)
        print(f"Adaptive confidence threshold: {adaptive_threshold:.4f}")
        
        # Filter detections by adaptive threshold
        valid_mask = matching_scores >= adaptive_threshold
        matching_boxes = matching_boxes[valid_mask]
        matching_masks = matching_masks[valid_mask]
        matching_scores = matching_scores[valid_mask]
        
        if len(matching_scores) == 0:
            print(f"No detections passed adaptive threshold filtering")
            os.unlink(temp_path)
            return {
                "success": False,
                "error": f"No {food_item} detections passed confidence threshold",
                "detections": [],
                "masks": []
            }
        
        print(f"After adaptive threshold filtering: {len(matching_scores)} detections remain")
        
        # For granular foods (scoop), keep only the largest mask BEFORE visualization
        if keep_largest_only and len(matching_masks) > 1:
            print(f"Keeping only the largest mask (keep_largest_only=True)")
            # Calculate mask sizes from numpy arrays
            mask_sizes = np.array([np.sum(mask > 0) for mask in matching_masks])
            largest_idx = np.argmax(mask_sizes)
            
            print(f"Largest mask size: {mask_sizes[largest_idx]} pixels (removed {len(matching_masks) - 1} smaller masks)")
            
            # Keep only the largest
            matching_boxes = matching_boxes[largest_idx:largest_idx+1]
            matching_masks = matching_masks[largest_idx:largest_idx+1]
            matching_scores = matching_scores[largest_idx:largest_idx+1]
        
        # Ensure all masks have correct dimensions matching the image
        target_height, target_width = image_np.shape[0], image_np.shape[1]
        resized_masks = []
        for i, mask in enumerate(matching_masks):
            # Squeeze any extra dimensions
            while mask.ndim > 2:
                mask = mask.squeeze(0)
            
            if mask.shape != (target_height, target_width):
                print(f"Resizing food mask {i} from {mask.shape} to {(target_height, target_width)}")
                zoom_y = target_height / mask.shape[0]
                zoom_x = target_width / mask.shape[1]
                resized_mask = zoom(mask.astype(float), (zoom_y, zoom_x), order=0) > 0.5
                resized_masks.append(resized_mask)
            else:
                # Ensure mask is boolean (same as TEST_SAM3.py: masks > 0)
                resized_masks.append(mask > 0)
        matching_masks = np.array(resized_masks)
        
        # Prepare results
        detection_results = []
        for i, (box, mask, score) in enumerate(zip(matching_boxes, matching_masks, matching_scores)):
            detection_results.append({
                "bbox": box.tolist(),
                "label": food_item,  # SAM-3 uses the prompt as the label
                "mask": mask.astype(bool),
                "score": float(score),
                "index": i
            })
        
        # Return all detections
        if detection_results:
            print(f"Found {len(detection_results)} detections of {food_item}")
            
            # Always save detection results as JSON if save_outputs is True
            if save_outputs:
                output_dir = "/home/felixog/robot-assisted-feeding/services/action_parameterization_service/results"
                os.makedirs(output_dir, exist_ok=True)
                
                # Encode masks as base64 PNG for JSON serialization
                import json
                detections_serializable = []
                for det in detection_results:
                    # Encode mask as PNG and then base64
                    mask_uint8 = (det["mask"].astype(np.uint8) * 255)
                    _, mask_png = cv2.imencode('.png', mask_uint8)
                    mask_base64 = base64.b64encode(mask_png).decode('utf-8')
                    
                    detections_serializable.append({
                        "bbox": det["bbox"],
                        "label": det["label"],
                        "score": det["score"],
                        "index": det["index"],
                        "mask_base64": mask_base64
                    })
                
                # Save detection results as JSON
                json_path = os.path.join(output_dir, f"sam3_detection_{timestamp}.json")
                with open(json_path, 'w') as f:
                    json.dump({
                        "success": True,
                        "detections": detections_serializable,
                        "image_size": (image.width, image.height),
                        "detected_item": food_item,
                        "timestamp": timestamp
                    }, f, indent=2)
                print(f"Detection results JSON saved to: {json_path}")
            
            # Generate visualization only if requested (for local testing)
            if save_visualization:
                output_dir = "/home/felixog/robot-assisted-feeding/services/action_parameterization_service/results"
                os.makedirs(output_dir, exist_ok=True)
                
                # Create visualization matching TEST_SAM3.py exactly
                image_pil = Image.fromarray(image_np)
                
                # Use matplotlib colormap for rainbow colors
                cmap = plt.get_cmap("hsv")
                colors_normalized = [cmap(i / (len(detection_results) + 1))[:3] for i in range(len(detection_results))]
                
                # Start with RGBA image for transparent overlay (like TEST_SAM3.py)
                final_overlay_image = image_pil.convert("RGBA")
                
                # Apply transparent overlays for each detection
                for idx, detection in enumerate(detection_results):
                    mask = detection["mask"]
                    
                    # Get rainbow color for this detection (RGB 0-255)
                    color_rgb = tuple(int(c * 255) for c in colors_normalized[idx])
                    
                    # Apply transparent overlay (same as TEST_SAM3.py overlay_mask_transparent)
                    alpha_val = 0.5
                    color_layer = Image.new("RGBA", final_overlay_image.size, color_rgb + (0,))
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    
                    try:
                        mask_pil = Image.fromarray(mask_uint8, mode='L')
                    except Exception:
                        mask_uint8 = np.squeeze(mask_uint8)
                        mask_pil = Image.fromarray(mask_uint8, mode='L')
                    
                    if mask_pil.size != final_overlay_image.size:
                        mask_pil = mask_pil.resize(final_overlay_image.size, resample=Image.NEAREST)
                    
                    # Apply alpha to mask
                    mask_pil = mask_pil.point(lambda x: int(255 * alpha_val) if x > 0 else 0)
                    color_layer.putalpha(mask_pil)
                    final_overlay_image = Image.alpha_composite(final_overlay_image, color_layer)
                
                # Convert to RGB for drawing bboxes and labels
                final_overlay_rgb = final_overlay_image.convert("RGB")
                final_overlay_np = np.array(final_overlay_rgb)
                
                # Draw bounding boxes and labels with clean font
                for idx, detection in enumerate(detection_results):
                    bbox = detection["bbox"]
                    label = detection["label"]
                    score = detection["score"]
                    
                    # Get same rainbow color for bbox
                    color_rgb = tuple(int(c * 255) for c in colors_normalized[idx])
                    
                    # Draw bounding box
                    cv2.rectangle(final_overlay_np, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_rgb, 2)
                    
                    # Add label with clean white text on semi-transparent black background
                    label_text = f"{label} #{idx+1} ({score:.2f})"
                    font = cv2.FONT_HERSHEY_PLAIN
                    font_scale = 1
                    thickness = 2
                    
                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                    
                    # Position for text (above bbox)
                    text_x = int(bbox[0])
                    text_y = int(bbox[1]) - 8
                    
                    # Draw semi-transparent black background rectangle for text
                    padding = 10
                    vertical_padding = 10
                    overlay = final_overlay_np.copy()
                    cv2.rectangle(overlay, 
                                (text_x, text_y - text_height - vertical_padding),
                                (text_x + text_width + padding * 2, int(bbox[1])),
                                (0, 0, 0), -1)
                    # Blend with 50% transparency
                    cv2.addWeighted(overlay, 0.5, final_overlay_np, 0.5, 0, final_overlay_np)
                    
                    # Draw white text on transparent background
                    cv2.putText(final_overlay_np, label_text, (text_x + padding, text_y - 5), 
                               font, font_scale, (255, 255, 255), thickness)
                
                # Save visualization
                vis_path = os.path.join(output_dir, f"sam3_detection_{timestamp}.png")
                cv2.imwrite(vis_path, cv2.cvtColor(final_overlay_np, cv2.COLOR_RGB2BGR))
                
                print(f"Visualization saved to: {vis_path}")
           
            # Clean up temp file
            os.unlink(temp_path)
            
            return {
                "success": True,
                "detections": detection_results,
                "image_size": (image.width, image.height),
                "detected_item": food_item
            }
        else:
            # Clean up temp file
            os.unlink(temp_path)
            
            return {
                "success": False,
                "error": f"No valid detections found for {food_item}",
                "detections": [],
                "masks": []
            }
        
    except Exception as e:
        print(f"Error in detect_and_segment_food_item: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Detection and segmentation failed: {str(e)}",
            "detections": [],
            "masks": []
        }


def get_random_detection_mask(detection_results):
    """
    Get a random detection mask from the results
    
    Args:
        detection_results: Results from detect_and_segment_food_item
        
    Returns:
        Single mask array or None if no detections
    """
    if not detection_results["success"] or not detection_results["detections"]:
        return None
    
    # Select random detection
    selected_detection = random.choice(detection_results["detections"])
    print(f"Selected detection: {selected_detection['label']} at bbox {selected_detection['bbox']}")
    
    return selected_detection["mask"]


if __name__ == "__main__":
    # Test with a sample image
    test_image_path = "/home/felixog/Downloads/test_plate2.png"
    
    if os.path.exists(test_image_path):
        # Convert test image to base64
        with open(test_image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Test detection and segmentation (outputs will be saved automatically)
        results = detect_and_segment_food_item(image_base64, "carrot", save_outputs=True)
        print("\n=== Detection Results ===")
        print(f"Success: {results['success']}")
        
        if results["success"]:
            print(f"Number of detections: {len(results['detections'])}")
            for i, det in enumerate(results['detections']):
                print(f"  Detection {i}: bbox={det['bbox']}, score={det['score']:.3f}")
            
            # Test getting random mask
            mask = get_random_detection_mask(results)
            if mask is not None:
                print(f"\nRandom mask selected with shape: {mask.shape}")
            else:
                print("\nFailed to get random mask")
        else:
            print(f"Detection failed: {results['error']}")
            if results.get("plate_detected", False):
                print("Note: Plate was detected successfully")
    else:
        print(f"Test image not found: {test_image_path}")
