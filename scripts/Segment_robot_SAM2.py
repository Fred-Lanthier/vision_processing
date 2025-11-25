import os
import sys
import time
import cv2
import numpy as np
import torch
from sam3.model_builder import build_sam3_video_predictor

# Try to import rospkg
try:
    import rospkg
    ROSPKG_AVAILABLE = True
except ImportError:
    ROSPKG_AVAILABLE = False

class SegmentRobotSAM2:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 [SAM2-Style] Initializing on {self.device}...")
        
        # Initialize SAM 3 High-Level Predictor (to get text prompt support)
        self.video_predictor = build_sam3_video_predictor()
        print("✅ Model loaded.")
        
        self.inference_state = None

    def process_video_sequence(self, images_dir, output_dir, prompt="robot"):
        # Get image files
        image_files = sorted([
            os.path.join(images_dir, f) 
            for f in os.listdir(images_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if not image_files:
            print("❌ No images found!")
            return

        print(f"📂 Found {len(image_files)} images.")
        
        # 1. Start Session (High Level)
        print("🎬 Starting session...")
        try:
            # This loads images into the state
            response = self.video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=images_dir,
                )
            )
            session_id = response['session_id']
            print(f"   ✅ Session started. ID: {session_id}")
            
            # DEBUG: Inspect object
            print(f"   🔍 Predictor type: {type(self.video_predictor)}")
            print(f"   🔍 Predictor dir: {dir(self.video_predictor)}")
            
        except Exception as e:
            print(f"❌ Error starting session: {e}")
            return

        # 2. Add Prompt (High Level) - to get initial mask
        print(f"   ➕ Adding text prompt '{prompt}' to frame 0...")
        try:
            response = self.video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=prompt
                )
            )
            # This updates the state with the mask for frame 0
        except Exception as e:
             print(f"❌ Error adding prompt: {e}")
             return

        # 3. Switch to Low-Level Tracker for Propagation
        print("   🌊 Propagating mask (SAM 2 Style)...")
        
        # Attempt to access the low-level tracker and state
        # The 'inference_state' needed by tracker.propagate_in_video is usually a dict.
        # We need to find it.
        # self.video_predictor is a Sam3VideoPredictor.
        # It likely has a 'state' or 'inference_states' attribute.
        
        try:
            # Access the underlying predictor (assuming rank 0)
            if hasattr(self.video_predictor, 'predictors'):
                # Multi-GPU wrapper
                predictor_core = self.video_predictor.predictors[0]
            else:
                predictor_core = self.video_predictor

            # Find inference state
            # Usually mapped by session_id
            if hasattr(predictor_core, 'inference_states'):
                inference_state = predictor_core.inference_states[session_id]
            elif hasattr(predictor_core, 'inference_state'):
                inference_state = predictor_core.inference_state
            elif hasattr(predictor_core, 'state'):
                inference_state = predictor_core.state
            else:
                print(f"⚠️ Could not find inference_state in {type(predictor_core)}")
                # Filter dir for relevant keys
                keys = [k for k in dir(predictor_core) if 'state' in k or 'infer' in k or 'session' in k]
                print(f"   Relevant keys: {keys}")
                
                # Check for private attributes
                if hasattr(predictor_core, '_inference_states'):
                     print("   Found _inference_states (private)")
                     inference_state = predictor_core._inference_states[session_id]
                elif hasattr(predictor_core, '_get_session'):
                     print("   Found _get_session method")
                     # _get_session likely returns the state
                     inference_state = predictor_core._get_session(session_id)
                else:
                     inference_state = None

            # Find tracker
            # predictor_core.model is likely Sam3VideoModel
            print(f"   🔍 Searching for tracker in {type(predictor_core)}...")
            if hasattr(predictor_core, 'model'):
                print(f"   🔍 Found .model: {type(predictor_core.model)}")
                if hasattr(predictor_core.model, 'tracker'):
                    tracker = predictor_core.model.tracker
                    print("   🔍 Found .model.tracker")
                else:
                    print(f"   ❌ .model has no tracker. Dir: {dir(predictor_core.model)}")
                    tracker = None
            elif hasattr(predictor_core, 'tracker'):
                tracker = predictor_core.tracker
                print("   🔍 Found .tracker")
            else:
                print(f"⚠️ Could not find tracker in {type(predictor_core)}")
                print(f"   Dir: {dir(predictor_core)}")
                tracker = None
                
            if tracker and inference_state:
                print("   ✅ Found Low-Level Tracker and State. Using SAM 2 API.")
                
                # Recursive search for SAM 2 state
                def find_sam2_state(d, depth=0):
                    if depth > 10: return None
                    if not isinstance(d, dict): return None
                    
                    # DEBUG: Print keys at this level
                    print(f"   🔎 Depth {depth} keys: {list(d.keys())}", flush=True)
                    
                    if 'obj_idx_to_id' in d: return d
                    
                    # Check specific keys first
                    if 'tracker_inference_states' in d:
                        tis = d['tracker_inference_states']
                        if isinstance(tis, dict):
                            for k, v in tis.items():
                                res = find_sam2_state(v, depth+1)
                                if res: return res
                    
                    if 'state' in d:
                        res = find_sam2_state(d['state'], depth+1)
                        if res: return res
                        
                    # General search (expensive but safe)
                    for k, v in d.items():
                        # print(f"   🔎 Depth {depth} Key {k} Type {type(v)}", flush=True)
                        if isinstance(v, dict):
                            # Avoid re-visiting 'state' or 'tracker_inference_states' if already checked
                            if k in ['state', 'tracker_inference_states']: continue
                            res = find_sam2_state(v, depth+1)
                            if res: return res
                        elif isinstance(v, list) and len(v) > 0:
                             # Check if it's a list of dicts
                             if isinstance(v[0], dict):
                                 res = find_sam2_state(v[0], depth+1)
                                 if res: return res
                    return None

                print("   🔍 Searching for SAM 2 state...", flush=True)
                found_state = find_sam2_state(inference_state)
                if found_state:
                    print("   ✅ Found SAM 2 state!", flush=True)
                    inference_state = found_state
                    print(f"   🆔 Passed State ID: {id(inference_state)}", flush=True)
                    print(f"   🔍 obj_idx_to_id value: {inference_state['obj_idx_to_id']}", flush=True)
                else:
                    print("   ❌ Could not find SAM 2 state in inference_state.", flush=True)
                    print(f"   Top level keys: {list(inference_state.keys())}", flush=True)

                # SAM 2 API: propagate_in_video(inference_state, ...)
                # Yields: (frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores)
                # Try positional arguments: inference_state, start_frame_idx, max_frame_num_to_track, reverse, propagate_preflight
                generator = tracker.propagate_in_video(
                    inference_state,
                    0,
                    len(image_files),
                    False,
                    True
                )
                use_low_level = True
            else:
                print("   ⚠️ Falling back to High-Level Generator (SAM 3 API).")
                # Fallback to the high-level generator we used before
                generator = self.video_predictor.handle_stream_request(
                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                        start_frame_idx=0
                    )
                )
                use_low_level = False
                
        except Exception as e:
            print(f"❌ Error setting up propagation: {e}")
            return

        # Buffer for results
        results_buffer = []
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        processed_count = 0
        
        # Enable Mixed Precision
        use_half = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_half else torch.float16
        print(f"⚡ Mixed Precision Enabled: {amp_dtype}")

        inference_start_time = time.time()
        pure_inference_time = 0
        inference_frames = 0
        
        step_start = time.time()
        
        with torch.autocast(device_type=self.device, dtype=amp_dtype):
            for step_result in generator:
                pure_inference_time += (time.time() - step_start)
                inference_frames += 1
                
                if use_low_level:
                    # Unpack SAM 2 tuple
                    # (frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores)
                    out_frame_idx, out_obj_ids, out_mask_logits, out_video_res_masks, out_scores = step_result
                    # Note: SAM 2 might return logits or probs. video_res_masks is usually the one to use.
                    # Let's use out_video_res_masks (high res)
                    masks = out_video_res_masks
                else:
                    # Unpack SAM 3 dict
                    out_frame_idx = step_result['frame_index']
                    outputs = step_result['outputs']
                    masks = outputs['out_binary_masks']

                # Process mask
                if isinstance(masks, torch.Tensor):
                    masks = masks.detach().cpu().numpy()
                
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
            
            if isinstance(frame_idx, int):
                image_path = image_files[frame_idx]
            else:
                image_path = os.path.join(images_dir, str(frame_idx))
                
            image_name = os.path.basename(image_path)
            
            image = cv2.imread(image_path)
            if image is None: continue
            
            image[mask] = [0, 255, 0]
            output_filename = f"video_seg_sam2_{image_name}"
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
    
    images_dir = os.path.join(package_path, 'scripts', 'images_trajectory')
    output_dir = os.path.join(package_path, 'scripts', 'results_trajectory_sam2')
    
    if not os.path.exists(images_dir):
         print(f"❌ Image directory not found: {images_dir}")
         return
    
    segmenter = SegmentRobotSAM2()
    segmenter.process_video_sequence(images_dir, output_dir)

if __name__ == "__main__":
    main()
