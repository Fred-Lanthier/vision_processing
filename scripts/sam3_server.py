#!/usr/bin/env python3
"""
SAM3 Segmentation Server

Loads SAM3 model ONCE at startup and provides a ROS service.
Other nodes call this service instead of loading SAM3 themselves.

Usage:
    Terminal 1: rosrun vision_processing sam3_server.py
    Terminal 2: rosrun vision_processing condition_pcd_node.py
    
The condition node will connect to this server automatically.
"""
import rospy
import numpy as np
import torch
import sys
from sensor_msgs.msg import Image

# Custom service (you need to build the package first)
from vision_processing.srv import Sam3Segment, Sam3SegmentResponse

# SAM3 imports
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("=" * 60)
    print("❌ SAM3 not found!")
    print("   Activate your SAM3 environment first:")
    print("   $ source ~/venv_sam3/bin/activate")
    print("=" * 60)
    sys.exit(1)

from PIL import Image as PILImage


class Sam3Server:
    def __init__(self):
        rospy.init_node('sam3_server')
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("  SAM3 SERVER STARTING")
        rospy.loginfo("=" * 60)
        
        # Create service FIRST so it's advertised even if model loading is slow
        self.service = rospy.Service('/sam3/segment', Sam3Segment, self.handle_request)
        self.model_ready = False

        # Device et Dtype 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        rospy.loginfo(f"Device: {self.device} | Autocast Dtype: {self.dtype}")
        
        # Load model (slow part - only done once!)
        rospy.loginfo("Loading SAM3 model... (30-60 seconds)")
        t0 = rospy.Time.now()
        
        self.sam_model = build_sam3_image_model()
        if hasattr(self.sam_model, "to"):
            self.sam_model.to(device=self.device)
        if hasattr(self.sam_model, "eval"):
            self.sam_model.eval()
            
        self.sam_processor = Sam3Processor(self.sam_model, confidence_threshold=0.1)
        self.model_ready = True

        load_time = (rospy.Time.now() - t0).to_sec()
        rospy.loginfo(f"✅ SAM3 loaded in {load_time:.1f}s")
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("🚀 SAM3 SERVER READY")
        rospy.loginfo("   Service: /sam3/segment")
        rospy.loginfo("=" * 60)
    
    def imgmsg_to_numpy(self, msg):
        """ROS Image → numpy RGB"""
        if msg.encoding == "rgb8":
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        elif msg.encoding == "bgr8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            return img[:, :, ::-1]
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    
    def numpy_to_imgmsg(self, arr):
        """numpy → ROS Image (mono8)"""
        msg = Image()
        msg.height, msg.width = arr.shape[:2]
        msg.encoding = "mono8"
        msg.is_bigendian = False
        msg.step = arr.shape[1]
        msg.data = arr.astype(np.uint8).tobytes()
        return msg
    
    def handle_request(self, req):
        """Process segmentation request"""
        resp = Sam3SegmentResponse()
        resp.success = False
        resp.confidence = 0.0
        resp.bbox = []

        if not self.model_ready:
            resp.message = "SAM3 model is still loading, please wait..."
            return resp
        
        try:
            # Convert image
            rgb = self.imgmsg_to_numpy(req.rgb_image)
            pil_img = PILImage.fromarray(rgb)
            
            threshold = req.confidence_threshold if req.confidence_threshold > 0 else 0.1
            
            # On garde l'autocast dynamique pour la vitesse et la compatibilité
            with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=self.dtype):
                state = self.sam_processor.set_image(pil_img)
                output = self.sam_processor.set_text_prompt(state=state, prompt=req.text_prompt)
            
            # Get scores
            raw_scores = output.get("scores", [])
            if len(raw_scores) == 0:
                resp.message = f"No detection for '{req.text_prompt}'"
                return resp
            
            scores = raw_scores.detach().cpu().numpy().flatten() if isinstance(raw_scores, torch.Tensor) else np.array(raw_scores).flatten()
            best_idx = np.argmax(scores)
            best_score = float(scores[best_idx])

            if best_score < threshold:
                resp.message = f"Score {best_score:.3f} < threshold {threshold:.3f}"
                resp.confidence = best_score
                return resp

            # Get masks
            masks = output.get("masks", [])
            masks = masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else np.array(masks)

            # top_k=0 or 1 → single best mask (backward-compatible default)
            top_k = max(1, int(req.top_k)) if req.top_k > 1 else 1
            top_k = min(top_k, len(scores))

            if top_k == 1:
                # Original behaviour
                mask = masks[best_idx]
                while mask.ndim > 2:
                    mask = mask[0]
                combined_mask = mask > 0
            else:
                # Union of the top-K scoring masks that clear the threshold
                top_indices = np.argsort(scores)[::-1][:top_k]
                valid = [i for i in top_indices if scores[i] >= threshold]
                if not valid:
                    valid = [best_idx]
                combined_mask = np.zeros_like(
                    masks[0].squeeze() if masks[0].ndim > 2 else masks[0], dtype=bool
                )
                for i in valid:
                    m = masks[i]
                    while m.ndim > 2:
                        m = m[0]
                    combined_mask |= (m > 0)
                rospy.loginfo(
                    f"  top-{top_k} union: used {len(valid)} masks "
                    f"scores={scores[top_indices].round(3).tolist()}"
                )

            mask_uint8 = combined_mask.astype(np.uint8) * 255
            
            # Bounding box
            rows, cols = np.any(mask_uint8, axis=1), np.any(mask_uint8, axis=0)
            if np.any(rows) and np.any(cols):
                y_idx, x_idx = np.where(rows)[0], np.where(cols)[0]
                resp.bbox = [float(x_idx[0]), float(y_idx[0]), float(x_idx[-1]), float(y_idx[-1])]
            
            # Success
            resp.success = True
            resp.message = f"Detected '{req.text_prompt}'"
            resp.mask = self.numpy_to_imgmsg(mask_uint8)
            resp.confidence = best_score
            
            rospy.loginfo(f"✅ '{req.text_prompt}' conf={best_score:.3f}")
            
        except Exception as e:
            resp.message = str(e)
            rospy.logerr(f"Error: {e}")
        
        return resp
    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    Sam3Server().run()
