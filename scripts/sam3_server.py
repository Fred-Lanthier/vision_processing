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
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        rospy.loginfo(f"Device: {self.device}")
        
        # Load model (slow part - only done once!)
        rospy.loginfo("Loading SAM3 model... (30-60 seconds)")
        t0 = rospy.Time.now()
        
        self.sam_model = build_sam3_image_model()
        if hasattr(self.sam_model, "to"):
            self.sam_model.to(self.device)
        self.sam_processor = Sam3Processor(self.sam_model, confidence_threshold=0.1)
        
        load_time = (rospy.Time.now() - t0).to_sec()
        rospy.loginfo(f"✅ SAM3 loaded in {load_time:.1f}s")
        
        # Create service
        self.service = rospy.Service('/sam3/segment', Sam3Segment, self.handle_request)
        
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
        
        try:
            # Convert image
            rgb = self.imgmsg_to_numpy(req.rgb_image)
            pil_img = PILImage.fromarray(rgb)
            
            threshold = req.confidence_threshold if req.confidence_threshold > 0 else 0.1
            
            # Run SAM3
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
            
            # Get mask
            masks = output.get("masks", [])
            masks = masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else np.array(masks)
            mask = masks[best_idx]
            while mask.ndim > 2:
                mask = mask[0]
            
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            
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
