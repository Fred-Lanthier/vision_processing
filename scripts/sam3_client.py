#!/usr/bin/env python3
"""
SAM3 Client Helper

Simple wrapper to call the SAM3 server service.
Import this in your condition nodes instead of loading SAM3 directly.

Usage:
    from sam3_client import Sam3Client
    
    client = Sam3Client()
    mask, score = client.segment(rgb_msg, "cube")
"""
import rospy
import numpy as np
from sensor_msgs.msg import Image
from vision_processing.srv import Sam3Segment, Sam3SegmentRequest


class Sam3Client:
    """
    Client for the SAM3 segmentation server.
    
    Call segment() with an RGB image and text prompt to get a segmentation mask.
    """
    
    def __init__(self, service_name='/sam3/segment', timeout=30.0):
        """
        Initialize the client.
        
        Args:
            service_name: ROS service name (default: /sam3/segment)
            timeout: How long to wait for server (seconds)
        """
        self.service_name = service_name
        
        rospy.loginfo(f"⏳ Waiting for SAM3 server ({service_name})...")
        
        try:
            rospy.wait_for_service(service_name, timeout=timeout)
            self.client = rospy.ServiceProxy(service_name, Sam3Segment)
            rospy.loginfo("✅ Connected to SAM3 server!")
        except rospy.ROSException:
            rospy.logerr("=" * 60)
            rospy.logerr("❌ SAM3 server not found!")
            rospy.logerr("   Start the server first:")
            rospy.logerr("   $ rosrun vision_processing sam3_server.py")
            rospy.logerr("=" * 60)
            raise RuntimeError("SAM3 server not available")
    
    def segment(self, rgb_image_msg, text_prompt, confidence_threshold=0.1):
        """
        Segment an object in the image using text prompt.
        
        Args:
            rgb_image_msg: sensor_msgs/Image (RGB)
            text_prompt: Object to detect (e.g., "cube", "food", "plate")
            confidence_threshold: Minimum confidence to accept detection
            
        Returns:
            tuple: (mask, confidence) where:
                - mask: numpy array (H, W) with values 0 or 255, or None if failed
                - confidence: float score, or 0.0 if failed
        """
        try:
            req = Sam3SegmentRequest()
            req.rgb_image = rgb_image_msg
            req.text_prompt = text_prompt
            req.confidence_threshold = confidence_threshold
            
            resp = self.client(req)
            
            if resp.success:
                # Convert mask to numpy
                mask = np.frombuffer(resp.mask.data, dtype=np.uint8)
                mask = mask.reshape(resp.mask.height, resp.mask.width)
                return mask, resp.confidence
            else:
                rospy.logdebug(f"SAM3: {resp.message}")
                return None, resp.confidence
                
        except rospy.ServiceException as e:
            rospy.logerr(f"SAM3 service call failed: {e}")
            return None, 0.0
    
    def segment_with_bbox(self, rgb_image_msg, text_prompt, confidence_threshold=0.1):
        """
        Same as segment() but also returns bounding box.
        
        Returns:
            tuple: (mask, confidence, bbox) where:
                - mask: numpy array (H, W) or None
                - confidence: float
                - bbox: [x_min, y_min, x_max, y_max] or []
        """
        try:
            req = Sam3SegmentRequest()
            req.rgb_image = rgb_image_msg
            req.text_prompt = text_prompt
            req.confidence_threshold = confidence_threshold
            
            resp = self.client(req)
            
            if resp.success:
                mask = np.frombuffer(resp.mask.data, dtype=np.uint8)
                mask = mask.reshape(resp.mask.height, resp.mask.width)
                return mask, resp.confidence, list(resp.bbox)
            else:
                return None, resp.confidence, []
                
        except rospy.ServiceException as e:
            rospy.logerr(f"SAM3 service call failed: {e}")
            return None, 0.0, []
