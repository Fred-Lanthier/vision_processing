#!/usr/bin/env python3
import rospy
from vision_processing.srv import ProcessImage, ProcessImageResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import os

# Add the scripts directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from TEST_SAM2 import SingleFoodDetector

class VisionServer:
    def __init__(self):
        rospy.init_node('vision_server')
        
        # Load models ONCE at startup
        rospy.loginfo("🚀 Loading models at startup...")
        model_path = os.path.join(script_dir, "sam_vit_l.pth")
        self.detector = SingleFoodDetector(model_type="vit_l", model_name=model_path)
        rospy.loginfo("✅ Models ready!")
        
        rospy.loginfo(f"Debug images will be saved to: {script_dir}")
        
        self.bridge = CvBridge()
        
        # Advertise the service
        self.service = rospy.Service('process_image', ProcessImage, self.handle_request)
        rospy.loginfo("Vision processing service ready - waiting for image requests...")
    
    def handle_request(self, req):
        response = ProcessImageResponse()
        
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(req.input_image, "bgr8")
            
            # Save temporarily in scripts directory
            temp_path = os.path.join(script_dir, "ros_image.jpg")
            cv2.imwrite(temp_path, cv_image)
            
            # Change working directory to scripts folder
            original_dir = os.getcwd()
            os.chdir(script_dir)
            
            # Run detection (images will be saved in script_dir)
            rospy.loginfo("Processing image...")
            result = self.detector.detect_individual_food_pieces(temp_path)
            
            # Change back to original directory
            os.chdir(original_dir)
            
            if result:
                response.success = True
                response.selected_food = result['selected_food']
                response.total_pieces = result['total_pieces']
                response.centroid = result['centroid']
                response.computation_time = result['computation_time']
                response.selected_piece_area = int(result['selected_piece']['area'])
                
                # Create visualization with mask overlay
                mask = result['selected_piece']['mask']
                overlay = cv_image.copy()
                overlay[mask > 0] = [0, 255, 0]
                blended = cv2.addWeighted(cv_image, 0.7, overlay, 0.3, 0)
                
                response.output_image = self.bridge.cv2_to_imgmsg(blended, "bgr8")
                
                rospy.loginfo(f"✅ Detected: {result['selected_food']}, "
                             f"Pieces: {result['total_pieces']}, "
                             f"Centroid: {result['centroid']}, "
                             f"Time: {result['computation_time']:.3f}s")
            else:
                response.success = False
                rospy.logwarn("❌ No detections found")
                
        except Exception as e:
            response.success = False
            rospy.logerr(f"❌ Detection failed: {str(e)}")
            os.chdir(original_dir)  # Make sure to change back
        
        return response
    
    def spin(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        server = VisionServer()
        server.spin()
    except rospy.ROSInterruptException:
        pass