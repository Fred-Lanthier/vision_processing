#!/home/flanthier/Documents/src/vision_processing/venv_py310/bin/python3
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
        sam2_checkpoint = os.path.expanduser(f"~/segment-anything-2/checkpoints/sam2.1_hiera_base_plus.pt")
        sam2_config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        self.detector = SingleFoodDetector(sam2_checkpoint, sam2_config, output_dir=script_dir)
        rospy.loginfo("✅ Models ready!")
        
        rospy.loginfo(f"Debug images will be saved to: {script_dir}")
        
        self.bridge = CvBridge()
        
        # Advertise the service
        self.service = rospy.Service('process_image', ProcessImage, self.handle_request)
        rospy.loginfo("Vision processing service ready - waiting for image requests...")
    
    def handle_request(self, req):
        response = ProcessImageResponse()
        original_dir = os.getcwd()
        
        try:
            # Extract raw image data from ROS message
            height = req.input_image.height
            width = req.input_image.width
            encoding = req.input_image.encoding
            
            # Convert raw bytes to numpy array (no cv_bridge needed)
            if encoding == "bgr8":
                cv_image = np.frombuffer(req.input_image.data, dtype=np.uint8).reshape(height, width, 3)
            elif encoding == "rgb8":
                cv_image = np.frombuffer(req.input_image.data, dtype=np.uint8).reshape(height, width, 3)
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            else:
                rospy.logerr(f"Unsupported encoding: {encoding}")
                response.success = False
                return response
            
            # Save temporarily in scripts directory
            temp_path = os.path.join(script_dir, "ros_image.jpg")
            cv2.imwrite(temp_path, cv_image)
            
            # Change working directory to scripts folder
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
                
                # Convert output image back to ROS message manually
                response.output_image.height = blended.shape[0]
                response.output_image.width = blended.shape[1]
                response.output_image.encoding = "bgr8"
                response.output_image.step = blended.shape[1] * 3
                response.output_image.data = blended.tobytes()
                
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
            os.chdir(original_dir)
        
        return response
    
    def spin(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        server = VisionServer()
        server.spin()
    except rospy.ROSInterruptException:
        pass