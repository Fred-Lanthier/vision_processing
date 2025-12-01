#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import sys
import os
import time

# --- SETUP PATH ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# --- ROS IMPORTS ---
# Assurez-vous d'avoir fait catkin_make après avoir créé le fichier .srv
from vision_processing.srv import ProcessRobot, ProcessRobotResponse
from sensor_msgs.msg import Image

from vision_processing.src.vision_tracking.Real_time_Robot_segmentation import SegmentRobotSAM2

class VisionServerRobot:
    def __init__(self):
        rospy.init_node('vision_server_robot')
        rospy.loginfo("🚀 Starting Robot Segmentation Server (SAM 3 + SAM 2)...")
        
        # 1. Initialize the Hybrid Model
        self.segmenter = SegmentRobotSAM2(temp_work_dir=os.path.join(script_dir, "temp_stream_buffer"))
        
        rospy.loginfo("✅ Model Loaded. Waiting for requests.")
        
        # 2. Create Service
        self.service = rospy.Service('process_robot', ProcessRobot, self.handle_request)
    
    def handle_request(self, req):
        response = ProcessRobotResponse()
        start_t = time.time()
        
        try:
            # ==========================================
            # A. DECODE RGB IMAGE (req.rgb_image)
            # ==========================================
            h_rgb = req.rgb_image.height
            w_rgb = req.rgb_image.width
            encoding_rgb = req.rgb_image.encoding
            
            img_buffer = np.frombuffer(req.rgb_image.data, dtype=np.uint8)
            
            if encoding_rgb == "bgr8":
                cv_rgb = img_buffer.reshape(h_rgb, w_rgb, 3)
            elif encoding_rgb == "rgb8":
                cv_rgb = img_buffer.reshape(h_rgb, w_rgb, 3)
                cv_rgb = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)
            elif encoding_rgb == "mono8":
                gray = img_buffer.reshape(h_rgb, w_rgb)
                cv_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                rospy.logerr(f"Unsupported RGB encoding: {encoding_rgb}")
                response.success = False
                return response

            # ==========================================
            # B. DECODE DEPTH IMAGE (req.depth_image)
            # ==========================================
            # Votre classe a besoin d'un array numpy en mm
            depth_map = None
            
            if req.depth_image.data: # Si on a reçu une image depth
                h_d = req.depth_image.height
                w_d = req.depth_image.width
                enc_d = req.depth_image.encoding
                
                # Cas 1 : 16UC1 (Standard RealSense uint16 en mm)
                if enc_d == "16UC1":
                    depth_buffer = np.frombuffer(req.depth_image.data, dtype=np.uint16)
                    depth_map = depth_buffer.reshape(h_d, w_d).astype(np.float32)
                    # Déjà en mm, pas de conversion nécessaire
                
                # Cas 2 : 32FC1 (Standard ROS float32 en mètres)
                elif enc_d == "32FC1":
                    depth_buffer = np.frombuffer(req.depth_image.data, dtype=np.float32)
                    depth_map = depth_buffer.reshape(h_d, w_d)
                    depth_map = depth_map * 1000.0 # Conversion m -> mm
                
                else:
                    rospy.logwarn(f"Depth encoding {enc_d} not optimized, skipping 3D generation.")

            # ==========================================
            # C. PROCESS FRAME
            # ==========================================
            rospy.loginfo(f"🖼️ Processing Robot Frame #{self.segmenter.frame_count}...")
            
            # Appel à votre classe
            result = self.segmenter.process_new_frame(cv_rgb, depth_array=depth_map)
            
            # ==========================================
            # D. BUILD RESPONSE
            # ==========================================
            if result and result["success"]:
                response.success = True
                
                # 1. Centroid
                cx, cy = result["centroid"]
                response.centroid = [float(cx), float(cy)]
                
                # 2. PCD Path (String)
                if result["pcd_path"]:
                    response.pcd_path = str(result["pcd_path"])
                    rospy.loginfo(f"   ☁️ PCD Saved: {result['pcd_path']}")
                else:
                    response.pcd_path = ""
                
                # 3. Debug Image (sensor_msgs/Image)
                mask = result["mask"]
                
                # Création visualisation (Vert pour le robot)
                overlay = cv_rgb.copy()
                if mask is not None:
                    overlay[mask] = [0, 255, 0]
                vis_img = cv2.addWeighted(cv_rgb, 0.7, overlay, 0.3, 0)
                
                # Encodage manuel vers ROS
                response.debug_image.height = vis_img.shape[0]
                response.debug_image.width = vis_img.shape[1]
                response.debug_image.encoding = "bgr8"
                response.debug_image.step = vis_img.shape[1] * 3
                response.debug_image.data = vis_img.tobytes()
                
            else:
                response.success = False
                rospy.logwarn("⚠️ Robot tracking failed.")

        except Exception as e:
            response.success = False
            rospy.logerr(f"❌ Server Error: {e}")
            import traceback
            traceback.print_exc()

        return response

    def spin(self):
        rospy.spin()

if __name__ == "__main__":
    server = VisionServerRobot()
    server.spin()