#!/usr/bin/env python3
import rospy
from vision_processing.srv import ProcessImage, ProcessImageResponse
from sensor_msgs.msg import Image
import cv2
import numpy as np
import sys
import os

# --- CONFIGURATION DU PATH ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import de la classe Food (Assurez-vous que le nom du fichier .py est bon)
# Ex: Si le fichier s'appelle FoodSegmenterIndividual.py :
from vision_processing.src.vision_segmentation.Single_Segmentation_Gemini_SAM3 import FoodSegmenterIndividual

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class VisionServerFood:
    def __init__(self):
        rospy.init_node('vision_server_food') # Nom unique
        rospy.loginfo("🚀 Démarrage Serveur Food (SAM 3 + Gemini)...")
        
        if not GOOGLE_API_KEY:
            rospy.logerr("❌ GOOGLE_API_KEY manquante !")
            sys.exit(1)
            
        self.segmenter = FoodSegmenterIndividual(google_api_key=GOOGLE_API_KEY)
        
        # Service spécifique à la nourriture
        self.service = rospy.Service('process_food', ProcessImage, self.handle_request)
        rospy.loginfo("✅ Serveur Food prêt.")
    
    def handle_request(self, req):
        response = ProcessImageResponse()
        
        # ... (Réception et conversion image identiques à votre code) ...
        # (Je condense pour la lisibilité)
        height = req.input_image.height
        width = req.input_image.width
        img_data = np.frombuffer(req.input_image.data, dtype=np.uint8)
        
        if req.input_image.encoding == "bgr8":
            cv_image = img_data.reshape(height, width, 3)
        elif req.input_image.encoding == "rgb8":
            cv_image = cv2.cvtColor(img_data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
        else:
            response.success = False
            return response
            
        temp_path = os.path.join(script_dir, "temp_food.jpg")
        cv2.imwrite(temp_path, cv_image)
        
        try:
            # TRAITEMENT
            result = self.segmenter.process_image(temp_path)
            
            if result["success"]:
                response.success = True
                response.selected_food = result["selected_food"]
                response.total_pieces = result["total_pieces"]
                response.centroid = result["centroid"]
                response.selected_piece_area = result["selected_piece_area"]
                response.computation_time = result["computation_time"]
                
                # --- CORRECTION ICI ---
                # On récupère l'image overlay (Masque transparent)
                out_img = result["selected_overlay_image"]
                
                if out_img is not None:
                    response.output_image.height = out_img.shape[0]
                    response.output_image.width = out_img.shape[1]
                    response.output_image.encoding = "bgr8"
                    response.output_image.step = out_img.shape[1] * 3
                    response.output_image.data = out_img.tobytes()
            else:
                response.success = False
                
        except Exception as e:
            rospy.logerr(f"Erreur: {e}")
            response.success = False
            
        return response

    def spin(self):
        rospy.spin()

if __name__ == "__main__":
    server = VisionServerFood()
    server.spin()
