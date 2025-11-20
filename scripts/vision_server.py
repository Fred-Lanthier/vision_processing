#!/home/flanthier/Github/src/vision_processing/venv_sam3/bin/python3
import rospy
# Assurez-vous que le package vision_processing est sourcé dans votre terminal
from vision_processing.srv import ProcessImage, ProcessImageResponse
from sensor_msgs.msg import Image
import cv2
import numpy as np
import sys
import os

# --- CONFIGURATION DU PATH ---
# Ajoute le dossier courant au path pour trouver SAM3_Gemini.py
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import de la nouvelle classe
from SAM3_Gemini import FoodSegmenterNative

# --- VOTRE CLÉ API GEMINI ---
# Idéalement, mettez ça dans un fichier de config ou variable d'env, mais pour l'instant :
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class VisionServer:
    def __init__(self):
        rospy.init_node('vision_server')
        
        rospy.loginfo("🚀 Démarrage du serveur Vision SAM 3 + Gemini...")
        
        # 1. Initialisation du modèle (Charge sur GPU)
        # On passe la clé API à l'initialisation
        if "AIza" not in GOOGLE_API_KEY:
            rospy.logerr("❌ CLÉ API GOOGLE MANQUANTE ! Editez vision_server.py")
            sys.exit(1)
            
        self.segmenter = FoodSegmenterNative(google_api_key=GOOGLE_API_KEY)
        
        rospy.loginfo("✅ Modèles SAM 3 et Client Gemini prêts !")
        
        # 2. Création du Service ROS
        self.service = rospy.Service('process_image', ProcessImage, self.handle_request)
        rospy.loginfo("📡 Service 'process_image' en attente de requêtes...")
    
    def handle_request(self, req):
        response = ProcessImageResponse()
        
        # Dossier temporaire pour l'échange (nécessaire pour votre logique actuelle)
        temp_img_path = os.path.join(script_dir, "ros_image.jpg")
        
        try:
            # --- A. Réception et Conversion de l'image ---
            height = req.input_image.height
            width = req.input_image.width
            encoding = req.input_image.encoding
            
            # Conversion manuelle Bytes -> Numpy (Evite les bugs cv_bridge/python3.10)
            dtype = np.uint8
            if encoding == "bgr8":
                cv_image = np.frombuffer(req.input_image.data, dtype=dtype).reshape(height, width, 3)
            elif encoding == "rgb8":
                cv_image = np.frombuffer(req.input_image.data, dtype=dtype).reshape(height, width, 3)
                # Convertir en BGR car OpenCV (et notre code SAM3) travaille souvent en BGR/RGB
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif encoding == "mono8":
                 cv_image = np.frombuffer(req.input_image.data, dtype=dtype).reshape(height, width)
                 cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            else:
                rospy.logerr(f"Encoding non supporté : {encoding}")
                response.success = False
                return response
            
            # Sauvegarde temporaire (pour que SAM3_Gemini puisse la lire avec PIL)
            cv2.imwrite(temp_img_path, cv_image)
            
            # --- B. Traitement (Inférence) ---
            rospy.loginfo("🖼️  Image reçue. Lancement de l'analyse...")
            
            # Appel à la classe FoodSegmenterNative
            result = self.segmenter.process_image(temp_img_path)
            
            # --- C. Remplissage de la Réponse ---
            if result["success"]:
                response.success = True
                response.selected_food = result["selected_food"]
                response.total_pieces = result["total_pieces"]
                response.centroid = result["centroid"] # [x, y]
                response.computation_time = result["computation_time"]
                response.selected_piece_area = result["selected_piece_area"]
                
                # --- D. Conversion de l'image de sortie (Numpy -> ROS Msg) ---
                processed_cv_img = result["processed_image"]
                if processed_cv_img is not None:
                    response.output_image.height = processed_cv_img.shape[0]
                    response.output_image.width = processed_cv_img.shape[1]
                    response.output_image.encoding = "bgr8"
                    response.output_image.step = processed_cv_img.shape[1] * 3
                    response.output_image.data = processed_cv_img.tobytes()
                
                rospy.loginfo(f"✅ Succès: '{result['selected_food']}' (Area: {result['selected_piece_area']}) en {result['computation_time']:.2f}s")
            else:
                response.success = False
                rospy.logwarn("⚠️ Analyse terminée mais aucun aliment détecté.")

        except Exception as e:
            response.success = False
            rospy.logerr(f"❌ Erreur critique serveur : {str(e)}")
            import traceback
            traceback.print_exc()
        
        return response
    
    def spin(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        server = VisionServer()
        server.spin()
    except rospy.ROSInterruptException:
        pass