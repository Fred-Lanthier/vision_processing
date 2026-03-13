#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
import sys
import os
import rospkg

# Import de ta classe SAM 3 existante
from sam3_client import Sam3Client

class Sam3DetectorNode:
    def __init__(self):
        rospy.init_node('sam3_detector_node', anonymous=True)
        
        self.target_object = rospy.get_param("~target_object", "cube")
        self.detection_confidence = 0.10
        
        rospy.loginfo("⏳ Chargement du modèle SAM 3...")
        self.sam3 = Sam3Client()
        rospy.loginfo("✅ SAM 3 Prêt. En attente de requêtes...")

        # Flag pour éviter d'accumuler les requêtes si SAM 3 est déjà en train de calculer
        self.is_processing = False

        self.sub_req = rospy.Subscriber('/vision/sam3_request', Image, self.request_cb, queue_size=1)
        self.pub_reply = rospy.Publisher('/vision/sam3_reply', Image, queue_size=1)

    def request_cb(self, msg):
        if self.is_processing:
            return 
            
        self.is_processing = True

        try:
            mask, score = self.sam3.segment(msg, self.target_object, self.detection_confidence)
            
            reply_msg = Image()
            reply_msg.header = msg.header
            reply_msg.height = msg.height
            reply_msg.width = msg.width
            reply_msg.encoding = "mono8"
            reply_msg.step = msg.width

            if mask is not None and score > self.detection_confidence:
                rospy.loginfo(f"🎯 Objet trouvé (Score: {score:.2f}) ! Envoi du masque au Tracker.")
                mask_np = (mask > 0).astype(np.uint8) * 255
                reply_msg.data = mask_np.tobytes()
            else:
                rospy.logwarn("❌ SAM 3 n'a rien trouvé. Renvoi d'un masque vide pour continuer la boucle.")
                # Création d'un masque tout noir (vide)
                empty_mask = np.zeros((msg.height, msg.width), dtype=np.uint8)
                reply_msg.data = empty_mask.tobytes()
                
            self.pub_reply.publish(reply_msg)
                
        except Exception as e:
            rospy.logerr(f"Erreur SAM 3 : {e}")
            
        self.is_processing = False

if __name__ == '__main__':
    Sam3DetectorNode()
    rospy.spin()