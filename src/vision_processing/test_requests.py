#!/usr/bin/env python3
import rospy
import rospkg
import cv2
import numpy as np
import os
import glob
import time
import sys

# Imports des services
from sensor_msgs.msg import Image
from vision_processing.srv import ProcessImage, ProcessImageRequest
from vision_processing.srv import ProcessRobot, ProcessRobotRequest

class GlobalTester:
    def __init__(self):
        rospy.init_node('global_vision_tester')
        
        # --- 1. Configuration des Chemins ---
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('vision_processing')
        
        # Chemins Food
        self.food_img_path = os.path.join(package_path, 'scripts', 'images', 'Filou.jpeg')
        
        # Chemins Robot (Séquence)
        self.robot_img_dir = os.path.join(package_path, 'scripts', 'images_trajectory')
        self.robot_depth_dir = os.path.join(package_path, 'scripts', 'Robot_depth_trajectory')
        
        rospy.loginfo("🛠️  Initialisation du Testeur Global...")

    def numpy_to_ros(self, cv_img, encoding):
        """Convertit numpy -> message ROS"""
        msg = Image()
        msg.height = cv_img.shape[0]
        msg.width = cv_img.shape[1]
        msg.encoding = encoding
        
        if encoding == "bgr8":
            msg.step = cv_img.shape[1] * 3
            msg.data = cv_img.tobytes()
        elif encoding == "16UC1": # Depth uint16 (mm)
            msg.step = cv_img.shape[1] * 2
            msg.data = cv_img.astype(np.uint16).tobytes()
        return msg

    # =========================================================================
    # TEST 1 : FOOD SEGMENTATION (SAM 3 + Gemini)
    # =========================================================================
    def run_food_test(self):
        print("\n" + "="*60)
        print("🍔 TEST 1 : SERVICE NOURRITURE (One-Shot)")
        print("="*60)
        
        service_name = '/process_food'
        
        if not os.path.exists(self.food_img_path):
            rospy.logerr(f"❌ Image Food introuvable : {self.food_img_path}")
            return

        # 1. Chargement
        print(f"📂 Chargement image : {os.path.basename(self.food_img_path)}")
        cv_img = cv2.imread(self.food_img_path)
        if cv_img is None: return

        # 2. Connexion Service
        print(f"⏳ En attente du service {service_name}...")
        try:
            rospy.wait_for_service(service_name, timeout=5)
        except rospy.ROSException:
            rospy.logerr("❌ Service Food non disponible !")
            return

        # 3. Envoi
        try:
            proxy = rospy.ServiceProxy(service_name, ProcessImage)
            req = ProcessImageRequest()
            req.input_image = self.numpy_to_ros(cv_img, "bgr8")
            
            print("📤 Envoi de la requête...")
            start_t = time.time()
            resp = proxy(req)
            duration = time.time() - start_t
            
            if resp.success:
                print(f"✅ SUCCÈS en {duration:.2f}s")
                print(f"   🍴 Aliment : {resp.selected_food}")
                print(f"   📊 Pièces : {resp.total_pieces}")
                print(f"   🎯 Centre : {resp.centroid}")
            else:
                print("❌ ECHEC : Le serveur a retourné success=False")
                
        except Exception as e:
            print(f"❌ Erreur critique : {e}")

    # =========================================================================
    # TEST 2 : ROBOT TRACKING (Séquence 50 frames)
    # =========================================================================
    def run_robot_sequence_test(self):
        print("\n" + "="*60)
        print("🤖 TEST 2 : SERVICE ROBOT (Séquence 50 Frames)")
        print("="*60)
        
        service_name = '/process_robot'
        
        # 1. Récupération des 50 premières images
        all_images = sorted(glob.glob(os.path.join(self.robot_img_dir, "*.jpg")))
        if len(all_images) < 50:
            rospy.logwarn(f"⚠️ Pas assez d'images pour le test complet ({len(all_images)} trouvées).")
            sequence = all_images
        else:
            sequence = all_images[:50]
            
        print(f"🎞️  Séquence chargée : {len(sequence)} frames.")

        # 2. Connexion Service
        print(f"⏳ En attente du service {service_name}...")
        try:
            rospy.wait_for_service(service_name, timeout=5)
        except rospy.ROSException:
            rospy.logerr("❌ Service Robot non disponible !")
            return

        proxy = rospy.ServiceProxy(service_name, ProcessRobot)
        latencies = []

        # 3. Boucle de test
        print("\n🚀 Démarrage de la séquence temps réel...")
        
        for i, img_path in enumerate(sequence):
            filename = os.path.basename(img_path)
            
            # Déduction du nom depth (On extrait le numéro)
            # Ex: "img_00120.jpg" -> "120" -> "static_depth_step_0120.npy"
            digits = "".join(filter(str.isdigit, filename))
            # On prend les 4 derniers chiffres pour matcher le format habituel
            if len(digits) >= 4:
                depth_num = digits[-4:]
            else:
                depth_num = digits.zfill(4)
                
            depth_name = f"static_depth_step_{depth_num}.npy"
            depth_path = os.path.join(self.robot_depth_dir, depth_name)
            
            if not os.path.exists(depth_path):
                print(f"⚠️ Frame {i}: Depth manquante ({depth_name}) -> Skip")
                continue

            # Load Data
            cv_rgb = cv2.imread(img_path)
            try:
                np_depth = np.load(depth_path)
            except:
                print(f"❌ Erreur lecture NPY: {depth_name}")
                continue
            
            # Prepare Request
            req = ProcessRobotRequest()
            req.rgb_image = self.numpy_to_ros(cv_rgb, "bgr8")
            req.depth_image = self.numpy_to_ros(np_depth, "16UC1")
            
            # Send
            print(f"   Frame {i} ({filename}) -> ", end="", flush=True)
            start_t = time.time()
            
            try:
                resp = proxy(req)
                end_t = time.time()
                duration = end_t - start_t
                latencies.append(duration)
                
                if resp.success:
                    pcd_info = "PCD OK" if resp.pcd_path else "NO PCD"
                    print(f"✅ {duration:.3f}s | {pcd_info}")
                else:
                    print(f"❌ ECHEC (Logic)")
            except Exception as e:
                print(f"❌ CRASH: {e}")

        # 4. Statistiques
        if latencies:
            print("\n📊 RÉSULTATS PERFORMANCE :")
            print(f"   Première Frame (Init + SAM 3) : {latencies[0]:.3f}s")
            if len(latencies) > 1:
                avg_tracking = sum(latencies[1:]) / (len(latencies)-1)
                fps = 1.0 / avg_tracking if avg_tracking > 0 else 0
                print(f"   Frames Suivantes (Tracking)   : {avg_tracking:.3f}s  (approx {fps:.1f} FPS)")
            print("="*60)

if __name__ == "__main__":
    try:
        tester = GlobalTester()
        
        # Lancer les deux tests à la suite
        tester.run_food_test()
        time.sleep(1) # Petite pause pour la lisibilité
        tester.run_robot_sequence_test()
        
        print("\n🏁 Tous les tests sont terminés.")
        
    except rospy.ROSInterruptException:
        pass