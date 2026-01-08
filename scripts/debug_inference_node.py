#!/usr/bin/env python3
import rospy
import message_filters
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
import time

class DebugInferenceNode:
    def __init__(self):
        rospy.init_node('debug_inference_node', anonymous=True)
        
        self.bridge = CvBridge()
        self.last_time = time.time()
        self.frame_count = 0
        self.current_fps = 0.0  # <--- AJOUT : On stocke le FPS ici pour s'en souvenir

        # 1. Configuration des Subscribers
        sub_wrist_rgb = message_filters.Subscriber("/synced/camera_wrist/rgb", Image)
        sub_wrist_depth = message_filters.Subscriber("/synced/camera_wrist/depth", Image)
        sub_static_rgb = message_filters.Subscriber("/synced/camera_static/rgb", Image)
        sub_static_depth = message_filters.Subscriber("/synced/camera_static/depth", Image)
        sub_ee_pose = message_filters.Subscriber("/synced/ee_pose", PoseStamped)
        sub_joints = message_filters.Subscriber("/synced/joint_states", JointState)

        # 2. Synchronisation
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_wrist_rgb, sub_wrist_depth, sub_static_rgb, sub_static_depth, sub_ee_pose, sub_joints],
            queue_size=10, slop=0.1, allow_headerless=False
        )
        
        self.ts.registerCallback(self.callback)
        
        rospy.loginfo("Debug Node Initialized. Waiting for synced streams...")

    def callback(self, w_rgb, w_depth, s_rgb, s_depth, ee_pose, joints):
        # --- Calcul de fréquence (Hz) ---
        self.frame_count += 1
        now = time.time()
        dt = now - self.last_time
        
        # On ne met à jour le calcul que toutes les secondes
        if dt > 1.0:
            self.current_fps = self.frame_count / dt  # Mise à jour de la variable membre
            self.frame_count = 0
            self.last_time = now

        try:
            # --- Conversion Images ---
            cv_w_rgb = self.bridge.imgmsg_to_cv2(w_rgb, "bgr8")
            cv_w_depth = self.bridge.imgmsg_to_cv2(w_depth, "passthrough")
            
            # --- Extraction Données Robot ---
            pos = ee_pose.pose.position
            ori = ee_pose.pose.orientation
            
            panda_joints = []
            for i, name in enumerate(joints.name):
                if "panda_joint" in name:
                    panda_joints.append(joints.position[i])

            # --- AFFICHAGE TERMINAL ---
            # On utilise self.current_fps qui garde la dernière valeur calculée
            print("\n" + "="*50)
            print(f"✅ FRAME SYNCHRONISÉ REÇU (Rate: {self.current_fps:.1f} Hz)") 
            print(f"⏰ Timestamp Maître : {w_rgb.header.stamp.to_sec():.4f}")
            print("-" * 20)
            
            print(f"📷 Wrist RGB : {cv_w_rgb.shape} | Depth : {cv_w_depth.shape}")
            print(f"📍 EE Position : [x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}]")
            print(f"🔄 EE Quat     : [x={ori.x:.3f}, y={ori.y:.3f}, z={ori.z:.3f}, w={ori.w:.3f}]")
            
            # Vérification Delta Temporel
            diff_cam = abs((w_rgb.header.stamp - s_rgb.header.stamp).to_sec())
            diff_pose = abs((w_rgb.header.stamp - ee_pose.header.stamp).to_sec())
            
            if diff_cam < 0.0001 and diff_pose < 0.0001:
                print("✨ SYNCHRO PARFAITE (Delta ~ 0s)")
            else:
                print(f"⚠️ ATTENTION JITTER: CamDiff={diff_cam:.4f}s, PoseDiff={diff_pose:.4f}s")

        except CvBridgeError as e:
            rospy.logerr(f"Erreur CV Bridge: {e}")

if __name__ == '__main__':
    try:
        DebugInferenceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass