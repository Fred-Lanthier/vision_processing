#!/usr/bin/env python3
"""
Lire l'état complet du robot Franka + caméra en temps réel
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped, TwistStamped
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

class RobotStateReader:
    def __init__(self):
        rospy.init_node('robot_state_reader')
        
        self.bridge = CvBridge()
        
        # Data storage
        self.joint_angles = None
        self.ee_position = None
        self.ee_orientation = None  # quaternion [x, y, z, w]
        self.ee_velocity = None
        self.rgb_image = None
        self.depth_image = None
        
        # Subscribers
        rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        rospy.Subscriber('/franka/end_effector_pose', PoseStamped, self.ee_pose_callback)
        rospy.Subscriber('/franka/end_effector_velocity', TwistStamped, self.ee_velocity_callback)
        rospy.Subscriber('/camera_wrist/color/image_raw', Image, self.rgb_callback)
        rospy.Subscriber('/camera_wrist/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        
        rospy.loginfo("Robot State Reader initialized!")
        rospy.loginfo("Subscribing to all robot state topics...")
        
    def joint_state_callback(self, msg):
        """Lire les angles des joints"""
        joint_angles = []
        for i, name in enumerate(msg.name):
            if 'panda_joint' in name:
                joint_angles.append(msg.position[i])
        
        if len(joint_angles) == 7:
            self.joint_angles = np.array(joint_angles)
    
    def ee_pose_callback(self, msg):
        """Lire la pose de l'effecteur"""
        self.ee_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        self.ee_orientation = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
    
    def ee_velocity_callback(self, msg):
        """Lire la vitesse de l'effecteur"""
        linear = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z
        ])
        
        angular = np.array([
            msg.twist.angular.x,
            msg.twist.angular.y,
            msg.twist.angular.z
        ])
        
        # Seuils pour considérer la vitesse comme nulle
        LINEAR_THRESHOLD = 0.001   # 1 mm/s
        ANGULAR_THRESHOLD = 0.01   # ~0.57 deg/s
        
        # Si vitesse trop faible, considérer comme nulle
        if np.linalg.norm(linear) < LINEAR_THRESHOLD:
            linear = np.zeros(3)
        
        if np.linalg.norm(angular) < ANGULAR_THRESHOLD:
            angular = np.zeros(3)
        
        self.ee_velocity = {
            'linear': linear,
            'angular': angular
        }
    
    def rgb_callback(self, msg):
        """Lire l'image RGB de la caméra"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.rgb_image = cv_image
        except Exception as e:
            rospy.logerr(f"Error converting RGB image: {e}")
    
    def depth_callback(self, msg):
        """Lire l'image de profondeur"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            self.depth_image = cv_image
        except Exception as e:
            rospy.logerr(f"Error converting depth image: {e}")
    
    def get_joint_angles_deg(self):
        """Obtenir les angles des joints en degrés"""
        if self.joint_angles is not None:
            return np.degrees(self.joint_angles)
        return None
    
    def get_ee_orientation_euler(self):
        """Obtenir l'orientation de l'effecteur en angles d'Euler (degrés)"""
        if self.ee_orientation is not None:
            r = R.from_quat(self.ee_orientation)
            roll, pitch, yaw = r.as_euler('xyz', degrees=True)
            return {'roll': roll, 'pitch': pitch, 'yaw': yaw}
        return None
    
    def print_state(self):
        """Afficher l'état complet du robot"""
        print("\n" + "="*60)
        print("ÉTAT DU ROBOT FRANKA")
        print("="*60)
        
        # Angles des joints
        if self.joint_angles is not None:
            print("\n📐 ANGLES DES JOINTS (degrés):")
            for i, angle in enumerate(self.get_joint_angles_deg()):
                print(f"  Joint {i+1}: {angle:7.2f}°")
        
        # Position de l'effecteur
        if self.ee_position is not None:
            print("\n📍 POSITION DE L'EFFECTEUR (m):")
            print(f"  x: {self.ee_position[0]:7.4f}")
            print(f"  y: {self.ee_position[1]:7.4f}")
            print(f"  z: {self.ee_position[2]:7.4f}")
        
        # Orientation de l'effecteur
        euler = self.get_ee_orientation_euler()
        if euler is not None:
            print("\n🔄 ORIENTATION DE L'EFFECTEUR (degrés):")
            print(f"  Roll:  {euler['roll']:7.2f}°")
            print(f"  Pitch: {euler['pitch']:7.2f}°")
            print(f"  Yaw:   {euler['yaw']:7.2f}°")
        
        # Vitesse de l'effecteur
        if self.ee_velocity is not None:
            lin_speed = np.linalg.norm(self.ee_velocity['linear'])
            ang_speed = np.linalg.norm(self.ee_velocity['angular'])
            
            print("\n⚡ VITESSE DE L'EFFECTEUR:")
            if lin_speed < 0.001 and ang_speed < 0.01:
                print("  Robot à l'ARRÊT")
            else:
                print(f"  Linéaire:  {lin_speed:7.4f} m/s")
                print(f"  Angulaire: {ang_speed:7.4f} rad/s")
        
        # Images
        print("\n📷 CAMÉRA:")
        if self.rgb_image is not None:
            print(f"  RGB:   {self.rgb_image.shape[1]}x{self.rgb_image.shape[0]} pixels")
        else:
            print("  RGB:   Pas d'image")
        
        if self.depth_image is not None:
            print(f"  Depth: {self.depth_image.shape[1]}x{self.depth_image.shape[0]} pixels")
        else:
            print("  Depth: Pas d'image")
    
    def show_camera_images(self):
        """Afficher les images de la caméra dans des fenêtres OpenCV"""
        if self.rgb_image is not None:
            cv2.imshow('Camera RGB', self.rgb_image)
        
        if self.depth_image is not None:
            # Normaliser depth pour affichage
            depth_display = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = depth_display.astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            cv2.imshow('Camera Depth', depth_colormap)
        
        cv2.waitKey(1)
    
    def save_current_images(self, prefix="capture"):
        """Sauvegarder les images actuelles"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.rgb_image is not None:
            filename = f"{prefix}_rgb_{timestamp}.png"
            cv2.imwrite(filename, self.rgb_image)
            rospy.loginfo(f"Saved RGB image: {filename}")
        
        if self.depth_image is not None:
            filename = f"{prefix}_depth_{timestamp}.png"
            cv2.imwrite(filename, self.depth_image)
            rospy.loginfo(f"Saved depth image: {filename}")

def main():
    reader = RobotStateReader()
    
    rospy.loginfo("Attente des données... (Ctrl+C pour quitter)")
    rospy.sleep(2)  # Attendre que les données arrivent
    
    rate = rospy.Rate(2)  # 2 Hz - affichage toutes les 0.5 secondes
    
    try:
        while not rospy.is_shutdown():
            # Afficher l'état du robot
            reader.print_state()
            
            # Afficher les images de la caméra
            reader.show_camera_images()
            
            # Sauvegarder les images avec 's'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                reader.save_current_images()
                rospy.loginfo("Images sauvegardées!")
            elif key == ord('q'):
                break
            
            rate.sleep()
    
    except KeyboardInterrupt:
        rospy.loginfo("Arrêt du lecteur...")
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()