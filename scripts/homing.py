#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import time

class DirectHoming:
    def __init__(self):
        rospy.init_node('direct_homing_node')
        
        # 1. Configuration
        # Cible Homing (Tes valeurs jointes du script précédent)
        self.home_joints = np.array([
            -0.000059, -0.125928, 0.000117, -2.193312, 
            -0.000251, 2.064780, 0.785511
        ])
        
        self.current_joints = None
        
        # 2. Communication
        self.pub = rospy.Publisher(
            '/joint_group_position_controller/command', 
            Float64MultiArray, 
            queue_size=1
        )
        self.sub = rospy.Subscriber("/joint_states", JointState, self.joint_cb)
        
        rospy.loginfo("⏳ Attente des joint states...")

    def joint_cb(self, msg):
        # Récupération propre des positions (filtrage gripper)
        positions = []
        for name, pos in zip(msg.name, msg.position):
            if "panda_joint" in name:
                positions.append(pos)
        if len(positions) == 7:
            self.current_joints = np.array(positions)

    def move_to_home(self, duration=4.0):
        """ Interpole linéairement de la position actuelle vers Home en 'duration' secondes """
        while self.current_joints is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
            
        start_joints = self.current_joints.copy()
        start_time = rospy.Time.now().to_sec()
        
        rate = rospy.Rate(100) # 100 Hz pour être très fluide
        
        rospy.loginfo("🚀 Début du Homing...")
        
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            t = now - start_time
            
            # Ratio d'avancement (0.0 à 1.0)
            alpha = min(t / duration, 1.0)
            
            # Interpolation
            cmd_joints = (1 - alpha) * start_joints + alpha * self.home_joints
            
            # Envoi
            msg = Float64MultiArray()
            msg.data = cmd_joints.tolist()
            self.pub.publish(msg)
            
            if alpha >= 1.0:
                break
                
            rate.sleep()
            
        rospy.loginfo("✅ Homing Terminé.")

if __name__ == '__main__':
    try:
        homer = DirectHoming()
        homer.move_to_home(duration=3.0) # 3 secondes pour y aller
    except rospy.ROSInterruptException:
        pass