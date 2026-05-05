#!/usr/bin/env python3
import rospy
import numpy as np
import math
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

class DirectHoming:
    def __init__(self):
        rospy.init_node('direct_homing_node')
        
        # Cible Homing
        self.home_joints = np.array([
            -0.000059, -0.125928, 0.000117, -2.193312, 
            -0.000251, 2.064780, 0.785511
        ])
        
        self.current_joints = None
        
        self.pub = rospy.Publisher(
            '/joint_group_position_controller/command', 
            Float64MultiArray, 
            queue_size=1
        )
        self.sub = rospy.Subscriber("/joint_states", JointState, self.joint_cb)
        
        rospy.loginfo("⏳ Attente des joint states...")

    def joint_cb(self, msg):
        positions = []
        for name, pos in zip(msg.name, msg.position):
            if "panda_joint" in name:
                positions.append(pos)
        if len(positions) == 7:
            self.current_joints = np.array(positions)

    def move_to_home(self, duration=4.0):
        while self.current_joints is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
            
        start_joints = self.current_joints.copy()
        start_time = rospy.Time.now().to_sec()
        rate = rospy.Rate(100)
        
        rospy.loginfo("🚀 Début du Homing...")
        
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            t = now - start_time
            
            alpha = min(t / duration, 1.0)
            
            # Adoucissement de la trajectoire
            smooth_alpha = (1.0 - math.cos(math.pi * alpha)) / 2.0
            cmd_joints = (1 - smooth_alpha) * start_joints + smooth_alpha * self.home_joints
            
            msg = Float64MultiArray()
            msg.data = cmd_joints.tolist()
            self.pub.publish(msg)
            
            # Si on a atteint la fin (alpha = 1.0)
            if alpha >= 1.0:
                # On s'assure d'envoyer la position EXACTE finale une dernière fois
                msg.data = self.home_joints.tolist()
                self.pub.publish(msg)
                break # On sort de la boucle
                
            rate.sleep()
            
        rospy.loginfo("✅ Homing Terminé ! Fermeture automatique du script.")
        # On donne une petite demi-seconde à ROS pour être sûr que 
        # le tout dernier message est bien parti avant de tuer le noeud.
        rospy.sleep(0.5) 

if __name__ == '__main__':
    try:
        homer = DirectHoming()
        homer.move_to_home(duration=4.0)
        # Le script se termine naturellement ici, le robot maintient sa position.
    except rospy.ROSInterruptException:
        pass