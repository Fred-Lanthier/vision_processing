#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

class DummyRobotNode:
    def __init__(self):
        rospy.init_node('dummy_robot_node')
        self.joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=1)
        rospy.Subscriber('/franka_control/safe_joint_velocities', Float32MultiArray, self.cmd_callback)
        
        # État initial (Pose "ready" du Panda)
        self.q = np.array([-0.000059, -0.125928, 0.000117, -2.193312, -0.000251, 2.064780, 0.785511])
        self.dq_cmd = np.zeros(7)
        self.last_time = rospy.get_time()
        
        # Noms des joints de la structure URDF
        self.joint_names = [f'panda_joint{i}' for i in range(1, 8)]
        self.joint_names.extend(['panda_finger_joint1', 'panda_finger_joint2'])
        
        self.rate = rospy.Rate(100) # 100 Hz simulation
    
    def cmd_callback(self, msg):
        self.dq_cmd = np.array(msg.data)
        
    def run(self):
        print("🤖 Dummy Robot démarré. À l'écoute des vitesses sécurisées...")
        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            if dt > 0:
                # Intégration d'Euler de la commande de vitesse
                self.q += self.dq_cmd * dt
            
            # Publish
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = self.joint_names
            # On simule aussi les doigts (immobiles) pour que robot_state_publisher soit content
            msg.position = list(self.q) + [0.0, 0.0]
            msg.velocity = list(self.dq_cmd) + [0.0, 0.0]
            
            self.joint_pub.publish(msg)
            self.rate.sleep()

if __name__ == '__main__':
    node = DummyRobotNode()
    node.run()