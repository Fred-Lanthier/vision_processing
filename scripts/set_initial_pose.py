#!/usr/bin/env python3
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

rospy.init_node('set_initial_pose')

# Attendre que le contrôleur soit prêt
rospy.sleep(3.0)

pub = rospy.Publisher('/panda_arm_controller/command', JointTrajectory, queue_size=1)
rospy.sleep(0.5)  # Laisser le publisher se connecter

# Créer la trajectoire
traj = JointTrajectory()
traj.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 
                    'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']

point = JointTrajectoryPoint()
point.positions = [-0.000059, -0.125928, 0.000117, -2.193312, -0.000251, 2.064780, 0.785511]
point.time_from_start = rospy.Duration(2.0)  # Prend 2 secondes pour y aller

traj.points.append(point)

rospy.loginfo("Envoi de la position initiale...")
pub.publish(traj)
rospy.loginfo("Position envoyée!")

rospy.sleep(3.0)  # Attendre que le mouvement se termine
