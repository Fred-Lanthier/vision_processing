#!/usr/bin/env python3
"""
Faire un cercle avec l'effecteur du Franka Panda dans Gazebo
1. Va d'abord à une position homing en contrôle de joints
2. Fait ensuite un cercle en cartésien avec orientation fixe
"""

import rospy
import rospkg
import moveit_commander
import sys
from geometry_msgs.msg import Pose
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import os

def create_pose_with_orientation(x, y, z, roll_deg, pitch_deg, yaw_deg):
    """Créer une pose avec position et orientation en angles Euler"""
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    
    # Créer rotation depuis Euler
    r = R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True)
    quat = r.as_quat()  # [qx, qy, qz, qw]
    
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    
    return pose

def create_pose_with_quaternion(xyz, quat):
    """Créer une pose avec position et orientation en quaternions"""
    pose = Pose()
    pose.position.x = xyz[0]
    pose.position.y = xyz[1]
    pose.position.z = xyz[2]
    
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    
    return pose
def move_through_waypoints(move_group, waypoints, speed=0.1):
    """Exécute une trajectoire cartésienne"""
    rospy.loginfo(f"Planification à travers {len(waypoints)} waypoints...")
    
    move_group.set_max_velocity_scaling_factor(speed)
    move_group.set_max_acceleration_scaling_factor(speed)
    
    # Planifier la trajectoire cartésienne
    (plan, fraction) = move_group.compute_cartesian_path(
        waypoints,
        0.01,  # Résolution de 1cm
        True    # Pas de détection de sauts
    )
    
    rospy.loginfo(f"Trajectoire planifiée: {fraction * 100:.1f}%")
    
    if fraction > 0.8:
        # Réajuster le timing
        plan = move_group.retime_trajectory(
            move_group.get_current_state(),
            plan,
            velocity_scaling_factor=speed,
            acceleration_scaling_factor=speed
        )
        
        rospy.loginfo("Exécution de la trajectoire...")
        success = move_group.execute(plan, wait=True)
        move_group.stop()
        
        if success:
            rospy.loginfo("✓ Trajectoire complétée!")
        return success
    else:
        rospy.logerr(f"Seulement {fraction * 100:.1f}% planifié. Abandon.")
        return False

def main():
    # Initialisation
    rospy.init_node('circle_trajectory')
    moveit_commander.roscpp_initialize(sys.argv)
    
    robot = moveit_commander.RobotCommander()
    move_group = moveit_commander.MoveGroupCommander("panda_arm")
    move_group.set_end_effector_link("panda_hand_tcp")
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("Trajectoire circulaire - Franka Panda")
    rospy.loginfo("=" * 60)
    rospy.loginfo(f"Planning frame: {move_group.get_planning_frame()}")
    rospy.loginfo(f"End effector: {move_group.get_end_effector_link()}")
    
    # Attendre que MoveIt soit prêt
    rospy.sleep(2)
    
    # ============================================
    # ÉTAPE 1: ALLER À LA POSITION HOMING (joints)
    # ============================================
    rospy.loginfo("\n--- ÉTAPE 1: Position homing (contrôle de joints) ---")
    
    # Position homing en radians
    joint_goal = [
        -0.000059,   # panda_joint1
        -0.125928,   # panda_joint2
        0.000117,    # panda_joint3
        -2.193312,   # panda_joint4
        -0.000251,   # panda_joint5
        2.064780,    # panda_joint6
        0.785511     # panda_joint7
    ]
    
    rospy.loginfo("Mouvement vers position homing...")
    move_group.go(joint_goal, wait=True)
    move_group.stop()
    
    rospy.loginfo("✓ Position homing atteinte!")
    rospy.sleep(1)
    
    # ============================================
    # ÉTAPE 2: OBTENIR LA POSE ACTUELLE
    # ============================================
    current_pose = move_group.get_current_pose().pose
    
    # Afficher orientation actuelle
    current_quat = [current_pose.orientation.x, current_pose.orientation.y,
                    current_pose.orientation.z, current_pose.orientation.w]
    r_current = R.from_quat(current_quat)
    current_roll, current_pitch, current_yaw = r_current.as_euler('xyz', degrees=True)
    
    rospy.loginfo(f"\n--- ÉTAPE 2: Pose actuelle ---")
    rospy.loginfo(f"Position: x={current_pose.position.x:.3f}, "
                  f"y={current_pose.position.y:.3f}, z={current_pose.position.z:.3f}")
    rospy.loginfo(f"Orientation: roll={current_roll:.1f}°, "
                  f"pitch={current_pitch:.1f}°, yaw={current_yaw:.1f}°")
    
    # ============================================
    # ÉTAPE 3: GÉNÉRER LE CERCLE
    # ============================================
    rospy.loginfo("\n--- ÉTAPE 3: Génération du cercle ---")
    
    # Paramètres du cercle
    center_x = current_pose.position.x
    center_y = current_pose.position.y
    center_z = current_pose.position.z
    radius = 0.2  # 5 cm de rayon
    n_points = 200  # 16 points
    
    # Orientation fixe: effecteur vers le bas
    # Pour Gazebo, roll=-180° fait pointer l'effecteur vers le bas
    target_roll = -180
    target_pitch = 0
    target_yaw = 0
    
    rospy.loginfo(f"Cercle: centre=({center_x:.3f}, {center_y:.3f}, {center_z:.3f}), rayon={radius}m")
    rospy.loginfo(f"Orientation: roll={target_roll}°, pitch={target_pitch}°, yaw={target_yaw}°")
    
    # Générer les waypoints du cercle
    waypoints = []
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    package_path = rospkg.RosPack().get_path('vision_processing')
    step = 10
    json_path = os.path.join(package_path, 'datas', 'Trajectories_preprocess', f'Trajectory_{step}', f'trajectory_{step}.json')
    with open(json_path, 'r') as f:
        datas = json.load(f)
        states = datas['states']
        for state in states:
            end_effector_position = state['end_effector_position']
            end_effector_orientation = state['end_effector_orientation']
            pose = create_pose_with_quaternion(end_effector_position, end_effector_orientation)
            if not waypoints:
                waypoints.append(pose)
            else:
                last_pose = waypoints[-1]
                import math
                dist = math.sqrt(
                    (pose.position.x - last_pose.position.x)**2 +
                    (pose.position.y - last_pose.position.y)**2 +
                    (pose.position.z - last_pose.position.z)**2
                )
                if dist >= 1e-5:
                    waypoints.append(pose)
            
    # for angle in angles:
    #     x = center_x + radius * np.cos(angle)
    #     y = center_y + radius * np.sin(angle)
    #     z = center_z
        
    #     # Créer la pose avec orientation contrôlée
    #     pose = create_pose_with_orientation(
    #         x, y, z,
    #         roll_deg=target_roll,
    #         pitch_deg=target_pitch,
    #         yaw_deg=target_yaw
    #     )
    #     waypoints.append(pose)
    
    # Fermer le cercle
    waypoints.append(waypoints[0])
    
    rospy.loginfo(f"Waypoints générés: {len(waypoints)} points")
    
    # Vérifier l'orientation du premier waypoint
    first_quat = [waypoints[0].orientation.x, waypoints[0].orientation.y,
                  waypoints[0].orientation.z, waypoints[0].orientation.w]
    r_check = R.from_quat(first_quat)
    check_roll, check_pitch, check_yaw = r_check.as_euler('xyz', degrees=True)
    rospy.loginfo(f"Vérification orientation waypoint: roll={check_roll:.1f}°, "
                  f"pitch={check_pitch:.1f}°, yaw={check_yaw:.1f}°")
    
    # ============================================
    # ÉTAPE 4: EXÉCUTER LE CERCLE
    # ============================================
    rospy.loginfo("\n--- ÉTAPE 4: Exécution du cercle ---")
    speed = 0.1  # 10% de la vitesse max
    success = move_through_waypoints(move_group, waypoints, speed=speed)
    
    if success:
        rospy.loginfo("=" * 60)
        rospy.loginfo("✓ CERCLE COMPLÉTÉ!")
        rospy.loginfo("=" * 60)
    else:
        rospy.logerr("✗ Échec de la trajectoire")
    
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
