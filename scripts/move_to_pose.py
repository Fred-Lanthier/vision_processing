#!/usr/bin/env python3
"""
Déplacer le Franka Panda à une pose cartésienne spécifique
"""

import rospy
import moveit_commander
import sys
import numpy as np
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R

def main():
    # Initialisation
    rospy.init_node('move_to_pose')
    moveit_commander.roscpp_initialize(sys.argv)
    
    robot = moveit_commander.RobotCommander()
    move_group = moveit_commander.MoveGroupCommander("panda_arm")
    move_group.set_end_effector_link("panda_hand_tcp")
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("Déplacement vers pose cible")
    rospy.loginfo("=" * 60)
    
    # Attendre que MoveIt soit prêt
    rospy.sleep(2)
    
    # ============================================
    # POSE ACTUELLE
    # ============================================
    current = move_group.get_current_pose().pose
    current_quat_scipy = [current.orientation.x, current.orientation.y, 
                          current.orientation.z, current.orientation.w]
    current_r = R.from_quat(current_quat_scipy)
    current_roll, current_pitch, current_yaw = current_r.as_euler('xyz', degrees=True)
    
    rospy.loginfo(f"Pose actuelle:")
    rospy.loginfo(f"  Position: x={current.position.x:.3f}, y={current.position.y:.3f}, z={current.position.z:.3f}")
    rospy.loginfo(f"  Quaternion: qw={current.orientation.w:.4f}, qx={current.orientation.x:.4f}, qy={current.orientation.y:.4f}, qz={current.orientation.z:.4f}")
    rospy.loginfo(f"  Euler XYZ: roll={current_roll:.1f}°, pitch={current_pitch:.1f}°, yaw={current_yaw:.1f}°")
    
    # ============================================
    # POSE CIBLE
    # ============================================
    target_pose = Pose()
    
    # Position
    target_pose.position.x = 0.5064181802418
    target_pose.position.y = 9.677804093958376e-05
    target_pose.position.z = 0.335494979650072
    
    # Quaternion depuis le JSON (ordre dans JSON: qw, qx, qy, qz)
    # Note: Dans ton JSON "end_effector_orientation", c'est probablement [qw, qx, qy, qz]
    # Quaternion cible du JSON
    qw_json = 0.9999962676013766
    qx_json = 6.836190407025131e-05
    qy_json = -0.0016231937042559697
    qz_json = -0.00010979196759587877
    
    # Créer la rotation depuis le quaternion JSON
    quaternion_scipy = np.array([qx_json, qy_json, qz_json, qw_json])
    r_json = R.from_quat(quaternion_scipy)
    
    # Ajouter une rotation de 180° autour de X pour inverser
    r_flip = R.from_euler('x', 180, degrees=True)
    r_final = r_flip * r_json
    
    # Convertir en quaternion pour ROS
    quat_final = r_final.as_quat()  # [qx, qy, qz, qw]
    
    # Assigner à ROS Pose
    target_pose.orientation.x = quat_final[0]
    target_pose.orientation.y = quat_final[1]
    target_pose.orientation.z = quat_final[2]
    target_pose.orientation.w = quat_final[3]
    
    # Convertir en Euler pour affichage
    roll, pitch, yaw = r_final.as_euler('xyz', degrees=True)
    
    rospy.loginfo(f"\nPose cible (AVEC rotation de 180° en X):")
    rospy.loginfo(f"  Position: x={target_pose.position.x:.4f}, y={target_pose.position.y:.6f}, z={target_pose.position.z:.4f}")
    rospy.loginfo(f"  Quaternion: qw={target_pose.orientation.w:.4f}, qx={target_pose.orientation.x:.6f}, qy={target_pose.orientation.y:.4f}, qz={target_pose.orientation.z:.6f}")
    rospy.loginfo(f"  Euler XYZ: roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°")
    
    # ============================================
    # PLANIFICATION
    # ============================================
    move_group.set_pose_target(target_pose)
    
    rospy.loginfo("\nPlanification...")
    move_group.set_planning_time(5.0)
    plan = move_group.plan()
    
    # Vérifier si la planification a réussi
    if isinstance(plan, tuple):
        success = plan[0]
        trajectory = plan[1]
    else:
        success = len(plan.joint_trajectory.points) > 0
        trajectory = plan
    
    # ============================================
    # EXÉCUTION
    # ============================================
    if success:
        rospy.loginfo("✓ Planification réussie!")
        rospy.loginfo("\nExécution du mouvement...")
        
        # Exécuter
        move_group.execute(trajectory, wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("✓ MOUVEMENT COMPLÉTÉ!")
        rospy.loginfo("=" * 60)
        
        # Afficher la pose finale
        final = move_group.get_current_pose().pose
        final_quat_scipy = [final.orientation.x, final.orientation.y, 
                           final.orientation.z, final.orientation.w]
        final_r = R.from_quat(final_quat_scipy)
        final_roll, final_pitch, final_yaw = final_r.as_euler('xyz', degrees=True)
        
        rospy.loginfo(f"\nPose finale:")
        rospy.loginfo(f"  Position: x={final.position.x:.4f}, y={final.position.y:.6f}, z={final.position.z:.4f}")
        rospy.loginfo(f"  Quaternion: qw={final.orientation.w:.4f}, qx={final.orientation.x:.6f}, qy={final.orientation.y:.4f}, qz={final.orientation.z:.6f}")
        rospy.loginfo(f"  Euler XYZ: roll={final_roll:.1f}°, pitch={final_pitch:.1f}°, yaw={final_yaw:.1f}°")
        
    else:
        rospy.logerr("✗ Planification échouée!")
        rospy.logerr("La pose cible n'est peut-être pas atteignable.")
    
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
