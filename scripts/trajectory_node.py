#!/usr/bin/env python3
import rospy
import moveit_commander
import sys
import numpy as np
import threading
import time
import math
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest

# --- VISUALISATION ---
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TrajectoryFollower:
    def __init__(self):
        rospy.init_node('trajectory_follower_node')
        
        # --- CONFIGURATION ---
        self.control_rate = 60.0  # Hz
        self.model_dt = 0.1       # Temps entre chaque point IA
        
        # VISUALISATION BLOQUANTE (Mets False une fois validé)
        self.ENABLE_VISUALIZATION = True
        
        self.current_joints = None
        self.lock = threading.Lock()
        self.trajectory_buffer = [] 
        self.trajectory_start_time = 0.0
        
        # --- MOVEIT SETUP ---
        moveit_commander.roscpp_initialize(sys.argv)
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        
        rospy.wait_for_service('/compute_ik')
        self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        
        # --- COMMS ---
        self.pub_joints = rospy.Publisher(
            '/joint_group_position_controller/command', 
            Float64MultiArray, 
            queue_size=1
        )
        
        self.sub_traj = rospy.Subscriber(
            "/diffusion/target_trajectory", 
            PoseArray, 
            self.traj_callback, 
            queue_size=1,
            buff_size=2**24 
        )
        
        self.sub_joints = rospy.Subscriber("/joint_states", JointState, self.joint_cb)
        
        rospy.loginfo("🚀 Suivi de Trajectoire (Target: panda_TCP) Prêt.")

    def joint_cb(self, msg):
        positions = []
        for name, pos in zip(msg.name, msg.position):
            if "panda_joint" in name:
                positions.append(pos)
        if len(positions) == 7:
            self.current_joints = np.array(positions)

    def visualize_trajectory(self, start_pose, predicted_poses, frame_id):
        if not self.ENABLE_VISUALIZATION: return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. Robot Actuel (Bleu)
        ax.scatter(start_pose.position.x, start_pose.position.y, start_pose.position.z, 
                   c='blue', s=100, label='Robot TCP (Start)', marker='o')
        
        # 2. Trajectoire Prédite (Rouge)
        xs = [p.position.x for p in predicted_poses]
        ys = [p.position.y for p in predicted_poses]
        zs = [p.position.z for p in predicted_poses]
        
        ax.plot(xs, ys, zs, c='red', linewidth=2, label='Prédiction IA (TCP)')
        ax.scatter(xs, ys, zs, c='red', s=30, marker='x')
        
        # 3. Saut
        ax.plot([start_pose.position.x, xs[0]], 
                [start_pose.position.y, ys[0]], 
                [start_pose.position.z, zs[0]], 'g--', linewidth=1, label='Saut')

        dist_jump = math.sqrt(
            (start_pose.position.x - xs[0])**2 +
            (start_pose.position.y - ys[0])**2 +
            (start_pose.position.z - zs[0])**2
        )
        
        ax.set_title(f"Target: panda_TCP | Saut: {dist_jump*100:.2f} cm")
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend()
        
        # Echelle égale
        all_x = xs + [start_pose.position.x]
        all_y = ys + [start_pose.position.y]
        all_z = zs + [start_pose.position.z]
        max_range = np.array([np.ptp(all_x), np.ptp(all_y), np.ptp(all_z)]).max() / 2.0
        mid_x = (np.max(all_x)+np.min(all_x)) * 0.5
        mid_y = (np.max(all_y)+np.min(all_y)) * 0.5
        mid_z = (np.max(all_z)+np.min(all_z)) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        print(f"\n--- DEBUG VISUEL (TCP) ---")
        print(f"📍 Start (TCP): [{start_pose.position.x:.3f}, {start_pose.position.y:.3f}, {start_pose.position.z:.3f}]")
        print(f"🎯 Target (TCP): [{xs[0]:.3f}, {ys[0]:.3f}, {zs[0]:.3f}]")
        print(f"⚠️ SAUT        : {dist_jump:.4f} m")

        plt.show()

    def traj_callback(self, msg):
        frame_id = msg.header.frame_id
        
        # --- 1. VISUALISATION AVEC POSE DU TCP ---
        if self.ENABLE_VISUALIZATION:
            try:
                # MODIFICATION ICI : On demande explicitement la pose du TCP pour le debug
                current_pose_stamped = self.move_group.get_current_pose("panda_TCP")
                self.visualize_trajectory(current_pose_stamped.pose, msg.poses, frame_id)
            except Exception as e:
                # Fallback si panda_TCP n'est pas trouvé (mais il devrait l'être)
                rospy.logwarn(f"Erreur visu TCP: {e}. Essai avec défaut...")
                current_pose_stamped = self.move_group.get_current_pose()
                self.visualize_trajectory(current_pose_stamped.pose, msg.poses, frame_id)
        # -----------------------------------------
        
        new_trajectory_joints = []
        if self.current_joints is not None:
            new_trajectory_joints.append(self.current_joints)
        else:
            return 

        limit_horizon = min(len(msg.poses), 6)
        prev_joints = self.current_joints 
        
        for i in range(limit_horizon):
            pose = msg.poses[i]
            
            # Sécurité Sol (TCP ne doit pas traverser la table)
            if pose.position.z < 0.01: # On permet d'aller très bas car c'est le TCP
                pose.position.z = 0.01

            sol = self.compute_fast_ik(pose, frame_id, seed=prev_joints)
            
            if sol is not None:
                sol_array = np.array(sol)
                new_trajectory_joints.append(sol_array)
                prev_joints = sol_array 
            else:
                break
        
        if len(new_trajectory_joints) > 1:
            with self.lock:
                self.trajectory_buffer = new_trajectory_joints
                self.trajectory_start_time = time.time()

    def compute_fast_ik(self, pose, frame_id, seed=None):
        req = GetPositionIKRequest()
        req.ik_request.group_name = "panda_arm"
        req.ik_request.pose_stamped.header.frame_id = frame_id
        req.ik_request.pose_stamped.pose = pose
        req.ik_request.avoid_collisions = False
        
        # --- C'EST ICI QUE LA MAGIE OPÈRE ---
        # On dit à MoveIt de résoudre l'IK pour le link panda_TCP
        # et non pour le poignet.
        req.ik_request.ik_link_name = "panda_TCP"
        # ------------------------------------
        
        if seed is not None:
            req.ik_request.robot_state.joint_state.name = [f"panda_joint{i+1}" for i in range(7)]
            if hasattr(seed, 'tolist'):
                req.ik_request.robot_state.joint_state.position = seed.tolist()
            else:
                req.ik_request.robot_state.joint_state.position = list(seed)
            
        try:
            resp = self.ik_service(req)
            if resp.error_code.val == 1:
                return list(resp.solution.joint_state.position)[:7]
            else:
                # Debug si IK échoue pour panda_TCP (souvent hors portée)
                # rospy.logwarn(f"IK Failed pour TCP (Code {resp.error_code.val})")
                return None
        except:
            return None

    def run(self):
        rate = rospy.Rate(self.control_rate)
        
        while not rospy.is_shutdown():
            if self.current_joints is None or not self.trajectory_buffer:
                rate.sleep()
                continue
            
            with self.lock:
                now = time.time()
                elapsed = now - self.trajectory_start_time
                
                idx_float = elapsed / self.model_dt
                idx_base = int(idx_float)
                alpha = idx_float - idx_base 
                
                if idx_base >= len(self.trajectory_buffer) - 1:
                    target_joints = self.trajectory_buffer[-1]
                else:
                    start_j = self.trajectory_buffer[idx_base]
                    end_j = self.trajectory_buffer[idx_base + 1]
                    target_joints = (1 - alpha) * start_j + alpha * end_j

            msg = Float64MultiArray()
            msg.data = target_joints.tolist()
            self.pub_joints.publish(msg)
            rate.sleep()

if __name__ == '__main__':
    TrajectoryFollower().run()