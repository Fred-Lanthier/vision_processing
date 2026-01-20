#!/usr/bin/env python3
import rospy
import moveit_commander
import sys
import numpy as np
import threading
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_msgs.msg import MoveItErrorCodes

# NOTE: Suppression totale de Matplotlib pour garantir le temps réel

class TrajectoryFollower:
    def __init__(self):
        rospy.init_node('trajectory_follower_node')
        
        # Définition de la transformation TCP -> Fork tip
        
        self.filtered_joints = None
        # Alpha 0.08 = Très fluide (absorbe les tremblements de fin de course)
        self.alpha_filter = 0.08 
        
        # --- CONFIGURATION ---
        self.control_rate = 60.0 
        self.model_dt = 0.1
        
        self.current_joints = None
        self.lock = threading.Lock()
        self.trajectory_buffer = [] 
        self.trajectory_start_time = rospy.Time(0) 
        
        moveit_commander.roscpp_initialize(sys.argv)
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        
        rospy.wait_for_service('/compute_ik')
        self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        
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
        
        rospy.loginfo("🚀 Suivi de Trajectoire ROBUSTE Prêt.")

    def joint_cb(self, msg):
        positions = []
        for name, pos in zip(msg.name, msg.position):
            if "panda_joint" in name:
                positions.append(pos)
        if len(positions) == 7:
            self.current_joints = np.array(positions)

    def traj_callback(self, msg):
        frame_id = msg.header.frame_id
        msg_stamp = msg.header.stamp
        
        new_trajectory_joints = []
        if self.current_joints is None: return 

        limit_horizon = len(msg.poses)
        prev_joints = self.current_joints 
        
        for i in range(limit_horizon):
            pose = msg.poses[i]
            if pose.position.z < 0.02: pose.position.z = 0.02

            # --- TENTATIVE 1 : IK RAPIDE (Avec Seed) ---
            # On essaye de rester proche de la config précédente
            sol = self.compute_ik(pose, frame_id, seed=prev_joints, timeout=0.03)
            
            # --- TENTATIVE 2 : IK ROBUSTE (Sans Seed - Fallback) ---
            if sol is None:
                # Si bloqué (singularité ou changement de config), on laisse MoveIt chercher librement
                sol = self.compute_ik(pose, frame_id, seed=None, timeout=0.05)

            if sol is not None:
                sol_array = np.array(sol)
                
                # Check Jump (Anti-Flip)
                max_diff = np.max(np.abs(sol_array - prev_joints))
                
                # Seuil large (0.85 rad) pour accepter le redressement rapide du poignet
                if max_diff > 0.85: 
                    rospy.logwarn(f"🛑 REJET SAUT (Pt {i}): Diff {max_diff:.2f} rad > 0.85")
                    break 
                
                new_trajectory_joints.append(sol_array)
                prev_joints = sol_array 
            else:
                rospy.logwarn(f"❌ IK TOTAL FAILURE (Pt {i}): Target inatteignable.")
                break
        
        if len(new_trajectory_joints) > 0:
            # --- DEADBAND (Anti-Tremblement final) ---
            # On regarde si le mouvement total demandé est significatif
            last_point = new_trajectory_joints[-1]
            current_pos = self.current_joints
            movement_magnitude = np.sum(np.abs(last_point - current_pos))
            
            # Si on bouge de moins de 0.15 rad (cumulé sur 7 joints), on ignore la nouvelle traj
            # Cela permet de stabiliser le robot une fois arrivé.
            if movement_magnitude < 0.15: 
                return

            with self.lock:
                self.trajectory_buffer = new_trajectory_joints
                self.trajectory_start_time = msg_stamp
        else:
            rospy.logwarn_throttle(1, "❄️ Buffer vide : Trajectoire rejetée ou IK invalide.")

    def compute_ik(self, pose, frame_id, seed=None, timeout=0.05):
        req = GetPositionIKRequest()
        req.ik_request.group_name = "panda_arm"
        req.ik_request.pose_stamped.header.frame_id = frame_id
        req.ik_request.pose_stamped.pose = pose
        req.ik_request.avoid_collisions = False 
        req.ik_request.ik_link_name = "panda_hand_tcp" 
        req.ik_request.timeout = rospy.Duration(timeout)
        
        if seed is not None:
            req.ik_request.robot_state.joint_state.name = [f"panda_joint{i+1}" for i in range(7)]
            seed_list = seed.tolist() if hasattr(seed, 'tolist') else list(seed)
            req.ik_request.robot_state.joint_state.position = seed_list
            
        try:
            resp = self.ik_service(req)
            if resp.error_code.val == MoveItErrorCodes.SUCCESS:
                return list(resp.solution.joint_state.position)[:7]
            else:
                return None
        except Exception as e:
            rospy.logerr(f"Service IK Exception: {e}")
            return None

    def run(self):
        rate = rospy.Rate(self.control_rate)
        
        while not rospy.is_shutdown():
            if self.current_joints is None:
                rate.sleep()
                continue
            
            target_joints = None
            
            with self.lock:
                if not self.trajectory_buffer:
                     target_joints = self.current_joints
                else:
                    now = rospy.Time.now()
                    elapsed = (now - self.trajectory_start_time).to_sec()
                    if elapsed < 0: elapsed = 0
                    
                    idx_float = elapsed / self.model_dt
                    idx_base = int(idx_float)
                    alpha = idx_float - idx_base 
                    
                    if idx_base >= len(self.trajectory_buffer) - 1:
                        target_joints = self.trajectory_buffer[-1]
                    else:
                        start_j = self.trajectory_buffer[idx_base]
                        end_j = self.trajectory_buffer[idx_base + 1]
                        target_joints = (1 - alpha) * start_j + alpha * end_j

            if self.filtered_joints is None:
                self.filtered_joints = self.current_joints

            if target_joints is not None:
                # Filtre passe-bas pour lisser les secousses
                self.filtered_joints = (1 - self.alpha_filter) * self.filtered_joints + \
                                    self.alpha_filter * target_joints
            
            msg = Float64MultiArray()
            msg.data = self.filtered_joints.tolist()
            self.pub_joints.publish(msg)
            
            rate.sleep()

if __name__ == '__main__':
    TrajectoryFollower().run()