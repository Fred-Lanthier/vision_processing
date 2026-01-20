#!/usr/bin/env python3
# """
# Corrected Trajectory Follower with Proper Action Chunking

# Key changes from original:
# 1. Execute a CHUNK of actions from each trajectory prediction (action_chunk_size)
# 2. Only re-plan AFTER executing the chunk (not every timestep)
# 3. Remove the slow low-pass filter
# 4. Proper timing synchronization
# """
# import rospy
# import rospkg
# import moveit_commander
# import sys
# import os
# import numpy as np
# import threading
# from geometry_msgs.msg import PoseStamped, PoseArray, Pose
# from std_msgs.msg import Float64MultiArray
# from sensor_msgs.msg import JointState
# from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
# from moveit_msgs.msg import MoveItErrorCodes
# from scipy.spatial.transform import Rotation as R

# # Import your utility function
# from utils import compute_T_child_parent_xacro


# class TrajectoryFollowerChunked:
#     """
#     Action Chunking approach following DP3 methodology:
#     - Receive a trajectory of N predicted poses
#     - Execute action_chunk_size poses from it
#     - Then re-plan
#     """
    
#     def __init__(self):
#         rospy.init_node('trajectory_follower_node')
        
#         # Load transforms (you'll need to compute these as before)
#         rospack = rospkg.RosPack()
#         package_path = rospack.get_path('vision_processing')
#         xacro_file = os.path.join(package_path, 'urdf', 'panda_camera.xacro')
        
#         # For now, using a placeholder - you should use your actual function
#         self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro_file, 'fork_tip', 'panda_TCP')
#         self.T_tcp_fork_tip_inv = np.linalg.inv(self.T_tcp_fork_tip)
        
#         # Placeholder transform (replace with your actual computation)
#         # self.T_tcp_fork_tip_inv = np.eye(4)
#         # rospy.logwarn("Using identity transform - replace with actual T_tcp_fork_tip_inv!")
        
#         # =============================================
#         # KEY PARAMETERS - These are what DP3 uses
#         # =============================================
#         self.control_rate = 10.0  # Hz - same as your inference rate
#         self.action_chunk_size = 12  # Execute 4 actions per trajectory (DP3 uses N_act=3)
#         self.prediction_horizon = 16  # Your model predicts 16 steps
        
#         # State tracking
#         self.current_joints = None
#         self.current_trajectory = None  # List of joint configurations
#         self.trajectory_index = 0  # Which action in the chunk we're executing
#         self.trajectory_lock = threading.Lock()
        
#         # Safety parameters
#         self.max_joint_jump = 0.3  # radians - reject jumps larger than this
#         self.z_min_safety = 0.002  # minimum Z height
        
#         # MoveIt setup
#         moveit_commander.roscpp_initialize(sys.argv)
#         self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        
#         rospy.wait_for_service('/compute_ik')
#         self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        
#         # Publishers
#         self.pub_joints = rospy.Publisher(
#             '/joint_group_position_controller/command', 
#             Float64MultiArray, 
#             queue_size=1
#         )
        
#         # Subscribers
#         self.sub_traj = rospy.Subscriber(
#             "/diffusion/target_trajectory", 
#             PoseArray, 
#             self.traj_callback, 
#             queue_size=1,
#             buff_size=2**24 
#         )
#         self.sub_joints = rospy.Subscriber("/joint_states", JointState, self.joint_cb)
        
#         rospy.loginfo("🚀 Trajectory Follower (CHUNKED) Ready")
#         rospy.loginfo(f"   Action chunk size: {self.action_chunk_size}")
#         rospy.loginfo(f"   Control rate: {self.control_rate} Hz")

#     def joint_cb(self, msg):
#         """Store current joint positions"""
#         positions = []
#         for name, pos in zip(msg.name, msg.position):
#             if "panda_joint" in name:
#                 positions.append(pos)
#         if len(positions) == 7:
#             self.current_joints = np.array(positions)

#     def transform_pose(self, pose_ros, transformation_matrix):
#         """Transform a ROS pose by a 4x4 matrix"""
#         p = np.array([pose_ros.position.x, pose_ros.position.y, pose_ros.position.z])
#         q = np.array([pose_ros.orientation.x, pose_ros.orientation.y, 
#                       pose_ros.orientation.z, pose_ros.orientation.w])
        
#         T_pose = np.eye(4)
#         T_pose[:3, :3] = R.from_quat(q).as_matrix()
#         T_pose[:3, 3] = p
        
#         T_new = T_pose @ transformation_matrix
        
#         new_pose = Pose()
#         new_pose.position.x, new_pose.position.y, new_pose.position.z = T_new[:3, 3]
        
#         new_q = R.from_matrix(T_new[:3, :3]).as_quat()
#         new_pose.orientation.x = new_q[0]
#         new_pose.orientation.y = new_q[1]
#         new_pose.orientation.z = new_q[2]
#         new_pose.orientation.w = new_q[3]
        
#         return new_pose

#     def compute_ik(self, pose, frame_id, seed=None, timeout=0.05):
#         """Compute inverse kinematics for a target pose"""
#         req = GetPositionIKRequest()
#         req.ik_request.group_name = "panda_arm"
#         req.ik_request.pose_stamped.header.frame_id = frame_id
#         req.ik_request.pose_stamped.pose = pose
#         req.ik_request.avoid_collisions = False 
#         req.ik_request.ik_link_name = "panda_hand_tcp" 
#         req.ik_request.timeout = rospy.Duration(timeout)
        
#         if seed is not None:
#             req.ik_request.robot_state.joint_state.name = [f"panda_joint{i+1}" for i in range(7)]
#             seed_list = seed.tolist() if hasattr(seed, 'tolist') else list(seed)
#             req.ik_request.robot_state.joint_state.position = seed_list
            
#         try:
#             resp = self.ik_service(req)
#             if resp.error_code.val == MoveItErrorCodes.SUCCESS:
#                 return np.array(list(resp.solution.joint_state.position)[:7])
#             return None
#         except Exception as e:
#             rospy.logerr(f"IK Service Exception: {e}")
#             return None

#     def traj_callback(self, msg):
#         """
#         When a new trajectory arrives:
#         1. Convert all poses to joint configurations (IK)
#         2. Store the first action_chunk_size valid configurations
#         3. Reset the trajectory index
#         """
#         if self.current_joints is None:
#             return
        
#         frame_id = msg.header.frame_id
#         new_trajectory_joints = []
        
#         # Use current joints as seed for first IK
#         prev_joints = self.current_joints.copy()
        
#         # Process up to action_chunk_size poses from the prediction
#         # Skip the first pose (index 0) as it's approximately current state
#         start_idx = 1
#         end_idx = min(len(msg.poses), start_idx + self.action_chunk_size)
        
#         for i, pose in enumerate(msg.poses[:5]):  # First 5 poses
#             rospy.loginfo(f"Pose {i}: pos=({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})")
            
#         for i in range(start_idx, end_idx):
#             pose_fork = msg.poses[i]
            
#             # Safety: enforce minimum Z
#             if pose_fork.position.z < self.z_min_safety:
#                 pose_fork.position.z = self.z_min_safety
            
#             # Transform from fork_tip frame to TCP frame
#             pose_tcp = self.transform_pose(pose_fork, self.T_tcp_fork_tip_inv)
            
#             # Compute IK with seed from previous solution (continuity)
#             sol = self.compute_ik(pose_tcp, frame_id, seed=prev_joints, timeout=0.03)
            
#             if sol is None:
#                 # Try without seed as fallback
#                 sol = self.compute_ik(pose_tcp, frame_id, seed=None, timeout=0.05)
            
#             if sol is not None:
#                 # Check for joint jumps (safety)
#                 max_diff = np.max(np.abs(sol - prev_joints))
#                 if max_diff > self.max_joint_jump:
#                     rospy.logwarn(f"Rejecting point {i}: joint jump {max_diff:.3f} rad")
#                     break
                
#                 new_trajectory_joints.append(sol)
#                 prev_joints = sol
#             else:
#                 rospy.logwarn(f"IK failed for point {i}")
#                 break
        
#         # Update trajectory if we got valid points
#         if len(new_trajectory_joints) > 0:
#             with self.trajectory_lock:
#                 self.current_trajectory = new_trajectory_joints
#                 self.trajectory_index = 0
#                 rospy.loginfo(f"New trajectory chunk: {len(new_trajectory_joints)} points")

#     def run(self):
#         """
#         Main control loop:
#         - Execute one action per timestep from the current trajectory chunk
#         - When chunk is exhausted, the next callback will provide new actions
#         """
#         rate = rospy.Rate(self.control_rate)
        
#         while not rospy.is_shutdown():
#             if self.current_joints is None:
#                 rate.sleep()
#                 continue
            
#             target_joints = None
            
#             with self.trajectory_lock:
#                 if self.current_trajectory is not None:
#                     if self.trajectory_index < len(self.current_trajectory):
#                         target_joints = self.current_trajectory[self.trajectory_index]
#                         self.trajectory_index += 1
#                     else:
#                         # Chunk exhausted - hold last position until new trajectory arrives
#                         target_joints = self.current_trajectory[-1]
            
#             if target_joints is not None:
#                 # Safety check: don't allow huge jumps
#                 # In control loop, add logging:
#                 rospy.loginfo(f"Current joints: {self.current_joints}")
#                 rospy.loginfo(f"Target joints: {target_joints}")
#                 rospy.loginfo(f"Diff: {np.max(np.abs(target_joints - self.current_joints)):.4f}")
#                 max_diff = np.max(np.abs(target_joints - self.current_joints))
#                 if max_diff < 0.5:  # ~28 degrees
#                     msg = Float64MultiArray()
#                     msg.data = target_joints.tolist()
#                     self.pub_joints.publish(msg)
#                 else:
#                     rospy.logwarn(f"Blocked command: joint diff {max_diff:.2f} rad too large")
            
#             rate.sleep()


# class TrajectoryFollowerInterpolated:
#     """
#     Alternative: Time-based interpolation approach
    
#     Instead of executing discrete actions, interpolate smoothly
#     through the trajectory based on elapsed time.
    
#     This can give smoother motion but requires accurate timing.
#     """
    
#     def __init__(self):
#         rospy.init_node('trajectory_follower_interpolated')
        
#         # Parameters
#         self.control_rate = 50.0  # Higher rate for interpolation
#         self.trajectory_duration = 1.6  # 16 steps at 10Hz = 1.6 seconds
        
#         # State
#         self.current_joints = None
#         self.trajectory_poses = None
#         self.trajectory_joints = None  # Pre-computed IK for all poses
#         self.trajectory_start_time = None
#         self.trajectory_lock = threading.Lock()
        
#         rospack = rospkg.RosPack()
#         package_path = rospack.get_path('vision_processing')
        
#         # Placeholder - use your actual transform
#         self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro_file, 'fork_tip', 'panda_TCP')
#         self.T_tcp_fork_tip_inv = np.linalg.inv(self.T_tcp_fork_tip)
        
#         moveit_commander.roscpp_initialize(sys.argv)
#         self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        
#         rospy.wait_for_service('/compute_ik')
#         self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        
#         self.pub_joints = rospy.Publisher(
#             '/joint_group_position_controller/command', 
#             Float64MultiArray, 
#             queue_size=1
#         )
        
#         self.sub_traj = rospy.Subscriber(
#             "/diffusion/target_trajectory", 
#             PoseArray, 
#             self.traj_callback, 
#             queue_size=1
#         )
#         self.sub_joints = rospy.Subscriber("/joint_states", JointState, self.joint_cb)
        
#         rospy.loginfo("🚀 Trajectory Follower (INTERPOLATED) Ready")

#     def joint_cb(self, msg):
#         positions = []
#         for name, pos in zip(msg.name, msg.position):
#             if "panda_joint" in name:
#                 positions.append(pos)
#         if len(positions) == 7:
#             self.current_joints = np.array(positions)

#     def transform_pose(self, pose_ros, transformation_matrix):
#         """Same as above"""
#         p = np.array([pose_ros.position.x, pose_ros.position.y, pose_ros.position.z])
#         q = np.array([pose_ros.orientation.x, pose_ros.orientation.y, 
#                       pose_ros.orientation.z, pose_ros.orientation.w])
#         T_pose = np.eye(4)
#         T_pose[:3, :3] = R.from_quat(q).as_matrix()
#         T_pose[:3, 3] = p
#         T_new = T_pose @ transformation_matrix
#         new_pose = Pose()
#         new_pose.position.x, new_pose.position.y, new_pose.position.z = T_new[:3, 3]
#         new_q = R.from_matrix(T_new[:3, :3]).as_quat()
#         new_pose.orientation.x = new_q[0]
#         new_pose.orientation.y = new_q[1]
#         new_pose.orientation.z = new_q[2]
#         new_pose.orientation.w = new_q[3]
#         return new_pose

#     def compute_ik(self, pose, frame_id, seed=None, timeout=0.05):
#         """Same as above"""
#         req = GetPositionIKRequest()
#         req.ik_request.group_name = "panda_arm"
#         req.ik_request.pose_stamped.header.frame_id = frame_id
#         req.ik_request.pose_stamped.pose = pose
#         req.ik_request.avoid_collisions = False
#         req.ik_request.ik_link_name = "panda_hand_tcp"
#         req.ik_request.timeout = rospy.Duration(timeout)
        
#         if seed is not None:
#             req.ik_request.robot_state.joint_state.name = [f"panda_joint{i+1}" for i in range(7)]
#             req.ik_request.robot_state.joint_state.position = seed.tolist()
            
#         try:
#             resp = self.ik_service(req)
#             if resp.error_code.val == MoveItErrorCodes.SUCCESS:
#                 return np.array(list(resp.solution.joint_state.position)[:7])
#             return None
#         except:
#             return None

#     def traj_callback(self, msg):
#         """Pre-compute IK for the entire trajectory"""
#         if self.current_joints is None:
#             return
        
#         frame_id = msg.header.frame_id
#         trajectory_joints = []
#         prev_joints = self.current_joints.copy()
        
#         # Compute IK for all poses
#         for i, pose_fork in enumerate(msg.poses):
#             if pose_fork.position.z < 0.002:
#                 pose_fork.position.z = 0.002
            
#             pose_tcp = self.transform_pose(pose_fork, self.T_tcp_fork_tip_inv)
#             sol = self.compute_ik(pose_tcp, frame_id, seed=prev_joints, timeout=0.02)
            
#             if sol is not None:
#                 trajectory_joints.append(sol)
#                 prev_joints = sol
#             else:
#                 # Interpolate from previous if IK fails
#                 if len(trajectory_joints) > 0:
#                     trajectory_joints.append(trajectory_joints[-1])
#                 else:
#                     trajectory_joints.append(self.current_joints)
        
#         if len(trajectory_joints) > 0:
#             with self.trajectory_lock:
#                 self.trajectory_joints = np.array(trajectory_joints)
#                 self.trajectory_start_time = rospy.Time.now()

#     def interpolate_trajectory(self, t_normalized):
#         """
#         Interpolate trajectory at normalized time t in [0, 1]
#         Uses linear interpolation between waypoints
#         """
#         if self.trajectory_joints is None:
#             return None
        
#         n = len(self.trajectory_joints)
#         t_scaled = t_normalized * (n - 1)
        
#         idx = int(t_scaled)
#         idx = max(0, min(idx, n - 2))
        
#         alpha = t_scaled - idx
        
#         return (1 - alpha) * self.trajectory_joints[idx] + alpha * self.trajectory_joints[idx + 1]

#     def run(self):
#         rate = rospy.Rate(self.control_rate)
        
#         while not rospy.is_shutdown():
#             if self.current_joints is None:
#                 rate.sleep()
#                 continue
            
#             target_joints = None
            
#             with self.trajectory_lock:
#                 if self.trajectory_joints is not None and self.trajectory_start_time is not None:
#                     elapsed = (rospy.Time.now() - self.trajectory_start_time).to_sec()
#                     t_normalized = elapsed / self.trajectory_duration
                    
#                     if t_normalized > 1.0:
#                         # Past end of trajectory - hold last position
#                         target_joints = self.trajectory_joints[-1]
#                     else:
#                         target_joints = self.interpolate_trajectory(t_normalized)
            
#             if target_joints is not None:
#                 max_diff = np.max(np.abs(target_joints - self.current_joints))
#                 if max_diff < 0.5:
#                     msg = Float64MultiArray()
#                     msg.data = target_joints.tolist()
#                     self.pub_joints.publish(msg)
            
#             rate.sleep()


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--mode', type=str, default='chunked', 
#                         choices=['chunked', 'interpolated'],
#                         help='Execution mode: chunked (DP3-style) or interpolated')
#     args, _ = parser.parse_known_args()
    
#     if args.mode == 'chunked':
#         TrajectoryFollowerChunked().run()
#     else:
#         TrajectoryFollowerInterpolated().run()






"""
Trajectory Follower with Smooth Blending

When a new trajectory arrives:
1. Find the closest pose to current position
2. Create a smooth blend from current → closest
3. Then continue executing the rest of the trajectory
"""
import rospy
import rospkg
import moveit_commander
import sys
import os
import numpy as np
import threading
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_msgs.msg import MoveItErrorCodes
from scipy.spatial.transform import Rotation as R

from utils import compute_T_child_parent_xacro


class TrajectoryFollowerSmooth:
    
    def __init__(self):
        rospy.init_node('trajectory_follower_node')
        
        # Load transforms
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('vision_processing')
        xacro_file = os.path.join(package_path, 'urdf', 'panda_camera.xacro')
        
        self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro_file, 'fork_tip', 'panda_TCP')
        self.T_tcp_fork_tip_inv = np.linalg.inv(self.T_tcp_fork_tip)
        
        # =============================================
        # KEY PARAMETERS
        # =============================================
        self.control_rate = 10.0  # Hz
        self.action_chunk_size = 8  # Execute 8 actions before re-planning
        self.blend_steps = 3  # Number of steps to smoothly blend to new trajectory
        
        # State tracking
        self.current_joints = None
        self.current_trajectory = None
        self.trajectory_index = 0
        self.is_committed = False
        self.lock = threading.Lock()
        
        # Safety parameters
        self.max_joint_jump = 0.3
        self.z_min_safety = 0.002
        
        # MoveIt setup
        moveit_commander.roscpp_initialize(sys.argv)
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        
        rospy.wait_for_service('/compute_ik')
        self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        
        # Publishers/Subscribers
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
        
        rospy.loginfo("🚀 Trajectory Follower (SMOOTH BLEND) Ready")
        rospy.loginfo(f"   Action chunk size: {self.action_chunk_size}")
        rospy.loginfo(f"   Blend steps: {self.blend_steps}")

    def joint_cb(self, msg):
        positions = []
        for name, pos in zip(msg.name, msg.position):
            if "panda_joint" in name:
                positions.append(pos)
        if len(positions) == 7:
            self.current_joints = np.array(positions)

    def transform_pose(self, pose_ros, transformation_matrix):
        p = np.array([pose_ros.position.x, pose_ros.position.y, pose_ros.position.z])
        q = np.array([pose_ros.orientation.x, pose_ros.orientation.y, 
                      pose_ros.orientation.z, pose_ros.orientation.w])
        
        T_pose = np.eye(4)
        T_pose[:3, :3] = R.from_quat(q).as_matrix()
        T_pose[:3, 3] = p
        
        T_new = T_pose @ transformation_matrix
        
        new_pose = Pose()
        new_pose.position.x, new_pose.position.y, new_pose.position.z = T_new[:3, 3]
        
        new_q = R.from_matrix(T_new[:3, :3]).as_quat()
        new_pose.orientation.x = new_q[0]
        new_pose.orientation.y = new_q[1]
        new_pose.orientation.z = new_q[2]
        new_pose.orientation.w = new_q[3]
        
        return new_pose

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
                return np.array(list(resp.solution.joint_state.position)[:7])
            return None
        except Exception as e:
            rospy.logerr(f"IK Service Exception: {e}")
            return None

    def find_closest_pose_index(self, trajectory_joints):
        """Find which pose in the trajectory is closest to current position"""
        if self.current_joints is None or len(trajectory_joints) == 0:
            return 0
        
        min_dist = float('inf')
        closest_idx = 0
        
        for i, joints in enumerate(trajectory_joints):
            dist = np.linalg.norm(joints - self.current_joints)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        return closest_idx

    def create_blend(self, start_joints, end_joints, num_steps):
        """
        Create smooth interpolation from start to end.
        Uses cosine interpolation for smoother acceleration/deceleration.
        """
        blend = []
        for i in range(1, num_steps + 1):
            # Cosine interpolation: smoother than linear
            t = i / num_steps
            t_smooth = (1 - np.cos(t * np.pi)) / 2  # S-curve from 0 to 1
            
            interpolated = start_joints + t_smooth * (end_joints - start_joints)
            blend.append(interpolated)
        
        return blend

    def process_trajectory(self, msg, frame_id):
        """Convert PoseArray to list of joint configurations"""
        if self.current_joints is None:
            return None
        
        trajectory_joints = []
        prev_joints = self.current_joints.copy()
        
        # Process ALL poses (we'll select which ones to use later)
        for i, pose_fork in enumerate(msg.poses):
            if pose_fork.position.z < self.z_min_safety:
                pose_fork.position.z = self.z_min_safety
            
            pose_tcp = self.transform_pose(pose_fork, self.T_tcp_fork_tip_inv)
            sol = self.compute_ik(pose_tcp, frame_id, seed=prev_joints, timeout=0.03)
            
            if sol is None:
                sol = self.compute_ik(pose_tcp, frame_id, seed=None, timeout=0.05)
            
            if sol is not None:
                max_diff = np.max(np.abs(sol - prev_joints))
                if max_diff > self.max_joint_jump:
                    rospy.logwarn(f"Joint jump too large at pose {i}: {max_diff:.3f} rad")
                    break
                
                trajectory_joints.append(sol)
                prev_joints = sol
            else:
                rospy.logwarn(f"IK failed for pose {i}")
                # Use last good solution to maintain trajectory length
                if len(trajectory_joints) > 0:
                    trajectory_joints.append(trajectory_joints[-1])
        
        return trajectory_joints if len(trajectory_joints) > 0 else None

    def traj_callback(self, msg):
        """
        When a new trajectory arrives:
        - If committed: ignore (will get fresh one after chunk completes)
        - If not committed: find closest pose, blend to it, then continue
        """
        with self.lock:
            if self.is_committed:
                # Ignore - we're executing current chunk
                rospy.logdebug("Committed - ignoring new trajectory")
                return
            
            # Process the full trajectory
            trajectory_joints = self.process_trajectory(msg, msg.header.frame_id)
            
            if trajectory_joints is None or len(trajectory_joints) < 2:
                return
            
            # Find closest pose to current position
            closest_idx = self.find_closest_pose_index(trajectory_joints)
            
            # Determine where to start in the trajectory (after closest point)
            start_idx = min(closest_idx + 1, len(trajectory_joints) - 1)
            
            # Get remaining trajectory from start_idx
            remaining_trajectory = trajectory_joints[start_idx:]
            
            if len(remaining_trajectory) == 0:
                return
            
            # Create smooth blend from current position to first point of remaining trajectory
            blend_target = remaining_trajectory[0]
            distance_to_target = np.linalg.norm(blend_target - self.current_joints)
            
            # Adjust blend steps based on distance (more steps for larger jumps)
            adaptive_blend_steps = max(1, min(self.blend_steps, int(distance_to_target / 0.05) + 1))
            
            if distance_to_target > 0.01:  # Only blend if we're not already there
                blend = self.create_blend(self.current_joints, blend_target, adaptive_blend_steps)
                # Combine: blend + remaining trajectory (skip first point since blend ends there)
                full_trajectory = blend + remaining_trajectory[1:]
            else:
                full_trajectory = remaining_trajectory
            
            # Limit to action_chunk_size
            chunk_size = min(self.action_chunk_size, len(full_trajectory))
            self.current_trajectory = full_trajectory[:chunk_size]
            self.trajectory_index = 0
            self.is_committed = True
            
            rospy.loginfo(f"✅ New trajectory: closest_idx={closest_idx}, blend_steps={adaptive_blend_steps}, chunk={len(self.current_trajectory)}")
            
            # Debug: log first few poses
            for i, pose in enumerate(msg.poses[:5]):
                rospy.loginfo(f"Pose {i}: pos=({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})")

    def run(self):
        rate = rospy.Rate(self.control_rate)
        
        while not rospy.is_shutdown():
            if self.current_joints is None:
                rate.sleep()
                continue
            
            with self.lock:
                if self.current_trajectory is not None and self.trajectory_index < len(self.current_trajectory):
                    # Execute current action
                    target_joints = self.current_trajectory[self.trajectory_index]
                    
                    max_diff = np.max(np.abs(target_joints - self.current_joints))
                    if max_diff < 0.5:
                        msg = Float64MultiArray()
                        msg.data = target_joints.tolist()
                        self.pub_joints.publish(msg)
                        
                        rospy.loginfo(f"Executing {self.trajectory_index + 1}/{len(self.current_trajectory)}, diff={max_diff:.4f}")
                    else:
                        rospy.logwarn(f"Blocked: diff {max_diff:.2f} too large")
                    
                    self.trajectory_index += 1
                    
                    # Check if chunk complete
                    if self.trajectory_index >= len(self.current_trajectory):
                        rospy.loginfo("✅ Chunk complete - ready for new trajectory")
                        self.is_committed = False
                        self.current_trajectory = None
                
                elif self.current_trajectory is not None:
                    # Hold last position
                    target_joints = self.current_trajectory[-1]
                    max_diff = np.max(np.abs(target_joints - self.current_joints))
                    if max_diff < 0.5:
                        msg = Float64MultiArray()
                        msg.data = target_joints.tolist()
                        self.pub_joints.publish(msg)
            
            rate.sleep()


if __name__ == '__main__':
    TrajectoryFollowerSmooth().run()