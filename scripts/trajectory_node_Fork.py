#!/usr/bin/env python3
"""
Trajectory Follower for FORK - Continuous Blending Approach

DIFFERENT PHILOSOPHY:
Instead of discrete chunks with commitment, this version:
1. Always accepts the latest trajectory
2. Continuously blends toward a "target" point that advances along the trajectory
3. Uses a pursuit-style algorithm (like Pure Pursuit for mobile robots)

This often produces smoother results for diffusion policies because:
- No hard transitions between chunks
- Natural velocity limiting through the blending
- Automatic handling of prediction noise

The key idea: Don't commit to waypoints, commit to a DIRECTION.
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


class TrajectoryFollowerContinuous:
    
    def __init__(self):
        rospy.init_node('trajectory_follower_fork_continuous')
        
        # --- FORK TRANSFORM ---
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('vision_processing')
        xacro_file = os.path.join(package_path, 'urdf', 'panda_camera.xacro')
        
        self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro_file, 'fork_tip', 'panda_TCP')
        self.T_tcp_fork_tip_inv = np.linalg.inv(self.T_tcp_fork_tip)
        rospy.loginfo("✅ Fork transform loaded")
        
        # =============================================
        # KEY PARAMETERS
        # =============================================
        
        # High control rate for smooth motion
        self.control_rate = 30.0  # Hz
        
        # LOOKAHEAD: How far ahead in the trajectory to target
        # Larger = smoother but slower to react
        # Smaller = more reactive but potentially jerky
        self.lookahead_steps = 3  # Target the 3rd waypoint ahead
        
        # BLEND RATE: How fast to blend toward target (per control step)
        # 0.0 = don't move, 1.0 = instant snap to target
        # 0.1-0.3 is usually good
        self.blend_rate = 0.15
        
        # Maximum joint velocity (rad/s) - limits motion speed
        self.max_joint_velocity = 0.7  # rad/s per joint
        
        # Safety
        self.max_joint_jump = 0.5
        self.z_min_safety = 0.002
        
        # State
        self.current_joints = None
        self.target_joints = None  # Current target we're blending toward
        self.latest_trajectory = None  # Most recent trajectory from inference
        self.trajectory_timestamp = None
        self.lock = threading.Lock()
        
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
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("🚀 Fork Trajectory Follower (Continuous Blending)")
        rospy.loginfo(f"   Control rate: {self.control_rate} Hz")
        rospy.loginfo(f"   Lookahead steps: {self.lookahead_steps}")
        rospy.loginfo(f"   Blend rate: {self.blend_rate}")
        rospy.loginfo(f"   Max velocity: {self.max_joint_velocity} rad/s")
        rospy.loginfo("=" * 60)

    def joint_cb(self, msg):
        positions = []
        for name, pos in zip(msg.name, msg.position):
            if "panda_joint" in name:
                positions.append(pos)
        if len(positions) == 7:
            self.current_joints = np.array(positions)
            
            # Initialize target to current position
            if self.target_joints is None:
                self.target_joints = self.current_joints.copy()

    def transform_fork_to_tcp(self, pose_fork):
        """Transform pose from fork_tip frame to TCP frame."""
        p = np.array([pose_fork.position.x, pose_fork.position.y, pose_fork.position.z])
        q = np.array([pose_fork.orientation.x, pose_fork.orientation.y, 
                      pose_fork.orientation.z, pose_fork.orientation.w])
        
        T_world_fork = np.eye(4)
        T_world_fork[:3, :3] = R.from_quat(q).as_matrix()
        T_world_fork[:3, 3] = p
        
        T_world_tcp = T_world_fork @ self.T_tcp_fork_tip_inv
        
        pose_tcp = Pose()
        pose_tcp.position.x, pose_tcp.position.y, pose_tcp.position.z = T_world_tcp[:3, 3]
        
        q_tcp = R.from_matrix(T_world_tcp[:3, :3]).as_quat()
        pose_tcp.orientation.x = q_tcp[0]
        pose_tcp.orientation.y = q_tcp[1]
        pose_tcp.orientation.z = q_tcp[2]
        pose_tcp.orientation.w = q_tcp[3]
        
        return pose_tcp

    def compute_ik(self, pose, frame_id, seed=None, timeout=0.03):
        req = GetPositionIKRequest()
        req.ik_request.group_name = "panda_arm"
        req.ik_request.pose_stamped.header.frame_id = frame_id
        req.ik_request.pose_stamped.pose = pose
        req.ik_request.avoid_collisions = False 
        req.ik_request.ik_link_name = "panda_hand_tcp" 
        req.ik_request.timeout = rospy.Duration(timeout)
        
        if seed is not None:
            req.ik_request.robot_state.joint_state.name = [f"panda_joint{i+1}" for i in range(7)]
            req.ik_request.robot_state.joint_state.position = seed.tolist()
            
        try:
            resp = self.ik_service(req)
            if resp.error_code.val == MoveItErrorCodes.SUCCESS:
                return np.array(list(resp.solution.joint_state.position)[:7])
            return None
        except:
            return None

    def process_trajectory(self, msg, frame_id):
        """Convert PoseArray to joint configurations."""
        if self.current_joints is None:
            return None
        
        trajectory_joints = []
        prev_joints = self.current_joints.copy()
        
        for i, pose_fork in enumerate(msg.poses):
            if pose_fork.position.z < self.z_min_safety:
                pose_fork.position.z = self.z_min_safety
            
            pose_tcp = self.transform_fork_to_tcp(pose_fork)
            sol = self.compute_ik(pose_tcp, frame_id, seed=prev_joints)
            
            if sol is not None:
                if np.max(np.abs(sol - prev_joints)) <= self.max_joint_jump:
                    trajectory_joints.append(sol)
                    prev_joints = sol
                else:
                    break
            else:
                if len(trajectory_joints) > 0:
                    trajectory_joints.append(trajectory_joints[-1])
        
        return trajectory_joints if len(trajectory_joints) > 0 else None

    def find_closest_index(self, trajectory):
        """Find index of closest point in trajectory to current position."""
        if self.current_joints is None or len(trajectory) == 0:
            return 0
        
        distances = [np.linalg.norm(t - self.current_joints) for t in trajectory]
        return np.argmin(distances)

    def traj_callback(self, msg):
        """
        Store the latest trajectory. No commitment - just update.
        """
        with self.lock:
            trajectory_joints = self.process_trajectory(msg, msg.header.frame_id)
            
            if trajectory_joints is not None and len(trajectory_joints) >= 2:
                self.latest_trajectory = trajectory_joints
                self.trajectory_timestamp = rospy.Time.now()

    def get_lookahead_target(self):
        """
        Find the target point to blend toward.
        
        Uses "carrot on a stick" approach:
        1. Find closest point on trajectory
        2. Look ahead by lookahead_steps
        3. Return that as target
        """
        if self.latest_trajectory is None or len(self.latest_trajectory) == 0:
            return None
        
        # Find where we are on the trajectory
        closest_idx = self.find_closest_index(self.latest_trajectory)
        
        # Look ahead
        target_idx = min(closest_idx + self.lookahead_steps, len(self.latest_trajectory) - 1)
        
        return self.latest_trajectory[target_idx]

    def compute_command(self):
        """
        Compute the joint command by blending toward the lookahead target.
        
        This is the core of the continuous blending approach:
        - Find where we want to go (lookahead target)
        - Blend smoothly toward it
        - Limit velocity
        """
        if self.current_joints is None or self.target_joints is None:
            return None
        
        # Get lookahead target from latest trajectory
        lookahead_target = self.get_lookahead_target()
        
        if lookahead_target is None:
            # No trajectory - hold position
            return self.target_joints
        
        # Blend current target toward lookahead target
        # This creates smooth transitions even when trajectory changes
        self.target_joints = (1 - self.blend_rate) * self.target_joints + self.blend_rate * lookahead_target
        
        # Compute command with velocity limiting
        direction = self.target_joints - self.current_joints
        distance = np.linalg.norm(direction)
        
        if distance < 0.001:
            return self.target_joints
        
        # Maximum step size based on velocity limit and control rate
        max_step = self.max_joint_velocity / self.control_rate
        
        if distance > max_step:
            # Limit velocity
            direction = direction / distance * max_step
            command = self.current_joints + direction
        else:
            command = self.target_joints
        
        return command

    def run(self):
        rate = rospy.Rate(self.control_rate)
        
        while not rospy.is_shutdown():
            if self.current_joints is None:
                rate.sleep()
                continue
            
            with self.lock:
                command = self.compute_command()
            
            if command is not None:
                # Safety check
                max_diff = np.max(np.abs(command - self.current_joints))
                
                if max_diff < 0.5:
                    msg = Float64MultiArray()
                    msg.data = command.tolist()
                    self.pub_joints.publish(msg)
            
            rate.sleep()


if __name__ == '__main__':
    TrajectoryFollowerContinuous().run()









# """
# Trajectory Follower for FORK with Smooth Blending + EMA Filter

# Key differences from base version:
# - Transforms fork_tip poses to TCP poses before IK
# - Uses T_tcp_fork_tip_inv transform
# - Lower z_min_safety for fork near food

# Smoothing features:
# 1. Temporal commitment: Execute N actions before accepting new trajectory
# 2. Skip to closest pose: Find where we are in new trajectory
# 3. Cosine blending: Smooth interpolation between chunks
# 4. EMA filter: Final smoothing layer on joint commands
# """
# import rospy
# import rospkg
# import moveit_commander
# import sys
# import os
# import numpy as np
# import threading
# from geometry_msgs.msg import PoseArray, Pose
# from std_msgs.msg import Float64MultiArray
# from sensor_msgs.msg import JointState
# from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
# from moveit_msgs.msg import MoveItErrorCodes
# from scipy.spatial.transform import Rotation as R

# # Utils for fork transform
# from utils import compute_T_child_parent_xacro


# class TrajectoryFollowerForkSmooth:
    
#     def __init__(self):
#         rospy.init_node('trajectory_follower_fork_node')
        
#         # --- FORK TRANSFORM ---
#         rospack = rospkg.RosPack()
#         package_path = rospack.get_path('vision_processing')
#         xacro_file = os.path.join(package_path, 'urdf', 'panda_camera.xacro')
        
#         self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro_file, 'fork_tip', 'panda_TCP')
#         self.T_tcp_fork_tip_inv = np.linalg.inv(self.T_tcp_fork_tip)
#         rospy.loginfo("✅ Fork transform loaded")
        
#         # =============================================
#         # KEY PARAMETERS
#         # =============================================
#         self.control_rate = 8.0  # Hz - matches inference rate
#         self.action_chunk_size = 8  # Execute 8 actions before re-planning
#         self.blend_steps = 5  # Steps to blend to new trajectory
        
#         # --- EMA FILTER PARAMETERS ---
#         self.ema_alpha = rospy.get_param("~ema_alpha", 0.5)
#         self.filtered_joints = None
        
#         # State tracking
#         self.current_joints = None
#         self.current_trajectory = None
#         self.trajectory_index = 0
#         self.is_committed = False
#         self.lock = threading.Lock()
        
#         # Safety parameters (lower z for fork near food)
#         self.max_joint_jump = 0.4  # radians
#         self.z_min_safety = 0.002  # Fork tip can be close to food
        
#         # MoveIt setup
#         moveit_commander.roscpp_initialize(sys.argv)
#         self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        
#         rospy.wait_for_service('/compute_ik')
#         self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        
#         # Publishers/Subscribers
#         self.pub_joints = rospy.Publisher(
#             '/joint_group_position_controller/command', 
#             Float64MultiArray, 
#             queue_size=1
#         )
        
#         self.sub_traj = rospy.Subscriber(
#             "/diffusion/target_trajectory", 
#             PoseArray, 
#             self.traj_callback, 
#             queue_size=1,
#             buff_size=2**24 
#         )
#         self.sub_joints = rospy.Subscriber("/joint_states", JointState, self.joint_cb)
        
#         rospy.loginfo("=" * 60)
#         rospy.loginfo("🚀 Fork Trajectory Follower (Smooth + EMA Filter)")
#         rospy.loginfo(f"   Action chunk size: {self.action_chunk_size}")
#         rospy.loginfo(f"   Blend steps: {self.blend_steps}")
#         rospy.loginfo(f"   EMA alpha: {self.ema_alpha}")
#         rospy.loginfo(f"   Z min safety: {self.z_min_safety}")
#         rospy.loginfo("=" * 60)

#     def joint_cb(self, msg):
#         positions = []
#         for name, pos in zip(msg.name, msg.position):
#             if "panda_joint" in name:
#                 positions.append(pos)
#         if len(positions) == 7:
#             self.current_joints = np.array(positions)
#             if self.filtered_joints is None:
#                 self.filtered_joints = self.current_joints.copy()

#     def transform_fork_to_tcp(self, pose_fork):
#         """
#         Transform a pose from fork_tip frame to TCP frame.
        
#         Args:
#             pose_fork: ROS Pose message in fork_tip frame
            
#         Returns:
#             ROS Pose message in TCP frame
#         """
#         # Extract position and orientation
#         p = np.array([pose_fork.position.x, pose_fork.position.y, pose_fork.position.z])
#         q = np.array([pose_fork.orientation.x, pose_fork.orientation.y, 
#                       pose_fork.orientation.z, pose_fork.orientation.w])
        
#         # Build transform matrix for fork pose
#         T_world_fork = np.eye(4)
#         T_world_fork[:3, :3] = R.from_quat(q).as_matrix()
#         T_world_fork[:3, 3] = p
        
#         # Transform to TCP frame: T_world_tcp = T_world_fork @ T_fork_tcp
#         # T_fork_tcp = inv(T_tcp_fork)
#         T_world_tcp = T_world_fork @ self.T_tcp_fork_tip_inv
        
#         # Convert back to Pose message
#         pose_tcp = Pose()
#         pose_tcp.position.x, pose_tcp.position.y, pose_tcp.position.z = T_world_tcp[:3, 3]
        
#         q_tcp = R.from_matrix(T_world_tcp[:3, :3]).as_quat()
#         pose_tcp.orientation.x = q_tcp[0]
#         pose_tcp.orientation.y = q_tcp[1]
#         pose_tcp.orientation.z = q_tcp[2]
#         pose_tcp.orientation.w = q_tcp[3]
        
#         return pose_tcp

#     def compute_ik(self, pose, frame_id, seed=None, timeout=0.05):
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

#     def find_closest_pose_index(self, trajectory_joints):
#         if self.current_joints is None or len(trajectory_joints) == 0:
#             return 0
        
#         min_dist = float('inf')
#         closest_idx = 0
        
#         for i, joints in enumerate(trajectory_joints):
#             dist = np.linalg.norm(joints - self.current_joints)
#             if dist < min_dist:
#                 min_dist = dist
#                 closest_idx = i
        
#         return closest_idx

#     def create_blend(self, start_joints, end_joints, num_steps):
#         """Cosine interpolation for smooth blending"""
#         blend = []
#         for i in range(1, num_steps + 1):
#             t = i / num_steps
#             t_smooth = (1 - np.cos(t * np.pi)) / 2  # S-curve
#             interpolated = start_joints + t_smooth * (end_joints - start_joints)
#             blend.append(interpolated)
#         return blend

#     def process_trajectory(self, msg, frame_id):
#         """
#         Convert PoseArray (fork_tip poses) to list of joint configurations.
#         Transforms each fork pose to TCP pose before IK.
#         """
#         if self.current_joints is None:
#             return None
        
#         trajectory_joints = []
#         prev_joints = self.current_joints.copy()
        
#         for i, pose_fork in enumerate(msg.poses):
#             # Safety: enforce minimum Z for fork tip
#             if pose_fork.position.z < self.z_min_safety:
#                 pose_fork.position.z = self.z_min_safety
            
#             # Transform from fork_tip frame to TCP frame
#             pose_tcp = self.transform_fork_to_tcp(pose_fork)
            
#             # Compute IK for TCP pose
#             sol = self.compute_ik(pose_tcp, frame_id, seed=prev_joints, timeout=0.03)
            
#             if sol is None:
#                 sol = self.compute_ik(pose_tcp, frame_id, seed=None, timeout=0.05)
            
#             if sol is not None:
#                 max_diff = np.max(np.abs(sol - prev_joints))
#                 if max_diff > self.max_joint_jump:
#                     rospy.logwarn(f"Joint jump too large at pose {i}: {max_diff:.3f} rad")
#                     break
                
#                 trajectory_joints.append(sol)
#                 prev_joints = sol
#             else:
#                 rospy.logwarn(f"IK failed for pose {i}")
#                 if len(trajectory_joints) > 0:
#                     trajectory_joints.append(trajectory_joints[-1])
        
#         return trajectory_joints if len(trajectory_joints) > 0 else None

#     def traj_callback(self, msg):
#         """
#         When a new trajectory arrives:
#         - If committed: ignore
#         - If not committed: find closest pose, blend to it, then continue
#         """
#         with self.lock:
#             if self.is_committed:
#                 rospy.logdebug("Committed - ignoring new trajectory")
#                 return
            
#             trajectory_joints = self.process_trajectory(msg, msg.header.frame_id)
            
#             if trajectory_joints is None or len(trajectory_joints) < 2:
#                 return
            
#             closest_idx = self.find_closest_pose_index(trajectory_joints)
#             start_idx = min(closest_idx + 1, len(trajectory_joints) - 1)
#             remaining_trajectory = trajectory_joints[start_idx:]
            
#             if len(remaining_trajectory) == 0:
#                 return
            
#             blend_target = remaining_trajectory[0]
#             distance_to_target = np.linalg.norm(blend_target - self.current_joints)
#             adaptive_blend_steps = max(1, min(self.blend_steps, int(distance_to_target / 0.05) + 1))
            
#             if distance_to_target > 0.01:
#                 blend = self.create_blend(self.current_joints, blend_target, adaptive_blend_steps)
#                 full_trajectory = blend + remaining_trajectory[1:]
#             else:
#                 full_trajectory = remaining_trajectory
            
#             chunk_size = min(self.action_chunk_size, len(full_trajectory))
#             self.current_trajectory = full_trajectory[:chunk_size]
#             self.trajectory_index = 0
#             self.is_committed = True
            
#             rospy.loginfo(f"✅ New trajectory: closest={closest_idx}, blend={adaptive_blend_steps}, chunk={len(self.current_trajectory)}")
            
#             # Debug: log first few fork poses
#             for i, pose in enumerate(msg.poses[:5]):
#                 rospy.loginfo(f"Fork pose {i}: ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})")

#     def apply_ema_filter(self, target_joints):
#         """
#         Apply Exponential Moving Average filter to joint commands.
#         """
#         if self.filtered_joints is None:
#             self.filtered_joints = target_joints.copy()
#             return target_joints
        
#         self.filtered_joints = self.ema_alpha * target_joints + (1 - self.ema_alpha) * self.filtered_joints
#         return self.filtered_joints

#     def run(self):
#         rate = rospy.Rate(self.control_rate)
        
#         while not rospy.is_shutdown():
#             if self.current_joints is None:
#                 rate.sleep()
#                 continue
            
#             with self.lock:
#                 if self.current_trajectory is not None and self.trajectory_index < len(self.current_trajectory):
#                     target_joints = self.current_trajectory[self.trajectory_index]
#                     smoothed_joints = self.apply_ema_filter(target_joints)
                    
#                     max_diff = np.max(np.abs(smoothed_joints - self.current_joints))
#                     if max_diff < 0.5:
#                         msg = Float64MultiArray()
#                         msg.data = smoothed_joints.tolist()
#                         self.pub_joints.publish(msg)
                        
#                         rospy.loginfo(f"Exec {self.trajectory_index + 1}/{len(self.current_trajectory)}, diff={max_diff:.4f}")
#                     else:
#                         rospy.logwarn(f"Blocked: diff {max_diff:.2f} too large")
                    
#                     self.trajectory_index += 1
                    
#                     if self.trajectory_index >= len(self.current_trajectory):
#                         rospy.loginfo("✅ Chunk complete")
#                         self.is_committed = False
#                         self.current_trajectory = None
                
#                 elif self.current_trajectory is not None:
#                     target_joints = self.current_trajectory[-1]
#                     smoothed_joints = self.apply_ema_filter(target_joints)
#                     max_diff = np.max(np.abs(smoothed_joints - self.current_joints))
#                     if max_diff < 0.5:
#                         msg = Float64MultiArray()
#                         msg.data = smoothed_joints.tolist()
#                         self.pub_joints.publish(msg)
            
#             rate.sleep()


# if __name__ == '__main__':
#     TrajectoryFollowerForkSmooth().run()






# """
# Trajectory Follower with Smooth Blending

# When a new trajectory arrives:
# 1. Find the closest pose to current position
# 2. Create a smooth blend from current → closest
# 3. Then continue executing the rest of the trajectory
# """
# import rospy
# import rospkg
# import moveit_commander
# import sys
# import os
# import numpy as np
# import threading
# from geometry_msgs.msg import PoseArray, Pose
# from std_msgs.msg import Float64MultiArray
# from sensor_msgs.msg import JointState
# from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
# from moveit_msgs.msg import MoveItErrorCodes
# from scipy.spatial.transform import Rotation as R

# from utils import compute_T_child_parent_xacro


# class TrajectoryFollowerSmooth:
    
#     def __init__(self):
#         rospy.init_node('trajectory_follower_node')
        
#         # Load transforms
#         rospack = rospkg.RosPack()
#         package_path = rospack.get_path('vision_processing')
#         xacro_file = os.path.join(package_path, 'urdf', 'panda_camera.xacro')
        
#         self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro_file, 'fork_tip', 'panda_TCP')
#         self.T_tcp_fork_tip_inv = np.linalg.inv(self.T_tcp_fork_tip)
        
#         # =============================================
#         # KEY PARAMETERS
#         # =============================================
#         self.control_rate = 10.0  # Hz
#         self.action_chunk_size = 8  # Execute 8 actions before re-planning
#         self.blend_steps = 3  # Number of steps to smoothly blend to new trajectory
        
#         # State tracking
#         self.current_joints = None
#         self.current_trajectory = None
#         self.trajectory_index = 0
#         self.is_committed = False
#         self.lock = threading.Lock()
        
#         # Safety parameters
#         self.max_joint_jump = 0.3
#         self.z_min_safety = 0.002
        
#         # MoveIt setup
#         moveit_commander.roscpp_initialize(sys.argv)
#         self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        
#         rospy.wait_for_service('/compute_ik')
#         self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        
#         # Publishers/Subscribers
#         self.pub_joints = rospy.Publisher(
#             '/joint_group_position_controller/command', 
#             Float64MultiArray, 
#             queue_size=1
#         )
        
#         self.sub_traj = rospy.Subscriber(
#             "/diffusion/target_trajectory", 
#             PoseArray, 
#             self.traj_callback, 
#             queue_size=1,
#             buff_size=2**24 
#         )
#         self.sub_joints = rospy.Subscriber("/joint_states", JointState, self.joint_cb)
        
#         rospy.loginfo("🚀 Trajectory Follower (SMOOTH BLEND) Ready")
#         rospy.loginfo(f"   Action chunk size: {self.action_chunk_size}")
#         rospy.loginfo(f"   Blend steps: {self.blend_steps}")

#     def joint_cb(self, msg):
#         positions = []
#         for name, pos in zip(msg.name, msg.position):
#             if "panda_joint" in name:
#                 positions.append(pos)
#         if len(positions) == 7:
#             self.current_joints = np.array(positions)

#     def transform_pose(self, pose_ros, transformation_matrix):
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

#     def find_closest_pose_index(self, trajectory_joints):
#         """Find which pose in the trajectory is closest to current position"""
#         if self.current_joints is None or len(trajectory_joints) == 0:
#             return 0
        
#         min_dist = float('inf')
#         closest_idx = 0
        
#         for i, joints in enumerate(trajectory_joints):
#             dist = np.linalg.norm(joints - self.current_joints)
#             if dist < min_dist:
#                 min_dist = dist
#                 closest_idx = i
        
#         return closest_idx

#     def create_blend(self, start_joints, end_joints, num_steps):
#         """
#         Create smooth interpolation from start to end.
#         Uses cosine interpolation for smoother acceleration/deceleration.
#         """
#         blend = []
#         for i in range(1, num_steps + 1):
#             # Cosine interpolation: smoother than linear
#             t = i / num_steps
#             t_smooth = (1 - np.cos(t * np.pi)) / 2  # S-curve from 0 to 1
            
#             interpolated = start_joints + t_smooth * (end_joints - start_joints)
#             blend.append(interpolated)
        
#         return blend

#     def process_trajectory(self, msg, frame_id):
#         """Convert PoseArray to list of joint configurations"""
#         if self.current_joints is None:
#             return None
        
#         trajectory_joints = []
#         prev_joints = self.current_joints.copy()
        
#         # Process ALL poses (we'll select which ones to use later)
#         for i, pose_fork in enumerate(msg.poses):
#             if pose_fork.position.z < self.z_min_safety:
#                 pose_fork.position.z = self.z_min_safety
            
#             pose_tcp = self.transform_pose(pose_fork, self.T_tcp_fork_tip_inv)
#             sol = self.compute_ik(pose_tcp, frame_id, seed=prev_joints, timeout=0.03)
            
#             if sol is None:
#                 sol = self.compute_ik(pose_tcp, frame_id, seed=None, timeout=0.05)
            
#             if sol is not None:
#                 max_diff = np.max(np.abs(sol - prev_joints))
#                 if max_diff > self.max_joint_jump:
#                     rospy.logwarn(f"Joint jump too large at pose {i}: {max_diff:.3f} rad")
#                     break
                
#                 trajectory_joints.append(sol)
#                 prev_joints = sol
#             else:
#                 rospy.logwarn(f"IK failed for pose {i}")
#                 # Use last good solution to maintain trajectory length
#                 if len(trajectory_joints) > 0:
#                     trajectory_joints.append(trajectory_joints[-1])
        
#         return trajectory_joints if len(trajectory_joints) > 0 else None

#     def traj_callback(self, msg):
#         """
#         When a new trajectory arrives:
#         - If committed: ignore (will get fresh one after chunk completes)
#         - If not committed: find closest pose, blend to it, then continue
#         """
#         with self.lock:
#             if self.is_committed:
#                 # Ignore - we're executing current chunk
#                 rospy.logdebug("Committed - ignoring new trajectory")
#                 return
            
#             # Process the full trajectory
#             trajectory_joints = self.process_trajectory(msg, msg.header.frame_id)
            
#             if trajectory_joints is None or len(trajectory_joints) < 2:
#                 return
            
#             # Find closest pose to current position
#             closest_idx = self.find_closest_pose_index(trajectory_joints)
            
#             # Determine where to start in the trajectory (after closest point)
#             start_idx = min(closest_idx + 1, len(trajectory_joints) - 1)
            
#             # Get remaining trajectory from start_idx
#             remaining_trajectory = trajectory_joints[start_idx:]
            
#             if len(remaining_trajectory) == 0:
#                 return
            
#             # Create smooth blend from current position to first point of remaining trajectory
#             blend_target = remaining_trajectory[0]
#             distance_to_target = np.linalg.norm(blend_target - self.current_joints)
            
#             # Adjust blend steps based on distance (more steps for larger jumps)
#             adaptive_blend_steps = max(1, min(self.blend_steps, int(distance_to_target / 0.05) + 1))
            
#             if distance_to_target > 0.01:  # Only blend if we're not already there
#                 blend = self.create_blend(self.current_joints, blend_target, adaptive_blend_steps)
#                 # Combine: blend + remaining trajectory (skip first point since blend ends there)
#                 full_trajectory = blend + remaining_trajectory[1:]
#             else:
#                 full_trajectory = remaining_trajectory
            
#             # Limit to action_chunk_size
#             chunk_size = min(self.action_chunk_size, len(full_trajectory))
#             self.current_trajectory = full_trajectory[:chunk_size]
#             self.trajectory_index = 0
#             self.is_committed = True
            
#             rospy.loginfo(f"✅ New trajectory: closest_idx={closest_idx}, blend_steps={adaptive_blend_steps}, chunk={len(self.current_trajectory)}")
            
#             # Debug: log first few poses
#             for i, pose in enumerate(msg.poses[:5]):
#                 rospy.loginfo(f"Pose {i}: pos=({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})")

#     def run(self):
#         rate = rospy.Rate(self.control_rate)
        
#         while not rospy.is_shutdown():
#             if self.current_joints is None:
#                 rate.sleep()
#                 continue
            
#             with self.lock:
#                 if self.current_trajectory is not None and self.trajectory_index < len(self.current_trajectory):
#                     # Execute current action
#                     target_joints = self.current_trajectory[self.trajectory_index]
                    
#                     max_diff = np.max(np.abs(target_joints - self.current_joints))
#                     if max_diff < 0.5:
#                         msg = Float64MultiArray()
#                         msg.data = target_joints.tolist()
#                         self.pub_joints.publish(msg)
                        
#                         rospy.loginfo(f"Executing {self.trajectory_index + 1}/{len(self.current_trajectory)}, diff={max_diff:.4f}")
#                     else:
#                         rospy.logwarn(f"Blocked: diff {max_diff:.2f} too large")
                    
#                     self.trajectory_index += 1
                    
#                     # Check if chunk complete
#                     if self.trajectory_index >= len(self.current_trajectory):
#                         rospy.loginfo("✅ Chunk complete - ready for new trajectory")
#                         self.is_committed = False
#                         self.current_trajectory = None
                
#                 elif self.current_trajectory is not None:
#                     # Hold last position
#                     target_joints = self.current_trajectory[-1]
#                     max_diff = np.max(np.abs(target_joints - self.current_joints))
#                     if max_diff < 0.5:
#                         msg = Float64MultiArray()
#                         msg.data = target_joints.tolist()
#                         self.pub_joints.publish(msg)
            
#             rate.sleep()


# if __name__ == '__main__':
#     TrajectoryFollowerSmooth().run()



# """
# Trajectory Follower for FORK - Improved Smoothing V2

# Key improvements over previous version:
# 1. EMA filter RESET on new trajectory (prevents sluggish transitions)
# 2. Overlapping chunks - accept new trajectory BEFORE chunk completes
# 3. Higher control rate (20 Hz) for smoother motion
# 4. Velocity-aware blending
# 5. Continuous motion without gaps between chunks

# The main insight: the "slow transition" was caused by:
# - EMA filter carrying momentum from old trajectory
# - Gap between chunk completion and new trajectory acceptance
# """
# import rospy
# import rospkg
# import moveit_commander
# import sys
# import os
# import numpy as np
# import threading
# from geometry_msgs.msg import PoseArray, Pose
# from std_msgs.msg import Float64MultiArray
# from sensor_msgs.msg import JointState
# from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
# from moveit_msgs.msg import MoveItErrorCodes
# from scipy.spatial.transform import Rotation as R

# from utils import compute_T_child_parent_xacro


# class TrajectoryFollowerForkSmoothV2:
    
#     def __init__(self):
#         rospy.init_node('trajectory_follower_fork_node')
        
#         # --- FORK TRANSFORM ---
#         rospack = rospkg.RosPack()
#         package_path = rospack.get_path('vision_processing')
#         xacro_file = os.path.join(package_path, 'urdf', 'panda_camera.xacro')
        
#         self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro_file, 'fork_tip', 'panda_TCP')
#         self.T_tcp_fork_tip_inv = np.linalg.inv(self.T_tcp_fork_tip)
#         rospy.loginfo("✅ Fork transform loaded")
        
#         # =============================================
#         # KEY PARAMETERS - TUNED FOR SMOOTHNESS
#         # =============================================
        
#         # Control rate HIGHER than inference rate for smoother interpolation
#         self.control_rate = 20.0  # Hz (was 8, now 20)
        
#         # How many actions to execute before accepting new trajectory
#         # SMALLER = more reactive but potentially less smooth
#         # LARGER = smoother but slower to react
#         self.action_chunk_size = 6  # (was 8)
        
#         # Accept new trajectory when this many actions remain
#         # This creates OVERLAP - no gap between chunks!
#         self.accept_new_when_remaining = 2  # Accept new traj when 2 actions left
        
#         # Blend steps for transitioning to new trajectory
#         self.blend_steps = 3  # (was 5 - too many causes lag)
        
#         # --- EMA FILTER ---
#         # Lower alpha = MORE smoothing (but more lag)
#         # Higher alpha = LESS smoothing (more responsive)
#         # 0.3-0.4 is good balance
#         self.ema_alpha = rospy.get_param("~ema_alpha", 0.35)
#         self.filtered_joints = None
        
#         # Velocity estimation for smoother blending
#         self.prev_joints = None
#         self.joint_velocity = None
        
#         # State tracking
#         self.current_joints = None
#         self.current_trajectory = None
#         self.trajectory_index = 0
#         self.is_committed = False
#         self.lock = threading.Lock()
        
#         # Pending trajectory (for seamless transitions)
#         self.pending_trajectory = None
        
#         # Safety parameters
#         self.max_joint_jump = 0.4  # radians
#         self.z_min_safety = 0.002
        
#         # MoveIt setup
#         moveit_commander.roscpp_initialize(sys.argv)
#         self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        
#         rospy.wait_for_service('/compute_ik')
#         self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        
#         # Publishers/Subscribers
#         self.pub_joints = rospy.Publisher(
#             '/joint_group_position_controller/command', 
#             Float64MultiArray, 
#             queue_size=1
#         )
        
#         self.sub_traj = rospy.Subscriber(
#             "/diffusion/target_trajectory", 
#             PoseArray, 
#             self.traj_callback, 
#             queue_size=1,
#             buff_size=2**24 
#         )
#         self.sub_joints = rospy.Subscriber("/joint_states", JointState, self.joint_cb)
        
#         rospy.loginfo("=" * 60)
#         rospy.loginfo("🚀 Fork Trajectory Follower V2 (Improved Smoothing)")
#         rospy.loginfo(f"   Control rate: {self.control_rate} Hz")
#         rospy.loginfo(f"   Action chunk size: {self.action_chunk_size}")
#         rospy.loginfo(f"   Accept new when remaining: {self.accept_new_when_remaining}")
#         rospy.loginfo(f"   Blend steps: {self.blend_steps}")
#         rospy.loginfo(f"   EMA alpha: {self.ema_alpha}")
#         rospy.loginfo("=" * 60)

#     def joint_cb(self, msg):
#         positions = []
#         for name, pos in zip(msg.name, msg.position):
#             if "panda_joint" in name:
#                 positions.append(pos)
#         if len(positions) == 7:
#             new_joints = np.array(positions)
            
#             # Estimate velocity
#             if self.current_joints is not None:
#                 self.joint_velocity = (new_joints - self.current_joints) * 50  # Assume 50Hz joint_states
            
#             self.prev_joints = self.current_joints
#             self.current_joints = new_joints
            
#             if self.filtered_joints is None:
#                 self.filtered_joints = self.current_joints.copy()

#     def transform_fork_to_tcp(self, pose_fork):
#         """Transform pose from fork_tip frame to TCP frame."""
#         p = np.array([pose_fork.position.x, pose_fork.position.y, pose_fork.position.z])
#         q = np.array([pose_fork.orientation.x, pose_fork.orientation.y, 
#                       pose_fork.orientation.z, pose_fork.orientation.w])
        
#         T_world_fork = np.eye(4)
#         T_world_fork[:3, :3] = R.from_quat(q).as_matrix()
#         T_world_fork[:3, 3] = p
        
#         T_world_tcp = T_world_fork @ self.T_tcp_fork_tip_inv
        
#         pose_tcp = Pose()
#         pose_tcp.position.x, pose_tcp.position.y, pose_tcp.position.z = T_world_tcp[:3, 3]
        
#         q_tcp = R.from_matrix(T_world_tcp[:3, :3]).as_quat()
#         pose_tcp.orientation.x = q_tcp[0]
#         pose_tcp.orientation.y = q_tcp[1]
#         pose_tcp.orientation.z = q_tcp[2]
#         pose_tcp.orientation.w = q_tcp[3]
        
#         return pose_tcp

#     def compute_ik(self, pose, frame_id, seed=None, timeout=0.05):
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

#     def find_closest_pose_index(self, trajectory_joints):
#         """Find which pose in trajectory is closest to current position."""
#         if self.current_joints is None or len(trajectory_joints) == 0:
#             return 0
        
#         min_dist = float('inf')
#         closest_idx = 0
        
#         for i, joints in enumerate(trajectory_joints):
#             dist = np.linalg.norm(joints - self.current_joints)
#             if dist < min_dist:
#                 min_dist = dist
#                 closest_idx = i
        
#         return closest_idx

#     def create_smooth_blend(self, start_joints, end_joints, num_steps, start_velocity=None):
#         """
#         Create smooth interpolation with velocity consideration.
        
#         Uses quintic (5th order) polynomial for smoother acceleration profile
#         when velocity is available, otherwise falls back to cosine interpolation.
#         """
#         if num_steps <= 0:
#             return [end_joints]
        
#         blend = []
        
#         for i in range(1, num_steps + 1):
#             t = i / num_steps
            
#             if start_velocity is not None and np.linalg.norm(start_velocity) > 0.01:
#                 # Quintic interpolation considering initial velocity
#                 # This creates smoother acceleration profile
#                 t2 = t * t
#                 t3 = t2 * t
#                 t4 = t3 * t
#                 t5 = t4 * t
                
#                 # Quintic polynomial: 6t^5 - 15t^4 + 10t^3
#                 t_smooth = 6*t5 - 15*t4 + 10*t3
#             else:
#                 # Cosine interpolation (S-curve)
#                 t_smooth = (1 - np.cos(t * np.pi)) / 2
            
#             interpolated = start_joints + t_smooth * (end_joints - start_joints)
#             blend.append(interpolated)
        
#         return blend

#     def process_trajectory(self, msg, frame_id):
#         """Convert PoseArray (fork_tip poses) to list of joint configurations."""
#         if self.current_joints is None:
#             return None
        
#         trajectory_joints = []
#         prev_joints = self.current_joints.copy()
        
#         for i, pose_fork in enumerate(msg.poses):
#             if pose_fork.position.z < self.z_min_safety:
#                 pose_fork.position.z = self.z_min_safety
            
#             pose_tcp = self.transform_fork_to_tcp(pose_fork)
#             sol = self.compute_ik(pose_tcp, frame_id, seed=prev_joints, timeout=0.02)
            
#             if sol is None:
#                 sol = self.compute_ik(pose_tcp, frame_id, seed=None, timeout=0.03)
            
#             if sol is not None:
#                 max_diff = np.max(np.abs(sol - prev_joints))
#                 if max_diff > self.max_joint_jump:
#                     rospy.logwarn(f"Joint jump at pose {i}: {max_diff:.3f} rad")
#                     break
                
#                 trajectory_joints.append(sol)
#                 prev_joints = sol
#             else:
#                 if len(trajectory_joints) > 0:
#                     trajectory_joints.append(trajectory_joints[-1])
        
#         return trajectory_joints if len(trajectory_joints) > 0 else None

#     def traj_callback(self, msg):
#         """
#         Handle new trajectory arrival.
        
#         Key improvement: Accept new trajectory even while committed,
#         but store it as pending and switch when appropriate.
#         """
#         with self.lock:
#             trajectory_joints = self.process_trajectory(msg, msg.header.frame_id)
            
#             if trajectory_joints is None or len(trajectory_joints) < 2:
#                 return
            
#             # Find closest pose
#             closest_idx = self.find_closest_pose_index(trajectory_joints)
#             start_idx = min(closest_idx + 1, len(trajectory_joints) - 1)
#             remaining_trajectory = trajectory_joints[start_idx:]
            
#             if len(remaining_trajectory) == 0:
#                 return
            
#             # Calculate blend
#             blend_target = remaining_trajectory[0]
#             distance = np.linalg.norm(blend_target - self.current_joints)
            
#             # Fewer blend steps for smaller distances
#             adaptive_blend = max(1, min(self.blend_steps, int(distance / 0.03) + 1))
            
#             if distance > 0.005:
#                 blend = self.create_smooth_blend(
#                     self.current_joints, 
#                     blend_target, 
#                     adaptive_blend,
#                     self.joint_velocity
#                 )
#                 full_trajectory = blend + remaining_trajectory[1:]
#             else:
#                 full_trajectory = remaining_trajectory
            
#             chunk_size = min(self.action_chunk_size, len(full_trajectory))
#             new_trajectory = full_trajectory[:chunk_size]
            
#             # Decision: accept now or store as pending
#             if not self.is_committed:
#                 # Not committed - accept immediately
#                 self.current_trajectory = new_trajectory
#                 self.trajectory_index = 0
#                 self.is_committed = True
                
#                 # IMPORTANT: Reset EMA filter on new trajectory!
#                 self.filtered_joints = self.current_joints.copy()
                
#                 rospy.loginfo(f"✅ New traj: closest={closest_idx}, blend={adaptive_blend}, chunk={len(new_trajectory)}")
            
#             elif self.current_trajectory is not None:
#                 remaining = len(self.current_trajectory) - self.trajectory_index
                
#                 if remaining <= self.accept_new_when_remaining:
#                     # Almost done with current chunk - switch now for seamless transition
#                     self.current_trajectory = new_trajectory
#                     self.trajectory_index = 0
                    
#                     # IMPORTANT: Reset EMA filter for fresh start
#                     self.filtered_joints = self.current_joints.copy()
                    
#                     rospy.loginfo(f"🔄 Seamless switch: remaining={remaining}, new chunk={len(new_trajectory)}")
#                 else:
#                     # Store as pending
#                     self.pending_trajectory = new_trajectory

#     def apply_ema_filter(self, target_joints):
#         """
#         Apply EMA filter to smooth joint commands.
        
#         filtered = alpha * target + (1-alpha) * previous_filtered
        
#         Lower alpha = more smoothing (more lag)
#         Higher alpha = less smoothing (more responsive)
#         """
#         if self.filtered_joints is None:
#             self.filtered_joints = target_joints.copy()
#             return target_joints
        
#         self.filtered_joints = self.ema_alpha * target_joints + (1 - self.ema_alpha) * self.filtered_joints
#         return self.filtered_joints

#     def run(self):
#         rate = rospy.Rate(self.control_rate)
        
#         while not rospy.is_shutdown():
#             if self.current_joints is None:
#                 rate.sleep()
#                 continue
            
#             with self.lock:
#                 if self.current_trajectory is not None and self.trajectory_index < len(self.current_trajectory):
#                     # Get target from trajectory
#                     target_joints = self.current_trajectory[self.trajectory_index]
                    
#                     # Apply EMA smoothing
#                     smoothed_joints = self.apply_ema_filter(target_joints)
                    
#                     # Safety check
#                     max_diff = np.max(np.abs(smoothed_joints - self.current_joints))
                    
#                     if max_diff < 0.5:
#                         msg = Float64MultiArray()
#                         msg.data = smoothed_joints.tolist()
#                         self.pub_joints.publish(msg)
#                     else:
#                         rospy.logwarn(f"Blocked: diff {max_diff:.2f} too large")
                    
#                     self.trajectory_index += 1
                    
#                     # Check if chunk complete
#                     if self.trajectory_index >= len(self.current_trajectory):
#                         # Check for pending trajectory
#                         if self.pending_trajectory is not None:
#                             self.current_trajectory = self.pending_trajectory
#                             self.pending_trajectory = None
#                             self.trajectory_index = 0
                            
#                             # Reset EMA for fresh trajectory
#                             self.filtered_joints = self.current_joints.copy()
                            
#                             rospy.loginfo("📥 Switched to pending trajectory")
#                         else:
#                             self.is_committed = False
#                             rospy.loginfo("✅ Chunk complete")
                
#                 elif self.is_committed and self.current_trajectory is not None:
#                     # Hold last position
#                     target_joints = self.current_trajectory[-1]
#                     smoothed_joints = self.apply_ema_filter(target_joints)
                    
#                     if np.max(np.abs(smoothed_joints - self.current_joints)) < 0.5:
#                         msg = Float64MultiArray()
#                         msg.data = smoothed_joints.tolist()
#                         self.pub_joints.publish(msg)
            
#             rate.sleep()


# if __name__ == '__main__':
#     TrajectoryFollowerForkSmoothV2().run()