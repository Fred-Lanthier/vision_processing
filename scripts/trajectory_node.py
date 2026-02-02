# #!/usr/bin/env python3
# """
# Trajectory Follower with Smooth Blending for panda_hand_tcp

# Key features:
# 1. Temporal commitment: Execute N actions before accepting new trajectory
# 2. Skip to closest pose: Find where we are in new trajectory
# 3. Smooth blending: Cosine interpolation to avoid jerky transitions
# 4. Direct TCP control: No fork tip transform needed
# """
# import rospy
# import moveit_commander
# import sys
# import numpy as np
# import threading
# from geometry_msgs.msg import PoseArray, Pose
# from std_msgs.msg import Float64MultiArray
# from sensor_msgs.msg import JointState
# from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
# from moveit_msgs.msg import MoveItErrorCodes


# class TrajectoryFollowerSmooth:
    
#     def __init__(self):
#         rospy.init_node('trajectory_follower_node')
        
#         # =============================================
#         # KEY PARAMETERS
#         # =============================================
#         self.control_rate = 10.0  # Hz - matches inference rate
#         self.action_chunk_size = 8  # Execute 8 actions before re-planning
#         self.blend_steps = 3  # Number of steps to smoothly blend to new trajectory
        
#         # State tracking
#         self.current_joints = None
#         self.current_trajectory = None  # List of joint configurations to execute
#         self.trajectory_index = 0  # Current position in the chunk
#         self.is_committed = False  # Are we committed to current trajectory?
#         self.lock = threading.Lock()
        
#         # Safety parameters
#         self.max_joint_jump = 0.5  # radians - reject jumps larger than this
#         self.z_min_safety = 0.02  # minimum Z height for TCP
        
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
        
#         rospy.loginfo("🚀 Trajectory Follower (SMOOTH BLEND) Ready")
#         rospy.loginfo(f"   Action chunk size: {self.action_chunk_size}")
#         rospy.loginfo(f"   Blend steps: {self.blend_steps}")
#         rospy.loginfo(f"   Control rate: {self.control_rate} Hz")

#     def joint_cb(self, msg):
#         """Store current joint positions"""
#         positions = []
#         for name, pos in zip(msg.name, msg.position):
#             if "panda_joint" in name:
#                 positions.append(pos)
#         if len(positions) == 7:
#             self.current_joints = np.array(positions)

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

#     def find_closest_pose_index(self, trajectory_joints):
#         """Find which pose in the trajectory is closest to current joint position"""
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
#             # Cosine interpolation: S-curve from 0 to 1
#             t = i / num_steps
#             t_smooth = (1 - np.cos(t * np.pi)) / 2
            
#             interpolated = start_joints + t_smooth * (end_joints - start_joints)
#             blend.append(interpolated)
        
#         return blend

#     def process_trajectory(self, msg, frame_id):
#         """Convert PoseArray to list of joint configurations"""
#         if self.current_joints is None:
#             return None
        
#         trajectory_joints = []
#         prev_joints = self.current_joints.copy()
        
#         # Process ALL poses from the prediction
#         for i, pose in enumerate(msg.poses):
#             # Safety: enforce minimum Z
#             if pose.position.z < self.z_min_safety:
#                 pose.position.z = self.z_min_safety
            
#             # Compute IK (pose is already in TCP frame, no transform needed)
#             sol = self.compute_ik(pose, frame_id, seed=prev_joints, timeout=0.03)
            
#             if sol is None:
#                 # Fallback: try without seed
#                 sol = self.compute_ik(pose, frame_id, seed=None, timeout=0.05)
            
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
            
#             # Adaptive blend steps: more steps for larger jumps
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
#                     # Execute current action in the chunk
#                     target_joints = self.current_trajectory[self.trajectory_index]
                    
#                     # Safety check
#                     max_diff = np.max(np.abs(target_joints - self.current_joints))
#                     if max_diff < 0.5:
#                         msg = Float64MultiArray()
#                         msg.data = target_joints.tolist()
#                         self.pub_joints.publish(msg)
                        
#                         rospy.loginfo(f"Executing {self.trajectory_index + 1}/{len(self.current_trajectory)}, diff={max_diff:.4f}")
#                     else:
#                         rospy.logwarn(f"Blocked: diff {max_diff:.2f} too large")
                    
#                     # Move to next action
#                     self.trajectory_index += 1
                    
#                     # Check if chunk complete
#                     if self.trajectory_index >= len(self.current_trajectory):
#                         rospy.loginfo("✅ Chunk complete - ready for new trajectory")
#                         self.is_committed = False
#                         self.current_trajectory = None
                
#                 elif self.current_trajectory is not None:
#                     # Hold last position while waiting for new trajectory
#                     target_joints = self.current_trajectory[-1]
#                     max_diff = np.max(np.abs(target_joints - self.current_joints))
#                     if max_diff < 0.5:
#                         msg = Float64MultiArray()
#                         msg.data = target_joints.tolist()
#                         self.pub_joints.publish(msg)
            
#             rate.sleep()


# if __name__ == '__main__':
#     TrajectoryFollowerSmooth().run()










#!/usr/bin/env python3
"""
Trajectory Follower with Smooth Blending + Joint EMA Filter

Two layers of smoothing:
1. Smooth blending between trajectory chunks (from previous version)
2. Exponential Moving Average on joint commands (final smoothing)

The EMA filter reduces high-frequency jitter without adding much lag.
"""
import rospy
import moveit_commander
import sys
import numpy as np
import threading
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_msgs.msg import MoveItErrorCodes


class TrajectoryFollowerSmooth:
    
    def __init__(self):
        rospy.init_node('trajectory_follower_node')
        
        # =============================================
        # KEY PARAMETERS
        # =============================================
        self.control_rate = 10.0  # Hz - matches inference rate
        self.action_chunk_size = 8  # Execute 8 actions before re-planning
        self.blend_steps = 5  # Steps to blend to new trajectory
        
        # --- EMA FILTER PARAMETERS ---
        # alpha = 0.0 → no filtering (raw commands)
        # alpha = 0.3 → light smoothing (recommended)
        # alpha = 0.5 → medium smoothing
        # alpha = 0.8 → heavy smoothing (more lag)
        self.ema_alpha = rospy.get_param("~ema_alpha", 0.5)
        self.filtered_joints = None
        
        # State tracking
        self.current_joints = None
        self.current_trajectory = None
        self.trajectory_index = 0
        self.is_committed = False
        self.lock = threading.Lock()
        
        # Safety parameters
        self.max_joint_jump = 0.5
        self.z_min_safety = 0.02
        
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
        rospy.loginfo("🚀 Trajectory Follower (Smooth + EMA Filter)")
        rospy.loginfo(f"   Action chunk size: {self.action_chunk_size}")
        rospy.loginfo(f"   Blend steps: {self.blend_steps}")
        rospy.loginfo(f"   EMA alpha: {self.ema_alpha}")
        rospy.loginfo("=" * 60)

    def joint_cb(self, msg):
        positions = []
        for name, pos in zip(msg.name, msg.position):
            if "panda_joint" in name:
                positions.append(pos)
        if len(positions) == 7:
            self.current_joints = np.array(positions)
            # Initialize filter state
            if self.filtered_joints is None:
                self.filtered_joints = self.current_joints.copy()

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
        """Cosine interpolation for smooth blending"""
        blend = []
        for i in range(1, num_steps + 1):
            t = i / num_steps
            t_smooth = (1 - np.cos(t * np.pi)) / 2  # S-curve
            interpolated = start_joints + t_smooth * (end_joints - start_joints)
            blend.append(interpolated)
        return blend

    def process_trajectory(self, msg, frame_id):
        if self.current_joints is None:
            return None
        
        trajectory_joints = []
        prev_joints = self.current_joints.copy()
        
        for i, pose in enumerate(msg.poses):
            if pose.position.z < self.z_min_safety:
                pose.position.z = self.z_min_safety
            
            sol = self.compute_ik(pose, frame_id, seed=prev_joints, timeout=0.03)
            
            if sol is None:
                sol = self.compute_ik(pose, frame_id, seed=None, timeout=0.05)
            
            if sol is not None:
                max_diff = np.max(np.abs(sol - prev_joints))
                if max_diff > self.max_joint_jump:
                    rospy.logwarn(f"Joint jump too large at pose {i}: {max_diff:.3f} rad")
                    break
                
                trajectory_joints.append(sol)
                prev_joints = sol
            else:
                rospy.logwarn(f"IK failed for pose {i}")
                if len(trajectory_joints) > 0:
                    trajectory_joints.append(trajectory_joints[-1])
        
        return trajectory_joints if len(trajectory_joints) > 0 else None

    def traj_callback(self, msg):
        with self.lock:
            if self.is_committed:
                rospy.logdebug("Committed - ignoring new trajectory")
                return
            
            trajectory_joints = self.process_trajectory(msg, msg.header.frame_id)
            
            if trajectory_joints is None or len(trajectory_joints) < 2:
                return
            
            closest_idx = self.find_closest_pose_index(trajectory_joints)
            start_idx = min(closest_idx + 1, len(trajectory_joints) - 1)
            remaining_trajectory = trajectory_joints[start_idx:]
            
            if len(remaining_trajectory) == 0:
                return
            
            blend_target = remaining_trajectory[0]
            distance_to_target = np.linalg.norm(blend_target - self.current_joints)
            adaptive_blend_steps = max(1, min(self.blend_steps, int(distance_to_target / 0.05) + 1))
            
            if distance_to_target > 0.01:
                blend = self.create_blend(self.current_joints, blend_target, adaptive_blend_steps)
                full_trajectory = blend + remaining_trajectory[1:]
            else:
                full_trajectory = remaining_trajectory
            
            chunk_size = min(self.action_chunk_size, len(full_trajectory))
            self.current_trajectory = full_trajectory[:chunk_size]
            self.trajectory_index = 0
            self.is_committed = True
            
            rospy.loginfo(f"✅ New trajectory: closest={closest_idx}, blend={adaptive_blend_steps}, chunk={len(self.current_trajectory)}")

    def apply_ema_filter(self, target_joints):
        """
        Apply Exponential Moving Average filter to joint commands.
        
        filtered = alpha * filtered_prev + (1 - alpha) * target
        
        Lower alpha = more smoothing (but more lag)
        Higher alpha = less smoothing (faster response)
        """
        if self.filtered_joints is None:
            self.filtered_joints = target_joints.copy()
            return target_joints
        
        # EMA formula: new = alpha * target + (1-alpha) * old
        self.filtered_joints = self.ema_alpha * target_joints + (1 - self.ema_alpha) * self.filtered_joints
        
        return self.filtered_joints

    def run(self):
        rate = rospy.Rate(self.control_rate)
        
        while not rospy.is_shutdown():
            if self.current_joints is None:
                rate.sleep()
                continue
            
            with self.lock:
                if self.current_trajectory is not None and self.trajectory_index < len(self.current_trajectory):
                    # Get target from trajectory
                    target_joints = self.current_trajectory[self.trajectory_index]
                    
                    # Apply EMA filter for additional smoothing
                    smoothed_joints = self.apply_ema_filter(target_joints)
                    
                    # Safety check
                    max_diff = np.max(np.abs(smoothed_joints - self.current_joints))
                    if max_diff < 0.5:
                        msg = Float64MultiArray()
                        msg.data = smoothed_joints.tolist()
                        self.pub_joints.publish(msg)
                        
                        rospy.loginfo(f"Exec {self.trajectory_index + 1}/{len(self.current_trajectory)}, diff={max_diff:.4f}")
                    else:
                        rospy.logwarn(f"Blocked: diff {max_diff:.2f} too large")
                    
                    self.trajectory_index += 1
                    
                    if self.trajectory_index >= len(self.current_trajectory):
                        rospy.loginfo("✅ Chunk complete")
                        self.is_committed = False
                        self.current_trajectory = None
                
                elif self.current_trajectory is not None:
                    # Hold last position with filtering
                    target_joints = self.current_trajectory[-1]
                    smoothed_joints = self.apply_ema_filter(target_joints)
                    max_diff = np.max(np.abs(smoothed_joints - self.current_joints))
                    if max_diff < 0.5:
                        msg = Float64MultiArray()
                        msg.data = smoothed_joints.tolist()
                        self.pub_joints.publish(msg)
            
            rate.sleep()


if __name__ == '__main__':
    TrajectoryFollowerSmooth().run()