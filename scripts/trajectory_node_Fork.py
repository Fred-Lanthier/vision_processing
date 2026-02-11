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