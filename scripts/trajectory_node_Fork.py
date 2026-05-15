#!/usr/bin/env python3
"""
Trajectory Follower for FORK - Continuous Blending Approach (OPTIMIZED)

Optimization: 
Instead of computing Inverse Kinematics (IK) for the entire trajectory array 
(which blocks the thread and causes massive latency/lag), this version computes 
the closest point in pure Cartesian space, selects the Lookahead target, 
and computes IK ONLY ONCE per control cycle.
"""
import rospy
import rospkg
import moveit_commander
import sys 
import os
import numpy as np
import threading
import tf
import tf.transformations as tft
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
        
        self.control_rate = 15.0  # Hz
        self.lookahead_steps = 3  # Target the 3rd waypoint ahead
        self.blend_rate = 0.25
        self.max_joint_velocity = 0.7  # rad/s per joint
        self.max_joint_jump = 0.5
        
        # State
        self.current_joints = None
        self.target_joints = None  
        self.last_command_joints = None
        self.latest_msg = None     # We now store the raw ROS message
        self.lock = threading.Lock()
        
        # TF Listener pour trouver la position Cartesienne instantanée
        self.tf_listener = tf.TransformListener()
        
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
            queue_size=1
        )
        self.sub_joints = rospy.Subscriber("/joint_states", JointState, self.joint_cb)
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("🚀 Fork Trajectory Follower (ULTRA-FAST Cartesian Blending)")
        rospy.loginfo("=" * 60)

    def joint_cb(self, msg):
        positions = []
        for name, pos in zip(msg.name, msg.position):
            if "panda_joint" in name:
                positions.append(pos)
        if len(positions) == 7:
            self.current_joints = np.array(positions)
            if self.target_joints is None:
                self.target_joints = self.current_joints.copy()
            if self.last_command_joints is None:
                self.last_command_joints = self.current_joints.copy()

    def transform_fork_to_tcp(self, pose_fork):
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

    def traj_callback(self, msg):
        """
        Stocke simplement le message. AUCUN CALCUL BLOQUANT ICI.
        """
        with self.lock:
            if len(msg.poses) > 0:
                self.latest_msg = msg

    def compute_command(self):
        if self.current_joints is None or self.latest_msg is None:
            return None
        
        # 1. Calculer la position Cartésienne ACTUELLE de la fourchette via TF
        try:
            (trans, rot) = self.tf_listener.lookupTransform('world', 'panda_hand_tcp', rospy.Time(0))
            T_world_tcp = tft.quaternion_matrix(rot)
            T_world_tcp[:3, 3] = trans
            T_world_fork = T_world_tcp @ self.T_tcp_fork_tip
            current_fork_pos = T_world_fork[:3, 3]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return self.target_joints # Fallback si TF échoue

        # 2. Trouver l'index le plus proche en espace Cartésien (Calcul matriciel ultra rapide)
        with self.lock:
            msg = self.latest_msg
            
        positions = np.array([[p.position.x, p.position.y, p.position.z] for p in msg.poses])
        distances = np.linalg.norm(positions - current_fork_pos, axis=1)
        closest_idx = np.argmin(distances)

        # 3. Appliquer le Lookahead
        target_idx = min(closest_idx + self.lookahead_steps, len(msg.poses) - 1)
        target_pose_fork = msg.poses[target_idx]

        # 4. Calculer l'IK UNIQUEMENT pour ce point cible précis (Gain de perf massif)
        pose_tcp = self.transform_fork_to_tcp(target_pose_fork)
        ik_sol = self.compute_ik(pose_tcp, msg.header.frame_id, seed=self.current_joints)

        # 5. Filtrage de sécurité
        if ik_sol is not None:
            if np.max(np.abs(ik_sol - self.current_joints)) <= self.max_joint_jump:
                target_lookahead_joints = ik_sol
            else:
                target_lookahead_joints = self.target_joints
        else:
            target_lookahead_joints = self.target_joints
        
        # 6. Continuous Blending
        self.target_joints = (1 - self.blend_rate) * self.target_joints + self.blend_rate * target_lookahead_joints
        
        # 7. Limite de Vélocité (Fixed to prevent stuttering)
        direction = self.target_joints - self.last_command_joints
        distance = np.linalg.norm(direction)
        if distance < 0.001:
            command = self.target_joints
        else:
            max_step = self.max_joint_velocity / self.control_rate
            if distance > max_step:
                direction = direction / distance * max_step
                command = self.last_command_joints + direction
            else:
                command = self.target_joints
        
        self.last_command_joints = command.copy()
        
        return command

    def run(self):
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            if self.current_joints is None:
                rate.sleep()
                continue
            
            command = self.compute_command()
            
            if command is not None:
                msg = Float64MultiArray()
                msg.data = command.tolist()
                self.pub_joints.publish(msg)
            
            rate.sleep()

if __name__ == '__main__':
    TrajectoryFollowerContinuous().run()


# #!/usr/bin/env python3
# """
# Trajectory Follower for FORK - Continuous Blending Approach (OPTIMIZED)

# Optimization: 
# Instead of computing Inverse Kinematics (IK) for the entire trajectory array 
# (which blocks the thread and causes massive latency/lag), this version computes 
# the closest point in pure Cartesian space, selects the Lookahead target, 
# and computes IK ONLY ONCE per control cycle.
# """
# import rospy
# import rospkg
# import moveit_commander
# import sys 
# import os
# import numpy as np
# import threading
# import tf
# import tf.transformations as tft
# from geometry_msgs.msg import PoseArray, Pose
# from std_msgs.msg import Float64MultiArray, Float32
# from sensor_msgs.msg import JointState
# from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
# from moveit_msgs.msg import MoveItErrorCodes
# from scipy.spatial.transform import Rotation as R

# from utils import compute_T_child_parent_xacro


# class TrajectoryFollowerContinuous:
    
#     def __init__(self):
#         rospy.init_node('trajectory_follower_fork_continuous')
        
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
        
#         self.control_rate = 30.0  # Hz
#         self.lookahead_steps = 3  # Target the 3rd waypoint ahead
#         self.blend_rate = 0.15
#         self.blend_rate_near = 0.50  # More aggressive when close to target
#         self.proximity_threshold = 0.030  # 30 mm — switch to aggressive blend
#         self.max_joint_velocity = 0.7  # rad/s per joint
#         self.max_joint_jump = 0.5
        
#         # Distance from condition node
#         self.fork_food_distance = float('inf')
        
#         # State
#         self.current_joints = None
#         self.target_joints = None  
#         self.latest_msg = None     # We now store the raw ROS message
#         self.lock = threading.Lock()
        
#         # TF Listener pour trouver la position Cartesienne instantanée
#         self.tf_listener = tf.TransformListener()
        
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
#             queue_size=1
#         )
#         self.sub_joints = rospy.Subscriber("/joint_states", JointState, self.joint_cb)
#         self.sub_dist = rospy.Subscriber("/vision/fork_food_distance", Float32, self.dist_cb, queue_size=1)
        
#         rospy.loginfo("=" * 60)
#         rospy.loginfo("🚀 Fork Trajectory Follower (ULTRA-FAST Cartesian Blending)")
#         rospy.loginfo("=" * 60)

#     def joint_cb(self, msg):
#         positions = []
#         for name, pos in zip(msg.name, msg.position):
#             if "panda_joint" in name:
#                 positions.append(pos)
#         if len(positions) == 7:
#             self.current_joints = np.array(positions)
#             if self.target_joints is None:
#                 self.target_joints = self.current_joints.copy()

#     def dist_cb(self, msg):
#         self.fork_food_distance = msg.data

#     def transform_fork_to_tcp(self, pose_fork):
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

#     def compute_ik(self, pose, frame_id, seed=None, timeout=0.03):
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
#         """
#         Stocke simplement le message. AUCUN CALCUL BLOQUANT ICI.
#         """
#         with self.lock:
#             if len(msg.poses) > 0:
#                 self.latest_msg = msg

#     def compute_command(self):
#         if self.current_joints is None or self.latest_msg is None:
#             return None
        
#         # 1. Calculer la position Cartésienne ACTUELLE de la fourchette via TF
#         try:
#             (trans, rot) = self.tf_listener.lookupTransform('world', 'panda_hand_tcp', rospy.Time(0))
#             T_world_tcp = tft.quaternion_matrix(rot)
#             T_world_tcp[:3, 3] = trans
#             T_world_fork = T_world_tcp @ self.T_tcp_fork_tip
#             current_fork_pos = T_world_fork[:3, 3]
#         except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
#             return self.target_joints # Fallback si TF échoue

#         # 2. Trouver l'index le plus proche en espace Cartésien (Calcul matriciel ultra rapide)
#         with self.lock:
#             msg = self.latest_msg
            
#         positions = np.array([[p.position.x, p.position.y, p.position.z] for p in msg.poses])
#         distances = np.linalg.norm(positions - current_fork_pos, axis=1)
#         closest_idx = np.argmin(distances)

#         # 3. Appliquer le Lookahead
#         target_idx = min(closest_idx + self.lookahead_steps, len(msg.poses) - 1)
#         target_pose_fork = msg.poses[target_idx]

#         # 4. Calculer l'IK UNIQUEMENT pour ce point cible précis (Gain de perf massif)
#         pose_tcp = self.transform_fork_to_tcp(target_pose_fork)
#         ik_sol = self.compute_ik(pose_tcp, msg.header.frame_id, seed=self.current_joints)

#         # 5. Filtrage de sécurité
#         if ik_sol is not None:
#             if np.max(np.abs(ik_sol - self.current_joints)) <= self.max_joint_jump:
#                 target_lookahead_joints = ik_sol
#             else:
#                 target_lookahead_joints = self.target_joints
#         else:
#             target_lookahead_joints = self.target_joints
        
#         # 6. Adaptive Blending — faster near target to maintain momentum
#         if self.fork_food_distance < self.proximity_threshold:
#             rate = self.blend_rate_near
#         else:
#             rate = self.blend_rate
#         self.target_joints = (1 - rate) * self.target_joints + rate * target_lookahead_joints
        
#         # 7. Limite de Vélocité
#         direction = self.target_joints - self.current_joints
#         distance = np.linalg.norm(direction)
#         if distance < 0.001:
#             return self.target_joints
            
#         max_step = self.max_joint_velocity / self.control_rate
#         if distance > max_step:
#             direction = direction / distance * max_step
#             command = self.current_joints + direction
#         else:
#             command = self.target_joints
        
#         return command

#     def run(self):
#         rate = rospy.Rate(self.control_rate)
#         while not rospy.is_shutdown():
#             if self.current_joints is None:
#                 rate.sleep()
#                 continue
            
#             command = self.compute_command()
            
#             if command is not None:
#                 msg = Float64MultiArray()
#                 msg.data = command.tolist()
#                 self.pub_joints.publish(msg)
            
#             rate.sleep()

# if __name__ == '__main__':
#     TrajectoryFollowerContinuous().run()