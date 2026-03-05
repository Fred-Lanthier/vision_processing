#!/usr/bin/env python3
"""
Trajectory Follower for FORK - With Face Tracking Handoff
==========================================================

Two modes:
  1. INFERENCE  — follows the diffusion policy trajectory (same as before)
  2. FACE_TRACKING — follows the face/mouth target position

The switch happens when the fork tip's Y coordinate (in world frame)
crosses a configurable threshold. This is a ONE-WAY switch:
once you enter FACE_TRACKING, you stay there.

Why Y axis?
  In most feeding setups, Y points toward the user. So "Y > threshold"
  means "the fork is close enough to the user to start tracking their mouth."
"""
import rospy
import rospkg
import moveit_commander
import sys
import os
import numpy as np
import threading
import tf as ros_tf
import tf.transformations as tft
from geometry_msgs.msg import PoseArray, Pose, PointStamped
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
        rospy.loginfo("Fork transform loaded")

        # =============================================
        # KEY PARAMETERS
        # =============================================

        self.control_rate = 30.0  # Hz

        # -- Blending (same as before) --
        self.lookahead_steps = 3
        self.blend_rate = 0.15
        self.max_joint_velocity = 0.7  # rad/s

        # -- Safety --
        self.max_joint_jump = 0.5
        self.z_min_safety = 0.000

        # =============================================
        # MODE SWITCHING PARAMETERS
        # =============================================

        # Y threshold in world frame (meters).
        # When fork tip Y > this value, switch to face tracking.
        # Tune this to your setup geometry.
        self.y_threshold = rospy.get_param("~y_threshold", 0.65)

        # In face tracking mode, approach the mouth but stop this far away (meters).
        # Prevents the fork from poking the user's face.
        self.face_approach_offset = rospy.get_param("~face_approach_offset", 0.05)

        # Blend rate for face tracking (can be different from inference mode).
        # Slower = smoother approach to mouth.
        self.face_blend_rate = rospy.get_param("~face_blend_rate", 0.08)

        # =============================================
        # STATE
        # =============================================

        # Mode: "INFERENCE" or "FACE_TRACKING"
        self.mode = "INFERENCE"

        self.current_joints = None
        self.target_joints = None
        self.latest_trajectory = None
        self.trajectory_timestamp = None

        # Face tracking state
        self.face_target_world = None          # (3,) position of mouth in world frame
        self.face_target_timestamp = None
        self.face_timeout = rospy.Duration(1.0)  # ignore stale face detections

        # Orientation to hold during face tracking.
        # We freeze the fork orientation at the moment of switching.
        self.frozen_orientation_quat = None

        self.lock = threading.Lock()

        # =============================================
        # TF LISTENER (to get fork tip pose)
        # =============================================
        self.tf_listener = ros_tf.TransformListener()

        # =============================================
        # MOVEIT
        # =============================================
        moveit_commander.roscpp_initialize(sys.argv)
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")

        rospy.wait_for_service('/compute_ik')
        self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)

        # =============================================
        # ROS I/O
        # =============================================

        self.pub_joints = rospy.Publisher(
            '/joint_group_position_controller/command',
            Float64MultiArray,
            queue_size=1,
        )

        # Diffusion trajectory input
        self.sub_traj = rospy.Subscriber(
            "/diffusion/target_trajectory",
            PoseArray,
            self.traj_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )

        # Face detection input
        self.sub_face = rospy.Subscriber(
            "/face_detection/mouth_position",
            PointStamped,
            self.face_callback,
            queue_size=1,
        )

        self.sub_joints = rospy.Subscriber("/joint_states", JointState, self.joint_cb)

        rospy.loginfo("=" * 60)
        rospy.loginfo("Fork Trajectory Follower (with Face Tracking)")
        rospy.loginfo(f"   Mode: {self.mode}")
        rospy.loginfo(f"   Y threshold: {self.y_threshold} m")
        rospy.loginfo(f"   Face approach offset: {self.face_approach_offset} m")
        rospy.loginfo(f"   Control rate: {self.control_rate} Hz")
        rospy.loginfo("=" * 60)

    # ==========================================================
    # CALLBACKS
    # ==========================================================

    def joint_cb(self, msg):
        positions = []
        for name, pos in zip(msg.name, msg.position):
            if "panda_joint" in name:
                positions.append(pos)
        if len(positions) == 7:
            self.current_joints = np.array(positions)
            if self.target_joints is None:
                self.target_joints = self.current_joints.copy()

    def traj_callback(self, msg):
        """Store latest diffusion trajectory (only used in INFERENCE mode)."""
        if self.mode != "INFERENCE":
            return  # ignore trajectories once we switched

        with self.lock:
            trajectory_joints = self.process_trajectory(msg, msg.header.frame_id)
            if trajectory_joints is not None and len(trajectory_joints) >= 2:
                self.latest_trajectory = trajectory_joints
                self.trajectory_timestamp = rospy.Time.now()

    def face_callback(self, msg):
        """
        Receive mouth position from face detection node.
        The message comes in the camera frame — we transform it to world frame using TF.
        """
        try:
            # Wait for the transform (up to 0.1s)
            self.tf_listener.waitForTransform(
                "/world", msg.header.frame_id, msg.header.stamp, rospy.Duration(0.1)
            )
            # Transform the point into world frame
            point_world = self.tf_listener.transformPoint("/world", msg)

            with self.lock:
                self.face_target_world = np.array([
                    point_world.point.x,
                    point_world.point.y,
                    point_world.point.z,
                ])
                self.face_target_timestamp = rospy.Time.now()

        except (ros_tf.LookupException, ros_tf.ConnectivityException,
                ros_tf.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0, f"Cannot transform face point: {e}")

    # ==========================================================
    # FORK TIP POSE (for mode switching check)
    # ==========================================================

    def get_fork_tip_pose_world(self):
        """
        Returns (position, quaternion) of fork tip in world frame.
        position: np.array (3,)
        quaternion: np.array (4,)  [x, y, z, w]
        """
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                '/world', '/panda_hand_tcp', rospy.Time(0)
            )
            T_world_tcp = tft.quaternion_matrix(rot)
            T_world_tcp[:3, 3] = trans
            T_world_fork = T_world_tcp @ self.T_tcp_fork_tip

            pos = T_world_fork[:3, 3]
            quat = tft.quaternion_from_matrix(T_world_fork)  # [x, y, z, w]
            return pos, quat

        except (ros_tf.LookupException, ros_tf.ConnectivityException,
                ros_tf.ExtrapolationException):
            return None, None

    # ==========================================================
    # MODE SWITCHING LOGIC
    # ==========================================================

    def check_mode_switch(self):
        """
        One-way switch: INFERENCE -> FACE_TRACKING
        Triggered when fork tip Y > y_threshold.
        """
        if self.mode != "INFERENCE":
            return

        fork_pos, fork_quat = self.get_fork_tip_pose_world()
        if fork_pos is None:
            return

        if fork_pos[1] > self.y_threshold:
            # Freeze the current fork orientation for face tracking.
            # The idea: the policy already oriented the fork correctly for feeding,
            # so we keep that orientation and only change position.
            self.frozen_orientation_quat = fork_quat

            self.mode = "FACE_TRACKING"
            rospy.loginfo("=" * 60)
            rospy.loginfo(f"MODE SWITCH -> FACE_TRACKING  (fork Y={fork_pos[1]:.3f} > {self.y_threshold})")
            rospy.loginfo("=" * 60)

    # ==========================================================
    # IK & TRAJECTORY PROCESSING (unchanged from original)
    # ==========================================================

    def transform_fork_to_tcp(self, pose_fork):
        """Transform pose from fork_tip frame to TCP frame."""
        p = np.array([pose_fork.position.x, pose_fork.position.y, pose_fork.position.z])
        q = np.array([
            pose_fork.orientation.x, pose_fork.orientation.y,
            pose_fork.orientation.z, pose_fork.orientation.w,
        ])

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

    # ==========================================================
    # LOOKAHEAD (INFERENCE MODE — unchanged)
    # ==========================================================

    def find_closest_index(self, trajectory):
        if self.current_joints is None or len(trajectory) == 0:
            return 0
        distances = [np.linalg.norm(t - self.current_joints) for t in trajectory]
        return np.argmin(distances)

    def get_lookahead_target(self):
        if self.latest_trajectory is None or len(self.latest_trajectory) == 0:
            return None
        closest_idx = self.find_closest_index(self.latest_trajectory)
        target_idx = min(closest_idx + self.lookahead_steps, len(self.latest_trajectory) - 1)
        return self.latest_trajectory[target_idx]

    # ==========================================================
    # FACE TRACKING TARGET
    # ==========================================================

    def get_face_tracking_target(self):
        """
        Compute a joint-space target that moves the fork toward the mouth.

        Strategy:
          - Position: mouth position (with a small offset so we don't poke the user)
          - Orientation: frozen from the moment of switching (the policy already
            oriented the fork for feeding, so we keep it)
          - Convert to IK -> joint target
        """
        if self.face_target_world is None:
            return None

        # Check staleness
        if self.face_target_timestamp is not None:
            age = rospy.Time.now() - self.face_target_timestamp
            if age > self.face_timeout:
                rospy.logwarn_throttle(2.0, "Face target is stale, holding position")
                return None

        # Build fork tip target pose in world frame
        # Position: mouth, pulled back along Y by the approach offset
        target_pos = self.face_target_world.copy()
        target_pos[1] -= self.face_approach_offset  # approach from in front

        # Safety floor
        if target_pos[2] < self.z_min_safety:
            target_pos[2] = self.z_min_safety

        # Orientation: use the frozen orientation from mode switch
        if self.frozen_orientation_quat is None:
            return None

        # Build a Pose message for the fork tip
        pose_fork = Pose()
        pose_fork.position.x = target_pos[0]
        pose_fork.position.y = target_pos[1]
        pose_fork.position.z = target_pos[2]
        pose_fork.orientation.x = self.frozen_orientation_quat[0]
        pose_fork.orientation.y = self.frozen_orientation_quat[1]
        pose_fork.orientation.z = self.frozen_orientation_quat[2]
        pose_fork.orientation.w = self.frozen_orientation_quat[3]

        # Transform fork tip pose -> TCP pose, then IK
        pose_tcp = self.transform_fork_to_tcp(pose_fork)
        sol = self.compute_ik(pose_tcp, "world", seed=self.current_joints)

        if sol is not None and np.max(np.abs(sol - self.current_joints)) <= self.max_joint_jump:
            return sol

        return None

    # ==========================================================
    # MAIN COMMAND COMPUTATION
    # ==========================================================

    def compute_command(self):
        """
        Core logic — same blending approach, different target source per mode.
        """
        if self.current_joints is None or self.target_joints is None:
            return None

        # Pick the target based on current mode
        if self.mode == "FACE_TRACKING":
            lookahead_target = self.get_face_tracking_target()
            blend = self.face_blend_rate
        else:
            lookahead_target = self.get_lookahead_target()
            blend = self.blend_rate

        if lookahead_target is None:
            return self.target_joints  # hold position

        # Blend current target toward the selected target
        self.target_joints = (1 - blend) * self.target_joints + blend * lookahead_target

        # Velocity limiting
        direction = self.target_joints - self.current_joints
        distance = np.linalg.norm(direction)

        if distance < 0.001:
            return self.target_joints

        max_step = self.max_joint_velocity / self.control_rate

        if distance > max_step:
            direction = direction / distance * max_step
            command = self.current_joints + direction
        else:
            command = self.target_joints

        return command

    # ==========================================================
    # MAIN LOOP
    # ==========================================================

    def run(self):
        rate = rospy.Rate(self.control_rate)

        while not rospy.is_shutdown():
            if self.current_joints is None:
                rate.sleep()
                continue

            with self.lock:
                # Check if we should switch modes
                self.check_mode_switch()

                # Compute and send command
                command = self.compute_command()

            if command is not None:
                max_diff = np.max(np.abs(command - self.current_joints))
                if max_diff < 0.5:
                    msg = Float64MultiArray()
                    msg.data = command.tolist()
                    self.pub_joints.publish(msg)

            rate.sleep()


if __name__ == '__main__':
    TrajectoryFollowerContinuous().run()