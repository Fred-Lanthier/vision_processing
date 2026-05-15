#!/usr/bin/env python3
import os
import sys
import tempfile
import threading

import numpy as np
import rospy
import rospkg
import tf
import tf.transformations as tft
import xacro
from geometry_msgs.msg import PointStamped, Pose, PoseArray
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Int32
from trajectory_msgs.msg import JointTrajectory
from scipy.spatial.transform import Rotation as R
from vision_processing import fast_ik_module

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
scripts_path = os.path.join(pkg_path, 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils import compute_T_child_parent_xacro


class NominalForkPoseFollower:
    def __init__(self):
        rospy.init_node('nominal_fork_pose_follower')

        # --- HARDCODED HOMING POSE ---
        self.home_joints = np.array([
            -0.000059, -0.125928, 0.000117, -2.193312, 
            -0.000251, 2.064780, 0.785511
        ], dtype=np.float64)

        xacro_file = os.path.join(pkg_path, 'urdf', 'panda_camera.xacro')
        self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro_file, 'fork_tip', 'panda_TCP')
        self.T_tcp_fork_tip_inv = np.linalg.inv(self.T_tcp_fork_tip)

        self.control_rate = rospy.get_param('~rate_hz', 150.0)
        self.lookahead_steps = rospy.get_param('~lookahead_steps', 3)
        self.blend_rate = rospy.get_param('~blend_rate', 0.15)
        self.blend_rate_near = rospy.get_param('~blend_rate_near', 0.50)
        self.proximity_threshold = rospy.get_param('~proximity_threshold', 0.030)
        self.max_joint_velocity = rospy.get_param('~max_joint_velocity', 0.7)
        self.max_joint_jump = rospy.get_param('~max_joint_jump', 0.5)
        self.ik_timeout = rospy.get_param('~ik_timeout', 0.03)
        self.print_error_hz = rospy.get_param('~print_error_hz', 5.0)
        self.tcp_frame = rospy.get_param('~tcp_frame', 'panda_TCP')
        self.ik_link_name = rospy.get_param('~ik_link_name', self.tcp_frame)
        self.use_planner_joint_trajectory = rospy.get_param('~use_planner_joint_trajectory', True)
        self.publish_nominal_topic = rospy.get_param('~publish_nominal_topic', True)
        self.publish_controller_command = rospy.get_param('~publish_controller_command', True)
        self.last_error_print = 0.0

        self.current_joints = None
        self.target_joints = self.home_joints.copy()
        self.last_command_joints = self.home_joints.copy()
        self.latest_msg = None
        self.latest_joint_traj = None
        self.fork_food_distance = float('inf')
        self.lock = threading.Lock()
        self.last_wait_log = 0.0
        self._joint_traj_idx = 0
        self._joint_waypoints = None   # np.array [N, 7]
        self._min_waypoint_idx = 0     # monotone progress guard

        self.tf_listener = tf.TransformListener()

        doc = xacro.process_file(xacro_file, mappings={'arm_id': 'panda'})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(doc.toxml())
            urdf_path = f.name
        self.ik_solver = fast_ik_module.FastIK(urdf_path, self.ik_link_name)
        rospy.loginfo("FastIK solver loaded for frame '%s'", self.ik_link_name)

        from std_msgs.msg import Float32
        rospy.Subscriber('/planner/nominal_fork_trajectory', PoseArray, self.traj_callback, queue_size=1)
        rospy.Subscriber('/planner/nominal_trajectory', JointTrajectory, self.joint_traj_callback, queue_size=1)
        rospy.Subscriber('/joint_states', JointState, self.joint_callback, queue_size=1)
        rospy.Subscriber('/vision/fork_food_distance', Float32, self.dist_callback, queue_size=1)

        self.cmd_pub = rospy.Publisher('/planner/nominal_joint_command', Float64MultiArray, queue_size=1)
        self.debug_cmd_pub = rospy.Publisher('/debug/cbf/input_joint_command', Float64MultiArray, queue_size=1)
        self.safe_cmd_pub = rospy.Publisher('/planner/safe_joint_command', Float64MultiArray, queue_size=1)
        self.controller_cmd_pub = rospy.Publisher('/joint_group_position_controller/command', Float64MultiArray, queue_size=1)
        self.debug_output_pub = rospy.Publisher('/debug/follower/final_joint_command', Float64MultiArray, queue_size=1)
        self.target_idx_pub = rospy.Publisher('/debug/nominal_target_index', Int32, queue_size=1)
        self.current_fork_pub = rospy.Publisher('/debug/nominal_current_fork_point', PointStamped, queue_size=1)
        self.target_fork_pub = rospy.Publisher('/debug/nominal_target_fork_point', PointStamped, queue_size=1)

        rospy.loginfo(
            'Nominal fork pose follower uses trajectory_node_Fork.py logic '
            'with tcp_frame=%s ik_link_name=%s use_planner_joint_trajectory=%s '
            'publish_controller_command=%s',
            self.tcp_frame,
            self.ik_link_name,
            self.use_planner_joint_trajectory,
            self.publish_controller_command,
        )

    def dist_callback(self, msg):
        self.fork_food_distance = msg.data

    def joint_callback(self, msg):
        pos_dict = {n: p for n, p in zip(msg.name, msg.position)}
        joints = []
        for i in range(1, 8):
            name = f'panda_joint{i}'
            if name not in pos_dict:
                return
            joints.append(pos_dict[name])

        with self.lock:
            self.current_joints = np.array(joints, dtype=np.float64)
            if self.target_joints is None:
                self.target_joints = self.current_joints.copy()
            if self.last_command_joints is None:
                self.last_command_joints = self.current_joints.copy()

    def traj_callback(self, msg):
        if not msg.poses:
            return
        with self.lock:
            self.latest_msg = msg
            if self.latest_joint_traj is None:
                # We don't reset target_joints here to avoid jumps, 
                # instead let compute_command handle the blending.
                pass
        rospy.loginfo("Received nominal fork trajectory with %d poses", len(msg.poses))

    def joint_traj_callback(self, msg):
        """Store joint waypoints and reset monotone progress index."""
        if not msg.points:
            return
        wps = np.array([p.positions[:7] for p in msg.points], dtype=np.float64)
        with self.lock:
            self.latest_joint_traj = msg
            self._joint_waypoints  = wps
            self._min_waypoint_idx = 0
        rospy.loginfo_throttle(2.0, "Received joint trajectory: %d waypoints", len(wps))

    def joint_waypoint(self, msg, idx):
        if idx >= len(msg.points):
            return None
        point = msg.points[idx]
        pos_dict = {n: p for n, p in zip(msg.joint_names, point.positions)}
        try:
            return np.array([pos_dict[f'panda_joint{i}'] for i in range(1, 8)], dtype=np.float64)
        except KeyError:
            return None

    def transform_fork_to_tcp(self, pose_fork):
        p = np.array([pose_fork.position.x, pose_fork.position.y, pose_fork.position.z], dtype=np.float64)
        q = np.array([
            pose_fork.orientation.x,
            pose_fork.orientation.y,
            pose_fork.orientation.z,
            pose_fork.orientation.w,
        ], dtype=np.float64)

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

    def compute_ik(self, pose, frame_id, seed):
        T = self.pose_to_matrix(pose)
        q_seed = np.concatenate([seed, [0.0, 0.0]])  # 7 → 9 DOF (arm + fingers)
        q9 = self.ik_solver.solve_single_ik(T, q_seed)
        q7 = np.array(q9[:7], dtype=np.float64)
        if not np.all(np.isfinite(q7)):
            return None
        return q7

    def current_fork_transform(self):
        try:
            trans, rot = self.tf_listener.lookupTransform('world', self.tcp_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
            rospy.logwarn_throttle(2.0, f"Waiting for TF world -> {self.tcp_frame}: {exc}")
            return None

        T_world_tcp = tft.quaternion_matrix(rot)
        T_world_tcp[:3, 3] = trans
        return T_world_tcp @ self.T_tcp_fork_tip

    def pose_to_matrix(self, pose):
        quat = np.array([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ], dtype=np.float64)
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat).as_matrix()
        T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return T

    def format_joints(self, joints):
        if joints is None:
            return "None"
        return np.array2string(
            np.asarray(joints, dtype=np.float64),
            precision=3,
            suppress_small=True,
            separator=", ",
        )

    def maybe_print_tracking_error(
        self,
        T_current_fork,
        closest_pose,
        closest_idx,
        target_idx,
        desired_joints=None,
        current_joints=None,
        command_joints=None,
    ):
        now = rospy.get_time()
        if self.print_error_hz <= 0.0:
            return
        if now - self.last_error_print < 1.0 / self.print_error_hz:
            return
        self.last_error_print = now

        T_desired = self.pose_to_matrix(closest_pose)
        pos_err = np.linalg.norm(T_desired[:3, 3] - T_current_fork[:3, 3])
        R_err = T_desired[:3, :3].T @ T_current_fork[:3, :3]
        angle_err = R.from_matrix(R_err).magnitude() * 180.0 / np.pi
        rospy.loginfo(
            "TRACKING ERROR | closest_idx=%d target_idx=%d pos=%.4f m angle=%.2f deg "
            "desired_joints=%s current_joints=%s command_joints=%s",
            closest_idx,
            target_idx,
            pos_err,
            angle_err,
            self.format_joints(desired_joints),
            self.format_joints(current_joints),
            self.format_joints(command_joints),
        )

    def publish_point(self, pub, xyz):
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'world'
        msg.point.x = float(xyz[0])
        msg.point.y = float(xyz[1])
        msg.point.z = float(xyz[2])
        pub.publish(msg)

    def compute_command(self):
        with self.lock:
            current_joints = self.current_joints.copy() if self.current_joints is not None else self.home_joints.copy()
            target_joints  = self.target_joints.copy()  if self.target_joints  is not None else current_joints.copy()
            msg            = self.latest_msg
            wps            = self._joint_waypoints
            min_idx        = self._min_waypoint_idx

        # ── Joint-trajectory branch (CASF pre-solved waypoints) ──────────────
        if self.use_planner_joint_trajectory:
            if wps is None:
                now = rospy.get_time()
                if now - self.last_wait_log > 2.0:
                    rospy.loginfo("Waiting for /planner/nominal_trajectory joint waypoints")
                    self.last_wait_log = now
                return self.home_joints.copy()

            # 1. Closest waypoint (monotone: never search before min_idx)
            dists       = np.linalg.norm(wps[min_idx:] - current_joints, axis=1)
            closest_idx = min_idx + int(np.argmin(dists))
            with self.lock:
                self._min_waypoint_idx = max(self._min_waypoint_idx, closest_idx)

            # 2. Lookahead
            target_idx = min(closest_idx + self.lookahead_steps, len(wps) - 1)
            target_lookahead_joints = wps[target_idx]

            # 3. Velocity cap from last_command_joints toward the lookahead waypoint.
            # No IIR — the velocity cap itself provides sufficient smoothing.
            last_cmd = self.last_command_joints if self.last_command_joints is not None else current_joints
            direction = target_lookahead_joints - last_cmd
            distance  = np.linalg.norm(direction)
            if distance < 1e-4:
                command = target_lookahead_joints
            else:
                max_step = self.max_joint_velocity / self.control_rate
                command  = target_lookahead_joints if distance <= max_step else (
                    last_cmd + direction / distance * max_step
                )

            with self.lock:
                self.last_command_joints = command.copy()

            self.target_idx_pub.publish(Int32(data=int(target_idx)))
            return command

        # ── Cartesian IK branch ───────────────────────────────────────────────
        if msg is None:
            return self.home_joints.copy()

        T_current_fork = self.current_fork_transform()
        if T_current_fork is None:
            return target_joints

        current_fork_pos = T_current_fork[:3, 3]
        positions = np.array([[p.position.x, p.position.y, p.position.z] for p in msg.poses], dtype=np.float64)

        closest_idx = int(np.argmin(np.linalg.norm(positions - current_fork_pos, axis=1)))
        target_idx  = min(closest_idx + self.lookahead_steps, len(msg.poses) - 1)

        target_pose_fork = msg.poses[target_idx]
        pose_tcp = self.transform_fork_to_tcp(target_pose_fork)
        ik_sol   = self.compute_ik(pose_tcp, msg.header.frame_id or 'world', seed=current_joints)

        if ik_sol is not None and np.max(np.abs(ik_sol - current_joints)) <= self.max_joint_jump:
            target_lookahead_joints = ik_sol
        else:
            target_lookahead_joints = target_joints

        # IIR blend + velocity cap (same pattern as joint branch)
        new_target = (1.0 - self.blend_rate) * target_joints + self.blend_rate * target_lookahead_joints
        with self.lock:
            self.target_joints = new_target.copy()

        last_cmd = self.last_command_joints if self.last_command_joints is not None else current_joints
        direction = new_target - last_cmd
        distance  = np.linalg.norm(direction)
        if distance < 1e-4:
            command = new_target
        else:
            max_step = self.max_joint_velocity / self.control_rate
            command  = new_target if distance <= max_step else (
                last_cmd + direction / distance * max_step
            )

        with self.lock:
            self.last_command_joints = command.copy()

        self.maybe_print_tracking_error(
            T_current_fork,
            msg.poses[closest_idx],
            closest_idx,
            target_idx,
            desired_joints=target_lookahead_joints,
            current_joints=current_joints,
            command_joints=command,
        )
        self.publish_point(self.target_fork_pub, positions[target_idx])
        self.publish_point(self.current_fork_pub, current_fork_pos)
        self.target_idx_pub.publish(Int32(data=int(target_idx)))
        return command

    def run(self):
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            if not rospy.get_param('/homing_done', False):
                rate.sleep()
                continue
                
            command = self.compute_command()
            if command is not None:
                msg = Float64MultiArray(data=command.tolist())
                if self.publish_nominal_topic:
                    self.cmd_pub.publish(msg)
                    self.debug_cmd_pub.publish(msg)
                self.safe_cmd_pub.publish(msg)
                self.debug_output_pub.publish(msg)
                if self.publish_controller_command:
                    self.controller_cmd_pub.publish(msg)
            rate.sleep()


if __name__ == '__main__':
    NominalForkPoseFollower().run()
