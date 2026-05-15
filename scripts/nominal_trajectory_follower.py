#!/usr/bin/env python3
import os
import sys
import tempfile
import time
import threading

import numpy as np
import rospy
import rospkg
import torch
import xacro
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Int32
from trajectory_msgs.msg import JointTrajectory

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
sys.path.insert(0, pkg_path)

from third_party.RDF.urdf_layer import URDFLayer


class NominalTrajectoryFollower:
    def __init__(self):
        rospy.init_node('nominal_trajectory_follower')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        urdf_path_raw = rospy.get_param('~urdf_path', os.path.join(pkg_path, 'urdf', 'panda_camera.xacro'))
        if urdf_path_raw.endswith('.xacro'):
            doc = xacro.process_file(urdf_path_raw, mappings={'arm_id': 'panda'})
            with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
                f.write(doc.toxml())
                urdf_path = f.name
        else:
            urdf_path = urdf_path_raw

        self.robot_layer = URDFLayer(
            urdf_path=urdf_path,
            device=self.device,
            package_dir=pkg_path,
            voxel_dir=os.path.join(pkg_path, 'third_party', 'RDF', 'panda_layer', 'meshes', 'voxel_128'),
        )

        self.joint_names = [f'panda_joint{i}' for i in range(1, 8)]
        self.current_q = None
        self.nominal_trajectory = None
        self.target_joints = None
        self.lock = threading.Lock()
        self.last_time = rospy.get_time()

        self.lookahead_steps = rospy.get_param('~lookahead_steps', 3)
        self.blend_rate = rospy.get_param('~blend_rate', 0.15)
        self.max_joint_velocity = rospy.get_param('~max_joint_velocity', 0.7)
        self.rate_hz = rospy.get_param('~rate_hz', 150.0)

        rospy.Subscriber('/joint_states', JointState, self.joint_callback, queue_size=1)
        rospy.Subscriber('/planner/nominal_trajectory', JointTrajectory, self.trajectory_callback, queue_size=1)

        self.cmd_pub = rospy.Publisher('/planner/nominal_joint_command', Float64MultiArray, queue_size=1)
        self.debug_cmd_pub = rospy.Publisher('/debug/cbf/input_joint_command', Float64MultiArray, queue_size=1)
        self.target_idx_pub = rospy.Publisher('/debug/nominal_target_index', Int32, queue_size=1)
        self.current_fork_pub = rospy.Publisher('/debug/nominal_current_fork_point', PointStamped, queue_size=1)
        self.target_fork_pub = rospy.Publisher('/debug/nominal_target_fork_point', PointStamped, queue_size=1)

        self.rate = rospy.Rate(self.rate_hz)
        rospy.loginfo('Nominal trajectory follower publishes /planner/nominal_joint_command')

    def joint_callback(self, msg):
        pos_dict = {n: p for n, p in zip(msg.name, msg.position)}
        q = []
        for name in self.joint_names:
            if name not in pos_dict:
                return
            q.append(pos_dict[name])
        with self.lock:
            self.current_q = torch.tensor(q, dtype=torch.float32, device=self.device).unsqueeze(0)

    def trajectory_callback(self, msg):
        if msg.points:
            with self.lock:
                self.nominal_trajectory = msg
                self.target_joints = None

    def get_fork_tip_pose(self, q7):
        if q7.dim() == 1:
            q7 = q7.unsqueeze(0)
        q_eval = q7
        if q_eval.shape[-1] == 7:
            q_eval = torch.cat([q_eval, torch.zeros((q_eval.shape[0], 2), device=self.device)], dim=-1)
        return self.robot_layer._native_forward_kinematics(q_eval).get('fork_tip', None)

    def publish_point(self, pub, xyz):
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'world'
        msg.point.x = float(xyz[0])
        msg.point.y = float(xyz[1])
        msg.point.z = float(xyz[2])
        pub.publish(msg)

    def compute_nominal_command(self, dt):
        with self.lock:
            if self.current_q is None or self.nominal_trajectory is None:
                return None
            current_q = self.current_q.clone()
            nominal_trajectory = self.nominal_trajectory
            target_joints = None if self.target_joints is None else self.target_joints.copy()

        curr_q_np = current_q.squeeze(0).detach().cpu().numpy()
        if target_joints is None:
            target_joints = curr_q_np.copy()

        T_current = self.get_fork_tip_pose(current_q)
        if T_current is None:
            rospy.logwarn_throttle(5, 'Nominal follower cannot find fork_tip in FK tree.')
            return None
        curr_fork = T_current[0, :3, 3].detach().cpu().numpy()

        waypoint_positions = []
        waypoint_joints = []
        for point in nominal_trajectory.points:
            if len(point.positions) < 7:
                continue
            q_waypoint = torch.tensor(point.positions[:7], device=self.device).float().unsqueeze(0)
            T_waypoint = self.get_fork_tip_pose(q_waypoint)
            if T_waypoint is None:
                continue
            waypoint_positions.append(T_waypoint[0, :3, 3].detach().cpu().numpy())
            waypoint_joints.append(np.array(point.positions[:7], dtype=np.float64))

        if not waypoint_positions:
            return None

        waypoint_positions = np.asarray(waypoint_positions)
        waypoint_joints = np.asarray(waypoint_joints)
        closest_idx = int(np.argmin(np.linalg.norm(waypoint_positions - curr_fork, axis=1)))
        target_idx = min(closest_idx + self.lookahead_steps, len(waypoint_joints) - 1)

        target_lookahead = waypoint_joints[target_idx]
        target_joints = (1.0 - self.blend_rate) * target_joints + self.blend_rate * target_lookahead

        direction = target_joints - curr_q_np
        distance = np.linalg.norm(direction)
        max_step = self.max_joint_velocity * dt
        if distance > max_step:
            q_next_nom = curr_q_np + (direction / distance) * max_step
        else:
            q_next_nom = target_joints

        with self.lock:
            self.target_joints = target_joints

        self.target_idx_pub.publish(Int32(data=target_idx))
        self.publish_point(self.current_fork_pub, curr_fork)
        self.publish_point(self.target_fork_pub, waypoint_positions[target_idx])

        return q_next_nom

    def run(self):
        while not rospy.is_shutdown():
            now = rospy.get_time()
            dt = now - self.last_time
            self.last_time = now
            if dt <= 0.0 or dt > 0.1:
                dt = 1.0 / self.rate_hz

            command = self.compute_nominal_command(dt)
            if command is not None:
                msg = Float64MultiArray(data=command.tolist())
                self.cmd_pub.publish(msg)
                self.debug_cmd_pub.publish(msg)

            self.rate.sleep()


if __name__ == '__main__':
    NominalTrajectoryFollower().run()
