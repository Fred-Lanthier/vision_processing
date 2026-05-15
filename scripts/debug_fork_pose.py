#!/usr/bin/env python3
import rospy
import rospkg
import os
import sys
import torch
import numpy as np
import xacro
import tempfile
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Float64MultiArray, Float32MultiArray, Int32
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, PoseArray

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
sys.path.insert(0, pkg_path)

from third_party.RDF.urdf_layer import URDFLayer
from scipy.spatial.transform import Rotation as R

class DebugForkPoseNode:
    def __init__(self):
        rospy.init_node('debug_fork_pose')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load URDF
        urdf_path_raw = rospy.get_param("~urdf_path", os.path.join(pkg_path, 'urdf', 'panda_camera.xacro'))
        if urdf_path_raw.endswith('.xacro'):
            doc = xacro.process_file(urdf_path_raw, mappings={'arm_id': 'panda'})
            with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
                f.write(doc.toxml())
                urdf_path = f.name
        else:
            urdf_path = urdf_path_raw

        voxel_dir = os.path.join(pkg_path, 'third_party', 'RDF', 'panda_layer', 'meshes', 'voxel_128')
        self.robot_layer = URDFLayer(urdf_path=urdf_path, device=self.device, package_dir=pkg_path, voxel_dir=voxel_dir)
        
        self.current_q = None
        self.commanded_q = None
        self.traj_q_list = None
        self.wanted_pose_list = None
        self.target_idx = None
        
        self.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']

        rospy.Subscriber('/joint_states', JointState, self.joint_cb)
        rospy.Subscriber('/joint_group_position_controller/command', Float64MultiArray, self.command_cb)
        rospy.Subscriber('/planner/nominal_trajectory', JointTrajectory, self.traj_cb)
        rospy.Subscriber('/planner/nominal_fork_trajectory', PoseArray, self.wanted_traj_cb)
        rospy.Subscriber('/diffusion/target_trajectory', PoseArray, self.wanted_traj_cb)
        rospy.Subscriber('/debug/nominal_target_index', Int32, self.target_idx_cb)
        
        self.viz_pub = rospy.Publisher('/viz/debug_fork_poses', MarkerArray, queue_size=1)
        
        self.rate = rospy.Rate(10)
        
    def joint_cb(self, msg):
        pos_dict = {n: p for n, p in zip(msg.name, msg.position)}
        q_list = []
        for jn in self.joint_names:
            if jn in pos_dict:
                q_list.append(pos_dict[jn])
        if len(q_list) == 7:
            self.current_q = torch.tensor(q_list, dtype=torch.float32, device=self.device)

    def command_cb(self, msg):
        if len(msg.data) >= 7:
            self.commanded_q = torch.tensor(msg.data[:7], dtype=torch.float32, device=self.device)

    def traj_cb(self, msg):
        q_list = []
        for p in msg.points:
            if len(p.positions) >= 7:
                q_list.append(p.positions[:7])
        if len(q_list) > 0:
            self.traj_q_list = torch.tensor(q_list, dtype=torch.float32, device=self.device)

    def wanted_traj_cb(self, msg):
        self.wanted_pose_list = msg.poses

    def target_idx_cb(self, msg):
        self.target_idx = msg.data

    def get_fork_pose(self, q):
        # Pad to 9
        q_pad = torch.cat([q, torch.zeros(2, device=self.device)])
        link_poses = self.robot_layer._native_forward_kinematics(q_pad.unsqueeze(0))
        if 'fork_tip' in link_poses:
            return link_poses['fork_tip'][0]
        return None

    def create_pose_marker(self, T, r, g, b, m_id, ns, is_sphere=False, scale=1.0):
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = m_id
        
        if isinstance(T, torch.Tensor):
            m.pose.position.x = T[0, 3].item()
            m.pose.position.y = T[1, 3].item()
            m.pose.position.z = T[2, 3].item()
            rot = R.from_matrix(T[:3, :3].cpu().numpy()).as_quat()
        else:
            # Assume it's a geometry_msgs/Pose
            m.pose = T
            rot = [T.orientation.x, T.orientation.y, T.orientation.z, T.orientation.w]
        
        m.pose.orientation.x = rot[0]
        m.pose.orientation.y = rot[1]
        m.pose.orientation.z = rot[2]
        m.pose.orientation.w = rot[3]
        
        if is_sphere:
            m.type = Marker.SPHERE
            m.scale.x = 0.02 * scale
            m.scale.y = 0.02 * scale
            m.scale.z = 0.02 * scale
        else:
            m.type = Marker.ARROW
            m.scale.x = 0.1 * scale # arrow length
            m.scale.y = 0.01 * scale # arrow width
            m.scale.z = 0.01 * scale # arrow height
            
        m.action = Marker.ADD
        
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = 1.0
        
        return m

    def run(self):
        rospy.loginfo("Debug node running... Open RViz and subscribe to /viz/debug_fork_poses")
        while not rospy.is_shutdown():
            markers = MarkerArray()
            
            # 1. Current pose (Blue Arrow) - ACTUAL
            if self.current_q is not None:
                T_curr = self.get_fork_pose(self.current_q)
                if T_curr is not None:
                    markers.markers.append(self.create_pose_marker(T_curr, 0.0, 0.0, 1.0, 0, "actual_fork_pose"))
                    
            # 2. Commanded pose from 150Hz node (Magenta Arrow)
            if self.commanded_q is not None:
                T_cmd = self.get_fork_pose(self.commanded_q)
                if T_cmd is not None:
                    markers.markers.append(self.create_pose_marker(T_cmd, 1.0, 0.0, 1.0, 1, "commanded_fk_pose"))
                    
            # 3. Trajectory from 8Hz node (Yellow Arrows)
            if self.traj_q_list is not None:
                for i in range(len(self.traj_q_list)):
                    T_traj = self.get_fork_pose(self.traj_q_list[i])
                    if T_traj is not None:
                        markers.markers.append(self.create_pose_marker(T_traj, 1.0, 1.0, 0.0, 100 + i, "joint_trajectory_fk_pose", is_sphere=False))

            # 4. Wanted trajectory from Cartesian Planner (Green Arrows) - WANTED
            if self.wanted_pose_list is not None:
                for i in range(len(self.wanted_pose_list)):
                    scale = 1.0
                    ns = "wanted_fork_trajectory"
                    # Highlight target index
                    if self.target_idx is not None and i == self.target_idx:
                        scale = 1.5
                        ns = "wanted_fork_target"
                    markers.markers.append(self.create_pose_marker(self.wanted_pose_list[i], 0.0, 1.0, 0.0, 200 + i, ns, scale=scale))

            # Delete old markers (using a DELALL marker at index 0 first, then adding)
            del_marker = Marker()
            del_marker.action = Marker.DELETEALL
            markers.markers.insert(0, del_marker)

            if len(markers.markers) > 1:
                self.viz_pub.publish(markers)

            # 5. Log errors at 0.5 Hz
            if self.current_q is not None and self.commanded_q is not None:
                T_curr = self.get_fork_pose(self.current_q)
                T_cmd = self.get_fork_pose(self.commanded_q)
                if T_curr is not None and T_cmd is not None:
                    pos_err = torch.norm(T_curr[:3, 3] - T_cmd[:3, 3]).item()
                    
                    R_curr = T_curr[:3, :3].cpu().numpy()
                    R_cmd = T_cmd[:3, :3].cpu().numpy()
                    R_err = R_curr.T @ R_cmd
                    angle_err = np.degrees(R.from_matrix(R_err).magnitude())
                    
                    # Homing target from homing.py
                    home_joints = np.array([-0.000059, -0.125928, 0.000117, -2.193312, -0.000251, 2.064780, 0.785511])
                    curr_q_np = self.current_q.cpu().numpy()
                    
                    rospy.loginfo_throttle(2.0, 
                        f"COMMAND TRACKING ERROR (Blue vs Magenta) | Pos: {pos_err*100:.2f} cm, Angle: {angle_err:.2f} deg\n"
                        f"JOINT STATE vs HOMING TARGET:\n"
                        f"  Actual: {np.array2string(curr_q_np, precision=3, suppress_small=True)}\n"
                        f"  Homing: {np.array2string(home_joints, precision=3, suppress_small=True)}"
                    )
                
            self.rate.sleep()

if __name__ == '__main__':
    DebugForkPoseNode().run()
