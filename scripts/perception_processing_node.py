#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import os
import sys
import rospkg
from sensor_msgs.msg import PointCloud2, JointState
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from vision_processing import fast_perception_module

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
sys.path.insert(0, pkg_path)

from third_party.RDF.urdf_layer import URDFLayer
from third_party.SDF_Bernstein_Basis.src.rdf_weights import RDF_Weights
from third_party.SDF_Bernstein_Basis.bernstein_core import BernsteinCore

class PerceptionProcessingNode:
    """
    Offloads heavy perception tasks (CropBox, Robot-Filtering, ROR) from the 150Hz loop.
    Runs at camera rate (~30Hz) to produce a 'Cleaned' obstacle cloud.
    """
    def __init__(self):
        rospy.init_node('perception_processing_node')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Parameters for CropBox
        self.crop_min = rospy.get_param("~crop_min", [0.0, -0.6, 0.0])
        self.crop_max = rospy.get_param("~crop_max", [0.8, 0.6, 1.0])
        
        # URDF and Bernstein for Robot Filtering
        urdf_path = rospy.get_param("~urdf_path", pkg_path + '/third_party/RDF/collision_avoidance_example/panda_urdf/panda.urdf')
        voxel_dir = rospy.get_param("~voxel_dir", pkg_path + '/third_party/RDF/panda_layer/meshes/voxel_128')
        weights_dir = rospy.get_param("~weights_dir", pkg_path + '/third_party/SDF_Bernstein_Basis/panda_test')
        
        self.robot_layer = URDFLayer(urdf_path=urdf_path, device=self.device, package_dir=None, voxel_dir=voxel_dir)
        weight_handler = RDF_Weights(device=self.device, dtype=torch.float32)
        weight_handler.init_robot_folder(weights_dir, robot_name='panda')
        link_names = ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7']
        weight_handler.add_models(link_names, robot_name='panda')
        self.bernstein_core = BernsteinCore(weight_handler, self.robot_layer, self.device, link_names)
        
        self.current_q = None
        self.raw_obstacles = None
        
        # Subscribers
        rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        rospy.Subscriber('/perception/obstacles', PointCloud2, self.obs_callback)
        
        # Publisher for cleaned obstacles
        self.cleaned_pub = rospy.Publisher('/perception/cleaned_obstacles', PointCloud2, queue_size=1)
        
        self.rate = rospy.Rate(30)
        rospy.loginfo("🧼 Perception Processing Node Initialized at 30Hz")

    def joint_callback(self, msg):
        self.current_q = torch.tensor(msg.position[:7], dtype=torch.float32, device=self.device).unsqueeze(0)

    def obs_callback(self, msg):
        try:
            points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, int(msg.point_step/4))
            self.raw_obstacles = torch.tensor(points[:, :3].copy(), device=self.device)
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error in Perception Processing Node (obs_callback): {e}")

    def run(self):
        while not rospy.is_shutdown():
            if self.raw_obstacles is not None and self.current_q is not None:
                pts = self.raw_obstacles
                
                # 1. GPU CropBox
                mask_x = (pts[:, 0] > self.crop_min[0]) & (pts[:, 0] < self.crop_max[0])
                mask_y = (pts[:, 1] > self.crop_min[1]) & (pts[:, 1] < self.crop_max[1])
                mask_z = (pts[:, 2] > self.crop_min[2]) & (pts[:, 2] < self.crop_max[2])
                pts_inside = pts[mask_x & mask_y & mask_z]
                
                if pts_inside.shape[0] > 0:
                    # 2. Robot-Self Filtering (Remove robot points from obstacle cloud)
                    base = torch.eye(4, device=self.device).unsqueeze(0)
                    sdf_vals = self.bernstein_core.get_whole_body_sdf_batch(pts_inside, base, self.current_q)
                    mask_not_robot = sdf_vals[0] > 0.04 # 4cm safety margin from meshes
                    pts_inside = pts_inside[mask_not_robot]
                
                if pts_inside.shape[0] > 0:
                    # 3. Radius Outlier Removal (C++)
                    pts_np = pts_inside.cpu().numpy().astype(np.float32)
                    pts_filtered_np = fast_perception_module.radius_outlier_removal(pts_np, 0.05, 5)
                    
                    # Publish Cleaned Cloud
                    header = Header()
                    header.stamp = rospy.Time.now()
                    header.frame_id = "world"
                    cloud_msg = pc2.create_cloud_xyz32(header, pts_filtered_np.tolist())
                    self.cleaned_pub.publish(cloud_msg)
                
            self.rate.sleep()

if __name__ == '__main__':
    node = PerceptionProcessingNode()
    node.run()
