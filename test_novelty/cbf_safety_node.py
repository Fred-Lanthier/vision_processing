#!/usr/bin/env python3

import rospy
import rospkg
import sys
import time
import torch
import numpy as np
import struct
from std_msgs.msg import Float32MultiArray, Header
from sensor_msgs.msg import JointState, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
sys.path.insert(0, pkg_path)

from third_party.SafeFlowMatcher.diffuser.models.rdf_cbf import RDF_CBF
from third_party.RDF.panda_layer.panda_layer import PandaLayer
from third_party.RDF.rdf_core import RDFCore
from third_party.RDF.rdf_barrier import RDF_Barrier
from third_party.RDF.urdf_layer import URDFLayer

from vision_processing import fast_perception_module

class CBFSafetyNode:
    def __init__(self):
        rospy.init_node('cbf_safety_node')
        self.device = torch.device('cuda')

        urdf_path = pkg_path + '/third_party/RDF/collision_avoidance_example/panda_urdf/panda.urdf' 
        self.robot_layer = URDFLayer(
            urdf_path=urdf_path,
            device=self.device,
            package_dir=None,
            voxel_dir=pkg_path + '/third_party/RDF/panda_layer/meshes/voxel_128'
        )
        
        model_path = pkg_path + '/third_party/RDF/models/BP_8.pt'
        model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.rdf_core = RDFCore(8, -1.0, 1.0, self.robot_layer, self.device, model)
        
        self.d_safe = float(rospy.get_param("~d_safe", 0.005))
        self.robot_filter_distance = self.d_safe / 2.0
        rdf_barrier = RDF_Barrier(self.rdf_core, d_safe=self.d_safe)
        self.cbf = RDF_CBF(rdf_barrier, self.robot_layer, gamma=10.0)
        
        # Graphe CUDA à 100 points (Taille fixe)
        self.cbf.setup_cuda_graph(batch_size=1, n_points=100)
        
        self.current_q = None
        self.target_x = None
        self.obs_points = None
        
        self.comp_times = []
        rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        rospy.Subscriber('/planning/target_pose', Float32MultiArray, self.target_callback)
        rospy.Subscriber('/perception/obstacles', Float32MultiArray, self.obs_callback)
        
        self.cmd_pub = rospy.Publisher('/franka_control/safe_joint_velocities', Float32MultiArray, queue_size=1)
        
        # Publishers RViz (Uniquement Jaune et Rouge)
        self.pub_inside_yellow = rospy.Publisher('/viz/obs_inside_yellow', PointCloud2, queue_size=1)
        self.pub_top100_red = rospy.Publisher('/viz/obs_top100_red', PointCloud2, queue_size=1)
        self.closest_pt_pub = rospy.Publisher('/viz/closest_point', Marker, queue_size=1)
        
        # FRÉQUENCE STABLE À 20 Hz
        self.rate_hz = 150.0
        self.rate = rospy.Rate(self.rate_hz) 
        self.last_time = rospy.get_time()

    def joint_callback(self, msg):
        self.current_q = torch.tensor(msg.position[:7], dtype=torch.float32, device=self.device).unsqueeze(0)

    def target_callback(self, msg):
        self.target_x = torch.tensor(msg.data, dtype=torch.float32, device=self.device).unsqueeze(0)

    def obs_callback(self, msg):
        flat_data = np.array(msg.data, dtype=np.float32)
        self.obs_points = torch.tensor(flat_data.reshape(-1, 3), device=self.device)

    def create_cloud_xyzrgb(self, points_tensor, color):
        """Créer un nuage XYZRGB pour RViz (sans les verts) via C++ ultra-rapide."""
        if points_tensor.shape[0] == 0:
            return None
            
        r, g, b = color
        pts_np = points_tensor.cpu().numpy().astype(np.float32)

        msg_bytes = fast_perception_module.create_cloud_xyzrgb(pts_np, r, g, b, "world")
        
        msg = PointCloud2()
        msg.deserialize(msg_bytes)
        msg.header.stamp = rospy.Time.now()
        
        return msg

    def run(self):
        # print(f"🛡️  CBF NODE : Bouclier actif à {self.rate_hz}Hz. Sécurités activées.")
        dt = 1.0 / self.rate_hz
        
        COLOR_YELLOW = (255, 255, 0)
        COLOR_RED = (255, 0, 0)
        
        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            dt = current_time - self.last_time
            self.last_time = current_time

            if dt <= 0.001: 
                self.rate.sleep()
                continue
            
            if self.current_q is not None and self.target_x is not None and self.obs_points is not None:
                base = torch.eye(4, device=self.device).unsqueeze(0)
                trans_list = self.robot_layer.get_transformations_each_link(base, self.current_q)
                
                x_now_pos = trans_list[-1][:, :3, 3] 
                z_now = trans_list[-1][:, :3, 2] 
                x_now_6d = torch.cat([x_now_pos, z_now], dim=-1) 
                
                # --- COMMANDE NOMINALE SÉCURISÉE ---
                v_pos = 5.0 * (self.target_x - x_now_pos)
                v_norm = torch.norm(v_pos, dim=-1, keepdim=True)
                if (v_norm > 0.2).any():
                    v_pos = v_pos / v_norm * 0.2
                
                # Cible de rotation vers le bas
                target_z = torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0)
                v_rot = 2.0 * (target_z - z_now)
                
                # CLAMPING VITAL DE LA ROTATION
                v_rot_norm = torch.norm(v_rot, dim=-1, keepdim=True)
                if (v_rot_norm > 0.5).any():  # Limite douce à 0.5 rad/s
                    v_rot = v_rot / v_rot_norm * 0.5
                
                u_raw_6d = torch.cat([v_pos, v_rot], dim=-1)
                x_next_naive_6d = x_now_6d + u_raw_6d * dt
                # -----------------------------------
                
                torch.cuda.synchronize()
                t_start = time.perf_counter()
                t_start_ror = time.perf_counter()
                # ==============================================================
                # PIPELINE DE PERCEPTION TEMPS RÉEL (GPU)
                # ==============================================================
                pts = self.obs_points
                
                # 1. Filtre CropBox (Ajuste ces valeurs selon ta zone de travail)
                mask_x = (pts[:, 0] > 0.0) & (pts[:, 0] < 0.8)
                mask_y = (pts[:, 1] > -0.6) & (pts[:, 1] < 0.6)
                mask_z = (pts[:, 2] > 0.0) & (pts[:, 2] < 1.0)
                crop_mask = mask_x & mask_y & mask_z
                
                pts_inside = pts[crop_mask]
                
                if pts_inside.shape[0] > 0:
                    # 2. Iso-surface Filtering
                    sdf_vals = self.rdf_core.get_whole_body_sdf_batch(pts_inside, base, self.current_q)
                    mask_not_robot = sdf_vals[0] > self.robot_filter_distance
                    pts_inside = pts_inside[mask_not_robot]
                
                if pts_inside.shape[0] > 0:
                    # 3. ROR (Radius Outlier Removal) ultra-rapide via PCL (C++)
                    pts_np = pts_inside.cpu().numpy().astype(np.float32)
                    pts_filtered_np = fast_perception_module.radius_outlier_removal(pts_np, 0.05, 5)
                    pts_inside = torch.tensor(pts_filtered_np, device=self.device)
                
                num_inside = pts_inside.shape[0]
                pts_yellow = torch.empty((0, 3), device=self.device)
                
                # 4. Échantillonnage ou Padding (100 points)
                if num_inside >= 100:
                    dist_to_ee = torch.norm(pts_inside - x_now_pos, dim=1)
                    _, top_idx = torch.topk(dist_to_ee, k=100, largest=False)
                    
                    top_mask = torch.zeros(num_inside, dtype=torch.bool, device=self.device)
                    top_mask[top_idx] = True
                    
                    cbf_points = pts_inside[top_mask] # ROUGE
                    pts_yellow = pts_inside[~top_mask] # JAUNE
                    
                elif num_inside > 0:
                    dist_to_ee = torch.norm(pts_inside - x_now_pos, dim=1)
                    closest_pt = pts_inside[torch.argmin(dist_to_ee)].unsqueeze(0)
                    padding = closest_pt.repeat(100 - num_inside, 1)
                    
                    cbf_points = torch.cat([pts_inside, padding], dim=0).contiguous() 
                else:
                    cbf_points = torch.tensor([[10.0, 10.0, 10.0]], device=self.device).repeat(100, 1)
                torch.cuda.synchronize()
                
                t_end_ror = time.perf_counter()
                t_ror = t_end_ror - t_start_ror
                print(f"ROR Time: {t_ror * 1000:.2f} ms")
                # ==============================================================
                # CBF GRAPHE CUDA
                # ==============================================================
                x_safe, q_safe, dq_nom, h_val, min_idx = self.cbf.apply(x_now_6d, x_next_naive_6d, self.current_q, cbf_points, dt)
                
                torch.cuda.synchronize()
                t_end = time.perf_counter()
                
                self.comp_times.append(t_end - t_start)
                if len(self.comp_times) >= 20:
                    avg_t = sum(self.comp_times) / len(self.comp_times)
                    print(f"⏱️ Lissage & CBF: {avg_t*1000:.2f} ms")
                    self.comp_times.clear()
                
                # --- Visualisation RViz (Jaune & Rouge) ---
                if num_inside > 0:
                    msg_yellow = self.create_cloud_xyzrgb(pts_yellow, COLOR_YELLOW)
                    msg_red = self.create_cloud_xyzrgb(cbf_points[:min(num_inside, 100)], COLOR_RED) 
                    
                    if msg_yellow: self.pub_inside_yellow.publish(msg_yellow)
                    if msg_red: self.pub_top100_red.publish(msg_red)
                    
                    closest_point = cbf_points[min_idx[0]].cpu().numpy()
                    marker = Marker()
                    marker.header.frame_id = "world"
                    marker.header.stamp = rospy.Time.now()
                    marker.ns = "critical_point"
                    marker.id = 1
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.pose.position.x = closest_point[0]
                    marker.pose.position.y = closest_point[1]
                    marker.pose.position.z = closest_point[2]
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = 0.05
                    marker.scale.y = 0.05
                    marker.scale.z = 0.05
                    marker.color.a = 1.0
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0 
                    self.closest_pt_pub.publish(marker)

                dq_safe = (q_safe - self.current_q) / dt
                self.cmd_pub.publish(Float32MultiArray(data=dq_safe.squeeze(0).cpu().numpy().tolist()))
                
            self.rate.sleep()

if __name__ == '__main__':
    node = CBFSafetyNode()
    node.run()
