#!/usr/bin/env python3
import rospy
import numpy as np
import message_filters
from sensor_msgs.msg import Image, PointCloud2, JointState, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from visualization_msgs.msg import Marker
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import tf
import tf.transformations as tft
import sys
import os
import rospkg
import fpsample
import torch
import time
from PIL import Image as PILImage
from utils import compute_T_child_parent_xacro

# --- IMPORTS SAM 3 & LOADER ---
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("❌ ERREUR: Activez venv_sam3 !")
    sys.exit(1)

rospack = rospkg.RosPack()
sys.path.append(os.path.join(rospack.get_path('vision_processing'), 'scripts'))
try:
    from Compute_3D_point_cloud_from_mesh import RobotMeshLoaderOptimized
    LOADER_AVAILABLE = True
except ImportError:
    LOADER_AVAILABLE = False

class MergedCloudNode:
    def __init__(self):
        rospy.init_node('merged_cloud_node', anonymous=True)
        
        self.target_object = "cube"
        self.cube_locked = False
        self.is_grasped = False
        self.static_cube_cloud = None
        self.current_cube_cloud = None
        self.contact_threshold = -0.02 

        # Offset manuel vectoriel pour la distance (Xacro)
        self.fork_tip_offset_vec = np.array([-0.0055, 0.0, 0.1296, 1.0]) 

        # Matrices calculées UNE SEULE FOIS au démarrage (Gain de temps CPU)
        
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('vision_processing')
        xacro_file = os.path.join(package_path, 'urdf', 'panda_camera.xacro')
        self.T_tcp_wrist = compute_T_child_parent_xacro(xacro_file, "camera_wrist_link", "panda_TCP")
        self.T_wrist_opt = compute_T_child_parent_xacro(xacro_file, "camera_wrist_optical_frame", "camera_wrist_link")
        self.T_tcp_cam = np.dot(self.T_tcp_wrist, self.T_wrist_opt)
        self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro_file, 'fork_tip', 'panda_TCP')

        self.T_world_tcp_at_grasp = None
        self.T_tcp_at_grasp_inv = None
        
        # SAM 3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_model = build_sam3_image_model()
        if hasattr(self.sam_model, "to"): self.sam_model.to(self.device)
        self.sam_processor = Sam3Processor(self.sam_model, confidence_threshold=0.1)

        # Robot Loader
        self.mesh_loader = None
        if LOADER_AVAILABLE:
            urdf_path = os.path.join(package_path, 'urdf', 'panda_camera.xacro')
            try: self.mesh_loader = RobotMeshLoaderOptimized(urdf_path)
            except: pass
        
        # Publishers
        self.pub_merged = rospy.Publisher('/vision/merged_cloud', PointCloud2, queue_size=1)
        self.pub_marker = rospy.Publisher('/vision/marker', Marker, queue_size=1)
        self.pub_fork_tip = rospy.Publisher('/vision/debug_fork_tip', PointStamped, queue_size=1)

        self.fx, self.fy, self.cx, self.cy = 604.9, 604.9, 320.0, 240.0
        self.sub_info = rospy.Subscriber("/camera_wrist/color/camera_info", CameraInfo, self.cam_info_cb)
        self.tf_listener = tf.TransformListener()

        # Subscribers (Note: on écoute le topic C++ /synced/ee_pose)
        sub_rgb = message_filters.Subscriber("/synced/camera_wrist/rgb", Image)
        sub_depth = message_filters.Subscriber("/synced/camera_wrist/depth", Image)
        sub_joints = message_filters.Subscriber("/synced/joint_states", JointState)

        # Synchronisation sur 4 topics
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth, sub_joints], queue_size=5, slop=0.1
        )
        self.ts.registerCallback(self.callback)
        rospy.loginfo("🚀 MERGED CLOUD (MODE C++ SYNC) PRÊT")

    def cam_info_cb(self, msg):
        self.fx = msg.K[0]; self.cx = msg.K[2]; self.fy = msg.K[4]; self.cy = msg.K[5]
        self.sub_info.unregister()

    def imgmsg_to_numpy(self, msg):
        if msg.encoding == "rgb8": return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        elif "32FC1" in msg.encoding: return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        return None

    def pose_to_matrix(self, pose_msg):
        p = pose_msg.pose.position; q = pose_msg.pose.orientation
        return tft.compose_matrix(translate=[p.x, p.y, p.z], angles=tft.euler_from_quaternion([q.x, q.y, q.z, q.w]))

    def publish_cloud(self, points, frame_id="world"):
        if points is None or len(points) == 0: return
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        cloud_msg = pc2.create_cloud_xyz32(header, points)
        self.pub_merged.publish(cloud_msg)
    
    def callback(self, rgb_msg, depth_msg, joint_msg):
        # 1. RÉCUPÉRATION POSE (Instantanée grâce au C++)
        # On ne fait plus de TF lookup, on trust le message reçu.
        (trans_world_tcp, rot_world_tcp) = self.tf_listener.lookupTransform("world", "panda_hand_tcp", rospy.Time(0))
        T_world_tcp = tft.compose_matrix(translate=trans_world_tcp, angles=tft.euler_from_quaternion(rot_world_tcp))
        
        # 2. CALCULS GÉOMÉTRIQUES LOCAUX (Multiplication matricielle simple)
        T_world_cam = T_world_tcp @ self.T_tcp_cam
        # Optionnel si besoin de la pose fourchette sous forme de matrice
        # T_world_fork = np.dot(T_world_tcp, self.T_tcp_fork_tip)

        # T_tcp_wrist = tft.compose_matrix(translate=[-0.052, 0.035, -0.045], angles=tft.euler_from_quaternion(tft.quaternion_from_euler(0, -np.pi/2, 0)))
        # T_wrist_opt = tft.compose_matrix(translate=[0, 0, 0], angles=tft.euler_from_quaternion(tft.quaternion_from_euler(-np.pi/2, 0, -np.pi/2)))
        # self.T_tcp_optical = np.dot(T_tcp_wrist, T_wrist_opt)
        
        # --- 3. ROBOT CLOUD ---
        current_robot_points = None
        if self.mesh_loader:
            try:
                joint_map = {name: joint_msg.position[i] for i, name in enumerate(joint_msg.name) if "panda" in name}
                current_robot_points = self.mesh_loader.create_point_cloud_fork_tip(joint_map)
            except: pass

        # --- 4. DÉTECTION CUBE (SAM 3) ---
        if not self.cube_locked:
            try:
                cv_rgb = self.imgmsg_to_numpy(rgb_msg)
                cv_depth = self.imgmsg_to_numpy(depth_msg)
                
                if cv_rgb is not None and cv_depth is not None:
                    pil_image = PILImage.fromarray(cv_rgb)
                    inference_state = self.sam_processor.set_image(pil_image)
                    output = self.sam_processor.set_text_prompt(state=inference_state, prompt=self.target_object)
                    
                    raw_scores = output["scores"]
                    if len(raw_scores) > 0:
                        scores = np.array(raw_scores).flatten() if not isinstance(raw_scores, torch.Tensor) else raw_scores.detach().cpu().numpy().flatten()
                        best_idx = np.argmax(scores)
                        
                        if scores[best_idx] > 0.20:
                            masks = np.array(output["masks"]) if not isinstance(output["masks"], torch.Tensor) else output["masks"].detach().cpu().numpy()
                            final_mask = masks[best_idx]
                            while final_mask.ndim > 2: final_mask = final_mask[0]
                            
                            z = cv_depth if cv_depth.dtype == np.float32 else cv_depth / 1000.0
                            valid = (final_mask > 0) & (z > 0.01) & (z < 2.0) & np.isfinite(z)
                            
                            if np.sum(valid) > 100:
                                v, u = np.where(valid)
                                z_val = z[valid]
                                x = (u - self.cx) * z_val / self.fx
                                y = (v - self.cy) * z_val / self.fy
                                points_cam = np.stack([x, y, z_val], axis=-1)
                                
                                # PROJECTION ROBUSTE (Utilise la matrice T_world_cam calculée plus haut)
                                ones = np.ones((points_cam.shape[0], 1))
                                points_world = np.dot(T_world_cam, np.hstack([points_cam, ones]).T).T[:, :3]
                                
                                self.static_cube_cloud = points_world.astype(np.float32)
                                self.cube_locked = True
                                
                                c = np.mean(points_world, axis=0)
                                m = Marker(); m.header.frame_id="world"; m.header.stamp=rospy.Time.now(); m.type=Marker.SPHERE; m.action=Marker.ADD
                                m.pose.position.x=c[0]; m.pose.position.y=c[1]; m.pose.position.z=c[2]
                                m.scale.x=0.05; m.scale.y=0.05; m.scale.z=0.05; m.color.a=1.0; m.color.g=1.0
                                self.pub_marker.publish(m)
                                print(f"✅ CUBE DÉTECTÉ à {c}")
            except Exception: pass

        # --- 5. VISU & GRASP ---
        merged = []
        if current_robot_points is not None: merged.append(current_robot_points)
        
        if self.cube_locked and self.static_cube_cloud is not None:
             # Offset manuel TCP (Calcul vectoriel rapide)
            fork_tip_world_hom = np.dot(T_world_tcp, self.fork_tip_offset_vec)
            fork_tip_pos = fork_tip_world_hom[:3]
            
            p_msg = PointStamped()
            p_msg.header.frame_id = "world"; p_msg.header.stamp = rospy.Time.now()
            p_msg.point.x = fork_tip_pos[0]; p_msg.point.y = fork_tip_pos[1]; p_msg.point.z = fork_tip_pos[2]
            self.pub_fork_tip.publish(p_msg)

            if not self.is_grasped:
                # Optimisation: Check sur échantillon réduit
                cloud_check = self.static_cube_cloud
                if len(cloud_check) > 500: cloud_check = cloud_check[::5]
                
                dists = fork_tip_pos[2] - np.max(cloud_check[:, 2])
                if dists < self.contact_threshold:
                    self.is_grasped = True
                    self.T_world_tcp_at_grasp = T_world_tcp
                    self.T_tcp_at_grasp_inv = np.linalg.inv(T_world_tcp)

            if self.is_grasped:
                T_motion = np.dot(T_world_tcp, self.T_tcp_at_grasp_inv)
                ones = np.ones((self.static_cube_cloud.shape[0], 1))
                pts_hom = np.hstack([self.static_cube_cloud, ones])
                self.current_cube_cloud = np.dot(T_motion, pts_hom.T).T[:, :3]
            else:
                self.current_cube_cloud = self.static_cube_cloud
                
            merged.append(self.current_cube_cloud)

        if len(merged) > 0:
            full_cloud = np.vstack(merged)
            if full_cloud.shape[0] > 1024:
                # OPTIMISATION: Random Sampling (O(1)) est bien plus rapide que FPS (O(N^2))
                # Si tu veux de la vitesse pure, décommente la ligne suivante :
                # indices = np.random.choice(full_cloud.shape[0], 1024, replace=False)
                
                # FPS (Plus lent mais plus joli)
                indices = fpsample.bucket_fps_kdline_sampling(full_cloud.astype(np.float32), 1024, h=7)
                full_cloud = full_cloud[indices]
                
            self.publish_cloud(full_cloud, frame_id="world")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    MergedCloudNode().run()