#!/usr/bin/env python3
import rospy
import numpy as np
import message_filters
from sensor_msgs.msg import Image, PointCloud2, JointState, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from visualization_msgs.msg import Marker
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tft
import sys
import os
import rospkg
import fpsample
import torch
import time
from PIL import Image as PILImage

# --- 1. IMPORTS & SÉCURITÉ ---
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
    print("⚠️ Attention: RobotMeshLoaderOptimized introuvable.")
    LOADER_AVAILABLE = False

class MergedCloudNode:
    def __init__(self):
        rospy.init_node('merged_cloud_node', anonymous=True)
        
        # --- CONFIGURATION ---
        self.target_object = "cube"
        
        # --- VARIABLES D'ÉTAT ---
        self.cube_locked = False
        self.is_grasped = False
        self.static_cube_cloud = None
        self.current_cube_cloud = None
        
        # Seuil de contact (en mètres)
        # 3 cm est suffisant pour détecter si la fourchette touche le cube
        self.contact_threshold = -0.01 

        # --- GÉOMÉTRIE FOURCHETTE (Basé sur ton Xacro) ---
        # Joint: fork_tip_joint -> origin xyz="-0.0055 0 0.1296"
        # C'est la position du bout de la fourchette par rapport au frame TCP
        self.fork_tip_offset_tcp = np.array([-0.0055, 0.0, 0.1296, 1.0]) 

        # Matrices pour le mouvement relatif
        self.T_world_tcp_at_grasp = None
        self.T_tcp_at_grasp_inv = None

        # --- SAM 3 ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        rospy.loginfo(f"⏳ Chargement SAM 3 sur {self.device}...")
        self.sam_model = build_sam3_image_model()
        if hasattr(self.sam_model, "to"): self.sam_model.to(self.device)
        self.sam_processor = Sam3Processor(self.sam_model, confidence_threshold=0.1)
        rospy.loginfo("✅ SAM 3 chargé.")

        # --- ROBOT LOADER ---
        self.mesh_loader = None
        if LOADER_AVAILABLE:
            urdf_path = os.path.join(rospack.get_path('vision_processing'), 'urdf', 'panda_camera.xacro')
            try: self.mesh_loader = RobotMeshLoaderOptimized(urdf_path)
            except: pass

        # --- PUBLISHERS ---
        self.pub_merged = rospy.Publisher('/vision/merged_cloud', PointCloud2, queue_size=1)
        self.pub_marker = rospy.Publisher('/vision/marker', Marker, queue_size=1)
        
        # Debug: Visualiser où le code pense que se trouve le bout de la fourchette
        self.pub_fork_tip = rospy.Publisher('/vision/debug_fork_tip', PointStamped, queue_size=1)

        # --- CAMÉRA INFO ---
        self.fx, self.fy, self.cx, self.cy = 604.9, 604.9, 320.0, 240.0
        self.sub_info = rospy.Subscriber("/camera_wrist/color/camera_info", CameraInfo, self.cam_info_cb)

        # --- TRANSFORMATIQUE ---
        T_tcp_wrist = tft.compose_matrix(translate=[-0.052, 0.035, -0.045], angles=tft.euler_from_quaternion(tft.quaternion_from_euler(0, -np.pi/2, 0)))
        T_wrist_opt = tft.compose_matrix(translate=[0, 0, 0], angles=tft.euler_from_quaternion(tft.quaternion_from_euler(-np.pi/2, 0, -np.pi/2)))
        self.T_tcp_optical = np.dot(T_tcp_wrist, T_wrist_opt)

        # --- SUBSCRIBERS ---
        sub_rgb = message_filters.Subscriber("/synced/camera_wrist/rgb", Image)
        sub_depth = message_filters.Subscriber("/synced/camera_wrist/depth", Image)
        sub_pose = message_filters.Subscriber("/synced/ee_pose", PoseStamped)
        sub_joints = message_filters.Subscriber("/synced/joint_states", JointState)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth, sub_pose, sub_joints], queue_size=5, slop=0.2
        )
        self.ts.registerCallback(self.callback)
        rospy.loginfo("🚀 Noeud de Fusion prêt avec détection de contact Fourchette.")

    def cam_info_cb(self, msg):
        self.fx = msg.K[0]; self.cx = msg.K[2]; self.fy = msg.K[4]; self.cy = msg.K[5]
        self.sub_info.unregister()

    def imgmsg_to_numpy(self, msg):
        if msg.encoding == "rgb8": return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        elif "32FC1" in msg.encoding: return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        return None

    def pose_to_matrix(self, pose_msg):
        p = pose_msg.position; q = pose_msg.orientation
        return tft.compose_matrix(translate=[p.x, p.y, p.z], angles=tft.euler_from_quaternion([q.x, q.y, q.z, q.w]))

    def publish_cloud(self, points, frame_id="world"):
        if points is None or len(points) == 0: return
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        cloud_msg = pc2.create_cloud_xyz32(header, points)
        self.pub_merged.publish(cloud_msg)

    def callback(self, rgb_msg, depth_msg, ee_pose_msg, joint_msg):
        
        # --- 1. ROBOT CLOUD ---
        current_robot_points = None
        if self.mesh_loader:
            try:
                joint_map = {name: joint_msg.position[i] for i, name in enumerate(joint_msg.name) if "panda" in name}
                current_robot_points = self.mesh_loader.create_point_cloud(joint_map)
            except: pass

        # --- 2. DETECCIÓN INITIALE DU CUBE (SAM 3) ---
        if not self.cube_locked:
            try:
                cv_rgb = self.imgmsg_to_numpy(rgb_msg)
                cv_depth = self.imgmsg_to_numpy(depth_msg)
                
                if cv_rgb is not None and cv_depth is not None:
                    pil_image = PILImage.fromarray(cv_rgb)
                    inference_state = self.sam_processor.set_image(pil_image)
                    output = self.sam_processor.set_text_prompt(state=inference_state, prompt=self.target_object)
                    
                    raw_scores = output["scores"]
                    raw_masks = output["masks"]
                    
                    if len(raw_scores) > 0:
                        scores = np.array(raw_scores).flatten() if not isinstance(raw_scores, torch.Tensor) else raw_scores.detach().cpu().numpy().flatten()
                        masks = np.array(raw_masks) if not isinstance(raw_masks, torch.Tensor) else raw_masks.detach().cpu().numpy()
                        
                        best_idx = np.argmax(scores)
                        if scores[best_idx] > 0.20:
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
                                
                                T_world_tcp = self.pose_to_matrix(ee_pose_msg.pose)
                                T_world_cam = np.dot(T_world_tcp, self.T_tcp_optical)
                                ones = np.ones((points_cam.shape[0], 1))
                                points_world = np.dot(T_world_cam, np.hstack([points_cam, ones]).T).T[:, :3]
                                
                                self.static_cube_cloud = points_world
                                self.cube_locked = True
                                
                                # Visu Marker
                                c = np.mean(points_world, axis=0)
                                m = Marker(); m.header.frame_id="world"; m.header.stamp=rospy.Time.now(); m.type=Marker.SPHERE; m.action=Marker.ADD
                                m.pose.position.x=c[0]; m.pose.position.y=c[1]; m.pose.position.z=c[2]
                                m.scale.x=0.05; m.scale.y=0.05; m.scale.z=0.05; m.color.a=1.0; m.color.g=1.0
                                self.pub_marker.publish(m)
                                print(f"\n✅ CUBE DÉTECTÉ ET VERROUILLÉ.")
            except Exception: pass

        # --- 3. GESTION DU GRASP (BASÉE SUR LE BOUT DE LA FOURCHETTE) ---
        self.current_cube_cloud = None
        
        if self.cube_locked and self.static_cube_cloud is not None:
            
            # A. Calcul de la position actuelle du BOUT DE LA FOURCHETTE
            T_world_tcp_current = self.pose_to_matrix(ee_pose_msg.pose)
            
            # Application de l'offset Xacro : P_world = T_world_tcp * P_local_offset
            fork_tip_world_hom = np.dot(T_world_tcp_current, self.fork_tip_offset_tcp)
            fork_tip_pos = fork_tip_world_hom[:3]
            
            # Debug: Publier où on pense que le bout est
            p_msg = PointStamped()
            p_msg.header.frame_id = "world"
            p_msg.header.stamp = rospy.Time.now()
            p_msg.point.x = fork_tip_pos[0]; p_msg.point.y = fork_tip_pos[1]; p_msg.point.z = fork_tip_pos[2]
            self.pub_fork_tip.publish(p_msg)

            # B. Logique de Contact
            if not self.is_grasped:
                # Calcul de distance : Bout fourchette <-> Nuage Cube
                # On utilise la norme L2 min entre le point unique (fourchette) et tous les points du cube
                dists = np.linalg.norm(self.static_cube_cloud - fork_tip_pos, axis=1)
                min_dist = np.min(dists)
                
                # Si le bout de la fourchette touche le cube
                if min_dist < self.contact_threshold:
                    print(f"\n🍴 CONTACT FOURCHETTE ! (Dist: {min_dist:.3f}m). Objet attaché.")
                    self.is_grasped = True
                    self.T_world_tcp_at_grasp = T_world_tcp_current
                    self.T_tcp_at_grasp_inv = np.linalg.inv(T_world_tcp_current)

            # C. Mise à jour position Cube
            if self.is_grasped:
                # Mouvement relatif du TCP (le cube suit le TCP)
                T_motion = np.dot(T_world_tcp_current, self.T_tcp_at_grasp_inv)
                
                ones = np.ones((self.static_cube_cloud.shape[0], 1))
                pts_hom = np.hstack([self.static_cube_cloud, ones])
                self.current_cube_cloud = np.dot(T_motion, pts_hom.T).T[:, :3]
            else:
                self.current_cube_cloud = self.static_cube_cloud

        # --- 4. FUSION ET PUBLICATION ---
        merged = []
        if current_robot_points is not None: merged.append(current_robot_points)
        if self.current_cube_cloud is not None: merged.append(self.current_cube_cloud)
            
        if len(merged) > 0:
            full_cloud = np.vstack(merged)
            if full_cloud.shape[0] > 1024:
                # --- DEBUT CHRONO ---
                start_t = time.time()
                
                # Ton code existant
                indices = fpsample.bucket_fps_kdline_sampling(full_cloud.astype(np.float32), 1024, h=7)
                full_cloud = full_cloud[indices]
                
                # --- FIN CHRONO ---
                end_t = time.time()
                duration = end_t - start_t
                
                # On affiche le temps en millisecondes
                print(f"⏱️ FPS Sampling ({len(indices)} pts): {duration:.4f} sec")
            
            self.publish_cloud(full_cloud, frame_id="world")
            
            if self.is_grasped:
                print(f"\r🚀 Transport en cours... ", end="")
            elif self.cube_locked:
                print(f"\r👀 Cible verrouillée, en approche... ", end="")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    MergedCloudNode().run()