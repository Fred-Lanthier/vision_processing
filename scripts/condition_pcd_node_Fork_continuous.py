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
import shutil
import rospkg
import fpsample
import torch
import cv2
import gc

from utils import compute_T_child_parent_xacro
from sam3_client import Sam3Client

# --- IMPORTS SAM 2 (State of the Art Video Tracking) ---
from sam2.build_sam import build_sam2_video_predictor

rospack = rospkg.RosPack()
sys.path.append(os.path.join(rospack.get_path('vision_processing'), 'scripts'))
try:
    from Compute_3D_point_cloud_from_mesh import RobotMeshLoaderOptimized
    LOADER_AVAILABLE = True
except ImportError:
    LOADER_AVAILABLE = False


class MergedCloudNode:
    def __init__(self, temp_work_dir="/tmp/robot_tracking_live"):
        rospy.init_node('merged_cloud_node', anonymous=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_object = "cube"
        self.detection_confidence = 0.10
        self.cube_locked = False
        
        # --- INIT SAM 3 ---
        self.sam3 = Sam3Client()
        
        # --- INIT SAM 2 VIDEO TRACKING ---
        rospy.loginfo(f"⏳ Chargement du modèle SAM 2 Video Predictor sur {self.device}...")
        self.sam2_checkpoint = os.path.expanduser("~/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt")
        self.sam2_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.sam2_predictor = build_sam2_video_predictor(self.sam2_config, self.sam2_checkpoint, device=self.device)
        
        # --- STATE OF THE ART MEMORY MANAGEMENT ---
        self.frame_count = 0
        self.session_frame_idx = 0
        self.MAX_MEMORY_FRAMES = 100
        self.last_valid_mask = None
        self.inference_state = None
        self.video_height = None
        self.video_width = None
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

        # Temp folder pour le Hack d'initialisation de SAM2
        self.work_dir = temp_work_dir
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
        os.makedirs(self.work_dir)

        rospy.loginfo("✅ SAM 2 Video Predictor Prêt.")

        # --- VARIABLES ROBOT / GRASPING ---
        self.is_grasped = False
        self.static_cube_cloud = None
        self.current_cube_cloud = None
        self.contact_threshold = -0.001
        self.static_cube_cloud_subsampled = None  
        self.fork_cloud_cache = None              
        self.last_joint_hash = None   

        self.fork_tip_offset_vec = np.array([-0.0055, 0.0, 0.1296, 1.0]) 

        package_path = rospack.get_path('vision_processing')
        xacro_file = os.path.join(package_path, 'urdf', 'panda_camera.xacro')
        self.T_tcp_wrist = compute_T_child_parent_xacro(xacro_file, "camera_wrist_link", "panda_TCP")
        self.T_wrist_opt = compute_T_child_parent_xacro(xacro_file, "camera_wrist_optical_frame", "camera_wrist_link")
        self.T_tcp_cam = np.dot(self.T_tcp_wrist, self.T_wrist_opt)
        self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro_file, 'fork_tip', 'panda_TCP')

        self.T_world_tcp_at_grasp = None
        self.T_tcp_at_grasp_inv = None

        self.mesh_loader = None
        if LOADER_AVAILABLE:
            try: self.mesh_loader = RobotMeshLoaderOptimized(xacro_file)
            except: pass
        
        # --- PUBLISHERS & SUBSCRIBERS ---
        self.pub_merged = rospy.Publisher('/vision/merged_cloud', PointCloud2, queue_size=1)
        self.pub_marker = rospy.Publisher('/vision/marker', Marker, queue_size=1)
        self.pub_fork_tip = rospy.Publisher('/vision/debug_fork_tip', PointStamped, queue_size=1)

        self.fx, self.fy, self.cx, self.cy = 604.9, 604.9, 320.0, 240.0
        self.sub_info = rospy.Subscriber("/camera_wrist/color/camera_info", CameraInfo, self.cam_info_cb)
        self.tf_listener = tf.TransformListener()

        sub_rgb = message_filters.Subscriber("/synced/camera_wrist/rgb", Image)
        sub_depth = message_filters.Subscriber("/synced/camera_wrist/depth", Image)
        sub_joints = message_filters.Subscriber("/synced/joint_states", JointState)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth, sub_joints], queue_size=5, slop=0.1
        )
        self.ts.registerCallback(self.callback)
        rospy.loginfo("🚀 MERGED CLOUD (MODE C++ SYNC) PRÊT")

    def cam_info_cb(self, msg):
        self.fx = msg.K[0]; self.cx = msg.K[2]; self.fy = msg.K[4]; self.cy = msg.K[5]
        self.sub_info.unregister()

    def imgmsg_to_numpy(self, msg):
        if msg.encoding == "rgb8": 
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
        elif "32FC1" in msg.encoding: 
            return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
        return None

    def publish_cloud(self, points, frame_id="world"):
        if points is None or len(points) == 0: return
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        cloud_msg = pc2.create_cloud_xyz32(header, points)
        self.pub_merged.publish(cloud_msg)
        
    def _initialize_tracking_session(self, cv_image, mask_np):
        """Initialise une nouvelle session SAM 2 (Démarrage ou Reset Mémoire)"""
        try:
            # Fake dossier vidéo pour tromper l'API Meta
            for f in os.listdir(self.work_dir):
                try: os.remove(os.path.join(self.work_dir, f))
                except: pass
            
            # Sauvegarde d'une image factice pour que init_state ne plante pas
            cv2.imwrite(os.path.join(self.work_dir, "00000.jpg"), cv_image)
            
            self.inference_state = self.sam2_predictor.init_state(video_path=self.work_dir)
            if isinstance(self.inference_state["images"], torch.Tensor):
                self.inference_state["images"] = [t for t in self.inference_state["images"]]
            
            if len(self.inference_state["images"]) > 0:
                self.video_height, self.video_width = self.inference_state["images"][0].shape[-2:]

            mask_input = torch.from_numpy(mask_np).bool().to(self.device)
            if mask_input.ndim > 2: mask_input = mask_input.squeeze()

            self.sam2_predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=1,
                mask=mask_input
            )
            self.session_frame_idx = 0
            return True
        except Exception as e:
            rospy.logerr(f"❌ Erreur d'initialisation SAM 2 : {e}")
            return False

    def callback(self, rgb_msg, depth_msg, joint_msg):
        # 1. RÉCUPÉRATION IMAGES ET TF
        cv_img = self.imgmsg_to_numpy(rgb_msg)
        if cv_img is None: return
        
        cv_depth = self.imgmsg_to_numpy(depth_msg)
        
        try:
            (trans_world_tcp, rot_world_tcp) = self.tf_listener.lookupTransform("world", "panda_hand_tcp", rospy.Time(0))
            T_world_tcp = tft.compose_matrix(translate=trans_world_tcp, angles=tft.euler_from_quaternion(rot_world_tcp))
            T_world_cam = T_world_tcp @ self.T_tcp_cam
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn_throttle(1, "TF non disponible")
            return

        # 2. CACHE DU ROBOT
        current_robot_points = None
        if self.mesh_loader:
            try:
                joint_map = {name: joint_msg.position[i] for i, name in enumerate(joint_msg.name) if "panda" in name}
                joint_hash = tuple(round(v, 4) for v in joint_map.values())
                if joint_hash != self.last_joint_hash:
                    self.fork_cloud_cache = self.mesh_loader.create_point_cloud_fork_tip(joint_map)
                    self.last_joint_hash = joint_hash
                current_robot_points = self.fork_cloud_cache
            except: pass

        # ==========================================================
        # 3. STATE OF THE ART VIDEO TRACKING LOGIC
        # ==========================================================
        current_mask = None
        is_resetting = False

        # Vérifier si on doit faire un Reset de Mémoire (Chunking)
        if self.cube_locked and self.frame_count > 0 and (self.frame_count % self.MAX_MEMORY_FRAMES == 0):
            rospy.loginfo(f"♻️ SAM 2 Memory Reset (Frame {self.frame_count})...")
            is_resetting = True
            
        # CAS A : Non verrouillé OU Reset de session nécessaire
        if not self.cube_locked or is_resetting or self.inference_state is None:
            
            if is_resetting and self.last_valid_mask is not None:
                # RESET : On réutilise le dernier masque connu (Très rapide)
                success = self._initialize_tracking_session(cv_img, self.last_valid_mask)
                if success:
                    current_mask = self.last_valid_mask
            else:
                # DÉMARRAGE : SAM 3 trouve le masque (Lent mais précis)
                rospy.loginfo_throttle(2, "🔍 SAM 3 cherche l'objet...")
                mask, score = self.sam3.segment(rgb_msg, self.target_object, self.detection_confidence)
                
                if mask is not None and score > self.detection_confidence:
                    mask_np = (mask > 0)
                    success = self._initialize_tracking_session(cv_img, mask_np)
                    if success:
                        self.cube_locked = True
                        current_mask = mask_np
                        self.last_valid_mask = mask_np
                        rospy.loginfo("🎯 Objet verrouillé par SAM 3 ! Démarrage du Video Tracking.")

        # CAS B : Tracking Vidéo Normal (Injection Tenseurs)
        else:
            target_h, target_w = self.video_height, self.video_width
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
            if img_rgb.shape[:2] != (target_h, target_w):
                img_rgb = cv2.resize(img_rgb, (target_w, target_h))
            
            # Préparation Tenseur Normalisé
            img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.to(self.device)
            img_tensor = (img_tensor - self.mean) / self.std
            
            # Injection dans l'état
            self.inference_state["images"].append(img_tensor)
            self.inference_state["num_frames"] = len(self.inference_state["images"])

            # Nettoyage RAM (Garder seulement les 6 dernières frames + Init)
            if len(self.inference_state["images"]) > 6:
                idx_to_clear = len(self.inference_state["images"]) - 7
                if self.inference_state["images"][idx_to_clear] is not None:
                    self.inference_state["images"][idx_to_clear] = None

            self.session_frame_idx += 1
            
            # Propagation
            try:
                with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                    for _, _, video_res_masks in self.sam2_predictor.propagate_in_video(
                        self.inference_state,
                        start_frame_idx=self.session_frame_idx,
                        max_frame_num_to_track=1
                    ):
                        raw_mask = (video_res_masks[0] > 0).cpu().numpy()
                        current_mask = raw_mask.squeeze()
                        if current_mask.ndim > 2: current_mask = current_mask[0]
                        break 
                        
                self.last_valid_mask = current_mask
                
                # Vérification grossière si l'objet est perdu (Masque vide)
                if np.sum(current_mask) < 50:
                    rospy.logwarn("⚠️ Masque vide ! Objet perdu. Relance SAM 3...")
                    self.cube_locked = False
                    
            except Exception as e:
                rospy.logerr(f"⚠️ Erreur de tracking SAM 2 : {e}")
                self.cube_locked = False

        self.frame_count += 1

        # ==========================================================
        # 4. CRÉATION DU NUAGE 3D ET PUBLICATIONS
        # ==========================================================
        if current_mask is not None and cv_depth is not None:
            z = cv_depth if cv_depth.dtype == np.float32 else cv_depth / 1000.0
            valid = current_mask & (z > 0.01) & (z < 2.0) & np.isfinite(z)
            
            if np.sum(valid) > 100:
                v, u = np.where(valid)
                z_val = z[valid]
                
                z_threshold = np.min(z_val) + 0.025
                slice_mask = z_val <= z_threshold
                v, u, z_val = v[slice_mask], u[slice_mask], z_val[slice_mask]
                
                if len(z_val) > 20:
                    x = (u - self.cx) * z_val / self.fx
                    y = (v - self.cy) * z_val / self.fy
                    points_cam = np.stack([x, y, z_val], axis=-1)
                    
                    ones = np.ones((points_cam.shape[0], 1))
                    points_world = np.dot(T_world_cam, np.hstack([points_cam, ones]).T).T[:, :3]
                    
                    self.current_cube_cloud = points_world

        # Échantillonnage équilibré
        NUM_TARGET_POINTS = 256
        pts_per_object = NUM_TARGET_POINTS // 2
        merged = []

        if current_robot_points is not None:
            if current_robot_points.shape[0] > pts_per_object:
                idx = np.random.choice(current_robot_points.shape[0], pts_per_object, replace=False)
                merged.append(current_robot_points[idx])
            else:
                merged.append(current_robot_points)

        if self.cube_locked and self.current_cube_cloud is not None:
            if self.current_cube_cloud.shape[0] > pts_per_object:
                idx = np.random.choice(self.current_cube_cloud.shape[0], pts_per_object, replace=False)
                merged.append(self.current_cube_cloud[idx])
            else:
                merged.append(self.current_cube_cloud)

        if len(merged) > 0:
            full_cloud = np.vstack(merged)
            
            if full_cloud.shape[0] > NUM_TARGET_POINTS:
                idx = np.random.choice(full_cloud.shape[0], NUM_TARGET_POINTS, replace=False)
                full_cloud = full_cloud[idx]
            elif full_cloud.shape[0] < NUM_TARGET_POINTS and full_cloud.shape[0] > 0:
                extra_idx = np.random.choice(full_cloud.shape[0], NUM_TARGET_POINTS - full_cloud.shape[0], replace=True)
                full_cloud = np.vstack([full_cloud, full_cloud[extra_idx]])

            self.publish_cloud(full_cloud, frame_id="world")
        
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    MergedCloudNode().run()