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

from utils import compute_T_child_parent_xacro

# --- IMPORTS SAM 2 ---
from sam2.build_sam import build_sam2_video_predictor

rospack = rospkg.RosPack()
sys.path.append(os.path.join(rospack.get_path('vision_processing'), 'scripts'))
try:
    from Compute_3D_point_cloud_from_mesh import RobotMeshLoaderOptimized
    LOADER_AVAILABLE = True
except ImportError:
    LOADER_AVAILABLE = False


class FastTrackingNode:
    def __init__(self, temp_work_dir="/tmp/robot_tracking_live"):
        rospy.init_node('fast_tracking_node', anonymous=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- INIT SAM 2 VIDEO TRACKING ---
        rospy.loginfo(f"⏳ Chargement du modèle SAM 2 Video Predictor sur {self.device}...")
        self.sam2_checkpoint = os.path.expanduser("~/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt")
        self.sam2_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.sam2_predictor = build_sam2_video_predictor(self.sam2_config, self.sam2_checkpoint, device=self.device)
        
        # --- STATE MANAGEMENT ---
        self.cube_locked = False
        self.waiting_for_sam3 = False
        self.pending_init_mask = None
        
        # --- MEMORY MANAGEMENT ---
        self.frame_count = 0
        self.session_frame_idx = 0
        self.MAX_MEMORY_FRAMES = 30
        self.last_valid_mask = None
        self.inference_state = None
        self.video_height = None
        self.video_width = None
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

        self.work_dir = temp_work_dir
        if os.path.exists(self.work_dir): shutil.rmtree(self.work_dir)
        os.makedirs(self.work_dir)
        rospy.loginfo("✅ SAM 2 Video Predictor Prêt.")

        # --- VARIABLES ROBOT / GRASPING ---
        self.last_known_cube_cloud = None # <-- LA MÉMOIRE 3D (FREEZE EFFECT)
        self.fork_cloud_cache = None              
        self.last_joint_hash = None   

        package_path = rospack.get_path('vision_processing')
        xacro_file = os.path.join(package_path, 'urdf', 'panda_camera.xacro')
        self.T_tcp_wrist = compute_T_child_parent_xacro(xacro_file, "camera_wrist_link", "panda_TCP")
        self.T_wrist_opt = compute_T_child_parent_xacro(xacro_file, "camera_wrist_optical_frame", "camera_wrist_link")
        self.T_tcp_cam = np.dot(self.T_tcp_wrist, self.T_wrist_opt)

        self.mesh_loader = None
        if LOADER_AVAILABLE:
            try: self.mesh_loader = RobotMeshLoaderOptimized(xacro_file)
            except: pass
        
        # --- PUBLISHERS & SUBSCRIBERS ---
        self.pub_merged = rospy.Publisher('/vision/merged_cloud', PointCloud2, queue_size=1)
        
        # Communication Asynchrone avec SAM 3
        self.pub_sam3_req = rospy.Publisher('/vision/sam3_request', Image, queue_size=1)
        self.sub_sam3_reply = rospy.Subscriber('/vision/sam3_reply', Image, self.sam3_reply_cb, queue_size=1)

        self.fx, self.fy, self.cx, self.cy = 604.9, 604.9, 320.0, 240.0
        self.sub_info = rospy.Subscriber("/camera_wrist/color/camera_info", CameraInfo, self.cam_info_cb)
        self.tf_listener = tf.TransformListener()

        sub_rgb = message_filters.Subscriber("/synced/camera_wrist/rgb", Image)
        sub_depth = message_filters.Subscriber("/synced/camera_wrist/depth", Image)
        sub_joints = message_filters.Subscriber("/synced/joint_states", JointState)

        self.ts = message_filters.ApproximateTimeSynchronizer([sub_rgb, sub_depth, sub_joints], queue_size=5, slop=0.1)
        self.ts.registerCallback(self.callback)
        rospy.loginfo("🚀 TRACKER HAUTE FRÉQUENCE PRÊT")

    def sam3_reply_cb(self, msg):
        """Callback asynchrone : Reçoit le masque de SAM 3 quand il est prêt"""
        mask_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        
        # On vérifie si SAM 3 a vraiment trouvé quelque chose (s'il y a des pixels blancs)
        if np.sum(mask_np) > 0:
            self.pending_init_mask = (mask_np > 0)
            rospy.loginfo("📥 Masque de SAM 3 reçu ! Reprise du tracking immédiate.")
        else:
            # SAM 3 n'a rien trouvé sur cette frame, on le laisse réessayer
            rospy.loginfo_throttle(1, "📥 L'objet n'est toujours pas là. On continue d'attendre...")
            
        # Dans tous les cas, on a reçu une réponse de SAM 3, on n'est plus en attente !
        self.waiting_for_sam3 = False

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
        try:
            for f in os.listdir(self.work_dir):
                try: os.remove(os.path.join(self.work_dir, f))
                except: pass
            
            cv2.imwrite(os.path.join(self.work_dir, "00000.jpg"), cv_image)
            
            self.inference_state = self.sam2_predictor.init_state(video_path=self.work_dir)
            if isinstance(self.inference_state["images"], torch.Tensor):
                self.inference_state["images"] = [t for t in self.inference_state["images"]]
            
            if len(self.inference_state["images"]) > 0:
                self.video_height, self.video_width = self.inference_state["images"][0].shape[-2:]

            mask_input = torch.from_numpy(mask_np).bool().to(self.device)
            if mask_input.ndim > 2: mask_input = mask_input.squeeze()

            self.sam2_predictor.add_new_mask(
                inference_state=self.inference_state, frame_idx=0, obj_id=1, mask=mask_input)
            self.session_frame_idx = 0
            return True
        except Exception as e:
            rospy.logerr(f"❌ Erreur d'initialisation SAM 2 : {e}")
            return False

    def callback(self, rgb_msg, depth_msg, joint_msg):
        # 1. RÉCUPÉRATION (30 FPS constant)
        cv_img = self.imgmsg_to_numpy(rgb_msg)
        if cv_img is None: return
        cv_depth = self.imgmsg_to_numpy(depth_msg)
        
        try:
            (trans_world_tcp, rot_world_tcp) = self.tf_listener.lookupTransform("world", "panda_hand_tcp", rospy.Time(0))
            T_world_tcp = tft.compose_matrix(translate=trans_world_tcp, angles=tft.euler_from_quaternion(rot_world_tcp))
            T_world_cam = T_world_tcp @ self.T_tcp_cam
        except:
            return

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
        # 2. LOGIQUE ASYNCHRONE ET TRACKING SAM 2
        # ==========================================================
        current_mask = None
        is_resetting = self.cube_locked and self.frame_count > 0 and (self.frame_count % self.MAX_MEMORY_FRAMES == 0)

        # A. Si SAM 3 vient de nous envoyer un masque tout frais !
        if self.pending_init_mask is not None:
            success = self._initialize_tracking_session(cv_img, self.pending_init_mask)
            if success:
                self.cube_locked = True
                current_mask = self.pending_init_mask
                self.last_valid_mask = self.pending_init_mask
            self.pending_init_mask = None

        # B. On a perdu l'objet : Demander de l'aide à SAM 3
        elif not self.cube_locked:
            if not self.waiting_for_sam3:
                self.pub_sam3_req.publish(rgb_msg)
                self.waiting_for_sam3 = True
                rospy.logwarn("⚠️ Cible perdue. Appel asynchrone à SAM 3... (Nuage 3D figé en attendant)")

        # C. Reset Mémoire interne de SAM 2 (Chunking)
        elif is_resetting:
            success = self._initialize_tracking_session(cv_img, self.last_valid_mask)
            if success: current_mask = self.last_valid_mask

        # D. Tracking Video Normal avec SAM 2
        else:
            target_h, target_w = self.video_height, self.video_width
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            if img_rgb.shape[:2] != (target_h, target_w): img_rgb = cv2.resize(img_rgb, (target_w, target_h))
            
            img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
            img_tensor = (img_tensor.to(self.device) - self.mean) / self.std
            
            self.inference_state["images"].append(img_tensor)
            self.inference_state["num_frames"] = len(self.inference_state["images"])

            if len(self.inference_state["images"]) > 6:
                idx_to_clear = len(self.inference_state["images"]) - 7
                if self.inference_state["images"][idx_to_clear] is not None:
                    self.inference_state["images"][idx_to_clear] = None

            self.session_frame_idx += 1
            
            try:
                with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                    for _, _, video_res_masks in self.sam2_predictor.propagate_in_video(
                        self.inference_state, start_frame_idx=self.session_frame_idx, max_frame_num_to_track=1
                    ):
                        raw_mask = (video_res_masks[0] > 0).cpu().numpy()
                        current_mask = raw_mask.squeeze()
                        if current_mask.ndim > 2: current_mask = current_mask[0]
                        break 
                        
                self.last_valid_mask = current_mask
                
                if np.sum(current_mask) < 50:
                    self.cube_locked = False # Déclenchera l'appel SAM 3 à la frame suivante
            except:
                self.cube_locked = False

        self.frame_count += 1

        # ==========================================================
        # 3. CRÉATION DU NUAGE 3D ET GESTION DU "FREEZE"
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
                    
                    # Mise à jour de la mémoire 3D !
                    self.last_known_cube_cloud = points_world

        # ==========================================================
        # 4. ÉCHANTILLONNAGE ET PUBLICATION (Toujours à 30 FPS !)
        # ==========================================================
        NUM_TARGET_POINTS = 256
        pts_per_object = NUM_TARGET_POINTS // 2
        merged = []

        if current_robot_points is not None:
            if current_robot_points.shape[0] > pts_per_object:
                idx = np.random.choice(current_robot_points.shape[0], pts_per_object, replace=False)
                merged.append(current_robot_points[idx])
            else:
                merged.append(current_robot_points)

        # On utilise le nuage connu (Même s'il est figé car SAM 3 cherche)
        if self.last_known_cube_cloud is not None:
            if self.last_known_cube_cloud.shape[0] > pts_per_object:
                idx = np.random.choice(self.last_known_cube_cloud.shape[0], pts_per_object, replace=False)
                merged.append(self.last_known_cube_cloud[idx])
            else:
                merged.append(self.last_known_cube_cloud)

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
    FastTrackingNode().run()