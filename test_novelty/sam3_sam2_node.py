#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import rospkg
import sys
import os
import cv2
import numpy as np
import torch
import threading
import time
from collections import deque
import pyrealsense2 as rs
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32MultiArray, Header
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import tf2_ros
from scipy.spatial.transform import Rotation

# Import du service SAM3 local
from vision_processing.srv import Sam3Segment, Sam3SegmentRequest

# Résolution des chemins pour le projet
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
muggled_sam_path = os.path.join(pkg_path, "third_party", "muggled_sam")
if muggled_sam_path not in sys.path:
    sys.path.insert(0, muggled_sam_path)

# Imports Muggled SAM / Samurai
try:
    from muggled_sam.make_sam import make_sam_from_state_dict
    from muggled_sam.demo_helpers.video_data_storage import SAMVideoObjectResults
except ImportError as e:
    print(f"❌ Erreur Import Muggled SAM : {e}")
    sys.exit(1)


class Sam3Sam2Node:
    def __init__(self):
        rospy.init_node('sam3_sam2_node')
        
        # --- PARAMETRES ROS ---
        self.sam2_model_path = rospy.get_param("~sam2_model_path", "/home/flanthier/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt")
        self.prompt = rospy.get_param("~prompt", "hand")
        self.target_frame = rospy.get_param("~target_frame", "world")
        self.camera_frame = rospy.get_param("~camera_frame", "camera_wrist_optical_frame")
        
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # --- INITIALISATION SAM 2 ---
        rospy.loginfo(f"⏳ Chargement du modèle SAM 2 Tiny depuis : {self.sam2_model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        self.imgenc_config_dict = {"max_side_length": None, "use_square_sizing": True}
        
        try:
            _, self.sammodel = make_sam_from_state_dict(self.sam2_model_path)
            self.sammodel.to(device=self.device, dtype=self.dtype)
        except Exception as e:
            rospy.logerr(f"❌ Impossible de charger SAM2 : {e}")
            sys.exit(1)
            
        self.obj_mem = None
        self.frame_idx = 0
        self.last_init_time = 0
        
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # --- INITIALISATION REALSENSE (NATIVE) ---
        rospy.loginfo("⏳ Démarrage du stream RealSense natif...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        try:
            profile = self.pipeline.start(config)
        except Exception as e:
            rospy.logerr(f"❌ RealSense introuvable ou erreur : {e}")
            sys.exit(1)
            
        # Alignement de la profondeur sur la couleur
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        # Intrinsèques
        color_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.cx = intrinsics.ppx
        self.cy = intrinsics.ppy
        
        # --- PUBLISHERS ROS ---
        self.pub_obs = rospy.Publisher('/perception/obstacles', PointCloud2, queue_size=1)
        self.pub_debug = rospy.Publisher('/vision/sam2_tracked_mask', Image, queue_size=1)
        self.pub_viz_obs = rospy.Publisher('/viz/sam2_obstacles', PointCloud2, queue_size=1)
        
        rospy.loginfo("✅ Nœud SAM3+SAM2 Prêt. Pipeline RealSense active.")
        
        # Lancement du Thread d'Intelligence Artificielle
        self.ai_thread = threading.Thread(target=self.ai_worker)
        self.ai_thread.daemon = True
        self.ai_thread.start()
        
        # Lancement de la boucle principale (Producteur de frames)
        self.run_loop()

    def init_with_sam3(self, rgb_cv):
        rospy.loginfo(f"⏳ Appel de SAM3 pour le prompt : '{self.prompt}' ...")
        rospy.wait_for_service('/sam3/segment', timeout=45.0)
        
        try:
            sam3_segment = rospy.ServiceProxy('/sam3/segment', Sam3Segment)
            req = Sam3SegmentRequest()
            req.rgb_image = self.bridge.cv2_to_imgmsg(rgb_cv, encoding="bgr8")
            req.text_prompt = self.prompt
            req.confidence_threshold = 0.1
            
            resp = sam3_segment(req)
            if not resp.success:
                rospy.logwarn(f"❌ SAM3 n'a pas trouvé '{self.prompt}'. Réessai à la prochaine frame...")
                return False
                
            rospy.loginfo(f"✅ SAM3 a détecté '{self.prompt}' avec confiance {resp.confidence:.2f}")
            
            init_mask_cv = self.bridge.imgmsg_to_cv2(resp.mask, "mono8")
            init_mask_norm = (init_mask_cv > 0).astype(np.float32)
            init_mask_tensor = torch.from_numpy(init_mask_norm).to(self.device).unsqueeze(0).unsqueeze(0)
            
            init_encoded_img, _, _ = self.sammodel.encode_image(rgb_cv, **self.imgenc_config_dict)
            
            # Initialisation stricte comme dans ton script SAM2.1
            init_mem = self.sammodel.initialize_from_mask(init_encoded_img, init_mask_tensor > 0)
            
            # Important : on crée SAMVideoObjectResults d'abord, puis on le transfère atomiquement
            new_obj_mem = SAMVideoObjectResults.create()
            new_obj_mem.store_prompt_result(self.frame_idx, init_mem)
            self.obj_mem = new_obj_mem
            
            rospy.loginfo("🚀 Tracking SAM 2.1 Initialisé !")
            return True
            
        except Exception as e:
            rospy.logerr(f"❌ Erreur Init SAM3 : {e}")
            return False

    def deproject_to_3d(self, mask_cv, depth_cv):
        # 1. Masques booléens (Vectorisé)
        mask_bool = mask_cv > 0
        valid_depth = (depth_cv > 100) & (depth_cv <= 3000)
        valid_mask = mask_bool & valid_depth
        
        ys, xs = np.where(valid_mask)
        
        if len(xs) == 0:
            return None
            
        ds = depth_cv[ys, xs]
        
        # 2. Sampler EXACTEMENT 100 points
        max_points = 500
        if len(xs) >= max_points:
            indices = np.random.choice(len(xs), max_points, replace=False)
        else:
            indices = np.random.choice(len(xs), max_points, replace=True)
            
        xs_sampled = xs[indices]
        ys_sampled = ys[indices]
        ds_sampled = ds[indices]
        
        # 3. Déprojection mathématique vectorisée (Ultra-rapide)
        Z = ds_sampled / 1000.0
        X = (xs_sampled - self.cx) * Z / self.fx
        Y = (ys_sampled - self.cy) * Z / self.fy
        
        points_3d = np.stack((X, Y, Z), axis=-1)
        return points_3d.astype(np.float32)

    def transform_to_world(self, points_3d_cam, frame_time):
        try:
            trans = self.tf_buffer.lookup_transform(self.target_frame, self.camera_frame, frame_time, rospy.Duration(0.1))
        except Exception as e:
            return None
            
        t_vec = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
        rw = trans.transform.rotation.w
        rx = trans.transform.rotation.x
        ry = trans.transform.rotation.y
        rz = trans.transform.rotation.z
        
        R = Rotation.from_quat([rx, ry, rz, rw]).as_matrix()
        points_world = (R @ points_3d_cam.T).T + t_vec
        return points_world.astype(np.float32)

    def ai_worker(self):
        """Thread Consommateur : S'occupe exclusivement des calculs PyTorch sans bloquer la lecture caméra"""
        rate = rospy.Rate(60) # Limite supérieure de FPS pour l'IA
        while not rospy.is_shutdown():
            with self.frame_lock:
                frame_data = self.latest_frame
                self.latest_frame = None # Frame consommée
                
            if frame_data is None:
                rate.sleep()
                continue
                
            rgb_cv, depth_cv, frame_time = frame_data
            
            # --- INITIALISATION SAM3 ---
            if self.obj_mem is None:
                if time.time() - self.last_init_time > 1.0:
                    self.last_init_time = time.time()
                    self.init_with_sam3(rgb_cv)
                continue
                
            # --- TRACKING SAM2.1 TINY ---
            try:
                encoded_imgs_list, _, _ = self.sammodel.encode_image(rgb_cv, **self.imgenc_config_dict)
                score, best_idx, preds, mem_enc, obj_ptr = self.sammodel.step_video_masking(
                    encoded_imgs_list, **self.obj_mem.to_dict()
                )
                
                if score.item() > 0:
                    self.obj_mem.store_frame_result(self.frame_idx, mem_enc, obj_ptr)
                    
                    MAX_MEMORY_FRAMES = 6
                    if hasattr(self.obj_mem, '_frame_results'):
                        while len(self.obj_mem._frame_results) > MAX_MEMORY_FRAMES:
                            self.obj_mem._frame_results.pop(0)
                    elif hasattr(self.obj_mem, 'frame_results'):
                        while len(self.obj_mem.frame_results) > MAX_MEMORY_FRAMES:
                            self.obj_mem.frame_results.pop(0)
                            
                    mask_f = preds[0, best_idx].cpu().float().numpy().squeeze()
                    mask_full = cv2.resize(mask_f, (rgb_cv.shape[1], rgb_cv.shape[0]), interpolation=cv2.INTER_LINEAR)
                    disp_mask_cv = ((mask_full > 0.0) * 255).astype(np.uint8)
                    
                    self.pub_debug.publish(self.bridge.cv2_to_imgmsg(disp_mask_cv, encoding="mono8"))
                    
                    points_cam = self.deproject_to_3d(disp_mask_cv, depth_cv)
                    if points_cam is not None:
                        points_world = self.transform_to_world(points_cam, frame_time)
                        if points_world is not None:
                            header = Header()
                            header.stamp = rospy.Time.now()
                            header.frame_id = self.target_frame
                            cloud_msg = pc2.create_cloud_xyz32(header, points_world.tolist())
                            
                            self.pub_obs.publish(cloud_msg)
                            self.pub_viz_obs.publish(cloud_msg)
                else:
                    rospy.logwarn_throttle(1.0, "⚠️ SAM2 a perdu la trace de la main ! Re-initialisation...")
                    self.obj_mem = None
                
                self.frame_idx += 1
            except Exception as e:
                rospy.logerr_throttle(1.0, f"Erreur dans le thread AI: {e}")

    def run_loop(self):
        """Thread Producteur : Ne fait QUE lire le buffer de la RealSense le plus vite possible"""
        try:
            while not rospy.is_shutdown():
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                    
                # Copie explicite pour éviter d'écraser la mémoire pendant que le thread AI travaille
                rgb_cv = np.asanyarray(color_frame.get_data()).copy()
                depth_cv = np.asanyarray(depth_frame.get_data()).copy()
                frame_time = rospy.Time.now()
                
                with self.frame_lock:
                    self.latest_frame = (rgb_cv, depth_cv, frame_time)
        finally:
            self.pipeline.stop()

if __name__ == '__main__':
    node = Sam3Sam2Node()