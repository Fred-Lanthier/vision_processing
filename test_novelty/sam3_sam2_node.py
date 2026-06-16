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
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
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
        self.target_prompt = rospy.get_param("~target_prompt", "green cube")
        self.obstacle_prompt = rospy.get_param("~obstacle_prompt", "person")
        self.tracking_rate_hz = float(rospy.get_param("~tracking_rate_hz", 5.0))
        self.sam3_init_retry_period = max(
            0.1, float(rospy.get_param("~sam3_init_retry_period", 8.0)))
        self.sam3_init_backoff_after_failures = int(rospy.get_param(
            "~sam3_init_backoff_after_failures", 2))
        self.sam3_init_backoff_period = max(
            self.sam3_init_retry_period,
            float(rospy.get_param("~sam3_init_backoff_period", 30.0)))
        self.sam2_max_side_length = rospy.get_param("~sam2_max_side_length", None)
        self.log_timing = bool(rospy.get_param("~log_timing", True))
        
        self.target_frame = rospy.get_param("~target_frame", "world")
        self.camera_frame = rospy.get_param("~camera_frame", "camera_wrist_optical_frame")
        
        self.publish_all_as_obstacles = rospy.get_param("~publish_all_as_obstacles", True)
        
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # --- INITIALISATION SAM 2 ---
        rospy.loginfo(f"⏳ Chargement du modèle SAM 2 Tiny depuis : {self.sam2_model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        self.imgenc_config_dict = {
            "max_side_length": self.sam2_max_side_length,
            "use_square_sizing": True,
        }
        
        try:
            _, self.sammodel = make_sam_from_state_dict(self.sam2_model_path)
            self.sammodel.to(device=self.device, dtype=self.dtype)
            self.sammodel.eval()
        except Exception as e:
            rospy.logerr(f"❌ Impossible de charger SAM2 : {e}")
            sys.exit(1)
            
        # DUAL TRACKING MEMORY
        self.obj_mem_target = None
        self.obj_mem_obstacle = None
        
        self.frame_idx = 0
        self.last_init_time = 0
        self.next_init_time = 0.0
        self.init_fail_count = 0
        
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.pipeline = None
        self.align = None
        self.fx = self.fy = self.cx = self.cy = None
        
        # --- PUBLISHERS ROS ---
        self.pub_debug = rospy.Publisher('/vision/sam2_tracked_mask', Image, queue_size=1)
        
        rospy.loginfo("✅ Nœud DUAL SAM3+SAM2 Prêt (Mask-Only Output).")
        
        self.ai_thread = threading.Thread(target=self.ai_worker)
        self.ai_thread.daemon = True
        self.ai_thread.start()
        self.run_loop()

    def init_with_sam3(self, rgb_cv):
        rospy.loginfo("⏳ Initialisation DUAL via SAM3...")
        service_name = '/sam3/segment'
        try:
            rospy.wait_for_service(service_name, timeout=90.0)
            sam3_segment = rospy.ServiceProxy(service_name, Sam3Segment)
        except rospy.ROSException:
            rospy.logerr(f"❌ Service {service_name} non disponible après 90s.")
            return False
        
        with torch.inference_mode():
            init_encoded_img, _, _ = self.sammodel.encode_image(rgb_cv, **self.imgenc_config_dict)

        # 1. INIT TARGET
        if self.obj_mem_target is None:
            req = Sam3SegmentRequest(rgb_image=self.bridge.cv2_to_imgmsg(rgb_cv, encoding="bgr8"), 
                                     text_prompt=self.target_prompt, confidence_threshold=0.1)
            resp = sam3_segment(req)
            if resp.success:
                mask_t = torch.from_numpy(self.bridge.imgmsg_to_cv2(resp.mask, "mono8") > 0).to(self.device).float().unsqueeze(0).unsqueeze(0)
                with torch.inference_mode():
                    init_mem, init_ptr = self.sammodel.initialize_from_mask(init_encoded_img, mask_t > 0)
                self.obj_mem_target = SAMVideoObjectResults.create().store_prompt_result(self.frame_idx, init_mem, init_ptr)
                rospy.loginfo(f"🎯 Target '{self.target_prompt}' initialisée.")
            else:
                rospy.logwarn(f"⚠️ Échec SAM3 Target: {resp.message}")

        # 2. INIT OBSTACLE (si non géré par publish_all_as_obstacles)
        if not self.publish_all_as_obstacles and self.obj_mem_obstacle is None:
            req = Sam3SegmentRequest(rgb_image=self.bridge.cv2_to_imgmsg(rgb_cv, encoding="bgr8"), 
                                     text_prompt=self.obstacle_prompt, confidence_threshold=0.1)
            resp = sam3_segment(req)
            if resp.success:
                mask_t = torch.from_numpy(self.bridge.imgmsg_to_cv2(resp.mask, "mono8") > 0).to(self.device).float().unsqueeze(0).unsqueeze(0)
                with torch.inference_mode():
                    init_mem, init_ptr = self.sammodel.initialize_from_mask(init_encoded_img, mask_t > 0)
                self.obj_mem_obstacle = SAMVideoObjectResults.create().store_prompt_result(self.frame_idx, init_mem, init_ptr)
                rospy.loginfo(f"🚧 Obstacle '{self.obstacle_prompt}' initialisé.")

        return self.obj_mem_target is not None

    def _record_sam3_init_result(self, success):
        now = time.time()
        self.last_init_time = now
        if success:
            self.init_fail_count = 0
            self.next_init_time = 0.0
            return

        self.init_fail_count += 1
        retry_period = self.sam3_init_retry_period
        max_fast_failures = self.sam3_init_backoff_after_failures
        if max_fast_failures > 0 and self.init_fail_count >= max_fast_failures:
            retry_period = self.sam3_init_backoff_period
            rospy.logwarn_throttle(
                10.0,
                "SAM3 target init failed %d times; backing off retries to %.1fs.",
                self.init_fail_count,
                retry_period,
            )
        self.next_init_time = now + retry_period

    def ai_worker(self):
        rate = rospy.Rate(max(self.tracking_rate_hz, 0.1))
        while not rospy.is_shutdown():
            with self.frame_lock:
                frame_data = self.latest_frame
                self.latest_frame = None
            if frame_data is None:
                rate.sleep()
                continue
            rgb_cv, depth_cv, frame_time = frame_data
            
            # Vérification de l'état d'initialisation
            needs_init = False
            if self.obj_mem_target is None:
                needs_init = True
                
            if needs_init:
                now = time.time()
                if now >= self.next_init_time:
                    try:
                        success = self.init_with_sam3(rgb_cv)
                    except Exception as e:
                        rospy.logerr_throttle(5.0, f"SAM3 initialization error: {e}")
                        success = False
                    self._record_sam3_init_result(success)
                if self.obj_mem_target is None:
                    # Publish empty mask if not initialized
                    empty_mask = np.zeros((rgb_cv.shape[0], rgb_cv.shape[1]), dtype=np.uint8)
                    msg_out = self.bridge.cv2_to_imgmsg(empty_mask, encoding="mono8")
                    msg_out.header.stamp = frame_time
                    msg_out.header.frame_id = self.camera_frame
                    self.pub_debug.publish(msg_out)
                    rate.sleep()
                    continue

            try:
                t0 = time.perf_counter()
                with torch.inference_mode():
                    encoded_imgs_list, _, _ = self.sammodel.encode_image(rgb_cv, **self.imgenc_config_dict)
                
                target_mask = None
                # --- PROCESS TARGET ---
                if self.obj_mem_target:
                    with torch.inference_mode():
                        score, best_idx, preds, mem_enc, obj_ptr = self.sammodel.step_video_masking(encoded_imgs_list, **self.obj_mem_target.to_dict())
                    if score.item() > 0:
                        self.obj_mem_target.store_frame_result(self.frame_idx, mem_enc, obj_ptr)
                        mask_f = cv2.resize(preds[0, best_idx].cpu().float().numpy().squeeze(), (rgb_cv.shape[1], rgb_cv.shape[0]))
                        target_mask = (mask_f > 0)
                        
                        # Publish 2D Mask
                        mask_img = (target_mask * 255).astype(np.uint8)
                        msg_out = self.bridge.cv2_to_imgmsg(mask_img, encoding="mono8")
                        msg_out.header.stamp = frame_time
                        msg_out.header.frame_id = self.camera_frame
                        self.pub_debug.publish(msg_out)
                    else:
                        self.obj_mem_target = None
                        self.next_init_time = 0.0
                        # Publish empty mask if tracking lost
                        empty_mask = np.zeros((rgb_cv.shape[0], rgb_cv.shape[1]), dtype=np.uint8)
                        msg_out = self.bridge.cv2_to_imgmsg(empty_mask, encoding="mono8")
                        msg_out.header.stamp = frame_time
                        msg_out.header.frame_id = self.camera_frame
                        self.pub_debug.publish(msg_out)
                
                self.frame_idx += 1
                if self.log_timing:
                    rospy.loginfo_throttle(5.0, f"⏱️ [TIMING] SAM2 Tracking: {(time.perf_counter() - t0)*1000:.2f} ms")
            except Exception as e:
                rospy.logerr_throttle(1.0, f"Erreur AI: {e}")
            rate.sleep()

    def ros_image_callback(self, rgb_msg, depth_msg):
        """Callback pour les topics ROS (Gazebo ou Driver RealSense ROS)"""
        try:
            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
            
            with self.frame_lock:
                self.latest_frame = (rgb_cv.copy(), depth_cv.copy(), rgb_msg.header.stamp)
        except Exception as e:
            rospy.logerr_throttle(5, f"Erreur décodage image ROS: {e}")

    def run_loop(self):
        """Thread Producteur : Soit RealSense native, soit ROS Topics"""
        rgb_topic = rospy.get_param("~rgb_topic", None)
        depth_topic = rospy.get_param("~depth_topic", None)
        
        if rgb_topic and depth_topic:
            rospy.loginfo(f"🛰️ Mode ROS : Souscription à {rgb_topic} et {depth_topic}")
            # On utilise message_filters pour la synchro
            from message_filters import Subscriber, ApproximateTimeSynchronizer
            self.sub_rgb = Subscriber(rgb_topic, Image)
            self.sub_depth = Subscriber(depth_topic, Image)
            self.ts = ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth], queue_size=10, slop=0.05)
            self.ts.registerCallback(self.ros_image_callback)
            
            # On récupère aussi les infos caméra pour les intrinsèques
            info_topic = rospy.get_param("~info_topic", rgb_topic.replace("image_raw", "camera_info"))
            from sensor_msgs.msg import CameraInfo
            try:
                info_msg = rospy.wait_for_message(info_topic, CameraInfo, timeout=5.0)
                self.fx = info_msg.K[0]; self.fy = info_msg.K[4]
                self.cx = info_msg.K[2]; self.cy = info_msg.K[5]
                rospy.loginfo(f"📸 Intrinsèques récupérées via {info_topic}")
            except:
                rospy.logwarn("⚠️ Impossible de lire CameraInfo, utilisation des valeurs par défaut.")
                self.fx, self.fy, self.cx, self.cy = 615.0, 615.0, 320.0, 240.0

            rospy.spin()
        else:
            if rs is None:
                rospy.logerr("❌ pyrealsense2 is required when rgb_topic/depth_topic are not provided.")
                return

            rospy.loginfo("⏳ Démarrage du stream RealSense natif...")
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            try:
                profile = self.pipeline.start(config)
                self.align = rs.align(rs.stream.color)
                intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                self.fx, self.fy, self.cx, self.cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
                
                while not rospy.is_shutdown():
                    frames = self.pipeline.wait_for_frames()
                    aligned_frames = self.align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    if not color_frame or not depth_frame: continue
                        
                    rgb_cv = np.asanyarray(color_frame.get_data()).copy()
                    depth_cv = np.asanyarray(depth_frame.get_data()).copy()
                    frame_time = rospy.Time.now()
                    
                    with self.frame_lock:
                        self.latest_frame = (rgb_cv, depth_cv, frame_time)
            except Exception as e:
                rospy.logerr(f"❌ RealSense introuvable ou erreur : {e}")
            finally:
                try: self.pipeline.stop()
                except: pass

if __name__ == '__main__':
    node = Sam3Sam2Node()
