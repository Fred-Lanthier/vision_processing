#!/usr/bin/env python3
import rospy
import numpy as np
import message_filters
import threading
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

# --- IMPORTS SAM 2 (State of the Art Video Tracking) ---
from sam2.build_sam import build_sam2_video_predictor

rospack = rospkg.RosPack()
sys.path.append(os.path.join(rospack.get_path('vision_processing'), 'scripts'))
try:
    from Compute_3D_point_cloud_from_mesh import RobotMeshLoaderOptimized
    from SDF import VoxelSDF_GPU
    from PersistentCloud import ObjectCloud, WorldCloud
    LOADER_AVAILABLE = True
except ImportError:
    LOADER_AVAILABLE = False


class MergedCloudNode:
    """
    Architecture — two parallel paths, sharing only lightweight data:

    CAMERA CALLBACK (runs at camera rate ~10-30 Hz):
      1. TF lookup
      2. SAM2 tracking → mask
      3. Unproject mask=True → target cloud (small, fast)
      4. Stash raw inputs for obstacle thread (just pointer copies)
      5. Cloud completion + publish merged cloud + publish centroid

    OBSTACLE THREAD (runs in background, processes whenever new data arrives):
      1. Pick up (mask, depth, T_world_cam, fork_cache) from stash
      2. Unproject ~mask → obstacle cloud (200k+ pts, slow on CPU)
      3. GPU robot body removal via cdist
      4. Downsample
      5. SDF update via cupy EDT
      6. Publish obstacle cloud

    The two paths NEVER block each other. The callback writes raw inputs
    atomically via a lock; the thread reads them and does all heavy work.
    """

    def __init__(self, temp_work_dir="/tmp/robot_tracking_live"):
        rospy.init_node('merged_cloud_node', anonymous=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_object = "cube"
        self.detection_confidence = 0.10
        self.cube_locked = False
        
        # --- SAM 3 ASYNC (via sam3_detector_node) ---
        self.sam3_pub = rospy.Publisher('/vision/sam3_request', Image, queue_size=1)
        self.sam3_sub = rospy.Subscriber('/vision/sam3_reply', Image, self.sam3_reply_cb, queue_size=1)
        self.waiting_for_sam3 = False
        self.sam3_request_time = None
        self.SAM3_TIMEOUT = 3.0
        self.sam3_result_mask = None
        
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

        self.work_dir = temp_work_dir
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
        os.makedirs(self.work_dir)

        rospy.loginfo("✅ SAM 2 Video Predictor Prêt.")

        # --- VARIABLES ROBOT ---
        self.current_cube_cloud = None
        self.fork_cloud_cache = None              
        self.last_joint_hash = None   

        # --- PERSISTENT CLOUDS (SLAM-like accumulators) ---
        # Target: object-centric (accumulates in local frame, moves with object)
        self.target_map = ObjectCloud(voxel_size=0.003, max_age=3000)
        # Environment: world-frame (static objects — table, bowl, walls)
        self.environment_map = WorldCloud(voxel_size=0.01, max_age=2000)
        self._env_cleaned_once = False  # one-time cleanup after SAM3 locks

        # --- CONTACT / GRASP DETECTION ---
        self.CONTACT_DISTANCE = 0.005
        self.is_grasped = False
        self.grasped_target_fork_local = None  # (M, 3) target in fork-tip frame

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

        # ==========================================================
        # SDF SYSTEM
        # ==========================================================
        self.sdf = VoxelSDF_GPU(
            voxel_size=0.01,
            bounds=((-0.2, 0.8), (-0.5, 0.5), (0.0, 0.8)),
            device=self.device
        )
        self.current_obstacle_cloud = None

        # --- Shared stash: callback WRITES, obstacle thread READS ---
        self._stash_lock = threading.Lock()
        self._stash_mask = None
        self._stash_depth = None
        self._stash_T = None
        self._stash_fork = None
        self._stash_dirty = False
        self._obstacle_event = threading.Event()

        # --- PUBLISHERS & SUBSCRIBERS ---
        self.pub_merged = rospy.Publisher('/vision/merged_cloud', PointCloud2, queue_size=1)
        self.pub_marker = rospy.Publisher('/vision/marker', Marker, queue_size=1)
        self.pub_fork_tip = rospy.Publisher('/vision/debug_fork_tip', PointStamped, queue_size=1)
        self.pub_target_centroid = rospy.Publisher('/vision/target_centroid', PointStamped, queue_size=1)
        self.pub_obstacle_cloud = rospy.Publisher('/vision/obstacle_cloud', PointCloud2, queue_size=1)
        self.pub_sdf_viz = rospy.Publisher('/vision/sdf_field', PointCloud2, queue_size=1)
        self.pub_sdf_slice = rospy.Publisher('/vision/sdf_slice', PointCloud2, queue_size=1)

        # SDF visualization params (changeable at runtime via rosparam set)
        #   rosparam set /sdf_viz_mode "slice"    → plane slice
        #   rosparam set /sdf_viz_mode "volume"   → 3D volume
        #   rosparam set /sdf_viz_mode "both"     → both topics
        #   rosparam set /sdf_viz_mode "off"      → disable (save GPU)
        #   rosparam set /sdf_slice_axis "z"       → x, y, or z
        #   rosparam set /sdf_slice_value 0.05     → height in meters
        #   rosparam set /sdf_viz_max_dist 0.15    → color range
        rospy.set_param('/sdf_viz_mode', 'both')
        rospy.set_param('/sdf_slice_axis', 'z')
        rospy.set_param('/sdf_slice_value', 0.05)
        rospy.set_param('/sdf_viz_max_dist', 0.15)

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

        self.sub_joints_for_mesh = rospy.Subscriber(
            "/joint_states", JointState, self.joint_mesh_cb, queue_size=1
        )

        # --- OBSTACLE THREAD ---
        self._obstacle_thread_running = True
        self._obstacle_thread = threading.Thread(
            target=self._obstacle_thread_fn, daemon=True
        )
        self._obstacle_thread.start()

        rospy.loginfo("🚀 MERGED CLOUD + SDF (TRULY PARALLEL) PRÊT")

    # ==================================================================
    # CAMERA INFO
    # ==================================================================
    def cam_info_cb(self, msg):
        self.fx = msg.K[0]; self.cx = msg.K[2]; self.fy = msg.K[4]; self.cy = msg.K[5]
        self.sub_info.unregister()

    # ==================================================================
    # SAM 3 REPLY
    # ==================================================================
    def sam3_reply_cb(self, msg):
        mask_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        if np.sum(mask_np) > 50:
            self.sam3_result_mask = (mask_np > 0)
            rospy.loginfo(f"🎯 SAM 3 reply received — valid mask ({np.sum(mask_np > 0)} px)")
        else:
            self.sam3_result_mask = None
            rospy.logwarn("❌ SAM 3 reply: empty mask, will retry next cycle")
        self.waiting_for_sam3 = False

    # ==================================================================
    # MESH FK (independent subscriber)
    # ==================================================================
    def joint_mesh_cb(self, joint_msg):
        if not self.mesh_loader:
            return
        try:
            joint_map = {name: joint_msg.position[i] 
                         for i, name in enumerate(joint_msg.name) if "panda" in name}
            joint_hash = tuple(round(v, 4) for v in joint_map.values())
            if joint_hash != self.last_joint_hash:
                self.fork_cloud_cache = self.mesh_loader.create_point_cloud_fork_tip(joint_map)
                self.last_joint_hash = joint_hash
        except Exception:
            pass

    # ==================================================================
    # OBSTACLE THREAD — entire obstacle pipeline in background
    # ==================================================================
    def _obstacle_thread_fn(self):
        """
        Background thread. Wakes when callback signals new data, then runs:
          1. Unproject ~mask → 3D obstacle cloud
          2. Remove robot body via GPU cdist
          3. Downsample
          4. SDF update (cupy EDT)
          5. Publish obstacle cloud

        ALL GPU work runs on a dedicated CUDA stream so it never
        blocks SAM2 on the default stream.
        """
        # Dedicated CUDA stream — this is the key fix.
        # Without it, every torch/cupy op here serializes with SAM2.
        stream = torch.cuda.Stream(device=self.device)

        # Pre-upload camera intrinsics to GPU (constant after init)
        fx_t = torch.tensor(self.fx, device=self.device, dtype=torch.float32)
        fy_t = torch.tensor(self.fy, device=self.device, dtype=torch.float32)
        cx_t = torch.tensor(self.cx, device=self.device, dtype=torch.float32)
        cy_t = torch.tensor(self.cy, device=self.device, dtype=torch.float32)

        while self._obstacle_thread_running and not rospy.is_shutdown():
            self._obstacle_event.wait(timeout=0.1)
            self._obstacle_event.clear()

            with self._stash_lock:
                if not self._stash_dirty:
                    continue
                mask = self._stash_mask
                depth = self._stash_depth
                T_world_cam = self._stash_T
                fork_cache = self._stash_fork
                self._stash_dirty = False

            if mask is None or depth is None:
                continue

            obstacle_np = None  # ensure clean scope each iteration
            ms_sdf = 0

            try:
                t0 = rospy.Time.now().to_sec()

                with torch.cuda.stream(stream):
                    # ── 1. Upload mask + depth to GPU ──
                    z_gpu = torch.from_numpy(depth).to(self.device, non_blocking=True)
                    mask_gpu = torch.from_numpy(mask).to(self.device, non_blocking=True)

                    valid = (~mask_gpu) & (z_gpu > 0.01) & (z_gpu < 1.5)

                    v_idx, u_idx = torch.where(valid)
                    z_val = z_gpu[v_idx, u_idx]

                    if len(z_val) < 100:
                        continue

                    # ── 2. GPU unprojection ──
                    x_cam = (u_idx.float() - cx_t) * z_val / fx_t
                    y_cam = (v_idx.float() - cy_t) * z_val / fy_t
                    pts_cam = torch.stack([x_cam, y_cam, z_val, torch.ones_like(z_val)], dim=1)

                    T_gpu = torch.from_numpy(T_world_cam.astype(np.float32)).to(
                        self.device, non_blocking=True
                    )
                    obstacle_gpu = (T_gpu @ pts_cam.T).T[:, :3]

                    # ── 3. GPU robot body removal ──
                    if fork_cache is not None and len(fork_cache) > 0:
                        # fork_cache comes from mesh loader — could be numpy or torch
                        if isinstance(fork_cache, np.ndarray):
                            rob_t = torch.from_numpy(fork_cache.astype(np.float32)).to(
                                self.device, non_blocking=True
                            )
                        else:
                            rob_t = fork_cache.to(self.device).float()

                        CHUNK = 50000
                        keep_mask = torch.ones(len(obstacle_gpu), dtype=torch.bool,
                                               device=self.device)
                        for i in range(0, len(obstacle_gpu), CHUNK):
                            chunk = obstacle_gpu[i:i + CHUNK]
                            dists = torch.cdist(chunk.unsqueeze(0), rob_t.unsqueeze(0)).squeeze(0)
                            min_dists = dists.min(dim=1).values
                            keep_mask[i:i + CHUNK] = min_dists > 0.02

                        obstacle_gpu = obstacle_gpu[keep_mask]

                    if len(obstacle_gpu) < 50:
                        continue

                    # ── 4. Downsample on GPU ──
                    if len(obstacle_gpu) > 20000:
                        perm = torch.randperm(len(obstacle_gpu), device=self.device)[:20000]
                        obstacle_gpu = obstacle_gpu[perm]

                # Sync before moving to CPU
                stream.synchronize()

                # ── 6. Move current frame to CPU, integrate into persistent map ──
                obstacle_np = obstacle_gpu.cpu().numpy().astype(np.float32)

                # The ~mask already excludes the target (SAM2 handles this).
                # Only problem: frames BEFORE SAM3 locked had no mask.
                # Solution: one-time cleanup when SAM3 first provides a centroid.
                if not self._env_cleaned_once:
                    target_c = self.target_map.get_centroid()
                    if target_c is not None:
                        # SAM3 just locked — purge pre-lock contamination
                        self.environment_map.exclude_near(target_c, radius=0.05)
                        self._env_cleaned_once = True
                        rospy.loginfo("Environment map: one-time target cleanup done")

                if len(obstacle_np) > 50:
                    self.environment_map.integrate(obstacle_np)

                # ── 7. Get the ACCUMULATED environment for SDF + publishing ──
                # This is the SLAM part: we use ALL remembered points,
                # not just the current field of view.
                accumulated_env = self.environment_map.get_points(min_confidence=1)

                if accumulated_env is not None and len(accumulated_env) > 100:
                    # Downsample accumulated cloud for SDF
                    if len(accumulated_env) > 20000:
                        idx = np.random.choice(len(accumulated_env), 20000, replace=False)
                        sdf_cloud = accumulated_env[idx]
                    else:
                        sdf_cloud = accumulated_env

                    # Re-run SDF on the full accumulated cloud
                    ms_sdf = self.sdf.update(sdf_cloud)
                    self.current_obstacle_cloud = sdf_cloud
                else:
                    self.current_obstacle_cloud = obstacle_np

                dt = rospy.Time.now().to_sec() - t0
                rospy.loginfo_throttle(2,
                    f"Obstacle thread: {dt*1000:.0f}ms total "
                    f"(SDF {ms_sdf:.0f}ms, env_map={self.environment_map.count()} voxels)")

                # Publish the ACCUMULATED cloud (not just current FOV)
                self._publish_cloud_on_topic(
                    self.current_obstacle_cloud, self.pub_obstacle_cloud
                )

                # 7. Publish SDF visualization (mode from rosparam)
                viz_mode = rospy.get_param('/sdf_viz_mode', 'off')
                if viz_mode != 'off':
                    max_dist = rospy.get_param('/sdf_viz_max_dist', 0.15)

                    # Volume mode
                    if viz_mode in ('volume', 'both'):
                        if self.pub_sdf_viz.get_num_connections() > 0:
                            pts, rgb, vals = self.sdf.get_visualization_points(
                                max_dist=max_dist, stride=3
                            )
                            if pts is not None:
                                msg = VoxelSDF_GPU.make_rviz_cloud_msg(pts, rgb)
                                if msg is not None:
                                    self.pub_sdf_viz.publish(msg)

                    # Slice mode
                    if viz_mode in ('slice', 'both'):
                        if self.pub_sdf_slice.get_num_connections() > 0:
                            axis = rospy.get_param('/sdf_slice_axis', 'z')
                            val = rospy.get_param('/sdf_slice_value', 0.05)
                            pts, rgb, vals = self.sdf.get_slice_points(
                                axis=axis, value=val, max_dist=max_dist
                            )
                            if pts is not None:
                                msg = VoxelSDF_GPU.make_rviz_cloud_msg(pts, rgb)
                                if msg is not None:
                                    self.pub_sdf_slice.publish(msg)

            except Exception as e:
                rospy.logerr(f"Obstacle thread error: {e}")

    # ==================================================================
    # UTILITIES
    # ==================================================================
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

    def _publish_cloud_on_topic(self, points, publisher, frame_id="world"):
        if points is None or len(points) == 0: return
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        cloud_msg = pc2.create_cloud_xyz32(header, points)
        publisher.publish(cloud_msg)

    def _publish_target_centroid(self, centroid):
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.point.x = float(centroid[0])
        msg.point.y = float(centroid[1])
        msg.point.z = float(centroid[2])
        self.pub_target_centroid.publish(msg)

    # ==================================================================
    # SAM 2 SESSION INIT
    # ==================================================================
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

    # ==================================================================
    # MAIN CALLBACK — fast path: SAM2 + target cloud + stash + merge
    # ==================================================================
    def callback(self, rgb_msg, depth_msg, joint_msg):
        # ==========================================================
        # 1. RÉCUPÉRATION IMAGES ET TF
        # ==========================================================
        cv_img = self.imgmsg_to_numpy(rgb_msg)
        if cv_img is None: return
        
        cv_depth = self.imgmsg_to_numpy(depth_msg)
        
        # Use the IMAGE timestamp for TF lookup to prevent drift.
        # Short timeout with fallback to Time(0) if exact stamp unavailable.
        img_stamp = rgb_msg.header.stamp
        try:
            self.tf_listener.waitForTransform("world", "panda_hand_tcp", img_stamp, rospy.Duration(0.03))
            (trans_world_tcp, rot_world_tcp) = self.tf_listener.lookupTransform("world", "panda_hand_tcp", img_stamp)
        except tf.Exception:
            # Fallback: use latest TF (tiny drift but doesn't block)
            try:
                (trans_world_tcp, rot_world_tcp) = self.tf_listener.lookupTransform("world", "panda_hand_tcp", rospy.Time(0))
            except tf.Exception:
                rospy.logwarn_throttle(1, "TF non disponible")
                return
        T_world_tcp = tft.quaternion_matrix(rot_world_tcp)
        T_world_tcp[:3, 3] = trans_world_tcp
        T_world_cam = T_world_tcp @ self.T_tcp_cam

        # ==========================================================
        # 2. ROBOT CLOUD
        # ==========================================================
        current_robot_points = self.fork_cloud_cache

        # ==========================================================
        # 2.5 CONTACT / GRASP DETECTION
        # ==========================================================
        fork_tip_world = (T_world_tcp @ self.T_tcp_fork_tip)[:3, 3]
        T_world_fork = T_world_tcp @ self.T_tcp_fork_tip
        skip_tracking = False

        if self.is_grasped:
            # Already grasped: skip vision, target tracks via FK (section 5)
            skip_tracking = True

        elif not self.is_grasped:
            accumulated_target = self.target_map.get_points_world(min_confidence=2)
            if accumulated_target is not None and len(accumulated_target) > 0:
                # Distance from fork tip to closest target point
                dists = np.linalg.norm(accumulated_target - fork_tip_world, axis=1)
                min_idx = np.argmin(dists)
                min_dist = dists[min_idx]
                closest_pt = accumulated_target[min_idx]

                rospy.loginfo_throttle(0.5, f"Fork→food min dist: {min_dist*1000:.1f}mm")

                if min_dist < self.CONTACT_DISTANCE:
                    skip_tracking = True

                # ── GRASP DETECTION: fork tip below closest target point ──
                # If fork_tip z < closest_point z, the fork has gone
                # THROUGH the food → it's spiked.
                if fork_tip_world[2] < closest_pt[2] and min_dist < 0.02:
                    rospy.loginfo("🍴 FOOD SPIKED — freezing target in fork frame")
                    self.is_grasped = True
                    skip_tracking = True

                    # Freeze the accumulated target cloud in fork-tip local frame
                    # T_fork_inv @ world_pts → fork-local coordinates
                    T_fork_inv = np.linalg.inv(T_world_fork)
                    target_world = self.target_map.get_points_world(min_confidence=1)
                    if target_world is not None:
                        ones = np.ones((len(target_world), 1))
                        pts_h = np.hstack([target_world, ones])  # (M, 4)
                        self.grasped_target_fork_local = (T_fork_inv @ pts_h.T).T[:, :3].astype(np.float32)

        # ==========================================================
        # 3. VIDEO TRACKING (SAM2)
        # ==========================================================
        current_mask = None
        is_resetting = False

        if skip_tracking:
            self.frame_count += 1

        else:
            if self.cube_locked and self.frame_count > 0 and (self.frame_count % self.MAX_MEMORY_FRAMES == 0):
                rospy.loginfo(f"♻️ SAM 2 Memory Reset (Frame {self.frame_count})...")
                is_resetting = True
            
            if not self.cube_locked or is_resetting or self.inference_state is None:
                
                if is_resetting and self.last_valid_mask is not None:
                    success = self._initialize_tracking_session(cv_img, self.last_valid_mask)
                    if success:
                        current_mask = self.last_valid_mask

                elif self.sam3_result_mask is not None:
                    mask_np = self.sam3_result_mask
                    self.sam3_result_mask = None
                    success = self._initialize_tracking_session(cv_img, mask_np)
                    if success:
                        self.cube_locked = True
                        current_mask = mask_np
                        self.last_valid_mask = mask_np
                        rospy.loginfo("🎯 Objet verrouillé par SAM 3 ! Démarrage du Video Tracking.")

                elif not self.waiting_for_sam3 or \
                     (self.sam3_request_time is not None and 
                      (rospy.Time.now() - self.sam3_request_time).to_sec() > self.SAM3_TIMEOUT):
                    if self.waiting_for_sam3:
                        rospy.logwarn("⏰ SAM 3 timeout — retrying...")
                    rospy.loginfo_throttle(2, "🔍 SAM 3 recherche en cours (non-bloquant)...")
                    self.sam3_pub.publish(rgb_msg)
                    self.waiting_for_sam3 = True
                    self.sam3_request_time = rospy.Time.now()

            else:
                target_h, target_w = self.video_height, self.video_width
                img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                
                if img_rgb.shape[:2] != (target_h, target_w):
                    img_rgb = cv2.resize(img_rgb, (target_w, target_h))
                
                img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
                img_tensor = img_tensor.to(self.device)
                img_tensor = (img_tensor - self.mean) / self.std
                
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
                            self.inference_state,
                            start_frame_idx=self.session_frame_idx,
                            max_frame_num_to_track=1
                        ):
                            raw_mask = (video_res_masks[0] > 0).cpu().numpy()
                            current_mask = raw_mask.squeeze()
                            if current_mask.ndim > 2: current_mask = current_mask[0]
                            break 
                            
                    self.last_valid_mask = current_mask
                    
                    if np.sum(current_mask) < 50:
                        rospy.logwarn("⚠️ Masque vide ! Objet perdu. Relance SAM 3...")
                        self.cube_locked = False
                        
                except Exception as e:
                    rospy.logerr(f"⚠️ Erreur de tracking SAM 2 : {e}")
                    self.cube_locked = False

            self.frame_count += 1

        # ==========================================================
        # 4. TARGET CLOUD (small, stays in callback)
        #    + STASH RAW INPUTS for obstacle thread
        # ==========================================================
        if current_mask is not None and cv_depth is not None:
            z = cv_depth if cv_depth.dtype == np.float32 else cv_depth / 1000.0

            # --- Target cloud (mask=True, small) ---
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

                    # Accumulate in object-local frame (moves with target)
                    self.target_map.integrate(points_world)

            # --- Stash for obstacle thread (just pointer copies, ~0ms) ---
            with self._stash_lock:
                self._stash_mask = current_mask
                self._stash_depth = z
                self._stash_T = T_world_cam.copy()
                self._stash_fork = self.fork_cloud_cache
                self._stash_dirty = True
            self._obstacle_event.set()

        # ==========================================================
        # 4.5 PUBLISH CENTROID
        # ==========================================================
        if self.is_grasped:
            # Grasped: centroid tracks with fork via FK
            grasped_centroid = (T_world_fork @ np.array([0, 0, 0, 1]))[:3]
            self._publish_target_centroid(grasped_centroid.astype(np.float32))
        else:
            target_centroid = self.target_map.get_centroid()
            if target_centroid is not None:
                self._publish_target_centroid(target_centroid)
                rospy.loginfo_throttle(2, f"Target map: {self.target_map.count()} voxels")

        # ==========================================================
        # 5. CLOUD COMPLETION + MERGING
        # ==========================================================
        NUM_TARGET_POINTS = 1024
        pts_per_object = NUM_TARGET_POINTS // 2
        merged = []

        # ── Robot points ──
        if current_robot_points is not None:
            if current_robot_points.shape[0] > pts_per_object:
                idx = np.random.choice(current_robot_points.shape[0], pts_per_object, replace=False)
                merged.append(current_robot_points[idx])
            else:
                merged.append(current_robot_points)

        # ── Target points ──
        if self.is_grasped and self.grasped_target_fork_local is not None:
            # GRASPED: transform frozen target from fork frame to world via FK
            ones = np.ones((len(self.grasped_target_fork_local), 1))
            pts_h = np.hstack([self.grasped_target_fork_local, ones])
            target_world = (T_world_fork @ pts_h.T).T[:, :3].astype(np.float32)

            if len(target_world) > pts_per_object:
                idx = np.random.choice(len(target_world), pts_per_object, replace=False)
                merged.append(target_world[idx])
            else:
                merged.append(target_world)
        else:
            # NOT GRASPED: use accumulated object-centric cloud
            accumulated_target = self.target_map.get_points_world(min_confidence=1)
            if accumulated_target is not None and len(accumulated_target) > 0:
                if len(accumulated_target) > pts_per_object:
                    idx = np.random.choice(len(accumulated_target), pts_per_object, replace=False)
                    merged.append(accumulated_target[idx])
                else:
                    merged.append(accumulated_target)

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
        self._obstacle_thread_running = False
        self._obstacle_event.set()

if __name__ == '__main__':
    MergedCloudNode().run()