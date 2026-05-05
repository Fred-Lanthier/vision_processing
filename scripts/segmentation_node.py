#!/usr/bin/env python3
"""
Segmentation Node
=================
Owns the GPU for: CUTIE tracking (Multi-Object) + SAM3 init.
Tracks BOTH the target (food) and the container (bowl/plate) using 
CUTIE for 2D mask propagation and two PersistentTargetModels for 3D ICP.
"""

# ── ANTI CPU-THRASHING ──
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import rospy
import numpy as np
import time
import torch
import cv2
import sys
from collections import deque

import rospkg
import message_filters
import tf
import tf.transformations as tft
from sensor_msgs.msg import Image, PointCloud2, JointState, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Float32MultiArray
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
from omegaconf import open_dict
from hydra import compose, initialize_config_dir

import cutie
from cutie.model.cutie import CUTIE
from cutie.utils.get_default_model import get_default_model
from cutie.inference.utils.args_utils import get_dataset_cfg
from cutie.utils.download_models import download_models_if_needed
from cutie.inference.inference_core import InferenceCore

rospack = rospkg.RosPack()
sys.path.append(os.path.join(rospack.get_path('vision_processing'), 'scripts'))
from utils import compute_T_child_parent_xacro
from PersistentTargetModel import PersistentTargetModel
from sam3_client import Sam3Client

from concurrent.futures import ThreadPoolExecutor, wait

try:
    from Compute_3D_point_cloud_from_mesh import RobotMeshLoaderOptimized
    MESH_LOADER_OK = True
except ImportError:
    MESH_LOADER_OK = False


def get_small_model() -> CUTIE:
    config_dir = os.path.abspath(os.path.join(cutie.__path__[0], 'config'))
    initialize_config_dir(version_base='1.3.2', config_dir=config_dir, job_name="eval_config")
    cfg = compose(config_name="eval_config", overrides=["model=small"])
    weight_dir = download_models_if_needed()
    with open_dict(cfg):
        cfg['weights'] = os.path.join(weight_dir, 'cutie-small-mega.pth')
    get_dataset_cfg(cfg)
    cutie_net = CUTIE(cfg).cuda().eval()
    model_weights = torch.load(cfg.weights, map_location='cuda')
    cutie_net.load_weights(model_weights)
    return cutie_net


class PerfTracker:
    def __init__(self, window=30):
        self._window = window
        self._durations = {}
        self._tick_times = {}
        self._intervals = {}

    def tick(self, stage):
        now = time.perf_counter()
        if stage in self._tick_times:
            dt = now - self._tick_times[stage]
            if stage not in self._intervals:
                self._intervals[stage] = deque(maxlen=self._window)
            self._intervals[stage].append(dt)
        self._tick_times[stage] = now

    def tock(self, stage):
        now = time.perf_counter()
        if stage in self._tick_times:
            dt = now - self._tick_times[stage]
            if stage not in self._durations:
                self._durations[stage] = deque(maxlen=self._window)
            self._durations[stage].append(dt)

    def avg_ms(self, stage):
        if stage not in self._durations or len(self._durations[stage]) == 0:
            return 0.0
        return np.mean(self._durations[stage]) * 1000

    def hz(self, stage):
        if stage not in self._intervals or len(self._intervals[stage]) == 0:
            return 0.0
        avg = np.mean(self._intervals[stage])
        return 1.0 / avg if avg > 0 else 0.0

    def summary(self, stages=None):
        if stages is None:
            stages = sorted(set(list(self._durations.keys()) + list(self._intervals.keys())))
        parts = []
        for s in stages:
            ms = self.avg_ms(s)
            hz = self.hz(s)
            if hz > 0:
                parts.append(f"{s}: {ms:.1f}ms ({hz:.0f}Hz)")
            elif ms > 0:
                parts.append(f"{s}: {ms:.1f}ms")
        return " | ".join(parts)


class SegmentationNode:

    def __init__(self):
        rospy.init_node('segmentation_node', anonymous=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.perf = PerfTracker(window=50)

        # ── Robot geometry ──
        pkg = rospack.get_path('vision_processing')
        xacro = os.path.join(pkg, 'urdf', 'panda_camera.xacro')
        self.T_tcp_cam = (
            compute_T_child_parent_xacro(xacro, "camera_wrist_link", "panda_TCP") @
            compute_T_child_parent_xacro(xacro, "camera_wrist_optical_frame", "camera_wrist_link")
        )
        self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro, 'fork_tip', 'panda_TCP')

        self.mesh_loader = None
        if MESH_LOADER_OK:
            try:
                self.mesh_loader = RobotMeshLoaderOptimized(xacro)
            except Exception as e:
                rospy.logwarn(f"Mesh loader fail: {e}")

        # ── Cutie SMALL ──
        rospy.loginfo("Loading Cutie VOS model (SMALL)...")
        self.cutie_model = get_default_model()
        self.processor = InferenceCore(self.cutie_model, cfg=self.cutie_model.cfg)
        self.processor.max_internal_size = 240
        self.cube_locked = False
        self.frame_count = 0
        rospy.loginfo("Cutie ready.")

        # ── SAM3 ──
        self.sam3_pub = rospy.Publisher('/vision/sam3_request', Image, queue_size=1)
        self.sam3_sub = rospy.Subscriber('/vision/sam3_reply', Image, self._sam3_cb, queue_size=1)
        self.sam3_result = None
        self.waiting_sam3 = False
        self.sam3_time = None
        self.sam3_direct = Sam3Client()
        
        # ── Masks tracking variables ──
        self.container_mask_init = None  
        self.last_valid_container_mask = None
        self.model_executor = ThreadPoolExecutor(max_workers=2)

        # ── Models (ICP 3D) ──
        # 1. Target (Nourriture)
        self.target_model = PersistentTargetModel(
            voxel_size=0.003, max_points=500, device=self.device,
            min_reference_size=300, stable_frames_needed=1,
            icp_max_translation=0.10
        )
        # 2. Container (Bol/Assiette)
        self.container_model = PersistentTargetModel(
            voxel_size=0.005, max_points=2500, device=self.device,
            min_reference_size=500, stable_frames_needed=1,
            icp_max_translation=0.15, icp_max_iterations=10
        )

        # ── Grasp state ──
        self.is_grasped = False
        self.grasped_local = None
        self.CONTACT_DIST = 0.005

        # ── Fork mesh cache ──
        self.fork_cloud = None
        self.last_joint_hash = None

        # ── Camera intrinsics ──
        self.fx = torch.tensor(604.9, device=self.device)
        self.fy = torch.tensor(604.9, device=self.device)
        self.cx = torch.tensor(320.0, device=self.device)
        self.cy = torch.tensor(240.0, device=self.device)

        # ── Publishers ──
        self.pub_target_bbox = rospy.Publisher('/vision/target_bbox', Float32MultiArray, queue_size=1)
        self.pub_container_bbox = rospy.Publisher('/vision/container_bbox', Float32MultiArray, queue_size=1, latch=True)
        self.pub_merged = rospy.Publisher('/vision/merged_cloud', PointCloud2, queue_size=1)
        self.pub_centroid = rospy.Publisher('/vision/target_centroid', PointStamped, queue_size=1)
        self.pub_perf = rospy.Publisher('/vision/seg_perf', String, queue_size=1)
        self.pub_container_cloud = rospy.Publisher('/vision/container_cloud', PointCloud2, queue_size=1)
        self.pub_target_cloud = rospy.Publisher('/vision/target_cloud', PointCloud2, queue_size=1)

        # ── Subscribers ──
        self.sub_info = rospy.Subscriber(
            "/camera_wrist/color/camera_info", CameraInfo, self._cam_info_cb
        )
        sub_rgb = message_filters.Subscriber("/synced/camera_wrist/rgb", Image)
        sub_dep = message_filters.Subscriber("/synced/camera_wrist/depth", Image)
        sub_jnt = message_filters.Subscriber("/synced/joint_states", JointState)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_dep, sub_jnt], queue_size=5, slop=0.1
        )
        self.ts.registerCallback(self._callback)
        self.sub_joints_mesh = rospy.Subscriber(
            "/joint_states", JointState, self._joint_cb, queue_size=1
        )

        # ── FLAG D'INITIALISATION ──
        self.is_initialized = False
        rospy.loginfo("Segmentation Node READY")


    # ==================================================================
    #  CALLBACKS
    # ==================================================================

    def _cam_info_cb(self, msg):
        self.fx = torch.tensor(msg.K[0], device=self.device)
        self.fy = torch.tensor(msg.K[4], device=self.device)
        self.cx = torch.tensor(msg.K[2], device=self.device)
        self.cy = torch.tensor(msg.K[5], device=self.device)
        self.sub_info.unregister()

    def _sam3_cb(self, msg):
        if not self.is_initialized:
            return 
            
        mask = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        self.sam3_result = (mask > 0) if np.sum(mask) > 50 else None
        self.waiting_sam3 = False

    def _joint_cb(self, msg):
        if not self.mesh_loader:
            return
        try:
            jmap = {n: msg.position[i] for i, n in enumerate(msg.name) if "joint" in n}
            jh = tuple(round(v, 4) for v in jmap.values())
            if jh != self.last_joint_hash:
                self.fork_cloud = self.mesh_loader.create_point_cloud_fork_tip(jmap)
                self.last_joint_hash = jh
        except Exception:
            pass


    # ==================================================================
    #  CUTIE HELPERS (MULTI-OBJECT)
    # ==================================================================

    def _prepare_image(self, cv_img):
        img_t = torch.from_numpy(cv_img).to(self.device, non_blocking=True)
        return img_t.permute(2, 0, 1).float().div_(255.0)

    def _init_cutie(self, cv_img, target_mask, container_mask):
        try:
            # with open_dict(self.cutie_model.cfg):
            #     self.cutie_model.cfg.use_long_term = True     # ← was False
            #     self.cutie_model.cfg.max_mem_frames = 5       # ← was 3
            #     self.cutie_model.cfg.mem_every = 3             # ← was 2
            #     self.cutie_model.cfg.max_long_term_elements = 5000

            self.processor = InferenceCore(self.cutie_model, cfg=self.cutie_model.cfg)
            self.processor.max_internal_size = 480
            image_t = self._prepare_image(cv_img)

            # Création du masque multi-classes : 0=Background, 1=Target, 2=Container
            mask_t = torch.zeros(target_mask.shape, dtype=torch.int64, device=self.device)
            
            if container_mask is not None:
                mask_t[torch.from_numpy(container_mask > 0)] = 2
                
            mask_t[torch.from_numpy(target_mask > 0)] = 1  # Target écrase le container s'ils se chevauchent

            objects = [1]
            if container_mask is not None:
                objects.append(2)

            with torch.inference_mode(), torch.amp.autocast('cuda'):
                self.processor.step(image_t, mask_t, objects=objects)
            return True
        except Exception as e:
            rospy.logerr(f"Cutie Init error: {e}")
            return False

    def _track_cutie(self, cv_img):
        try:
            image_t = self._prepare_image(cv_img)
            with torch.inference_mode(), torch.amp.autocast('cuda'):
                output_prob = self.processor.step(image_t)
                mask_t = self.processor.output_prob_to_mask(output_prob)

            mask_cpu = mask_t.to('cpu', non_blocking=True)
            torch.cuda.current_stream().synchronize()
            mask_np = mask_cpu.numpy().squeeze().astype(np.uint8)
            return mask_np
        except Exception as e:
            rospy.logerr(f"Cutie Track error: {e}")
            return None

    def _run_tracking(self, cv_img, rgb_msg):
        t_mask, c_mask = None, None

        # 1. Start CUTIE immediately for the container, even before the food is found
        if not getattr(self, '_cutie_started', False) and self.container_mask_init is not None:
            empty_target = np.zeros_like(self.container_mask_init)
            if self._init_cutie(cv_img, empty_target, self.container_mask_init):
                self._cutie_started = True
                self.last_valid_container_mask = self.container_mask_init.copy()

        # 2. If CUTIE is started, TRACK EVERYTHING (This kills the ghost!)
        if getattr(self, '_cutie_started', False):
            mask_np = self._track_cutie(cv_img)
            if mask_np is not None:
                t_mask = (mask_np == 1)
                c_mask = (mask_np == 2)
                
                # Keep container memory alive
                if np.sum(c_mask) > 100:
                    self.last_valid_container_mask = c_mask.copy()
                else:
                    c_mask = None
                    
                # If target is lost, signal SAM 3, but CUTIE keeps tracking the container
                if np.sum(t_mask) < 50:
                    self.cube_locked = False
                    t_mask = None

        # 3. Handle SAM 3 requests for the missing Target
        if not self.cube_locked:
            if self.sam3_result is not None:
                # SAM 3 found it! Combine with the LIVE container mask
                c_mask_to_use = getattr(self, 'last_valid_container_mask', None)
                if c_mask_to_use is None:
                    c_mask_to_use = np.zeros_like(self.sam3_result)
                    
                success = self._init_cutie(cv_img, self.sam3_result, c_mask_to_use)
                if success:
                    self.cube_locked = True
                    self._cutie_started = True
                    t_mask = self.sam3_result
                    c_mask = c_mask_to_use
                self.sam3_result = None

            elif not self.waiting_sam3 or (self.sam3_time and (rospy.Time.now() - self.sam3_time).to_sec() > 3.0):
                self.sam3_pub.publish(rgb_msg)
                self.waiting_sam3 = True
                self.sam3_time = rospy.Time.now()

        return t_mask, c_mask


    # ==================================================================
    #  MAIN CALLBACK
    # ==================================================================

    def _callback(self, rgb_msg, depth_msg, joint_msg):
        if not self.is_initialized:
            return

        self.perf.tick('callback')

        # ── 1. Parse + TF ──
        self.perf.tick('tf_lookup')
        cv_img = self._decode_img(rgb_msg)
        if cv_img is None: return
        cv_depth = self._decode_depth(depth_msg)
        T_world_tcp = self._lookup_tf(rgb_msg.header.stamp)
        if T_world_tcp is None: return
        T_world_cam = T_world_tcp @ self.T_tcp_cam
        T_world_fork = T_world_tcp @ self.T_tcp_fork_tip
        fork_tip = T_world_fork[:3, 3]
        self.perf.tock('tf_lookup')

        robot_pts = self.fork_cloud

        # ── 2. Contact / grasp (Pour la cible) ──
        self.perf.tick('grasp_check')
        skip_tracking = self.is_grasped
        if not self.is_grasped:
            tw = self.target_model.get_points()
            if tw is not None and len(tw) > 0:
                target_np = tw.cpu().numpy()
                dists = np.linalg.norm(target_np - fork_tip, axis=1)
                min_dist = dists.min()
                closest = target_np[dists.argmin()]
                if min_dist < self.CONTACT_DIST:
                    skip_tracking = True
                if fork_tip[2] < closest[2] and min_dist < 0.02:
                    self.is_grasped = True
                    skip_tracking = True
                    T_inv = np.linalg.inv(T_world_fork)
                    ones = np.ones((len(target_np), 1))
                    self.grasped_local = (T_inv @ np.hstack([target_np, ones]).T).T[:, :3].astype(np.float32)
        self.perf.tock('grasp_check')

        # ── 3. Cutie tracking (Multi-Object) ──
        self.perf.tick('cutie')
        target_mask, container_mask = None, None
        if not skip_tracking:
            target_mask, container_mask = self._run_tracking(cv_img, rgb_msg)
        self.frame_count += 1
        self.perf.tock('cutie')

        # ── 4. Target & Container 3D Models Update (ICP - Multithreaded) ──
        self.perf.tick('models_update')
        if cv_depth is not None:
            z = cv_depth if cv_depth.dtype == np.float32 else cv_depth / 1000.0
            
            f_fx = float(self.fx)
            f_fy = float(self.fy)
            f_cx = float(self.cx)
            f_cy = float(self.cy)

            # Hand the tasks to the worker threads
            future_target = self.model_executor.submit(
                self.target_model.update,
                target_mask, z, T_world_cam, f_fx, f_fy, f_cx, f_cy
            )
            future_container = self.model_executor.submit(
                self.container_model.update,
                container_mask, z, T_world_cam, f_fx, f_fy, f_cx, f_cy
            )
            
            # Wait for both threads to finish before moving to Step 5
            wait([future_target, future_container])
            try:
                future_target.result()
                future_container.result()
            except Exception as e:
                rospy.logerr(f"ICP Thread Error: {e}")
        self.perf.tock('models_update')

        # Publish Container Cloud
        cw = self.container_model.get_points()
        if cw is not None and len(cw) > 0:
            self._pub_cloud(cw.cpu().numpy(), self.pub_container_cloud)
            
        # Publish Target Cloud for RViz (Add this block)
        tw = self.target_model.get_points()
        if tw is not None and len(tw) > 0:
            self._pub_cloud(tw.cpu().numpy(), self.pub_target_cloud)

        # ── 5. Publish Bounding Boxes ──
        self.perf.tick('pub_bboxes')
        
        # BBox Nourriture
        tw = self.target_model.get_points()
        if tw is not None and len(tw) > 0:
            pts_np = tw.cpu().numpy()
            margin = 0.005
            bmin = pts_np.min(axis=0) - margin
            bmax = pts_np.max(axis=0) + margin
            msg = Float32MultiArray(data=list(bmin) + list(bmax))
            self.pub_target_bbox.publish(msg)

        # BBox Contenant
        cw = self.container_model.get_points()
        if cw is not None and len(cw) > 0:
            pts_np = cw.cpu().numpy()
            margin = 0.03
            bmin = pts_np.min(axis=0) - margin
            bmax = pts_np.max(axis=0) + margin
            msg = Float32MultiArray(data=list(bmin) + list(bmax))
            self.pub_container_bbox.publish(msg)
            
        self.perf.tock('pub_bboxes')

        # ── 6. Publish centroid ──
        if self.is_grasped:
            c = (T_world_fork @ np.array([0, 0, 0, 1]))[:3]
            self._pub_centroid(c.astype(np.float32))
        else:
            c = self.target_model.get_centroid_np()
            if c is not None:
                self._pub_centroid(c)

        # ── 7. Merge + publish condition cloud ──
        self.perf.tick('merge')
        self._publish_merged(robot_pts, T_world_fork)
        self.perf.tock('merge')

        self.perf.tock('callback')

        # ── 8. Perf summary ──
        if self.frame_count % 30 == 0:
            stages = ['callback', 'tf_lookup', 'cutie', 'models_update', 'pub_bboxes', 'merge']
            summary = "SEG: " + self.perf.summary(stages)
            rospy.loginfo(f"\n{summary}")
            self.pub_perf.publish(String(data=summary))


    # ==================================================================
    #  MERGE + PUBLISH
    # ==================================================================

    def _publish_merged(self, robot_pts, T_world_fork):
        N, half = 1024, 512
        parts = []

        if robot_pts is not None and len(robot_pts) > 0:
            rp = robot_pts if isinstance(robot_pts, np.ndarray) else robot_pts.cpu().numpy()
            idx = np.random.choice(len(rp), min(half, len(rp)), replace=False)
            parts.append(rp[idx])

        if self.is_grasped and self.grasped_local is not None:
            ones = np.ones((len(self.grasped_local), 1))
            tw = (T_world_fork @ np.hstack([self.grasped_local, ones]).T).T[:, :3].astype(np.float32)
        else:
            pts = self.target_model.get_points()
            tw = pts.cpu().numpy() if pts is not None else None

        if tw is not None and len(tw) > 0:
            idx = np.random.choice(len(tw), min(half, len(tw)), replace=False)
            parts.append(tw[idx])

        if not parts: return
        cloud = np.vstack(parts).astype(np.float32)
        if len(cloud) > N:
            cloud = cloud[np.random.choice(len(cloud), N, replace=False)]
        elif len(cloud) < N and len(cloud) > 0:
            extra = np.random.choice(len(cloud), N - len(cloud), replace=True)
            cloud = np.vstack([cloud, cloud[extra]])
        self._pub_cloud(cloud, self.pub_merged)


    # ==================================================================
    #  UTILITIES
    # ==================================================================

    def _decode_img(self, msg):
        if msg.encoding == "rgb8":
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
        return None

    def _decode_depth(self, msg):
        if "32FC1" in msg.encoding:
            return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
        return None

    def _lookup_tf(self, stamp):
        try:
            t, r = self.tf_listener.lookupTransform("world", "panda_hand_tcp", rospy.Time(0))
        except tf.Exception:
            return None
        T = tft.quaternion_matrix(r)
        T[:3, 3] = t
        return T

    def _pub_cloud(self, points, publisher, frame="world"):
        if points is None or len(points) == 0: return
        hdr = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id=frame)
        publisher.publish(pc2.create_cloud_xyz32(hdr, points.astype(np.float32)))

    def _pub_centroid(self, c):
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.point.x, msg.point.y, msg.point.z = float(c[0]), float(c[1]), float(c[2])
        self.pub_centroid.publish(msg)


    # ==================================================================
    #  CONTAINER DETECTION (Initialisation)
    # ==================================================================

    def _detect_container(self):
        rospy.loginfo("[Container] Waiting for camera frame...")
        try:
            rgb_msg = rospy.wait_for_message("/synced/camera_wrist/rgb", Image, timeout=10.0)
            depth_msg = rospy.wait_for_message("/synced/camera_wrist/depth", Image, timeout=10.0)
        except rospy.ROSException:
            rospy.logwarn("[Container] No camera frame, skipping.")
            return

        cv_depth = self._decode_depth(depth_msg)
        if cv_depth is None: return

        try:
            t, r = self.tf_listener.lookupTransform("world", "panda_hand_tcp", rospy.Time(0))
        except tf.Exception:
            rospy.logwarn("[Container] No TF, skipping.")
            return
        T = tft.quaternion_matrix(r)
        T[:3, 3] = t
        T_world_cam = T @ self.T_tcp_cam

        prompt_pub = rospy.Publisher('/vision/sam3_prompt', String, queue_size=1)
        rospy.sleep(0.3)

        for prompt in ["bowl", "plate", "container", "gray circle"]:
            rospy.loginfo(f"[Container] Trying SAM3: '{prompt}'...")
            prompt_pub.publish(String(data=prompt))
            rospy.sleep(0.1)

            self.sam3_pub.publish(rgb_msg)
            try:
                reply = rospy.wait_for_message("/vision/sam3_reply", Image, timeout=5.0)
            except rospy.ROSException:
                continue

            mask = np.frombuffer(reply.data, dtype=np.uint8).reshape(reply.height, reply.width)
            if np.sum(mask > 0) < 50:
                continue

            # Sauvegarde du masque initial pour CUTIE
            self.container_mask_init = (mask > 0)

            # Publication de la BBox initiale
            z = cv_depth if cv_depth.dtype == np.float32 else cv_depth / 1000.0
            valid = (mask > 0) & (z > 0.01) & (z < 2.0) & np.isfinite(z)
            if np.sum(valid) >= 50:
                v, u = np.where(valid)
                zv = z[valid]
                fx, fy = self.fx.item(), self.fy.item()
                cx, cy = self.cx.item(), self.cy.item()
                pts_cam = np.stack([(u-cx)*zv/fx, (v-cy)*zv/fy, zv], axis=-1).astype(np.float32)
                ones = np.ones((len(pts_cam), 1), dtype=np.float32)
                pts_world = (T_world_cam @ np.hstack([pts_cam, ones]).T).T[:, :3]

                margin = 0.03
                bbox_min = (pts_world.min(axis=0) - margin).astype(np.float32)
                bbox_max = (pts_world.max(axis=0) + margin).astype(np.float32)

                msg = Float32MultiArray(data=list(bbox_min) + list(bbox_max))
                self.pub_container_bbox.publish(msg)
                rospy.loginfo(f"[Container] Found! Init mask saved. Bbox={bbox_min} -> {bbox_max}")

            prompt_pub.publish(String(data="green cube"))
            rospy.sleep(0.2)
            return

        rospy.logwarn("[Container] Not detected. Defaulting to target only.")
        prompt_pub.publish(String(data="green cube"))
        rospy.sleep(0.2)


    # ==================================================================
    #  RUN
    # ==================================================================

    def run(self):
        self.tf_listener = tf.TransformListener()
        rospy.sleep(1.0)
        self._detect_container()
        
        del self.sam3_direct
        torch.cuda.empty_cache()
        rospy.loginfo("[Memory] SAM3 client freed from GPU")
        
        self.is_initialized = True
        rospy.loginfo("Segmentation node spinning.")
        rospy.spin()


if __name__ == '__main__':
    SegmentationNode().run()