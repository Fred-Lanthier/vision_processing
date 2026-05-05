#!/usr/bin/env python3
"""
OmniGuide Perception Node (v3)
===============================

Outputs:
  /vision/merged_cloud      — fork + target (1024 pts) for the policy
  /vision/target_centroid   — 3D attractor
  /vision/obstacle_cloud    — accumulated environment for RViz
  /vision/sdf_viz           — SDF volume visualization
  /vision/sdf_slice         — SDF slice visualization
  /vision/perf              — per-stage Hz and ms

Threading model:
  CALLBACK THREAD: TF + Cutie + target model + merge/publish
  OBSTACLE THREAD: environment unproject + accumulate + SDF + RViz
"""

import rospy
import numpy as np
import threading
import time
import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import cv2
import os
import sys

import rospkg
import message_filters
import tf
import tf.transformations as tft
from sensor_msgs.msg import Image, PointCloud2, JointState, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
from omegaconf import open_dict
from omegaconf import OmegaConf
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
from SDF import VoxelSDF_GPU
from ContainerTracker import ContainerTracker
from PersistentTargetModel import PersistentTargetModel
from sam3_client import Sam3Client

try:
    from Compute_3D_point_cloud_from_mesh import RobotMeshLoaderOptimized
    MESH_LOADER_OK = True
except ImportError:
    MESH_LOADER_OK = False

def get_small_model() -> CUTIE:
    """
    A helper function to get the SMALL model for quick testing
    """
    # 1. Obtenir le chemin absolu du dossier 'config' de Cutie
    config_dir = os.path.abspath(os.path.join(cutie.__path__[0], 'config'))
    
    # 2. Initialiser Hydra avec ce dossier de configuration
    initialize_config_dir(version_base='1.3.2', config_dir=config_dir, job_name="eval_config")
    
    # 3. Charger la config de base MAIS écraser l'architecture pour charger 'small.yaml'
    cfg = compose(config_name="eval_config", overrides=["model=small"])

    # 4. Obtenir le dossier weights et cibler ton nouveau fichier
    weight_dir = download_models_if_needed()
    with open_dict(cfg):
        cfg['weights'] = os.path.join(weight_dir, 'cutie-small-mega.pth')
        
    get_dataset_cfg(cfg)

    # 5. Créer l'architecture et charger les poids (load_weights gère les clés internes automatiquement)
    cutie_net = CUTIE(cfg).cuda().eval()
    model_weights = torch.load(cfg.weights, map_location='cuda')
    cutie_net.load_weights(model_weights)

    return cutie_net

# =====================================================================
#  PERFORMANCE TRACKER
# =====================================================================

class PerfTracker:
    def __init__(self, window=30):
        self._window = window
        self._durations = {}
        self._tick_times = {}
        self._intervals = {}
        from collections import deque
        self._deque = deque

    def tick(self, stage):
        now = time.perf_counter()
        if stage in self._tick_times:
            dt = now - self._tick_times[stage]
            if stage not in self._intervals:
                self._intervals[stage] = self._deque(maxlen=self._window)
            self._intervals[stage].append(dt)
        self._tick_times[stage] = now

    def tock(self, stage):
        now = time.perf_counter()
        if stage in self._tick_times:
            dt = now - self._tick_times[stage]
            if stage not in self._durations:
                self._durations[stage] = self._deque(maxlen=self._window)
            self._durations[stage].append(dt)

    def avg_ms(self, stage):
        if stage not in self._durations or len(self._durations[stage]) == 0:
            return 0.0
        return np.mean(self._durations[stage]) * 1000

    def hz(self, stage):
        if stage not in self._intervals or len(self._intervals[stage]) == 0:
            return 0.0
        avg_interval = np.mean(self._intervals[stage])
        return 1.0 / avg_interval if avg_interval > 0 else 0.0

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


# =====================================================================
#  ENVIRONMENT ACCUMULATOR (confidence grid on GPU)
# =====================================================================

class EnvironmentGrid:
    def __init__(self, voxel_size, bounds, device, max_confidence=20):
        self.voxel_size = voxel_size
        self.device = device
        self.max_confidence = max_confidence

        self.mins = torch.tensor([b[0] for b in bounds], device=device, dtype=torch.float32)
        self.maxs = torch.tensor([b[1] for b in bounds], device=device, dtype=torch.float32)
        extents = self.maxs - self.mins
        self.shape = tuple(int(torch.ceil(extents[i] / voxel_size).item()) for i in range(3))

        self.confidence = torch.zeros(self.shape, device=device, dtype=torch.int16)
        rospy.loginfo(f"EnvironmentGrid: {self.shape}, "
                      f"VRAM={self.confidence.nelement()*2/1e6:.1f}MB")

    def _to_flat(self, points):
        idx = ((points - self.mins) / self.voxel_size).long()
        for d in range(3):
            idx[:, d].clamp_(0, self.shape[d] - 1)
        flat = (idx[:, 0] * self.shape[1] * self.shape[2]
                + idx[:, 1] * self.shape[2]
                + idx[:, 2])
        return torch.unique(flat)

    def integrate(self, points_world):
        if points_world is None or len(points_world) == 0:
            return
        flat = self._to_flat(points_world)
        self.confidence.view(-1)[flat] = torch.clamp(
            self.confidence.view(-1)[flat] + 1, max=self.max_confidence
        )

    def clear_robot(self, robot_points, margin=0.02):
        if robot_points is None or len(robot_points) == 0:
            return
        flat = self._to_flat(robot_points)
        self.confidence.view(-1)[flat] = 0
        margin_v = max(1, int(margin / self.voxel_size))
        if margin_v > 1:
            mask = torch.zeros_like(self.confidence, dtype=torch.float32)
            mask.view(-1)[flat] = 1.0
            k = 2 * margin_v + 1
            dilated = F.max_pool3d(
                mask.unsqueeze(0).unsqueeze(0),
                kernel_size=k, stride=1, padding=margin_v
            ).squeeze()
            self.confidence[dilated > 0] = 0

    def clear_points(self, points_world):
        if points_world is None or len(points_world) == 0:
            return
        flat = self._to_flat(points_world)
        self.confidence.view(-1)[flat] = 0

    def decay(self, amount=1):
        self.confidence = torch.clamp(self.confidence - amount, min=0)

    def get_occupied_points(self, min_confidence=2, max_points=25000):
        mask = self.confidence >= min_confidence
        idx = torch.nonzero(mask, as_tuple=False)
        if len(idx) == 0:
            return None
        pts = idx.float() * self.voxel_size + self.mins + self.voxel_size / 2
        if len(pts) > max_points:
            perm = torch.randperm(len(pts), device=self.device)[:max_points]
            pts = pts[perm]
        return pts

    def count(self):
        return (self.confidence >= 1).sum().item()


# =====================================================================
#  GPU UNPROJECTION
# =====================================================================

def gpu_unproject(depth, mask, T, fx, fy, cx, cy, z_min=0.01, z_max=1.5):
    valid = mask & (depth > z_min) & (depth < z_max)
    v, u = torch.where(valid)
    if len(v) < 10:
        return None
    z = depth[v, u]
    pts_cam = torch.stack([
        (u.float() - cx) * z / fx,
        (v.float() - cy) * z / fy,
        z,
        torch.ones_like(z)
    ], dim=1)
    return (T @ pts_cam.T).T[:, :3]

def cpu_unproject(depth_np, mask_np, T_np, fx, fy, cx, cy, z_min=0.01, z_max=1.5):
    """Same math, runs on CPU. Fast enough for obstacle cloud."""
    valid = mask_np & (depth_np > z_min) & (depth_np < z_max)
    v, u = np.where(valid)
    if len(v) < 10:
        return None
    z = depth_np[v, u]
    pts_cam = np.stack([
        (u - cx) * z / fx,
        (v - cy) * z / fy,
        z,
        np.ones_like(z)
    ], axis=1).astype(np.float32)
    pts_world = (T_np @ pts_cam.T).T[:, :3]
    return pts_world

# =====================================================================
#  MAIN NODE
# =====================================================================

class OmniGuidePerceptionNode:

    def __init__(self):
        rospy.init_node('omniguide_perception', anonymous=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Timing ──
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

        # ── Cutie (food tracking) ──
        # rospy.loginfo("Loading Cutie VOS model...")
        # self.cutie_model = get_default_model().to(self.device)
        # self.processor = InferenceCore(self.cutie_model, cfg=self.cutie_model.cfg)

       # ── Cutie SMALL ──
        rospy.loginfo("Loading Cutie VOS model (SMALL)...")
        self.cutie_model = get_small_model()
        self.processor = InferenceCore(self.cutie_model, cfg=self.cutie_model.cfg)

        self.processor.max_internal_size = 240
        self.cube_locked = False
        self.last_mask = None
        self.frame_count = 0
        rospy.loginfo("Cutie ready.")

        # ── SAM3 client (used for both food and container) ──
        self.sam3 = Sam3Client()

        # ── SAM3 food detection (async via topics to sam3_detector_node) ──
        self.sam3_pub = rospy.Publisher('/vision/sam3_request', Image, queue_size=1)
        self.sam3_sub = rospy.Subscriber('/vision/sam3_reply', Image, self._sam3_cb, queue_size=1)
        self.sam3_result = None
        self.waiting_sam3 = False
        self.sam3_time = None

        # ── Container (one-shot bbox, detected at startup) ──
        self.container = ContainerTracker(device=self.device, margin=0.10)

        # ── SDF ──
        sdf_bounds = ((-0.2, 0.8), (-0.5, 0.5), (0.0, 0.8))
        self.sdf = VoxelSDF_GPU(voxel_size=0.01, bounds=sdf_bounds, device=self.device)

        # ── Environment accumulator ──
        self.env_grid = EnvironmentGrid(
            voxel_size=0.01, bounds=sdf_bounds, device=self.device
        )
        self._decay_counter = 0

        # ── Target model ──
        self.target_model = PersistentTargetModel(
            voxel_size=0.003, max_points=15000, device=self.device,
            min_reference_size=300,
            stable_frames_needed=5,
        )

        # ── Grasp state ──
        self.is_grasped = False
        self.grasped_local = None
        self.CONTACT_DIST = 0.005

        # ── Fork mesh cache ──
        self.fork_cloud = None
        self.last_joint_hash = None

        # ── Camera intrinsics (defaults, overwritten by CameraInfo) ──
        self.fx = torch.tensor(604.9, device=self.device)
        self.fy = torch.tensor(604.9, device=self.device)
        self.cx = torch.tensor(320.0, device=self.device)
        self.cy = torch.tensor(240.0, device=self.device)

        # ── Obstacle thread stash ──
        self._stash_lock = threading.Lock()
        self._stash = None
        self._obstacle_event = threading.Event()
        self._running = True

        # ── Publishers ──
        self.pub_merged = rospy.Publisher('/vision/merged_cloud', PointCloud2, queue_size=1)
        self.pub_centroid = rospy.Publisher('/vision/target_centroid', PointStamped, queue_size=1)
        self.pub_obstacle = rospy.Publisher('/vision/obstacle_cloud', PointCloud2, queue_size=1)
        self.pub_sdf_viz = rospy.Publisher('/vision/sdf_viz', PointCloud2, queue_size=1)
        self.pub_sdf_slice = rospy.Publisher('/vision/sdf_slice', PointCloud2, queue_size=1)
        self.pub_perf = rospy.Publisher('/vision/perf', String, queue_size=1)

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

        # ── SDF visualization defaults ──
        rospy.set_param('/sdf_viz_mode', 'both')
        rospy.set_param('/sdf_slice_axis', 'x')
        rospy.set_param('/sdf_slice_value', 0.42)
        rospy.set_param('/sdf_viz_max_dist', 0.15)

        # ── Start obstacle thread ──
        self._obstacle_thread = threading.Thread(target=self._obstacle_loop, daemon=True)
        self._obstacle_thread.start()

        rospy.loginfo("OmniGuide Perception Node READY")

    # ==================================================================
    #  SMALL CALLBACKS
    # ==================================================================

    def _cam_info_cb(self, msg):
        self.fx = torch.tensor(msg.K[0], device=self.device)
        self.fy = torch.tensor(msg.K[4], device=self.device)
        self.cx = torch.tensor(msg.K[2], device=self.device)
        self.cy = torch.tensor(msg.K[5], device=self.device)
        self.sub_info.unregister()

    def _sam3_cb(self, msg):
        mask = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        self.sam3_result = (mask > 0) if np.sum(mask) > 50 else None
        self.waiting_sam3 = False

    # ==================================================================
    #  CUTIE HELPERS
    # ==================================================================

    def _prepare_image(self, cv_img):
        # cv_img is already RGB (rgb8 encoding from RealSense)
        img_t = torch.from_numpy(cv_img).to(self.device, non_blocking=True)
        return img_t.permute(2, 0, 1).float().div_(255.0)

    def _joint_cb(self, msg):
        if not self.mesh_loader:
            return
        try:
            # Correction silencieuse pour correspondre au nommage 'jointX' au lieu de 'panda_jointX'
            jmap = {n: msg.position[i] for i, n in enumerate(msg.name) if "joint" in n}
            jh = tuple(round(v, 4) for v in jmap.values())
            if jh != self.last_joint_hash:
                self.fork_cloud = self.mesh_loader.create_point_cloud_fork_tip(jmap)
                self.last_joint_hash = jh
        except Exception:
            pass

    def _init_cutie(self, cv_img, mask_np):
        try:
            # On force la configuration Cutie
            with open_dict(self.cutie_model.cfg):
                # ── LA LIGNE CRITIQUE : Coupe la mémoire infinie ──
                self.cutie_model.cfg.use_long_term = False 
                self.cutie_model.cfg.max_mem_frames = 3
                self.cutie_model.cfg.mem_every = 2
                self.cutie_model.cfg.stagger_updates = 2
                
            self.processor = InferenceCore(self.cutie_model, cfg=self.cutie_model.cfg)
            self.processor.max_internal_size = 240

            image_t = self._prepare_image(cv_img)
            mask_t = torch.from_numpy(mask_np > 0).to(self.device)
            objects = [1]
            mask_t = mask_t.long() * objects[0]

            with torch.inference_mode(), torch.cuda.amp.autocast():
                self.processor.step(image_t, mask_t, objects=objects)

            return True
        except Exception as e:
            rospy.logerr(f"Cutie Init error: {e}")
            return False
        
    def _track_cutie(self, cv_img):
        try:
            image_t = self._prepare_image(cv_img)
            with torch.inference_mode(), torch.cuda.amp.autocast():
                output_prob = self.processor.step(image_t)
                mask_t = self.processor.output_prob_to_mask(output_prob)

            # Pin memory + non_blocking avoids a full GPU sync
            mask_np = mask_t.to('cpu', non_blocking=True)
            torch.cuda.current_stream().synchronize()  # wait only for default stream
            mask_np = mask_np.numpy().squeeze().astype(np.uint8)
            return mask_np > 0
        except Exception as e:
            rospy.logerr(f"Cutie Track error: {e}")
            return None

    def _run_tracking(self, cv_img, rgb_msg):
        if not self.cube_locked:
            if self.sam3_result is not None:
                if self._init_cutie(cv_img, self.sam3_result):
                    self.cube_locked = True
                    self.last_mask = self.sam3_result
                    self.sam3_result = None
                    return self.last_mask

            if not self.waiting_sam3 or (
                self.sam3_time and (rospy.Time.now() - self.sam3_time).to_sec() > 3.0
            ):
                self.sam3_pub.publish(rgb_msg)
                self.waiting_sam3 = True
                self.sam3_time = rospy.Time.now()
            return None

        # Tracking continu sans interruption
        try:
            mask = self._track_cutie(cv_img)
            if mask is not None and np.sum(mask) > 50:
                self.last_mask = mask
                return mask
            else:
                self.cube_locked = False
                return None
        except Exception as e:
            rospy.logerr(f"Cutie error: {e}")
            self.cube_locked = False
            return None

    # ==================================================================
    #  MAIN CALLBACK
    # ==================================================================

    def _callback(self, rgb_msg, depth_msg, joint_msg):
        self.perf.tick('callback')

        # ── 1. Parse + TF ──
        self.perf.tick('tf_lookup')
        cv_img = self._decode_img(rgb_msg)
        if cv_img is None:
            return
        cv_depth = self._decode_depth(depth_msg)
        T_world_tcp = self._lookup_tf(rgb_msg.header.stamp)
        if T_world_tcp is None:
            return
        T_world_cam = T_world_tcp @ self.T_tcp_cam
        T_world_fork = T_world_tcp @ self.T_tcp_fork_tip
        fork_tip = T_world_fork[:3, 3]
        self.perf.tock('tf_lookup')

        robot_pts = self.fork_cloud

        # ── 2. Contact / grasp ──
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

        # ── 3. Cutie tracking (food) ──
        self.perf.tick('cutie')
        current_mask = None
        if not skip_tracking:
            current_mask = self._run_tracking(cv_img, rgb_msg)
        self.frame_count += 1
        self.perf.tock('cutie')

        # ── 4. Target model update ──
        self.perf.tick('target_cloud')
        if cv_depth is not None:
            z = cv_depth if cv_depth.dtype == np.float32 else cv_depth / 1000.0

            target_result = self.target_model.update(
                mask=current_mask,
                depth=z,
                T_world_cam=T_world_cam,
                fx=self.fx.item(), fy=self.fy.item(),
                cx=self.cx.item(), cy=self.cy.item(),
            )

            if target_result.get('jumped', False):
                rospy.logwarn("Target jump detected! Flushing Environment SDF.")
                self.env_grid.confidence.zero_()

            rospy.loginfo_throttle(2,
                f"Target: state={target_result.get('state', '?')}, "
                f"mode={target_result['mode']}, "
                f"depth_valid={target_result['depth_valid_ratio']:.0%}, "
                f"model={target_result['model_size']} voxels"
            )

            # Stash for obstacle thread
            if current_mask is not None:
                with self._stash_lock:
                    self._stash = {
                        'mask': current_mask,
                        'depth': z.astype(np.float32),
                        'T': T_world_cam.astype(np.float32),
                        'fork': self.fork_cloud,
                    }
                self._obstacle_event.set()
        self.perf.tock('target_cloud')

        # ── 5. Publish centroid ──
        if self.is_grasped:
            c = (T_world_fork @ np.array([0, 0, 0, 1]))[:3]
            self._pub_centroid(c.astype(np.float32))
        else:
            c = self.target_model.get_centroid_np()
            if c is not None:
                self._pub_centroid(c)

        # ── 6. Merge + publish condition cloud ──
        self.perf.tick('merge')
        self._publish_merged(robot_pts, T_world_fork)
        self.perf.tock('merge')

        self.perf.tock('callback')

        # ── 7. Publish perf summary every ~2s ──
        if self.frame_count % 30 == 0:
            cb_stages = ['callback', 'tf_lookup', 'cutie', 'target_cloud',
                         'grasp_check', 'merge']
            obs_stages = ['obstacle_loop', 'obs_unproject', 'obs_accumulate',
                          'obs_robot_clear', 'obs_sdf_update', 'obs_publish']
            summary = (
                "CB:  " + self.perf.summary(cb_stages) +
                "\nSDF: " + self.perf.summary(obs_stages)
            )
            rospy.loginfo(f"\n{summary}")
            self.pub_perf.publish(String(data=summary))

    # ==================================================================
    #  OBSTACLE THREAD
    # ==================================================================

    def _obstacle_loop(self):
        stream = torch.cuda.Stream(device=self.device)

        while self._running and not rospy.is_shutdown():
            self._obstacle_event.wait(timeout=0.2)
            self._obstacle_event.clear()

            with self._stash_lock:
                stash = self._stash
                self._stash = None
            if stash is None:
                continue

            self.perf.tick('obstacle_loop')

            try:
                # ── 1. Unproject ──
                self.perf.tick('obs_unproject')

                depth_np = stash['depth']
                T_np = stash['T']
                mask_np = stash['mask']

                fx = self.fx.item()
                fy = self.fy.item()
                cx = self.cx.item()
                cy = self.cy.item()

                if mask_np is not None:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
                    dilated = cv2.dilate(mask_np.astype(np.uint8), kernel, iterations=1) > 0
                    
                    target_3d_np = cpu_unproject(depth_np, dilated, T_np, fx, fy, cx, cy)
                    obs_mask = ~dilated
                else:
                    target_3d_np = None
                    obs_mask = np.ones(depth_np.shape, dtype=bool)

                obs_pts_np = cpu_unproject(depth_np, obs_mask, T_np, fx, fy, cx, cy)
                self.perf.tock('obs_unproject')

                if obs_pts_np is None or len(obs_pts_np) < 50:
                    self.perf.tock('obstacle_loop')
                    continue

                # Container filter (still on GPU, convert once)
                obs_pts = torch.from_numpy(obs_pts_np).to(self.device)
                if self.container.has_bbox():
                    obs_pts = self.container.filter_points_gpu(obs_pts)
                    if obs_pts is None or len(obs_pts) < 50:
                        self.perf.tock('obstacle_loop')
                        continue

                # Clear target voxels
                self.perf.tick('obs_accumulate')
                if target_3d_np is not None and len(target_3d_np) > 0:
                    target_3d = torch.from_numpy(target_3d_np).to(self.device)
                    self.env_grid.clear_points(target_3d)

                # Integrate
                self.env_grid.integrate(obs_pts)
                self.perf.tock('obs_accumulate')
                
                # ── 4. Clear robot body voxels ──
                self.perf.tick('obs_robot_clear')
                fork = stash['fork']
                if fork is not None and len(fork) > 0:
                    if isinstance(fork, np.ndarray):
                        fork_t = torch.from_numpy(fork.astype(np.float32)).to(
                            self.device, non_blocking=True
                        )
                    else:
                        fork_t = fork.to(self.device).float()
                    self.env_grid.clear_robot(fork_t, margin=0.02)
                self.perf.tock('obs_robot_clear')

                # ── 5. Decay ──
                self._decay_counter += 1
                if self._decay_counter % 50 == 0:
                    self.env_grid.decay(amount=1)

                # ── 6. SDF update ──
                self.perf.tick('obs_sdf_update')
                self._sdf_counter = getattr(self, '_sdf_counter', 0) + 1
                if self._sdf_counter % 5 == 0:  # SDF at ~3-5 Hz instead of every frame
                    acc_pts = self.env_grid.get_occupied_points(min_confidence=2)
                    if acc_pts is not None and len(acc_pts) > 100:
                        self.sdf.update(acc_pts)
                    self.perf.tock('obs_sdf_update')
                else:
                    pass

                # ── 7. Publish for RViz ──
                self.perf.tick('obs_publish')

                # Ne publier vers RViz qu'une fois sur trois pour libérer le GPU
                if self._decay_counter % 3 == 0:
                    viz_pts = self.env_grid.get_occupied_points(max_points=20000)
                    if viz_pts is not None:
                        self._pub_cloud(viz_pts.cpu().numpy(), self.pub_obstacle)

                    viz_mode = rospy.get_param('/sdf_viz_mode', 'both')
                    if viz_mode != 'off':
                        max_dist = rospy.get_param('/sdf_viz_max_dist', 0.15)

                        if viz_mode in ('volume', 'both'):
                            pts, rgb, _ = self.sdf.get_visualization_points(
                                max_dist=max_dist, stride=3
                            )
                            if pts is not None:
                                msg = VoxelSDF_GPU.make_rviz_cloud_msg(pts, rgb)
                                if msg:
                                    self.pub_sdf_viz.publish(msg)

                        if viz_mode in ('slice', 'both'):
                            axis = rospy.get_param('/sdf_slice_axis', 'z')
                            val = rospy.get_param('/sdf_slice_value', 0.05)
                            pts, rgb, _ = self.sdf.get_slice_points(
                                axis=axis, value=val, max_dist=max_dist
                            )
                            if pts is not None:
                                msg = VoxelSDF_GPU.make_rviz_cloud_msg(pts, rgb)
                                if msg:
                                    self.pub_sdf_slice.publish(msg)

                self.perf.tock('obs_publish')

            except Exception as e:
                rospy.logerr(f"Obstacle thread error: {e}")

            self.perf.tock('obstacle_loop')

    # ==================================================================
    #  MERGE + PUBLISH (1024 pts)
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

        if not parts:
            return
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
        if points is None or len(points) == 0:
            return
        hdr = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id=frame)
        publisher.publish(pc2.create_cloud_xyz32(hdr, points.astype(np.float32)))

    def _pub_centroid(self, c):
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.point.x, msg.point.y, msg.point.z = float(c[0]), float(c[1]), float(c[2])
        self.pub_centroid.publish(msg)

    # ==================================================================
    #  CONTAINER DETECTION (one-shot at startup)
    # ==================================================================

    def _detect_container(self):
        """
        Try to detect the container (bowl/plate) via SAM3 service.
        Called once at startup. Uses the first available camera frame.
        """
        rospy.loginfo("[Container] Waiting for camera frame to detect container...")
        try:
            rgb_msg = rospy.wait_for_message("/synced/camera_wrist/rgb", Image, timeout=10.0)
            depth_msg = rospy.wait_for_message("/synced/camera_wrist/depth", Image, timeout=10.0)
        except rospy.ROSException:
            rospy.logwarn("[Container] No camera frame received, skipping container detection.")
            return

        cv_img = self._decode_img(rgb_msg)
        cv_depth = self._decode_depth(depth_msg)
        if cv_img is None or cv_depth is None:
            rospy.logwarn("[Container] Bad image data, skipping container detection.")
            return

        # Get TF
        try:
            t, r = self.tf_listener.lookupTransform("world", "panda_hand_tcp", rospy.Time(0))
        except tf.Exception:
            rospy.logwarn("[Container] No TF available, skipping container detection.")
            return
        T = tft.quaternion_matrix(r)
        T[:3, 3] = t
        T_world_cam = T @ self.T_tcp_cam

        # Try a few prompts
        for prompt in ["bowl", "plate", "container", "white circle containing green cube"]:
            rospy.loginfo(f"[Container] Trying SAM3 with prompt: '{prompt}'...")
            mask, score = self.sam3.segment(rgb_msg, prompt, confidence_threshold=0.05)
            if mask is not None:
                z = cv_depth if cv_depth.dtype == np.float32 else cv_depth / 1000.0
                success = self.container.init_from_mask(
                    mask > 0, z, T_world_cam,
                    self.fx.item(), self.fy.item(),
                    self.cx.item(), self.cy.item()
                )
                if success:
                    rospy.loginfo(f"[Container] Detected with prompt '{prompt}' (score={score:.2f})")
                    return

        rospy.logwarn("[Container] Could not detect container. SDF will use full workspace.")

    # ==================================================================
    #  RUN
    # ==================================================================

    def run(self):
        self.tf_listener = tf.TransformListener()
        rospy.sleep(1.0)  # wait for TF buffer to fill

        # One-shot container detection (blocking, before main loop)
        self._detect_container()

        rospy.loginfo("Starting main loop.")
        rospy.spin()
        self._running = False
        self._obstacle_event.set()


if __name__ == '__main__':
    OmniGuidePerceptionNode().run()