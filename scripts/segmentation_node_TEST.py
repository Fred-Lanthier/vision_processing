#!/usr/bin/env python3
"""
Segmentation Node v3 — Thread-based SAM3 + SAM2.1 (like the standalone script)
================================================================================
Architecture (matches your working standalone video_segmentation_from_detections_SAM2.py):

  Thread A: SAM3 DETECTOR
    - Runs in a continuous loop
    - Reads latest RGB frame from a shared buffer
    - For each prompt, runs detection
    - Pushes results to a queue

  Thread B: SAM2.1 TRACKER
    - Runs in a continuous loop
    - Reads latest RGB frame from a shared buffer
    - Consumes new detections from the queue
    - Tracks all objects, produces masks
    - Publishes masks to a shared dict

  Synced Callback (main ROS thread):
    - Triggered by synchronized RGB + Depth + JointState
    - Reads latest masks from shared dict
    - Runs PersistentTargetModel (ICP) for target and container
    - Publishes point clouds, bboxes, centroids

This is the SAME architecture as your standalone script, just reading
from ROS topics instead of a webcam.
"""

# ── ANTI CPU-THRASHING ──
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import rospy
import numpy as np
import time
import threading
import queue
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
from concurrent.futures import ThreadPoolExecutor, wait
from cv_bridge import CvBridge

rospack = rospkg.RosPack()
sys.path.append(os.path.join(rospack.get_path('vision_processing'), 'scripts'))
from utils import compute_T_child_parent_xacro
from PersistentTargetModel_TEST import PersistentTargetModel

# ── SAM3 + SAM2.1 (both loaded in this node) ──
rospack = rospkg.RosPack()
sys.path.append(os.path.join(rospack.get_path('vision_processing'), 'third_party/muggled_sam'))
from muggled_sam.make_sam import make_sam_from_state_dict
from muggled_sam.demo_helpers.video_data_storage import SAMVideoObjectResults
from muggled_sam.demo_helpers.bounding_boxes import get_2box_iou

try:
    from Compute_3D_point_cloud_from_mesh import RobotMeshLoaderOptimized
    MESH_LOADER_OK = True
except ImportError:
    MESH_LOADER_OK = False


# =====================================================================
#  CONFIG
# =====================================================================
DETECTION_MODEL_PATH = "/home/flanthier/.cache/huggingface/hub/models--facebook--sam3/snapshots/2afe64078f4420bdfbc063162d1336003efadc81/sam3.pt"
TRACKING_MODEL_PATH  = "/home/flanthier/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt"
 
# Detection
DETECTION_SCORE_THRESHOLD  = 0.35
EXISTING_BOX_IOU_THRESHOLD = 0.25
 
# Tracking
REMOVE_AFTER_N_MISSED     = 8
MAX_MEMORY_FRAMES         = 6
SAME_LABEL_MERGE_IOU      = 0.20
REID_MEMORY_DURATION      = 15.0
 
# Adaptive detection cadence
DET_INTERVAL_STABLE  = 0.20
DET_INTERVAL_HUNTING = 0.05
 
# Prompts
TARGET_PROMPT    = "green cube"     # overridden by rosparam /vision/target_prompt
CONTAINER_PROMPTS = ["bowl", "plate", "container"]
 
# Colors (BGR)
COLORS = [
    (0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 165, 255), (200, 100, 255),
]
 
 
# =====================================================================
#  PERF TRACKER
# =====================================================================
class PerfTracker:
    def __init__(self, window=30):
        self._window, self._durations, self._tick_times, self._intervals = window, {}, {}, {}
 
    def tick(self, s):
        now = time.perf_counter()
        if s in self._tick_times:
            if s not in self._intervals: self._intervals[s] = deque(maxlen=self._window)
            self._intervals[s].append(now - self._tick_times[s])
        self._tick_times[s] = now
 
    def tock(self, s):
        now = time.perf_counter()
        if s in self._tick_times:
            if s not in self._durations: self._durations[s] = deque(maxlen=self._window)
            self._durations[s].append(now - self._tick_times[s])
 
    def avg_ms(self, s):
        return np.mean(self._durations[s]) * 1000 if s in self._durations and self._durations[s] else 0.0
 
    def hz(self, s):
        if s not in self._intervals or not self._intervals[s]: return 0.0
        avg = np.mean(self._intervals[s])
        return 1.0 / avg if avg > 0 else 0.0
 
    def summary(self, stages):
        parts = []
        for s in stages:
            ms, hz = self.avg_ms(s), self.hz(s)
            if hz > 0: parts.append(f"{s}: {ms:.1f}ms ({hz:.0f}Hz)")
            elif ms > 0: parts.append(f"{s}: {ms:.1f}ms")
        return " | ".join(parts)
 
 
# =====================================================================
#  TRACKED OBJECT (same as standalone script)
# =====================================================================
class TrackedObject:
    def __init__(self, obj_id, label, color, memory, init_box_norm):
        self.obj_id = obj_id
        self.label = label
        self.color = color
        self.memory: SAMVideoObjectResults = memory
        self.missed = 0
        self.last_mask_np = None
        self.last_box_norm = init_box_norm
        self.last_score = 0.0
        self.stored_frame_count = 0
 
 
class ReIDGhost:
    def __init__(self, obj_id, label, color, last_box_norm, purge_time):
        self.obj_id, self.label, self.color = obj_id, label, color
        self.last_box_norm, self.purge_time = last_box_norm, purge_time
 
 
# =====================================================================
#  SHARED FRAME BUFFER (replaces FastWebcam for ROS)
# =====================================================================
class FrameBuffer:
    """Thread-safe latest-frame buffer. ROS callback writes, threads read."""
    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
 
    def put(self, frame):
        with self._lock:
            self._frame = frame
 
    def get(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None
 
 
# =====================================================================
#  HELPER
# =====================================================================
def box_from_mask(mask_np, shape_hw):
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0: return None
    h, w = shape_hw
    return torch.tensor([[xs.min()/w, ys.min()/h], [xs.max()/w, ys.max()/h]], dtype=torch.float32)
 
def prune_memory(memory, max_frames):
    store = getattr(memory, '_frame_results', None) or getattr(memory, 'frame_results', None)
    if store is not None:
        while len(store) > max_frames:
            store.pop(0)
 
 
# =====================================================================
#  SAM3 DETECTOR THREAD (continuous loop, like standalone)
# =====================================================================
def detector_thread(detmodel, frame_buf, prompts_ref, det_queue, shutdown_event):
    rospy.loginfo("[DETECTOR] SAM3 thread started (continuous loop).")
    enc_cfg = {"max_side_length": 512, "use_square_sizing": True}
 
    while not shutdown_event.is_set() and not rospy.is_shutdown():
        frame = frame_buf.get()
        if frame is None:
            time.sleep(0.01)
            continue
 
        with prompts_ref["lock"]:
            prompts = list(prompts_ref["prompts"])
            interval = prompts_ref.get("det_interval", DET_INTERVAL_STABLE)
 
        if not prompts:
            time.sleep(0.1)
            continue
 
        try:
            det_enc, _, _ = detmodel.encode_detection_image(frame, **enc_cfg)
 
            for prompt_text in prompts:
                exemplars = detmodel.encode_exemplars(det_enc, text=prompt_text)
                masks, boxes, _, _ = detmodel.generate_detections(
                    det_enc, exemplars,
                    detection_filter_threshold=DETECTION_SCORE_THRESHOLD,
                )
                n = masks.shape[1] if masks is not None else 0
                if n > 0:
                    det_queue.put({
                        "masks": masks,
                        "boxes": boxes,
                        "source_frame": frame,
                        "prompt": prompt_text,
                    })
        except Exception as e:
            rospy.logerr(f"[DETECTOR] Error: {e}")
 
        time.sleep(interval)
 
 
# =====================================================================
#  SAM2.1 TRACKER THREAD (continuous loop, like standalone)
# =====================================================================
def tracker_thread(track_model, frame_buf, det_queue, mask_state, prompts_ref, shutdown_event):
    rospy.loginfo("[TRACKER] SAM2.1 thread started (continuous loop).")
    enc_cfg = {"max_side_length": 512, "use_square_sizing": True}
 
    tracked: dict[int, TrackedObject] = {}
    ghosts: list[ReIDGhost] = []
    next_id = 0
    frame_idx = 0
 
    while not shutdown_event.is_set() and not rospy.is_shutdown():
        frame = frame_buf.get()
        if frame is None:
            time.sleep(0.01)
            continue
 
        h, w = frame.shape[:2]
        now = time.time()
 
        # Check prompt change → clear all
        with prompts_ref["lock"]:
            if prompts_ref.get("changed", False):
                prompts_ref["changed"] = False
                tracked.clear()
                ghosts.clear()
                rospy.loginfo("[TRACKER] Prompts changed → cleared all objects.")
 
        # ── A) INGEST NEW DETECTIONS ──
        while not det_queue.empty():
            try:
                det = det_queue.get_nowait()
            except queue.Empty:
                break
 
            new_masks = det["masks"]
            new_boxes = det["boxes"]
            src_frame = det["source_frame"]
            prompt    = det["prompt"]
            n_det     = new_masks.shape[1]
 
            src_enc, _, _ = track_model.encode_image(src_frame, **enc_cfg)
 
            known_boxes = []
            for obj in tracked.values():
                if obj.label == prompt and obj.last_box_norm is not None:
                    known_boxes.append(obj.last_box_norm.to(new_boxes.device))
 
            for i in range(n_det):
                new_box = new_boxes[0, i]
                is_known = any(get_2box_iou(new_box, kb) > EXISTING_BOX_IOU_THRESHOLD for kb in known_boxes)
                if is_known:
                    continue
 
                # Re-ID check
                reused_ghost = None
                for ghost in ghosts:
                    if ghost.label != prompt: continue
                    if (now - ghost.purge_time) > REID_MEMORY_DURATION: continue
                    label_ghosts = [g for g in ghosts if g.label == prompt and (now - g.purge_time) < REID_MEMORY_DURATION]
                    same_region = False
                    if ghost.last_box_norm is not None:
                        same_region = get_2box_iou(new_box, ghost.last_box_norm.to(new_boxes.device)) > 0.05
                    if same_region or len(label_ghosts) == 1:
                        reused_ghost = ghost
                        break
 
                raw_mask = new_masks[0, i]
                init_mem = track_model.initialize_from_mask(src_enc, raw_mask > 0)
                obj_mem = SAMVideoObjectResults.create()
                obj_mem.store_prompt_result(frame_idx, init_mem)
 
                if reused_ghost is not None:
                    obj = TrackedObject(reused_ghost.obj_id, prompt, reused_ghost.color, obj_mem, new_box.cpu().float())
                    ghosts.remove(reused_ghost)
                    rospy.loginfo(f"[TRACKER] Re-identified '{prompt}' → restored ID {obj.obj_id}")
                else:
                    color = COLORS[next_id % len(COLORS)]
                    obj = TrackedObject(next_id, prompt, color, obj_mem, new_box.cpu().float())
                    rospy.loginfo(f"[TRACKER] New '{prompt}' → ID {next_id}")
                    next_id += 1
 
                tracked[obj.obj_id] = obj
 
        # ── B) PROPAGATE TRACKING ──
        if tracked:
            enc_imgs, _, _ = track_model.encode_image(frame, **enc_cfg)
            dead_ids = []
 
            for obj in tracked.values():
                score, best_idx, preds, mem_enc, obj_ptr = \
                    track_model.step_video_masking(enc_imgs, **obj.memory.to_dict())
                obj.last_score = score.item()
 
                if score.item() < 0:
                    obj.missed += 1
                    if obj.missed > REMOVE_AFTER_N_MISSED:
                        dead_ids.append(obj.obj_id)
                else:
                    obj.missed = 0
                    obj.memory.store_frame_result(frame_idx, mem_enc, obj_ptr)
                    obj.stored_frame_count += 1
                    prune_memory(obj.memory, MAX_MEMORY_FRAMES)
 
                    mask_f = preds[0, best_idx].cpu().float().numpy().squeeze()
                    mask_full = cv2.resize(mask_f, (w, h), interpolation=cv2.INTER_LINEAR)
                    obj.last_mask_np = mask_full
                    obj.last_box_norm = box_from_mask(mask_full > 0, (h, w))
 
            for did in dead_ids:
                if did in tracked:
                    dead_obj = tracked.pop(did)
                    ghosts.append(ReIDGhost(dead_obj.obj_id, dead_obj.label, dead_obj.color,
                                           dead_obj.last_box_norm, now))
 
            ghosts[:] = [g for g in ghosts if (now - g.purge_time) < REID_MEMORY_DURATION]
 
        # ── C) ANTI-DRIFT MERGE ──
        labels_seen = {}
        for obj in list(tracked.values()):
            if obj.last_mask_np is None: continue
            labels_seen.setdefault(obj.label, []).append(obj)
 
        merge_kills = []
        for label, objs in labels_seen.items():
            if len(objs) < 2: continue
            for i in range(len(objs)):
                if objs[i].obj_id in merge_kills: continue
                for j in range(i+1, len(objs)):
                    if objs[j].obj_id in merge_kills: continue
                    mask_a = objs[i].last_mask_np > 0
                    mask_b = objs[j].last_mask_np > 0
                    inter = np.logical_and(mask_a, mask_b).sum()
                    union = np.logical_or(mask_a, mask_b).sum()
                    if union > 0 and inter / union > SAME_LABEL_MERGE_IOU:
                        victim = objs[j] if objs[i].last_score >= objs[j].last_score else objs[i]
                        merge_kills.append(victim.obj_id)
 
        for mid in merge_kills:
            tracked.pop(mid, None)
 
        # ── D) PUBLISH MASKS ──
        result = {}
        for obj in tracked.values():
            if obj.last_mask_np is not None:
                result[obj.label] = {
                    "id": obj.obj_id,
                    "label": obj.label,
                    "mask": (obj.last_mask_np > 0),  # bool H×W
                    "score": obj.last_score,
                }
 
        with mask_state["lock"]:
            mask_state["masks"] = result
 
        # Adaptive detection speed
        with prompts_ref["lock"]:
            active_labels = {o.label for o in tracked.values()}
            all_found = all(p in active_labels for p in prompts_ref["prompts"])
            prompts_ref["det_interval"] = DET_INTERVAL_STABLE if all_found else DET_INTERVAL_HUNTING
 
        frame_idx += 1
 
 
# =====================================================================
#  SEGMENTATION NODE
# =====================================================================
class SegmentationNode:
 
    def __init__(self):
        rospy.init_node('segmentation_node', anonymous=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
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
 
        # ── Load BOTH models in this node ──
        rospy.loginfo("Loading SAM 3.1 (detector)...")
        _, sam3 = make_sam_from_state_dict(DETECTION_MODEL_PATH)
        sam3.to(device=self.device, dtype=self.dtype)
        self.detmodel = sam3.make_detector_model()
 
        rospy.loginfo("Loading SAM 2.1 Tiny (tracker)...")
        _, self.sam2_model = make_sam_from_state_dict(TRACKING_MODEL_PATH)
        self.sam2_model.to(device=self.device, dtype=self.dtype)
        rospy.loginfo("Both models loaded.")
 
        # ── Shared state between threads ──
        self.frame_buf = FrameBuffer()
        self.det_queue = queue.Queue()
        self.shutdown_event = threading.Event()
 
        # mask_state: tracker thread writes masks here, callback reads them
        self.mask_state = {"masks": {}, "lock": threading.Lock()}
 
        # prompts_ref: what to detect
        target_prompt = rospy.get_param('/vision/target_prompt', TARGET_PROMPT)
        all_prompts = CONTAINER_PROMPTS + [target_prompt]
        self.target_prompt = target_prompt
 
        self.prompts_ref = {
            "prompts":      all_prompts,
            "det_interval": DET_INTERVAL_HUNTING,
            "changed":      False,
            "lock":         threading.Lock(),
        }
 
        # ── 3D Models (ICP) ──
        self.target_model = PersistentTargetModel(
                    name="TARGET",
                    voxel_size=0.001, max_points=1000, device=self.device,
                    min_reference_size=100, stable_frames_needed=3,
                    icp_max_translation=0.10, icp_max_iterations=10,
                    icp_max_dist_factor=10.0,
                    allow_fusion=False,
                    icp_fail_reset=20,       # was 5 → gives ~1.7s for tracker to recover
                )
        self.container_model = PersistentTargetModel(
                    name="CONTAINER",
                    voxel_size=0.005, max_points=2500, device=self.device,
                    min_reference_size=100, stable_frames_needed=3,
                    icp_max_translation=0.15, icp_max_iterations=10,
                    allow_fusion=False,
                    depth_variance_max=0.15,
                    icp_fail_reset=20,
                )
 
        # ── Grasp state ──
        self.is_grasped = False
        self.grasped_local = None
        self.CONTACT_DIST = 0.005
        self.frame_count = 0
 
        # ── Fork mesh ──
        self.fork_cloud = None
        self.last_joint_hash = None
 
        # ── Camera intrinsics ──
        self.fx = torch.tensor(604.9, device=self.device)
        self.fy = torch.tensor(604.9, device=self.device)
        self.cx = torch.tensor(320.0, device=self.device)
        self.cy = torch.tensor(240.0, device=self.device)
 
        # ── Thread pool for parallel ICP ──
        self.model_executor = ThreadPoolExecutor(max_workers=2)
 
        # ── Publishers ──
        self.pub_target_bbox = rospy.Publisher('/vision/target_bbox', Float32MultiArray, queue_size=1)
        self.pub_container_bbox = rospy.Publisher('/vision/container_bbox', Float32MultiArray, queue_size=1, latch=True)
        self.pub_merged = rospy.Publisher('/vision/merged_cloud', PointCloud2, queue_size=1)
        self.pub_centroid = rospy.Publisher('/vision/target_centroid', PointStamped, queue_size=1)
        self.pub_perf = rospy.Publisher('/vision/seg_perf', String, queue_size=1)
        self.pub_container_cloud = rospy.Publisher('/vision/container_cloud', PointCloud2, queue_size=1)
        self.pub_target_cloud = rospy.Publisher('/vision/target_cloud', PointCloud2, queue_size=1)

        self.bridge = CvBridge()
        self.pub_mask_viz = rospy.Publisher('/vision/sam2_segmentation_viz', Image, queue_size=1)
        
        # ── Subscribers ──
        self.sub_info = rospy.Subscriber("/camera_wrist/color/camera_info", CameraInfo, self._cam_info_cb)
 
        # Raw RGB for the frame buffer (feeds both SAM3 and SAM2.1 threads)
        self.sub_rgb_raw = rospy.Subscriber("/synced/camera_wrist/rgb", Image, self._rgb_cb, queue_size=1)
 
        # Synced callback for depth + joints → 3D
        sub_rgb = message_filters.Subscriber("/synced/camera_wrist/rgb", Image)
        sub_dep = message_filters.Subscriber("/synced/camera_wrist/depth", Image)
        sub_jnt = message_filters.Subscriber("/synced/joint_states", JointState)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_dep, sub_jnt], queue_size=5, slop=0.1
        )
        self.ts.registerCallback(self._synced_callback)
 
        self.sub_joints_mesh = rospy.Subscriber("/joint_states", JointState, self._joint_cb, queue_size=1)
 
        rospy.loginfo("Segmentation Node v3 READY (threaded SAM3+SAM2.1)")
 
    # ==================================================================
    #  ROS CALLBACKS
    # ==================================================================
 
    def _cam_info_cb(self, msg):
        self.fx = torch.tensor(msg.K[0], device=self.device)
        self.fy = torch.tensor(msg.K[4], device=self.device)
        self.cx = torch.tensor(msg.K[2], device=self.device)
        self.cy = torch.tensor(msg.K[5], device=self.device)
        self.sub_info.unregister()
 
    def _rgb_cb(self, msg):
        """Feed latest RGB frame to the shared buffer for SAM3/SAM2.1 threads."""
        if msg.encoding == "rgb8":
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
            self.frame_buf.put(frame)
 
    def _joint_cb(self, msg):
        if not self.mesh_loader: return
        try:
            jmap = {n: msg.position[i] for i, n in enumerate(msg.name) if "joint" in n}
            jh = tuple(round(v, 4) for v in jmap.values())
            if jh != self.last_joint_hash:
                self.fork_cloud = self.mesh_loader.create_point_cloud_fork_tip(jmap)
                self.last_joint_hash = jh
        except Exception:
            pass
 
    # ==================================================================
    #  SYNCED CALLBACK (depth → 3D → ICP → publish)
    # ==================================================================
 
    def _synced_callback(self, rgb_msg, depth_msg, joint_msg):
        self.perf.tick('callback')
 
        # ── 1. Parse + TF ──
        self.perf.tick('tf_lookup')
        cv_depth = self._decode_depth(depth_msg)
        T_world_tcp = self._lookup_tf()
        if cv_depth is None or T_world_tcp is None:
            return
        T_world_cam = T_world_tcp @ self.T_tcp_cam
        T_world_fork = T_world_tcp @ self.T_tcp_fork_tip
        fork_tip = T_world_fork[:3, 3]
        self.perf.tock('tf_lookup')
 
        robot_pts = self.fork_cloud
        self.frame_count += 1
 
        # ── 2. Read latest masks from tracker thread ──
        with self.mask_state["lock"]:
            masks = dict(self.mask_state["masks"])
 
        depth_h, depth_w = cv_depth.shape[:2]
 
        # Find target mask and container mask, resize to depth resolution
        target_mask = None
        container_mask = None
 
        if self.target_prompt in masks:
            raw = masks[self.target_prompt]["mask"]
            # CRITICAL: mask comes from tracker thread at RGB resolution,
            # depth may be a different resolution. Resize to match.
            if raw.shape[:2] != (depth_h, depth_w):
                raw = cv2.resize(raw.astype(np.uint8), (depth_w, depth_h),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
            target_mask = raw
 
        for cp in CONTAINER_PROMPTS:
            if cp in masks:
                raw = masks[cp]["mask"]
                if raw.shape[:2] != (depth_h, depth_w):
                    raw = cv2.resize(raw.astype(np.uint8), (depth_w, depth_h),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
                container_mask = raw
                break

        # ── AJOUT : Exclusion mutuelle (La cible coupe le contenant) ──
        if target_mask is not None and container_mask is not None:
            # Opération booléenne : Le contenant devient (Contenant ET NON Cible)
            container_mask = container_mask & (~target_mask)

        # ── AJOUT : Génération de l'image de visualisation (Overlay) ──
        self.perf.tick('viz_overlay')
        try:
            # 1. Décoder l'image RGB synchronisée (qui est au format rgb8)
            cv_rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, 3).copy()
            
            alpha = 0.5 # Transparence
            
            # 2. Colorier le masque de la cible (en Vert)
            if target_mask is not None:
                color_target = np.array([0, 255, 0], dtype=np.uint8) # RGB: Vert
                # Redimensionnement de sécurité au cas où l'image RGB et Depth ont des résolutions différentes
                t_mask_viz = target_mask
                if t_mask_viz.shape[:2] != cv_rgb.shape[:2]:
                    t_mask_viz = cv2.resize(t_mask_viz.astype(np.uint8), (cv_rgb.shape[1], cv_rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                roi = cv_rgb[t_mask_viz]
                cv_rgb[t_mask_viz] = (roi * (1.0 - alpha) + color_target * alpha).astype(np.uint8)

            # 3. Colorier le masque du contenant (en Bleu)
            if container_mask is not None:
                color_container = np.array([0, 0, 255], dtype=np.uint8) # RGB: Bleu
                c_mask_viz = container_mask
                if c_mask_viz.shape[:2] != cv_rgb.shape[:2]:
                    c_mask_viz = cv2.resize(c_mask_viz.astype(np.uint8), (cv_rgb.shape[1], cv_rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                roi = cv_rgb[c_mask_viz]
                cv_rgb[c_mask_viz] = (roi * (1.0 - alpha) + color_container * alpha).astype(np.uint8)

            # 4. Conversion et Publication
            viz_msg = self.bridge.cv2_to_imgmsg(cv_rgb, encoding="rgb8")
            viz_msg.header = rgb_msg.header # On garde le même timestamp et frame_id
            self.pub_mask_viz.publish(viz_msg)
            
        except Exception as e:
            rospy.logwarn(f"Erreur lors de la génération de l'image de viz: {e}")
            
        self.perf.tock('viz_overlay')

        # ── DIAGNOSTIC LOG (every 30 frames) ──
        if self.frame_count % 30 == 0:
            t_px = int(np.sum(target_mask)) if target_mask is not None else 0
            c_px = int(np.sum(container_mask)) if container_mask is not None else 0
            available = list(masks.keys())
            rospy.loginfo(f"[MASKS] available={available} target={t_px}px container={c_px}px "
                          f"depth={depth_h}x{depth_w} "
                          f"target_state={self.target_model.get_state()} "
                          f"container_state={self.container_model.get_state()}")
 
        # ── 3. Contact / grasp ──
        # self.perf.tick('grasp_check')
        # skip_target_update = self.is_grasped
        # if not self.is_grasped:
        #     tw = self.target_model.get_points()
        #     if tw is not None and len(tw) > 0:
        #         target_np = tw.cpu().numpy()
        #         dists = np.linalg.norm(target_np - fork_tip, axis=1)
        #         min_dist = dists.min()
        #         closest = target_np[dists.argmin()]
        #         if min_dist < self.CONTACT_DIST:
        #             skip_target_update = True
        #         if fork_tip[2] < closest[2] and min_dist < 0.02:
        #             self.is_grasped = True
        #             skip_target_update = True
        #             T_inv = np.linalg.inv(T_world_fork)
        #             ones = np.ones((len(target_np), 1))
        #             self.grasped_local = (T_inv @ np.hstack([target_np, ones]).T).T[:, :3].astype(np.float32)
        # self.perf.tock('grasp_check')
        self.perf.tick('grasp_check')
        skip_target_update = False
        self.perf.tock('grasp_check')

        # ── 4. 3D Models Update (ICP — multithreaded) ──
        self.perf.tick('models_update')
        z = cv_depth if cv_depth.dtype == np.float32 else cv_depth / 1000.0
        f_fx, f_fy = float(self.fx), float(self.fy)
        f_cx, f_cy = float(self.cx), float(self.cy)
 
        t_mask_for_icp = target_mask if not skip_target_update else None
 
        future_target = self.model_executor.submit(
            self.target_model.update, t_mask_for_icp, z, T_world_cam, f_fx, f_fy, f_cx, f_cy
        )
        future_container = self.model_executor.submit(
            self.container_model.update, container_mask, z, T_world_cam, f_fx, f_fy, f_cx, f_cy
        )
        wait([future_target, future_container])
        try:
            future_target.result()
            future_container.result()
        except Exception as e:
            rospy.logerr(f"ICP Thread Error: {e}")
        self.perf.tock('models_update')
 
        # ── 5. Publish clouds ──
        cw = self.container_model.get_points()
        if cw is not None and len(cw) > 0:
            self._pub_cloud(cw.cpu().numpy(), self.pub_container_cloud)
 
        tw = self.target_model.get_points()
        if tw is not None and len(tw) > 0:
            self._pub_cloud(tw.cpu().numpy(), self.pub_target_cloud)
 
        # ── 6. Publish bboxes ──
        self.perf.tick('pub_bboxes')
        if tw is not None and len(tw) > 0:
            pts_np = tw.cpu().numpy()
            margin = 0.005
            bmin = pts_np.min(axis=0) - margin
            bmax = pts_np.max(axis=0) + margin
            self.pub_target_bbox.publish(Float32MultiArray(data=list(bmin) + list(bmax)))
 
        if cw is not None and len(cw) > 0:
            pts_np = cw.cpu().numpy()
            margin = 0.03
            bmin = pts_np.min(axis=0) - margin
            bmax = pts_np.max(axis=0) + margin
            self.pub_container_bbox.publish(Float32MultiArray(data=list(bmin) + list(bmax)))
        self.perf.tock('pub_bboxes')
 
        # ── 7. Publish centroid ──
        if self.is_grasped:
            c = (T_world_fork @ np.array([0, 0, 0, 1]))[:3]
            self._pub_centroid(c.astype(np.float32))
        else:
            c = self.target_model.get_centroid_np()
            if c is not None:
                self._pub_centroid(c)
 
        # ── 8. Merge + publish condition cloud ──
        self.perf.tick('merge')
        self._publish_merged(robot_pts, T_world_fork)
        self.perf.tock('merge')
 
        self.perf.tock('callback')
 
        # ── 9. Perf ──
        if self.frame_count % 30 == 0:
            stages = ['callback', 'tf_lookup', 'models_update', 'pub_bboxes', 'merge']
            summary = "SEG: " + self.perf.summary(stages)
            # Add tracker/detector info
            t_state = self.target_model.get_state()
            c_state = self.container_model.get_state()
            t_count = self.target_model.count()
            c_count = self.container_model.count()
            summary += f" | target={t_state}({t_count}pts) container={c_state}({c_count}pts)"
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
        elif 0 < len(cloud) < N:
            extra = np.random.choice(len(cloud), N - len(cloud), replace=True)
            cloud = np.vstack([cloud, cloud[extra]])
        self._pub_cloud(cloud, self.pub_merged)
 
    # ==================================================================
    #  UTILITIES
    # ==================================================================
 
    def _decode_depth(self, msg):
        if "32FC1" in msg.encoding:
            return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
        return None
 
    def _lookup_tf(self):
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
    #  RUN
    # ==================================================================
 
    def run(self):
        self.tf_listener = tf.TransformListener()
        rospy.sleep(1.0)
 
        # Start detector thread
        threading.Thread(
            target=detector_thread,
            args=(self.detmodel, self.frame_buf, self.prompts_ref,
                  self.det_queue, self.shutdown_event),
            daemon=True,
        ).start()
 
        # Start tracker thread
        threading.Thread(
            target=tracker_thread,
            args=(self.sam2_model, self.frame_buf, self.det_queue,
                  self.mask_state, self.prompts_ref, self.shutdown_event),
            daemon=True,
        ).start()
 
        rospy.loginfo(f"Tracking prompts: {self.prompts_ref['prompts']}")
        rospy.loginfo("Segmentation node spinning (threaded SAM3 + SAM2.1).")
 
        rospy.on_shutdown(lambda: self.shutdown_event.set())
        rospy.spin()
 
 
if __name__ == '__main__':
    SegmentationNode().run()