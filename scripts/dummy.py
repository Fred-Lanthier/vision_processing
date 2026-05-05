#!/usr/bin/env python3
"""
SDF Node (CPU-only, pure NumPy)
================================
Environment accumulation + SDF field computation.
Zero GPU usage — CUTIE gets the full GPU in segmentation_node.

Subscribes:
  /vision/target_mask              (Image, mono8) — from segmentation_node
  /vision/container_bbox           (Float32MultiArray) — [xmin,ymin,zmin,xmax,ymax,zmax]
  /synced/camera_wrist/depth       (Image, 32FC1)
  /camera_wrist/color/camera_info  (CameraInfo)
  /joint_states                    (JointState) — for fork mesh clearing

Publishes:
  /vision/obstacle_cloud   — accumulated environment for RViz
  /vision/sdf_viz          — SDF volume visualization
  /vision/sdf_slice        — SDF slice visualization
  /vision/sdf_perf         — per-stage timing
"""

# ── Block GPU before importing ANYTHING that might touch CUDA ──
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.nice(10)
import rospy
import numpy as np
import threading
import time
import traceback
import cv2
import sys
from collections import deque
from scipy.ndimage import distance_transform_edt as scipy_edt

import rospkg
import message_filters
import tf
import tf.transformations as tft
from sensor_msgs.msg import Image, PointCloud2, PointField, JointState, CameraInfo
from std_msgs.msg import String, Float32MultiArray
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg

rospack = rospkg.RosPack()
sys.path.append(os.path.join(rospack.get_path('vision_processing'), 'scripts'))
from utils import compute_T_child_parent_xacro

try:
    from Compute_3D_point_cloud_from_mesh import RobotMeshLoaderOptimized
    MESH_LOADER_OK = True
except ImportError:
    MESH_LOADER_OK = False


# =====================================================================
#  PERF TRACKER
# =====================================================================

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


# =====================================================================
#  NUMPY ENVIRONMENT GRID
# =====================================================================

class EnvironmentGridNP:
    """
    Pure NumPy confidence grid.
    
    Each voxel stores a confidence counter (int8, max 20).
    clear_robot uses index expansion instead of 3D max_pool.
    """

    def __init__(self, voxel_size, bounds, max_confidence=20):
        self.voxel_size = voxel_size
        self.max_confidence = max_confidence

        self.mins = np.array([b[0] for b in bounds], dtype=np.float32)
        self.maxs = np.array([b[1] for b in bounds], dtype=np.float32)
        self.shape = tuple(int(np.ceil((self.maxs[i] - self.mins[i]) / voxel_size)) for i in range(3))
        D, H, W = self.shape
        self._strides = np.array([H * W, W, 1], dtype=np.int64)

        self.confidence = np.zeros(self.shape, dtype=np.int8)
        rospy.loginfo(f"EnvironmentGridNP: {self.shape}, RAM={self.confidence.nbytes/1e6:.1f}MB")

    def _pts_to_flat(self, points):
        idx = ((points - self.mins) / self.voxel_size).astype(np.int64)
        idx[:, 0] = np.clip(idx[:, 0], 0, self.shape[0] - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, self.shape[1] - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, self.shape[2] - 1)
        return np.unique(idx @ self._strides)

    def _pts_to_ijk(self, points):
        idx = ((points - self.mins) / self.voxel_size).astype(np.int64)
        idx[:, 0] = np.clip(idx[:, 0], 0, self.shape[0] - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, self.shape[1] - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, self.shape[2] - 1)
        return idx

    def integrate(self, points):
        if points is None or len(points) == 0:
            return
        flat = self._pts_to_flat(points)
        c = self.confidence.ravel()
        c[flat] = np.clip(c[flat] + 1, 0, self.max_confidence)

    def clear_points(self, points):
        if points is None or len(points) == 0:
            return
        flat = self._pts_to_flat(points)
        self.confidence.ravel()[flat] = 0

    def clear_robot(self, points, margin=0.02):
        """
        Clear robot voxels + margin by expanding voxel indices.
        For margin=0.02, voxel_size=0.01 → expand ±2 voxels → 5³=125 offsets.
        Much faster than F.max_pool3d on CPU.
        """
        if points is None or len(points) == 0:
            return
        margin_v = max(1, int(margin / self.voxel_size))
        ijk = np.unique(self._pts_to_ijk(points), axis=0)

        if margin_v <= 1:
            flat = ijk @ self._strides
            self.confidence.ravel()[flat] = 0
            return

        # Build offset cube
        r = np.arange(-margin_v, margin_v + 1)
        di, dj, dk = np.meshgrid(r, r, r, indexing='ij')
        offsets = np.stack([di.ravel(), dj.ravel(), dk.ravel()], axis=1)

        # Expand: (N,1,3) + (1,K,3) → (N*K, 3)
        expanded = (ijk[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        expanded[:, 0] = np.clip(expanded[:, 0], 0, self.shape[0] - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, self.shape[1] - 1)
        expanded[:, 2] = np.clip(expanded[:, 2], 0, self.shape[2] - 1)

        flat = np.unique(expanded @ self._strides)
        self.confidence.ravel()[flat] = 0

    def decay(self, amount=1):
        self.confidence = np.clip(self.confidence - amount, 0, self.max_confidence).astype(np.int8)

    def get_occupied_points(self, min_confidence=2, max_points=25000):
        mask = self.confidence >= min_confidence
        idx = np.argwhere(mask)
        if len(idx) == 0:
            return None
        pts = idx.astype(np.float32) * self.voxel_size + self.mins + self.voxel_size / 2
        if len(pts) > max_points:
            pts = pts[np.random.choice(len(pts), max_points, replace=False)]
        return pts


# =====================================================================
#  NUMPY SDF
# =====================================================================

class VoxelSDF_NP:
    """Pure NumPy SDF using scipy EDT."""

    def __init__(self, voxel_size, bounds):
        self.voxel_size = voxel_size
        self.mins = np.array([b[0] for b in bounds], dtype=np.float32)
        self.maxs = np.array([b[1] for b in bounds], dtype=np.float32)
        self.grid_shape = np.ceil((self.maxs - self.mins) / voxel_size).astype(int)
        self.sdf_grid = None
        self._last_ms = 0.0

    def update(self, points_np):
        t0 = time.perf_counter()
        D, H, W = self.grid_shape
        idx = ((points_np - self.mins) / self.voxel_size).astype(int)
        valid = (
            (idx[:, 0] >= 0) & (idx[:, 0] < D) &
            (idx[:, 1] >= 0) & (idx[:, 1] < H) &
            (idx[:, 2] >= 0) & (idx[:, 2] < W)
        )
        idx = idx[valid]
        occ = np.zeros((D, H, W), dtype=bool)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        self.sdf_grid = scipy_edt(~occ).astype(np.float32) * self.voxel_size
        self._last_ms = (time.perf_counter() - t0) * 1000
        return self._last_ms

    def query(self, positions_np):
        if self.sdf_grid is None:
            return np.full(len(positions_np), 999.0, dtype=np.float32)
        idx = ((positions_np - self.mins) / self.voxel_size).astype(int)
        D, H, W = self.sdf_grid.shape
        idx[:, 0] = np.clip(idx[:, 0], 0, D - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, H - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, W - 1)
        return self.sdf_grid[idx[:, 0], idx[:, 1], idx[:, 2]]

    # ── RViz helpers ──

    @staticmethod
    def _sdf_colormap(vals, max_dist):
        rgb = np.zeros((len(vals), 3), dtype=np.uint8)
        on_surface = vals < 1e-4
        rgb[on_surface] = [255, 255, 255]
        barrier = ~on_surface
        if np.any(barrier):
            t = np.clip(vals[barrier] / max_dist, 0, 1)
            rgb[barrier, 0] = ((1.0 - t) * 255).astype(np.uint8)
            rgb[barrier, 1] = (t * 255).astype(np.uint8)
        return rgb

    def get_visualization_points(self, max_dist=0.20, stride=2):
        if self.sdf_grid is None:
            return None, None, None
        sub = self.sdf_grid[::stride, ::stride, ::stride]
        mask = sub <= max_dist
        if not np.any(mask):
            return None, None, None
        pts = np.argwhere(mask).astype(np.float32) * stride * self.voxel_size + self.mins
        vals = sub[mask]
        return pts, self._sdf_colormap(vals, max_dist), vals

    def get_slice_points(self, axis="z", value=0.1, max_dist=0.20):
        if self.sdf_grid is None:
            return None, None, None
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        vi = int(round((value - self.mins[axis_idx]) / self.voxel_size))
        vi = max(0, min(vi, self.sdf_grid.shape[axis_idx] - 1))
        if axis_idx == 0:
            sl = self.sdf_grid[vi, :, :]
            other = (1, 2)
        elif axis_idx == 1:
            sl = self.sdf_grid[:, vi, :]
            other = (0, 2)
        else:
            sl = self.sdf_grid[:, :, vi]
            other = (0, 1)
        mask = sl <= max_dist
        if not np.any(mask):
            return None, None, None
        idx2d = np.argwhere(mask)
        M = len(idx2d)
        pts = np.zeros((M, 3), dtype=np.float32)
        pts[:, axis_idx] = value
        pts[:, other[0]] = idx2d[:, 0] * self.voxel_size + self.mins[other[0]]
        pts[:, other[1]] = idx2d[:, 1] * self.voxel_size + self.mins[other[1]]
        return pts, self._sdf_colormap(sl[mask], max_dist), sl[mask]

    @staticmethod
    def make_rviz_cloud_msg(points_xyz, points_rgb, frame_id="world", stamp=None):
        import rospy
        N = len(points_xyz)
        if N == 0:
            return None
        r = points_rgb[:, 0].astype(np.uint32)
        g = points_rgb[:, 1].astype(np.uint32)
        b = points_rgb[:, 2].astype(np.uint32)
        rgb_packed = ((r << 16) | (g << 8) | b).view(np.float32)
        cloud_data = np.zeros(N, dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)
        ])
        cloud_data['x'] = points_xyz[:, 0]
        cloud_data['y'] = points_xyz[:, 1]
        cloud_data['z'] = points_xyz[:, 2]
        cloud_data['rgb'] = rgb_packed
        msg = PointCloud2()
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp or rospy.Time.now()
        msg.height = 1
        msg.width = N
        msg.fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * N
        msg.data = cloud_data.tobytes()
        msg.is_dense = True
        return msg


# =====================================================================
#  CPU UNPROJECT
# =====================================================================

def cpu_unproject(depth_np, mask_np, T_np, fx, fy, cx, cy, z_min=0.01, z_max=1.5):
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
    return (T_np @ pts_cam.T).T[:, :3].astype(np.float32)


# =====================================================================
#  CONTAINER BBOX (received via topic)
# =====================================================================

class ContainerBBox:
    def __init__(self):
        self.bbox_min = None
        self.bbox_max = None
        self.locked = False

    def set_bbox(self, bbox_min, bbox_max):
        self.bbox_min = bbox_min.astype(np.float32)
        self.bbox_max = bbox_max.astype(np.float32)
        self.locked = True
        rospy.loginfo(f"[Container] Received bbox: {self.bbox_min} -> {self.bbox_max}")

    def filter_points(self, points_np):
        if not self.locked:
            return points_np
        inside = np.all((points_np >= self.bbox_min) & (points_np <= self.bbox_max), axis=1)
        return points_np[inside]

    def has_bbox(self):
        return self.locked


# =====================================================================
#  SDF NODE
# =====================================================================

class SDFNode:

    def __init__(self):
        rospy.init_node('sdf_node', anonymous=True)
        self.perf = PerfTracker(window=50)

        # ── Robot geometry ──
        pkg = rospack.get_path('vision_processing')
        xacro = os.path.join(pkg, 'urdf', 'panda_camera.xacro')
        self.T_tcp_cam = (
            compute_T_child_parent_xacro(xacro, "camera_wrist_link", "panda_TCP") @
            compute_T_child_parent_xacro(xacro, "camera_wrist_optical_frame", "camera_wrist_link")
        )

        self.mesh_loader = None
        if MESH_LOADER_OK:
            try:
                self.mesh_loader = RobotMeshLoaderOptimized(xacro)
            except Exception as e:
                rospy.logwarn(f"Mesh loader fail: {e}")

        self.fork_cloud = None
        self.last_joint_hash = None

        # ── Container bbox (from segmentation node) ──
        self.container = ContainerBBox()

        # ── SDF + Grid (pure numpy) ──
        sdf_bounds = ((-0.2, 0.8), (-0.5, 0.5), (0.0, 0.8))
        self.sdf = VoxelSDF_NP(voxel_size=0.02, bounds=sdf_bounds)
        self.env_grid = EnvironmentGridNP(voxel_size=0.02, bounds=sdf_bounds)
        self._decay_counter = 0

        # ── Camera intrinsics ──
        self.fx = 604.9
        self.fy = 604.9
        self.cx = 320.0
        self.cy = 240.0

        # ── Background thread ──
        self._stash_lock = threading.Lock()
        self._stash = None
        self._event = threading.Event()
        self._running = True
        self._frame_count = 0

        # ── Subscribers ──
        self.sub_info = rospy.Subscriber(
            "/camera_wrist/color/camera_info", CameraInfo, self._cam_info_cb
        )
        self.sub_bbox = rospy.Subscriber(
            "/vision/container_bbox", Float32MultiArray, self._bbox_cb, queue_size=1
        )
        sub_mask = message_filters.Subscriber("/vision/target_mask", Image)
        sub_dep = message_filters.Subscriber("/synced/camera_wrist/depth", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_mask, sub_dep], queue_size=20, slop=0.2
        )
        self.ts.registerCallback(self._mask_depth_cb)

        self.sub_depth_only = rospy.Subscriber(
            "/synced/camera_wrist/depth", Image, self._depth_only_cb, queue_size=1
        )
        self._last_mask_time = rospy.Time(0)

        self.sub_joints = rospy.Subscriber(
            "/joint_states", JointState, self._joint_cb, queue_size=1
        )

        # ── Start background thread ──
        self._bg_thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._bg_thread.start()

        rospy.loginfo("SDF Node READY (CPU-only, no GPU)")

    # ==================================================================
    #  CALLBACKS
    # ==================================================================

    def _cam_info_cb(self, msg):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        self.sub_info.unregister()

    def _bbox_cb(self, msg):
        if len(msg.data) == 6:
            self.container.set_bbox(
                np.array(msg.data[:3], dtype=np.float32),
                np.array(msg.data[3:], dtype=np.float32)
            )

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

    def _mask_depth_cb(self, mask_msg, depth_msg):
        self._last_mask_time = mask_msg.header.stamp
        depth_np = self._decode_depth(depth_msg)
        if depth_np is None:
            return
        mask_np = np.frombuffer(mask_msg.data, dtype=np.uint8).reshape(
            mask_msg.height, mask_msg.width) > 0
        T = self._get_T_world_cam()
        if T is None:
            return
        self._push_stash(mask_np, depth_np, T)

    def _depth_only_cb(self, depth_msg):
        if (depth_msg.header.stamp - self._last_mask_time).to_sec() < 0.2:
            return
        depth_np = self._decode_depth(depth_msg)
        if depth_np is None:
            return
        T = self._get_T_world_cam()
        if T is None:
            return
        self._push_stash(None, depth_np, T)

    def _push_stash(self, mask, depth_np, T):
        self.perf.tick('unproject')
        z = depth_np if depth_np.dtype == np.float32 else depth_np / 1000.0

        if mask is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
            target_pts = cpu_unproject(z, dilated, T, self.fx, self.fy, self.cx, self.cy)
            obs_pts = cpu_unproject(z, ~dilated, T, self.fx, self.fy, self.cx, self.cy)
        else:
            target_pts = None
            obs_pts = cpu_unproject(z, np.ones(z.shape, dtype=bool), T,
                                    self.fx, self.fy, self.cx, self.cy)
        self.perf.tock('unproject')

        if obs_pts is None or len(obs_pts) < 50:
            return
        with self._stash_lock:
            self._stash = {'obs_pts': obs_pts, 'target_pts': target_pts, 'fork': self.fork_cloud}
        self._event.set()

    # ==================================================================
    #  UTILITIES
    # ==================================================================

    def _decode_depth(self, msg):
        if "32FC1" in msg.encoding:
            return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
        return None

    def _get_T_world_cam(self):
        try:
            t, r = self.tf_listener.lookupTransform("world", "panda_hand_tcp", rospy.Time(0))
        except tf.Exception:
            return None
        T = tft.quaternion_matrix(r)
        T[:3, 3] = t
        return T @ self.T_tcp_cam

    def _pub_cloud(self, points, publisher, frame="world"):
        if points is None or len(points) == 0:
            return
        hdr = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id=frame)
        publisher.publish(pc2.create_cloud_xyz32(hdr, points.astype(np.float32)))

    def run(self):
        self.tf_listener = tf.TransformListener()
        rospy.sleep(1.0)
        rospy.loginfo("SDF node spinning (waiting for container bbox from segmentation node).")
        rospy.spin()
        self._running = False
        self._event.set()


if __name__ == '__main__':
    SDFNode().run()