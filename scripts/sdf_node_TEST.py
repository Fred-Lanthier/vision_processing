#!/usr/bin/env python3
"""
SDF Node v2 (CPU-only, pure NumPy)
====================================
Environment accumulation + SDF field computation.

Changes from v1:
  - Two-layer grid: STRUCTURAL (table/walls, updates every 2s) + DYNAMIC
    (robot-area obstacles, updates every frame). EDT runs on the merged
    occupancy but the structural layer is cached, making the per-frame
    cost much lower.
  - Depth-adaptive strided unprojection for the full scene
  - Cleaner ROI mask logic
"""

# ── ANTI CPU-THRASHING ──
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
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
from visualization_msgs.msg import Marker, MarkerArray

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
        if points is None or len(points) == 0:
            return
        margin_v = max(1, int(margin / self.voxel_size))
        ijk = np.unique(self._pts_to_ijk(points), axis=0)

        if margin_v <= 1:
            flat = ijk @ self._strides
            self.confidence.ravel()[flat] = 0
            return

        r = np.arange(-margin_v, margin_v + 1)
        di, dj, dk = np.meshgrid(r, r, r, indexing='ij')
        offsets = np.stack([di.ravel(), dj.ravel(), dk.ravel()], axis=1)
        expanded = (ijk[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        expanded[:, 0] = np.clip(expanded[:, 0], 0, self.shape[0] - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, self.shape[1] - 1)
        expanded[:, 2] = np.clip(expanded[:, 2], 0, self.shape[2] - 1)
        flat = expanded @ self._strides
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

    def get_occupancy_bool(self):
        """Return raw bool occupancy grid (for SDF computation)."""
        return self.confidence >= 1

    def clear_bbox(self, bbox_min, bbox_max):
        idx_min = np.maximum(0, ((bbox_min - self.mins) / self.voxel_size).astype(int))
        idx_max = np.minimum(self.shape, ((bbox_max - self.mins) / self.voxel_size).astype(int) + 1)
        self.confidence[idx_min[0]:idx_max[0], idx_min[1]:idx_max[1], idx_min[2]:idx_max[2]] = 0


class TargetBBox:
    def __init__(self):
        self.bbox_min = None
        self.bbox_max = None
        self.locked = False

    def set_bbox(self, bbox_min, bbox_max):
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.locked = True

    def filter_out_points(self, points_np):
        if not self.locked:
            return points_np
        inside = np.all((points_np >= self.bbox_min) & (points_np <= self.bbox_max), axis=1)
        return points_np[~inside]

    def has_bbox(self):
        return self.locked


class ContainerBBox:
    def __init__(self):
        self.bbox_min = None
        self.bbox_max = None
        self.locked = False

    def set_bbox(self, bbox_min, bbox_max):
        self.bbox_min = bbox_min.astype(np.float32)
        self.bbox_max = bbox_max.astype(np.float32)
        self.locked = True

    def filter_points(self, pts):
        if not self.has_bbox():
            return pts
        mask = ((pts[:, 0] >= self.bbox_min[0]) & (pts[:, 0] <= self.bbox_max[0]) &
                (pts[:, 1] >= self.bbox_min[1]) & (pts[:, 1] <= self.bbox_max[1]) &
                (pts[:, 2] >= self.bbox_min[2]) & (pts[:, 2] <= self.bbox_max[2]))
        return pts[mask]

    def has_bbox(self):
        return self.locked


# =====================================================================
#  NUMPY SDF
# =====================================================================
class VoxelSDF_NP:
    def __init__(self, voxel_size, bounds):
        self.voxel_size = voxel_size
        self.mins = np.array([b[0] for b in bounds], dtype=np.float32)
        self.maxs = np.array([b[1] for b in bounds], dtype=np.float32)
        self.grid_shape = np.ceil((self.maxs - self.mins) / voxel_size).astype(int)
        self.sdf_grid = None
        self._last_ms = 0.0

    def update_from_occupancy(self, occupancy_bool):
        """
        Compute SDF from a pre-built bool occupancy grid.
        This avoids re-voxelizing points every frame.
        """
        t0 = time.perf_counter()
        self.sdf_grid = scipy_edt(~occupancy_bool).astype(np.float32) * self.voxel_size
        self._last_ms = (time.perf_counter() - t0) * 1000
        return self._last_ms

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
def cpu_unproject(depth_np, T_np, fx, fy, cx, cy, valid_mask=None, z_min=0.01, z_max=1.5):
    if valid_mask is not None:
        valid = valid_mask & np.isfinite(depth_np)
    else:
        valid = (depth_np > z_min) & (depth_np < z_max) & np.isfinite(depth_np)

    v, u = np.where(valid)
    if len(v) < 10:
        return None

    z = depth_np[v, u]
    pts_cam = np.empty((len(z), 3), dtype=np.float32)
    pts_cam[:, 0] = (u - cx) * z / fx
    pts_cam[:, 1] = (v - cy) * z / fy
    pts_cam[:, 2] = z

    R = T_np[:3, :3].astype(np.float32)
    t = T_np[:3, 3].astype(np.float32)
    return pts_cam @ R.T + t


def cpu_unproject_strided(depth_np, T_np, fx, fy, cx, cy,
                          stride=4, z_min=0.01, z_max=1.5):
    sub = depth_np[::stride, ::stride]
    valid = (sub > z_min) & (sub < z_max) & np.isfinite(sub)
    vs, us = np.where(valid)
    if len(vs) < 10:
        return None

    u_full = (us * stride).astype(np.float32)
    v_full = (vs * stride).astype(np.float32)
    z = sub[valid]

    pts_cam = np.empty((len(z), 3), dtype=np.float32)
    pts_cam[:, 0] = (u_full - cx) * z / fx
    pts_cam[:, 1] = (v_full - cy) * z / fy
    pts_cam[:, 2] = z

    R = T_np[:3, :3].astype(np.float32)
    t = T_np[:3, 3].astype(np.float32)
    return pts_cam @ R.T + t


# =====================================================================
#  SDF NODE
# =====================================================================
class SDFNode:

    # How often to refresh the structural layer (seconds)
    STRUCTURAL_REFRESH_SEC = 2.0

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

        # ── Grids ──
        sdf_bounds = ((-0.2, 0.8), (-0.5, 0.5), (0.0, 0.8))
        self.sdf = VoxelSDF_NP(voxel_size=0.01, bounds=sdf_bounds)

        # Dynamic grid: rebuilt every frame (container area + robot exclusion)
        self.env_grid = EnvironmentGridNP(voxel_size=0.01, bounds=sdf_bounds)

        # Structural grid: cached full-scene geometry, refreshed every 2s
        # This stores table, walls, etc. that don't change frame-to-frame
        self.structural_grid = EnvironmentGridNP(voxel_size=0.01, bounds=sdf_bounds)
        self._last_structural_time = 0.0

        # Container
        self.container = ContainerBBox()
        self._container_cloud = None
        self._container_cloud_lock = threading.Lock()

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

        # ── Publishers ──
        self.pub_obstacle = rospy.Publisher('/vision/obstacle_cloud', PointCloud2, queue_size=1)
        self.pub_sdf_viz = rospy.Publisher('/vision/sdf_viz', PointCloud2, queue_size=1)
        self.pub_sdf_slice = rospy.Publisher('/vision/sdf_slice', PointCloud2, queue_size=1)
        self.pub_perf = rospy.Publisher('/vision/sdf_perf', String, queue_size=1)
        self.pub_markers = rospy.Publisher('/vision/bboxes_viz', MarkerArray, queue_size=1, latch=True)

        # ── Target/Container Bbox ──
        self.sub_bbox = rospy.Subscriber(
            "/vision/container_bbox", Float32MultiArray, self._bbox_cb, queue_size=1
        )
        self.target_bbox = TargetBBox()
        self.sub_target_bbox = rospy.Subscriber(
            '/vision/target_bbox', Float32MultiArray, self._target_bbox_cb, queue_size=1
        )

        # ── Subscribers ──
        self.sub_info = rospy.Subscriber("/camera_wrist/color/camera_info", CameraInfo, self._cam_info_cb)
        self.sub_depth = rospy.Subscriber("/synced/camera_wrist/depth", Image, self._depth_cb, queue_size=1)
        self.sub_joints = rospy.Subscriber("/joint_states", JointState, self._joint_cb, queue_size=1)
        self.sub_container_cloud = rospy.Subscriber(
            '/vision/container_cloud', PointCloud2, self._container_cloud_cb, queue_size=1
        )

        # ── SDF viz defaults ──
        rospy.set_param('/sdf_viz_mode', 'both')
        rospy.set_param('/sdf_slice_axis', 'x')
        rospy.set_param('/sdf_slice_value', 0.42)
        rospy.set_param('/sdf_viz_max_dist', 0.15)

        # ── Start background thread ──
        self._bg_thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._bg_thread.start()

        rospy.loginfo("SDF Node v2 READY (two-layer grid)")

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

    def _target_bbox_cb(self, msg):
        if len(msg.data) == 6:
            self.target_bbox.set_bbox(
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

    def _container_cloud_cb(self, msg):
        pts = np.array(
            list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float32
        )
        if len(pts) > 0:
            with self._container_cloud_lock:
                self._container_cloud = pts

    def _depth_cb(self, depth_msg):
        depth_np = self._decode_depth(depth_msg)
        if depth_np is None:
            return
        T = self._get_T_world_cam()
        if T is None:
            return
        with self._stash_lock:
            self._stash = {'depth_np': depth_np, 'T': T, 'fork': self.fork_cloud}
        self._event.set()

    # ==================================================================
    #  ROI MASK
    # ==================================================================
    def _get_roi_mask(self, depth_np, T_world_cam):
        if not self.container.has_bbox():
            return None

        bmin = self.container.bbox_min
        bmax = self.container.bbox_max
        corners = np.array([
            [bmin[0], bmin[1], bmin[2]], [bmax[0], bmin[1], bmin[2]],
            [bmin[0], bmax[1], bmin[2]], [bmax[0], bmax[1], bmin[2]],
            [bmin[0], bmin[1], bmax[2]], [bmax[0], bmin[1], bmax[2]],
            [bmin[0], bmax[1], bmax[2]], [bmax[0], bmax[1], bmax[2]]
        ])
        T_cam_world = np.linalg.inv(T_world_cam)
        corners_h = np.hstack([corners, np.ones((8, 1))])
        corners_cam = (T_cam_world @ corners_h.T).T[:, :3]

        z_cam = corners_cam[:, 2]
        valid_z = z_cam > 0.01
        if not np.any(valid_z):
            return None

        z_min = max(0.01, z_cam.min() - 0.05)
        z_max = z_cam.max() + 0.05

        u = (corners_cam[valid_z, 0] * self.fx / z_cam[valid_z]) + self.cx
        v = (corners_cam[valid_z, 1] * self.fy / z_cam[valid_z]) + self.cy

        u_min = max(0, int(np.floor(u.min())) - 10)
        u_max = min(depth_np.shape[1], int(np.ceil(u.max())) + 10)
        v_min = max(0, int(np.floor(v.min())) - 10)
        v_max = min(depth_np.shape[0], int(np.ceil(v.max())) + 10)

        mask = np.zeros_like(depth_np, dtype=bool)
        if u_min < u_max and v_min < v_max:
            roi = depth_np[v_min:v_max, u_min:u_max]
            mask[v_min:v_max, u_min:u_max] = (roi >= z_min) & (roi <= z_max)
        return mask

    # ==================================================================
    #  MARKERS
    # ==================================================================
    def _publish_markers(self):
        msg = MarkerArray()
        if self.container.has_bbox():
            m1 = self._create_marker(self.container.bbox_min, self.container.bbox_max,
                                     0, (0.0, 1.0, 0.0, 0.3), "container")
            msg.markers.append(m1)
        if hasattr(self, 'target_bbox') and self.target_bbox.has_bbox():
            m2 = self._create_marker(self.target_bbox.bbox_min, self.target_bbox.bbox_max,
                                     1, (1.0, 0.0, 0.0, 0.3), "target")
            msg.markers.append(m2)
        if len(msg.markers) > 0:
            self.pub_markers.publish(msg)

    def _create_marker(self, bmin, bmax, m_id, color, ns):
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = m_id
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = (bmin[0] + bmax[0]) / 2.0
        m.pose.position.y = (bmin[1] + bmax[1]) / 2.0
        m.pose.position.z = (bmin[2] + bmax[2]) / 2.0
        m.pose.orientation.w = 1.0
        m.scale.x = max(0.001, bmax[0] - bmin[0])
        m.scale.y = max(0.001, bmax[1] - bmin[1])
        m.scale.z = max(0.001, bmax[2] - bmin[2])
        m.color.r, m.color.g, m.color.b, m.color.a = color
        return m

    # ==================================================================
    #  BACKGROUND LOOP (two-layer grid)
    # ==================================================================
    def _bg_loop(self):
        while self._running and not rospy.is_shutdown():
            self._event.wait(timeout=0.2)
            self._event.clear()

            with self._stash_lock:
                stash = self._stash
                self._stash = None
            if stash is None:
                continue

            self.perf.tick('sdf_loop')
            self._frame_count += 1

            try:
                depth_np = stash['depth_np']
                T = stash['T']
                fork = stash['fork']
                z = depth_np if depth_np.dtype == np.float32 else depth_np / 1000.0
                now = time.time()

                # ── 1a. Container ROI (high density, every frame) ──
                self.perf.tick('unproject')
                roi_mask = self._get_roi_mask(z, T)
                container_obs = cpu_unproject(
                    z, T, self.fx, self.fy, self.cx, self.cy, valid_mask=roi_mask
                )

                # ── 1b. Full scene (low density) ──
                full_obs = cpu_unproject_strided(
                    z, T, self.fx, self.fy, self.cx, self.cy,
                    stride=4, z_min=0.01, z_max=3.0
                )
                self.perf.tock('unproject')

                # ── 2. STRUCTURAL LAYER (refresh every N seconds) ──
                # Table, walls, etc. — these barely change so we cache them
                self.perf.tick('structural')
                if (now - self._last_structural_time) > self.STRUCTURAL_REFRESH_SEC:
                    self.structural_grid.confidence[:] = 0
                    if full_obs is not None and len(full_obs) > 0:
                        filtered = full_obs
                        if self.target_bbox.has_bbox():
                            filtered = self.target_bbox.filter_out_points(filtered)
                        if len(filtered) > 0:
                            self.structural_grid.integrate(filtered)
                    # Clear robot from structural layer too
                    if fork is not None and len(fork) > 0:
                        fork_np = fork if isinstance(fork, np.ndarray) else fork.numpy()
                        self.structural_grid.clear_robot(fork_np.astype(np.float32), margin=0.03)
                    self._last_structural_time = now
                self.perf.tock('structural')

                # ── 3. DYNAMIC LAYER (fresh every frame) ──
                self.perf.tick('accumulate')
                self.env_grid.confidence[:] = 0

                # Container-area observations
                if container_obs is not None and len(container_obs) > 0:
                    if self.container.has_bbox():
                        container_obs = self.container.filter_points(container_obs)
                    if self.target_bbox.has_bbox():
                        container_obs = self.target_bbox.filter_out_points(container_obs)
                    if len(container_obs) > 0:
                        self.env_grid.integrate(container_obs)

                # ICP-tracked container reference cloud
                with self._container_cloud_lock:
                    ref_cloud = self._container_cloud
                if ref_cloud is not None and len(ref_cloud) > 0:
                    self.env_grid.integrate(ref_cloud)

                self.perf.tock('accumulate')

                # ── 4. Clear robot from dynamic layer ──
                self.perf.tick('robot_clear')
                if fork is not None and len(fork) > 0:
                    fork_np = fork if isinstance(fork, np.ndarray) else fork.numpy()
                    self.env_grid.clear_robot(fork_np.astype(np.float32), margin=0.02)
                self.perf.tock('robot_clear')

                # ── 5. Clear target bbox from dynamic layer ──
                if self.target_bbox.has_bbox():
                    self.env_grid.clear_bbox(
                        self.target_bbox.bbox_min, self.target_bbox.bbox_max
                    )

                # ── 6. MERGE LAYERS → SDF ──
                # Combine structural (cached) + dynamic (fresh) into one occupancy,
                # then run EDT once. The structural layer is free (already computed),
                # so this is barely more expensive than just the dynamic layer alone.
                self.perf.tick('sdf_update')
                merged_occ = (
                    self.structural_grid.get_occupancy_bool() |
                    self.env_grid.get_occupancy_bool()
                )

                # Clear target from merged occupancy too (belt and suspenders)
                if self.target_bbox.has_bbox():
                    bmin = self.target_bbox.bbox_min
                    bmax = self.target_bbox.bbox_max
                    idx_min = np.maximum(0, ((bmin - self.env_grid.mins) / self.env_grid.voxel_size).astype(int))
                    idx_max = np.minimum(self.env_grid.shape, ((bmax - self.env_grid.mins) / self.env_grid.voxel_size).astype(int) + 1)
                    merged_occ[idx_min[0]:idx_max[0], idx_min[1]:idx_max[1], idx_min[2]:idx_max[2]] = False

                if np.any(merged_occ):
                    self.sdf.update_from_occupancy(merged_occ)
                self.perf.tock('sdf_update')

                # ── 7. Publish to RViz ──
                self.perf.tick('publish')
                viz_pts = self.env_grid.get_occupied_points(min_confidence=1, max_points=20000)
                self._publish_markers()

                if full_obs is not None and len(full_obs) > 0:
                    combined_viz = np.vstack((viz_pts, full_obs)) if viz_pts is not None else full_obs
                    self._pub_cloud(combined_viz, self.pub_obstacle)
                elif viz_pts is not None:
                    self._pub_cloud(viz_pts, self.pub_obstacle)

                if self._frame_count % 3 == 0:
                    viz_mode = rospy.get_param('/sdf_viz_mode', 'both')
                    if viz_mode != 'off':
                        max_dist = rospy.get_param('/sdf_viz_max_dist', 0.15)

                        if viz_mode in ('volume', 'both'):
                            pts, rgb, _ = self.sdf.get_visualization_points(
                                max_dist=max_dist, stride=3
                            )
                            if pts is not None:
                                msg = VoxelSDF_NP.make_rviz_cloud_msg(pts, rgb)
                                if msg:
                                    self.pub_sdf_viz.publish(msg)

                        if viz_mode in ('slice', 'both'):
                            axis = rospy.get_param('/sdf_slice_axis', 'z')
                            val = rospy.get_param('/sdf_slice_value', 0.05)
                            pts, rgb, _ = self.sdf.get_slice_points(
                                axis=axis, value=val, max_dist=max_dist
                            )
                            if pts is not None:
                                msg = VoxelSDF_NP.make_rviz_cloud_msg(pts, rgb)
                                if msg:
                                    self.pub_sdf_slice.publish(msg)
                self.perf.tock('publish')

            except Exception as e:
                rospy.logerr(f"SDF thread error: {e}\n{traceback.format_exc()}")

            self.perf.tock('sdf_loop')

            if self._frame_count % 30 == 0:
                stages = ['sdf_loop', 'unproject', 'structural', 'accumulate',
                          'robot_clear', 'sdf_update', 'publish']
                summary = "SDF: " + self.perf.summary(stages)
                rospy.loginfo(f"\n{summary}")
                self.pub_perf.publish(String(data=summary))

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
        rospy.loginfo("SDF node spinning (two-layer grid, waiting for container bbox).")
        rospy.spin()
        self._running = False
        self._event.set()


if __name__ == '__main__':
    SDFNode().run()
