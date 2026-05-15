#!/usr/bin/env python3
"""
Persistent obstacle map — sparse WorldCloud + GPU per-voxel freespace check.

Voxel lifecycle:
  - Camera sees background beyond a voxel  → evicted immediately (one depth frame)
  - Camera occluded (robot arm in front)   → voxel unchanged, persists
  - Voxel outside camera FOV              → voxel unchanged, persists indefinitely
  - Target voxels (SAM2)                  → separate _tgt_world, never freespace-evicted

No age-based eviction. No free-count counters. Memory bounded by max_points.

Two maps:
  obs_world  — WorldCloud at user-defined voxel_size (5 mm default).
  _tgt_world — WorldCloud for the target food item.
"""
import os
import sys

import rospy
import numpy as np
import torch
import tf2_ros
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo, Image
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge

sys.path.insert(0, os.path.dirname(__file__))
from PersistentCloud import WorldCloud


# ── Shared helper ─────────────────────────────────────────────────────────────

def _make_cloud_msg(header, pts_np):
    """Zero-copy PointCloud2 from numpy [N, 3] float32."""
    pts = np.ascontiguousarray(pts_np, dtype=np.float32)
    msg = PointCloud2()
    msg.header       = header
    msg.height       = 1
    msg.width        = len(pts)
    msg.is_bigendian = False
    msg.point_step   = 12
    msg.row_step     = 12 * len(pts)
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]
    msg.data     = pts.tobytes()
    msg.is_dense = True
    return msg


# ── GPU per-voxel freespace check ─────────────────────────────────────────────

@torch.no_grad()
def _gpu_freespace_check(pts_world, depth_img, R_cw, t_cw,
                         fx, fy, cx, cy, free_margin, device):
    """
    Project committed voxels into the depth image.
    Returns [N] bool — True = camera sees background beyond this voxel → evict.

    Occluded voxels (robot arm closer than the voxel) and out-of-FOV voxels
    both return False so they are left untouched.
    """
    if len(pts_world) == 0:
        return np.zeros(0, dtype=bool)

    pts = torch.from_numpy(pts_world.astype(np.float32)).to(device)
    R   = torch.from_numpy(R_cw.astype(np.float32)).to(device)
    t   = torch.from_numpy(t_cw.astype(np.float32)).to(device)

    pts_c = (R @ pts.T).T + t          # [N, 3] camera frame
    z     = pts_c[:, 2]

    eps = z.clamp(min=0.01)
    u   = pts_c[:, 0] * fx / eps + cx
    v   = pts_c[:, 1] * fy / eps + cy

    H, W = depth_img.shape
    ui   = u.round().long()
    vi   = v.round().long()
    in_fov = (z > 0.01) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

    d_meas = torch.zeros(len(pts), dtype=torch.float32, device=device)
    if in_fov.any():
        depth_t = torch.from_numpy(depth_img.astype(np.float32)).to(device)
        ui_c    = ui[in_fov].clamp(0, W - 1)
        vi_c    = vi[in_fov].clamp(0, H - 1)
        d_meas[in_fov] = depth_t[vi_c, ui_c]

    has_depth = in_fov & (d_meas > 0.001) & (d_meas < 4.5)

    # Evict only when camera clearly sees background beyond the voxel.
    # Occluded (d_meas < z) and out-of-FOV both return False — preserved.
    is_free = has_depth & (d_meas > z + free_margin)
    return is_free.cpu().numpy()


# ── ROS node ──────────────────────────────────────────────────────────────────

class PersistentCloudNode:
    def __init__(self):
        rospy.init_node('persistent_cloud_node')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            rospy.logwarn("CUDA unavailable — freespace check runs on CPU (slower)")
        self.device = device

        # ── Obstacle map parameters ───────────────────────────────────────────
        obs_voxel_size  = float(rospy.get_param("~voxel_size",     0.005))
        obs_free_margin = float(rospy.get_param("~free_margin",    obs_voxel_size * 2))
        obs_max_voxels  = int(rospy.get_param("~max_voxels",       50000))
        obs_commit_thresh = int(rospy.get_param("~commit_threshold", 2))

        # ── Target map parameters ─────────────────────────────────────────────
        tgt_voxel_size    = float(rospy.get_param("~target_voxel_size",       0.005))
        tgt_commit_thresh = int(rospy.get_param("~target_commit_threshold",   2))
        tgt_max_voxels    = int(rospy.get_param("~target_max_voxels",         20000))

        # ── Shared ────────────────────────────────────────────────────────────
        self.obs_voxel_size    = obs_voxel_size
        self.tgt_voxel_size    = tgt_voxel_size
        self.obs_free_margin   = obs_free_margin
        self.obs_commit_thresh = obs_commit_thresh
        self.tgt_commit_thresh = tgt_commit_thresh
        self.obs_max_pts       = obs_max_voxels
        self.camera_frame      = rospy.get_param("~camera_frame", "camera_wrist_optical_frame")
        self.world_frame       = rospy.get_param("~world_frame",  "world")

        # ── Robot self-filter ─────────────────────────────────────────────────
        # Removes depth-projected points on robot body links before integration.
        # Excludes panda_hand/TCP so real obstacles near the fork are kept.
        self._body_link_frames = [
            'panda_link0', 'panda_link1', 'panda_link2', 'panda_link3',
            'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7'
        ]
        self.robot_body_radius = float(rospy.get_param("~robot_body_radius", 0.07))
        self._body_link_pos    = None   # [K, 3], updated each depth frame

        # ── Obstacle accumulator ──────────────────────────────────────────────
        self.obs_world = WorldCloud(voxel_size=obs_voxel_size, max_points=obs_max_voxels)

        # ── Target accumulator ────────────────────────────────────────────────
        self._tgt_world = WorldCloud(voxel_size=tgt_voxel_size, max_points=tgt_max_voxels)

        # Live fallback clouds (used before any voxels are committed)
        self._live_obs    = np.empty((0, 3), dtype=np.float32)
        self._live_target = np.empty((0, 3), dtype=np.float32)

        # ── Camera state ──────────────────────────────────────────────────────
        self.fx = self.fy = self.cx = self.cy = None
        self.depth_cv    = None
        self.depth_stamp = None
        self.bridge      = CvBridge()

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ── ROS wiring ────────────────────────────────────────────────────────
        depth_topic  = rospy.get_param("~depth_topic",  "/camera_wrist/aligned_depth_to_color/image_raw")
        info_topic   = rospy.get_param("~info_topic",   "/camera_wrist/color/camera_info")
        cloud_topic  = rospy.get_param("~cloud_topic",  "/perception/cleaned_obstacles")
        target_topic = rospy.get_param("~target_topic", "/perception/target")

        rospy.Subscriber(info_topic,   CameraInfo,  self._info_cb,   queue_size=1)
        rospy.Subscriber(depth_topic,  Image,       self._depth_cb,  queue_size=1)
        rospy.Subscriber(cloud_topic,  PointCloud2, self._obs_cb,    queue_size=1)
        rospy.Subscriber(target_topic, PointCloud2, self._target_cb, queue_size=1)

        self.pub        = rospy.Publisher('/perception/persistent_obstacles', PointCloud2, queue_size=1)
        self.target_pub = rospy.Publisher('/perception/persistent_target',    PointCloud2, queue_size=1)

        rospy.loginfo(
            "Persistent Cloud Node ready | "
            "obs: %.1fmm voxel, commit=%d, free_margin=%.1fmm | "
            "self-filter radius=%.0fcm | device=%s",
            obs_voxel_size * 1000, obs_commit_thresh, obs_free_margin * 1000,
            self.robot_body_radius * 100, device)

    # ── Robot link position helper ────────────────────────────────────────────

    def _get_link_positions(self, frames):
        """Return [K, 3] world-frame positions for the given TF frames, or None."""
        positions = []
        for frame in frames:
            try:
                tr = self.tf_buffer.lookup_transform(
                    self.world_frame, frame, rospy.Time(0), rospy.Duration(0.05))
                t = tr.transform.translation
                positions.append([t.x, t.y, t.z])
            except Exception:
                pass
        return np.array(positions, dtype=np.float32) if positions else None

    # ── Camera callbacks ──────────────────────────────────────────────────────

    def _info_cb(self, msg):
        self.fx, self.cx = msg.K[0], msg.K[2]
        self.fy, self.cy = msg.K[4], msg.K[5]

    def _depth_cb(self, msg):
        """Decode depth, update robot positions, run GPU freespace eviction."""
        if self.fx is None:
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32)
            if msg.encoding != "32FC1":
                img /= 1000.0
        except Exception as e:
            rospy.logwarn_throttle(5, f"Depth decode error: {e}")
            return

        R_cw, t_cw = self._get_world_to_cam(msg.header.stamp)
        if R_cw is None:
            return

        # Refresh cached robot link positions (also used by _obs_cb self-filter)
        self._body_link_pos = self._get_link_positions(self._body_link_frames)

        # Freespace eviction: immediately remove voxels where camera sees background
        pts, hashes = self.obs_world.get_points_and_hashes(min_confidence=1)
        if pts is not None:
            is_free = _gpu_freespace_check(
                pts, img, R_cw, t_cw,
                self.fx, self.fy, self.cx, self.cy,
                self.obs_free_margin, self.device)
            if is_free.any():
                self.obs_world.evict_hashes(np.sort(hashes[is_free]))

    # ── Cloud callbacks ───────────────────────────────────────────────────────

    def _obs_cb(self, msg):
        try:
            raw = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, int(msg.point_step / 4))
            pts = raw[:, :3].copy()
        except Exception as e:
            rospy.logerr_throttle(5, f"Obstacle cloud parse error: {e}")
            return

        # Self-filter: remove depth-projected points on robot body links
        if len(pts) > 0 and self._body_link_pos is not None:
            dists = np.linalg.norm(
                pts[:, None, :] - self._body_link_pos[None, :, :], axis=-1)  # [N, K]
            pts = pts[dists.min(axis=1) > self.robot_body_radius]

        self._live_obs = pts
        if len(pts) > 0:
            self.obs_world.integrate(pts)
        self._publish(msg.header.stamp)

    def _target_cb(self, msg):
        try:
            raw = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, int(msg.point_step / 4))
            pts = raw[:, :3].copy()
        except Exception as e:
            rospy.logwarn_throttle(5, f"Target cloud parse error: {e}")
            return

        self._live_target = pts
        if len(pts) > 0:
            self._tgt_world.integrate(pts)
        self._publish_target(msg.header.stamp)

    # ── TF helper ─────────────────────────────────────────────────────────────

    def _get_world_to_cam(self, stamp):
        """Returns (R_cw [3,3], t_cw [3]) — world→camera — or (None, None)."""
        try:
            tr = self.tf_buffer.lookup_transform(
                self.camera_frame, self.world_frame, stamp, rospy.Duration(0.1))
            t  = tr.transform.translation
            r  = tr.transform.rotation
            R  = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix().astype(np.float32)
            tv = np.array([t.x, t.y, t.z], dtype=np.float32)
            return R, tv
        except Exception as e:
            rospy.logwarn_throttle(5, f"TF {self.camera_frame}←{self.world_frame} failed: {e}")
            return None, None

    # ── Publishing ────────────────────────────────────────────────────────────

    def _publish(self, stamp):
        header = Header(stamp=stamp, frame_id=self.world_frame)

        tgt_pts = self._tgt_world.get_points(min_confidence=self.tgt_commit_thresh)
        if tgt_pts is None:
            tgt_pts = self._live_target

        obs_pts = self.obs_world.get_points(min_confidence=self.obs_commit_thresh)
        if obs_pts is None:
            obs_pts = self._live_obs

        # Carve target region out of obstacle cloud
        if obs_pts is not None and len(obs_pts) > 0 \
                and tgt_pts is not None and len(tgt_pts) > 0:
            excl  = self.obs_voxel_size + self.tgt_voxel_size
            dists = np.linalg.norm(
                obs_pts[:, None, :] - tgt_pts[None, :, :], axis=-1)
            obs_pts = obs_pts[dists.min(axis=1) >= excl]

        if obs_pts is not None and len(obs_pts) > self.obs_max_pts:
            obs_pts = obs_pts[:self.obs_max_pts]

        if obs_pts is not None and len(obs_pts) > 0:
            self.pub.publish(_make_cloud_msg(header, obs_pts))

        rospy.loginfo_throttle(5.0,
            "Obs committed: %d | Target committed: %d",
            self.obs_world.count(), self._tgt_world.count())

    def _publish_target(self, stamp):
        header  = Header(stamp=stamp, frame_id=self.world_frame)
        tgt_pts = self._tgt_world.get_points(min_confidence=self.tgt_commit_thresh)
        if tgt_pts is None:
            tgt_pts = self._live_target
        if tgt_pts is not None and len(tgt_pts) > 0:
            self.target_pub.publish(_make_cloud_msg(header, tgt_pts))

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    PersistentCloudNode().run()
