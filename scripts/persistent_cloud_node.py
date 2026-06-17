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
from PersistentCloud import WorldCloud, _to_hashes


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
        self.log_counts        = bool(rospy.get_param("~log_counts", True))
        self.freespace_evict_enabled = bool(rospy.get_param(
            "~freespace_evict_enabled", True))
        self.publish_evicted_debug = bool(rospy.get_param(
            "~publish_evicted_debug", False))
        self.log_freespace_evictions = bool(rospy.get_param(
            "~log_freespace_evictions", False))

        # ── Robot self-filter ─────────────────────────────────────────────────
        # Removes depth-projected points on robot body links before integration.
        # Excludes panda_hand/TCP so real obstacles near the fork are kept.
        self._body_link_frames = [
            'panda_link0', 'panda_link1', 'panda_link2', 'panda_link3',
            'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7'
        ]
        # Reduced from 0.07 — cleaned_obstacles already has accurate SDF self-filter;
        # this sphere catch-all at 3 cm only removes residual self-projections on the
        # arm links, without erasing bowl-rim points near link7 during close approach.
        self.robot_body_radius = float(rospy.get_param("~robot_body_radius", 0.03))
        # Publish dilation: add 6 face-adjacent voxel-center offsets at publish time.
        # Fills gaps between observed surface samples without bloating stored state.
        self.dilate_published_cloud = bool(rospy.get_param("~dilate_published_cloud", True))
        # Target exclusion sphere: any obs_world point within this radius of the
        # target centroid is treated as target, not obstacle.  Applied at:
        #   1. integration time (never stored in obs_world)
        #   2. target-update time (evict already-stored contaminated voxels)
        #   3. publish time (O(N) centroid-sphere guard, replaces O(N×M) pairwise)
        # Minimum exclusion sphere radius (hard floor — never go below this).
        # The actual radius used at runtime is the 95th-percentile distance from
        # the target centroid to its own point cloud, plus one voxel of margin.
        # This adapts automatically to the target's physical size so neither the
        # cube surface nor the close bowl rim are treated as obstacles.
        self.target_exclusion_radius_min = float(rospy.get_param("~target_exclusion_radius", 0.005))
        self._tgt_centroid = None   # latest target centroid in world frame
        # Carve obstacle points coincident with the target by point-to-point
        # proximity to the actual target cloud, NOT a centroid sphere. A sphere
        # large enough to cover the target also deletes nearby obstacles (e.g.
        # the walls of a slim container the target sits in). Only points within
        # this margin of a real target point are removed.
        self._tgt_carve_margin = max(
            self.target_exclusion_radius_min, obs_voxel_size // 2)

        # Synthetic floor under the target: fills the depth-occluded region
        # below the food so the CBF cannot drive the fork too deep.
        # Height is inferred from the low target Z percentile when possible,
        # with the obstacle-ring estimate kept as a fallback. Points are added
        # only to the published cloud — obs_world is never modified.
        self.generate_synthetic_floor  = bool(rospy.get_param("~generate_synthetic_floor", True))
        self.floor_sample_radius       = float(rospy.get_param("~floor_sample_radius",    0.12))
        self.floor_grid_radius         = float(rospy.get_param("~floor_grid_radius",      0.05))
        self.floor_grid_resolution     = float(rospy.get_param("~floor_grid_resolution",  obs_voxel_size))
        self.floor_height_percentile   = float(rospy.get_param("~floor_height_percentile", 15.0))
        self.floor_footprint_mode      = str(rospy.get_param("~floor_footprint_mode", "target")).lower()
        self.floor_footprint_margin    = float(rospy.get_param("~floor_footprint_margin", 0.0))
        self.floor_height_filter_tau   = max(
            0.0, float(rospy.get_param("~floor_height_filter_tau", 0.0)))
        self.floor_height_max_step     = max(
            0.0, float(rospy.get_param("~floor_height_max_step", 0.0)))
        self._floor_z_filtered         = None
        self._floor_z_stamp            = None
        if self.floor_footprint_mode not in ("target", "circle"):
            rospy.logwarn(
                "Unknown floor_footprint_mode=%s, falling back to target",
                self.floor_footprint_mode)
            self.floor_footprint_mode = "target"

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
        self.evicted_pub = rospy.Publisher(
            '/perception/persistent_evicted_freespace', PointCloud2, queue_size=1)

        # Publish the current map state at a fixed rate, independent of the
        # obstacle-cloud input rate (which is capped at the SAM2 tracking rate,
        # ~5 Hz, and can stall for seconds on detection failures).
        pub_rate_hz = float(rospy.get_param("~publish_rate_hz", 30.0))
        self._last_pub_stamp = None
        rospy.Timer(rospy.Duration(1.0 / pub_rate_hz), self._timer_pub_cb)

        rospy.loginfo(
            "Persistent Cloud Node ready | "
            "obs: %.1fmm voxel, commit=%d, free_margin=%.1fmm | "
            "freespace_evict=%s | self-filter radius=%.0fcm | device=%s",
            obs_voxel_size * 1000, obs_commit_thresh, obs_free_margin * 1000,
            self.freespace_evict_enabled, self.robot_body_radius * 100, device)

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

        if not self.freespace_evict_enabled:
            return

        # Freespace eviction: immediately remove voxels where camera sees background
        pts, hashes = self.obs_world.get_points_and_hashes(min_confidence=1)
        if pts is not None:
            is_free = _gpu_freespace_check(
                pts, img, R_cw, t_cw,
                self.fx, self.fy, self.cx, self.cy,
                self.obs_free_margin, self.device)
            if is_free.any():
                evicted_pts = pts[is_free]
                self.obs_world.evict_hashes(np.sort(hashes[is_free]))
                if self.publish_evicted_debug:
                    header = Header(stamp=msg.header.stamp, frame_id=self.world_frame)
                    self.evicted_pub.publish(_make_cloud_msg(header, evicted_pts))
                if self.log_freespace_evictions:
                    rospy.loginfo_throttle(
                        1.0,
                        "Persistent freespace eviction: evicted=%d / checked=%d",
                        int(is_free.sum()), int(len(is_free)))

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

        # Integration-time target exclusion: never store target-region points in
        # obs_world.  The freespace eviction cannot remove them once committed
        # because the target itself keeps the depth value occupied.
        if len(pts) > 0:
            pts = self._carve_against_target(pts)

        self._live_obs = pts
        if len(pts) > 0:
            self.obs_world.integrate(pts)
        self._last_pub_stamp = msg.header.stamp

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
            new_centroid = pts.mean(axis=0).astype(np.float32)

            # Evict obs_world voxels coincident with the target cloud.  Necessary
            # on first target acquisition and whenever the target shifts.
            if self._tgt_centroid is None or \
                    np.linalg.norm(new_centroid - self._tgt_centroid) > self.obs_voxel_size:
                self._evict_target_region(pts)
            self._tgt_centroid = new_centroid
        self._last_pub_stamp = msg.header.stamp

    def _carve_against_target(self, obs_pts):
        """Remove obstacle points coincident with the actual target cloud.

        Point-to-point margin against the target points (see _tgt_carve_margin),
        not a sphere around the centroid, so obstacles near the target (e.g. a
        slim container's walls) are preserved.
        """
        if obs_pts is None or len(obs_pts) == 0:
            return obs_pts
        tgt = self._tgt_world.get_points(min_confidence=1)
        if tgt is None or len(tgt) == 0:
            tgt = self._live_target
        if tgt is None or len(tgt) == 0:
            return obs_pts
        try:
            from scipy.spatial import cKDTree
            d, _ = cKDTree(tgt).query(obs_pts, k=1)
            return obs_pts[d >= self._tgt_carve_margin]
        except Exception:
            keep = np.ones(len(obs_pts), dtype=bool)
            for p in tgt:
                keep &= np.linalg.norm(obs_pts - p, axis=1) >= self._tgt_carve_margin
            return obs_pts[keep]

    def _evict_target_region(self, tgt_pts):
        """Evict obs_world voxels coincident with the target cloud (point-to-point
        margin, not a centroid sphere)."""
        pts, hashes = self.obs_world.get_points_and_hashes(min_confidence=1)
        if pts is None or tgt_pts is None or len(tgt_pts) == 0:
            return
        try:
            from scipy.spatial import cKDTree
            d, _ = cKDTree(tgt_pts).query(pts, k=1)
            contaminated = hashes[d < self._tgt_carve_margin]
        except Exception:
            mask = np.zeros(len(pts), dtype=bool)
            for p in tgt_pts:
                mask |= np.linalg.norm(pts - p, axis=1) < self._tgt_carve_margin
            contaminated = hashes[mask]
        if len(contaminated) > 0:
            self.obs_world.evict_hashes(np.sort(contaminated))

    def _timer_pub_cb(self, event):
        stamp = self._last_pub_stamp if self._last_pub_stamp is not None else rospy.Time.now()
        self._publish(stamp)
        self._publish_target(stamp)

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

    # ── Synthetic floor ───────────────────────────────────────────────────────

    @staticmethod
    def _points_in_polygon(points_xy, poly_xy):
        """Vectorized ray-casting test for a convex or non-convex 2D polygon."""
        if points_xy is None or len(points_xy) == 0 or poly_xy is None or len(poly_xy) < 3:
            return np.zeros(0, dtype=bool)
        x = points_xy[:, 0]
        y = points_xy[:, 1]
        inside = np.zeros(len(points_xy), dtype=bool)
        xj, yj = poly_xy[-1]
        eps = 1e-12
        for xi, yi in poly_xy:
            crosses = ((yi > y) != (yj > y))
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + eps) + xi
            inside ^= crosses & (x < x_intersect)
            xj, yj = xi, yi
        return inside

    def _generate_target_footprint_patch(self, tgt_pts, floor_z):
        """Generate floor points only below the target's projected XY support."""
        if tgt_pts is None or len(tgt_pts) < 3:
            return None

        cx, cy, _ = self._tgt_centroid
        xy = tgt_pts[:, :2].astype(np.float32)

        # Keep the support bounded by the same radius used by the old disk mode.
        # This prevents a single target outlier from creating a large synthetic
        # obstacle while still allowing the footprint mode to replace the circle.
        if self.floor_grid_radius > 0.0:
            d2 = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
            bounded = xy[d2 <= self.floor_grid_radius ** 2]
            if len(bounded) >= 3:
                xy = bounded

        res = max(self.floor_grid_resolution, 1e-4)
        margin = max(self.floor_footprint_margin, 0.0)
        xy_min = xy.min(axis=0) - margin
        xy_max = xy.max(axis=0) + margin

        xs = np.arange(xy_min[0], xy_max[0] + res * 0.5, res, dtype=np.float32)
        ys = np.arange(xy_min[1], xy_max[1] + res * 0.5, res, dtype=np.float32)
        if len(xs) == 0 or len(ys) == 0:
            return None

        xx, yy = np.meshgrid(xs, ys)
        grid_xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

        try:
            from scipy.spatial import ConvexHull
            if np.linalg.matrix_rank(xy - xy.mean(axis=0, keepdims=True)) < 2:
                raise ValueError("Target footprint is nearly collinear.")
            hull = ConvexHull(xy)
            poly = xy[hull.vertices]
            keep = self._points_in_polygon(grid_xy, poly)
        except Exception:
            # Degenerate footprint: fall back to occupied target XY voxels.
            idx = np.floor(grid_xy / res).astype(np.int64)
            tgt_idx = np.floor(xy / res).astype(np.int64)
            keep = np.isin(
                idx[:, 0].astype(np.int64) * 73856093
                ^ idx[:, 1].astype(np.int64) * 19349663,
                tgt_idx[:, 0].astype(np.int64) * 73856093
                ^ tgt_idx[:, 1].astype(np.int64) * 19349663)

        if margin > 0.0:
            try:
                from scipy.spatial import cKDTree
                nearest, _ = cKDTree(xy).query(grid_xy, k=1)
                keep |= nearest <= margin
            except Exception:
                pass

        grid_xy = grid_xy[keep]
        if len(grid_xy) == 0:
            return None

        zz = np.full(len(grid_xy), floor_z, dtype=np.float32)
        return np.column_stack([grid_xy[:, 0], grid_xy[:, 1], zz]).astype(np.float32)

    def _generate_circular_floor_patch(self, floor_z):
        """Generate the legacy circular floor patch centered under the target."""
        cx, cy, _ = self._tgt_centroid
        res = max(self.floor_grid_resolution, 1e-4)
        r = self.floor_grid_radius
        xs = np.arange(cx - r, cx + r + res * 0.5, res, dtype=np.float32)
        ys = np.arange(cy - r, cy + r + res * 0.5, res, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        xx = xx.ravel()
        yy = yy.ravel()
        in_circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        xx = xx[in_circle]
        yy = yy[in_circle]
        zz = np.full(len(xx), floor_z, dtype=np.float32)
        return np.column_stack([xx, yy, zz])

    def _stabilize_floor_z(self, floor_z):
        if self.floor_height_filter_tau <= 0.0 and self.floor_height_max_step <= 0.0:
            return floor_z

        now = rospy.get_time()
        if self._floor_z_filtered is None:
            self._floor_z_filtered = float(floor_z)
            self._floor_z_stamp = now
            return self._floor_z_filtered

        prev = float(self._floor_z_filtered)
        dt = max(0.0, now - float(self._floor_z_stamp or now))
        z = float(floor_z)

        if self.floor_height_filter_tau > 0.0 and dt > 0.0:
            alpha = dt / (self.floor_height_filter_tau + dt)
            z = prev + alpha * (z - prev)

        if self.floor_height_max_step > 0.0:
            dz = np.clip(z - prev, -self.floor_height_max_step,
                         self.floor_height_max_step)
            z = prev + float(dz)

        self._floor_z_filtered = z
        self._floor_z_stamp = now
        return z

    def _generate_floor_patch(self, obs_pts):
        """
        Return a [N,3] float32 grid of synthetic floor points below the target.

        Algorithm:
          1. Estimate floor height from the low target Z percentile.
          2. Fall back to nearby obstacle-ring points if the target cloud is absent.
          3. Fill either the target XY footprint or the legacy circular patch.

        Returns None when there is insufficient ring data to infer the floor.
        """
        if not self.generate_synthetic_floor:
            return None
        if self._tgt_centroid is None:
            return None

        cx, cy, cz = self._tgt_centroid
        # Floor height = bottom of the target itself (the food resting on the
        # container base), NOT the percentile of surrounding ring obstacle
        # points: those pick up the table/floor at z~0 and place the synthetic
        # floor far below a tall container's interior. A low percentile of the
        # target Z is robust to SAM2 mask outliers. Fall back to the ring
        # estimate only when no target cloud is available.
        tgt = self._tgt_world.get_points(min_confidence=1)
        if tgt is None or len(tgt) == 0:
            tgt = self._live_target
        if tgt is not None and len(tgt) >= 5:
            floor_z = float(np.percentile(tgt[:, 2], 20.0))
        else:
            if obs_pts is None or len(obs_pts) == 0:
                return None
            dx = obs_pts[:, 0] - cx
            dy = obs_pts[:, 1] - cy
            horiz_sq = dx * dx + dy * dy
            ring_mask = (horiz_sq < self.floor_sample_radius ** 2) & (obs_pts[:, 2] < cz)
            if ring_mask.sum() < 5:
                return None
            floor_z = float(np.percentile(obs_pts[ring_mask, 2], self.floor_height_percentile))

        floor_z = self._stabilize_floor_z(floor_z)

        if self.floor_footprint_mode == "target" and tgt is not None:
            patch = self._generate_target_footprint_patch(tgt, floor_z)
            if patch is not None:
                return patch

        return self._generate_circular_floor_patch(floor_z)

    # ── Publishing ────────────────────────────────────────────────────────────

    def _publish(self, stamp):
        header = Header(stamp=stamp, frame_id=self.world_frame)

        tgt_pts = self._tgt_world.get_points(min_confidence=self.tgt_commit_thresh)
        if tgt_pts is None:
            tgt_pts = self._live_target

        obs_pts = self.obs_world.get_points(min_confidence=self.obs_commit_thresh)
        if obs_pts is None:
            obs_pts = self._live_obs

        # Publish-time target exclusion: O(N) centroid-sphere guard.
        # Belt-and-suspenders on top of the integration-time filter — catches
        # any live_obs fallback points and handles centroid drift between evictions.
        if obs_pts is not None and len(obs_pts) > 0:
            obs_pts = self._carve_against_target(obs_pts)

        # Synthetic floor: infer floor height from ring obstacle points and fill
        # the depth-occluded region below the target so the CBF cannot drive the
        # fork through the bottom of the bowl.  Added after carving so the
        # floor patch is never zeroed by the exclusion sphere.
        floor_patch = self._generate_floor_patch(obs_pts)
        if floor_patch is not None and len(floor_patch) > 0:
            obs_pts = np.vstack([obs_pts, floor_patch]) \
                if obs_pts is not None and len(obs_pts) > 0 else floor_patch

        # Surface dilation: add 6 face-adjacent neighbor centers at publish time.
        # Fills per-scan-line gaps (up to 2×voxel_size wide) so the CBF sees a
        # continuous surface rather than isolated points with void between them.
        # Storage is unchanged — dilation is purely for the downstream CBF consumer.
        if obs_pts is not None and len(obs_pts) > 0 and self.dilate_published_cloud:
            vs = self.obs_voxel_size
            offsets = np.array([
                [vs, 0, 0], [-vs, 0, 0],
                [0, vs, 0], [0, -vs, 0],
                [0, 0, vs], [0, 0, -vs],
            ], dtype=np.float32)
            dilated = (obs_pts[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
            all_pts = np.vstack([obs_pts, dilated])
            h_all = _to_hashes(all_pts, vs)
            _, unique_idx = np.unique(h_all, return_index=True)
            obs_pts = all_pts[unique_idx].astype(np.float32)

        if obs_pts is not None and len(obs_pts) > self.obs_max_pts:
            obs_pts = obs_pts[:self.obs_max_pts]

        if obs_pts is not None and len(obs_pts) > 0:
            self.pub.publish(_make_cloud_msg(header, obs_pts))

        if self.log_counts:
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
