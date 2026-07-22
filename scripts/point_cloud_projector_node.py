#!/usr/bin/env python3
import rospy
import time
import numpy as np
import cv2
import message_filters
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, PointField
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import tf2_ros
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from fork_filter import ForkFilter
from robot_self_filter import RobotSelfFilter

import rospkg
_pkg = rospkg.RosPack().get_path('vision_processing')
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)
from pipeline_timing import TimingPublisher

from vision_processing.srv import Sam3Segment, Sam3SegmentRequest


class PointCloudProjectorNode:
    def __init__(self):
        rospy.init_node('point_cloud_projector_node')

        self.target_frame = rospy.get_param("~target_frame", "world")
        self.camera_frame = rospy.get_param("~camera_frame", "camera_wrist_optical_frame")
        self.max_points = int(rospy.get_param("~max_points", 5000))
        self.max_target_points = int(rospy.get_param("~max_target_points", 2000))
        self.obstacle_sample_multiplier = float(rospy.get_param("~obstacle_sample_multiplier", 3.0))
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 0.01))
        self.run_3d_fork_filter_backup = bool(rospy.get_param("~run_3d_fork_filter_backup", False))
        self.log_timing = bool(rospy.get_param("~log_timing", True))
        self.log_events = bool(rospy.get_param("~log_events", True))
        # Per-stage timing -> /pipeline/timing/* (recorded in the bag for section 5.6)
        self.timing = TimingPublisher(enabled=rospy.get_param("~publish_timing", True))
        self.rng = np.random.default_rng()

        # SAM3 fork mask params
        self._sam3_prompt     = rospy.get_param("~fork_sam3_prompt",       "Fork teeth")
        self._sam3_confidence = float(rospy.get_param("~fork_sam3_confidence",  0.4))
        self._mask_dilation   = int(rospy.get_param("~fork_mask_dilation_px",   0))
        self._enable_fork_sam3_mask = bool(rospy.get_param("~enable_fork_sam3_mask", False))
        # Master switch for ALL fork removal (SAM3 mask, geometric capsule, 3D backup).
        # Default True = unchanged feeding behaviour. Set False for the pick-and-place
        # gripper (no fork): otherwise the geometric capsule mask zeroes depth at the
        # tool region and wipes the target cloud, and the 3D backup runs on a bad
        # (no fork_tip TF) capsule.
        self._enable_fork_filter = bool(rospy.get_param("~enable_fork_filter", True))

        # 3D fork backup: an SDF shell around the TRUE fork surface (fork_tip.stl)
        # is the geometrically-correct backstop for the 2D mask's near-fork leaks.
        # Falls back to the analytic capsule if disabled or the mesh fails to load.
        self._fork_sdf_filter = bool(rospy.get_param("~fork_sdf_filter", False))
        self._fork_sdf_margin = float(rospy.get_param("~fork_sdf_margin", 0.005))
        _fork_mesh_default = os.path.join(
            _pkg, "src/vision_processing/diffusion_model_train/fork_tip.stl")
        self._fork_mesh_path = rospy.get_param("~fork_mesh_path", _fork_mesh_default)

        # Same 5 mm surface shell as the fork backup below, but around every
        # protected link instead of only the fork. Obstacle cloud only: the
        # target cloud is the object being grasped and must survive contact.
        self.robot_self_filter = None
        if bool(rospy.get_param("~robot_self_filter", False)):
            robot_xml = rospy.get_param(
                rospy.get_param("~robot_description_param", "robot_description"), "")
            self.robot_self_filter = RobotSelfFilter(
                robot_xml   = robot_xml,
                link_names  = list(rospy.get_param(
                    "~robot_self_filter_links",
                    ['panda_link2', 'panda_link3', 'panda_link4', 'panda_link5',
                     'panda_link6', 'panda_link7', 'panda_hand',
                     'panda_rightfinger', 'panda_leftfinger'])),
                margin      = float(rospy.get_param("~robot_self_filter_margin", 0.005)),
                resolution  = float(rospy.get_param("~robot_self_filter_resolution", 0.0)),
                logger      = lambda msg: rospy.loginfo("[robot_self_filter] " + msg),
            )

        # Geometric fallback params (used only if SAM3 permanently unavailable)
        self.fork_filter = ForkFilter(
            capsule_length   = float(rospy.get_param("~fork_capsule_length", 0.15)),
            capsule_radius   = float(rospy.get_param("~fork_capsule_radius", 0.03)),
            prong_axis_local = rospy.get_param("~fork_prong_axis",           [0.0, 0.0, -1.0]),
            mask_dilation_px = self._mask_dilation,
            sdf_filter       = self._fork_sdf_filter,
            sdf_margin       = self._fork_sdf_margin,
            fork_mesh_path   = self._fork_mesh_path,
            sdf_surface_samples = int(rospy.get_param("~fork_sdf_samples", 4000)),
        )

        # pixel_mask starts None → no 2D masking until SAM3 delivers
        self._last_valid_target_mask = None  # last SAM2 mask with sufficient coverage
        # (Strategy B 3D capsule handles interim filtering)
        self._mask_ready   = False
        self._sam3_attempts = 0
        self._max_sam3_attempts = 20       # ~20 × 10 s = 3 min before geometric fallback
        self._latest_rgb   = None
        self.img_H = self.img_W = None
        self._geometric_mask_pending = False

        self.bridge      = CvBridge()
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.fx = self.fy = self.cx = self.cy = None

        # Publishers
        self.pub_target   = rospy.Publisher('/perception/target',   PointCloud2, queue_size=1)
        self.pub_obs      = rospy.Publisher('/perception/obstacles', PointCloud2, queue_size=1)
        self.pub_marker   = rospy.Publisher('/viz/fork_capsule',     Marker,      queue_size=1, latch=True)
        self.pub_mask_dbg = rospy.Publisher('/viz/fork_mask',        Image,       queue_size=1, latch=True)

        # Topics
        depth_topic = rospy.get_param("~depth_topic", "/camera_wrist/aligned_depth_to_color/image_raw")
        mask_topic  = rospy.get_param("~mask_topic",  "/vision/sam2_tracked_mask")
        info_topic  = rospy.get_param("~info_topic",  "/camera_wrist/color/camera_info")
        rgb_topic   = rospy.get_param("~rgb_topic",   "/camera_wrist/color/image_raw")

        rospy.Subscriber(info_topic, CameraInfo, self._info_cb)
        rospy.Subscriber(rgb_topic,  Image,      self._rgb_cb)

        sub_depth = message_filters.Subscriber(depth_topic, Image)
        sub_mask  = message_filters.Subscriber(mask_topic,  Image)
        self.ts   = message_filters.ApproximateTimeSynchronizer(
                        [sub_depth, sub_mask], queue_size=5, slop=0.1)
        self.ts.registerCallback(self.sync_cb)

        if not self._enable_fork_filter:
            if self.log_events:
                rospy.loginfo("✅ Point Cloud Projector ready. Fork filter DISABLED (gripper / no fork).")
        elif self._enable_fork_sam3_mask:
            # First SAM3 attempt after 15 s — gives the model time to finish loading.
            # sam3_server registers the ROS service BEFORE loading weights, so
            # wait_for_service() returning does NOT mean the model is ready.
            rospy.Timer(rospy.Duration(15.0), self._try_sam3_mask, oneshot=True)
            if self.log_events:
                rospy.loginfo("✅ Point Cloud Projector ready. SAM3 fork mask scheduled in 15 s.")
        else:
            self._geometric_mask_pending = True
            if self.log_events:
                rospy.loginfo("✅ Point Cloud Projector ready. Geometric fork mask pending CameraInfo.")

    def _link_pose_lookup(self, stamp):
        """Return lookup(link) -> (R, t) placing a link in the CAMERA frame."""
        def lookup(link):
            try:
                tr = self.tf_buffer.lookup_transform(
                    self.camera_frame, link, stamp,
                    rospy.Duration(self.tf_timeout))
            except Exception as e:
                rospy.logwarn_throttle(
                    10, f"[robot_self_filter] TF {self.camera_frame}<-{link} "
                        f"failed ({e}); link not filtered this frame.")
                return None
            t, r = tr.transform.translation, tr.transform.rotation
            return (Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix(),
                    np.array([t.x, t.y, t.z]))
        return lookup

    # ── Incoming data caches ───────────────────────────────────────────────────

    def _info_cb(self, msg):
        self.fx = msg.K[0]; self.cx = msg.K[2]
        self.fy = msg.K[4]; self.cy = msg.K[5]
        self.img_H = msg.height
        self.img_W = msg.width
        if self._geometric_mask_pending and not self._mask_ready:
            self._use_geometric_fallback()
            self._geometric_mask_pending = False
            if self.log_events:
                rospy.loginfo("✅ Geometric fork mask initialized from CameraInfo.")

    def _rgb_cb(self, msg):
        self._latest_rgb = msg   # always keep the freshest frame for SAM3

    # ── SAM3 fork mask — timer-driven, non-blocking ────────────────────────────

    def _try_sam3_mask(self, event=None):
        """Non-blocking timer callback. Retries every 10 s until SAM3 gives a
        valid fork mask or the maximum number of attempts is exhausted."""

        if self._mask_ready:
            return

        self._sam3_attempts += 1
        if self.log_events:
            rospy.loginfo(
                f"[fork_filter] SAM3 mask attempt {self._sam3_attempts}/{self._max_sam3_attempts} …"
            )

        # Prerequisites: need intrinsics + at least one RGB frame
        if self.fx is None or self._latest_rgb is None:
            if self.log_events:
                rospy.logwarn("[fork_filter] Camera not ready yet, retrying in 10 s.")
            rospy.Timer(rospy.Duration(10.0), self._try_sam3_mask, oneshot=True)
            return

        try:
            sam3  = rospy.ServiceProxy('/sam3/segment', Sam3Segment)
            req   = Sam3SegmentRequest(
                rgb_image            = self._latest_rgb,
                text_prompt          = self._sam3_prompt,
                confidence_threshold = self._sam3_confidence,
                top_k                = 4,   # union of all 4 fork teeth
            )
            resp  = sam3(req)

            if resp.success and resp.confidence >= self._sam3_confidence:
                raw  = self.bridge.imgmsg_to_cv2(resp.mask, desired_encoding="mono8")
                k    = self._mask_dilation
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
                dilated = cv2.dilate(raw, kern)

                self.fork_filter.pixel_mask = dilated.astype(bool)
                self._mask_ready = True
                self._publish_mask_debug(dilated)
                if self.log_events:
                    rospy.loginfo(
                        f"[fork_filter] ✅ SAM3 fork mask ready — "
                        f"confidence={resp.confidence:.3f}  "
                        f"{100*dilated.astype(bool).mean():.1f}% of image masked"
                    )
                return

            if self.log_events:
                rospy.logwarn(
                    f"[fork_filter] SAM3 not ready yet "
                    f"(success={resp.success}, confidence={resp.confidence:.3f}). "
                    f"Retrying in 10 s."
                )

        except Exception as e:
            if self.log_events:
                rospy.logwarn(f"[fork_filter] SAM3 call failed ({e}). Retrying in 10 s.")

        # Schedule next attempt or give up and use geometric fallback
        if self._sam3_attempts < self._max_sam3_attempts:
            rospy.Timer(rospy.Duration(10.0), self._try_sam3_mask, oneshot=True)
        else:
            if self.log_events:
                rospy.logwarn(
                    "[fork_filter] SAM3 unreachable after max attempts. "
                    "Switching to geometric capsule mask."
                )
            self._use_geometric_fallback()

    def _use_geometric_fallback(self):
        mask = self.fork_filter.compute_pixel_mask(
            self.fx, self.fy, self.cx, self.cy, self.img_H, self.img_W
        )
        self._mask_ready = True
        self._publish_capsule_marker()
        self._publish_mask_debug((mask * 255).astype(np.uint8))
        if self.log_events:
            rospy.loginfo(
                f"[fork_filter] Geometric mask active — "
                f"P1={self.fork_filter.P1.round(3)} P2={self.fork_filter.P2.round(3)}"
            )

    # ── Debug publishers ────────────────────────────────────────────────────────

    def _publish_mask_debug(self, mask_uint8):
        msg = self.bridge.cv2_to_imgmsg(mask_uint8, encoding="mono8")
        msg.header.stamp    = rospy.Time.now()
        msg.header.frame_id = self.camera_frame
        self.pub_mask_dbg.publish(msg)

    def _publish_capsule_marker(self):
        P1, P2 = self.fork_filter.capsule_endpoints_camera_frame()
        m             = Marker()
        m.header.frame_id = self.camera_frame
        m.header.stamp    = rospy.Time.now()
        m.ns              = "fork_capsule"
        m.id              = 0
        m.type            = Marker.LINE_LIST
        m.action          = Marker.ADD
        m.scale.x         = 2.0 * self.fork_filter.capsule_radius
        m.color           = ColorRGBA(0.0, 1.0, 0.0, 0.5)
        m.points.append(Point(x=P1[0], y=P1[1], z=P1[2]))
        m.points.append(Point(x=P2[0], y=P2[1], z=P2[2]))
        self.pub_marker.publish(m)

    # ── 3D projection helpers ──────────────────────────────────────────────────

    @staticmethod
    def xyz_array_to_cloud(points, frame_id, stamp):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)

        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = points.shape[0]
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = False
        msg.data = points.tobytes()
        return msg

    def deproject_to_3d(self, mask, z, max_points=None, crop_to_mask=False):
        """mask -> camera-frame points. ``z`` is the depth in METRES float32
        (converted ONCE per frame in sync_cb; it used to be re-converted
        full-frame inside every call, twice per frame).

        crop_to_mask: restrict all per-pixel work to the mask's bounding
        box. For a target mask covering ~1% of the frame this removes ~99%
        of the boolean/nonzero work; output is bit-identical (row-major
        order inside the bbox equals the full-frame order restricted to the
        mask, so even the sampling RNG draws the same points).
        """
        if self.fx is None:
            return None
        v0 = u0 = 0
        if crop_to_mask:
            rows = np.flatnonzero(mask.any(axis=1))
            if rows.size == 0:
                return None
            cols = np.flatnonzero(mask.any(axis=0))
            v0, v1 = int(rows[0]), int(rows[-1]) + 1
            u0, u1 = int(cols[0]), int(cols[-1]) + 1
            mask = mask[v0:v1, u0:u1]
            z = z[v0:v1, u0:u1]
        valid = (mask > 0) & (z > 0.01) & (z < 3.0) & np.isfinite(z)
        n_valid = int(np.count_nonzero(valid))
        if n_valid < 10:
            return None
        v_idx, u_idx = np.nonzero(valid)
        if max_points is not None and n_valid > max_points:
            idx = self.rng.choice(n_valid, int(max_points), replace=False)
            v_idx = v_idx[idx]
            u_idx = u_idx[idx]
        z_val = z[v_idx, u_idx]
        x = (u_idx + u0 - self.cx) * z_val / self.fx
        y = (v_idx + v0 - self.cy) * z_val / self.fy
        return np.stack([x, y, z_val], axis=-1).astype(np.float32, copy=False)

    def lookup_world_transform(self, stamp):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                stamp,
                rospy.Duration(self.tf_timeout),
            )
        except Exception as stamped_error:
            try:
                trans = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    self.camera_frame,
                    rospy.Time(0),
                    rospy.Duration(0.0),
                )
                rospy.logwarn_throttle(
                    5,
                    f"TF stamped lookup failed in projector ({stamped_error}); using latest transform.",
                )
            except Exception as latest_error:
                rospy.logwarn_throttle(5, f"TF Error in projector: {latest_error}")
                return None

        t = trans.transform.translation
        r = trans.transform.rotation
        R = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix().astype(np.float32)
        offset = np.array([t.x, t.y, t.z], dtype=np.float32)
        return R, offset

    @staticmethod
    def transform_to_world(points_cam, world_tf):
        if world_tf is None:
            return None
        R, offset = world_tf
        return points_cam @ R.T + offset

    def downsample_points(self, points, max_points):
        if points is None or len(points) <= max_points:
            return points
        idx = self.rng.choice(len(points), int(max_points), replace=False)
        return points[idx]

    @staticmethod
    def _elapsed_ms(newer, older):
        return (newer - older) * 1000.0

    def _log_timing(self, total_start, stages, obs_in, obs_pub, target_pub):
        self.timing.publish('projection', (time.perf_counter() - total_start) * 1000.0)
        if not self.log_timing:
            return
        rospy.loginfo_throttle(
            5.0,
            "⏱️ [TIMING] PointCloud Projector: "
            f"{(time.perf_counter() - total_start) * 1000.0:.2f} ms | "
            f"decode={stages['decode']:.2f} tf={stages['tf']:.2f} "
            f"target={stages['target']:.2f} obs_deproj={stages['obs_deproj']:.2f} "
            f"fork_filter={stages['fork_filter']:.2f} "
            f"self_filter={stages['self_filter']:.2f} "
            f"publish={stages['publish']:.2f} ms | "
            f"obs_in={obs_in} obs_pub={obs_pub} target_pub={target_pub}"
        )

    def transform_to_world_legacy(self, points_cam, stamp):
        try:
            world_tf = self.lookup_world_transform(stamp)
            return self.transform_to_world(points_cam, world_tf)
        except Exception as e:
            rospy.logwarn_throttle(5, f"TF Error in projector: {e}")
            return None

    # ── Main callback ──────────────────────────────────────────────────────────

    def sync_cb(self, depth_msg, mask_msg):
        if self.fx is None:
            return
            
        t0 = time.perf_counter()
        t_stage = t0

        depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        mask_cv  = self.bridge.imgmsg_to_cv2(mask_msg,  desired_encoding="mono8")
        stamp    = depth_msg.header.stamp
        stages = {
            "decode": self._elapsed_ms(time.perf_counter(), t_stage),
            "tf": 0.0,
            "target": 0.0,
            "obs_deproj": 0.0,
            "fork_filter": 0.0,
            "self_filter": 0.0,
            "publish": 0.0,
        }

        # ── STRATEGY C: zero fork pixels in depth image ────────────────────────
        # pixel_mask is None until SAM3 delivers — no masking in the meantime.
        if self.fork_filter.pixel_mask is not None:
            depth_cv = depth_cv.copy()
            depth_cv[self.fork_filter.pixel_mask] = 0
        # ──────────────────────────────────────────────────────────────────────

        # Depth -> metres float32 ONCE per frame (was re-converted full-frame
        # inside every deproject_to_3d call: 2x per frame).
        z_m = depth_cv.astype(np.float32)
        if depth_cv.dtype != np.float32:
            z_m /= 1000.0

        t_stage = time.perf_counter()
        world_tf = self.lookup_world_transform(stamp)
        stages["tf"] = self._elapsed_ms(time.perf_counter(), t_stage)
        if world_tf is None:
            self._log_timing(t0, stages, 0, 0, 0)
            return

        # 1. TARGET — pixels inside SAM2 mask
        tracking_active = np.sum(mask_cv) > 50
        target_pub = 0
        t_stage = time.perf_counter()
        if tracking_active:
            self._last_valid_target_mask = mask_cv.copy()
            target_pts_cam = self.deproject_to_3d(
                mask_cv, z_m, max_points=self.max_target_points,
                crop_to_mask=True
            )
            if target_pts_cam is not None:
                target_w = self.transform_to_world(target_pts_cam, world_tf)
                if target_w is not None:
                    t_pub = time.perf_counter()
                    self.pub_target.publish(
                        self.xyz_array_to_cloud(target_w, self.target_frame, rospy.Time.now())
                    )
                    stages["publish"] += self._elapsed_ms(time.perf_counter(), t_pub)
                    target_pub = len(target_w)
        else:
            # Publish empty cloud to explicitly clear downstream state
            self.pub_target.publish(
                self.xyz_array_to_cloud(np.empty((0, 3)), self.target_frame, rospy.Time.now())
            )
        stages["target"] = self._elapsed_ms(time.perf_counter(), t_stage)

        # 2. OBSTACLES — pixels outside SAM2 mask.
        # When tracking is lost, fall back to the last valid mask so the target
        # region is not reclassified as an obstacle and the CBF does not block approach.
        # Skipped entirely when nothing subscribes to the obstacle topic: the
        # PP launch runs two projector instances whose obstacle outputs are
        # unused (the CBF cloud is synthetic there), yet this full-frame pass
        # dominated the 'projection' timing stat (p95 ~19 ms). Subscriber-
        # gated, so the feeding pipeline (perception_processing subscribes)
        # is byte-identical.
        t_stage = time.perf_counter()
        obs_pts_cam = None
        obs_in = 0
        if self.pub_obs.get_num_connections() > 0:
            mask_for_obs = mask_cv if tracking_active else self._last_valid_target_mask
            if mask_for_obs is not None:
                obs_mask = np.ones_like(mask_cv) * 255
                obs_mask[mask_for_obs > 0] = 0
            else:
                obs_mask = np.ones_like(mask_cv) * 255
            obs_pre_sample = max(self.max_points, int(self.max_points * self.obstacle_sample_multiplier))
            obs_pts_cam = self.deproject_to_3d(obs_mask, z_m, max_points=obs_pre_sample)
            obs_in = 0 if obs_pts_cam is None else len(obs_pts_cam)
        stages["obs_deproj"] = self._elapsed_ms(time.perf_counter(), t_stage)

        # ── STRATEGY B: 3D fork backup (SDF shell or capsule) ─────────────────
        # The 2D mask cannot represent "within N mm of the fork in 3D", so near-
        # fork points still leak when the fork is close to a surface (inside the
        # container). Run the 3D backup always when the SDF shell is active; the
        # capsule stays a startup-only fallback (before the pixel mask exists).
        t_stage = time.perf_counter()
        use_3d_fork_backup = self._enable_fork_filter and (
                              self.run_3d_fork_filter_backup
                              or self.fork_filter.sdf_active
                              or self.fork_filter.pixel_mask is None)
        if obs_pts_cam is not None and use_3d_fork_backup:
            obs_pts_cam = self.fork_filter.filter_camera_frame_points(obs_pts_cam)
        stages["fork_filter"] = self._elapsed_ms(time.perf_counter(), t_stage)
        # ──────────────────────────────────────────────────────────────────────

        # ── Same shell, every protected link (TF-posed, so it follows the arm) ─
        t_stage = time.perf_counter()
        if obs_pts_cam is not None and self.robot_self_filter is not None:
            obs_pts_cam = self.robot_self_filter.filter_points(
                obs_pts_cam, self._link_pose_lookup(stamp))
        stages["self_filter"] = self._elapsed_ms(time.perf_counter(), t_stage)
        # ──────────────────────────────────────────────────────────────────────

        obs_pub = 0
        if obs_pts_cam is not None and len(obs_pts_cam) > 0:
            obs_pts_cam = self.downsample_points(obs_pts_cam, self.max_points)
            obs_w = self.transform_to_world(obs_pts_cam, world_tf)
            if obs_w is not None:
                t_pub = time.perf_counter()
                self.pub_obs.publish(
                    self.xyz_array_to_cloud(obs_w, self.target_frame, rospy.Time.now())
                )
                stages["publish"] += self._elapsed_ms(time.perf_counter(), t_pub)
                obs_pub = len(obs_w)

        self._log_timing(t0, stages, obs_in, obs_pub, target_pub)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    PointCloudProjectorNode().run()
