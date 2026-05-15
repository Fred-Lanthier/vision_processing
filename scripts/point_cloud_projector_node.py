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
        self.rng = np.random.default_rng()

        # SAM3 fork mask params
        self._sam3_prompt     = rospy.get_param("~fork_sam3_prompt",       "Fork teeth")
        self._sam3_confidence = float(rospy.get_param("~fork_sam3_confidence",  0.4))
        self._mask_dilation   = int(rospy.get_param("~fork_mask_dilation_px",   0))
        self._enable_fork_sam3_mask = bool(rospy.get_param("~enable_fork_sam3_mask", False))

        # Geometric fallback params (used only if SAM3 permanently unavailable)
        self.fork_filter = ForkFilter(
            capsule_length   = float(rospy.get_param("~fork_capsule_length", 0.15)),
            capsule_radius   = float(rospy.get_param("~fork_capsule_radius", 0.03)),
            prong_axis_local = rospy.get_param("~fork_prong_axis",           [0.0, 0.0, -1.0]),
            mask_dilation_px = self._mask_dilation,
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

        if self._enable_fork_sam3_mask:
            # First SAM3 attempt after 15 s — gives the model time to finish loading.
            # sam3_server registers the ROS service BEFORE loading weights, so
            # wait_for_service() returning does NOT mean the model is ready.
            rospy.Timer(rospy.Duration(15.0), self._try_sam3_mask, oneshot=True)
            rospy.loginfo("✅ Point Cloud Projector ready. SAM3 fork mask scheduled in 15 s.")
        else:
            self._geometric_mask_pending = True
            rospy.loginfo("✅ Point Cloud Projector ready. Geometric fork mask pending CameraInfo.")

    # ── Incoming data caches ───────────────────────────────────────────────────

    def _info_cb(self, msg):
        self.fx = msg.K[0]; self.cx = msg.K[2]
        self.fy = msg.K[4]; self.cy = msg.K[5]
        self.img_H = msg.height
        self.img_W = msg.width
        if self._geometric_mask_pending and not self._mask_ready:
            self._use_geometric_fallback()
            self._geometric_mask_pending = False
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
        rospy.loginfo(
            f"[fork_filter] SAM3 mask attempt {self._sam3_attempts}/{self._max_sam3_attempts} …"
        )

        # Prerequisites: need intrinsics + at least one RGB frame
        if self.fx is None or self._latest_rgb is None:
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
                rospy.loginfo(
                    f"[fork_filter] ✅ SAM3 fork mask ready — "
                    f"confidence={resp.confidence:.3f}  "
                    f"{100*dilated.astype(bool).mean():.1f}% of image masked"
                )
                return

            rospy.logwarn(
                f"[fork_filter] SAM3 not ready yet "
                f"(success={resp.success}, confidence={resp.confidence:.3f}). "
                f"Retrying in 10 s."
            )

        except Exception as e:
            rospy.logwarn(f"[fork_filter] SAM3 call failed ({e}). Retrying in 10 s.")

        # Schedule next attempt or give up and use geometric fallback
        if self._sam3_attempts < self._max_sam3_attempts:
            rospy.Timer(rospy.Duration(10.0), self._try_sam3_mask, oneshot=True)
        else:
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

    def deproject_to_3d(self, mask, depth_img, max_points=None):
        if self.fx is None:
            return None
        z = depth_img.astype(np.float32)
        if depth_img.dtype != np.float32:
            z /= 1000.0
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
        x = (u_idx - self.cx) * z_val / self.fx
        y = (v_idx - self.cy) * z_val / self.fy
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
        rospy.loginfo_throttle(
            5.0,
            "⏱️ [TIMING] PointCloud Projector: "
            f"{(time.perf_counter() - total_start) * 1000.0:.2f} ms | "
            f"decode={stages['decode']:.2f} tf={stages['tf']:.2f} "
            f"target={stages['target']:.2f} obs_deproj={stages['obs_deproj']:.2f} "
            f"fork_filter={stages['fork_filter']:.2f} publish={stages['publish']:.2f} ms | "
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
            "publish": 0.0,
        }

        # ── STRATEGY C: zero fork pixels in depth image ────────────────────────
        # pixel_mask is None until SAM3 delivers — no masking in the meantime.
        if self.fork_filter.pixel_mask is not None:
            depth_cv = depth_cv.copy()
            depth_cv[self.fork_filter.pixel_mask] = 0
        # ──────────────────────────────────────────────────────────────────────

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
                mask_cv, depth_cv, max_points=self.max_target_points
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
        mask_for_obs = mask_cv if tracking_active else self._last_valid_target_mask
        if mask_for_obs is not None:
            obs_mask = np.ones_like(mask_cv) * 255
            obs_mask[mask_for_obs > 0] = 0
        else:
            obs_mask = np.ones_like(mask_cv) * 255

        t_stage = time.perf_counter()
        obs_pre_sample = max(self.max_points, int(self.max_points * self.obstacle_sample_multiplier))
        obs_pts_cam = self.deproject_to_3d(obs_mask, depth_cv, max_points=obs_pre_sample)
        obs_in = 0 if obs_pts_cam is None else len(obs_pts_cam)
        stages["obs_deproj"] = self._elapsed_ms(time.perf_counter(), t_stage)

        # ── STRATEGY B: 3D capsule backup ─────────────────────────────────────
        # Once the fork pixel mask is active, the fork is already removed before
        # projection. Keep the 3D capsule as startup/fallback protection only.
        t_stage = time.perf_counter()
        use_3d_fork_backup = self.run_3d_fork_filter_backup or self.fork_filter.pixel_mask is None
        if obs_pts_cam is not None and use_3d_fork_backup:
            obs_pts_cam = self.fork_filter.filter_camera_frame_points(obs_pts_cam)
        stages["fork_filter"] = self._elapsed_ms(time.perf_counter(), t_stage)
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
