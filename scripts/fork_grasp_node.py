#!/usr/bin/env python3
"""
fork_grasp_node.py — weld the target food to the fork on contact (Gazebo).

The fork has no collision geometry, so it cannot physically hold the food. This
node detects contact by comparing two world-frame clouds the pipeline already
publishes:
  * /vision/fork_cloud            — fork mesh points (condition_pcd_from_perception)
  * /perception/persistent_target — accumulated target voxels (persistent_cloud)
When the minimum distance between any fork point and any target point drops below
contact_distance, it welds the food to the fork with a dynamic fixed joint via
gazebo_ros_link_attacher (/link_attacher_node/attach). This fires for contact
anywhere on the fork and anywhere on the target. Release with /fork_grasp/release.

On grasp it also snapshots the target's point cloud into the fork (`box_frame`)
frame and republishes it each tick on /fork_grasp/grasped_cloud. The CBF turns
that cloud into an extra protected + self-filtered "link" (distance-to-nearest-
grasped-point SDF) that conforms to the real target shape and rides the fork.
RViz can display the same topic to visualise the protected geometry.

Requirements:
  * gazebo_ros_link_attacher built/sourced; its world plugin
    (libgazebo_ros_link_attacher.so) loaded in worlds/feeding.world.
  * fork_tip preserved as a Gazebo link (disableFixedJointLumping in the xacro).

Services:
  /fork_grasp/grab    (std_srvs/Trigger) — force an attach now.
  /fork_grasp/release (std_srvs/Trigger) — detach (deliver the food).
"""
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

try:
    from gazebo_ros_link_attacher.srv import Attach, AttachRequest
except ImportError:  # package not built yet — fail loudly with guidance.
    Attach = None
    AttachRequest = None


def _cloud_to_np(msg):
    pts = np.array(
        list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
        dtype=np.float32)
    return pts if len(pts) else None


class ForkGraspNode:
    def __init__(self):
        # Link-attacher endpoints.
        self.robot_model = rospy.get_param("~robot_model", "panda")
        self.fork_link = rospy.get_param("~fork_link", "fork_tip")
        self.food_model = rospy.get_param("~food_model", "food_cube")
        self.food_link = rospy.get_param("~food_link", "base_link")

        # Clouds (both world frame).
        self.fork_topic = rospy.get_param("~fork_topic", "/vision/fork_cloud")
        self.target_topic = rospy.get_param(
            "~target_topic", "/perception/persistent_target")
        # Contact = min(|fork_i - target_j|) below this (m). The persistent
        # target is voxelised (~10 mm), so keep this >= one voxel.
        self.contact_distance = float(rospy.get_param("~contact_distance", 0.012))
        self.contact_cycles = int(rospy.get_param("~contact_cycles", 2))
        self.cloud_timeout = float(rospy.get_param("~cloud_timeout", 1.0))
        self.auto_attach = bool(rospy.get_param("~auto_attach", True))
        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))

        # Grasped object = the target's own point cloud, snapshotted in the fork
        # frame at attach and re-published each tick (fresh stamp) so RViz and the
        # CBF follow the fork via TF instead of freezing at the attach pose.
        self.grasp_cloud_topic = rospy.get_param(
            "~grasp_cloud_topic", "/fork_grasp/grasped_cloud")
        self.box_frame = rospy.get_param("~box_frame", "fork_tip")
        self.world_frame = rospy.get_param("~world_frame", "world")
        # Snapshot source: the LIVE target (tight, current), not the accumulated
        # persistent target (which smears since it is never freespace-evicted).
        self.grasp_fit_topic = rospy.get_param("~box_fit_topic", "/perception/target")
        # Drop snapshot points farther than this from the cloud median (smear).
        self.grasp_crop_radius = float(rospy.get_param("~grasp_crop_radius", 0.05))
        # Grasped-cloud source. "perception" = snapshot of the perceived target
        # (visible faces only: the UNDERSIDE that drags on the pile/bowl during
        # the exit is missing, so h_food under-reports contact -- run102 read
        # +7 mm while the food was physically dragging). "ground_truth" = full
        # analytic box surface at the food's live Gazebo pose (the same
        # "perfect perception" ablation as gt_cube_cloud_node in PP); falls
        # back to the perception snapshot if the Gazebo service is missing.
        self.grasp_cloud_source = str(rospy.get_param(
            "~grasp_cloud_source", "perception")).lower()
        # Default = the food_cube COLLISION box (0.03^3), which is what
        # physically contacts; the visual box is 0.025^3.
        self.gt_half = np.array([
            float(rospy.get_param("~gt_half_x", 0.015)),
            float(rospy.get_param("~gt_half_y", 0.015)),
            float(rospy.get_param("~gt_half_z", 0.015))])
        self.gt_num_points = int(rospy.get_param("~gt_num_points", 300))
        # Gazebo world -> TF world (robot base on the table top at z=0.75).
        self.gt_table_z_offset = float(rospy.get_param("~gt_table_z_offset", 0.75))
        self._gt_state_srv = None
        self.publish_viz = bool(rospy.get_param("~publish_viz", True))
        # SDF envelope viz: one sphere per grasped point of radius
        # grasp_point_radius (the CBF's SDF=0 surface). Set envelope_radius to
        # grasp_point_radius + d_safe to instead see the CBF keep-out boundary.
        self.envelope_radius = float(rospy.get_param("~envelope_radius", 0.012))
        self.pub_grasp_cloud = rospy.Publisher(
            self.grasp_cloud_topic, PointCloud2, queue_size=1, latch=True)
        self.pub_envelope = rospy.Publisher(
            "/viz/grasp_sdf_envelope", Marker, queue_size=1, latch=True)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.attached = False
        self._near_count = 0
        self._fork_pts = None
        self._fork_stamp = 0.0
        self._target_pts = None
        self._target_stamp = 0.0
        self._fit_pts = None              # live target cloud (world frame)
        self._fit_stamp = 0.0
        self._grasp_cloud_fork = None     # snapshot in fork frame [M,3] (cached)

        self._attach_srv = None
        self._detach_srv = None
        self._connect_attacher()

        rospy.Subscriber(self.fork_topic, PointCloud2, self._fork_cb, queue_size=1)
        rospy.Subscriber(self.target_topic, PointCloud2, self._target_cb,
                         queue_size=1)
        if self.grasp_fit_topic and self.grasp_fit_topic != self.target_topic:
            rospy.Subscriber(self.grasp_fit_topic, PointCloud2, self._fit_cb,
                             queue_size=1)
        rospy.Service("/fork_grasp/grab", Trigger, self._grab_srv)
        rospy.Service("/fork_grasp/release", Trigger, self._release_srv)

        rospy.loginfo(
            "fork_grasp_node ready: weld %s/%s <- %s/%s when min|%s , %s| < "
            "%.3f m for %d cycles (auto_attach=%s)",
            self.robot_model, self.fork_link, self.food_model, self.food_link,
            self.fork_topic, self.target_topic, self.contact_distance,
            self.contact_cycles, self.auto_attach)

        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.rate_hz, 1.0)),
                                 self._tick)

    # -- service plumbing ----------------------------------------------------
    def _connect_attacher(self):
        if Attach is None:
            rospy.logerr("gazebo_ros_link_attacher not importable — clone it "
                         "into the workspace src and rebuild (catkin_make).")
            return
        try:
            rospy.wait_for_service("/link_attacher_node/attach", timeout=10.0)
            self._attach_srv = rospy.ServiceProxy(
                "/link_attacher_node/attach", Attach)
            self._detach_srv = rospy.ServiceProxy(
                "/link_attacher_node/detach", Attach)
            rospy.loginfo("Connected to /link_attacher_node attach/detach.")
        except rospy.ROSException:
            rospy.logwarn("/link_attacher_node not available yet — is the world "
                          "plugin libgazebo_ros_link_attacher.so loaded? Will "
                          "retry lazily on first attach.")

    def _make_request(self):
        req = AttachRequest()
        req.model_name_1 = self.robot_model
        req.link_name_1 = self.fork_link
        req.model_name_2 = self.food_model
        req.link_name_2 = self.food_link
        return req

    # -- callbacks -----------------------------------------------------------
    def _fork_cb(self, msg):
        self._fork_pts = _cloud_to_np(msg)
        self._fork_stamp = rospy.get_time()

    def _target_cb(self, msg):
        self._target_pts = _cloud_to_np(msg)
        self._target_stamp = rospy.get_time()

    def _fit_cb(self, msg):
        self._fit_pts = _cloud_to_np(msg)
        self._fit_stamp = rospy.get_time()

    def _contact_distance(self):
        """Min distance between any fork point and any target point, or None if
        either cloud is missing or stale."""
        now = rospy.get_time()
        if self._fork_pts is None or self._target_pts is None:
            return None
        if (now - self._fork_stamp) > self.cloud_timeout or \
                (now - self._target_stamp) > self.cloud_timeout:
            return None
        fork, tgt = self._fork_pts, self._target_pts
        if cKDTree is not None:
            d, _ = cKDTree(tgt).query(fork, k=1)
            return float(np.min(d))
        # Fallback: brute-force min distance.
        diff = fork[:, None, :] - tgt[None, :, :]
        return float(np.sqrt((diff ** 2).sum(-1)).min())

    # -- attach / detach -----------------------------------------------------
    def attach(self):
        if self.attached:
            return True
        if self._attach_srv is None:
            self._connect_attacher()
        if self._attach_srv is None:
            return False
        try:
            self._attach_srv(self._make_request())
            self.attached = True
            rospy.loginfo("Spiked food: welded %s/%s to %s/%s.",
                          self.food_model, self.food_link,
                          self.robot_model, self.fork_link)
            self._snapshot_grasp_cloud()
            return True
        except rospy.ServiceException as e:
            rospy.logwarn("attach failed: %s", e)
            return False

    def detach(self):
        if not self.attached:
            return True
        if self._detach_srv is None:
            return False
        try:
            self._detach_srv(self._make_request())
            self.attached = False
            self._near_count = 0
            rospy.loginfo("Released food: detached %s from %s.",
                          self.food_model, self.robot_model)
            self._clear_grasp_cloud()
            return True
        except rospy.ServiceException as e:
            rospy.logwarn("detach failed: %s", e)
            return False

    # -- grasped cloud (protected/self-filtered by the CBF) ------------------
    def _gt_box_points_world(self):
        """Full analytic box surface at the food's live Gazebo pose, in the TF
        world frame. Unlike the perception snapshot this includes the bottom
        faces, so the food barrier can see pile/bowl contact under the carried
        food."""
        try:
            from gazebo_msgs.srv import GetModelState
            if self._gt_state_srv is None:
                rospy.wait_for_service('/gazebo/get_model_state', timeout=2.0)
                self._gt_state_srv = rospy.ServiceProxy(
                    '/gazebo/get_model_state', GetModelState)
            state = self._gt_state_srv(self.food_model, '')
            if not state.success:
                raise RuntimeError(state.status_message)
        except Exception as e:
            rospy.logwarn(
                "GT grasp cloud unavailable (%s); using perception snapshot.", e)
            return None
        q = state.pose.orientation
        Rm = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        pos = np.array([state.pose.position.x, state.pose.position.y,
                        state.pose.position.z - self.gt_table_z_offset])
        half = self.gt_half
        rng = np.random.default_rng(0)
        # Area-weighted face sampling (same scheme as gt_cube_cloud_node).
        areas = np.array([half[1] * half[2], half[0] * half[2], half[0] * half[1]])
        n = self.gt_num_points
        loc = (rng.random((n, 3)) * 2.0 - 1.0) * half
        ax = rng.choice(3, size=n, p=areas / areas.sum())
        sgn = np.where(rng.random(n) < 0.5, 1.0, -1.0)
        loc[np.arange(n), ax] = half[ax] * sgn
        return (loc @ Rm.T + pos).astype(np.float32)

    def _snapshot_grasp_cloud(self):
        """Snapshot the target into the fork frame so it rides the fork. Uses the
        live target (tight) and crops smear/outliers around the cloud median."""
        now = rospy.get_time()
        gt_pts = None
        if self.grasp_cloud_source == "ground_truth":
            gt_pts = self._gt_box_points_world()
        if gt_pts is not None:
            pts = gt_pts
        elif self._fit_pts is not None and (now - self._fit_stamp) <= self.cloud_timeout:
            pts = self._fit_pts                       # live target: tight, current
        else:
            pts = self._target_pts                    # fallback: persistent target
        if pts is None or len(pts) < 3:
            rospy.logwarn("No target points at grasp; skipping protected cloud.")
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.box_frame, self.world_frame, rospy.Time(0),
                rospy.Duration(0.2))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Grasp cloud TF lookup failed (%s); skipping.", e)
            return
        q = tf.transform.rotation
        t = tf.transform.translation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        t = np.array([t.x, t.y, t.z], dtype=np.float32)
        p_fork = ((R @ pts.T).T + t).astype(np.float32)   # world -> fork frame

        # Median crop targets perception smear; the analytic GT box has none,
        # and the crop radius (0.02) would clip its corners (0.026 from center).
        if gt_pts is None:
            med = np.median(p_fork, axis=0)
            keep = np.linalg.norm(p_fork - med, axis=1) <= self.grasp_crop_radius
            if keep.sum() >= 3:
                p_fork = p_fork[keep]

        self._grasp_cloud_fork = p_fork
        self._publish_grasp_cloud()
        rospy.loginfo("Protected grasp cloud: %d pts in %s frame.",
                      len(p_fork), self.box_frame)

    def _publish_grasp_cloud(self):
        """(Re)publish the cached snapshot in the fork frame with a fresh stamp,
        so the CBF and RViz transform it through the live fork TF."""
        if self._grasp_cloud_fork is None:
            return
        header = Header(stamp=rospy.Time.now(), frame_id=self.box_frame)
        self.pub_grasp_cloud.publish(
            pc2.create_cloud_xyz32(header, self._grasp_cloud_fork))
        if self.publish_viz:
            self._publish_envelope(Marker.ADD)

    def _publish_envelope(self, action):
        """SPHERE_LIST envelope of the grasped-cloud SDF: one sphere per point of
        diameter 2*envelope_radius (the union = the SDF=0 isosurface)."""
        m = Marker()
        m.header.frame_id = self.box_frame
        m.header.stamp = rospy.Time.now()
        m.ns = "grasp_sdf_envelope"
        m.id = 0
        m.type = Marker.SPHERE_LIST
        m.action = action
        m.pose.orientation.w = 1.0
        d = 2.0 * self.envelope_radius
        m.scale.x, m.scale.y, m.scale.z = d, d, d
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.3, 0.35
        if action == Marker.ADD and self._grasp_cloud_fork is not None:
            m.points = [Point(float(p[0]), float(p[1]), float(p[2]))
                        for p in self._grasp_cloud_fork]
        self.pub_envelope.publish(m)

    def _clear_grasp_cloud(self):
        self._grasp_cloud_fork = None
        header = Header(stamp=rospy.Time.now(), frame_id=self.box_frame)
        self.pub_grasp_cloud.publish(
            pc2.create_cloud_xyz32(header, np.empty((0, 3), dtype=np.float32)))
        if self.publish_viz:
            self._publish_envelope(Marker.DELETE)

    def _grab_srv(self, _req):
        ok = self.attach()
        return TriggerResponse(success=ok,
                               message="attached" if ok else "attach failed")

    def _release_srv(self, _req):
        ok = self.detach()
        return TriggerResponse(success=ok,
                               message="detached" if ok else "detach failed")

    # -- main loop -----------------------------------------------------------
    def _tick(self, _evt):
        if self.attached:
            # Republish the snapshot each tick (fresh stamp) so the CBF and RViz
            # keep transforming it through the live fork TF. A once-latched cloud
            # would freeze at its attach-time stamp and appear static.
            self._publish_grasp_cloud()
            return
        if not self.auto_attach:
            return
        dist = self._contact_distance()
        if dist is not None and dist < self.contact_distance:
            self._near_count += 1
            if self._near_count >= self.contact_cycles:
                self.attach()
        else:
            self._near_count = 0


if __name__ == "__main__":
    rospy.init_node("fork_grasp_node")
    ForkGraspNode()
    rospy.spin()
