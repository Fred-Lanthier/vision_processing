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
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_srvs.srv import Trigger, TriggerResponse

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

        self.attached = False
        self._near_count = 0
        self._fork_pts = None
        self._fork_stamp = 0.0
        self._target_pts = None
        self._target_stamp = 0.0

        self._attach_srv = None
        self._detach_srv = None
        self._connect_attacher()

        rospy.Subscriber(self.fork_topic, PointCloud2, self._fork_cb, queue_size=1)
        rospy.Subscriber(self.target_topic, PointCloud2, self._target_cb,
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
            return True
        except rospy.ServiceException as e:
            rospy.logwarn("detach failed: %s", e)
            return False

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
        if self.attached or not self.auto_attach:
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
