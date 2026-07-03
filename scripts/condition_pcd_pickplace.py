#!/home/flanthier/Github/src/vision_processing/venv_sam3/bin/python3
"""
condition_pcd_pickplace.py
==========================
PICK-AND-PLACE conditioning node. Builds the merged point cloud the Flow-Matching
pick-and-place policy was trained on:  GRIPPER (panda_hand) + RED CUBE + BROWN BOX,
in WORLD frame (the planner translation-centers it on the TCP, exactly like the
fork pipeline -- see casf_generative_node_PC.py).

This is the pick-and-place analogue of condition_pcd_from_perception.py. Differences:
  * tool = the panda_hand GRIPPER, not the fork. The gripper cloud is the SAME
    parametric geometry used to build the training data
    (diffusion_model_train/Data_preprocess_PickPlace.py:build_tool_cloud_tcp),
    placed at the live TCP pose (TF world->panda_TCP) with the live finger opening.
  * TWO targets: the pick cube (/perception/target_cube) and the place box
    (/perception/target_box), both already in world frame from the projector.

Publishes:
  /vision/merged_cloud  (PointCloud2, world)  -> planner conditioning input
  /vision/gripper_cloud  (PointCloud2, world) -> grasp/contact node (tool cloud)

The feeding system is untouched (separate node, separate topics).
"""
import os
import sys
import numpy as np
import rospy
import rospkg
import tf2_ros
from sensor_msgs.msg import PointCloud2, JointState
from std_msgs.msg import Header, Float32, String
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation as R

# ---- Gripper geometry (panda_hand), TCP frame. Mirrors build_tool_cloud_tcp. ----
TCP_FROM_HAND_Z = 0.1034
FINGER_BASE_Z_HAND = 0.0584
FINGER_LEN = 0.054
FINGER_HALF_X = 0.012
FINGER_THICK_Y = 0.012
HAND_HALF_X = 0.020
HAND_HALF_Y = 0.045
HAND_BACK_Z_HAND = -0.010


def _sample_box_surface(half, center, n, rng):
    half = np.asarray(half, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    areas = np.array([half[1] * half[2], half[0] * half[2], half[0] * half[1]])
    areas = areas / areas.sum()
    pts = np.empty((n, 3))
    for i in range(n):
        ax = rng.choice(3, p=areas)
        p = (rng.random(3) * 2 - 1) * half
        p[ax] = half[ax] * (1 if rng.random() < 0.5 else -1)
        pts[i] = p + center
    return pts


def build_gripper_tcp(q_left, q_right, n_body, n_finger, rng):
    """panda_hand gripper cloud in the TCP frame (z=approach, +-y=fingers)."""
    def to_tcp_z(z_hand):
        return z_hand - TCP_FROM_HAND_Z

    body_zc = 0.5 * (HAND_BACK_Z_HAND + FINGER_BASE_Z_HAND)
    body_half = [HAND_HALF_X, HAND_HALF_Y, 0.5 * (FINGER_BASE_Z_HAND - HAND_BACK_Z_HAND)]
    body = _sample_box_surface(body_half, [0.0, 0.0, to_tcp_z(body_zc)], n_body, rng)

    fz_center = FINGER_BASE_Z_HAND + 0.5 * FINGER_LEN
    finger_half = [FINGER_HALF_X, 0.5 * FINGER_THICK_Y, 0.5 * FINGER_LEN]
    left = _sample_box_surface(finger_half, [0.0, q_left + 0.5 * FINGER_THICK_Y, to_tcp_z(fz_center)],
                               n_finger, rng)
    right = _sample_box_surface(finger_half, [0.0, -(q_right + 0.5 * FINGER_THICK_Y), to_tcp_z(fz_center)],
                                n_finger, rng)
    return np.vstack([body, left, right]).astype(np.float32)


class ConditionPcdPickPlace:
    def __init__(self):
        rospy.init_node('condition_pcd_pickplace', anonymous=True)

        self.num_points = int(rospy.get_param("~num_points", 1024))
        # Fractions of the merged cloud (gripper / cube / box). Box = remainder.
        # Loosely matches the Data_preprocess_PickPlace sampling (tool-heavy, box > cube).
        self.gripper_fraction = float(rospy.get_param("~gripper_fraction", 0.40))
        self.cube_fraction = float(rospy.get_param("~cube_fraction", 0.20))
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.tcp_frame = rospy.get_param("~tcp_frame", "panda_TCP")
        self.target_timeout = float(rospy.get_param("~target_timeout", 5.0))
        self.hold_on_loss = bool(rospy.get_param("~hold_last_target_on_loss", True))
        # Cube cloud source: SAM projector (default) or the ground-truth
        # cloud from gt_cube_cloud_node.py (use_gt_cube in launch).
        self.cube_topic = rospy.get_param("~cube_topic", "/perception/target_cube")
        # While the cube is grasped (state from gripper_grasp_node_pp), freeze
        # its cloud in the TCP frame and move it with the hand: matches training,
        # where the GT cube cloud follows the gripper during the carry, whereas
        # live SAM either loses the occluded cube (stale persistent cloud at the
        # pick location) or sees only slivers of it.
        self.follow_cube_on_grasp = bool(rospy.get_param("~follow_cube_on_grasp", True))
        self.grasp_state = "PRE_GRASP"
        self.cube_tcp_snapshot = None

        self.rng = np.random.default_rng(0)
        self.cube_cloud = None;  self.cube_stamp = None
        self.box_cloud = None;   self.box_stamp = None
        self.q_left = 0.04;      self.q_right = 0.04

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub_merged = rospy.Publisher('/vision/merged_cloud', PointCloud2, queue_size=1)
        self.pub_gripper = rospy.Publisher('/vision/gripper_cloud', PointCloud2, queue_size=1)
        self.pub_dist = rospy.Publisher('/vision/gripper_cube_distance', Float32, queue_size=1)

        rospy.Subscriber(self.cube_topic, PointCloud2, self._cube_cb)
        rospy.Subscriber('/perception/target_box', PointCloud2, self._box_cb)
        rospy.Subscriber('/joint_states', JointState, self._joint_cb)
        rospy.Subscriber('/pp_grasp/state', String, self._grasp_state_cb)

        self.rate = rospy.Rate(30)
        rospy.loginfo("🚀 Condition PCD PickPlace (gripper + cube + box) ready "
                      f"[cube <- {self.cube_topic}]")

    # -- callbacks --------------------------------------------------------- #
    def _read_xyz(self, msg):
        pts = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, int(msg.point_step / 4))
        return pts[:, :3].copy()

    def _cube_cb(self, msg):
        try:
            p = self._read_xyz(msg)
            if len(p) > 0:
                self.cube_cloud = p; self.cube_stamp = rospy.get_time()
        except Exception as e:
            rospy.logerr_throttle(5, f"cube cloud: {e}")

    def _box_cb(self, msg):
        try:
            p = self._read_xyz(msg)
            if len(p) > 0:
                self.box_cloud = p; self.box_stamp = rospy.get_time()
        except Exception as e:
            rospy.logerr_throttle(5, f"box cloud: {e}")

    def _grasp_state_cb(self, msg):
        self.grasp_state = msg.data
        if msg.data not in ("CLOSING", "GRASPED"):
            self.cube_tcp_snapshot = None  # back to live perception

    def _joint_cb(self, msg):
        d = {n: p for n, p in zip(msg.name, msg.position)}
        if 'panda_finger_joint1' in d:
            self.q_left = float(d['panda_finger_joint1'])
        if 'panda_finger_joint2' in d:
            self.q_right = float(d['panda_finger_joint2'])

    # -- helpers ----------------------------------------------------------- #
    def stable_sample(self, points, count):
        if points is None or points.shape[0] == 0 or count <= 0:
            return np.empty((0, 3), dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32)
        if pts.shape[0] > count:
            order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
            ordered = pts[order]
            idx = np.linspace(0, ordered.shape[0] - 1, count, dtype=np.int64)
            return ordered[idx]
        if pts.shape[0] < count:
            return pts[np.arange(count, dtype=np.int64) % pts.shape[0]]
        return pts

    def gripper_world(self, n_body, n_finger):
        """Gripper cloud transformed into world frame via TF world<-panda_TCP."""
        try:
            tr = self.tf_buffer.lookup_transform(self.world_frame, self.tcp_frame,
                                                 rospy.Time(0), rospy.Duration(0.05))
        except Exception:
            return None
        t = tr.transform.translation
        q = tr.transform.rotation
        Rm = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        p = np.array([t.x, t.y, t.z])
        self._tcp_Rm, self._tcp_p = Rm, p  # reused for the carried cube cloud
        n_f = max(1, n_finger // 2)
        g_tcp = build_gripper_tcp(self.q_left, self.q_right, n_body, n_f, self.rng)
        return (g_tcp @ Rm.T + p).astype(np.float32)

    def _fresh(self, stamp):
        return stamp is not None and (self.target_timeout <= 0.0
                                      or (rospy.get_time() - stamp) <= self.target_timeout)

    def publish(self, pub, points):
        h = Header(); h.stamp = rospy.Time.now(); h.frame_id = self.world_frame
        pub.publish(pc2.create_cloud_xyz32(h, points if points is not None and len(points) else np.empty((0, 3))))

    # -- main loop --------------------------------------------------------- #
    def run(self):
        n_grip = int(self.num_points * self.gripper_fraction)
        n_cube = int(self.num_points * self.cube_fraction)
        n_box = self.num_points - n_grip - n_cube
        n_body = int(n_grip * 0.6); n_finger = n_grip - n_body

        while not rospy.is_shutdown():
            grip = self.gripper_world(n_body, n_finger)
            if grip is not None:
                self.publish(self.pub_gripper, grip)

            # Carried cube: freeze the last cloud in the TCP frame at grasp time
            # and move it with the hand until release (mirrors training GT).
            cube_used = self.cube_cloud
            if (self.follow_cube_on_grasp and grip is not None
                    and self.grasp_state in ("CLOSING", "GRASPED")):
                if self.cube_tcp_snapshot is None and self.cube_cloud is not None:
                    self.cube_tcp_snapshot = \
                        (self.cube_cloud - self._tcp_p) @ self._tcp_Rm
                if self.cube_tcp_snapshot is not None:
                    cube_used = (self.cube_tcp_snapshot @ self._tcp_Rm.T
                                 + self._tcp_p).astype(np.float32)

            cube_ok = cube_used is not None and (self._fresh(self.cube_stamp)
                                                 or self.hold_on_loss
                                                 or self.cube_tcp_snapshot is not None)
            box_ok = self.box_cloud is not None and (self._fresh(self.box_stamp) or self.hold_on_loss)

            if grip is not None and cube_ok and box_ok:
                merged = [self.stable_sample(grip, n_grip),
                          self.stable_sample(cube_used, n_cube),
                          self.stable_sample(self.box_cloud, n_box)]
                full = self.stable_sample(np.vstack(merged), self.num_points)
                self.publish(self.pub_merged, full)

                dmin = float(np.min(np.linalg.norm(cube_used - grip.mean(axis=0), axis=1)))
                self.pub_dist.publish(Float32(data=dmin))
            elif grip is None:
                rospy.loginfo_throttle(10, "Waiting for TF world->TCP (gripper cloud)...")
            else:
                rospy.loginfo_throttle(10, "Waiting for cube + box detections...")
                self.publish(self.pub_merged, np.empty((0, 3)))

            self.rate.sleep()


if __name__ == '__main__':
    ConditionPcdPickPlace().run()
