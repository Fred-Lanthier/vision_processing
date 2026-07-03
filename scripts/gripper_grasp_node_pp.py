#!/home/flanthier/Github/src/vision_processing/venv_sam3/bin/python3
"""
gripper_grasp_node_pp.py — close the gripper on the cube AND weld on contact (PICK-AND-PLACE).

Gazebo's contact solver does not hold an object by finger friction reliably, so a
pure franka_gripper grasp lets the cube slip. We therefore do a hybrid:

  1. Send a `franka_gripper/grasp` so the fingers physically CLOSE on the cube.
     Their real joint positions (franka_gripper/joint_states -> /joint_states) feed
     condition_pcd_pickplace, so the conditioning gripper cloud enters the trained
     "carrying" distribution (fingers rest at ~0.02 on the 0.04 cube).
  2. When the fingers have actually closed onto the cube (finger width drops to ~the
     cube width = CONTACT), WELD red_cube to panda_hand via gazebo_ros_link_attacher
     so the hold is rigid and never slips. Then cancel the force grasp and command
     a visual hold at the cube width, so the fingers do not keep closing through
     the cube. (panda_hand is kept a standalone Gazebo link in panda_pickplace.xacro
     so the attacher can find it.)

Release opens the fingers and detaches the weld.

Trigger: GRASP when the cube's top passes the TCP at the bottom of the descent.
Release: when the carried cube is over the box and descended to its top.
Manual: /pp_grasp/grab, /pp_grasp/release (std_srvs/Trigger).
"""
import numpy as np
import rospy
import tf2_ros
import actionlib
from sensor_msgs.msg import PointCloud2, JointState
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

try:
    from franka_gripper.msg import (GraspAction, GraspGoal, GraspEpsilon,
                                    MoveAction, MoveGoal)
except ImportError:
    GraspAction = None

try:
    from gazebo_ros_link_attacher.srv import Attach, AttachRequest
except ImportError:
    Attach = None
    AttachRequest = None


def _cloud_to_np(msg):
    pts = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
                   dtype=np.float32)
    return pts if len(pts) else None


class GripperGraspNodePP:
    def __init__(self):
        rospy.init_node("gripper_grasp_node_pp")

        self.robot_model = rospy.get_param("~robot_model", "panda")
        self.gripper_link = rospy.get_param("~gripper_link", "panda_hand")
        self.cube_model = rospy.get_param("~cube_model", "red_cube")
        self.cube_link = rospy.get_param("~cube_link", "base_link")

        self.cube_topic = rospy.get_param("~cube_topic", "/perception/target_cube")
        self.box_topic = rospy.get_param("~box_topic", "/perception/target_box")
        self.tcp_frame = rospy.get_param("~tcp_frame", "panda_TCP")
        self.world_frame = rospy.get_param("~world_frame", "world")

        # franka_gripper action namespace (FrankaGripperSim controller).
        self.gripper_ns = rospy.get_param("~gripper_ns", "/franka_gripper")
        self.grasp_width = float(rospy.get_param("~grasp_width", 0.04))    # cube full width (m)
        self.grasp_epsilon_inner = float(rospy.get_param("~grasp_epsilon_inner", 0.02))
        self.grasp_epsilon_outer = float(rospy.get_param("~grasp_epsilon_outer", 0.02))
        self.grasp_speed = float(rospy.get_param("~grasp_speed", 0.1))     # m/s
        self.grasp_force = float(rospy.get_param("~grasp_force", 40.0))    # N
        self.open_width = float(rospy.get_param("~open_width", 0.08))      # fully open (m)
        self.hold_after_weld = bool(rospy.get_param("~hold_after_weld", True))
        self.hold_width = float(rospy.get_param("~hold_width", self.grasp_width))
        self.hold_speed = float(rospy.get_param("~hold_speed", min(self.grasp_speed, 0.05)))

        # Weld-on-contact: weld once the fingers have closed in onto the cube,
        # i.e. each finger joint <= this (open is 0.04; on the 0.04 cube ~0.02).
        # A short settle timeout welds anyway so we never get stuck mid-close.
        default_contact_finger = 0.5 * self.hold_width + 0.004
        self.contact_weld_finger = float(rospy.get_param("~contact_weld_finger",
                                                         default_contact_finger))
        self.grasp_settle_timeout = float(rospy.get_param("~grasp_settle_timeout", 1.0))

        self.cloud_timeout = float(rospy.get_param("~cloud_timeout", 1.0))
        self.confirm_cycles = int(rospy.get_param("~confirm_cycles", 2))
        self.release_xy_radius = float(rospy.get_param("~release_xy_radius", 0.10))
        self.release_z_clearance = float(rospy.get_param("~release_z_clearance", 0.10))
        self.release_min_grasp_time = float(rospy.get_param("~release_min_grasp_time", 1.0))
        self.auto = bool(rospy.get_param("~auto", True))
        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))

        self.cube_pts = None; self.cube_stamp = 0.0
        self.box_pts = None;  self.box_stamp = 0.0
        self.finger_width = None           # latest panda_finger_joint1 position (m), per finger
        self.state = "PRE_GRASP"           # PRE_GRASP -> CLOSING -> GRASPED -> RELEASED
        self.grasp_count = 0; self.release_count = 0
        self.close_time = 0.0; self.grasp_time = 0.0
        self.prev_tcp_z = None             # to detect the bottom of the descent
        self.descent_eps = float(rospy.get_param("~descent_eps", 0.0004))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self._grasp_client = None; self._move_client = None
        self._attach_srv = None; self._detach_srv = None
        self._connect_gripper()
        self._connect_attacher()

        # Latched grasp state (PRE_GRASP/CLOSING/GRASPED/RELEASED): lets the
        # conditioning node attach the cube cloud to the hand while carried.
        self.pub_state = rospy.Publisher('/pp_grasp/state', String,
                                         queue_size=1, latch=True)
        self.pub_state.publish(String(data=self.state))

        rospy.Subscriber(self.cube_topic, PointCloud2, self._cube_cb, queue_size=1)
        rospy.Subscriber(self.box_topic, PointCloud2, self._box_cb, queue_size=1)
        rospy.Subscriber('/joint_states', JointState, self._js_cb, queue_size=1)
        rospy.Service("/pp_grasp/grab", Trigger, lambda r: self._srv(self.start_grasp))
        rospy.Service("/pp_grasp/release", Trigger, lambda r: self._srv(self.release))

        rospy.loginfo("gripper_grasp_node_pp ready: close on %s, weld near finger=%.3f, "
                      "then hold width=%.3f. force=%.1fN auto=%s.",
                      self.cube_model, self.contact_weld_finger, self.hold_width,
                      self.grasp_force, self.auto)
        rospy.Timer(rospy.Duration(1.0 / max(self.rate_hz, 1.0)), self._tick)

    # -- franka_gripper action clients --------------------------------------
    def _connect_gripper(self):
        if GraspAction is None:
            rospy.logerr("franka_gripper messages not importable — build/source franka_gripper.")
            return
        try:
            self._grasp_client = actionlib.SimpleActionClient(
                self.gripper_ns + "/grasp", GraspAction)
            self._move_client = actionlib.SimpleActionClient(
                self.gripper_ns + "/move", MoveAction)
            if self._grasp_client.wait_for_server(rospy.Duration(10.0)):
                rospy.loginfo("Connected to %s grasp/move action servers.", self.gripper_ns)
            else:
                rospy.logwarn("%s/grasp not up yet — retrying lazily.", self.gripper_ns)
        except Exception as e:
            rospy.logwarn("franka_gripper connect failed: %s", e)

    # -- link attacher (the reliable hold) ----------------------------------
    def _connect_attacher(self):
        if Attach is None:
            rospy.logerr("gazebo_ros_link_attacher not importable — build it in the workspace.")
            return
        try:
            rospy.wait_for_service("/link_attacher_node/attach", timeout=10.0)
            self._attach_srv = rospy.ServiceProxy("/link_attacher_node/attach", Attach)
            self._detach_srv = rospy.ServiceProxy("/link_attacher_node/detach", Attach)
            rospy.loginfo("Connected to /link_attacher_node attach/detach.")
        except rospy.ROSException:
            rospy.logwarn("/link_attacher_node not up yet — retrying lazily on first weld.")

    def _attach_request(self):
        req = AttachRequest()
        req.model_name_1 = self.robot_model; req.link_name_1 = self.gripper_link
        req.model_name_2 = self.cube_model;  req.link_name_2 = self.cube_link
        return req

    # -- grasp / weld / release ---------------------------------------------
    def _set_state(self, state):
        self.state = state
        self.pub_state.publish(String(data=state))

    def start_grasp(self):
        """Begin closing the gripper on the cube; weld happens on contact."""
        if self.state in ("CLOSING", "GRASPED"):
            return True
        if self._grasp_client is None:
            self._connect_gripper()
        if self._grasp_client is None:
            return False
        goal = GraspGoal()
        goal.width = self.grasp_width
        goal.epsilon = GraspEpsilon(inner=self.grasp_epsilon_inner,
                                    outer=self.grasp_epsilon_outer)
        goal.speed = self.grasp_speed
        goal.force = self.grasp_force
        self._grasp_client.send_goal(goal)   # non-blocking
        self._set_state("CLOSING"); self.close_time = rospy.get_time()
        rospy.loginfo("✊ Closing gripper on %s (width=%.3f force=%.1fN) — weld on contact.",
                      self.cube_model, self.grasp_width, self.grasp_force)
        return True

    def _hold_gripper_width(self):
        """Stop the force grasp and keep the fingers at the visual cube width."""
        if not self.hold_after_weld:
            return True
        if self._move_client is None or self._grasp_client is None:
            self._connect_gripper()
        if self._move_client is None:
            return False
        try:
            if self._grasp_client is not None:
                self._grasp_client.cancel_all_goals()
            self._move_client.send_goal(MoveGoal(width=self.hold_width, speed=self.hold_speed))
            rospy.loginfo("✋ Holding gripper at width=%.3f after weld.", self.hold_width)
            return True
        except Exception as e:
            rospy.logwarn("hold-width command failed: %s", e)
            return False

    def _weld(self):
        """Rigidly weld the cube to the hand (reliable hold once contacted)."""
        if self._attach_srv is None:
            self._connect_attacher()
        if self._attach_srv is None:
            return False
        try:
            self._attach_srv(self._attach_request())
            self._set_state("GRASPED"); self.grasp_time = rospy.get_time()
            self._hold_gripper_width()
            rospy.loginfo("🤝 Contact — welded %s to %s/%s (finger=%.4f).",
                          self.cube_model, self.robot_model, self.gripper_link,
                          self.finger_width if self.finger_width is not None else -1.0)
            return True
        except rospy.ServiceException as e:
            rospy.logwarn("weld failed: %s", e); return False

    def release(self):
        """Detach the weld and open the gripper over the box."""
        if self.state not in ("GRASPED", "CLOSING"):
            return True
        if self._detach_srv is not None and self.state == "GRASPED":
            try:
                self._detach_srv(self._attach_request())
            except rospy.ServiceException as e:
                rospy.logwarn("detach failed: %s", e)
        if self._move_client is not None:
            self._move_client.send_goal(MoveGoal(width=self.open_width, speed=self.grasp_speed))
        self._set_state("RELEASED")
        rospy.loginfo("👐 Released %s: detached + opened gripper to width=%.3f.",
                      self.cube_model, self.open_width)
        return True

    def _srv(self, fn):
        ok = fn()
        return TriggerResponse(success=ok, message="ok" if ok else "failed")

    # -- callbacks -----------------------------------------------------------
    def _cube_cb(self, m): self.cube_pts = _cloud_to_np(m); self.cube_stamp = rospy.get_time()
    def _box_cb(self, m):  self.box_pts = _cloud_to_np(m);  self.box_stamp = rospy.get_time()

    def _js_cb(self, m):
        idx = {n: i for i, n in enumerate(m.name)}
        if 'panda_finger_joint1' in idx:
            self.finger_width = float(m.position[idx['panda_finger_joint1']])

    def _fresh(self, pts, stamp):
        return pts is not None and (rospy.get_time() - stamp) <= self.cloud_timeout

    def _tcp_z(self):
        try:
            tr = self.tf_buffer.lookup_transform(self.world_frame, self.tcp_frame,
                                                 rospy.Time(0), rospy.Duration(0.05))
            return float(tr.transform.translation.z)
        except Exception:
            return None

    # -- main loop -----------------------------------------------------------
    def _tick(self, _evt):
        if not self.auto or self.state == "RELEASED":
            return

        if self.state == "PRE_GRASP":
            tcp_z = self._tcp_z()
            if tcp_z is None or not self._fresh(self.cube_pts, self.cube_stamp):
                self.grasp_count = 0; self.prev_tcp_z = tcp_z; return
            cube_top_z = float(np.percentile(self.cube_pts[:, 2], 90))
            descending = self.prev_tcp_z is not None and (tcp_z < self.prev_tcp_z - self.descent_eps)
            self.prev_tcp_z = tcp_z
            # Close only at the BOTTOM of the descent: the cube's top has passed
            # the TCP AND the gripper has stopped going down (about to lift).
            if tcp_z <= cube_top_z and not descending:
                self.grasp_count += 1
                if self.grasp_count >= self.confirm_cycles:
                    self.start_grasp()
            else:
                self.grasp_count = 0

        elif self.state == "CLOSING":
            # Weld once the fingers have closed onto the cube (contact), or after a
            # short settle timeout (cube is already at the TCP by the trigger).
            contacted = (self.finger_width is not None
                         and self.finger_width <= self.contact_weld_finger)
            if contacted or (rospy.get_time() - self.close_time) > self.grasp_settle_timeout:
                self._weld()

        elif self.state == "GRASPED":
            if rospy.get_time() - self.grasp_time < self.release_min_grasp_time:
                return
            if not (self._fresh(self.cube_pts, self.cube_stamp)
                    and self._fresh(self.box_pts, self.box_stamp)):
                self.release_count = 0; return
            cube_xy = self.cube_pts[:, :2].mean(0)
            cube_bottom_z = float(np.percentile(self.cube_pts[:, 2], 10))
            box_xy = self.box_pts[:, :2].mean(0)
            box_top_z = float(np.percentile(self.box_pts[:, 2], 90))
            if (np.linalg.norm(cube_xy - box_xy) < self.release_xy_radius
                    and cube_bottom_z <= box_top_z + self.release_z_clearance):
                self.release_count += 1
                if self.release_count >= self.confirm_cycles:
                    self.release()
            else:
                self.release_count = 0


if __name__ == "__main__":
    GripperGraspNodePP()
    rospy.spin()
