#!/usr/bin/env python3
"""
Move fork_tip toward target centroid — one-shot, fixed orientation.

Strategy (same as homing.py):
  1. Wait for one /vision/target_centroid
  2. Compute IK ONCE for the goal pose (fork_tip at target, fixed vertical orientation)
  3. Cosine-interpolate in joint space from current → goal
  4. Hold at goal forever
"""

import rospy
import rospkg
import moveit_commander
import sys
import os
import math
import numpy as np
import tf
import tf.transformations as tft
from geometry_msgs.msg import PointStamped, Pose
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_msgs.msg import MoveItErrorCodes
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.join(rospkg.RosPack().get_path('vision_processing'), 'scripts'))
from utils import compute_T_child_parent_xacro


class MoveToTarget:

    def __init__(self):
        rospy.init_node('move_to_target', anonymous=True)

        # ── Fork transform ──
        rospack = rospkg.RosPack()
        xacro = os.path.join(rospack.get_path('vision_processing'), 'urdf', 'panda_camera.xacro')
        self.T_tcp_fork = compute_T_child_parent_xacro(xacro, 'fork_tip', 'panda_TCP')

        # ── Parameters ──
        self.z_offset = rospy.get_param('~z_offset', 0.05)
        self.duration = rospy.get_param('~duration', 4.0)  # seconds for the motion

        # Fixed vertical orientation: RPY = [180°, 0°, 0°] → quat [1, 0, 0, 0]
        self.fixed_quat = np.array(
            rospy.get_param('~orientation', [1.0, 0.0, 0.0, 0.0]),
            dtype=np.float64
        )
        self.fixed_quat /= np.linalg.norm(self.fixed_quat)

        # ── State ──
        self.current_joints = None

        # ── MoveIt IK only ──
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.wait_for_service('/compute_ik')
        self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)

        # ── Pub / Sub ──
        self.pub_joints = rospy.Publisher(
            '/joint_group_position_controller/command',
            Float64MultiArray, queue_size=1
        )
        rospy.Subscriber('/joint_states', JointState, self._joint_cb)

        rospy.loginfo(f"MoveToTarget ready. z_offset={self.z_offset}m "
                      f"duration={self.duration}s quat={self.fixed_quat}")

    def _joint_cb(self, msg):
        positions = []
        for name, pos in zip(msg.name, msg.position):
            if "panda_joint" in name:
                positions.append(pos)
        if len(positions) == 7:
            self.current_joints = np.array(positions)

    def _compute_ik(self, pose_tcp, seed, timeout=0.1):
        req = GetPositionIKRequest()
        req.ik_request.group_name = "panda_arm"
        req.ik_request.pose_stamped.header.frame_id = "world"
        req.ik_request.pose_stamped.pose = pose_tcp
        req.ik_request.avoid_collisions = False
        req.ik_request.ik_link_name = "panda_hand_tcp"
        req.ik_request.timeout = rospy.Duration(timeout)
        req.ik_request.robot_state.joint_state.name = [
            f"panda_joint{i+1}" for i in range(7)
        ]
        req.ik_request.robot_state.joint_state.position = seed.tolist()

        try:
            resp = self.ik_service(req)
            if resp.error_code.val == MoveItErrorCodes.SUCCESS:
                return np.array(list(resp.solution.joint_state.position)[:7])
            return None
        except Exception:
            return None

    def _fork_goal_to_tcp_pose(self, fork_pos):
        R_world_tcp = R.from_quat(self.fixed_quat).as_matrix()
        offset_world = R_world_tcp @ self.T_tcp_fork[:3, 3]
        tcp_pos = fork_pos - offset_world

        pose = Pose()
        pose.position.x = tcp_pos[0]
        pose.position.y = tcp_pos[1]
        pose.position.z = tcp_pos[2]
        pose.orientation.x = self.fixed_quat[0]
        pose.orientation.y = self.fixed_quat[1]
        pose.orientation.z = self.fixed_quat[2]
        pose.orientation.w = self.fixed_quat[3]
        return pose

    def move_to(self, goal_joints, duration):
        """Cosine-interpolate from current joints to goal, same as homing.py."""
        while self.current_joints is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        start_joints = self.current_joints.copy()
        start_time = rospy.Time.now().to_sec()
        rate = rospy.Rate(100)

        rospy.loginfo(f"Moving... duration={duration}s")

        while not rospy.is_shutdown():
            t = rospy.Time.now().to_sec() - start_time
            alpha = min(t / duration, 1.0)

            # Cosine smoothing (same as homing.py)
            smooth = (1.0 - math.cos(math.pi * alpha)) / 2.0
            cmd = (1 - smooth) * start_joints + smooth * goal_joints

            msg = Float64MultiArray()
            msg.data = cmd.tolist()
            self.pub_joints.publish(msg)

            if alpha >= 1.0:
                # Send exact final position
                msg.data = goal_joints.tolist()
                self.pub_joints.publish(msg)
                break

            rate.sleep()

        rospy.loginfo("Motion complete.")

    def run(self):
        rospy.sleep(1.0)

        # ── Wait for joints ──
        while self.current_joints is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Got joint states.")

        # ── Wait for one centroid ──
        rospy.loginfo("Waiting for /vision/target_centroid ...")
        msg = rospy.wait_for_message('/vision/target_centroid', PointStamped,
                                      timeout=30.0)
        target = np.array([msg.point.x, msg.point.y, msg.point.z])
        goal_fork = target.copy()
        goal_fork[2] += self.z_offset
        rospy.loginfo(f"Target: {target} → fork goal: {goal_fork}")

        # ── Compute IK once ──
        tcp_pose = self._fork_goal_to_tcp_pose(goal_fork)
        rospy.loginfo(f"TCP goal: [{tcp_pose.position.x:.3f}, "
                      f"{tcp_pose.position.y:.3f}, {tcp_pose.position.z:.3f}]")

        goal_joints = self._compute_ik(tcp_pose, seed=self.current_joints, timeout=1.0)
        if goal_joints is None:
            rospy.logerr("IK failed for target pose. Aborting.")
            return

        rospy.loginfo(f"IK solution: {np.round(goal_joints, 3)}")

        # ── Execute smooth motion ──
        self.move_to(goal_joints, self.duration)

        # ── Hold position ──
        rospy.loginfo("Holding position. Ctrl+C to stop.")
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            msg = Float64MultiArray()
            msg.data = goal_joints.tolist()
            self.pub_joints.publish(msg)
            rate.sleep()


if __name__ == '__main__':
    try:
        MoveToTarget().run()
    except rospy.ROSInterruptException:
        pass