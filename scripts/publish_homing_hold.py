#!/usr/bin/env python3
import time

import rospy
from controller_manager_msgs.srv import ListControllers
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class HomingHoldPublisher:
    def __init__(self):
        rospy.init_node("publish_homing_hold")

        self.controller_name = rospy.get_param(
            "~controller_name", "position_joint_trajectory_controller")
        self.command_topic = rospy.get_param(
            "~command_topic", f"/{self.controller_name}/command")
        self.wait_timeout = float(rospy.get_param("~wait_timeout", 20.0))
        self.publish_repeats = int(rospy.get_param("~publish_repeats", 20))
        self.publish_period = float(rospy.get_param("~publish_period", 0.05))
        self.hold_duration = float(rospy.get_param("~hold_duration", 2.0))
        self.lead_time = float(rospy.get_param("~lead_time", 0.10))
        self.joint_names = rospy.get_param("~joint_names", [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ])
        self.homing_position = rospy.get_param("~homing_position", [
            -0.000059,
            -0.125928,
            0.000117,
            -2.193312,
            -0.000251,
            2.064780,
            0.785511,
        ])

        self.pub = rospy.Publisher(self.command_topic, JointTrajectory, queue_size=1)

    def wait_for_controller(self):
        rospy.wait_for_service("/controller_manager/list_controllers", timeout=self.wait_timeout)
        list_controllers = rospy.ServiceProxy(
            "/controller_manager/list_controllers", ListControllers)

        deadline = time.time() + self.wait_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            controllers = list_controllers().controller
            for controller in controllers:
                if controller.name == self.controller_name and controller.state == "running":
                    return True
            time.sleep(0.05)
        return False

    def hold_message(self):
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now() + rospy.Duration(self.lead_time)
        msg.joint_names = list(self.joint_names)

        p0 = JointTrajectoryPoint()
        p0.positions = list(self.homing_position)
        p0.time_from_start = rospy.Duration(0.0)
        msg.points.append(p0)

        p1 = JointTrajectoryPoint()
        p1.positions = list(self.homing_position)
        p1.time_from_start = rospy.Duration(self.hold_duration)
        msg.points.append(p1)
        return msg

    def run(self):
        rospy.loginfo("Waiting for %s before publishing homing hold.",
                      self.controller_name)
        if not self.wait_for_controller():
            rospy.logerr("Timed out waiting for %s; homing hold was not published.",
                         self.controller_name)
            return

        rospy.loginfo("Publishing homing hold to %s.", self.command_topic)
        for _ in range(max(self.publish_repeats, 1)):
            msg = self.hold_message()
            self.pub.publish(msg)
            time.sleep(self.publish_period)
        rospy.loginfo("Initial homing hold command published.")


if __name__ == "__main__":
    try:
        HomingHoldPublisher().run()
    except Exception as exc:
        rospy.logerr("Failed to publish homing hold: %s", exc)
