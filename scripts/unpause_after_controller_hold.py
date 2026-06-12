#!/usr/bin/env python3
import time

import rospy
from controller_manager_msgs.srv import ListControllers
from gazebo_msgs.srv import GetWorldProperties
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class UnpauseAfterControllerHold:
    def __init__(self):
        rospy.init_node("unpause_after_controller_hold")

        self.controller_name = rospy.get_param(
            "~controller_name", "position_joint_trajectory_controller")
        self.command_topic = rospy.get_param(
            "~command_topic", f"/{self.controller_name}/command")
        self.command_type = rospy.get_param("~command_type", "trajectory")
        self.wait_timeout = float(rospy.get_param("~wait_timeout", 20.0))
        self.expected_models = set(rospy.get_param("~expected_models", []))
        self.model_settle_time = float(rospy.get_param("~model_settle_time", 1.5))
        self.publish_repeats = int(rospy.get_param("~publish_repeats", 5))
        self.publish_period = float(rospy.get_param("~publish_period", 0.05))
        self.hold_duration = float(rospy.get_param("~hold_duration", 2.0))
        self.lead_time = float(rospy.get_param("~lead_time", 0.25))
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

        if self.command_type == "trajectory":
            msg_type = JointTrajectory
        elif self.command_type == "group_position":
            msg_type = Float64MultiArray
        else:
            raise ValueError(
                f"Unsupported homing command_type: {self.command_type}")

        self.pub = rospy.Publisher(self.command_topic, msg_type, queue_size=1)

    def wait_for_models(self):
        if not self.expected_models:
            return True

        rospy.wait_for_service("/gazebo/get_world_properties", timeout=self.wait_timeout)
        get_world_properties = rospy.ServiceProxy(
            "/gazebo/get_world_properties", GetWorldProperties)

        deadline = time.time() + self.wait_timeout
        missing = set(self.expected_models)
        while not rospy.is_shutdown() and time.time() < deadline:
            response = get_world_properties()
            missing = self.expected_models.difference(response.model_names)
            if not missing:
                return True
            time.sleep(0.05)

        rospy.logerr("Timed out waiting for Gazebo models: %s",
                     ", ".join(sorted(missing)))
        return False

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
        if self.command_type == "group_position":
            return Float64MultiArray(data=list(self.homing_position))

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

    def publish_hold(self):
        for _ in range(max(self.publish_repeats, 1)):
            msg = self.hold_message()
            self.pub.publish(msg)
            time.sleep(self.publish_period)

    def unpause(self):
        rospy.wait_for_service("/gazebo/unpause_physics", timeout=self.wait_timeout)
        rospy.ServiceProxy("/gazebo/unpause_physics", Empty)()

    def pause(self):
        rospy.wait_for_service("/gazebo/pause_physics", timeout=self.wait_timeout)
        rospy.ServiceProxy("/gazebo/pause_physics", Empty)()

    def run(self):
        rospy.loginfo("Waiting for Gazebo models before unpausing physics.")
        if not self.wait_for_models():
            rospy.logerr("Leaving Gazebo paused because the scene is incomplete.")
            return

        if self.model_settle_time > 0.0:
            rospy.loginfo("Scene complete; waiting %.2f wall-seconds for spawn finalization.",
                          self.model_settle_time)
            time.sleep(self.model_settle_time)

        # gazebo_ros_control only processes controller state changes while the
        # physics update loop is running. All scene models are present here, so
        # it is safe to unpause before waiting for the controller.
        self.unpause()
        rospy.loginfo("Gazebo unpaused; waiting for %s to start.",
                      self.controller_name)
        if not self.wait_for_controller():
            rospy.logerr("Timed out waiting for %s; pausing Gazebo again.",
                         self.controller_name)
            self.pause()
            return

        rospy.loginfo("Publishing homing hold to %s.", self.command_topic)
        self.publish_hold()
        rospy.loginfo("Gazebo startup completed with the homing hold active.")


if __name__ == "__main__":
    try:
        UnpauseAfterControllerHold().run()
    except Exception as exc:
        rospy.logerr("Failed to unpause safely: %s", exc)
