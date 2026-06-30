#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


class FakeFingerPublisher:
    def __init__(self):
        rospy.init_node('fake_finger_publisher')
        # Fixed finger opening (m per finger). Default 0.0 = feeding behaviour.
        # Pick-and-place sets 0.04 (wide open) so the gripper conditioning cloud
        # matches the training APPROACH phase and the hand stays visibly open.
        self.finger_pos = float(rospy.get_param('~finger_position', 0.0))
        # Pick-and-place only: the grasp node commands the CONDITIONING finger
        # width here (0.04 open -> ~0.0237 closed at grasp) so the merged cloud
        # enters the trained "carrying" distribution and the policy lifts. The
        # fingers are not physically actuated (the cube is welded), so this is a
        # conditioning/visualization signal only. Feeding never publishes this
        # topic, so its behaviour is unchanged.
        self.pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        rospy.Subscriber('/pp_grasp/finger_position', Float64, self._cmd_cb, queue_size=1)

    def _cmd_cb(self, msg):
        self.finger_pos = float(msg.data)

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = ['panda_finger_joint1', 'panda_finger_joint2']
            msg.position = [self.finger_pos, self.finger_pos]
            msg.velocity = [0.0, 0.0]
            msg.effort = [0.0, 0.0]
            self.pub.publish(msg)
            rate.sleep()


if __name__ == '__main__':
    FakeFingerPublisher().run()
