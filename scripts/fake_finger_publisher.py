#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState

def main():
    rospy.init_node('fake_finger_publisher')
    pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    rate = rospy.Rate(30)
    
    while not rospy.is_shutdown():
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['panda_finger_joint1', 'panda_finger_joint2']
        msg.position = [0.0, 0.0]
        msg.velocity = [0.0, 0.0]
        msg.effort = [0.0, 0.0]
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    main()
