#!/usr/bin/env python3
"""
Mock Face Detection for Simulation
====================================
In Gazebo, MediaPipe can't detect the mannequin's face.
So instead, this node reads the person's mouth_link TF directly
and publishes it on the same topic as the real face detection node.

This way, your trajectory follower works identically in sim and real —
it always listens to /face_detection/mouth_position.

Usage:
  rosrun vision_processing mock_face_detection.py

  Real robot:  run face_detection_ros.py     (uses webcam + MediaPipe)
  Simulation:  run mock_face_detection.py    (uses TF from person model)
"""
import rospy
import tf
import numpy as np
from geometry_msgs.msg import PointStamped


class MockFaceDetection:
    def __init__(self):
        rospy.init_node("mock_face_detection")

        self.tf_listener = tf.TransformListener()

        # The person model has tf_prefix="person", so the mouth frame is:
        self.mouth_frame = rospy.get_param("~mouth_frame", "person/mouth_link")

        # Publish on same topic as real face detection
        self.pub = rospy.Publisher(
            "/face_detection/mouth_position", PointStamped, queue_size=1
        )

        rate = rospy.get_param("~rate", 10)
        rospy.Timer(rospy.Duration(1.0 / rate), self.loop)
        rospy.loginfo(f"Mock face detection: reading TF '{self.mouth_frame}' -> publishing mouth position")

    def loop(self, event):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                "/world", self.mouth_frame, rospy.Time(0)
            )

            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world"  # already in world frame
            msg.point.x = trans[0]
            msg.point.y = trans[1]
            msg.point.z = trans[2]
            self.pub.publish(msg)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn_throttle(5.0, f"Waiting for TF: /world -> {self.mouth_frame}")


if __name__ == "__main__":
    MockFaceDetection()
    rospy.spin()