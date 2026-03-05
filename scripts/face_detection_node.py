#!/usr/bin/env python3
"""
Face Detection ROS Node
========================
Publishes the 3D position of the user's mouth in a camera frame.

The trajectory follower will use TF to transform this into the world frame.

You need a static transform publisher for your webcam:
  rosrun tf static_transform_publisher X Y Z yaw pitch roll /world /webcam_frame 100

Where X, Y, Z, yaw, pitch, roll define where your webcam is in the world.
"""
import rospy
import cv2
import mediapipe as mp
import math
import numpy as np
from geometry_msgs.msg import PointStamped


class FaceDetectionNode:
    def __init__(self):
        rospy.init_node("face_detection_node")

        # --- Camera parameters ---
        # These are rough estimates without calibration.
        # For better accuracy, do a checkerboard calibration.
        self.camera_id = rospy.get_param("~camera_id", 0)
        self.frame_id = rospy.get_param("~camera_frame", "webcam_frame")
        self.REAL_EYE_DIST_M = 0.064  # Average eye distance in meters

        # Open camera to get resolution
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            rospy.logerr("Cannot open camera")
            return

        ret, frame = self.cap.read()
        if not ret:
            rospy.logerr("Cannot read from camera")
            return

        h, w, _ = frame.shape
        self.img_w = w
        self.img_h = h

        # Approximate intrinsics (no calibration)
        # fx ~ image width for a ~60° FOV webcam
        self.fx = w
        self.fy = w  # square pixels assumption
        self.cx = w / 2.0
        self.cy = h / 2.0

        rospy.loginfo(f"Camera: {w}x{h}, fx={self.fx:.0f}, frame={self.frame_id}")

        # MediaPipe face mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ROS publisher
        self.pub_mouth = rospy.Publisher(
            "/face_detection/mouth_position", PointStamped, queue_size=1
        )

        # Optional: show debug window
        self.show_debug = rospy.get_param("~show_debug", True)

        # Run at ~30 Hz
        rate = rospy.get_param("~rate", 30)
        rospy.Timer(rospy.Duration(1.0 / rate), self.loop)
        rospy.loginfo("Face detection node ready")

    def pixel_to_metric(self, u, v, eye_dist_px):
        """
        Pinhole camera model: (u, v, eye_distance_px) -> (X, Y, Z) in meters.

        Z = fx * real_eye_dist / eye_dist_px   (depth from similar triangles)
        X = (u - cx) * Z / fx                  (horizontal)
        Y = (v - cy) * Z / fy                  (vertical)
        """
        if eye_dist_px < 1.0:
            return None

        Z = (self.fx * self.REAL_EYE_DIST_M) / eye_dist_px
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        return (X, Y, Z)

    def loop(self, event):
        ret, image = self.cap.read()
        if not ret:
            return

        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return

        lm = results.multi_face_landmarks[0]
        w, h = self.img_w, self.img_h

        # Mouth center (between upper and lower lip landmarks)
        top_lip = lm.landmark[13]
        bot_lip = lm.landmark[14]
        mouth_u = int((top_lip.x + bot_lip.x) / 2.0 * w)
        mouth_v = int((top_lip.y + bot_lip.y) / 2.0 * h)

        # Eye corners for depth estimation
        eye_l = lm.landmark[33]   # left eye outer corner
        eye_r = lm.landmark[263]  # right eye outer corner
        eye_dist_px = math.sqrt(
            (eye_l.x * w - eye_r.x * w) ** 2 + (eye_l.y * h - eye_r.y * h) ** 2
        )

        coords = self.pixel_to_metric(mouth_u, mouth_v, eye_dist_px)
        if coords is None:
            return

        X, Y, Z = coords

        # Publish
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.point.x = X
        msg.point.y = Y
        msg.point.z = Z
        self.pub_mouth.publish(msg)

        # Debug visualization
        if self.show_debug:
            cv2.circle(image, (mouth_u, mouth_v), 5, (0, 255, 255), -1)
            cv2.putText(
                image,
                f"Mouth: X={X:.2f} Y={Y:.2f} Z={Z:.2f} m",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Face Detection", image)
            cv2.waitKey(1)

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    node = FaceDetectionNode()
    rospy.on_shutdown(node.cleanup)
    rospy.spin()