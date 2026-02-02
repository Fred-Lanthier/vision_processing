#!/usr/bin/env python3
"""
Condition PCD Node (SAM3 Client Version)

This version uses the SAM3 server instead of loading the model directly.
Startup is now instant since SAM3 is already loaded in the server.

Requirements:
    1. Start SAM3 server: rosrun vision_processing sam3_server.py
    2. Then run this:    rosrun vision_processing condition_pcd_node.py
"""
import rospy
import numpy as np
import message_filters
from sensor_msgs.msg import Image, PointCloud2, JointState, CameraInfo
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Header
import sensor_msgs.point_cloud2 as pc2
import tf
import tf.transformations as tft
import sys
import os
import rospkg
import fpsample

# Import SAM3 client (instead of loading model)
from sam3_client import Sam3Client

rospack = rospkg.RosPack()
sys.path.append(os.path.join(rospack.get_path('vision_processing'), 'scripts'))

try:
    from Compute_3D_point_cloud_from_mesh import RobotMeshLoaderOptimized
    LOADER_AVAILABLE = True
except ImportError:
    LOADER_AVAILABLE = False


class MergedCloudNode:
    def __init__(self):
        rospy.init_node('merged_cloud_node', anonymous=True)
        
        # Detection parameters
        self.target_object = "cube"
        self.detection_confidence = 0.20
        
        # State
        self.cube_locked = False
        self.is_grasped = False
        self.static_cube_cloud = None
        self.current_cube_cloud = None
        self.contact_threshold = 0.01

        self.fork_tip_offset_tcp = np.array([-0.0055, 0.0, 0.1296, 1.0])
        self.T_world_tcp_at_grasp = None
        self.T_tcp_at_grasp_inv = None

        # --- SAM3 CLIENT (connects to server) ---
        self.sam3 = Sam3Client()  # This waits for server automatically
        
        # Robot Mesh Loader
        self.mesh_loader = None
        if LOADER_AVAILABLE:
            urdf_path = os.path.join(rospack.get_path('vision_processing'), 'urdf', 'panda_camera.xacro')
            try:
                self.mesh_loader = RobotMeshLoaderOptimized(urdf_path)
            except:
                pass

        # Publishers
        self.pub_merged = rospy.Publisher('/vision/merged_cloud', PointCloud2, queue_size=1)
        self.pub_marker = rospy.Publisher('/vision/marker', Marker, queue_size=1)
        self.pub_fork_tip = rospy.Publisher('/vision/debug_fork_tip', PointStamped, queue_size=1)
        self.pub_debug_cam_axis = rospy.Publisher('/debug/camera_axis', Marker, queue_size=1)
        self.pub_debug_tcp_axis = rospy.Publisher('/debug/tcp_z_axis', Marker, queue_size=1)

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = 604.9, 604.9, 320.0, 240.0
        self.sub_info = rospy.Subscriber("/camera_wrist/color/camera_info", CameraInfo, self.cam_info_cb)

        # TF
        self.tf_listener = tf.TransformListener()

        # Synchronized subscribers
        sub_rgb = message_filters.Subscriber("/synced/camera_wrist/rgb", Image)
        sub_depth = message_filters.Subscriber("/synced/camera_wrist/depth", Image)
        sub_joints = message_filters.Subscriber("/synced/joint_states", JointState)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth, sub_joints], queue_size=5, slop=0.1
        )
        self.ts.registerCallback(self.callback)
        
        rospy.loginfo("🚀 Condition PCD Node Ready (using SAM3 server)")

    def cam_info_cb(self, msg):
        self.fx = msg.K[0]
        self.cx = msg.K[2]
        self.fy = msg.K[4]
        self.cy = msg.K[5]
        self.sub_info.unregister()

    def imgmsg_to_numpy(self, msg):
        if msg.encoding == "rgb8":
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        elif "32FC1" in msg.encoding:
            return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        return None

    def publish_cloud(self, points, frame_id="world"):
        if points is None or len(points) == 0:
            return
        header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
        cloud_msg = pc2.create_cloud_xyz32(header, points)
        self.pub_merged.publish(cloud_msg)

    def publish_debug_cam_axis(self, T_world_cam, frame_id="world"):
        start = tft.translation_from_matrix(T_world_cam)
        end = np.dot(T_world_cam, np.array([0, 0, 0.5, 1.0]))
        
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = "camera_axis"
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.points = [Point(*start), Point(*end[:3])]
        m.scale.x, m.scale.y, m.scale.z = 0.01, 0.02, 0.05
        m.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)
        self.pub_debug_cam_axis.publish(m)

    def publish_debug_tcp_axis(self, T_world_tcp, frame_id="world"):
        start = tft.translation_from_matrix(T_world_tcp)
        end = np.dot(T_world_tcp, np.array([0, 0, 0.2, 1.0]))
        
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = "tcp_axis"
        m.id = 1
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.points = [Point(*start), Point(*end[:3])]
        m.scale.x, m.scale.y, m.scale.z = 0.01, 0.02, 0.05
        m.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)
        self.pub_debug_tcp_axis.publish(m)

    def callback(self, rgb_msg, depth_msg, joint_msg):
        target_time = joint_msg.header.stamp
        
        # --- TF ---
        try:
            self.tf_listener.waitForTransform("world", "camera_wrist_optical_frame", target_time, rospy.Duration(0.1))
            (trans_cam, rot_cam) = self.tf_listener.lookupTransform("world", "camera_wrist_optical_frame", target_time)
            
            self.tf_listener.waitForTransform("world", "panda_TCP", target_time, rospy.Duration(0.1))
            (trans_tcp, rot_tcp) = self.tf_listener.lookupTransform("world", "panda_TCP", target_time)

            self.tf_listener.waitForTransform("world", "panda_hand_tcp", target_time, rospy.Duration(0.1))
            (trans_hand, rot_hand) = self.tf_listener.lookupTransform("world", "panda_hand_tcp", target_time)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        T_world_cam = tft.compose_matrix(translate=trans_cam, angles=tft.euler_from_quaternion(rot_cam))
        T_world_tcp = tft.compose_matrix(translate=trans_tcp, angles=tft.euler_from_quaternion(rot_tcp))
        T_world_hand = tft.compose_matrix(translate=trans_hand, angles=tft.euler_from_quaternion(rot_hand))

        self.publish_debug_cam_axis(T_world_cam)
        self.publish_debug_tcp_axis(T_world_hand)

        # --- ROBOT CLOUD ---
        current_robot_points = None
        if self.mesh_loader:
            try:
                joint_map = {name: joint_msg.position[i] for i, name in enumerate(joint_msg.name) if "panda" in name}
                current_robot_points = self.mesh_loader.create_point_cloud(joint_map)
            except:
                pass

        # --- CUBE DETECTION (via SAM3 server) ---
        if not self.cube_locked:
            try:
                cv_depth = self.imgmsg_to_numpy(depth_msg)
                
                if cv_depth is not None:
                    # Call SAM3 server (fast - model already loaded!)
                    mask, score = self.sam3.segment(rgb_msg, self.target_object, self.detection_confidence)
                    
                    if mask is not None and score > self.detection_confidence:
                        z = cv_depth if cv_depth.dtype == np.float32 else cv_depth / 1000.0
                        valid = (mask > 0) & (z > 0.01) & (z < 2.0) & np.isfinite(z)
                        
                        if np.sum(valid) > 100:
                            v, u = np.where(valid)
                            z_val = z[valid]
                            x = (u - self.cx) * z_val / self.fx
                            y = (v - self.cy) * z_val / self.fy
                            points_cam = np.stack([x, y, z_val], axis=-1)
                            
                            ones = np.ones((points_cam.shape[0], 1))
                            points_world = np.dot(T_world_cam, np.hstack([points_cam, ones]).T).T[:, :3]
                            
                            self.static_cube_cloud = points_world
                            self.cube_locked = True
                            
                            c = np.mean(points_world, axis=0)
                            m = Marker()
                            m.header.frame_id = "world"
                            m.header.stamp = rospy.Time.now()
                            m.type = Marker.SPHERE
                            m.action = Marker.ADD
                            m.pose.position.x, m.pose.position.y, m.pose.position.z = c
                            m.scale.x = m.scale.y = m.scale.z = 0.05
                            m.color.a = 1.0
                            m.color.g = 1.0
                            self.pub_marker.publish(m)
                            
                            rospy.loginfo(f"✅ CUBE DETECTED at {c} (score={score:.3f})")
            except Exception as e:
                rospy.logwarn_throttle(5, f"Detection error: {e}")

        # --- GRASP & MERGE ---
        merged = []
        if current_robot_points is not None:
            merged.append(current_robot_points)
        
        if self.cube_locked and self.static_cube_cloud is not None:
            fork_tip = np.dot(T_world_tcp, self.fork_tip_offset_tcp)[:3]
            
            p_msg = PointStamped()
            p_msg.header.frame_id = "world"
            p_msg.header.stamp = rospy.Time.now()
            p_msg.point.x, p_msg.point.y, p_msg.point.z = fork_tip
            self.pub_fork_tip.publish(p_msg)

            if not self.is_grasped:
                dists = np.linalg.norm(self.static_cube_cloud - fork_tip, axis=1)
                if np.min(dists) < self.contact_threshold:
                    self.is_grasped = True
                    self.T_world_tcp_at_grasp = T_world_tcp
                    self.T_tcp_at_grasp_inv = np.linalg.inv(T_world_tcp)

            if self.is_grasped:
                T_motion = np.dot(T_world_tcp, self.T_tcp_at_grasp_inv)
                ones = np.ones((self.static_cube_cloud.shape[0], 1))
                self.current_cube_cloud = np.dot(T_motion, np.hstack([self.static_cube_cloud, ones]).T).T[:, :3]
            else:
                self.current_cube_cloud = self.static_cube_cloud
                
            merged.append(self.current_cube_cloud)

        if len(merged) > 0:
            full_cloud = np.vstack(merged)
            if full_cloud.shape[0] > 1024:
                indices = fpsample.bucket_fps_kdline_sampling(full_cloud.astype(np.float32), 1024, h=7)
                full_cloud = full_cloud[indices]
            self.publish_cloud(full_cloud)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    MergedCloudNode().run()