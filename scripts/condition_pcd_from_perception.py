#!/usr/bin/env python3
import rospy
import numpy as np
import message_filters
from sensor_msgs.msg import PointCloud2, JointState
from std_msgs.msg import Header, Float32
import sensor_msgs.point_cloud2 as pc2
import tf
import tf.transformations as tft
import sys
import os
import rospkg

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
scripts_path = os.path.join(pkg_path, 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils import compute_T_child_parent_xacro
try:
    from Compute_3D_point_cloud_from_mesh import RobotMeshLoaderOptimized
    LOADER_AVAILABLE = True
except ImportError:
    LOADER_AVAILABLE = False

class ConditionPcdFromPerception:
    def __init__(self):
        rospy.init_node('condition_pcd_from_perception', anonymous=True)
        
        self.num_points = rospy.get_param("~num_points", 1024)
        self.target_cloud = None
        self.target_stamp = None
        self.target_timeout = rospy.get_param("~target_timeout", 2.0)
        self.fork_cloud_cache = None
        self.last_joint_hash = None

        package_path = rospack.get_path('vision_processing')
        urdf_path = os.path.join(package_path, 'urdf', 'panda_camera.xacro')
        
        # Robot Loader
        self.mesh_loader = None
        if LOADER_AVAILABLE:
            try:
                self.mesh_loader = RobotMeshLoaderOptimized(urdf_path)
            except Exception as e:
                rospy.logerr(f"Could not load mesh loader: {e}")
        
        # Publishers
        self.pub_merged = rospy.Publisher('/vision/merged_cloud', PointCloud2, queue_size=1)
        self.pub_dist = rospy.Publisher('/vision/fork_food_distance', Float32, queue_size=1)
        
        # Subscribers
        rospy.Subscriber('/perception/target', PointCloud2, self.target_callback)
        rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        
        self.rate = rospy.Rate(30)
        rospy.loginfo("🚀 Condition PCD from Perception Node PRÊT")

    def target_callback(self, msg):
        try:
            points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, int(msg.point_step/4))
            pts_3d = points[:, :3]
            if len(pts_3d) > 0:
                self.target_cloud = pts_3d
                self.target_stamp = rospy.get_time()
            # If the cloud is empty, we do not immediately clear it.
            # We allow the run() loop's target_timeout to handle expiration.
        except Exception as e:
            rospy.logerr_throttle(5, f"Error reading target cloud: {e}")

    def joint_callback(self, joint_msg):
        if not self.mesh_loader:
            return
        try:
            # Proper joint mapping by name
            joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
            pos_dict = {n: p for n, p in zip(joint_msg.name, joint_msg.position)}
            q_dict = {}
            for jn in joint_names:
                if jn in pos_dict:
                    q_dict[jn] = pos_dict[jn]
                else:
                    return # Wait for complete state
            
            joint_hash = tuple(round(v, 4) for v in q_dict.values())
            if joint_hash != self.last_joint_hash:
                self.fork_cloud_cache = self.mesh_loader.create_point_cloud_fork_tip(q_dict)
                self.last_joint_hash = joint_hash
        except Exception as e:
            rospy.logerr_throttle(5, f"Error in condition joint_callback: {e}")

    def publish_cloud(self, points, frame_id="world"):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        if points is None or len(points) == 0:
            cloud_msg = pc2.create_cloud_xyz32(header, np.empty((0, 3)))
        else:
            cloud_msg = pc2.create_cloud_xyz32(header, points)
        self.pub_merged.publish(cloud_msg)

    def run(self):
        pts_per_object = self.num_points // 2
        
        while not rospy.is_shutdown():
            # Expire stale target so a lost SAM2 mask doesn't keep guiding the planner
            if self.target_stamp is not None and (rospy.get_time() - self.target_stamp) > self.target_timeout:
                self.target_cloud = None
                self.target_stamp = None
                rospy.logwarn_throttle(5.0, "Target cloud expired (no update for %.1fs) — waiting for detection." % self.target_timeout)

            # Only proceed if we have BOTH the fork component AND the target detection
            if self.fork_cloud_cache is not None and self.target_cloud is not None:
                merged = []
                
                # 1. Robot points (Fork)
                pts = self.fork_cloud_cache
                if pts.shape[0] > pts_per_object:
                    idx = np.random.choice(pts.shape[0], pts_per_object, replace=False)
                    merged.append(pts[idx])
                else:
                    merged.append(pts)
                
                # 2. Target points (Cube)
                # PUBLISH DISTANCE
                dists = np.linalg.norm(self.target_cloud - np.mean(self.fork_cloud_cache, axis=0), axis=1)
                min_dist = np.min(dists)
                self.pub_dist.publish(Float32(data=min_dist))

                pts = self.target_cloud
                if pts.shape[0] > pts_per_object:
                    idx = np.random.choice(pts.shape[0], pts_per_object, replace=False)
                    merged.append(pts[idx])
                else:
                    merged.append(pts)
                
                # 3. Assemble and Pad/Sample
                if len(merged) > 0:
                    full_cloud = np.vstack(merged)
                    
                    if full_cloud.shape[0] > self.num_points:
                        idx = np.random.choice(full_cloud.shape[0], self.num_points, replace=False)
                        full_cloud = full_cloud[idx]
                    elif full_cloud.shape[0] < self.num_points and full_cloud.shape[0] > 0:
                        extra_idx = np.random.choice(full_cloud.shape[0], self.num_points - full_cloud.shape[0], replace=True)
                        full_cloud = np.vstack([full_cloud, full_cloud[extra_idx]])
                    
                    self.publish_cloud(full_cloud)
            elif self.fork_cloud_cache is None:
                rospy.loginfo_throttle(10, "Waiting for fork mesh...")
            elif self.target_cloud is None:
                rospy.loginfo_throttle(10, "Waiting for target detection before publishing to planner...")
                self.publish_cloud(np.empty((0, 3)))
            
            self.rate.sleep()

if __name__ == '__main__':
    node = ConditionPcdFromPerception()
    node.run()
