#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import rospkg
import time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from vision_processing import fast_perception_module

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')


class PerceptionProcessingNode:
    """
    Offloads heavy perception tasks (CropBox + ROR) from the 150Hz loop.
    Runs at camera rate (~30Hz) to produce a cleaned obstacle cloud.

    Robot self-filtering is NOT done here: the fork is masked upstream by
    SAM3 in point_cloud_projector_node (depth zeroing + 3D capsule backup),
    and the wrist camera cannot see other arm links.
    """
    def __init__(self):
        rospy.init_node('perception_processing_node')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.crop_min = rospy.get_param("~crop_min", [0.0, -0.6, 0.0])
        self.crop_max = rospy.get_param("~crop_max", [0.8, 0.6, 1.0])
        self.log_timing = bool(rospy.get_param("~log_timing", True))

        self.raw_obstacles = None

        rospy.Subscriber('/perception/obstacles', PointCloud2, self.obs_callback)
        self.cleaned_pub = rospy.Publisher('/perception/cleaned_obstacles', PointCloud2, queue_size=1)

        self.rate = rospy.Rate(30)
        rospy.loginfo("Perception Processing Node ready at 30 Hz (CropBox + ROR).")

    def obs_callback(self, msg):
        try:
            points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, int(msg.point_step / 4))
            self.raw_obstacles = torch.tensor(points[:, :3].copy(), device=self.device)
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error in obs_callback: {e}")

    def run(self):
        while not rospy.is_shutdown():
            if self.raw_obstacles is not None:
                try:
                    t0 = time.perf_counter()
                    pts = self.raw_obstacles

                    # 1. GPU CropBox
                    mask = (
                        (pts[:, 0] > self.crop_min[0]) & (pts[:, 0] < self.crop_max[0]) &
                        (pts[:, 1] > self.crop_min[1]) & (pts[:, 1] < self.crop_max[1]) &
                        (pts[:, 2] > self.crop_min[2]) & (pts[:, 2] < self.crop_max[2])
                    )
                    pts_inside = pts[mask]

                    # 2. Radius Outlier Removal (C++)
                    if pts_inside.shape[0] > 0:
                        pts_np = pts_inside.cpu().numpy().astype(np.float32)
                        pts_filtered_np = fast_perception_module.radius_outlier_removal(pts_np, 0.05, 5)
                    else:
                        pts_filtered_np = np.empty((0, 3), dtype=np.float32)

                    header = Header(stamp=rospy.Time.now(), frame_id="world")
                    self.cleaned_pub.publish(pc2.create_cloud_xyz32(header, pts_filtered_np.tolist()))
                    if self.log_timing:
                        rospy.loginfo_throttle(5.0, f"⏱️ [TIMING] Perception Filter (Crop + ROR): {(time.perf_counter() - t0)*1000:.2f} ms")
                except Exception as e:
                    rospy.logerr_throttle(5.0, f"Error in perception run loop: {e}")

            self.rate.sleep()


if __name__ == '__main__':
    PerceptionProcessingNode().run()
