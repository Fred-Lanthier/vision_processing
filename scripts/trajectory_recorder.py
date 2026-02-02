#!/usr/bin/env python3
"""
Trajectory Recorder Node

Records all relevant data during policy execution for later analysis:
- End-effector poses (TCP and fork_tip)
- Joint positions and velocities
- Inference timing
- Point cloud snapshots

Usage:
    rosrun vision_processing trajectory_recorder.py _output_dir:=/path/to/save

Data is saved as JSON files for easy plotting.
"""
import rospy
import numpy as np
import json
import os
import time
import tf
import tf.transformations as tft
from datetime import datetime
from collections import deque

from sensor_msgs.msg import JointState, PointCloud2
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64MultiArray

# For fork transform
import rospkg
import sys
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
sys.path.append(os.path.join(pkg_path, 'scripts'))

try:
    from utils import compute_T_child_parent_xacro
    FORK_TRANSFORM_AVAILABLE = True
except ImportError:
    FORK_TRANSFORM_AVAILABLE = False


class TrajectoryRecorder:
    def __init__(self):
        rospy.init_node('trajectory_recorder')
        
        # Output directory
        self.output_dir = rospy.get_param("~output_dir", os.path.expanduser("~/trajectory_recordings"))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Recording state
        self.is_recording = False
        self.record_start_time = None
        
        # Data buffers
        self.tcp_poses = []          # [(t, x, y, z, qx, qy, qz, qw), ...]
        self.fork_poses = []         # [(t, x, y, z, qx, qy, qz, qw), ...]
        self.joint_positions = []    # [(t, j1, j2, ..., j7), ...]
        self.joint_velocities = []   # [(t, v1, v2, ..., v7), ...]
        self.joint_commands = []     # [(t, j1, j2, ..., j7), ...]
        self.predicted_trajectories = []  # [(t, [poses...]), ...]
        self.inference_times = []    # [(t, inference_ms), ...]
        
        # Fork transform
        self.T_tcp_fork_tip = None
        if FORK_TRANSFORM_AVAILABLE:
            try:
                xacro_file = os.path.join(pkg_path, 'urdf', 'panda_camera.xacro')
                self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro_file, 'fork_tip', 'panda_TCP')
                rospy.loginfo("✅ Fork transform loaded")
            except Exception as e:
                rospy.logwarn(f"Could not load fork transform: {e}")
        
        # TF listener
        self.tf_listener = tf.TransformListener()
        
        # Subscribers
        self.sub_joints = rospy.Subscriber("/joint_states", JointState, self.joint_callback)
        self.sub_trajectory = rospy.Subscriber("/diffusion/target_trajectory", PoseArray, self.trajectory_callback)
        self.sub_command = rospy.Subscriber("/joint_group_position_controller/command", Float64MultiArray, self.command_callback)
        
        # Timer for pose recording (50 Hz)
        self.pose_timer = rospy.Timer(rospy.Duration(0.02), self.record_poses)
        
        # Inference time tracking
        self.last_trajectory_time = None
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("🎬 Trajectory Recorder Ready")
        rospy.loginfo(f"   Output directory: {self.output_dir}")
        rospy.loginfo("   Commands:")
        rospy.loginfo("     rosservice call /recorder/start")
        rospy.loginfo("     rosservice call /recorder/stop")
        rospy.loginfo("=" * 60)
        
        # Start recording immediately (or use services for manual control)
        self.start_recording()
    
    def start_recording(self):
        """Start recording data"""
        self.is_recording = True
        self.record_start_time = rospy.Time.now()
        
        # Clear buffers
        self.tcp_poses = []
        self.fork_poses = []
        self.joint_positions = []
        self.joint_velocities = []
        self.joint_commands = []
        self.predicted_trajectories = []
        self.inference_times = []
        
        rospy.loginfo("🔴 Recording STARTED")
    
    def stop_recording(self):
        """Stop recording and save data"""
        self.is_recording = False
        self.save_data()
        rospy.loginfo("⬛ Recording STOPPED")
    
    def get_relative_time(self):
        """Get time since recording started"""
        if self.record_start_time is None:
            return 0.0
        return (rospy.Time.now() - self.record_start_time).to_sec()
    
    def record_poses(self, event):
        """Record TCP and fork poses from TF"""
        if not self.is_recording:
            return
        
        t = self.get_relative_time()
        
        # Record TCP pose
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/world', '/panda_hand_tcp', rospy.Time(0))
            self.tcp_poses.append([t] + list(trans) + list(rot))
        except:
            pass
        
        # Record fork pose (if transform available)
        if self.T_tcp_fork_tip is not None:
            try:
                (trans_tcp, rot_tcp) = self.tf_listener.lookupTransform('/world', '/panda_hand_tcp', rospy.Time(0))
                T_world_tcp = tft.quaternion_matrix(rot_tcp)
                T_world_tcp[:3, 3] = trans_tcp
                T_world_fork = T_world_tcp @ self.T_tcp_fork_tip
                
                pos = T_world_fork[:3, 3]
                quat = tft.quaternion_from_matrix(T_world_fork)
                self.fork_poses.append([t] + list(pos) + list(quat))
            except:
                pass
    
    def joint_callback(self, msg):
        """Record joint states"""
        if not self.is_recording:
            return
        
        t = self.get_relative_time()
        
        # Extract panda joints
        positions = []
        velocities = []
        for i, name in enumerate(msg.name):
            if "panda_joint" in name:
                positions.append(msg.position[i])
                if len(msg.velocity) > i:
                    velocities.append(msg.velocity[i])
        
        if len(positions) == 7:
            self.joint_positions.append([t] + positions)
            if len(velocities) == 7:
                self.joint_velocities.append([t] + velocities)
    
    def command_callback(self, msg):
        """Record joint commands"""
        if not self.is_recording:
            return
        
        t = self.get_relative_time()
        if len(msg.data) == 7:
            self.joint_commands.append([t] + list(msg.data))
    
    def trajectory_callback(self, msg):
        """Record predicted trajectories and compute inference time"""
        if not self.is_recording:
            return
        
        t = self.get_relative_time()
        
        # Compute inference time (time between predictions)
        current_time = time.time()
        if self.last_trajectory_time is not None:
            inference_interval = (current_time - self.last_trajectory_time) * 1000  # ms
            self.inference_times.append([t, inference_interval])
        self.last_trajectory_time = current_time
        
        # Record predicted trajectory
        poses_data = []
        for pose in msg.poses:
            poses_data.append([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
            ])
        self.predicted_trajectories.append([t, poses_data])
    
    def save_data(self):
        """Save all recorded data to JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        data = {
            'metadata': {
                'timestamp': timestamp,
                'duration_sec': self.get_relative_time(),
                'num_tcp_poses': len(self.tcp_poses),
                'num_fork_poses': len(self.fork_poses),
                'num_joint_samples': len(self.joint_positions),
                'num_predictions': len(self.predicted_trajectories),
            },
            'tcp_poses': self.tcp_poses,
            'fork_poses': self.fork_poses,
            'joint_positions': self.joint_positions,
            'joint_velocities': self.joint_velocities,
            'joint_commands': self.joint_commands,
            'predicted_trajectories': self.predicted_trajectories,
            'inference_times': self.inference_times,
        }
        
        filename = os.path.join(self.output_dir, f"trajectory_{timestamp}.json")
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        rospy.loginfo(f"💾 Data saved to: {filename}")
        rospy.loginfo(f"   Duration: {data['metadata']['duration_sec']:.2f}s")
        rospy.loginfo(f"   TCP poses: {len(self.tcp_poses)}")
        rospy.loginfo(f"   Fork poses: {len(self.fork_poses)}")
        rospy.loginfo(f"   Joint samples: {len(self.joint_positions)}")
        rospy.loginfo(f"   Predictions: {len(self.predicted_trajectories)}")
        
        return filename
    
    def run(self):
        rospy.on_shutdown(self.stop_recording)
        rospy.spin()


if __name__ == '__main__':
    TrajectoryRecorder().run()
