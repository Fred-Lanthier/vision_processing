#!/usr/bin/env python3
"""
Diffusion Policy Inference Node for FORK — Translation-Centered
================================================================

Key change: Translation-only centering instead of full SE(3) local frame.
  - PCD: subtract fork tip position (world orientation preserved)
  - Poses: subtract fork tip position (rotations stay in world frame)
  - Prediction: add fork tip position back (rotations already in world)

This matches the training data loader (Data_Loader_Fork.py).
"""
import rospy
import numpy as np
import torch
import collections
import json
import os
import sys
import rospkg
import tf
import tf.transformations as tft
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, Pose
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from utils import compute_T_child_parent_xacro

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
sys.path.append(os.path.join(pkg_path, 'src', 'vision_processing', 'diffusion_model_train'))

try:
    from Train_Fork import DP3AgentRobust, Normalizer
except ImportError as e:
    rospy.logerr(f"Import Error: {e}")
    sys.exit(1)


# ==============================================================
# MATH UTILITIES (same as training)
# ==============================================================

def rotation_matrix_to_ortho6d(matrix):
    return np.concatenate([matrix[..., 0], matrix[..., 1]], axis=-1)

def ortho6d_to_rotation_matrix(d6):
    x_raw, y_raw = d6[..., 0:3], d6[..., 3:6]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


# ==============================================================
# TEMPORAL ENSEMBLER (unchanged)
# ==============================================================

class TemporalEnsembler:
    """
    Temporal Ensembling for smoothing predictions.
    Operates on WORLD FRAME actions.
    """
    def __init__(self, pred_horizon, action_dim, buffer_size=3, weights='exponential'):
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.buffer = collections.deque(maxlen=buffer_size)
        self.weights_type = weights
        self.step_counter = 0
    
    def add_prediction(self, prediction):
        self.buffer.append((self.step_counter, prediction.copy()))
        self.step_counter += 1
    
    def get_ensembled_trajectory(self, num_steps):
        if len(self.buffer) == 0: return None
        
        trajectory = []
        for i in range(num_steps):
            actions = []
            weights = []
            
            for (timestamp, prediction) in self.buffer:
                age = self.step_counter - timestamp - 1
                pred_index = age + i
                
                if 0 <= pred_index < self.pred_horizon:
                    actions.append(prediction[pred_index])
                    if self.weights_type == 'exponential':
                        weights.append(np.exp(-0.5 * age))
                    else:
                        weights.append(1.0)
            
            if not actions:
                if trajectory: trajectory.append(trajectory[-1])
                continue

            actions = np.array(actions)
            weights = np.array(weights)
            weights /= weights.sum()
            
            avg_action = np.average(actions, axis=0, weights=weights)
            trajectory.append(avg_action)
            
        return np.array(trajectory)


# ==============================================================
# ROS NODE
# ==============================================================

class DiffusionInferenceNodeFork:
    def __init__(self):
        rospy.init_node("diffusion_inference_node_fork")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # URDF transforms
        package_path = rospack.get_path('vision_processing')
        xacro_file = os.path.join(package_path, 'urdf', 'panda_camera.xacro')
        self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro_file, 'fork_tip', 'panda_TCP')
        
        # Load model
        model_name = rospy.get_param("~model_name", "Last_Fork_256_points_Relative.ckpt")
        self.model_path = os.path.join(pkg_path, 'models', model_name)
        
        if not os.path.exists(self.model_path):
            rospy.logerr(f"Model not found: {self.model_path}")
            sys.exit(1)

        rospy.loginfo(f"Loading Model: {model_name}")
        payload = torch.load(self.model_path, map_location=self.device)
        
        cfg = payload.get('config', {})
        self.obs_horizon = cfg.get('obs_horizon', 2)
        self.pred_horizon = cfg.get('pred_horizon', 16)
        self.action_dim = cfg.get('action_dim', 9)
        self.num_points = cfg.get('num_points', 1024)
        
        self.model = DP3AgentRobust(
            action_dim=self.action_dim, 
            robot_state_dim=cfg.get('robot_state_dim', 9),
            obs_horizon=self.obs_horizon, 
            pred_horizon=self.pred_horizon
        ).to(self.device)

        state_dict = payload.get('state_dict', payload)
        self.model.load_state_dict(state_dict, strict=False)
        
        if not hasattr(self.model, 'normalizer') or not self.model.normalizer.is_initialized:
            if 'stats' in payload:
                self.model.normalizer = Normalizer(payload['stats']).to(self.device)
            else:
                rospy.logwarn("Using default stats json file")
                json_path = os.path.join(pkg_path, "normalization_stats_fork.json")
                with open(json_path, 'r') as f: stats = json.load(f)
                self.model.normalizer = Normalizer(stats).to(self.device)
        
        self.model.eval()

        # Scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='sample'
        )
        self.num_inference_steps = rospy.get_param("~num_inference_steps", 10)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        # Ensembler
        self.ensembler = TemporalEnsembler(
            pred_horizon=self.pred_horizon,
            action_dim=self.action_dim,
            buffer_size=3
        )

        # ROS setup
        self.tf_listener = tf.TransformListener()
        self.obs_queue = collections.deque(maxlen=self.obs_horizon)
        self.latest_cloud = None
        
        self.sub_cloud = rospy.Subscriber("/vision/merged_cloud", PointCloud2, self.cloud_callback, queue_size=1)
        self.pub_trajectory = rospy.Publisher("/diffusion/target_trajectory", PoseArray, queue_size=1)
        
        self.control_rate = rospy.get_param("~control_rate", 10.0)
        rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control_loop)
        
        rospy.loginfo("Node Ready (Translation-Centered Inference)")

    def cloud_callback(self, msg):
        """
        Receives the merged cloud in WORLD frame from condition_pcd_node.
        We do NOT transform it here — centering happens in control_loop.
        """
        import sensor_msgs.point_cloud2 as pc2
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        points = np.array(points, dtype=np.float32)
        
        if points.shape[0] == 0: return

        if points.shape[0] > self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            self.latest_cloud = points[indices]
        else:
            extra = np.random.choice(points.shape[0], self.num_points - points.shape[0], replace=True)
            self.latest_cloud = np.concatenate([points, points[extra]], axis=0)

    def get_current_fork_pose(self):
        """
        Returns fork tip position (3,) and rotation matrix (3,3) in world frame.
        """
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/world', '/panda_hand_tcp', rospy.Time(0))
            T_world_tcp = tft.quaternion_matrix(rot)
            T_world_tcp[:3, 3] = trans
            T_world_fork = T_world_tcp @ self.T_tcp_fork_tip
            
            self.fork_pos = T_world_fork[:3, 3]
            self.fork_rot = T_world_fork[:3, :3]
            return self.fork_pos, self.fork_rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None, None

    def control_loop(self, event):
        if self.latest_cloud is None: return

        # 1. Get current fork tip pose in world
        fork_pos, fork_rot = self.get_current_fork_pose()
        if fork_pos is None: return
        
        # Build 9D pose (world frame)
        rot_6d = rotation_matrix_to_ortho6d(fork_rot)
        curr_pose_9d = np.concatenate([fork_pos, rot_6d.flatten()])
        
        # Observation queue (world frame)
        self.obs_queue.append(curr_pose_9d)
        if len(self.obs_queue) < self.obs_horizon: return

        # =====================================================
        # TRANSLATION-ONLY CENTERING (matches training)
        # =====================================================
        
        # A. Center point cloud: subtract fork tip position
        #    World orientation is PRESERVED
        pcd_centered = self.latest_cloud - fork_pos
        
        # B. Center observation poses: subtract position only
        #    Rotations (dims 3:9) stay in world frame
        obs_seq_world = np.stack(self.obs_queue)
        obs_seq_centered = obs_seq_world.copy()
        obs_seq_centered[:, :3] -= fork_pos
        # obs_seq_centered[-1, :3] is now [0, 0, 0]
        # obs_seq_centered[-1, 3:9] is the current world rotation (NOT identity)

        # =====================================================
        # INFERENCE
        # =====================================================
        
        t_pcd = torch.from_numpy(pcd_centered).unsqueeze(0).float().to(self.device)
        t_obs = torch.from_numpy(obs_seq_centered).unsqueeze(0).float().to(self.device)
        
        t_obs_norm = self.model.normalizer.normalize(t_obs, 'agent_pos')
        
        with torch.no_grad():
            feat_pcd = self.model.point_encoder(t_pcd)
            feat_robot = self.model.robot_mlp(t_obs_norm.reshape(1, -1))
            cond = torch.cat([feat_pcd, feat_robot], dim=-1)
            
            noisy = torch.randn((1, self.pred_horizon, self.action_dim), device=self.device)
            for t in self.noise_scheduler.timesteps:
                noise_pred = self.model.noise_pred_net(noisy, t, cond)
                noisy = self.noise_scheduler.step(noise_pred, t, noisy).prev_sample
            
            action_centered = self.model.normalizer.unnormalize(noisy, 'action')
            action_centered_np = action_centered[0].cpu().numpy()  # (pred_horizon, 9)

        # =====================================================
        # BACK TO WORLD FRAME
        # =====================================================
        # Just add fork tip position back to positions.
        # Rotations are already in world frame — no conversion needed.
        
        action_world_np = action_centered_np.copy()
        action_world_np[:, :3] += fork_pos
        # action_world_np[:, 3:9] unchanged (already world frame)
        
        # ── DIAGNOSTIC BLOCK ──
        # 1. What does the model actually predict? (centered frame)
        span = action_centered_np[-1, :3] - action_centered_np[0, :3]
        rospy.loginfo(f"[FM] Centered span: [{span[0]:.4f}, {span[1]:.4f}, {span[2]:.4f}] = {np.linalg.norm(span)*100:.1f}cm")

        # 2. Are rotations valid? (should be ~unit vectors)
        rot_col1_norm = np.linalg.norm(action_centered_np[0, 3:6])
        rot_col2_norm = np.linalg.norm(action_centered_np[0, 6:9])
        rospy.loginfo(f"[FM] Rot norms: col1={rot_col1_norm:.3f}, col2={rot_col2_norm:.3f} (should be ~1.0)")

        # 3. After adding fork_pos back, is the trajectory reachable?
        rospy.loginfo(f"[FM] World pos[0]: {action_world_np[0,:3]}")
        rospy.loginfo(f"[FM] World pos[-1]: {action_world_np[-1,:3]}")
        
        # Temporal ensembling (world frame)
        self.ensembler.add_prediction(action_world_np)
        smooth_traj = self.ensembler.get_ensembled_trajectory(num_steps=8)
        
        if smooth_traj is not None:
            self.publish_trajectory(smooth_traj)

    def publish_trajectory(self, trajectory):
        msg = PoseArray()
        msg.header.frame_id = "world"
        msg.header.stamp = rospy.Time.now()
        
        for pose_9d in trajectory:
            pos = pose_9d[:3]
            rot_6d = pose_9d[3:]
            
            rot_mat = ortho6d_to_rotation_matrix(rot_6d[None, None, :])[0, 0]
            M = np.eye(4)
            M[:3, :3] = rot_mat
            quat = tft.quaternion_from_matrix(M)
            
            p = Pose()
            p.position.x, p.position.y, p.position.z = pos
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
            msg.poses.append(p)
            
        self.pub_trajectory.publish(msg)
        last_dist = np.linalg.norm(trajectory[-1, :3])
        first_dist = np.linalg.norm(trajectory[0, :3])
        rospy.loginfo(f"Fork tip position: {np.linalg.norm(self.fork_pos):.4f}")
        rospy.loginfo(f"Last position distance: {last_dist:.4f}")
        rospy.loginfo(f"First position distance: {first_dist:.4f}")

if __name__ == "__main__":
    import sensor_msgs.point_cloud2 as pc2
    try:
        DiffusionInferenceNodeFork()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass