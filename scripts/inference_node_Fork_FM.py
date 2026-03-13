#!/usr/bin/env python3
"""
Flow Matching Inference Node for FORK — Translation-Centered
=============================================================

Matched to FlowMatchingAgent from Train_Fork_FM.py.

What changed vs the Diffusion version:
───────────────────────────────────────
  REMOVED:
    - DDIMScheduler (diffusers)
    - The iterative denoising loop
    - noise_pred_net

  REPLACED BY:
    - model.predict_action(obs_dict, num_steps, method)
    - Which internally does ODE integration: euler / midpoint / rk4
    - velocity_net predicts v_θ(x_t, t, cond) instead of noise

  The model handles everything:
    - Normalization of obs and actions
    - Encoding (point cloud + robot state → conditioning)
    - ODE integration loop
    - Unnormalization of output
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

from utils import compute_T_child_parent_xacro

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
sys.path.append(os.path.join(pkg_path, 'src', 'vision_processing', 'diffusion_model_train'))

try:
    from Train_Fork_FM import FlowMatchingAgent, Normalizer
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
            actions, weights = [], []
            
            for (timestamp, prediction) in self.buffer:
                age = self.step_counter - timestamp - 1
                pred_index = age + i
                
                if 0 <= pred_index < self.pred_horizon:
                    actions.append(prediction[pred_index])
                    w = np.exp(-0.5 * age) if self.weights_type == 'exponential' else 1.0
                    weights.append(w)
            
            if not actions:
                if trajectory: trajectory.append(trajectory[-1])
                continue

            actions = np.array(actions)
            weights = np.array(weights)
            weights /= weights.sum()
            trajectory.append(np.average(actions, axis=0, weights=weights))
            
        return np.array(trajectory)


# ==============================================================
# ROS NODE
# ==============================================================

class FlowMatchingInferenceNodeFork:
    def __init__(self):
        rospy.init_node("flow_matching_inference_node_fork")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # URDF transforms
        package_path = rospack.get_path('vision_processing')
        xacro_file = os.path.join(package_path, 'urdf', 'panda_camera.xacro')
        self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro_file, 'fork_tip', 'panda_TCP')
        
        # =====================================================
        # LOAD MODEL
        # =====================================================
        model_name = rospy.get_param("~model_name", "last_fm_model_high_dim_CFM_relative.ckpt")
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
        self.num_points = cfg.get('num_points', 256)
        
        # ── Rebuild model with SAME architecture as training ──
        # FlowMatchingAgent contains: point_encoder, robot_mlp, velocity_net, normalizer
        # predict_action() handles: normalization → encoding → ODE loop → unnormalization
        self.model = FlowMatchingAgent(
            action_dim=self.action_dim, 
            obs_horizon=self.obs_horizon, 
            pred_horizon=self.pred_horizon,
            encoder_output_dim=cfg.get('encoder_output_dim', 64),
            diffusion_step_embed_dim=cfg.get('diffusion_step_embed_dim', 256),
            down_dims=cfg.get('down_dims', [256, 512, 1024]),
            kernel_size=cfg.get('kernel_size', 5),
            n_groups=cfg.get('n_groups', 8),
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(payload['model_state_dict'])
        
        # Load normalizer stats if not already in state_dict
        if not self.model.normalizer.is_initialized and 'stats' in payload:
            self.model.normalizer.load_stats_from_dict(payload['stats'])
        
        self.model.eval()
        rospy.loginfo(f"Normalizer initialized: {self.model.normalizer.is_initialized.item()}")

        # =====================================================
        # ODE SOLVER CONFIG
        # =====================================================
        # 'euler', 'midpoint', or 'rk4'
        self.ode_method = rospy.get_param("~ode_method", "euler")
        self.num_ode_steps = rospy.get_param("~num_ode_steps", 10)
        
        rospy.loginfo(f"ODE: {self.ode_method}, {self.num_ode_steps} steps")

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
        
        self.sub_cloud = rospy.Subscriber(
            "/vision/merged_cloud", PointCloud2, self.cloud_callback, queue_size=1
        )
        self.pub_trajectory = rospy.Publisher(
            "/diffusion/target_trajectory", PoseArray, queue_size=1
        )
        
        self.control_rate = rospy.get_param("~control_rate", 10.0)
        rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control_loop)
        
        rospy.loginfo("Node Ready (Flow Matching — Translation-Centered)")

    def cloud_callback(self, msg):
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
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/world', '/panda_hand_tcp', rospy.Time(0))
            T_world_tcp = tft.quaternion_matrix(rot)
            T_world_tcp[:3, 3] = trans
            T_world_fork = T_world_tcp @ self.T_tcp_fork_tip
            return T_world_fork[:3, 3], T_world_fork[:3, :3]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None, None

    def control_loop(self, event):
        if self.latest_cloud is None: return

        self.fork_pos, self.fork_rot = self.get_current_fork_pose()
        if self.fork_pos is None: return
        
        rot_6d = rotation_matrix_to_ortho6d(self.fork_rot)
        curr_pose_9d = np.concatenate([self.fork_pos, rot_6d.flatten()])
        
        self.obs_queue.append(curr_pose_9d)
        if len(self.obs_queue) < self.obs_horizon: return

        # =====================================================
        # TRANSLATION-ONLY CENTERING
        # =====================================================
        pcd_centered = self.latest_cloud - self.fork_pos

        obs_seq_world = np.stack(self.obs_queue)
        obs_seq_centered = obs_seq_world.copy()
        obs_seq_centered[:, :3] -= self.fork_pos

        # =====================================================
        # FLOW MATCHING INFERENCE
        # =====================================================
        t_pcd = torch.from_numpy(pcd_centered).unsqueeze(0).float().to(self.device)
        t_obs = torch.from_numpy(obs_seq_centered).unsqueeze(0).float().to(self.device)
        
        # Normalize obs (point cloud is NOT normalized — same as training)
        t_obs_norm = self.model.normalizer.normalize(t_obs)
        
        with torch.no_grad():
            # ── Encode ONCE ──
            obs_for_encoding = {
                'point_cloud': t_pcd,
                'agent_pos': t_obs_norm
            }
            global_cond = self.model.encode_obs(obs_for_encoding)
            
            # ── ODE integration reusing the same conditioning ──
            x = torch.randn(
                1, self.pred_horizon, self.action_dim, device=self.device
            )
            dt = 1.0 / self.num_ode_steps
            
            for i in range(self.num_ode_steps):
                t_val = i * dt
                t_tensor = torch.tensor([t_val], device=self.device)
                
                if self.ode_method == 'euler':
                    v = self.model.velocity_net(x, t_tensor, global_cond)
                    x = x + v * dt
                    
                elif self.ode_method == 'midpoint':
                    t_mid_tensor = torch.tensor([t_val + dt / 2], device=self.device)
                    v1 = self.model.velocity_net(x, t_tensor, global_cond)
                    x_mid = x + v1 * (dt / 2)
                    v2 = self.model.velocity_net(x_mid, t_mid_tensor, global_cond)
                    x = x + v2 * dt
            
            # Unnormalize back to physical units
            action_centered = self.model.normalizer.unnormalize(x)
            action_centered_np = action_centered[0].cpu().numpy()

        for t in range(action_centered_np.shape[0]):
            col1 = action_centered_np[t, 3:6]
            col2 = action_centered_np[t, 6:9]
            
            # Normalize first column
            col1 = col1 / (np.linalg.norm(col1) + 1e-8)
            
            # Gram-Schmidt: make col2 orthogonal to col1, then normalize
            col2 = col2 - np.dot(col2, col1) * col1
            col2 = col2 / (np.linalg.norm(col2) + 1e-8)
            
            action_centered_np[t, 3:6] = col1
            action_centered_np[t, 6:9] = col2

        # =====================================================
        # BACK TO WORLD FRAME
        # =====================================================
        action_world_np = action_centered_np.copy()
        action_world_np[:, :3] += self.fork_pos
        # Rotations (dims 3:9) already in world frame — untouched

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
        last_dist = np.linalg.norm(trajectory[-1, :3])
        first_dist = np.linalg.norm(trajectory[0, :3])
        rospy.loginfo(f"Fork tip position: {np.linalg.norm(self.fork_pos):.4f}")

        rospy.loginfo(f"Last position distance: {last_dist:.4f}")
        rospy.loginfo(f"First position distance: {first_dist:.4f}")
        
        self.pub_trajectory.publish(msg)


if __name__ == "__main__":
    import sensor_msgs.point_cloud2 as pc2
    try:
        FlowMatchingInferenceNodeFork()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass