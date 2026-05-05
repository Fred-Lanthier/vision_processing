#!/usr/bin/env python3
"""
Flow Matching Inference Node for FORK — Translation-Centered
=============================================================

Updated for 27D Second-Order Dynamics and Path Visualization:
─────────────────────────────────────────────────────────────
  - Generates 27D dynamics (Pos/Rot + Vel + Acc) internally.
  - Publishes both geometry_msgs/PoseArray and nav_msgs/Path 
    for better RViz visualization.
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
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from nav_msgs.msg import Path

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
# MATH UTILITIES
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
# TEMPORAL ENSEMBLER
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
        model_name = rospy.get_param("~model_name", "best_fm_model_27D_dynamics.ckpt")
        self.model_path = os.path.join(pkg_path, 'models', model_name)
        
        if not os.path.exists(self.model_path):
            rospy.logerr(f"Model not found: {self.model_path}")
            sys.exit(1)

        rospy.loginfo(f"Loading Model: {model_name}")
        payload = torch.load(self.model_path, map_location=self.device)
        
        cfg = payload.get('config', {})
        self.obs_horizon = cfg.get('obs_horizon', 2)
        self.pred_horizon = cfg.get('pred_horizon', 16)
        
        self.obs_dim = cfg.get('obs_dim', 9)
        self.action_dim = cfg.get('action_dim', 27) 
        self.num_points = cfg.get('num_points', 256)
        
        self.model = FlowMatchingAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim, 
            obs_horizon=self.obs_horizon, 
            pred_horizon=self.pred_horizon,
            encoder_output_dim=cfg.get('encoder_output_dim', 64),
            diffusion_step_embed_dim=cfg.get('diffusion_step_embed_dim', 256),
            down_dims=cfg.get('down_dims', [256, 512, 1024]),
            kernel_size=cfg.get('kernel_size', 5),
            n_groups=cfg.get('n_groups', 8),
        ).to(self.device)

        self.model.load_state_dict(payload['model_state_dict'])
        
        if not self.model.normalizer.is_initialized and 'stats' in payload:
            self.model.normalizer.load_stats_from_dict(payload['stats'])
        
        self.model.eval()
        rospy.loginfo(f"Normalizer initialized: {self.model.normalizer.is_initialized.item()}")

        # =====================================================
        # ODE SOLVER CONFIG
        # =====================================================
        self.ode_method = rospy.get_param("~ode_method", "midpoint")
        self.num_ode_steps = rospy.get_param("~num_ode_steps", 10)
        
        rospy.loginfo(f"ODE: {self.ode_method}, {self.num_ode_steps} steps")

        self.ensembler = TemporalEnsembler(
            pred_horizon=self.pred_horizon,
            action_dim=self.obs_dim, 
            buffer_size=3
        )

        self.tf_listener = tf.TransformListener()
        self.obs_queue = collections.deque(maxlen=self.obs_horizon)
        self.latest_cloud = None
        
        self.sub_cloud = rospy.Subscriber(
            "/vision/merged_cloud", PointCloud2, self.cloud_callback, queue_size=1
        )
        
        # ── Publishers ──
        self.pub_trajectory = rospy.Publisher(
            "/diffusion/target_trajectory", PoseArray, queue_size=1
        )
        self.pub_path = rospy.Publisher(
            "/diffusion/target_path", Path, queue_size=1
        )
        
        self.control_rate = rospy.get_param("~control_rate", 10.0)
        rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control_loop)
        
        rospy.loginfo("Node Ready (27D Flow Matching Dynamics — 9D Execution)")

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
        
        with torch.no_grad():
            obs_dict = {
                'point_cloud': t_pcd,
                'agent_pos': t_obs
            }
            
            result = self.model.predict_action(
                obs_dict, 
                num_steps=self.num_ode_steps, 
                method=self.ode_method
            )
            
            action_centered_np = result['action'].cpu().numpy()[0]

        for t in range(action_centered_np.shape[0]):
            col1 = action_centered_np[t, 3:6]
            col2 = action_centered_np[t, 6:9]
            
            col1 = col1 / (np.linalg.norm(col1) + 1e-8)
            col2 = col2 - np.dot(col2, col1) * col1
            col2 = col2 / (np.linalg.norm(col2) + 1e-8)
            
            action_centered_np[t, 3:6] = col1
            action_centered_np[t, 6:9] = col2

        # =====================================================
        # BACK TO WORLD FRAME
        # =====================================================
        action_world_np = action_centered_np.copy()
        action_world_np[:, :3] += self.fork_pos

        self.ensembler.add_prediction(action_world_np)
        smooth_traj = self.ensembler.get_ensembled_trajectory(num_steps=8)
        
        if smooth_traj is not None:
            self.publish_trajectory(smooth_traj)

    def publish_trajectory(self, trajectory):
        msg_pose_array = PoseArray()
        msg_pose_array.header.frame_id = "world"
        msg_pose_array.header.stamp = rospy.Time.now()

        # Initialisation du message Path
        msg_path = Path()
        msg_path.header = msg_pose_array.header

        for pose_9d in trajectory:
            pos = pose_9d[:3]
            rot_6d = pose_9d[3:]
            rot_mat = ortho6d_to_rotation_matrix(rot_6d[None, None, :])[0, 0]
            M = np.eye(4)
            M[:3, :3] = rot_mat
            quat = tft.quaternion_from_matrix(M)

            # Création de la Pose basique
            p = Pose()
            p.position.x, p.position.y, p.position.z = pos
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
            msg_pose_array.poses.append(p)

            # Création de la PoseStamped pour le Path
            ps = PoseStamped()
            ps.header = msg_path.header
            ps.pose = p
            msg_path.poses.append(ps)

        rospy.loginfo(f"Fork tip: {np.linalg.norm(self.fork_pos):.4f}")
        
        # Publication simultanée
        self.pub_trajectory.publish(msg_pose_array)
        self.pub_path.publish(msg_path)


if __name__ == "__main__":
    import sensor_msgs.point_cloud2 as pc2
    try:
        FlowMatchingInferenceNodeFork()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass