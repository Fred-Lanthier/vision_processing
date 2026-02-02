#!/usr/bin/env python3
# import rospy
# import numpy as np
# import torch
# import message_filters
# import tf
# import tf.transformations as tft
# import os
# import sys
# import rospkg
# import collections
# import json
# import sensor_msgs.point_cloud2 as pc2
# from sensor_msgs.msg import PointCloud2
# from geometry_msgs.msg import PoseArray, Pose
# from diffusers.schedulers.scheduling_ddim import DDIMScheduler
# from utils import compute_T_child_parent_xacro

# # --- 1. IMPORTS DYNAMIQUES ---
# rospack = rospkg.RosPack()
# pkg_path = rospack.get_path('vision_processing')
# sys.path.append(os.path.join(pkg_path, 'src', 'vision_processing', 'diffusion_model_train'))

# try:
#     from Train_Fork import DP3AgentRobust, Normalizer
# except ImportError as e:
#     rospy.logerr(f"Erreur Import: {e}")
#     sys.exit(1)

# # --- 2. UTILITAIRES ---
# def rotation_matrix_to_ortho6d(matrices):
#     x_raw = matrices[..., 0]; y_raw = matrices[..., 1]
#     return np.concatenate([x_raw, y_raw], axis=-1)

# def ortho6d_to_rotation_matrix(d6):
#     x_raw = d6[..., 0:3]; y_raw = d6[..., 3:6]
#     x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
#     z = np.cross(x, y_raw); z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
#     y = np.cross(z, x)
#     return np.stack([x, y, z], axis=-1)

# class DiffusionInferenceNode:
#     def __init__(self):
#         rospy.init_node("diffusion_inference_node")

#         # --- COMPUTE T_FORK_TCP ---
#         rospack = rospkg.RosPack()
#         package_path = rospack.get_path('vision_processing')
#         xacro_file = os.path.join(package_path, 'urdf', 'panda_camera.xacro')
#         self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro_file, 'fork_tip', 'panda_TCP')

#         # --- DEVICE ---
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # --- MODEL ---
#         model_name = rospy.get_param("~model_name", "dp3_policy_last_diffusers_fork_only_SAMPLE_NO_AUG.ckpt")
#         self.model_path = os.path.join(pkg_path, 'models', model_name)
        
#         if not os.path.exists(self.model_path):
#             rospy.logerr(f"Modèle introuvable: {self.model_path}")
#             sys.exit(1)

#         rospy.loginfo(f"📂 Chargement: {self.model_path}")
#         payload = torch.load(self.model_path, map_location=self.device)
        
#         # --- CONFIGURATION ---
#         cfg = payload.get('config', {})
#         self.obs_horizon = cfg.get('obs_horizon', 2)
#         self.pred_horizon = cfg.get('pred_horizon', 16)
#         self.action_dim = cfg.get('action_dim', 9)
#         self.num_points = cfg.get('num_points', 1024)
#         self.robot_state_dim = cfg.get('robot_state_dim', 9)
#         self.action_horizon = 16

#         if self.action_horizon > self.pred_horizon:
#             self.action_horizon = self.pred_horizon
            
#         # --- MODÈLE ---
#         self.model = DP3AgentRobust(
#             action_dim=self.action_dim, 
#             robot_state_dim=self.robot_state_dim,
#             obs_horizon=self.obs_horizon, 
#             pred_horizon=self.pred_horizon,
#             stats=None 
#         ).to(self.device)

#         weights = payload['state_dict'] if 'state_dict' in payload else payload
#         self.model.load_state_dict(weights, strict=False)
#         self.model.eval()

#         # Fallback Normalizer
#         if not hasattr(self.model, 'normalizer') or not self.model.normalizer.is_initialized:
#             if 'stats' in payload:
#                 self.model.normalizer = Normalizer(payload['stats']).to(self.device)
#             else:
#                 json_path = os.path.join(pkg_path, "normalization_stats_fork_only.json")
#                 with open(json_path, 'r') as f: stats = json.load(f)
#                 self.model.normalizer = Normalizer(stats).to(self.device)

#         # --- SCHEDULER (DDIM pour rapidité) ---
#         self.noise_scheduler = DDIMScheduler(
#             num_train_timesteps=100,
#             beta_schedule='squaredcos_cap_v2',
#             clip_sample=True,
#             prediction_type='sample'
#         )
#         self.noise_scheduler.set_timesteps(10)

#         # --- ROS ---
#         self.tf_listener = tf.TransformListener()
#         self.obs_queue = collections.deque(maxlen=self.obs_horizon)
#         self.latest_cloud = None
        
#         self.sub_cloud = rospy.Subscriber("/vision/merged_cloud", PointCloud2, self.cloud_callback, queue_size=1)
#         self.pub_trajectory = rospy.Publisher("/diffusion/target_trajectory", PoseArray, queue_size=1)
        
#         # Timer 10Hz
#         self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
#         rospy.loginfo("🚀 Inference Node Prêt.")

#     def cloud_callback(self, msg):
#         """ Lecture ultra-rapide du PointCloud2 sans ros_numpy """
#         # 1. Lire les dimensions
#         self.latest_cloud_stamp = msg.header.stamp
#         point_step = msg.point_step
#         data = msg.data

#         # 2. Créer une vue Numpy brute sur les données binaires
#         # On suppose que les champs sont x, y, z en float32 (standard)
#         # Offset typique : x=0, y=4, z=8
#         dtype_list = [
#             ('x', np.float32), 
#             ('y', np.float32), 
#             ('z', np.float32),
#             ('dummy', np.float32) # Souvent il y a un padding ou une couleur (rgb)
#         ]
        
#         # Ajustement si point_step != 16 octets (ex: juste xyz = 12 octets)
#         if point_step == 12:
#              dtype_list = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
        
#         # Lecture buffer
#         try:
#             # On force le buffer en tableau structuré
#             cloud_arr = np.frombuffer(data, dtype=dtype_list)
#         except ValueError:
#             # Si le buffer ne matche pas, on fallback (plus lent mais robuste)
#             # C'est rare si tu envoies du xyz32 standard
#             return 

#         # 3. Extraction vectorisée
#         points = np.zeros((cloud_arr.shape[0], 3), dtype=np.float32)
#         points[:, 0] = cloud_arr['x']
#         points[:, 1] = cloud_arr['y']
#         points[:, 2] = cloud_arr['z']
        
#         # 4. Filtrage NaNs (Vectorisé)
#         mask = np.isfinite(points).all(axis=1)
#         points = points[mask]
        
#         # 5. Sampling (Ton code existant)
#         n_points = points.shape[0]
#         if n_points == 0:
#             final_points = np.zeros((self.num_points, 3), dtype=np.float32)
#         elif n_points < self.num_points:
#             # Padding
#             extra_indices = np.random.choice(n_points, self.num_points - n_points, replace=True)
#             extra_points = points[extra_indices]
#             final_points = np.concatenate([points, extra_points], axis=0)
#         else:
#             # Downsampling
#             indices = np.random.choice(n_points, self.num_points, replace=False)
#             final_points = points[indices]
            
#         self.latest_cloud = final_points

#     def get_fork_pose(self):
#         try:
#             (trans_tcp, rot_tcp) = self.tf_listener.lookupTransform('/world', '/panda_hand_tcp', rospy.Time(0))
#             T_world_tcp = tft.quaternion_matrix(rot_tcp)
#             T_world_tcp[:3, 3] = trans_tcp
#             T_world_fork_tip = T_world_tcp @ self.T_tcp_fork_tip
#             mat = T_world_fork_tip[:3, :3]
#             rot_6d = rotation_matrix_to_ortho6d(mat)
#             return np.concatenate([T_world_fork_tip[:3, 3], rot_6d.flatten()])
#         except: return None

#     def control_loop(self, event):
#         if self.latest_cloud is None: return

#         # 1. Historique Fourchette
#         current_pose = self.get_fork_pose()
#         if current_pose is None: return
#         rospy.loginfo(f"[DEBUG] Robot fork_tip in world frame: {current_pose[:3]}")
#         rospy.loginfo(f"[DEBUG] Point cloud centroid: {np.mean(self.latest_cloud, axis=0)}")
#         self.obs_queue.append(current_pose)
#         if len(self.obs_queue) < self.obs_horizon: return

#         # 2. Batching
#         pcd_tensor = torch.from_numpy(self.latest_cloud).unsqueeze(0).float().to(self.device)
#         agent_tensor = torch.from_numpy(np.stack(self.obs_queue)).unsqueeze(0).float().to(self.device)

#         # 3. Normalisation
#         norm_agent_pos = self.model.normalizer.normalize(agent_tensor, 'agent_pos')

#         # 4. Inférence (Encodage conditionnel Hors-Boucle)
#         with torch.no_grad():
#             point_features = self.model.point_encoder(pcd_tensor)
#             robot_features = self.model.robot_mlp(norm_agent_pos.reshape(1, -1))
#             global_cond = torch.cat([point_features, robot_features], dim=-1)

#             noisy_action = torch.randn((1, self.pred_horizon, self.action_dim), device=self.device)
            
#             for t in self.noise_scheduler.timesteps:
#                 model_output = self.model.noise_pred_net(noisy_action, t, global_cond)
#                 noisy_action = self.noise_scheduler.step(model_output, t, noisy_action).prev_sample

#             # 5. Denormalisation
#             final_action = self.model.normalizer.unnormalize(noisy_action, 'action')
            
#             for i in range(min(16, final_action.shape[1])):
#                 pos = final_action[0, i, :3].cpu().detach().numpy()
#                 rospy.loginfo(f"final_action shape: {final_action.shape}")
#                 rospy.loginfo(f"Pose {i}: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")
#         # 6. Publication
#         action_chunk = final_action[0, :self.action_horizon].cpu().numpy()
#         self.publish_trajectory(action_chunk)

#     def publish_trajectory(self, action_chunk):
#         """ Publie un PoseArray contenant les actions futures """
#         msg = PoseArray()
#         msg.header.frame_id = "world" # ou "world" selon ta config MoveIt
#         # msg.header.stamp = rospy.Time.now()
#         msg.header.stamp = self.latest_cloud_stamp
#         # On convertit tout le chunk en poses
#         for i in range(len(action_chunk)):
#             action_9d = action_chunk[i]
#             pos = action_9d[:3]
            
#             # Conversion Rotation 6D -> Matrice 3x3 -> Quaternion
#             # Note: ortho6d_to_rotation_matrix attend une dimension (..., 6)
#             rot_6d = action_9d[3:]
#             # Ajout dimensions pour le broadcast si besoin, mais ici c'est un vecteur simple
#             # On appelle la fonction helper
#             rot_mat = ortho6d_to_rotation_matrix(rot_6d[None, None, :])[0, 0]
            
#             M = np.eye(4)
#             M[:3, :3] = rot_mat
#             quat = tft.quaternion_from_matrix(M)
#             norm = np.linalg.norm(quat)
#             quat = quat / norm
            
#             p = Pose()
#             p.position.x = pos[0]
#             p.position.y = pos[1]
#             p.position.z = pos[2]
#             p.orientation.x = quat[0]
#             p.orientation.y = quat[1]
#             p.orientation.z = quat[2]
#             p.orientation.w = quat[3]
            
#             msg.poses.append(p)
            
#         self.pub_trajectory.publish(msg)

# if __name__ == "__main__":
#     DiffusionInferenceNode()
#     rospy.spin()


"""
Diffusion Policy Inference Node for FORK with Temporal Ensembling

Key differences from base version:
- Controls fork_tip frame instead of panda_hand_tcp
- Uses T_tcp_fork_tip transform to compute fork pose
- Imports from Train_Fork

Smoothing features:
- Temporal ensembling averages overlapping predictions
- Results in much smoother motion
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

# Utils for fork transform
from utils import compute_T_child_parent_xacro

# --- IMPORTS ---
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
sys.path.append(os.path.join(pkg_path, 'src', 'vision_processing', 'diffusion_model_train'))

try:
    from Train_Fork import DP3AgentRobust, Normalizer
except ImportError as e:
    rospy.logerr(f"Import Error: {e}")
    sys.exit(1)


def rotation_matrix_to_ortho6d(matrix):
    """Convert 3x3 rotation matrix to 6D representation"""
    return np.concatenate([matrix[..., 0], matrix[..., 1]], axis=-1)


def ortho6d_to_rotation_matrix(d6):
    """Convert 6D representation to 3x3 rotation matrix"""
    x_raw, y_raw = d6[..., 0:3], d6[..., 3:6]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


class TemporalEnsembler:
    """
    Temporal Ensembling for Diffusion Policy
    
    Maintains a buffer of recent predictions and averages overlapping 
    actions to produce smoother outputs.
    """
    
    def __init__(self, pred_horizon, action_dim, buffer_size=5, weights='exponential'):
        """
        Args:
            pred_horizon: Number of steps in each prediction
            action_dim: Dimension of each action (9 for pos+rot6d)
            buffer_size: Number of past predictions to keep
            weights: 'uniform', 'exponential', or 'linear'
        """
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.weights_type = weights
        
        self.buffer = collections.deque(maxlen=buffer_size)
        self.step_counter = 0
    
    def add_prediction(self, prediction):
        """Add a new prediction to the buffer."""
        self.buffer.append((self.step_counter, prediction.copy()))
        self.step_counter += 1
    
    def get_ensembled_action(self, future_step=0):
        """Get the ensembled action for a given future step."""
        if len(self.buffer) == 0:
            return None
        
        actions = []
        weights = []
        
        for (timestamp, prediction) in self.buffer:
            age = self.step_counter - timestamp - 1
            pred_index = age + future_step
            
            if 0 <= pred_index < self.pred_horizon:
                actions.append(prediction[pred_index])
                
                if self.weights_type == 'exponential':
                    w = np.exp(-0.5 * age)
                elif self.weights_type == 'linear':
                    w = 1.0 / (age + 1)
                else:
                    w = 1.0
                    
                weights.append(w)
        
        if len(actions) == 0:
            return None
        
        actions = np.array(actions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return np.average(actions, axis=0, weights=weights)
    
    def get_ensembled_trajectory(self, num_steps):
        """Get ensembled trajectory for multiple future steps."""
        trajectory = []
        for i in range(num_steps):
            action = self.get_ensembled_action(future_step=i)
            if action is not None:
                trajectory.append(action)
            elif len(trajectory) > 0:
                trajectory.append(trajectory[-1])
        
        return np.array(trajectory) if len(trajectory) > 0 else None
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.step_counter = 0


class DiffusionInferenceNodeFork:
    def __init__(self):
        rospy.init_node("diffusion_inference_node_fork")
        
        # --- FORK TRANSFORM ---
        package_path = rospack.get_path('vision_processing')
        xacro_file = os.path.join(package_path, 'urdf', 'panda_camera.xacro')
        self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro_file, 'fork_tip', 'panda_TCP')
        rospy.loginfo("✅ Fork transform loaded")
        
        # --- DEVICE ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- MODEL ---
        model_name = rospy.get_param("~model_name", "dp3_policy_last_diffusers_fork_only_SAMPLE_NO_AUG.ckpt")
        self.model_path = os.path.join(pkg_path, 'models', model_name)
        
        if not os.path.exists(self.model_path):
            rospy.logerr(f"Model not found: {self.model_path}")
            sys.exit(1)

        rospy.loginfo(f"📂 Loading: {self.model_path}")
        payload = torch.load(self.model_path, map_location=self.device)
        
        # --- CONFIGURATION ---
        cfg = payload.get('config', {})
        self.obs_horizon = cfg.get('obs_horizon', 2)
        self.pred_horizon = cfg.get('pred_horizon', 16)
        self.action_dim = cfg.get('action_dim', 9)
        self.num_points = cfg.get('num_points', 1024)
        self.robot_state_dim = cfg.get('robot_state_dim', 9)
        self.action_horizon = 16

        # --- MODEL ---
        self.model = DP3AgentRobust(
            action_dim=self.action_dim, 
            robot_state_dim=self.robot_state_dim,
            obs_horizon=self.obs_horizon, 
            pred_horizon=self.pred_horizon,
            stats=None 
        ).to(self.device)

        weights = payload.get('state_dict', payload)
        self.model.load_state_dict(weights, strict=False)
        self.model.eval()

        # Normalizer
        if not hasattr(self.model, 'normalizer') or not self.model.normalizer.is_initialized:
            if 'stats' in payload:
                self.model.normalizer = Normalizer(payload['stats']).to(self.device)
            else:
                json_path = os.path.join(pkg_path, "normalization_stats_fork_only.json")
                with open(json_path, 'r') as f:
                    stats = json.load(f)
                self.model.normalizer = Normalizer(stats).to(self.device)

        # --- DDIM SCHEDULER ---
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='sample'
        )
        self.num_inference_steps = rospy.get_param("~num_inference_steps", 10)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        # --- TEMPORAL ENSEMBLING ---
        self.ensembler = TemporalEnsembler(
            pred_horizon=self.pred_horizon,
            action_dim=self.action_dim,
            buffer_size=rospy.get_param("~ensemble_buffer_size", 5),
            weights=rospy.get_param("~ensemble_weights", 'exponential')
        )
        
        self.publish_horizon = rospy.get_param("~publish_horizon", 16)

        # --- ROS ---
        self.tf_listener = tf.TransformListener()
        self.obs_queue = collections.deque(maxlen=self.obs_horizon)
        self.latest_cloud = None
        self.latest_cloud_stamp = None
        
        self.sub_cloud = rospy.Subscriber("/vision/merged_cloud", PointCloud2, self.cloud_callback, queue_size=1)
        self.pub_trajectory = rospy.Publisher("/diffusion/target_trajectory", PoseArray, queue_size=1)
        
        self.control_rate = rospy.get_param("~control_rate", 10.0)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control_loop)
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("🚀 Fork Inference Node (with Temporal Ensembling)")
        rospy.loginfo(f"   Ensemble buffer: {self.ensembler.buffer_size}")
        rospy.loginfo(f"   Ensemble weights: {self.ensembler.weights_type}")
        rospy.loginfo(f"   Inference steps: {self.num_inference_steps}")
        rospy.loginfo(f"   Control rate: {self.control_rate} Hz")
        rospy.loginfo("=" * 60)

    def cloud_callback(self, msg):
        """Fast point cloud parsing"""
        self.latest_cloud_stamp = msg.header.stamp
        point_step = msg.point_step
        data = msg.data

        if point_step == 12:
            dtype_list = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
        else:
            dtype_list = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('dummy', np.float32)]
        
        try:
            cloud_arr = np.frombuffer(data, dtype=dtype_list)
        except ValueError:
            return

        points = np.zeros((cloud_arr.shape[0], 3), dtype=np.float32)
        points[:, 0] = cloud_arr['x']
        points[:, 1] = cloud_arr['y']
        points[:, 2] = cloud_arr['z']
        
        mask = np.isfinite(points).all(axis=1)
        points = points[mask]
        
        n_points = points.shape[0]
        if n_points == 0:
            self.latest_cloud = np.zeros((self.num_points, 3), dtype=np.float32)
        elif n_points < self.num_points:
            extra = np.random.choice(n_points, self.num_points - n_points, replace=True)
            self.latest_cloud = np.concatenate([points, points[extra]], axis=0)
        else:
            indices = np.random.choice(n_points, self.num_points, replace=False)
            self.latest_cloud = points[indices]

    def get_fork_pose(self):
        """
        Get fork tip pose as 9D vector (pos + rot6d).
        Computes fork_tip pose from TCP pose using transform.
        """
        try:
            (trans_tcp, rot_tcp) = self.tf_listener.lookupTransform('/world', '/panda_hand_tcp', rospy.Time(0))
            
            # Build TCP transform matrix
            T_world_tcp = tft.quaternion_matrix(rot_tcp)
            T_world_tcp[:3, 3] = trans_tcp
            
            # Compute fork tip transform
            T_world_fork_tip = T_world_tcp @ self.T_tcp_fork_tip
            
            # Extract position and rotation
            pos = T_world_fork_tip[:3, 3]
            rot_mat = T_world_fork_tip[:3, :3]
            rot_6d = rotation_matrix_to_ortho6d(rot_mat)
            
            return np.concatenate([pos, rot_6d.flatten()])
        except:
            return None

    def control_loop(self, event):
        if self.latest_cloud is None:
            return

        # Get current fork pose
        current_pose = self.get_fork_pose()
        if current_pose is None:
            return
            
        self.obs_queue.append(current_pose)
        if len(self.obs_queue) < self.obs_horizon:
            return

        # Prepare tensors
        pcd_tensor = torch.from_numpy(self.latest_cloud).unsqueeze(0).float().to(self.device)
        agent_tensor = torch.from_numpy(np.stack(self.obs_queue)).unsqueeze(0).float().to(self.device)

        # Normalize
        norm_agent_pos = self.model.normalizer.normalize(agent_tensor, 'agent_pos')

        # Inference
        with torch.no_grad():
            point_features = self.model.point_encoder(pcd_tensor)
            robot_features = self.model.robot_mlp(norm_agent_pos.reshape(1, -1))
            global_cond = torch.cat([point_features, robot_features], dim=-1)

            noisy_action = torch.randn((1, self.pred_horizon, self.action_dim), device=self.device)
            
            for t in self.noise_scheduler.timesteps:
                model_output = self.model.noise_pred_net(noisy_action, t, global_cond)
                noisy_action = self.noise_scheduler.step(model_output, t, noisy_action).prev_sample

            final_action = self.model.normalizer.unnormalize(noisy_action, 'action')

        # Add to temporal ensemble buffer
        raw_prediction = final_action[0].cpu().numpy()
        self.ensembler.add_prediction(raw_prediction)
        
        # Get ensembled (smoothed) trajectory
        ensembled_trajectory = self.ensembler.get_ensembled_trajectory(self.publish_horizon)
        
        if ensembled_trajectory is not None:
            self.publish_trajectory(ensembled_trajectory)

    def publish_trajectory(self, action_chunk):
        """Publish trajectory as PoseArray (fork_tip poses)"""
        msg = PoseArray()
        msg.header.frame_id = "world"
        msg.header.stamp = self.latest_cloud_stamp if self.latest_cloud_stamp else rospy.Time.now()
        
        for action_9d in action_chunk:
            pos = action_9d[:3]
            rot_6d = action_9d[3:]
            
            # 6D to rotation matrix to quaternion
            rot_mat = ortho6d_to_rotation_matrix(rot_6d[None, None, :])[0, 0]
            M = np.eye(4)
            M[:3, :3] = rot_mat
            quat = tft.quaternion_from_matrix(M)
            quat = quat / np.linalg.norm(quat)
            
            p = Pose()
            p.position.x, p.position.y, p.position.z = pos
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
            msg.poses.append(p)
            
        self.pub_trajectory.publish(msg)


if __name__ == "__main__":
    DiffusionInferenceNodeFork()
    rospy.spin()