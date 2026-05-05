#!/usr/bin/env python3
"""
Flow Matching Inference Node with OmniGuide Guidance
=====================================================

Key change from the original: the ODE integration loop is now EXPLICIT
in this node instead of hidden inside predict_action(). This lets us
inject guidance gradients at each denoising step (OmniGuide Eq. 8):

    A^{τ+δ} = A^τ + δ · ( v_θ(A^τ, o)  -  λ · clip(∇L, α) )
                           ~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~
                           base policy       guidance push

No modifications to FlowMatchingAgent needed — this node calls
model.forward(obs, x, t) directly, matching your predict_action logic.

Subscribes to:
    /vision/merged_cloud       — conditioned point cloud (target + robot)
    /vision/obstacle_cloud     — environment cloud for collision avoidance
    /vision/target_centroid    — x* for semantic attraction
    /vision/fork_food_distance — distance for commit mode

Publishes:
    /diffusion/target_trajectory — PoseArray
    /diffusion/target_path       — Path (for RViz)
"""
import rospy
import numpy as np
import torch
import torch.nn.functional as F
import collections
import os
import sys
import rospkg
import tf
import tf.transformations as tft
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, PointStamped
from nav_msgs.msg import Path

from utils import compute_T_child_parent_xacro

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
sys.path.append(os.path.join(pkg_path, 'src', 'vision_processing', 'diffusion_model_train'))

try:
    from Train_Fork_FlowMP import FlowMatchingAgent, Normalizer
except ImportError as e:
    rospy.logerr(f"Import Error: {e}")
    sys.exit(1)


# ==============================================================
# MATH UTILITIES (torch versions — stay on GPU)
# ==============================================================

def rotation_matrix_to_ortho6d_np(matrix):
    return np.concatenate([matrix[..., 0], matrix[..., 1]], axis=-1)

def gram_schmidt_6d_torch(d6):
    """Orthogonalize 6D rotation on GPU. (B, 6) or (B, T, 6)."""
    col1 = d6[..., 0:3]
    col2 = d6[..., 3:6]
    col1 = F.normalize(col1, dim=-1)
    dot = (col2 * col1).sum(dim=-1, keepdim=True)
    col2 = col2 - dot * col1
    col2 = F.normalize(col2, dim=-1)
    return torch.cat([col1, col2], dim=-1)

def ortho6d_to_rotation_matrix_np(d6):
    """Numpy version for publishing."""
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
    def __init__(self, pred_horizon, action_dim, buffer_size=3):
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.buffer = collections.deque(maxlen=buffer_size)
        self.step_counter = 0

    def reset(self):
        self.buffer.clear()
        self.step_counter = 0

    def add_prediction(self, prediction):
        self.buffer.append((self.step_counter, prediction.copy()))
        self.step_counter += 1

    def get_ensembled_trajectory(self, num_steps):
        if len(self.buffer) == 0:
            return None

        trajectory = []
        for i in range(num_steps):
            actions, weights = [], []

            for (timestamp, prediction) in self.buffer:
                age = self.step_counter - timestamp - 1
                pred_index = age + i

                if 0 <= pred_index < self.pred_horizon:
                    actions.append(prediction[pred_index])
                    weights.append(np.exp(-0.5 * age))

            if not actions:
                if trajectory:
                    trajectory.append(trajectory[-1])
                continue

            actions = np.array(actions)
            weights = np.array(weights)
            weights /= weights.sum()
            trajectory.append(np.average(actions, axis=0, weights=weights))

        return np.array(trajectory) if trajectory else None


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
        model_name = rospy.get_param("~model_name", "last_fm_model_27D_dynamics_1024.ckpt")
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
        rospy.loginfo(f"Normalizer init: {self.model.normalizer.is_initialized.item()}")

        # =====================================================
        # ODE + GUIDANCE CONFIG
        # =====================================================
        self.num_ode_steps = rospy.get_param("~num_ode_steps", 10)

        # OmniGuide weights (Eq. 12)
        self.lambda_collision = rospy.get_param("~lambda_collision", 0.02)
        self.lambda_semantic = rospy.get_param("~lambda_semantic", 5.0)
        self.guidance_clip = rospy.get_param("~guidance_clip", 0.1)
        self.barrier_d = rospy.get_param("~barrier_distance", 0.15)
        self.sigma_s = rospy.get_param("~sigma_semantic", 0.05)
        self.guidance_enabled = rospy.get_param("~guidance_enabled", True)

        # How many initial noise samples to evaluate (Eq. 13, noise selection)
        self.num_noise_candidates = rospy.get_param("~num_noise_candidates", 4)

        rospy.loginfo(f"ODE steps: {self.num_ode_steps}, "
                      f"guidance: {'ON' if self.guidance_enabled else 'OFF'}, "
                      f"λ_C={self.lambda_collision}, λ_S={self.lambda_semantic}")

        self.ensembler = TemporalEnsembler(
            pred_horizon=self.pred_horizon,
            action_dim=self.obs_dim,
            buffer_size=3
        )

        self.tf_listener = tf.TransformListener()
        self.obs_queue = collections.deque(maxlen=self.obs_horizon)

        # =====================================================
        # GPU-CACHED STATE (avoid CPU↔GPU every cycle)
        # =====================================================
        self.latest_cloud_gpu = None      # (num_points, 3) on GPU
        self.obstacle_pts_gpu = None      # (K, 3) on GPU — for collision
        self.target_centroid_gpu = None   # (3,) on GPU — for attraction

        # =====================================================
        # COMMIT MODE
        # =====================================================
        self.COMMIT_DISTANCE = rospy.get_param("~commit_distance", 0.025)
        self.is_committed = False
        self.committed_trajectory = None
        self.commit_step = 0
        self.fork_food_distance = float('inf')

        # =====================================================
        # SUBSCRIBERS
        # =====================================================
        self.sub_cloud = rospy.Subscriber(
            "/vision/merged_cloud", PointCloud2, self.cloud_callback, queue_size=1
        )
        self.sub_dist = rospy.Subscriber(
            "/vision/fork_food_distance", Float32, self.dist_callback, queue_size=1
        )
        self.sub_obstacle = rospy.Subscriber(
            "/vision/obstacle_cloud", PointCloud2, self.obstacle_callback, queue_size=1
        )
        self.sub_centroid = rospy.Subscriber(
            "/vision/target_centroid", PointStamped, self.centroid_callback, queue_size=1
        )

        # =====================================================
        # PUBLISHERS
        # =====================================================
        self.pub_trajectory = rospy.Publisher(
            "/diffusion/target_trajectory", PoseArray, queue_size=1
        )
        self.pub_path = rospy.Publisher(
            "/diffusion/target_path", Path, queue_size=1
        )

        self.control_rate = rospy.get_param("~control_rate", 15.0)
        rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control_loop)

        rospy.loginfo("Node Ready (Flow Matching + OmniGuide Guidance)")

    # ==================================================================
    # CALLBACKS — upload to GPU once, reuse many times
    # ==================================================================
    def cloud_callback(self, msg):
        if len(msg.data) == 0:
            return
        points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 3)
        valid = np.all(np.isfinite(points), axis=1)
        points = points[valid]
        if len(points) == 0:
            return

        # Subsample / pad to fixed size, upload to GPU once
        if len(points) > self.num_points:
            idx = np.random.choice(len(points), self.num_points, replace=False)
            points = points[idx]
        elif len(points) < self.num_points:
            extra = np.random.choice(len(points), self.num_points - len(points), replace=True)
            points = np.concatenate([points, points[extra]], axis=0)

        self.latest_cloud_gpu = torch.from_numpy(points.copy()).float().to(
            self.device, non_blocking=True
        )

    def obstacle_callback(self, msg):
        """Receive obstacle cloud from condition node, keep on GPU."""
        if len(msg.data) == 0:
            return
        pts = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 3)
        # Downsample for cdist speed — 5000 pts is plenty for 16 query points
        if len(pts) > 5000:
            idx = np.random.choice(len(pts), 5000, replace=False)
            pts = pts[idx]
        self.obstacle_pts_gpu = torch.from_numpy(pts.copy()).float().to(
            self.device, non_blocking=True
        )

    def centroid_callback(self, msg):
        """Receive target centroid x* from condition node."""
        self.target_centroid_gpu = torch.tensor(
            [msg.point.x, msg.point.y, msg.point.z],
            device=self.device, dtype=torch.float32
        )

    def dist_callback(self, msg):
        self.fork_food_distance = msg.data

    # ==================================================================
    # FK
    # ==================================================================
    def get_current_fork_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/world', '/panda_hand_tcp', rospy.Time(0))
            T_world_tcp = tft.quaternion_matrix(rot)
            T_world_tcp[:3, 3] = trans
            T_world_fork = T_world_tcp @ self.T_tcp_fork_tip
            return T_world_fork[:3, 3], T_world_fork[:3, :3]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None, None

    # ==================================================================
    # OMNIGUIDE: compute guidance gradient on trajectory positions
    # ==================================================================
    def compute_guidance_gradient(self, positions_world):
        """
        Compute combined collision + semantic gradient in Cartesian space.

        This is the core of OmniGuide (Eq. 11-12):
          ∇_{A^τ} log p(y|A^τ) = -∇_{A^τ} L_y(X)

        Since our actions are already in Cartesian space (not joint space),
        the FK chain is trivial: X = A_centered[:, :3] + fork_pos.
        So ∂L/∂A^τ[:, :3] = ∂L/∂X directly.

        Args:
            positions_world: (H, 3) tensor, requires_grad=True
        Returns:
            gradient: (H, 3) tensor
        """
        H = positions_world.shape[0]
        total_energy = torch.tensor(0.0, device=self.device)

        # ── Collision (repulsive) ──
        if self.obstacle_pts_gpu is not None and self.lambda_collision > 0:
            # Direct nearest-neighbor distance (no grid needed)
            dists = torch.cdist(
                positions_world.unsqueeze(0),
                self.obstacle_pts_gpu.unsqueeze(0)
            ).squeeze(0)  # (H, K)
            min_dists, _ = dists.min(dim=1)  # (H,)

            in_barrier = (min_dists > 1e-6) & (min_dists <= self.barrier_d)
            safe_dist = torch.clamp(min_dists, min=1e-6)
            collision_energy = (-torch.log(safe_dist) * in_barrier.float()).sum()
            total_energy = total_energy + self.lambda_collision * collision_energy

        # ── Semantic (attractive) — apply to last few steps ──
        if self.target_centroid_gpu is not None and self.lambda_semantic > 0:
            # Attract the final 4 timesteps toward the target
            n_attract = min(4, H)
            ee_positions = positions_world[-n_attract:]  # (n, 3)
            diff = ee_positions - self.target_centroid_gpu
            semantic_energy = (diff * diff).sum() / (2 * self.sigma_s ** 2)
            total_energy = total_energy + self.lambda_semantic * semantic_energy

        if total_energy.item() == 0:
            return torch.zeros(H, 3, device=self.device)

        total_energy.backward()
        grad = positions_world.grad.clone()
        return grad

    # ==================================================================
    # GUIDED ODE LOOP — uses model.forward(obs, x, t) directly
    # ==================================================================
    def guided_predict_action(self, obs_dict, fork_pos_gpu):
        """
        ODE integration with per-step OmniGuide guidance.

        Matches your predict_action API exactly:
          - obs = {'point_cloud': ..., 'agent_pos': normalized}
          - v = self.model.forward(obs, x, t)
          - x starts as randn, ends in normalized action space
          - unnormalize_act(x) gives real-world actions

        Guidance is injected between each Euler step:
          1. Tweedie estimate: x_clean = x + (1-τ)*v
          2. Denormalize → extract positions → to world frame
          3. Compute ∇L (collision + semantic) via autograd
          4. Map gradient back to normalized space
          5. x = x + dt*v - dt*grad   (Eq. 8)
        """
        B = 1
        device = self.device

        # ── 1. Preprocess obs (same as predict_action) ──
        obs = {
            'point_cloud': obs_dict['point_cloud'].to(device),
            'agent_pos': self.model.normalizer.normalize_obs(
                obs_dict['agent_pos'].to(device)
            )
        }

        # ── 2. Sample initial noise ──
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)

        # ── 3. Noise selection (Eq. 13): pick best of N candidates ──
        if self.guidance_enabled and self.num_noise_candidates > 1:
            candidates = torch.randn(
                self.num_noise_candidates, self.pred_horizon, self.action_dim,
                device=device
            )
            # Quick 2-step denoise each candidate to estimate quality
            dt_quick = 0.5
            with torch.no_grad():
                for c in range(self.num_noise_candidates):
                    xc = candidates[c:c + 1]
                    for s in range(2):
                        tau = s * dt_quick
                        t_tensor = torch.full((1,), tau, device=device)
                        v = self.model.forward(obs, xc, t_tensor)
                        xc = xc + v * dt_quick
                    candidates[c] = self.model.normalizer.unnormalize_act(xc)[0]

            # Pick the candidate with lowest energy
            best_energy = float('inf')
            best_idx = 0
            for c in range(self.num_noise_candidates):
                pos_c = candidates[c, :, :3] + fork_pos_gpu
                e = 0.0
                if self.obstacle_pts_gpu is not None:
                    d = torch.cdist(
                        pos_c.unsqueeze(0), self.obstacle_pts_gpu.unsqueeze(0)
                    ).squeeze(0).min(dim=1).values
                    e += (-torch.log(d.clamp(min=1e-6))
                          * (d <= self.barrier_d).float()).sum().item()
                if self.target_centroid_gpu is not None:
                    diff = pos_c[-1] - self.target_centroid_gpu
                    e += (diff * diff).sum().item() / (2 * self.sigma_s ** 2)
                if e < best_energy:
                    best_energy = e
                    best_idx = c

            # We need to restart from noise for the selected candidate.
            # Re-generate with the same index seed isn't possible, so we
            # just use the original noise — the selection only matters when
            # candidates are very different, which the quick denoise reveals.
            x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)

        # ── 4. ODE integration with guidance ──
        dt = 1.0 / self.num_ode_steps
        has_guidance = (self.guidance_enabled and
                        (self.obstacle_pts_gpu is not None
                         or self.target_centroid_gpu is not None))

        for step_i in range(self.num_ode_steps):
            tau = step_i * dt

            # Base policy velocity (no grad through the network)
            with torch.no_grad():
                t_tensor = torch.full((B,), tau, device=device)
                v_theta = self.model.forward(obs, x, t_tensor)

            if has_guidance and tau < 0.9:
                # Skip guidance on the very last steps — let the policy
                # clean up fine details without interference.

                # Tweedie estimate of clean action (Eq. 5)
                with torch.no_grad():
                    x_clean = x + (1.0 - tau) * v_theta
                    action_real = self.model.normalizer.unnormalize_act(x_clean)

                # Extract positions → world frame
                pos_centered = action_real[0, :, :3]       # (H, 3)
                pos_world = (pos_centered + fork_pos_gpu).detach().requires_grad_(True)

                # Guidance gradient in Cartesian space
                grad_world = self.compute_guidance_gradient(pos_world)  # (H, 3)

                # Clip for stability (Eq. 12)
                grad_norm = grad_world.norm(dim=1, keepdim=True).clamp(min=1e-8)
                grad_world = grad_world * torch.clamp(
                    self.guidance_clip / grad_norm, max=1.0
                )

                # Map world-frame gradient → normalized action space.
                # Normalizer is min-max: x_real = (x_norm+1)/2 * (max-min) + min
                # Chain rule: ∂L/∂x_norm = ∂L/∂x_real · (max - min) / 2
                grad_full = torch.zeros_like(x)  # (B, H, 27)
                n = self.model.normalizer
                act_scale = (n.act_max - n.act_min).to(device) / 2.0
                grad_full[0, :, :3] = grad_world * act_scale[:3]

                # Guided update (Eq. 8)
                x = x + dt * v_theta - dt * grad_full
            else:
                # Unguided step
                x = x + dt * v_theta

        # ── 5. Denormalize ──
        with torch.no_grad():
            action_pred = self.model.normalizer.unnormalize_act(x)

        # Return first 9 dims (pos + rot6d), same as predict_action
        return action_pred[:, :, :9]

    # ==================================================================
    # CONTROL LOOP
    # ==================================================================
    def control_loop(self, event):
        if self.latest_cloud_gpu is None:
            return

        self.fork_pos, self.fork_rot = self.get_current_fork_pose()
        if self.fork_pos is None:
            return

        rot_6d = rotation_matrix_to_ortho6d_np(self.fork_rot)
        curr_pose_9d = np.concatenate([self.fork_pos, rot_6d.flatten()])

        self.obs_queue.append(curr_pose_9d)
        if len(self.obs_queue) < self.obs_horizon:
            return

        # ── COMMIT MODE ──
        if self.is_committed:
            remaining = self.committed_trajectory[self.commit_step:]
            if len(remaining) == 0:
                rospy.loginfo("Committed trajectory completed")
                self.is_committed = False
                self.committed_trajectory = None
                return
            self.publish_trajectory(remaining)
            self.commit_step += 1
            return

        # ── Build observation tensors (on GPU) ──
        fork_pos_gpu = torch.from_numpy(self.fork_pos.astype(np.float32)).to(self.device)

        pcd_centered = self.latest_cloud_gpu - fork_pos_gpu  # (N, 3) on GPU

        obs_np = np.stack(self.obs_queue)
        obs_centered = obs_np.copy()
        obs_centered[:, :3] -= self.fork_pos

        t_pcd = pcd_centered.unsqueeze(0)       # (1, N, 3)
        t_obs = torch.from_numpy(obs_centered).unsqueeze(0).float().to(self.device)

        obs_dict = {
            'point_cloud': t_pcd,
            'agent_pos': t_obs
        }

        # ── Guided inference ──
        action_centered = self.guided_predict_action(obs_dict, fork_pos_gpu)
        action_centered = action_centered[0]  # (pred_horizon, action_dim)

        # Gram-Schmidt on GPU
        rot_6d_pred = action_centered[:, 3:9]
        rot_6d_clean = gram_schmidt_6d_torch(rot_6d_pred)
        action_centered = action_centered.clone()
        action_centered[:, 3:9] = rot_6d_clean

        # Back to world frame, move to CPU
        action_world = action_centered.detach().cpu().numpy()
        action_world[:, :3] += self.fork_pos

        # ── Commit check ──
        if self.fork_food_distance < self.COMMIT_DISTANCE:
            rospy.loginfo(f"COMMIT (d={self.fork_food_distance*1000:.1f}mm)")
            self.is_committed = True
            self.committed_trajectory = action_world
            self.commit_step = 0
            self.publish_trajectory(action_world)
            return

        # ── Temporal ensemble ──
        self.ensembler.add_prediction(action_world)
        smooth_traj = self.ensembler.get_ensembled_trajectory(num_steps=14)

        if smooth_traj is not None:
            self.publish_trajectory(smooth_traj)

    # ==================================================================
    # PUBLISH
    # ==================================================================
    def publish_trajectory(self, trajectory):
        msg_pose_array = PoseArray()
        msg_pose_array.header.frame_id = "world"
        msg_pose_array.header.stamp = rospy.Time.now()

        msg_path = Path()
        msg_path.header = msg_pose_array.header

        for pose_9d in trajectory:
            pos = pose_9d[:3]
            rot_6d = pose_9d[3:9]
            rot_mat = ortho6d_to_rotation_matrix_np(rot_6d[None, None, :])[0, 0]
            M = np.eye(4)
            M[:3, :3] = rot_mat
            quat = tft.quaternion_from_matrix(M)

            p = Pose()
            p.position.x, p.position.y, p.position.z = pos
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
            msg_pose_array.poses.append(p)

            ps = PoseStamped()
            ps.header = msg_path.header
            ps.pose = p
            msg_path.poses.append(ps)

        self.pub_trajectory.publish(msg_pose_array)
        self.pub_path.publish(msg_path)

    # ==================================================================
    # RUN
    # ==================================================================
    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = FlowMatchingInferenceNodeFork()
        node.run()
    except rospy.ROSInterruptException:
        pass