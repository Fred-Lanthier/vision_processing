#!/usr/bin/env python3
import rospy
import torch
import numpy as np
import time
import os
import sys
import rospkg
import collections
import xacro
import tempfile
from sensor_msgs.msg import JointState, PointCloud2
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

# ==============================================================
# TEMPORAL ENSEMBLER
# ==============================================================
class TemporalEnsembler:
    def __init__(self, pred_horizon, buffer_size=3, weights='exponential'):
        self.pred_horizon = pred_horizon
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

# Standard imports for this workspace
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
sys.path.insert(0, pkg_path)
sys.path.append(os.path.join(pkg_path, 'src', 'vision_processing', 'diffusion_model_train'))

from Train_Fork_FlowMP import FlowMatchingAgent, Normalizer
from vision_processing import fast_ik_module
from third_party.SDF_Bernstein_Basis.src.rdf_weights import RDF_Weights
from third_party.SDF_Bernstein_Basis.bernstein_core import BernsteinCore
from third_party.SDF_Bernstein_Basis.bernstein_barrier import BernsteinBarrier
from third_party.RDF.urdf_layer import URDFLayer
from utils import compute_T_child_parent_xacro
from Compute_3D_point_cloud_from_mesh import RobotMeshLoaderOptimized

def decode_9d_to_se3_gpu(traj_9d, eps=1e-6):
    """
    Orthogonalisation Gram-Schmidt 100% vectorisée sur GPU.
    """
    original_shape = traj_9d.shape
    if len(original_shape) == 3:
        B, H, D = original_shape
        traj = traj_9d.reshape(B * H, D)
    else:
        traj = traj_9d
        
    pos = traj[:, :3]
    c1 = traj[:, 3:6]
    c2 = traj[:, 6:9]
    
    # Gram-Schmidt avec guards de division
    b1_norm = torch.norm(c1, dim=-1, keepdim=True).clamp(min=eps)
    b1 = c1 / b1_norm
    
    dot_product = torch.sum(b1 * c2, dim=-1, keepdim=True)
    b2 = c2 - dot_product * b1
    
    b2_norm = torch.norm(b2, dim=-1, keepdim=True).clamp(min=eps)
    b2 = b2 / b2_norm
    
    b3 = torch.cross(b1, b2, dim=-1)
    
    # Construction du batch
    N = traj.shape[0]
    T = torch.eye(4, device=traj_9d.device).unsqueeze(0).repeat(N, 1, 1)
    T[:, :3, 0] = b1
    T[:, :3, 1] = b2
    T[:, :3, 2] = b3
    T[:, :3, 3] = pos
    
    return T

class CASFGenerativeNode:
    def __init__(self):
        rospy.init_node('casf_generative_node')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paths and Parameters
        urdf_path_raw = rospy.get_param("~urdf_path", pkg_path + '/urdf/panda_camera.xacro')
        
        # Global Xacro -> URDF conversion
        if urdf_path_raw.endswith('.xacro'):
            rospy.loginfo(f"⏳ Processing XACRO: {urdf_path_raw}")
            # Explicitly pass arm_id mapping
            doc = xacro.process_file(urdf_path_raw, mappings={'arm_id': 'panda'})
            with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
                f.write(doc.toxml())
                urdf_path = f.name
            rospy.loginfo(f"🔥 Processed XACRO to temporary URDF: {urdf_path}")
        else:
            urdf_path = urdf_path_raw

        # Debug: Check joints in URDF
        from urdf_parser_py.urdf import URDF as URDFParser
        with open(urdf_path, 'r') as f:
            robot_urdf = URDFParser.from_xml_string(f.read())
            urdf_joints = [j.name for j in robot_urdf.joints if j.type != 'fixed']
            rospy.loginfo(f"🔍 URDF non-fixed joints: {urdf_joints}")

        voxel_dir = rospy.get_param("~voxel_dir", pkg_path + '/third_party/RDF/panda_layer/meshes/voxel_128')
        weights_dir = rospy.get_param("~weights_dir", pkg_path + '/third_party/SDF_Bernstein_Basis/panda_test')
        model_name = rospy.get_param("~model_name", "last_fm_model_27D_dynamics_1024.ckpt")
        model_path = os.path.join(pkg_path, 'models', model_name)
        
        # 1. Initialize Bernstein Safety components
        self.robot_layer = URDFLayer(urdf_path=urdf_path, device=self.device, package_dir=pkg_path, voxel_dir=voxel_dir)
        weight_handler = RDF_Weights(device=self.device, dtype=torch.float32)
        weight_handler.init_robot_folder(weights_dir, robot_name='panda')
        link_names = ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7']
        weight_handler.add_models(link_names, robot_name='panda')
        self.bernstein_core = BernsteinCore(weight_handler, self.robot_layer, self.device, link_names)
        self.barrier = BernsteinBarrier(self.bernstein_core, d_safe=0.05)
        
        # 2. Initialize IK Solver (Ultra-fast Pybind11 Pinocchio)
        self.ik_solver = fast_ik_module.FastIK(urdf_path, "fork_tip")
        self._q_warm = None
        
        # Debug: Print Pinocchio model joints
        try:
            pin_joints = self.ik_solver.get_joint_names()
            rospy.loginfo(f"🤖 Pinocchio IK joints: {pin_joints}")
        except:
            pass
        
        # 3. Initialize Fork-related components
        self.mesh_loader = RobotMeshLoaderOptimized(urdf_path)
        fork_pts = self.mesh_loader.static_point_clouds.get('fork_tip', np.zeros((0, 3)))
        self.fork_pts_local = torch.from_numpy(fork_pts).float().to(self.device)

        # 4. Initialize Flow Matching Agent
        rospy.loginfo(f"🛡️  SafeGenerativeNode: Loading Model: {model_name}")
        payload = torch.load(model_path, map_location=self.device)
        cfg = payload.get('config', {})
        self.action_dim = cfg.get('action_dim', 27)
        self.obs_dim = cfg.get('obs_dim', 9)
        self.fm_agent = FlowMatchingAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            obs_horizon=cfg.get('obs_horizon', 2),
            pred_horizon=cfg.get('pred_horizon', 16),
            encoder_output_dim=cfg.get('encoder_output_dim', 64),
            diffusion_step_embed_dim=cfg.get('diffusion_step_embed_dim', 256),
            down_dims=cfg.get('down_dims', [256, 512, 1024]),
        ).to(self.device)
        self.fm_agent.load_state_dict(payload['model_state_dict'])
        self.normalizer = self.fm_agent.normalizer
        if not self.normalizer.is_initialized and 'stats' in payload:
            self.normalizer.load_stats_from_dict(payload['stats'])
        self.fm_agent.eval()

        # 4. State variables
        self.current_q = None
        self.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        self.obstacle_pts_gpu = None
        self.obs_queue = collections.deque(maxlen=cfg.get('obs_horizon', 2))
        self.num_points_pcd = cfg.get('num_points', 256)
        self.latest_cloud_gpu = None
        self.ensembler = TemporalEnsembler(pred_horizon=cfg.get('pred_horizon', 16), buffer_size=3)

        
        # 5. Subscribers & Publishers
        rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        rospy.Subscriber('/perception/cleaned_obstacles', PointCloud2, self.obstacle_callback)
        rospy.Subscriber('/vision/merged_cloud', PointCloud2, self.cloud_callback)
        
        self.traj_pub = rospy.Publisher('/planner/nominal_trajectory', JointTrajectory, queue_size=1)
        self.viz_traj_pub = rospy.Publisher('/viz/nominal_trajectory_3d', MarkerArray, queue_size=1)
        
        # Control Loop at 8Hz (Planner rate)
        self.rate_hz = 8.0
        rospy.Timer(rospy.Duration(1.0/self.rate_hz), self.control_loop)
        rospy.loginfo("🚀 Safe Generative Planner Node Initialized at 8Hz")

    def joint_callback(self, msg):
        try:
            # Proper joint mapping by name
            pos_dict = {n: p for n, p in zip(msg.name, msg.position)}
            q_list = []
            for jn in self.joint_names:
                if jn in pos_dict:
                    q_list.append(pos_dict[jn])
                else:
                    return # Silently ignore messages that don't have all arm joints (e.g. from fake_finger_publisher)
            self.current_q = torch.tensor(q_list, dtype=torch.float32, device=self.device)
        except Exception as e:
            rospy.logerr_throttle(5, f"Error in joint_callback: {e}")

    def obstacle_callback(self, msg):
        # Unpack obstacles for CBF Corrector in ODE
        # PointCloud2 to Numpy (Assume XYZ float32 from create_cloud_xyz32)
        try:
            points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, int(msg.point_step/4))
            points = points[:, :3] # Take only X, Y, Z
            if len(points) > 2000:
                idx = np.random.choice(len(points), 2000, replace=False)
                points = points[idx]
            self.obstacle_pts_gpu = torch.from_numpy(points.copy()).float().to(self.device)
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error unpacking obstacles: {e}")

    def cloud_callback(self, msg):
        # Conditioned point cloud for FM Encoder
        try:
            points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, int(msg.point_step/4))
            points = points[:, :3]
            if len(points) > self.num_points_pcd:
                idx = np.random.choice(len(points), self.num_points_pcd, replace=False)
                points = points[idx]
            elif len(points) < self.num_points_pcd and len(points) > 0:
                extra = np.random.choice(len(points), self.num_points_pcd - len(points), replace=True)
                points = np.concatenate([points, points[extra]], axis=0)
            
            if len(points) > 0:
                self.latest_cloud_gpu = torch.from_numpy(points.copy()).float().to(self.device)
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error unpacking merged cloud: {e}")

    def control_loop(self, event):
        if self.current_q is None or self.latest_cloud_gpu is None:
            return

        # Get precise fork_tip pose directly from the kinematics tree
        base = torch.eye(4, device=self.device).unsqueeze(0)
        
        # Pad current_q if needed
        q_eval = self.current_q.clone().unsqueeze(0) # Ensure it is shape [1, 7]
        if q_eval.shape[-1] == 7:
            q_eval = torch.cat([q_eval, torch.zeros((1, 2), device=self.device)], dim=-1)
            
        link_poses = self.robot_layer._native_forward_kinematics(q_eval)
        
        if 'fork_tip' not in link_poses:
            rospy.logwarn_throttle(5, "FK failed: 'fork_tip' not found in URDF kinematics tree.")
            return
            
        T_fork = link_poses['fork_tip'][0]
        
        # Check for NaNs early
        if not torch.isfinite(T_fork).all():
            rospy.logerr_throttle(5, "NaN or Inf detected in Fork transformation! Checking current_q...")
            return

        pos_fork = T_fork[:3, 3]
        r1 = T_fork[:3, 0]
        r2 = T_fork[:3, 1]
        curr_pose_9d = torch.cat([pos_fork, r1, r2]).detach().cpu().numpy()
        
        self.obs_queue.append(curr_pose_9d)
        if len(self.obs_queue) < self.obs_queue.maxlen:
            return

        # A. Proprioception: Centered at CURRENT fork tip
        obs_np = np.stack(self.obs_queue)
        obs_centered = obs_np.copy()
        obs_centered[:, :3] -= curr_pose_9d[:3] # Centering at current fork tip
        t_obs = torch.from_numpy(obs_centered).unsqueeze(0).float().to(self.device)
        
        # B. Vision: Merged Fork + Target, Centered at current fork tip
        pcd_target_centered = self.latest_cloud_gpu - pos_fork
        pcd_fork_centered = self.fork_pts_local
        
        # Merge
        pcd_merged_centered = torch.cat([pcd_target_centered, pcd_fork_centered], dim=0)
        
        # Sample / Pad to fixed size
        num_pts = pcd_merged_centered.shape[0]
        if num_pts > self.num_points_pcd:
            idx = torch.randperm(num_pts, device=self.device)[:self.num_points_pcd]
            pcd_final_centered = pcd_merged_centered[idx]
        else:
            pad_idx = torch.randint(0, num_pts, (self.num_points_pcd - num_pts,), device=self.device)
            pcd_final_centered = torch.cat([pcd_merged_centered, pcd_merged_centered[pad_idx]], dim=0)
        
        obs_dict = {
            'point_cloud': pcd_final_centered.unsqueeze(0),
            'agent_pos': t_obs
        }

        # Predict safe trajectory
        try:
            A = self.generate_safe_trajectory(obs_dict, pos_fork, self.obstacle_pts_gpu)
            if A is not None and not torch.isnan(A).any():
                self.publish_joint_trajectory(A, pos_fork)
            elif A is not None:
                rospy.logwarn_throttle(5, "Generated trajectory contains NaNs, skipping...")
        except Exception as e:
            rospy.logerr_throttle(5, f"Trajectory generation error: {e}")

    def generate_safe_trajectory(self, obs_dict, curr_pos, P_obs, N_STEPS=10, T0=0.6, C=0.5, lam=1e-5):
        """Modified SAD-Flower Predictor-Corrector loop for real-time 8Hz."""
        B = 1
        H = 16
        dt = 1.0 / N_STEPS
        dev = self.device
        
        # Validate inputs
        if torch.isnan(obs_dict['point_cloud']).any() or torch.isnan(obs_dict['agent_pos']).any():
            rospy.logerr_throttle(5, "Input observations contain NaNs!")
            return None
            
        obs_norm = {
            'point_cloud': obs_dict['point_cloud'].to(dev),
            'agent_pos': self.normalizer.normalize_obs(obs_dict['agent_pos'].to(dev))
        }

        A = torch.randn(B, H, self.action_dim, device=dev)
        # 9D Warm-start (Bras + effecteur)
        if self._q_warm is None:
            q_init = self.current_q.cpu().numpy()
            if len(q_init) == 7:
                q_init = np.concatenate([q_init, [0.0, 0.0]])
            q_warm = np.tile(q_init, (H, 1))
        else:
            q_warm = self._q_warm
        
        t_start = time.perf_counter()
        t_fm_total = 0.0
        t_ik_total = 0.0
        t_qp_total = 0.0

        for i in range(N_STEPS):
            t_val = i * dt
            t_tensor = torch.full((B,), t_val, device=dev)
            
            t0 = time.perf_counter()
            with torch.no_grad():
                # Midpoint ODE step for much better Flow Matching accuracy
                v1 = self.fm_agent.forward(obs_norm, A, t_tensor)
                A_mid = A + v1 * (dt / 2)
                t_mid_tensor = torch.full((B,), t_val + dt / 2, device=dev)
                v2 = self.fm_agent.forward(obs_norm, A_mid, t_mid_tensor)
                v = v2 # We use v2 as the final velocity for the step
            t_fm_total += (time.perf_counter() - t0)
            
            # Check model output
            if not torch.isfinite(v).all():
                rospy.logerr_throttle(5, f"FM model produced non-finite values at step {i}!")
                return None
            
            # Predictor step
            if t_val < T0 or P_obs is None:
                A = A + v * dt
            else:
                # Corrector step (Physics-aware projection)
                phi_t = C / (1.0 - t_val + 1e-4) ** 2

                A_unnorm = self.normalizer.unnormalize_act(A)
                A_world = A_unnorm.clone()
                A_world[..., :3] += curr_pos

                # Check A_world before SE3 decode
                if not torch.isfinite(A_world).all():
                    rospy.logerr_throttle(5, f"A_world contains non-finite values at step {i}!")
                    return None

                # Additional check for 9D degenerate rotation (c1, c2 columns)
                c1_norm = torch.norm(A_world[..., 3:6], dim=-1)
                c2_norm = torch.norm(A_world[..., 6:9], dim=-1)
                if (c1_norm < 1e-4).any() or (c2_norm < 1e-4).any():
                    rospy.logerr_throttle(5, f"Degenerate 9D rotation detected at step {i} (norms too small)!")
                    return None

                A_next_unnorm = self.normalizer.unnormalize_act(A + v * dt)
                A_next_world = A_next_unnorm.clone()
                A_next_world[..., :3] += curr_pos
                v_world = (A_next_world - A_world).reshape(H, self.action_dim) # displacement

                t1 = time.perf_counter()
                # --- Analytical Jacobians from C++ FastIK ---
                # We only need the first 9 dimensions for the pose
                se3_gpu = decode_9d_to_se3_gpu(A_world[..., :9].reshape(H, 9))
                se3_np = se3_gpu.cpu().numpy()

                # --- Guard NaN : remplace les SE3 corrompus par des valeurs de secours ---
                # On utilise la position actuelle et une rotation identité
                curr_pos_np = curr_pos.cpu().numpy()
                fallback_rotation = np.eye(3)

                for k in range(H):
                    if not np.isfinite(se3_np[k]).all():
                        # Si la rotation est corrompue (NaN ou Inf), on met l'identité
                        if not np.isfinite(se3_np[k, :3, :3]).all():
                            se3_np[k, :3, :3] = fallback_rotation
                        # Si la position est AUSSI corrompue, on met la position actuelle
                        if not np.isfinite(se3_np[k, :3, 3]).all():
                            se3_np[k, :3, 3] = curr_pos_np

                se3_list = [se3_np[j] for j in range(H)]

                try:
                    # solve_batch_with_jacobians returns (q_np, jacobians_list)
                    q_np, Js_list = self.ik_solver.solve_batch_with_jacobians(se3_list, q_warm[0])
                    q_warm = q_np
                    t_ik_total += (time.perf_counter() - t1)

                    t2 = time.perf_counter()
                    q_batch = torch.from_numpy(q_np[:, :7]).float().to(dev).requires_grad_(True)
                    J = torch.from_numpy(np.stack(Js_list)).float().to(dev)[:, :, :7] # [H, 9, 7]

                    # Bernstein Safety Barrier
                    h_q, grad_h_q, _ = self.barrier(q_batch, P_obs)
                    h_q = h_q.detach()
                    grad_h_q = grad_h_q.detach()

                    with torch.no_grad():
                        # Map gradient from joint space to pose space: grad_h_x = J * (JtJ)^-1 * grad_h_q
                        JtJ = torch.bmm(J.transpose(1, 2), J) + lam * torch.eye(7, device=dev).unsqueeze(0)
                        grad_h_x = torch.bmm(J, torch.linalg.solve(JtJ, grad_h_q.unsqueeze(-1))).squeeze(-1)
                        
                        # ==============================================================
                        # CASF: Control-Aware Score-based Flow (Riemannian Metric Warping)
                        # ==============================================================
                        
                        # 1. Isolate the positional part of the task-space gradient
                        n_x_pos = torch.zeros((B, H, self.action_dim), device=dev)
                        n_x_pos[0, :, :3] = grad_h_x[:, :3]

                        # Normalize it to act as a pure directional vector
                        grad_norm_x = n_x_pos.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        n_x_pos = n_x_pos / grad_norm_x

                        # 2. Compute exponential influence weight based on distance h_q
                        # h_q > 0 means safe, h_q <= 0 means penetration
                        kappa = rospy.get_param("~casf_kappa", 50.0)
                        alpha = rospy.get_param("~casf_alpha", 5.0)
                        
                        # We use -h_q so that as distance decreases (approaching obstacle), weight increases.
                        # Clamp the penetration to avoid numerical explosion inside the obstacle
                        w = torch.exp(-kappa * h_q.clamp(min=-0.05)).view(B, H, 1)

                        # 3. Woodbury Matrix Identity for M^-1 * v
                        # M = I + (w * alpha) * n_x_pos * n_x_pos^T
                        # M^-1 v = v - (w*alpha * (n^T v) / (1 + w*alpha * n^T n)) * n
                        w_alpha = w * alpha
                        dot_nv = (v_world * n_x_pos).sum(dim=-1, keepdim=True)
                        dot_nn = (n_x_pos ** 2).sum(dim=-1, keepdim=True)

                        # Apply the metric warp to the raw FM displacement
                        M_inv_disp = v_world - (w_alpha * dot_nv) / (1.0 + w_alpha * dot_nn) * n_x_pos

                        A_world_next = A_world + M_inv_disp

                        A_world_next[..., :3] -= curr_pos
                        A = self.normalizer.normalize_act(A_world_next)
                    t_qp_total += (time.perf_counter() - t2)
                except Exception as e:
                    rospy.logerr_throttle(5, f"Error in corrector step at {i}: {e}")
                    return None

        # Final check
        if not torch.isfinite(A).all():
            rospy.logwarn_throttle(5, "Planner aborted: Final A contains non-finite values.")
            return None

        self._q_warm = q_warm

        t_total = time.perf_counter() - t_start
        rospy.loginfo_throttle(10, f"✅ Plan generated in {t_total*1000:.2f}ms | Stats: FM Inference={t_fm_total*1000:.2f}ms, C++ IK={t_ik_total*1000:.2f}ms, CBF QP={t_qp_total*1000:.2f}ms")
        return A


    def publish_joint_trajectory(self, A_norm, pos_fork):
        """Publish full 16-point joint trajectory for the 150Hz Safety Shield."""
        A_unnorm = self.normalizer.unnormalize_act(A_norm)
        A_world = A_unnorm.clone()
        # pos_fork is already a torch tensor from control_loop
        A_world[..., :3] += pos_fork
        A_world_np = A_world.squeeze(0).detach().cpu().numpy()
        
        self.ensembler.add_prediction(A_world_np)
        smoothed_traj_np = self.ensembler.get_ensembled_trajectory(num_steps=16)
        if smoothed_traj_np is None:
            smoothed_traj_np = A_world_np
            
        # Re-normalize rotation columns after averaging
        for t in range(16):
            col1 = smoothed_traj_np[t, 3:6]
            col2 = smoothed_traj_np[t, 6:9]
            col1 = col1 / (np.linalg.norm(col1) + 1e-8)
            col2 = col2 - np.dot(col2, col1) * col1
            col2 = col2 / (np.linalg.norm(col2) + 1e-8)
            smoothed_traj_np[t, 3:6] = col1
            smoothed_traj_np[t, 6:9] = col2
        
        se3_list = []
        for i in range(16):
            se3 = decode_9d_to_se3_gpu(
                torch.from_numpy(smoothed_traj_np[i, :9]).to(self.device).unsqueeze(0)
            )[0].cpu().numpy()
            
            # Guard : vérifier que R est orthogonale (det ≈ 1 et trace entre -1.0 et 3.0)
            R = se3[:3, :3]
            det = np.linalg.det(R)
            tr = np.trace(R)
            if not np.isfinite(det) or abs(det - 1.0) > 0.15 or not (-1.0 <= tr <= 3.0):
                # Fallback : rotation identité, position du waypoint
                se3[:3, :3] = np.eye(3)
                if not np.isfinite(se3[:3, 3]).all():
                    se3[:3, 3] = pos_fork.cpu().numpy()
            se3_list.append(se3)
        
        # Pad current_q to 9 if it has 7 elements (IK solver expects 9: 7 arm + 2 fingers)
        q_init = self.current_q.cpu().numpy()
        if len(q_init) == 7:
            q_init = np.concatenate([q_init, [0.0, 0.0]])
            
        q_traj = self.ik_solver.solve_batch(se3_list, q_init)
        
        if len(q_traj) == 0: return
        
        # --- Visualization (MarkerArray) ---
        marker_array = MarkerArray()
        
        # Line strip for trajectory path
        line_marker = Marker()
        line_marker.header.frame_id = "world"
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = "nominal_trajectory"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.01 # Line width
        line_marker.color.a = 1.0
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        
        for i, se3 in enumerate(se3_list):
            p = Point()
            p.x = se3[0, 3]
            p.y = se3[1, 3]
            p.z = se3[2, 3]
            line_marker.points.append(p)
            
            # Spheres for waypoints
            sphere_marker = Marker()
            sphere_marker.header.frame_id = "world"
            sphere_marker.header.stamp = rospy.Time.now()
            sphere_marker.ns = "nominal_waypoints"
            sphere_marker.id = i + 1
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.pose.position = p
            sphere_marker.scale.x = 0.02
            sphere_marker.scale.y = 0.02
            sphere_marker.scale.z = 0.02
            sphere_marker.color.a = 1.0
            sphere_marker.color.r = 1.0
            sphere_marker.color.g = 1.0
            sphere_marker.color.b = 0.0
            marker_array.markers.append(sphere_marker)
            
        marker_array.markers.append(line_marker)
        self.viz_traj_pub.publish(marker_array)
        # ----------------------------------
        
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        
        dt_wp = 0.3 # 300ms between waypoints for interpolation (Lowers max speed of the robot)
        for i in range(len(q_traj)):
            point = JointTrajectoryPoint()
            point.positions = q_traj[i][:7].tolist()
            point.time_from_start = rospy.Duration(i * dt_wp)
            msg.points.append(point)
            
        self.traj_pub.publish(msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = SafeGenerativeNode()
    node.run()
