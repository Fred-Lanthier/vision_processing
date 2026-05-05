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
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

# Standard imports for this workspace
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
sys.path.insert(0, pkg_path)
sys.path.append(os.path.join(pkg_path, 'src', 'vision_processing', 'diffusion_model_train'))

from Train_Fork_FM import FlowMatchingAgent, Normalizer
from vision_processing import fast_ik_module
from third_party.SDF_Bernstein_Basis.src.rdf_weights import RDF_Weights
from third_party.SDF_Bernstein_Basis.bernstein_core import BernsteinCore
from third_party.SDF_Bernstein_Basis.bernstein_barrier import BernsteinBarrier
from third_party.RDF.urdf_layer import URDFLayer
from utils import compute_T_child_parent_xacro
from Compute_3D_point_cloud_from_mesh import RobotMeshLoaderOptimized

def decode_9d_to_se3_gpu(traj_9d):
    """
    Orthogonalisation Gram-Schmidt 100% vectorisée sur GPU.
    Zéro boucle for, zéro transfert CPU pendant le calcul.
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
    
    # Gram-Schmidt vectorisé
    b1 = torch.nn.functional.normalize(c1, dim=-1)
    dot_product = torch.sum(b1 * c2, dim=-1, keepdim=True)
    b2 = c2 - dot_product * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    
    # Construction du batch de matrices 4x4
    N = traj.shape[0]
    T = torch.eye(4, device=traj_9d.device).unsqueeze(0).repeat(N, 1, 1)
    T[:, :3, 0] = b1
    T[:, :3, 1] = b2
    T[:, :3, 2] = b3
    T[:, :3, 3] = pos
    
    return T

class SafeGenerativeNode:
    def __init__(self):
        rospy.init_node('safe_generative_node')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paths and Parameters
        urdf_path_raw = rospy.get_param("~urdf_path", pkg_path + '/urdf/panda_camera.xacro')
        
        # Global Xacro -> URDF conversion to fix FastIK and other C++ components
        if urdf_path_raw.endswith('.xacro'):
            rospy.loginfo(f"⏳ Processing XACRO: {urdf_path_raw}")
            doc = xacro.process_file(urdf_path_raw)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
                f.write(doc.toxml())
                urdf_path = f.name
            rospy.loginfo(f"🔥 Processed XACRO to temporary URDF: {urdf_path}")
        else:
            urdf_path = urdf_path_raw

        voxel_dir = rospy.get_param("~voxel_dir", pkg_path + '/third_party/RDF/panda_layer/meshes/voxel_128')
        weights_dir = rospy.get_param("~weights_dir", pkg_path + '/third_party/SDF_Bernstein_Basis/panda_test')
        model_name = rospy.get_param("~model_name", "best_fm_model_high_dim_CFM_relative_dropout.ckpt")
        model_path = os.path.join(pkg_path, 'models', model_name)
        
        # 1. Initialize Bernstein Safety components
        # package_dir set to pkg_path because URDF is now in /tmp
        self.robot_layer = URDFLayer(urdf_path=urdf_path, device=self.device, package_dir=pkg_path, voxel_dir=voxel_dir)
        weight_handler = RDF_Weights(device=self.device, dtype=torch.float32)
        weight_handler.init_robot_folder(weights_dir, robot_name='panda')
        link_names = ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7']
        weight_handler.add_models(link_names, robot_name='panda')
        self.bernstein_core = BernsteinCore(weight_handler, self.robot_layer, self.device, link_names)
        self.barrier = BernsteinBarrier(self.bernstein_core, d_safe=0.08)
        
        # 2. Initialize IK Solver (Ultra-fast Pybind11 Pinocchio)
        # Now solving for fork_tip to match the model's 9D pose
        self.ik_solver = fast_ik_module.FastIK(urdf_path, "fork_tip")
        self._q_warm = None
        
        # 3. Initialize Fork-related components
        rospy.loginfo("🍴 Loading Fork Mesh and Transform...")
        self.T_tcp_fork_tip = torch.from_numpy(compute_T_child_parent_xacro(urdf_path, 'fork_tip', 'panda_TCP')).float().to(self.device)
        self.mesh_loader = RobotMeshLoaderOptimized(urdf_path)
        # We only need the fork points in its local frame
        fork_pts = self.mesh_loader.static_point_clouds.get('fork_tip', np.zeros((0, 3)))
        self.fork_pts_local = torch.from_numpy(fork_pts).float().to(self.device)
        rospy.loginfo(f"🍴 Fork Mesh loaded with {len(self.fork_pts_local)} points.")

        # 4. Initialize Flow Matching Agent
        rospy.loginfo(f"🛡️  SafeGenerativeNode: Loading Model: {model_name}")
        payload = torch.load(model_path, map_location=self.device)
        cfg = payload.get('config', {})
        self.fm_agent = FlowMatchingAgent(
            action_dim=cfg.get('action_dim', 9),
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
        self.obstacle_pts_gpu = None
        self.obs_queue = collections.deque(maxlen=cfg.get('obs_horizon', 2))
        self.num_points_pcd = cfg.get('num_points', 256)
        self.latest_cloud_gpu = None
        
        # 5. Subscribers & Publishers
        rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        rospy.Subscriber('/perception/obstacles', PointCloud2, self.obstacle_callback)
        # On s'abonne aussi à obstacles pour le conditionnement si merged_cloud n'est pas là
        rospy.Subscriber('/perception/obstacles', PointCloud2, self.cloud_callback)
        # rospy.Subscriber('/vision/merged_cloud', PointCloud2, self.cloud_callback)
        
        self.traj_pub = rospy.Publisher('/planner/nominal_trajectory', JointTrajectory, queue_size=1)
        
        # Control Loop at 8Hz (Planner rate)
        self.rate_hz = 8.0
        rospy.Timer(rospy.Duration(1.0/self.rate_hz), self.control_loop)
        rospy.loginfo("🚀 Safe Generative Planner Node Initialized at 8Hz")

    def joint_callback(self, msg):
        self.current_q = torch.tensor(msg.position[:7], dtype=torch.float32, device=self.device)

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

        # Prepare EE and Fork-Tip poses
        base = torch.eye(4, device=self.device).unsqueeze(0)
        T_ee = self.robot_layer.get_transformations_each_link(base, self.current_q.unsqueeze(0))[-1][0]
        
        # Fork Tip Pose (world)
        T_fork = T_ee @ self.T_tcp_fork_tip
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
        # Target cloud is already in world frame (from cloud_callback)
        pcd_target_centered = self.latest_cloud_gpu - pos_fork
        
        # Fork points are static in fork_tip frame, so they are ALREADY centered at fork tip
        pcd_fork_centered = self.fork_pts_local
        
        # Merge
        pcd_merged_centered = torch.cat([pcd_target_centered, pcd_fork_centered], dim=0)
        
        # Sample / Pad to fixed size (e.g., 1024)
        num_pts = pcd_merged_centered.shape[0]
        if num_pts > self.num_points_pcd:
            idx = torch.randperm(num_pts, device=self.device)[:self.num_points_pcd]
            pcd_final_centered = pcd_merged_centered[idx]
        else:
            # Padding
            pad_idx = torch.randint(0, num_pts, (self.num_points_pcd - num_pts,), device=self.device)
            pcd_final_centered = torch.cat([pcd_merged_centered, pcd_merged_centered[pad_idx]], dim=0)
        
        obs_dict = {
            'point_cloud': pcd_final_centered.unsqueeze(0),
            'agent_pos': t_obs
        }

        # Predict safe trajectory via PC-Integrator
        A = self.generate_safe_trajectory(obs_dict, pos_fork, self.obstacle_pts_gpu)

        if A is not None:
            self.publish_joint_trajectory(A, pos_fork)

    def generate_safe_trajectory(self, obs_dict, curr_pos, P_obs, N_STEPS=10, T0=0.6, C=0.5, lam=1e-5):
        """Modified SAD-Flower Predictor-Corrector loop for real-time 8Hz."""
        B = 1
        H = 16
        dt = 1.0 / N_STEPS
        dev = self.device
        
        obs_norm = {
            'point_cloud': obs_dict['point_cloud'].to(dev),
            'agent_pos': self.normalizer.normalize(obs_dict['agent_pos'].to(dev))
        }

        A = torch.randn(B, H, 9, device=dev)
        # 9D Warm-start (Bras + effecteur)
        if self._q_warm is None:
            q_init = self.current_q.cpu().numpy()
            if len(q_init) == 7:
                q_init = np.concatenate([q_init, [0.0, 0.0]])
            q_warm = np.tile(q_init, (H, 1))
        else:
            q_warm = self._q_warm
        
        t_start = time.perf_counter()
        
        for i in range(N_STEPS):
            t = i * dt
            t_tensor = torch.full((B,), t, device=dev)
            
            with torch.no_grad():
                v = self.fm_agent.forward(obs_norm, A, t_tensor)
            
            # Predictor step
            if t < T0 or P_obs is None:
                A = A + v * dt
            else:
                # Corrector step (Physics-aware projection)
                phi_t = C / (1.0 - t + 1e-4) ** 2
                
                A_unnorm = self.normalizer.unnormalize(A)
                A_world = A_unnorm.clone()
                A_world[..., :3] += curr_pos
                
                A_next_unnorm = self.normalizer.unnormalize(A + v * dt)
                A_next_world = A_next_unnorm.clone()
                A_next_world[..., :3] += curr_pos
                v_world = (A_next_world - A_world).reshape(H, 9)
                
                # Fast batch IK
                se3_gpu = decode_9d_to_se3_gpu(A_world.reshape(H, 9))
                se3_list = [se3_gpu[j].cpu().numpy() for j in range(H)]
                q_np = self.ik_solver.solve_batch(se3_list, q_warm[0])
                q_warm = q_np
                
                q_batch = torch.from_numpy(q_np[:, :7]).float().to(dev).requires_grad_(True)
                
                # Bernstein Safety Barrier
                h_q, grad_h_q, _ = self.barrier(q_batch, P_obs)
                
                with torch.no_grad():
                    # Exact Analytical Gradient Mapping from Joint to 9D Pose Space
                    J = torch.func.vmap(torch.func.jacfwd(self.fk_9d_single))(q_batch.detach())
                    
                    JJt = torch.bmm(J, J.transpose(1, 2)) + lam * torch.eye(9, device=dev).unsqueeze(0)
                    rhs = torch.bmm(J, grad_h_q.unsqueeze(-1))
                    grad_h_x = torch.linalg.solve(JJt, rhs).squeeze(-1)
                    
                    # Closed-form QP in R9
                    dot = (grad_h_x * v_world).sum(-1)
                    constr = dot + phi_t * h_q.detach()
                    denom = (grad_h_x**2).sum(-1).clamp(min=1e-8)
                    lam_qp = torch.clamp(-constr / denom, min=0)
                    u_world = v_world + lam_qp.unsqueeze(-1) * grad_h_x
                    
                    A_world_next = A_world + u_world.reshape(B, H, 9) * dt
                    A_world_next[..., :3] -= curr_pos
                    A = self.normalizer.normalize(A_world_next)

        self._q_warm = q_warm
        rospy.loginfo_throttle(10, f"✅ Plan generated in {(time.perf_counter() - t_start)*1000:.2f}ms")
        return A

    def fk_9d_single(self, q_single):
        """FK for 9D target used in autograd gradient mapping (fork_tip)."""
        q_b = q_single.unsqueeze(0)
        base = torch.eye(4, device=self.device, dtype=q_single.dtype).unsqueeze(0)
        T_ee = self.robot_layer.get_transformations_each_link(base, q_b)[-1][0]
        T_fork = T_ee @ self.T_tcp_fork_tip.to(q_single.dtype)
        # 9D: [Pos X,Y,Z, Rot Col 1, Rot Col 2]
        return torch.cat([T_fork[:3, 3], T_fork[:3, 0], T_fork[:3, 1]])

    def publish_joint_trajectory(self, A_norm, pos_fork):
        """Publish full 16-point joint trajectory for the 150Hz Safety Shield."""
        A_unnorm = self.normalizer.unnormalize(A_norm)
        A_world = A_unnorm.clone()
        # pos_fork is already a torch tensor from control_loop
        A_world[..., :3] += pos_fork
        A_world_np = A_world.squeeze(0).detach().cpu().numpy()
        
        # Batch IK for final joint trajectory (solving for fork_tip)
        se3_list = [decode_9d_to_se3_gpu(torch.from_numpy(A_world_np[i]).to(self.device).unsqueeze(0))[0].cpu().numpy() for i in range(16)]
        
        # Pad current_q to 9 if it has 7 elements (IK solver expects 9: 7 arm + 2 fingers)
        q_init = self.current_q.cpu().numpy()
        if len(q_init) == 7:
            q_init = np.concatenate([q_init, [0.0, 0.0]])
            
        q_traj = self.ik_solver.solve_batch(se3_list, q_init)
        
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        
        dt_wp = 0.1 # 100ms between waypoints for interpolation
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
