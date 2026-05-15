#!/usr/bin/env python3
import rospy
import tf
import tf.transformations as tft
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
from geometry_msgs.msg import Point, Pose, PoseArray
from std_msgs.msg import Header, Float64MultiArray
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

# ==============================================================
# TEMPORAL ENSEMBLER
# ==============================================================
class TemporalEnsembler:
    def __init__(self, pred_horizon, buffer_size=3, weights='exponential'):
        self.pred_horizon = pred_horizon
        self.buffer = collections.deque(maxlen=buffer_size)
        self.weights_type = weights
        self.wp_dt = 0.1  # 1.6 seconds / 16 waypoints
    
    def add_prediction(self, prediction, current_time):
        self.buffer.append((current_time, prediction.copy()))
    
    def get_ensembled_trajectory(self, num_steps, current_time):
        if len(self.buffer) == 0: return None
        
        trajectory = []
        for i in range(num_steps):
            actions, weights = [], []
            
            for (timestamp, prediction) in self.buffer:
                # How much time has passed since this prediction was made
                age_sec = current_time - timestamp
                # Convert time to waypoint indices
                index_offset = int(round(age_sec / self.wp_dt))
                pred_index = index_offset + i
                
                if 0 <= pred_index < self.pred_horizon:
                    actions.append(prediction[pred_index])
                    w = np.exp(-1.0 * age_sec) if self.weights_type == 'exponential' else 1.0
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
scripts_path = os.path.join(pkg_path, 'scripts')
sys.path.insert(0, pkg_path)
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)
sys.path.append(os.path.join(pkg_path, 'src', 'vision_processing', 'diffusion_model_train'))

from Train_Fork_FM import FlowMatchingAgent, Normalizer
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
        rospy.init_node('casf_generative_node_PC')
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
        model_name = rospy.get_param("~model_name", "best_fm_model_9D_dynamics_1024.ckpt")
        # model_name = rospy.get_param("~model_name", "best_fm_model_high_dim_CFM_relative_dropout.ckpt")
        model_path = os.path.join(pkg_path, 'models', model_name)
        
        # 1. Initialize Bernstein Safety components
        self.robot_layer = URDFLayer(urdf_path=urdf_path, device=self.device, package_dir=pkg_path, voxel_dir=voxel_dir)
        weight_handler = RDF_Weights(device=self.device, dtype=torch.float32)
        weight_handler.init_robot_folder(weights_dir, robot_name='panda')
        link_names = ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 'panda_hand', 'panda_leftfinger', 'panda_rightfinger', 'fork_tip']
        weight_handler.add_models(link_names, robot_name='panda')
        self.bernstein_core = BernsteinCore(weight_handler, self.robot_layer, self.device, link_names)
        self.d_safe = float(rospy.get_param("~d_safe", 0.005))
        # alpha=0.001 matches the CBF 150Hz node: softmin bias ≈ alpha*log(N) ≈ 4.6mm for N=100,
        # acceptable for d_safe in mm range. Default 0.01 would bias h by 46mm, firing too early.
        self.barrier = BernsteinBarrier(self.bernstein_core, d_safe=self.d_safe, alpha=0.001)
        # Body links (link0→panda_hand) used for self-filter; end-effector links excluded so
        # real obstacle points near the fork tip are never silently removed.
        self.n_body_links = 9  # indices 0-8 in link_names (panda_link0 … panda_hand)
        
        # 2. Initialize IK Solver (Ultra-fast Pybind11 Pinocchio)
        # The FM model predicts fork-tip poses, but the robot IK target is the TCP.
        # This matches scripts/trajectory_node_Fork.py:
        #   T_world_fork = T_world_tcp @ T_tcp_fork_tip
        #   T_world_tcp = T_world_fork @ inv(T_tcp_fork_tip)
        xacro_file = os.path.join(pkg_path, 'urdf', 'panda_camera.xacro')
        self.T_tcp_fork_tip = compute_T_child_parent_xacro(xacro_file, 'fork_tip', 'panda_TCP')
        if self.T_tcp_fork_tip is None:
            raise RuntimeError("Could not resolve fixed transform panda_TCP -> fork_tip")
        self.T_tcp_fork_tip_inv = np.linalg.inv(self.T_tcp_fork_tip)
        self.ik_solver = fast_ik_module.FastIK(urdf_path, "panda_TCP")
        rospy.loginfo("✅ FastIK targets panda_TCP; FM fork-tip poses are converted to TCP poses")
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
        self.action_dim = cfg.get('action_dim', 9)
        self.obs_dim = cfg.get('obs_dim', 9)
        self.fm_agent = FlowMatchingAgent(
            action_dim=self.action_dim,
            obs_horizon=cfg.get('obs_horizon', 2),
            pred_horizon=cfg.get('pred_horizon', 16),
            encoder_output_dim=cfg.get('encoder_output_dim', 64),
            diffusion_step_embed_dim=cfg.get('diffusion_step_embed_dim', 256),
            down_dims=cfg.get('down_dims', [256, 512, 1024]),
            stats=payload.get('stats', None)
        ).to(self.device)
        self.fm_agent.load_state_dict(payload['model_state_dict'])
        self.normalizer = self.fm_agent.normalizer
        if not self.normalizer.is_initialized and 'stats' in payload:
            self.normalizer.load_stats_from_dict(payload['stats'])
        self.fm_agent.eval()

        # 4. State variables
        self.current_q = None
        self.commanded_q = None
        self.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        self.obstacle_pts_gpu = None
        self.obs_queue = collections.deque(maxlen=cfg.get('obs_horizon', 2))
        self.num_points_pcd = cfg.get('num_points', 256)
        self.obstacle_pre_sample_max = int(rospy.get_param("~obstacle_pre_sample_max", 2000))
        self.planner_cbf_max_obstacles = int(rospy.get_param("~planner_cbf_max_obstacles", 512))
        self.casf_spatial_max_obstacles = int(rospy.get_param("~casf_spatial_max_obstacles", 100))
        self.check_finite_debug = bool(rospy.get_param("~check_finite_debug", False))
        self.merged_cloud_contains_fork = rospy.get_param("~merged_cloud_contains_fork", True)
        self.single_shot = rospy.get_param("~single_shot", True)
        self.tcp_frame = rospy.get_param("~tcp_frame", "panda_TCP")
        # CASF ODE parameters (improvement #1: read once, not per-iteration)
        self.casf_kappa = float(rospy.get_param("~casf_kappa", 50.0))
        self.casf_alpha = float(rospy.get_param("~casf_alpha", 5.0))
        # Improvement #3: correction / prediction step counts exposed as params
        self.n_pred = int(rospy.get_param("~n_pred", 5))
        self.n_corr = int(rospy.get_param("~n_corr", 5))
        self.alpha_corr = float(rospy.get_param("~alpha_corr", 2.0))
        self.plan_published = False
        self.is_committed = False
        self.fork_food_distance = float('inf')
        self.latest_cloud_gpu = None
        self.ensembler = TemporalEnsembler(pred_horizon=cfg.get('pred_horizon', 16), buffer_size=3)
        self.last_pose_source_log = 0.0

        
        # 5. Subscribers & Publishers
        rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        rospy.Subscriber('/joint_group_position_controller/command', Float64MultiArray, self.command_callback)
        rospy.Subscriber('/perception/cleaned_obstacles', PointCloud2, self.obstacle_callback)
        rospy.Subscriber('/vision/merged_cloud', PointCloud2, self.cloud_callback)
        from std_msgs.msg import Float32
        rospy.Subscriber('/vision/fork_food_distance', Float32, self.dist_callback)
        
        self.traj_pub = rospy.Publisher('/planner/nominal_trajectory', JointTrajectory, queue_size=1, latch=False)
        self.fork_pose_traj_pub = rospy.Publisher('/planner/nominal_fork_trajectory', PoseArray, queue_size=1, latch=True)
        self.viz_traj_pub = rospy.Publisher('/viz/nominal_trajectory_3d', MarkerArray, queue_size=1, latch=True)
        self.pub_model_pcd = rospy.Publisher('/viz/model_input_pcd', PointCloud2, queue_size=1, latch=True)
        
        # Control Loop at 3Hz (Planner rate)
        self.rate_hz = 3.0
        rospy.Timer(rospy.Duration(1.0/self.rate_hz), self.control_loop)
        rospy.loginfo("🚀 Safe Generative Planner Node Initialized at 3Hz")

    def command_callback(self, msg):
        try:
            from std_msgs.msg import Float64MultiArray
            if len(msg.data) >= 7:
                self.commanded_q = torch.tensor(msg.data[:7], dtype=torch.float32, device=self.device)
        except Exception as e:
            rospy.logerr_throttle(5, f"Error in command_callback: {e}")

    def dist_callback(self, msg):
        self.fork_food_distance = msg.data

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
            if len(points) > self.obstacle_pre_sample_max:
                idx = np.random.choice(len(points), self.obstacle_pre_sample_max, replace=False)
                points = points[idx]
            self.obstacle_pts_gpu = torch.from_numpy(points.copy()).float().to(self.device)
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error unpacking obstacles: {e}")

    def cloud_callback(self, msg):
        # Conditioned point cloud for FM Encoder
        try:
            points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, int(msg.point_step/4))
            points = points[:, :3]
            
            # Explicitly clear the cache if perception sends an empty cloud
            if len(points) == 0:
                self.latest_cloud_gpu = None
                return
                
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
        t_capture = rospy.get_time()
        
        if self.single_shot and self.plan_published:
            return
            
        if self.is_committed:
            # We already published the open-loop strike, do not replan.
            return

        if self.current_q is None or self.latest_cloud_gpu is None:
            return

        # Get precise fork_tip pose directly from the kinematics tree
        base = torch.eye(4, device=self.device).unsqueeze(0)
        
        # Revert to physical current_q so the Point Cloud (from perception) perfectly aligns with the base pose
        q_source = self.current_q
        q_eval = q_source.clone().unsqueeze(0) # Ensure it is shape [1, 7]
        if q_eval.shape[-1] == 7:
            q_eval = torch.cat([q_eval, torch.zeros((1, 2), device=self.device)], dim=-1)
            
        link_poses = self.robot_layer._native_forward_kinematics(q_eval)
        
        if 'fork_tip' not in link_poses:
            rospy.logwarn_throttle(5, "FK failed: 'fork_tip' not found in URDF kinematics tree.")
            return
            
        # Strictly use Forward Kinematics (FK) to avoid TF latency jitter
        T_fork = link_poses['fork_tip'][0]
        
        # Check for NaNs early
        if self.check_finite_debug and not torch.isfinite(T_fork).all():
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
        
        # B. Vision: translate-center while preserving the world orientation.
        # /vision/merged_cloud from condition_pcd_from_perception.py already contains
        # the current fork mesh and target in world frame, matching the training
        # Merged_Fork preprocessing. Do not append a second canonical local fork.
        pcd_merged_centered = self.latest_cloud_gpu - pos_fork
        if not self.merged_cloud_contains_fork:
            pcd_merged_centered = torch.cat([pcd_merged_centered, self.fork_pts_local], dim=0)
        
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

        # --- VISUALIZE MODEL INPUT POINT CLOUD ---
        # Shift back to world frame for visualization
        pcd_world = pcd_final_centered + pos_fork
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"
        cloud_msg = pc2.create_cloud_xyz32(header, pcd_world.cpu().numpy())
        self.pub_model_pcd.publish(cloud_msg)

        # Predict safe trajectory
        try:
            A = self.generate_safe_trajectory(obs_dict, pos_fork, self.obstacle_pts_gpu, q_source)
            if A is not None and not (self.check_finite_debug and torch.isnan(A).any()):
                # Check for Commit Mode
                COMMIT_DISTANCE = 0.035 # 3.5 cm
                use_ensembler = True
                if self.fork_food_distance < COMMIT_DISTANCE:
                    rospy.loginfo(f"🔒 COMMIT MODE (d={self.fork_food_distance*1000:.1f}mm) — executing final strike open-loop")
                    self.is_committed = True
                    use_ensembler = False

                self.publish_joint_trajectory(A, pos_fork, q_source, t_capture, use_ensembler=use_ensembler)
                self.plan_published = True
                
                if self.single_shot:
                    rospy.loginfo("✅ Single-shot trajectory generated and latched. Planner will not replan.")
            elif A is not None:
                rospy.logwarn_throttle(5, "Generated trajectory contains NaNs, skipping...")
        except Exception as e:
            rospy.logerr_throttle(5, f"Trajectory generation error: {e}")

    def fork_se3_to_tcp_se3(self, T_world_fork):
        return T_world_fork @ self.T_tcp_fork_tip_inv

    def generate_safe_trajectory(self, obs_dict, curr_pos, P_obs, q_source, lam=1e-5):
        """PC-CASF: Prediction-Correction with CASF Metric Warping."""
        B = 1
        H = 16
        dev = self.device
        N_PRED = self.n_pred
        N_CORR = self.n_corr
        alpha_corr = self.alpha_corr
        
        # Validate inputs
        if self.check_finite_debug and (
            torch.isnan(obs_dict['point_cloud']).any() or torch.isnan(obs_dict['agent_pos']).any()
        ):
            rospy.logerr_throttle(5, "Input observations contain NaNs!")
            return None
            
        obs_norm = {
            'point_cloud': obs_dict['point_cloud'].to(dev),
            'agent_pos': self.normalizer.normalize(obs_dict['agent_pos'].to(dev))
        }

        # ---------------------------------------------------------
        # PHASE 1: PREDICTION (Pure Denoising)
        # ---------------------------------------------------------
        # Start from pure noise
        A = torch.randn(B, H, self.action_dim, device=dev)
        dt_pred = 1.0 / N_PRED
        
        t_start = time.perf_counter()
        t_fm_encode = 0.0
        t_fm_pred = 0.0

        t0 = time.perf_counter()
        with torch.inference_mode():
            global_cond = self.fm_agent.encode_obs(obs_norm)
        t_fm_encode = time.perf_counter() - t0

        for i in range(N_PRED):
            t_val = i * dt_pred
            t_tensor = torch.full((B,), t_val, device=dev)
            
            t0 = time.perf_counter()
            with torch.inference_mode():
                # Euler ODE step. The observation encoder is cached once per rollout.
                v = self.fm_agent.velocity_net(A, t_tensor, global_cond)
            t_fm_pred += (time.perf_counter() - t0)
            
            if self.check_finite_debug and not torch.isfinite(v).all():
                rospy.logerr_throttle(5, f"FM model produced non-finite values during prediction step {i}!")
                return None
            
            A = A + v * dt_pred

        # ---------------------------------------------------------
        # PHASE 2: CORRECTION (Soft CASF Deflection)
        # ---------------------------------------------------------
        # Initialize IK warm-start
        if self._q_warm is None:
            q_init = q_source.cpu().numpy()
            if len(q_init) == 7:
                q_init = np.concatenate([q_init, [0.0, 0.0]])
            q_warm = np.tile(q_init, (H, 1))
        else:
            q_warm = self._q_warm
            
        dt_corr = 1.0 / N_CORR
        t_fm_corr = 0.0
        t_ik_total = 0.0
        t_casf_total = 0.0
        
        if P_obs is not None and P_obs.shape[0] > 0:
            if P_obs.shape[0] > self.planner_cbf_max_obstacles:
                with torch.no_grad():
                    dist_to_fork = torch.norm(P_obs - curr_pos, dim=1)
                    _, top_idx = torch.topk(
                        dist_to_fork,
                        k=self.planner_cbf_max_obstacles,
                        largest=False,
                    )
                    P_obs = P_obs[top_idx]

            # Step 1: Self-filter — remove depth-noise points on the robot arm body.
            # Mirrors the CBF 150Hz self-filter: keep only points whose minimum SDF
            # over body links (link0…panda_hand, indices 0:n_body_links) exceeds 5mm.
            # End-effector links are excluded so real obstacle points near the fork are kept.
            with torch.no_grad():
                q9 = torch.cat([q_source.unsqueeze(0),
                                 torch.zeros((1, 2), device=dev)], dim=-1)
                _, sdf_per_link = self.bernstein_core.get_whole_body_sdf_batch(
                    P_obs,
                    torch.eye(4, device=dev).unsqueeze(0),
                    q9,
                    return_per_link=True
                )
                # sdf_per_link: [1, K, N_obs]; slice body links only
                sdf_body = sdf_per_link[0, :self.n_body_links, :].min(dim=0).values
                # Only remove points inside the mesh (SDF < 0 = depth-noise self-hits).
                # Real obstacle points at the surface have SDF >= 0 and must be kept.
                not_self = sdf_body > -0.003
                P_obs = P_obs[not_self]

            # Step 2: Spatial filter — top-100 closest to fork tip (fast proxy for danger)
            if P_obs.shape[0] > 0:
                dist_to_fork = torch.norm(P_obs - curr_pos, dim=1)
                k_spatial = min(self.casf_spatial_max_obstacles, self.planner_cbf_max_obstacles)
                if P_obs.shape[0] > k_spatial:
                    _, top_idx = torch.topk(dist_to_fork, k=k_spatial, largest=False)
                    P_obs_filtered = P_obs[top_idx]
                else:
                    P_obs_filtered = P_obs
            else:
                P_obs_filtered = P_obs

            if P_obs_filtered.shape[0] == 0:
                rospy.logdebug_throttle(5.0, "CASF: all obstacle points removed by self-filter, skipping correction.")

            # range(0) is empty — loop body is skipped when self-filter removed everything
            for i in range(N_CORR if P_obs_filtered.shape[0] > 0 else 0):
                # We scale time t_val from 0 to 1 over the correction phase to vanish the flow
                tau = (i + 0.5) * dt_corr 
                t_tensor = torch.full((B,), tau, device=dev)
                
                t0 = time.perf_counter()
                with torch.inference_mode():
                    v_theta = self.fm_agent.velocity_net(A, t_tensor, global_cond)
                t_fm_corr += (time.perf_counter() - t0)
                
                # Restorative flow: alpha * (1 - tau) * v_theta
                v_nom = alpha_corr * (1.0 - tau) * v_theta
                
                # 1. Unnormalize to get physical poses of Fork Tip
                A_unnorm = self.normalizer.unnormalize(A)
                A_world = A_unnorm.clone()
                A_world[..., :3] += curr_pos
                
                if self.check_finite_debug and not torch.isfinite(A_world).all():
                    rospy.logerr_throttle(5, f"A_world contains non-finite values in correction step {i}!")
                    return None

                # 2. Get Fork Tip SE(3) matrices
                t1 = time.perf_counter()
                se3_fork_gpu = decode_9d_to_se3_gpu(A_world.view(H, 9))
                se3_fork_np = se3_fork_gpu.cpu().numpy()
                
                se3_tcp_list = []
                for k in range(H):
                    T_world_fork = se3_fork_np[k]
                    if not np.isfinite(T_world_fork).all():
                        T_world_fork[:3, :3] = np.eye(3)
                        if not np.isfinite(T_world_fork[:3, 3]).all():
                            T_world_fork[:3, 3] = curr_pos.cpu().numpy()
                    se3_tcp_list.append(self.fork_se3_to_tcp_se3(T_world_fork))
                
                # 3. Solve IK for the robot TCP pose corresponding to each fork-tip pose.
                try:
                    q_np, Js_list = self.ik_solver.solve_batch_with_jacobians(se3_tcp_list, q_warm[0])
                    q_warm = q_np
                except Exception as e:
                    rospy.logerr_throttle(5, f"FastIK failed in correction step {i}: {e}")
                    return None
                t_ik_total += (time.perf_counter() - t1)
                
                # 4. Evaluate RDF and Apply CASF (Position Only)
                t2 = time.perf_counter()
                q_batch = torch.from_numpy(q_np[:, :7]).float().to(dev).requires_grad_(True)
                
                # J_full is [H, 6, 9]. We slice the first 3 rows for pos, and first 7 cols for arm joints.
                J_full = torch.from_numpy(np.stack(Js_list)).float().to(dev) 
                J_pos = J_full[:, :3, :7] # [H, 3, 7]
                
                h_q, grad_h_q, _ = self.barrier(q_batch, P_obs_filtered)
                h_q = h_q.detach()
                grad_h_q = grad_h_q.detach()
                
                with torch.no_grad():
                    # Pull back gradient: grad_h_pos = J_pos * (J_pos^T J_pos + lam I)^-1 * grad_h_q
                    JtJ = torch.bmm(J_pos.transpose(1, 2), J_pos) + lam * torch.eye(7, device=dev).unsqueeze(0)
                    grad_h_pos = torch.bmm(J_pos, torch.linalg.solve(JtJ, grad_h_q.unsqueeze(-1))).squeeze(-1) # [H, 3]
                    
                    n_pos = torch.zeros((B, H, 3), device=dev)
                    n_pos[0, :, :] = grad_h_pos
                    
                    grad_norm = n_pos.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    n_pos = n_pos / grad_norm
                    
                    # Weight based on distance (params cached at init — no ROS param call in hot path)
                    w = torch.exp(-self.casf_kappa * h_q.clamp(min=-0.05)).view(B, H, 1)
                    w_alpha = w * self.casf_alpha
                    
                    # Unnormalize v_nom to get physical displacement
                    A_next_unnorm = self.normalizer.unnormalize(A + v_nom * dt_corr)
                    A_next_world = A_next_unnorm.clone()
                    A_next_world[..., :3] += curr_pos
                    v_world = (A_next_world - A_world).reshape(B, H, 9)
                    
                    v_world_pos = v_world[..., :3]
                    
                    # Woodbury CASF Warp purely on Position
                    dot_nv = (v_world_pos * n_pos).sum(dim=-1, keepdim=True)
                    dot_nn = (n_pos ** 2).sum(dim=-1, keepdim=True)
                    
                    M_inv_disp_pos = v_world_pos - (w_alpha * dot_nv) / (1.0 + w_alpha * dot_nn) * n_pos
                    
                    # Reconstruct safe physical A
                    A_world_next = A_world.clone()
                    A_world_next[..., :3] = A_world[..., :3] + M_inv_disp_pos # Pos warped
                    # FIX: Truly leave rotation untouched by CASF restorative flow
                    A_world_next[..., 3:] = A_world[..., 3:] 
                    
                    # Back to relative, normalized A
                    A_world_next[..., :3] -= curr_pos
                    A = self.normalizer.normalize(A_world_next)
                    
                t_casf_total += (time.perf_counter() - t2)
                
        # Final orthogonalization check
        if self.check_finite_debug and not torch.isfinite(A).all():
            rospy.logwarn_throttle(5, "Planner aborted: Final A contains non-finite values.")
            return None

        self._q_warm = q_warm

        t_total = time.perf_counter() - t_start
        rospy.loginfo_throttle(5.0, f"⏱️ [TIMING] PC-CASF Plan generated in {t_total*1000:.2f}ms | FM Encode={t_fm_encode*1000:.2f}ms, Pred FM={t_fm_pred*1000:.2f}ms, Corr FM={t_fm_corr*1000:.2f}ms, IK={t_ik_total*1000:.2f}ms, CASF={t_casf_total*1000:.2f}ms")
        return A

    def publish_joint_trajectory(self, A_norm, pos_fork, q_source, t_capture, use_ensembler=True):
        """Publish full 16-point joint trajectory for the 150Hz Safety Shield."""
        current_time = rospy.get_time()
        
        A_unnorm = self.normalizer.unnormalize(A_norm)
        A_world = A_unnorm.clone()
        # pos_fork is already a torch tensor from control_loop
        A_world[..., :3] += pos_fork
        A_world_np = A_world.squeeze(0).detach().cpu().numpy()
        
        if use_ensembler:
            # We add the prediction tagged with the time the point cloud was captured
            self.ensembler.add_prediction(A_world_np, t_capture)
            # We evaluate the ensembled trajectory at the CURRENT time
            smoothed_traj_np = self.ensembler.get_ensembled_trajectory(num_steps=16, current_time=current_time)
            if smoothed_traj_np is None:
                smoothed_traj_np = A_world_np
        else:
            smoothed_traj_np = A_world_np
            
        # ================= NO INTERPOLATION (16 POINTS) =================
        # We preserve the original 16-point trajectory to maintain the correct Cartesian lookahead distance
        num_interp_points = smoothed_traj_np.shape[0]
        interpolated_traj_np = smoothed_traj_np

        # Re-normalize rotation columns after averaging
        for t in range(num_interp_points):
            col1 = interpolated_traj_np[t, 3:6]
            col2 = interpolated_traj_np[t, 6:9]
            col1 = col1 / (np.linalg.norm(col1) + 1e-8)
            col2 = col2 - np.dot(col2, col1) * col1
            col2 = col2 / (np.linalg.norm(col2) + 1e-8)
            interpolated_traj_np[t, 3:6] = col1
            interpolated_traj_np[t, 6:9] = col2
        
        se3_fork_list = []
        se3_tcp_list = []
        for i in range(num_interp_points):
            se3 = decode_9d_to_se3_gpu(
                torch.from_numpy(interpolated_traj_np[i, :9]).to(self.device).unsqueeze(0)
            )[0].cpu().numpy()
            
            # Guard : vérifier que R est orthogonale (det ≈ 1 et trace entre -1.0 et 3.0)
            R_mat = se3[:3, :3]
            det = np.linalg.det(R_mat)
            tr = np.trace(R_mat)
            if not np.isfinite(det) or abs(det - 1.0) > 0.15 or not (-1.0 <= tr <= 3.0):
                # Fallback : rotation identité, position du waypoint
                se3[:3, :3] = np.eye(3)
                if not np.isfinite(se3[:3, 3]).all():
                    se3[:3, 3] = pos_fork.cpu().numpy()
            se3_fork_list.append(se3)
            se3_tcp_list.append(self.fork_se3_to_tcp_se3(se3))

        current_fork_pos = pos_fork.detach().cpu().numpy()
        first_pos_err = np.linalg.norm(se3_fork_list[0][:3, 3] - current_fork_pos)
        last_pos_err = np.linalg.norm(se3_fork_list[-1][:3, 3] - current_fork_pos)
        log_fn = rospy.logwarn if first_pos_err > 0.01 else rospy.logdebug
        log_fn(
            "PUBLISHED NOMINAL START | first_from_current=%.4f m last_from_current=%.4f m "
            "current=[%.3f %.3f %.3f] first=[%.3f %.3f %.3f] last=[%.3f %.3f %.3f]",
            first_pos_err,
            last_pos_err,
            current_fork_pos[0],
            current_fork_pos[1],
            current_fork_pos[2],
            se3_fork_list[0][0, 3],
            se3_fork_list[0][1, 3],
            se3_fork_list[0][2, 3],
            se3_fork_list[-1][0, 3],
            se3_fork_list[-1][1, 3],
            se3_fork_list[-1][2, 3],
        )
        
        # Pad q_init to 9 if it has 7 elements (IK solver expects 9: 7 arm + 2 fingers)
        q_init = q_source.cpu().numpy()
        if len(q_init) == 7:
            q_init = np.concatenate([q_init, [0.0, 0.0]])
            
        q_traj = self.ik_solver.solve_batch(se3_tcp_list, q_init)
        
        if len(q_traj) == 0: return
        
        # Unwrap joints to prevent 2*pi jumps and ensure continuity with current_q
        for i in range(len(q_traj)):
            prev = q_init if i == 0 else q_traj[i-1]
            diff = q_traj[i] - prev
            # Wrap diff to [-pi, pi]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            q_traj[i] = prev + diff

        first_joint_err = np.linalg.norm(q_traj[0][:7] - q_init[:7])
        log_fn = rospy.logwarn if first_joint_err > 0.1 else rospy.logdebug
        log_fn(
            "PUBLISHED NOMINAL JOINT START | first_ik_from_current=%.4f rad "
            "current=[%.3f %.3f %.3f %.3f %.3f %.3f %.3f] "
            "first_ik=[%.3f %.3f %.3f %.3f %.3f %.3f %.3f]",
            first_joint_err,
            q_init[0], q_init[1], q_init[2], q_init[3], q_init[4], q_init[5], q_init[6],
            q_traj[0][0], q_traj[0][1], q_traj[0][2], q_traj[0][3],
            q_traj[0][4], q_traj[0][5], q_traj[0][6],
        )
        
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
        line_marker.color.g = 0.25
        line_marker.color.b = 1.0
        
        for i, se3 in enumerate(se3_fork_list):
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
            sphere_marker.color.r = 0.0
            sphere_marker.color.g = 0.25
            sphere_marker.color.b = 1.0
            marker_array.markers.append(sphere_marker)
            
        marker_array.markers.append(line_marker)
        self.viz_traj_pub.publish(marker_array)
        # ----------------------------------

        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "world"
        for se3 in se3_fork_list:
            pose = Pose()
            pose.position.x = float(se3[0, 3])
            pose.position.y = float(se3[1, 3])
            pose.position.z = float(se3[2, 3])
            quat = R.from_matrix(se3[:3, :3]).as_quat()
            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])
            pose_array.poses.append(pose)
        self.fork_pose_traj_pub.publish(pose_array)
        
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        
        # The ensembler has already aligned the trajectory to start exactly at current_time.
        # We publish the 16 waypoints spaced by 0.1s.
        dt_wp = 1.6 / 16.0
        for i in range(len(q_traj)):
            point = JointTrajectoryPoint()
            point.positions = q_traj[i][:7].tolist()
            point.time_from_start = rospy.Duration(i * dt_wp)
            msg.points.append(point)
            
        self.traj_pub.publish(msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = CASFGenerativeNode()
    node.run()
