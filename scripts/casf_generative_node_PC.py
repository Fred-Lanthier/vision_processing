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
import traceback
import xacro
import tempfile
import xml.etree.ElementTree as ET
from sensor_msgs.msg import JointState, PointCloud2
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, PoseArray
from std_msgs.msg import Bool, Header, Float64MultiArray
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d


def _bool_param(name, default=False):
    value = rospy.get_param(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def stable_numpy_sample(points, count):
    """Deterministic point selection/padding for stable planner inputs."""
    if points is None or len(points) == 0 or count <= 0:
        return np.empty((0, 3), dtype=np.float32)

    n_pts = len(points)
    if n_pts > count:
        score = points[:, 0] * 1.0e6 + points[:, 1] * 1.0e3 + points[:, 2]
        order = np.argsort(score, kind="mergesort")
        take = np.linspace(0, n_pts - 1, count, dtype=np.int64)
        return points[order[take]]

    if n_pts < count:
        idx = np.arange(count, dtype=np.int64) % n_pts
        return points[idx]

    return points


# ==============================================================
# TEMPORAL ENSEMBLER
# ==============================================================
class TemporalEnsembler:
    def __init__(self, pred_horizon, buffer_size=3, weights='exponential',
                 waypoint_dt=0.1, decay=1.0):
        self.pred_horizon = pred_horizon
        self.buffer = collections.deque(maxlen=buffer_size)
        self.weights_type = weights
        self.wp_dt = float(waypoint_dt)
        self.decay = float(decay)
    
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
                    w = np.exp(-self.decay * age_sec) if self.weights_type == 'exponential' else 1.0
                    weights.append(w)
            
            if not actions:
                if trajectory: trajectory.append(trajectory[-1])
                continue

            actions = np.array(actions)
            weights = np.array(weights)
            weights /= weights.sum()
            trajectory.append(np.average(actions, axis=0, weights=weights))

        if not trajectory:
            return None
        while len(trajectory) < num_steps:
            trajectory.append(trajectory[-1])
        return np.array(trajectory)

# Standard imports for this workspace
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
scripts_path = os.path.join(pkg_path, 'scripts')
sys.path.insert(0, pkg_path)
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)
sys.path.append(os.path.join(pkg_path, 'src', 'vision_processing', 'diffusion_model_train'))
from pipeline_timing import TimingPublisher

# Two checkpoint families share the same U-Net architecture but use different
# normalizers:
#   - Train_Fork_FM.py            → Normalizer with pos_min/pos_max, normalize()/unnormalize()
#                                   and stats keyed {'agent_pos','action'} (shared pos stats).
#   - Train_Fork_FlowMP_9D.py     → Normalizer with obs_min/obs_max/act_min/act_max,
#                                   normalize_obs()/normalize_act()/unnormalize_act()
#                                   and stats keyed {'obs','action'} (separate obs/act stats).
# The node auto-detects which family a checkpoint belongs to (see model loading
# below) so both model types load without code changes.
# Each agent builds its OWN normalizer from its OWN module:
#   FlowMatchingAgentFM → FM Normalizer (pos_min/pos_max)
#   FlowMatchingAgent9D → 9D Normalizer (obs_min/act_min, separate obs/act stats)
# We never import a Normalizer directly, so there is no risk of applying the
# wrong one to a model.
from Train_Fork_FM import FlowMatchingAgent as FlowMatchingAgentFM
from Train_Fork_FlowMP_9D import FlowMatchingAgent as FlowMatchingAgent9D
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
        # Per-stage timing -> /pipeline/timing/* (recorded in the bag for section 5.6)
        self.timing = TimingPublisher(enabled=rospy.get_param('~publish_timing', True))

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

        # Read only the standard joint fields needed for this debug message.
        # urdf_parser_py warns about valid Franka-specific URDF extensions.
        with open(urdf_path, 'r') as f:
            robot_xml = ET.fromstring(f.read())
            urdf_joints = [
                joint.get('name')
                for joint in robot_xml.findall('joint')
                if joint.get('type') != 'fixed'
            ]
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
        # CBF protected links. Default = fork feeding rig (incl. fork_tip). For the
        # pick-and-place gripper (no fork) pass ~cbf_link_names WITHOUT fork_tip.
        default_link_names = [
            'panda_link2','panda_link3','panda_link4', 'panda_link5', 'panda_link6', 'panda_link7',
            'panda_hand', 'fork_tip',
        ]
        link_names = list(rospy.get_param("~cbf_link_names", default_link_names))
        weight_handler.add_models(link_names, robot_name='panda')
        self.bernstein_core = BernsteinCore(weight_handler, self.robot_layer, self.device, link_names)
        self.d_safe = float(rospy.get_param("~d_safe", 0.005))
        self.barrier_alpha = float(rospy.get_param("~barrier_alpha", 0.001))
        self.barrier = BernsteinBarrier(self.bernstein_core, d_safe=self.d_safe, alpha=self.barrier_alpha)
        # Body links = all but a trailing tool link (the fork). With no fork in the
        # list (gripper case) all links are body links -> n_body_links == len.
        self.n_body_links = int(rospy.get_param("~n_body_links", 5))

        # 2. Initialize IK Solver (Ultra-fast Pybind11 Pinocchio)
        # The FM model predicts the CONTROLLED link's pose; the robot IK target is the
        # TCP. controlled_link defaults to 'fork_tip' (feeding); pick-and-place sets
        # ~controlled_link:=panda_TCP so the model predicts the TCP directly (identity).
        #   T_world_ctrl = T_world_tcp @ T_tcp_ctrl ;  T_world_tcp = T_world_ctrl @ inv(T_tcp_ctrl)
        self.controlled_link = str(rospy.get_param("~controlled_link", "fork_tip"))
        if self.controlled_link == 'panda_TCP':
            # The model predicts the TCP directly (pick-and-place) -> identity.
            self.T_tcp_fork_tip = np.eye(4)
        else:
            self.T_tcp_fork_tip = compute_T_child_parent_xacro(urdf_path_raw, self.controlled_link, 'panda_TCP')
            if self.T_tcp_fork_tip is None:
                raise RuntimeError(f"Could not resolve fixed transform panda_TCP -> {self.controlled_link}")
        self.T_tcp_fork_tip_inv = np.linalg.inv(self.T_tcp_fork_tip)
        self.ik_solver = fast_ik_module.FastIK(urdf_path, "panda_TCP")
        rospy.loginfo(f"✅ FastIK targets panda_TCP; FM '{self.controlled_link}' poses converted to TCP poses")
        self._q_warm = None
        
        # Debug: Print Pinocchio model joints
        try:
            pin_joints = self.ik_solver.get_joint_names()
            rospy.loginfo(f"🤖 Pinocchio IK joints: {pin_joints}")
        except:
            pass
        
        # 3. Initialize Fork-related components
        self.mesh_loader = RobotMeshLoaderOptimized(urdf_path)
        fork_pts = self.mesh_loader.static_point_clouds.get(self.controlled_link, np.zeros((0, 3)))
        self.fork_pts_local = torch.from_numpy(fork_pts).float().to(self.device)

        # 4. Initialize Flow Matching Agent
        rospy.loginfo(f"🛡️  SafeGenerativeNode: Loading Model: {model_name}")
        payload = torch.load(model_path, map_location=self.device)
        cfg = payload.get('config', {})
        self.action_dim = cfg.get('action_dim', 9)
        self.obs_dim = cfg.get('obs_dim', 9)
        self.pred_horizon = int(cfg.get('pred_horizon', 16))
        stats = payload.get('stats', None)

        # Auto-detect the normalizer family from the checkpoint. The 9D-dynamics
        # models store separate obs/act stats (normalizer.obs_min / act_min);
        # the original FM models store a single pos_min/pos_max.
        state_dict = payload['model_state_dict']
        self.norm_is_9d = (
            any(k.startswith('normalizer.obs_') or k.startswith('normalizer.act_')
                for k in state_dict)
            or (isinstance(stats, dict) and 'obs' in stats and 'action' in stats)
        )
        AgentCls = FlowMatchingAgent9D if self.norm_is_9d else FlowMatchingAgentFM
        rospy.loginfo(
            "🧭 Detected %s normalizer for checkpoint %s",
            "9D-dynamics (obs/act)" if self.norm_is_9d else "FM (shared pos)",
            model_name,
        )

        agent_kwargs = dict(
            action_dim=self.action_dim,
            obs_horizon=cfg.get('obs_horizon', 2),
            pred_horizon=self.pred_horizon,
            encoder_output_dim=cfg.get('encoder_output_dim', 64),
            diffusion_step_embed_dim=cfg.get('diffusion_step_embed_dim', 256),
            down_dims=cfg.get('down_dims', [256, 512, 1024]),
            stats=stats,
        )
        if self.norm_is_9d:
            agent_kwargs['obs_dim'] = self.obs_dim
        self.fm_agent = AgentCls(**agent_kwargs).to(self.device)
        self.fm_agent.load_state_dict(state_dict)
        self.normalizer = self.fm_agent.normalizer
        if not self.normalizer.is_initialized and stats is not None:
            self.normalizer.load_stats_from_dict(stats)
        self.fm_agent.eval()

        # 4. State variables
        self.current_q = None
        self.commanded_q = None
        self.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        self.obstacle_pts_gpu = None
        self.obs_horizon = cfg.get('obs_horizon', 2)
        self.obs_queue = collections.deque(maxlen=self.obs_horizon)
        self.num_points_pcd = cfg.get('num_points', 256)
        self.fm_use_torch_compile = _bool_param("~fm_use_torch_compile", True)
        self.obstacle_pre_sample_max = int(rospy.get_param("~obstacle_pre_sample_max", 2000))
        self.planner_cbf_max_obstacles = int(rospy.get_param("~planner_cbf_max_obstacles", 512))
        self.casf_spatial_max_obstacles = int(rospy.get_param("~casf_spatial_max_obstacles", 100))
        self.check_finite_debug = bool(rospy.get_param("~check_finite_debug", False))
        self.merged_cloud_contains_fork = rospy.get_param("~merged_cloud_contains_fork", True)
        self.single_shot = _bool_param("~single_shot", True)
        self.tcp_frame = rospy.get_param("~tcp_frame", "panda_TCP")
        # CASF ODE parameters (improvement #1: read once, not per-iteration)
        self.casf_kappa = float(rospy.get_param("~casf_kappa", 50.0))
        self.casf_alpha = float(rospy.get_param("~casf_alpha", 5.0))
        self.enable_casf_correction = _bool_param("~enable_casf_correction", True)
        # Improvement #3: correction / prediction step counts exposed as params
        self.n_pred = int(rospy.get_param("~n_pred", 5))
        self.n_corr = int(rospy.get_param("~n_corr", 5))
        self.alpha_corr = float(rospy.get_param("~alpha_corr", 2.0))
        self.casf_barrier_memory_fraction = float(rospy.get_param("~casf_barrier_memory_fraction", 0.75))
        self.casf_barrier_min_obstacles = max(1, int(rospy.get_param("~casf_barrier_min_obstacles", 8)))
        self.commit_distance = float(rospy.get_param("~commit_distance", 0.035))
        self.planner_rate_hz = float(rospy.get_param("~planner_rate_hz", 3.0))
        self.trajectory_waypoint_rate_hz = float(rospy.get_param("~trajectory_waypoint_rate_hz", 10.0))
        if self.planner_rate_hz <= 0.0:
            raise ValueError("~planner_rate_hz must be > 0")
        if self.trajectory_waypoint_rate_hz <= 0.0:
            raise ValueError("~trajectory_waypoint_rate_hz must be > 0")
        self.trajectory_waypoint_dt = 1.0 / self.trajectory_waypoint_rate_hz
        self.trajectory_retime_mode = str(rospy.get_param(
            "~trajectory_retime_mode", "joint_speed")).lower()
        if self.trajectory_retime_mode not in ("uniform", "joint_arc", "joint_speed"):
            raise ValueError(
                "~trajectory_retime_mode must be 'uniform', 'joint_arc', or 'joint_speed'")
        self.trajectory_retime_joint_speed = float(rospy.get_param(
            "~trajectory_retime_joint_speed", 0.25))
        self.trajectory_retime_min_segment_fraction = float(rospy.get_param(
            "~trajectory_retime_min_segment_fraction", 0.02))
        self.trajectory_retime_joint_weights = self._parse_joint_weights(
            rospy.get_param("~trajectory_retime_joint_weights", "1 1 1 1 1 1 1"))
        self.exit_lift_enabled = _bool_param("~planner_exit_lift_enabled", False)
        self.exit_lift_radius_xy = float(rospy.get_param(
            "~planner_exit_lift_radius_xy", 0.16))
        self.exit_lift_release_radius_xy = float(rospy.get_param(
            "~planner_exit_lift_release_radius_xy", 0.12))
        self.exit_lift_clearance = float(rospy.get_param(
            "~planner_exit_lift_clearance", 0.025))
        self.exit_lift_trigger_depth = float(rospy.get_param(
            "~planner_exit_lift_trigger_depth", 0.008))
        self.exit_lift_max_lift = float(rospy.get_param(
            "~planner_exit_lift_max_lift", 0.12))
        self.exit_lift_waypoints = max(2, int(rospy.get_param(
            "~planner_exit_lift_waypoints", 6)))
        self.exit_lift_hold_fraction = float(rospy.get_param(
            "~planner_exit_lift_hold_fraction", 0.70))
        self.exit_lift_min_obstacles = max(1, int(rospy.get_param(
            "~planner_exit_lift_min_obstacles", 12)))
        self.exit_lift_z_quantile = float(rospy.get_param(
            "~planner_exit_lift_z_quantile", 0.95))
        self.exit_lift_below_margin = float(rospy.get_param(
            "~planner_exit_lift_below_margin", 0.05))
        self.log_planner_exit_lift = _bool_param("~log_planner_exit_lift", True)
        self.ensembler_buffer_size = max(1, int(rospy.get_param("~ensembler_buffer_size", 3)))
        self.ensembler_decay = float(rospy.get_param("~ensembler_decay", 1.0))
        self.use_temporal_ensembler = _bool_param("~use_temporal_ensembler", False)
        # Dynamic-scene gate: when enabled, the ensembler blend is applied only
        # while the CBF reports a moving obstacle closing on the robot
        # (/cbf_safety/env_hdot = max per-link dynamic-hdot tightening, m/s;
        # zero for static scenes by deadband design). Supersedes the constant
        # use_temporal_ensembler flag. The gate holds for ensembler_gate_hold
        # seconds past the last reading above ensembler_env_hdot_on.
        self.ensembler_dynamic_gate = _bool_param("~ensembler_dynamic_gate", False)
        self.ensembler_env_hdot_on = float(rospy.get_param(
            "~ensembler_env_hdot_on", 0.02))
        self.ensembler_gate_hold = float(rospy.get_param(
            "~ensembler_gate_hold", 2.0))
        self.latest_env_hdot = 0.0
        self.latest_env_hdot_stamp = 0.0
        self.dynamic_scene_until = 0.0
        self.dynamic_scene_active = False
        self.anchor_cartesian_start = _bool_param("~anchor_cartesian_start", False)
        self.log_planner_cartesian_debug = _bool_param("~log_planner_cartesian_debug", False)
        self.fm_noise_seed = int(rospy.get_param("~fm_noise_seed", 7))
        self.log_trajectory_timing = _bool_param("~log_trajectory_timing", True)
        self.wait_for_cbf_ready = _bool_param("~wait_for_cbf_ready", True)
        self.cbf_ready = not self.wait_for_cbf_ready
        self.hold_last_condition_on_loss = _bool_param(
            "~hold_last_condition_on_loss", True)
        self.condition_hold_max_age = float(rospy.get_param(
            "~condition_hold_max_age", 0.0))
        # Reference-state conditioning: clamp the conditioning joint state to a
        # band of +-condition_reference_clamp rad (per joint) around the
        # trajectory executor's nominal command, so a CBF evasion cannot drag
        # the FM model's input off the training manifold. The CBF + tracking
        # feedback absorb the actual deviation. 0.0 = disabled (condition on
        # the raw measured state, legacy behavior).
        self.condition_reference_clamp = float(rospy.get_param(
            "~condition_reference_clamp", 0.0))
        self.condition_reference_timeout = float(rospy.get_param(
            "~condition_reference_timeout", 0.5))
        self.condition_reference_topic = rospy.get_param(
            "~condition_reference_topic", "/planner/nominal_joint_command")
        # Points of the merged cloud closer than this to the measured-state
        # fork surface are treated as fork points and re-posed to the
        # reference state, keeping the conditioning cloud consistent with the
        # conditioned joint state.
        self.condition_fork_match_radius = float(rospy.get_param(
            "~condition_fork_match_radius", 0.02))
        self.nominal_ref_q = None
        self.nominal_ref_stamp = 0.0
        self.condition_ref_active = False
        self.plan_seq = 0
        self.plan_published = False
        self.is_committed = False
        self.fork_food_distance = float('inf')
        self.latest_cloud_gpu = None
        self.latest_cloud_stamp = None
        self.ensembler = TemporalEnsembler(
            pred_horizon=self.pred_horizon,
            buffer_size=self.ensembler_buffer_size,
            waypoint_dt=self.trajectory_waypoint_dt,
            decay=self.ensembler_decay,
        )
        self.last_pose_source_log = 0.0

        # 4.5 Compile + CUDA-graph the FM velocity net (shrinks the planner's
        # GPU burst so the shared GPU is free for the 150 Hz CBF loop).
        self._setup_fm_compile()

        # 5. Subscribers & Publishers
        rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        rospy.Subscriber('/joint_group_position_controller/command', Float64MultiArray, self.command_callback)
        rospy.Subscriber('/perception/persistent_obstacles', PointCloud2, self.obstacle_callback)
        rospy.Subscriber('/vision/merged_cloud', PointCloud2, self.cloud_callback)
        rospy.Subscriber('/cbf_safety/ready', Bool, self.cbf_ready_callback)
        if self.condition_reference_clamp > 0.0:
            rospy.Subscriber(self.condition_reference_topic, Float64MultiArray,
                             self.nominal_ref_callback)
        from std_msgs.msg import Float32
        rospy.Subscriber('/vision/fork_food_distance', Float32, self.dist_callback)
        if self.ensembler_dynamic_gate:
            rospy.Subscriber('/cbf_safety/env_hdot', Float32,
                             self.env_hdot_callback)
        
        self.traj_pub = rospy.Publisher('/planner/nominal_trajectory', JointTrajectory, queue_size=1, latch=False)
        self.fork_pose_traj_pub = rospy.Publisher('/planner/nominal_fork_trajectory', PoseArray, queue_size=1, latch=True)
        self.viz_traj_pub = rospy.Publisher('/viz/nominal_trajectory_3d', MarkerArray, queue_size=1, latch=True)
        self.pub_model_pcd = rospy.Publisher('/viz/model_input_pcd', PointCloud2, queue_size=1, latch=True)
        
        rospy.Timer(rospy.Duration(1.0 / self.planner_rate_hz), self.control_loop)
        rospy.loginfo(
            "🚀 Safe Generative Planner Node Initialized | planner=%.2fHz waypoint=%.2fHz "
            "pred_horizon=%d commit_distance=%.3fm casf_correction=%s",
            self.planner_rate_hz,
            self.trajectory_waypoint_rate_hz,
            self.pred_horizon,
            self.commit_distance,
            self.enable_casf_correction,
        )

    def _setup_fm_compile(self):
        """Compile + CUDA-graph the FM velocity net.

        The planner runs at ~8 Hz but each replan does N_PRED forward passes
        through velocity_net (55-85 ms total, batch-1 launch-overhead bound).
        That burst monopolizes the GPU the 150 Hz CBF loop shares, inflating
        its red-point preprocess to 40-52 ms. reduce-overhead mode captures
        each forward as a CUDA graph (graph trees), collapsing per-kernel
        launch overhead so the burst drops toward ~15-20 ms and stops
        colliding with the safety loop. Math is unchanged (same fp32 graph).

        Compilation + capture are forced now via a warmup so the first live
        replan does not stall. Any failure falls back to the eager net.
        """
        self._velocity_net_eager = self.fm_agent.velocity_net
        if not self.fm_use_torch_compile:
            rospy.loginfo("FM torch.compile disabled (~fm_use_torch_compile=false).")
            return
        if self.device.type != "cuda":
            rospy.loginfo("FM torch.compile skipped: planner not on CUDA.")
            return
        try:
            t0 = time.perf_counter()
            self.fm_agent.velocity_net = torch.compile(
                self.fm_agent.velocity_net, mode="reduce-overhead")
            # Warm up with prediction-shaped dummy inputs. Shapes MUST match the
            # live call (B=1, pred_horizon, action_dim) or the graph re-records
            # on the first real replan.
            B, H = 1, self.pred_horizon
            dt_pred = 1.0 / max(self.n_pred, 1)
            dummy_obs = {
                'point_cloud': torch.randn(B, self.num_points_pcd, 3, device=self.device),
                'agent_pos': torch.zeros(B, self.obs_horizon, 9, device=self.device),
            }
            with torch.inference_mode():
                global_cond = self.fm_agent.encode_obs(dummy_obs)
                for _ in range(3):  # 3 passes: trace, then capture, then replay
                    A = torch.randn(B, H, self.action_dim, device=self.device)
                    for i in range(self.n_pred):
                        t_tensor = torch.full((B,), i * dt_pred, device=self.device)
                        v = self.fm_agent.velocity_net(A, t_tensor, global_cond)
                        A = A + v * dt_pred
            torch.cuda.synchronize()
            rospy.loginfo(
                "✅ FM velocity_net compiled (reduce-overhead) + warmed up in %.1fs",
                time.perf_counter() - t0)
        except Exception as e:
            rospy.logwarn("FM torch.compile failed (%s); using eager velocity_net.", e)
            self.fm_agent.velocity_net = self._velocity_net_eager

    def cbf_ready_callback(self, msg):
        if msg.data and not self.cbf_ready:
            rospy.loginfo("[TRAJ TIMING] CBF ready received; planner may publish trajectories.")
        self.cbf_ready = bool(msg.data)

    # ----- Normalizer dispatch (handles both FM and 9D-dynamics checkpoints) -----
    def norm_obs(self, x):
        """Normalize the proprioceptive observation (agent_pos)."""
        if self.norm_is_9d:
            return self.normalizer.normalize_obs(x)
        return self.normalizer.normalize(x)

    def norm_act(self, x):
        """Normalize an action / trajectory sample."""
        if self.norm_is_9d:
            return self.normalizer.normalize_act(x)
        return self.normalizer.normalize(x)

    def unnorm_act(self, x):
        """Un-normalize an action / trajectory sample back to physical units."""
        if self.norm_is_9d:
            return self.normalizer.unnormalize_act(x)
        return self.normalizer.unnormalize(x)

    def stable_torch_sample(self, points, count):
        """Deterministic point selection/padding for stable FM conditioning."""
        if points is None or int(points.shape[0]) == 0 or count <= 0:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)

        pts = points
        n_pts = int(pts.shape[0])
        if n_pts > count:
            # Deterministic pseudo-lexicographic order. This is intentionally
            # simple and cheap; the upstream point cloud is already spatially
            # filtered and approximately uniform.
            score = pts[:, 0] * 1.0e6 + pts[:, 1] * 1.0e3 + pts[:, 2]
            order = torch.argsort(score)
            take = torch.linspace(
                0, n_pts - 1, count, device=pts.device, dtype=torch.float32
            ).long()
            return pts[order[take]]

        if n_pts < count:
            idx = torch.arange(count, device=pts.device, dtype=torch.long) % n_pts
            return pts[idx]

        return pts

    def _parse_joint_weights(self, value):
        if isinstance(value, str):
            weights = np.fromstring(value.replace(",", " "), sep=" ", dtype=np.float64)
        else:
            weights = np.asarray(value, dtype=np.float64).reshape(-1)
        if weights.size != 7:
            rospy.logwarn(
                "Invalid trajectory_retime_joint_weights=%s; using all ones.",
                value,
            )
            weights = np.ones(7, dtype=np.float64)
        weights = np.where(np.isfinite(weights), weights, 1.0)
        return np.maximum(weights, 1e-3)

    def _retime_joint_trajectory(self, q_traj):
        q_arm = np.asarray(q_traj, dtype=np.float64)[:, :7]
        n_points = q_arm.shape[0]
        if n_points == 0:
            return np.empty((0,), dtype=np.float64), {}
        if n_points == 1:
            return np.zeros((1,), dtype=np.float64), {
                "mode": "single",
                "weighted_path": 0.0,
                "min_dt": 0.0,
                "max_dt": 0.0,
            }

        nominal_total = (n_points - 1) * self.trajectory_waypoint_dt
        if self.trajectory_retime_mode == "uniform":
            times = np.arange(n_points, dtype=np.float64) * self.trajectory_waypoint_dt
            return times, {
                "mode": "uniform",
                "weighted_path": 0.0,
                "min_dt": self.trajectory_waypoint_dt,
                "max_dt": self.trajectory_waypoint_dt,
            }

        weighted_dq = np.diff(q_arm, axis=0) * self.trajectory_retime_joint_weights.reshape(1, 7)
        segment_lengths = np.linalg.norm(weighted_dq, axis=1)
        weighted_path = float(np.sum(segment_lengths))
        if weighted_path <= 1e-9 or not np.isfinite(weighted_path):
            times = np.arange(n_points, dtype=np.float64) * self.trajectory_waypoint_dt
            return times, {
                "mode": "uniform_fallback",
                "weighted_path": weighted_path,
                "min_dt": self.trajectory_waypoint_dt,
                "max_dt": self.trajectory_waypoint_dt,
            }

        min_fraction = max(0.0, self.trajectory_retime_min_segment_fraction)
        min_length = min_fraction * weighted_path / max(1, n_points - 1)
        segment_weights = np.maximum(segment_lengths, min_length)
        if self.trajectory_retime_mode == "joint_speed":
            speed = max(self.trajectory_retime_joint_speed, 1e-6)
            segment_dt = segment_weights / speed
            times = np.concatenate(([0.0], np.cumsum(segment_dt)))
            return times, {
                "mode": "joint_speed",
                "weighted_path": weighted_path,
                "target_speed": speed,
                "min_dt": float(np.min(segment_dt)),
                "max_dt": float(np.max(segment_dt)),
            }

        segment_dt = nominal_total * segment_weights / max(np.sum(segment_weights), 1e-9)
        times = np.concatenate(([0.0], np.cumsum(segment_dt)))
        times[-1] = nominal_total
        return times, {
            "mode": "joint_arc",
            "weighted_path": weighted_path,
            "min_dt": float(np.min(segment_dt)),
            "max_dt": float(np.max(segment_dt)),
        }

    def initial_fm_noise(self, batch_size, horizon, action_dim):
        if self.fm_noise_seed < 0:
            return torch.randn(batch_size, horizon, action_dim, device=self.device)

        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(self.fm_noise_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.fm_noise_seed)
            return torch.randn(batch_size, horizon, action_dim, device=self.device)

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

    def env_hdot_callback(self, msg):
        self.latest_env_hdot = float(msg.data)
        self.latest_env_hdot_stamp = rospy.get_time()

    def _update_dynamic_scene_gate(self):
        """True while a moving obstacle is (or recently was) closing on the
        robot. A fresh env_hdot reading above the ON threshold arms the gate
        for ensembler_gate_hold seconds; it re-arms as long as readings stay
        above threshold, and expires quietly once the scene is static again."""
        now = rospy.get_time()
        if (self.latest_env_hdot >= self.ensembler_env_hdot_on
                and now - self.latest_env_hdot_stamp < 1.0):
            self.dynamic_scene_until = now + self.ensembler_gate_hold
        active = now < self.dynamic_scene_until
        if active != self.dynamic_scene_active:
            if active:
                rospy.loginfo(
                    "🌀 Moving obstacle detected (env_hdot %.3f m/s >= %.3f) "
                    "— temporal ensembler ON",
                    self.latest_env_hdot, self.ensembler_env_hdot_on)
            else:
                rospy.loginfo(
                    "Moving obstacle cleared for %.1fs — temporal ensembler OFF",
                    self.ensembler_gate_hold)
            self.dynamic_scene_active = active
        return active

    def nominal_ref_callback(self, msg):
        try:
            if len(msg.data) >= 7:
                self.nominal_ref_q = torch.tensor(
                    msg.data[:7], dtype=torch.float32, device=self.device)
                self.nominal_ref_stamp = rospy.get_time()
        except Exception as e:
            rospy.logerr_throttle(5, f"Error in nominal_ref_callback: {e}")

    def _conditioning_q(self):
        """Measured q, clamped to +-condition_reference_clamp (rad, per joint)
        around the executor's nominal command. Sets self.condition_ref_active
        when the clamp changed the state, so the cloud path can re-pose the
        fork points to match."""
        q_meas = self.current_q
        self.condition_ref_active = False
        c = self.condition_reference_clamp
        if c <= 0.0 or self.nominal_ref_q is None:
            return q_meas
        if rospy.get_time() - self.nominal_ref_stamp > self.condition_reference_timeout:
            rospy.logwarn_throttle(
                5.0, "Conditioning reference stale; conditioning on measured state.")
            return q_meas
        dev = q_meas - self.nominal_ref_q
        if not bool((dev.abs() > c).any()):
            return q_meas
        self.condition_ref_active = True
        rospy.loginfo_throttle(
            2.0,
            "Conditioning on reference state (max joint deviation %.3f rad > clamp %.3f rad).",
            float(dev.abs().max()), c)
        return self.nominal_ref_q + torch.clamp(dev, -c, c)

    def _fork_cloud_at(self, q):
        q_dict = {jn: float(v)
                  for jn, v in zip(self.joint_names, q.detach().cpu().numpy())}
        pts = self.mesh_loader.create_point_cloud_fork_tip(q_dict)
        return torch.from_numpy(pts).float().to(self.device)

    def _repose_fork_points(self, cloud_world, q_ref):
        """The merged cloud's fork-mesh points sit at the MEASURED joint state;
        the conditioning state q_ref may differ (clamped toward the nominal).
        Swap the points near the measured fork surface for a fork sampled at
        q_ref, so cloud and conditioned state stay consistent as in training."""
        try:
            fork_meas = self._fork_cloud_at(self.current_q)
            if fork_meas.shape[0] == 0:
                return cloud_world
            d = torch.cdist(cloud_world, fork_meas)
            is_fork = d.min(dim=1).values < self.condition_fork_match_radius
            n_fork = int(is_fork.sum())
            if n_fork == 0:
                return cloud_world
            fork_ref = self.stable_torch_sample(self._fork_cloud_at(q_ref), n_fork)
            return torch.cat([cloud_world[~is_fork], fork_ref], dim=0)
        except Exception as e:
            rospy.logwarn_throttle(
                5.0, f"Fork repose failed ({e}); using raw merged cloud.")
            return cloud_world

    def obstacle_callback(self, msg):
        # Unpack obstacles for CBF Corrector in ODE
        # PointCloud2 to Numpy (Assume XYZ float32 from create_cloud_xyz32)
        try:
            points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, int(msg.point_step/4))
            points = points[:, :3] # Take only X, Y, Z
            if len(points) > self.obstacle_pre_sample_max:
                points = stable_numpy_sample(points, self.obstacle_pre_sample_max)
            self.obstacle_pts_gpu = torch.from_numpy(points.copy()).float().to(self.device)
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error unpacking obstacles: {e}")

    def cloud_callback(self, msg):
        # Conditioned point cloud for FM Encoder
        try:
            points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, int(msg.point_step/4))
            points = points[:, :3]
            
            # Empty condition clouds usually mean the target detector lost the
            # object. Keep the last valid conditioning cloud so replanning does
            # not stop during short SAM3/SAM2 dropouts.
            if len(points) == 0:
                if self.hold_last_condition_on_loss and self.latest_cloud_gpu is not None:
                    if self.latest_cloud_stamp is None:
                        age = 0.0
                    else:
                        age = rospy.get_time() - self.latest_cloud_stamp
                    if self.condition_hold_max_age <= 0.0 or age <= self.condition_hold_max_age:
                        rospy.logwarn_throttle(
                            5.0,
                            "Condition cloud empty; continuing with last valid conditioning cloud "
                            "(age %.1fs).",
                            age,
                        )
                        return
                self.latest_cloud_gpu = None
                self.latest_cloud_stamp = None
                return
                
            points = stable_numpy_sample(points, self.num_points_pcd)
            
            if len(points) > 0:
                self.latest_cloud_gpu = torch.from_numpy(points.copy()).float().to(self.device)
                self.latest_cloud_stamp = rospy.get_time()
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error unpacking merged cloud: {e}")

    def control_loop(self, event):
        t_capture = rospy.get_time()
        
        if self.single_shot and self.plan_published:
            return
            
        if self.is_committed:
            # We already published the open-loop strike, do not replan.
            return

        if not self.cbf_ready:
            rospy.loginfo_throttle(
                2.0,
                "Waiting for CBF ready before publishing planner trajectory...")
            return

        if self.current_q is None or self.latest_cloud_gpu is None:
            return

        # Get precise fork_tip pose directly from the kinematics tree
        base = torch.eye(4, device=self.device).unsqueeze(0)
        
        # Conditioning state: measured q, optionally clamped toward the
        # nominal reference (condition_reference_clamp > 0) so a CBF evasion
        # cannot drag the FM input off the training manifold. Cloud/base
        # alignment is restored by _repose_fork_points below when clamped.
        q_source = self._conditioning_q()
        q_eval = q_source.clone().unsqueeze(0) # Ensure it is shape [1, 7]
        if q_eval.shape[-1] == 7:
            q_eval = torch.cat([q_eval, torch.zeros((1, 2), device=self.device)], dim=-1)
            
        link_poses = self.robot_layer._native_forward_kinematics(q_eval)
        
        if self.controlled_link not in link_poses:
            rospy.logwarn_throttle(5, f"FK failed: '{self.controlled_link}' not found in URDF kinematics tree.")
            return

        # Strictly use Forward Kinematics (FK) to avoid TF latency jitter
        T_fork = link_poses[self.controlled_link][0]
        
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
        cloud_world = self.latest_cloud_gpu
        if self.condition_ref_active and self.merged_cloud_contains_fork:
            cloud_world = self._repose_fork_points(cloud_world, q_source)
        pcd_merged_centered = cloud_world - pos_fork
        if not self.merged_cloud_contains_fork:
            pcd_merged_centered = torch.cat([pcd_merged_centered, self.fork_pts_local], dim=0)
        
        # Sample / pad deterministically. Random point resampling changes the
        # model input enough to make a receding-horizon FM policy flip modes.
        pcd_final_centered = self.stable_torch_sample(
            pcd_merged_centered, self.num_points_pcd)
        
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
                use_ensembler = self.use_temporal_ensembler
                if self.ensembler_dynamic_gate:
                    use_ensembler = self._update_dynamic_scene_gate()
                if self.fork_food_distance < self.commit_distance:
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
            rospy.logerr_throttle(
                5,
                "Trajectory generation error: %r\n%s",
                e,
                traceback.format_exc(),
            )

    def fork_se3_to_tcp_se3(self, T_world_fork):
        return T_world_fork @ self.T_tcp_fork_tip_inv

    def cap_casf_obstacles_for_memory(self, P_obs, curr_pos, horizon):
        """Keep the original CASF path, but avoid barrier allocations that cannot fit."""
        if (
            P_obs is None or P_obs.shape[0] == 0
            or not torch.cuda.is_available()
            or self.casf_barrier_memory_fraction <= 0.0
        ):
            return P_obs

        try:
            free_bytes, _ = torch.cuda.mem_get_info()
        except Exception:
            return P_obs

        k_links = int(getattr(self.bernstein_core, "K", 1))
        n_func = int(getattr(self.bernstein_core, "n_func", 1))
        bytes_per_point = int(horizon) * k_links * (n_func ** 3) * P_obs.element_size()
        if bytes_per_point <= 0:
            return P_obs

        budget_bytes = int(free_bytes * self.casf_barrier_memory_fraction)
        max_points = budget_bytes // bytes_per_point
        max_points = min(int(P_obs.shape[0]), max(self.casf_barrier_min_obstacles, int(max_points)))

        if max_points >= int(P_obs.shape[0]):
            return P_obs

        dist_to_fork = torch.norm(P_obs - curr_pos, dim=1)
        _, top_idx = torch.topk(dist_to_fork, k=max_points, largest=False, sorted=True)
        rospy.logwarn_throttle(
            2.0,
            "CASF barrier capped obstacle points from %d to %d due to free VRAM budget",
            int(P_obs.shape[0]),
            int(max_points),
        )
        return P_obs[top_idx]

    def evaluate_casf_barrier(self, q_batch, P_obs):
        """Original combined CASF barrier with OOM backoff."""
        P_eval = P_obs
        while True:
            q_eval = q_batch.detach().clone().requires_grad_(True)
            try:
                h_q, grad_h_q, _ = self.barrier(q_eval, P_eval)
                return h_q.detach(), grad_h_q.detach()
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() or int(P_eval.shape[0]) <= self.casf_barrier_min_obstacles:
                    raise
                torch.cuda.empty_cache()
                new_count = max(self.casf_barrier_min_obstacles, int(P_eval.shape[0]) // 2)
                rospy.logwarn_throttle(
                    2.0,
                    "CASF barrier OOM with %d obstacle points; retrying with %d",
                    int(P_eval.shape[0]),
                    int(new_count),
                )
                P_eval = P_eval[:new_count]

    def apply_container_exit_lift(self, traj_np, current_fork_pos):
        """Lift fork-tip waypoints above the local rim before lateral exit."""
        if (
            not self.exit_lift_enabled
            or traj_np is None
            or traj_np.shape[0] < 2
            or self.obstacle_pts_gpu is None
            or self.obstacle_pts_gpu.shape[0] < self.exit_lift_min_obstacles
        ):
            return traj_np

        obs = self.obstacle_pts_gpu.detach()
        if obs.shape[1] < 3:
            return traj_np

        current = torch.as_tensor(
            current_fork_pos, dtype=obs.dtype, device=obs.device)
        with torch.no_grad():
            dxy = torch.norm(obs[:, :2] - current[:2], dim=1)
            z = obs[:, 2]
            keep = (
                (dxy <= self.exit_lift_radius_xy)
                & (z >= current[2] - self.exit_lift_below_margin)
            )
            local_z = z[keep]
            if int(local_z.numel()) < self.exit_lift_min_obstacles:
                return traj_np
            local_z_np = local_z.float().cpu().numpy()

        q = float(np.clip(self.exit_lift_z_quantile, 0.50, 1.0))
        rim_z = float(np.quantile(local_z_np, q))
        current_z = float(current_fork_pos[2])
        target_z = rim_z + self.exit_lift_clearance
        if self.exit_lift_max_lift > 0.0:
            target_z = min(target_z, current_z + self.exit_lift_max_lift)
        lift_dz = target_z - current_z
        if lift_dz < self.exit_lift_trigger_depth:
            return traj_np

        out = traj_np.copy()
        original = traj_np.copy()
        h = int(out.shape[0])
        n_lift = min(h, self.exit_lift_waypoints)
        hold_fraction = float(np.clip(self.exit_lift_hold_fraction, 0.0, 0.95))
        release_radius = max(0.0, self.exit_lift_release_radius_xy)

        for i in range(n_lift):
            s = float(i + 1) / float(n_lift)
            smooth = s * s * (3.0 - 2.0 * s)
            z_min = current_z + smooth * lift_dz
            out[i, 2] = max(out[i, 2], z_min)

            if hold_fraction > 0.0:
                if s <= hold_fraction:
                    xy_release = 0.0
                else:
                    xy_release = (s - hold_fraction) / (1.0 - hold_fraction)
                    xy_release = xy_release * xy_release
                out[i, :2] = (
                    current_fork_pos[:2]
                    + xy_release * (original[i, :2] - current_fork_pos[:2])
                )

        if release_radius > 0.0:
            for i in range(n_lift, h):
                if np.linalg.norm(out[i, :2] - current_fork_pos[:2]) <= release_radius:
                    out[i, 2] = max(out[i, 2], target_z)

        if self.log_planner_exit_lift:
            rospy.loginfo_throttle(
                1.0,
                "Planner exit lift applied | obs=%d rim_z=%.3f fork_z=%.3f "
                "target_z=%.3f dz=%.3f waypoints=%d max_delta=%.3fm",
                int(local_z_np.shape[0]),
                rim_z,
                current_z,
                target_z,
                lift_dz,
                n_lift,
                float(np.linalg.norm(out[:, :3] - original[:, :3], axis=1).max()),
            )
        return out

    def generate_safe_trajectory(self, obs_dict, curr_pos, P_obs, q_source, lam=1e-5):
        """PC-CASF: Prediction-Correction with CASF Metric Warping."""
        B = 1
        H = self.pred_horizon
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
            'agent_pos': self.norm_obs(obs_dict['agent_pos'].to(dev))
        }

        # ---------------------------------------------------------
        # PHASE 1: PREDICTION (Pure Denoising)
        # ---------------------------------------------------------
        # Start from pure noise. In servo mode the noise is fixed by default so
        # repeated replans do not choose different left/right modes for the same
        # scene.
        A = self.initial_fm_noise(B, H, self.action_dim)
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
        
        if self.enable_casf_correction and P_obs is not None and P_obs.shape[0] > 0:
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
            # Mirrors the CBF self-filter: keep only points whose minimum SDF
            # over body links (link0…panda_hand) exceeds 5mm.
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

            # Step 2: Spatial filter — closest points to fork tip (fast proxy for danger)
            if P_obs.shape[0] > 0:
                dist_to_fork = torch.norm(P_obs - curr_pos, dim=1)
                k_spatial = min(self.casf_spatial_max_obstacles, self.planner_cbf_max_obstacles)
                if P_obs.shape[0] > k_spatial:
                    _, top_idx = torch.topk(dist_to_fork, k=k_spatial, largest=False, sorted=True)
                    P_obs_filtered = P_obs[top_idx]
                else:
                    P_obs_filtered = P_obs
            else:
                P_obs_filtered = P_obs

            P_obs_filtered = self.cap_casf_obstacles_for_memory(P_obs_filtered, curr_pos, H)

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
                A_unnorm = self.unnorm_act(A)
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
                q_batch = torch.from_numpy(q_np[:, :7]).float().to(dev)
                
                # J_full is [H, 6, 9]. We slice the first 3 rows for pos, and first 7 cols for arm joints.
                J_full = torch.from_numpy(np.stack(Js_list)).float().to(dev) 
                J_pos = J_full[:, :3, :7] # [H, 3, 7]
                
                h_q, grad_h_q = self.evaluate_casf_barrier(q_batch, P_obs_filtered)
                
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
                    A_next_unnorm = self.unnorm_act(A + v_nom * dt_corr)
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
                    A = self.norm_act(A_world_next)
                    
                t_casf_total += (time.perf_counter() - t2)
                
        # Final orthogonalization check
        if self.check_finite_debug and not torch.isfinite(A).all():
            rospy.logwarn_throttle(5, "Planner aborted: Final A contains non-finite values.")
            return None

        self._q_warm = q_warm

        t_total = time.perf_counter() - t_start
        self.timing.publish('fm_encode', t_fm_encode * 1000.0)
        self.timing.publish('fm_generation', t_fm_pred * 1000.0)
        self.timing.publish('fm_correction', t_fm_corr * 1000.0)
        self.timing.publish('ik_correction', t_ik_total * 1000.0)
        self.timing.publish('casf_warp', t_casf_total * 1000.0)
        self.timing.publish('planner_total', t_total * 1000.0)
        plan_label = "PC-CASF" if self.enable_casf_correction else "PC-FM"
        rospy.loginfo_throttle(5.0, f"⏱️ [TIMING] {plan_label} Plan generated in {t_total*1000:.2f}ms | FM Encode={t_fm_encode*1000:.2f}ms, Pred FM={t_fm_pred*1000:.2f}ms, Corr FM={t_fm_corr*1000:.2f}ms, IK_corr={t_ik_total*1000:.2f}ms, CASF={t_casf_total*1000:.2f}ms")
        return A

    def publish_joint_trajectory(self, A_norm, pos_fork, q_source, t_capture, use_ensembler=True):
        """Publish full joint trajectory for the safety shield."""
        t_build_start = time.perf_counter()
        current_time = rospy.get_time()

        A_unnorm = self.unnorm_act(A_norm)
        A_world = A_unnorm.clone()
        # pos_fork is already a torch tensor from control_loop
        A_world[..., :3] += pos_fork
        A_world_np = A_world.squeeze(0).detach().cpu().numpy()
        raw_traj_np = A_world_np.copy()
        
        # Always feed the buffer (tagged with the point-cloud capture time) so
        # the dynamic-scene gate can switch the blend on mid-run and only ever
        # mixes recent predictions; stale ones age out of the horizon window.
        self.ensembler.add_prediction(A_world_np, t_capture)
        if use_ensembler:
            # We evaluate the ensembled trajectory at the CURRENT time
            smoothed_traj_np = self.ensembler.get_ensembled_trajectory(
                num_steps=A_world_np.shape[0],
                current_time=current_time,
            )
            if smoothed_traj_np is None or smoothed_traj_np.shape[0] == 0:
                rospy.logwarn_throttle(
                    5.0,
                    "Temporal ensembler has no valid current waypoint; using latest raw trajectory.")
                smoothed_traj_np = A_world_np
        else:
            smoothed_traj_np = A_world_np
        ensembled_traj_np = smoothed_traj_np.copy()

        current_fork_pos = pos_fork.detach().cpu().numpy()
        if self.anchor_cartesian_start:
            # Optional legacy behavior. It removes Cartesian start error, but
            # it also bends the model path. Keep it off when diagnosing FM
            # trajectory geometry.
            start_offset = smoothed_traj_np[0, :3] - current_fork_pos
            H_traj = smoothed_traj_np.shape[0]
            for i in range(H_traj):
                decay = 1.0 - float(i) / float(H_traj)
                smoothed_traj_np[i, :3] -= start_offset * decay
        anchored_traj_np = smoothed_traj_np.copy()
        smoothed_traj_np = self.apply_container_exit_lift(
            smoothed_traj_np, current_fork_pos)
        lifted_traj_np = smoothed_traj_np.copy()

        if self.log_planner_cartesian_debug:
            rospy.loginfo_throttle(
                0.5,
                "FM CART DEBUG | use_ens=%s cart_anchor=%s fork=[%.3f %.3f %.3f] "
                "raw0=[%.3f %.3f %.3f] rawN=[%.3f %.3f %.3f] "
                "ens0=[%.3f %.3f %.3f] ensN=[%.3f %.3f %.3f] "
                "anch0=[%.3f %.3f %.3f] anchN=[%.3f %.3f %.3f] "
                "lift0=[%.3f %.3f %.3f] liftN=[%.3f %.3f %.3f]",
                use_ensembler,
                self.anchor_cartesian_start,
                current_fork_pos[0], current_fork_pos[1], current_fork_pos[2],
                raw_traj_np[0, 0], raw_traj_np[0, 1], raw_traj_np[0, 2],
                raw_traj_np[-1, 0], raw_traj_np[-1, 1], raw_traj_np[-1, 2],
                ensembled_traj_np[0, 0], ensembled_traj_np[0, 1], ensembled_traj_np[0, 2],
                ensembled_traj_np[-1, 0], ensembled_traj_np[-1, 1], ensembled_traj_np[-1, 2],
                anchored_traj_np[0, 0], anchored_traj_np[0, 1], anchored_traj_np[0, 2],
                anchored_traj_np[-1, 0], anchored_traj_np[-1, 1], anchored_traj_np[-1, 2],
                lifted_traj_np[0, 0], lifted_traj_np[0, 1], lifted_traj_np[0, 2],
                lifted_traj_np[-1, 0], lifted_traj_np[-1, 1], lifted_traj_np[-1, 2],
            )
            
        # ================= NO INTERPOLATION =================
        # Preserve the model trajectory to maintain the learned Cartesian lookahead distance.
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
        log_fn = (
            rospy.logwarn
            if self.anchor_cartesian_start and first_pos_err > 0.01
            else rospy.logdebug
        )
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
            
        t_ik_start = time.perf_counter()
        q_traj = self.ik_solver.solve_batch(se3_tcp_list, q_init)
        t_publish_ik = time.perf_counter() - t_ik_start
        # Canonical IK cost: the batch solve that turns the FM Cartesian fork-tip
        # trajectory into the joint position trajectory q_traj. Runs every plan,
        # so this is the IK number for the §5.6 timing table (the correction-loop
        # IK is published separately as 'ik_correction' and is 0 when CASF is off).
        self.timing.publish('ik', t_publish_ik * 1000.0)

        if len(q_traj) == 0: return
        
        # Unwrap joints to prevent 2*pi jumps and ensure continuity with current_q
        for i in range(len(q_traj)):
            prev = q_init if i == 0 else q_traj[i-1]
            diff = q_traj[i] - prev
            # Wrap diff to [-pi, pi]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            q_traj[i] = prev + diff

        # Publish the measured current joints as waypoint 0 so every nominal
        # trajectory starts continuously without warping the Cartesian FM path.
        q_traj[0][:7] = q_init[:7]

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
        self.plan_seq += 1
        msg.header.seq = self.plan_seq
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        time_stamps, retime_stats = self._retime_joint_trajectory(q_traj)
        
        for i in range(len(q_traj)):
            point = JointTrajectoryPoint()
            point.positions = q_traj[i][:7].tolist()
            point.time_from_start = rospy.Duration(float(time_stamps[i]))
            msg.points.append(point)
            
        self.traj_pub.publish(msg)
        self.timing.publish('trajectory_build', (time.perf_counter() - t_build_start) * 1000.0)
        if self.log_trajectory_timing:
            if len(q_traj) > 1:
                q_arm = np.asarray(q_traj)[:, :7]
                segment_lengths = np.linalg.norm(np.diff(q_arm, axis=0), axis=1)
                joint_path = float(np.sum(segment_lengths))
                first_step = float(segment_lengths[0])
                max_step = float(np.max(segment_lengths))
            else:
                joint_path = 0.0
                first_step = 0.0
                max_step = 0.0
            expected_duration = (
                msg.points[-1].time_from_start.to_sec()
                if msg.points else 0.0
            )
            rospy.loginfo(
                "[TRAJ TIMING] published id=%d points=%d expected_duration=%.3fs "
                "waypoint_dt=%.3fs retime=%s speed=%.3frad/s dt_range=[%.3f, %.3f]s "
                "capture_to_publish=%.3fs build=%.1fms publish_ik=%.1fms "
                "joint_path=%.3frad weighted_path=%.3f first_step=%.3frad max_step=%.3frad",
                self.plan_seq,
                len(msg.points),
                expected_duration,
                self.trajectory_waypoint_dt,
                retime_stats.get("mode", "unknown"),
                retime_stats.get("target_speed", 0.0),
                retime_stats.get("min_dt", 0.0),
                retime_stats.get("max_dt", 0.0),
                rospy.get_time() - t_capture,
                (time.perf_counter() - t_build_start) * 1000.0,
                t_publish_ik * 1000.0,
                joint_path,
                retime_stats.get("weighted_path", 0.0),
                first_step,
                max_step,
            )

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = CASFGenerativeNode()
    node.run()
