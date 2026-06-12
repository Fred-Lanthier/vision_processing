#!/usr/bin/env python3

import rospy
import rospkg
import os
import sys
import time
import json
import torch
import numpy as np
import struct
import xacro
import tempfile
import xml.etree.ElementTree as ET
from std_msgs.msg import Bool, Float32MultiArray, Float64MultiArray, Header, String
from sensor_msgs.msg import JointState, PointCloud2, PointField
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs.point_cloud2 as pc2

try:
    import trimesh
except ImportError:
    trimesh = None

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
sys.path.insert(0, pkg_path)

from third_party.SafeFlowMatcher.diffuser.models.rdf_cbf import RDF_CBF # Le solveur CBF est conservé
from third_party.RDF.urdf_layer import URDFLayer # Le module de cinématique est conservé pour l'Autograd

# import depuis le nouveau projet Bernstein
from third_party.SDF_Bernstein_Basis.src.rdf_weights import RDF_Weights
from third_party.SDF_Bernstein_Basis.bernstein_core import BernsteinCore
from third_party.SDF_Bernstein_Basis.bernstein_barrier import BernsteinBarrier

from vision_processing import fast_perception_module


def _bool_param(name, default=False):
    value = rospy.get_param(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


# Layout of /cbf_safety/diagnostics (Float32MultiArray). Same base fields as
# cbf_safety_node_Bernstein.py so analysis scripts work across both nodes;
# concepts this node does not implement (escape, output EMA/low-pass, single
# QP multiplier) are published as 0/NaN. The multicbf extensions are appended
# after the base scalars. Full layout is published as JSON on the latched
# topic /cbf_safety/diagnostics_layout.
DIAG_VEC_FIELDS = [
    "dq_ff",          # feedforward from nominal position differencing
    "dq_fb",          # shaped tracking feedback (see _shape_tracking_feedback)
    "dq_base",        # filtered nominal velocity fed to the solver
    "dq_escape",      # always zero (no escape mechanism in this node)
    "dq_cbf_delta",   # solver output minus dq_base (before repair)
    "dq_pre_filter",  # solver output before clamp/repair
    "dq_final",       # velocity actually published
]
DIAG_SCALAR_FIELDS = [
    "h",                # min barrier value (hard min in multi_projected)
    "h_corr",           # bias-compensated h (== h in multi_projected mode)
    "lam",              # NaN (no single multiplier in this node)
    "cap_active",       # NaN
    "recovery_used",    # NaN
    "grad_degenerate",  # NaN
    "constr_pre",       # CBF constraint value of dq_base (most-critical grad)
    "repair_applied",   # 1 if post-clamp repair modified the command
    "constr_final",     # constraint value after repair (NaN if repair off)
    "grad_h_norm",      # |grad_h| of the most critical constraint
    "escape_active",    # always 0
    "ema_applied",      # always 0
    "velocity_filter_active",  # always 0
    "vel_source",       # always 1 (position differencing)
    "velocity_age",     # NaN
    "dt",               # control loop dt (s)
    "selected_count",   # obstacle points fed to the solver
    "min_obs_dist",     # min obstacle distance from the selection stage
    "cbf_ms",           # solver + repair time
    "q_cmd_ms",         # command construction time
    "total_ms",         # solver start to safe-command publish
    # --- multicbf extensions ---
    "cmd_hdot",         # grad_h . dq_safe (commanded h-rate)
    "real_hdot",        # grad_h . dq_measured (executed h-rate)
    "real_constr",      # CBF constraint evaluated on the measured velocity
    "h_rate",           # finite-difference dh/dt
    "alignment",        # cos(angle) between dq_base and dq_safe
    "q_lead",           # |q_cmd - q_cur|
    "dq_real_norm",     # |measured joint velocity|
    "active_constraints",  # number of half-spaces in the QP this cycle
    "solver_mode",      # 0 = fast_tangent, 1 = multi_projected
    "monitor_only",     # 1 = barrier evaluated but no correction applied
    "contact_stop",     # 1 = latched contact stop active (robot frozen)
]


class CBFSafetyNode:
    def __init__(self):
        rospy.init_node('cbf_safety_node')
        self.device = torch.device('cuda')
        self._logged_first_nominal = False
        self._logged_first_safe_publish = False

        urdf_path_raw = pkg_path + '/urdf/panda_camera.xacro' 

        # Global Xacro -> URDF conversion
        if urdf_path_raw.endswith('.xacro'):
            doc = xacro.process_file(urdf_path_raw)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
                f.write(doc.toxml())
                urdf_path = f.name
        else:
            urdf_path = urdf_path_raw

        self.robot_layer = URDFLayer(
            urdf_path=urdf_path,
            device=self.device,
            package_dir=pkg_path,
            voxel_dir=pkg_path + '/third_party/RDF/panda_layer/meshes/voxel_128'
        )
        self._log_cuda_memory("CBF after URDFLayer")
        
        # 1. Chargement de weights via la nouvelle API
        weight_handler = RDF_Weights(device=self.device, dtype=torch.float32)
        weight_handler.init_robot_folder(pkg_path + '/third_party/SDF_Bernstein_Basis/panda_test', robot_name='panda')
        
        # 2. On défini les links que l'on veut charger (correspondant aux links cinématiques dans l'ordre)
        # Pruning: We only protect the upper arm and wrist (Links 4-7 + Fork)
        link_names = ['panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 'panda_hand', 'fork_tip']
        self.protected_link_names = link_names
        self.fork_link_index = link_names.index('fork_tip')
        weight_handler.add_models(link_names, robot_name='panda')
        self._log_cuda_memory("CBF after Bernstein weights")
        
        # 3. Création du pont Core
        self.bernstein_core = BernsteinCore(weight_handler, self.robot_layer, self.device, link_names)
        self._log_cuda_memory("CBF after Bernstein core")
        
        # 4. Création du pont Barrier (qui gère autograd comme l'ancien RDF_Barrier)
        self.d_safe = float(rospy.get_param("~d_safe", 0.005))

        # alpha: softmin temperature over obstacle points.
        # Bias ≈ alpha*log(N). Small alpha → near hard-min → gradient snaps to the
        # single nearest obstacle point and jumps direction as the robot moves (oscillation).
        # Larger alpha blends the gradient over the k nearest points, giving a smooth,
        # stable correction direction at the cost of a larger bias.
        # Effective trigger clearance = d_safe + alpha*log(N).
        # Tune with cbf_graph_points because the soft-min bias depends on N.
        barrier_alpha = float(rospy.get_param("~barrier_alpha", 0.005))
        self.barrier = BernsteinBarrier(self.bernstein_core, d_safe=self.d_safe, alpha=barrier_alpha)
        self._log_cuda_memory("CBF after Bernstein barrier")

        # kappa: class-K coefficient in the CBF constraint ∇h·dq + κh ≥ 0.
        # High kappa forces hard corrections even when h is slightly positive (safe).
        # With a jumpy gradient, high kappa amplifies oscillation near obstacles.
        # Recommended: 1.0–3.0. Must be read BEFORE setup_cuda_graph (baked into graph).
        self.cbf_kappa = float(rospy.get_param("~cbf_kappa", 2.0))
        self.cbf_recovery_kappa = float(rospy.get_param("~cbf_recovery_kappa", 3.0))
        self.cbf_constraint_margin = float(rospy.get_param("~cbf_constraint_margin", 0.0))
        self.cbf_recovery_switch_margin = float(rospy.get_param(
            "~cbf_recovery_switch_margin", 0.004))
        self.cbf_solver_mode = str(rospy.get_param(
            "~cbf_solver_mode", "fast_tangent")).lower()
        if self.cbf_solver_mode not in ("fast_tangent", "multi_projected"):
            raise ValueError("~cbf_solver_mode must be 'fast_tangent' or 'multi_projected'")
        self.cbf_active_constraints = max(1, int(rospy.get_param(
            "~cbf_active_constraints", 4)))
        self.cbf_h_activate = float(rospy.get_param("~cbf_h_activate", 0.006))
        self.cbf_max_inward_speed = float(rospy.get_param(
            "~cbf_max_inward_speed", 0.20))
        self.cbf_recovery_speed = float(rospy.get_param(
            "~cbf_recovery_speed", 0.0))
        self.cbf_recovery_depth = float(rospy.get_param(
            "~cbf_recovery_depth", 0.01))
        self.cbf_max_correction_speed = float(rospy.get_param(
            "~cbf_max_correction_speed", 0.35))
        self.cbf_projection_iters = max(1, int(rospy.get_param(
            "~cbf_projection_iters", 3)))
        self.cbf_projection_relaxation = float(rospy.get_param(
            "~cbf_projection_relaxation", 1.0))
        self.cbf_enforce_final_constraint = _bool_param("~cbf_enforce_final_constraint", True)
        self.cbf_integrate_from_current = _bool_param("~cbf_integrate_from_current", True)
        self.cbf_command_dt = float(rospy.get_param("~cbf_command_dt", 0.0))
        self.cbf_passthrough_when_inactive = _bool_param(
            "~cbf_passthrough_when_inactive", True)
        self.cbf_stateful_position_command = _bool_param("~cbf_stateful_position_command", True)
        self.cbf_direct_position_mode = str(rospy.get_param(
            "~cbf_direct_position_mode", "nominal_correction")).lower()
        if self.cbf_direct_position_mode not in ("integrated", "nominal_correction"):
            raise ValueError("~cbf_direct_position_mode must be 'integrated' or 'nominal_correction'")
        self.cbf_direct_feedback_when_reactive = _bool_param(
            "~cbf_direct_feedback_when_reactive", True)
        self.cbf_tracking_feedback_mode = str(rospy.get_param(
            "~cbf_tracking_feedback_mode", "tangent_near")).lower()
        if self.cbf_tracking_feedback_mode not in ("full", "none", "tangent", "tangent_near"):
            raise ValueError(
                "~cbf_tracking_feedback_mode must be 'full', 'none', 'tangent', or 'tangent_near'")
        self.cbf_tangent_feedback_min_speed = float(rospy.get_param(
            "~cbf_tangent_feedback_min_speed", 0.02))
        self.cbf_position_correction_dt = float(rospy.get_param(
            "~cbf_position_correction_dt", 0.02))
        self.cbf_position_correction_filter_tau = float(rospy.get_param(
            "~cbf_position_correction_filter_tau", 0.08))
        self.cbf_max_position_correction = float(rospy.get_param(
            "~cbf_max_position_correction", 0.025))
        self.cbf_use_reactive_position_step = _bool_param(
            "~cbf_use_reactive_position_step", True)
        self.cbf_reactive_position_dt = float(rospy.get_param(
            "~cbf_reactive_position_dt", 0.04))
        self.cbf_max_command_lead = float(rospy.get_param("~cbf_max_command_lead", 0.08))
        self.cbf_gradient_check_period = float(rospy.get_param("~cbf_gradient_check_period", 1.0))
        self.cbf_gradient_check_eps = float(rospy.get_param("~cbf_gradient_check_eps", 1e-3))
        self.log_cbf_events = _bool_param("~log_cbf_events", False)
        self.log_cbf_command_timing = _bool_param("~log_cbf_command_timing", True)
        self.log_trajectory_timing = _bool_param("~log_trajectory_timing", True)
        self.active_plan_id = 0
        self.active_plan_stamp = None
        self.active_plan_duration = 0.0
        self._cbf_first_nominal_for_plan = False
        self._cbf_first_safe_for_plan = False
        self._last_gradient_check = 0.0
        self.last_cbf_position_delta = None

        self.cbf_graph_points = max(1, int(rospy.get_param("~cbf_graph_points", 100)))
        preprocess_max_points = int(rospy.get_param("~preprocess_max_points", 512))
        self.preprocess_max_points = (
            max(self.cbf_graph_points, preprocess_max_points)
            if preprocess_max_points > 0 else 0
        )

        # Setup the new fast CUDA graph
        self.setup_cuda_graph(batch_size=1, n_points=self.cbf_graph_points)
        self._log_cuda_memory("CBF after CUDA graph")
        
        self.joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
            'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]
        self.current_q = None
        self.nominal_q = None
        self.target_x = None
        self.obs_points = torch.empty((0, 3), dtype=torch.float32, device=self.device)

        self.max_joint_velocity = float(rospy.get_param("~max_joint_velocity", 0.7))
        self.cbf_kp = float(rospy.get_param("~cbf_kp", 10.0))
        self.cbf_feedback_coupling = float(rospy.get_param("~cbf_feedback_coupling", 0.1))
        self.max_feedback_velocity = float(rospy.get_param("~max_feedback_velocity", 0.05))
        self.cbf_filter_tau = float(rospy.get_param("~cbf_filter_tau", 0.05))
        # Output low-pass on the safe velocity (0 = off). Applied before the
        # final-constraint repair so the smoothed command is re-validated.
        self.cbf_velocity_filter_tau = float(rospy.get_param("~cbf_velocity_filter_tau", 0.0))
        self._dq_safe_filtered = None
        self.real_velocity_filter_tau = float(rospy.get_param("~real_velocity_filter_tau", 0.08))
        self.last_nominal_q = None
        self.dq_nom_filtered = None
        self.last_nominal_tangent_dir = None
        self.last_h_value_for_feedback = float("inf")
        # Retimed feedforward from the trajectory executor (constant joint
        # speed). Position differencing is only the fallback when stale.
        self.use_nominal_velocity_topic = _bool_param("~use_nominal_velocity_topic", True)
        self.nominal_velocity_topic = str(rospy.get_param(
            "~nominal_velocity_topic", "/planner/nominal_joint_velocity"))
        self.nominal_velocity_timeout = float(rospy.get_param(
            "~nominal_velocity_timeout", 0.25))
        self.nominal_feedforward_gain = float(rospy.get_param(
            "~nominal_feedforward_gain", 1.0))
        self.nominal_dq_ff = None
        self.nominal_dq_stamp = 0.0

        self.enable_cbf = _bool_param("~enable_cbf", True)
        # Monitor-only baseline mode: the barrier, diagnostics and the
        # counterfactual correction are still computed, but the nominal
        # velocity is passed through unmodified ("sans CBF" runs).
        self.cbf_monitor_only = _bool_param("~cbf_monitor_only", False)
        # Perception-referenced contact stop: latch a permanent position hold
        # the first time the estimated true clearance (h_corr + d_safe) drops
        # below ~contact_stop_clearance for ~contact_stop_cycles consecutive
        # cycles. Needed because the fork has no Gazebo collision geometry.
        self.contact_stop_enabled = _bool_param("~contact_stop_enabled", False)
        self.contact_stop_clearance = float(rospy.get_param("~contact_stop_clearance", 0.0))
        self.contact_stop_cycles = max(1, int(rospy.get_param("~contact_stop_cycles", 2)))
        self._contact_counter = 0
        self._contact_stopped = False
        self._contact_q = None
        self.publish_controller_command = _bool_param("~publish_controller_command", False)
        self.preprocess_rate_hz = float(rospy.get_param("~preprocess_rate_hz", 30.0))
        self.viz_rate_hz = float(rospy.get_param("~viz_rate_hz", 5.0))
        self.publish_debug_topics = _bool_param("~publish_debug_topics", False)
        self.publish_diagnostics = _bool_param("~publish_diagnostics", True)
        self.publish_viz_topics = _bool_param("~publish_viz_topics", True)
        self.profile_sync = _bool_param("~profile_sync", False)
        self.cuda_memory_log_period = float(rospy.get_param("~cuda_memory_log_period", 10.0))
        self._last_cuda_memory_log = 0.0
        self.tcp_filter_radius = float(rospy.get_param("~tcp_filter_radius", 0.40))
        self.fork_filter_radius = float(rospy.get_param("~fork_filter_radius", 0.15))
        self.nominal_hold_deadband = float(rospy.get_param("~nominal_hold_deadband", 1e-4))
        self.cbf_candidate_filter = str(rospy.get_param("~cbf_candidate_filter", "sphere")).lower()
        if self.cbf_candidate_filter not in ("sphere", "sdf"):
            raise ValueError("~cbf_candidate_filter must be 'sphere' or 'sdf'")
        self.cbf_sdf_candidate_max_dist = float(rospy.get_param("~cbf_sdf_candidate_max_dist", 0.15))
        self.cbf_sdf_prune_chunk_size = max(1, int(rospy.get_param("~cbf_sdf_prune_chunk_size", 4096)))
        self.cbf_sdf_self_filter_margin = float(rospy.get_param("~cbf_sdf_self_filter_margin", -0.003))
        self.cbf_sdf_self_filter_all_links = _bool_param("~cbf_sdf_self_filter_all_links", False)
        self.cbf_sdf_prefilter_radius = float(rospy.get_param("~cbf_sdf_prefilter_radius", 0.45))
        self.cbf_sdf_prefilter_max_points = int(rospy.get_param("~cbf_sdf_prefilter_max_points", 1024))
        self.cbf_selection_metric = str(rospy.get_param("~cbf_selection_metric", "fork_mesh")).lower()
        if self.cbf_selection_metric not in ("distance", "sdf", "fork_sdf", "fork_mesh"):
            raise ValueError("~cbf_selection_metric must be 'distance', 'sdf', 'fork_sdf', or 'fork_mesh'")
        self.cbf_cluster_mode = str(rospy.get_param("~cbf_cluster_mode", "local")).lower()
        if self.cbf_cluster_mode not in ("local", "topk"):
            raise ValueError("~cbf_cluster_mode must be 'local' or 'topk'")
        self.fork_mesh_selection_points = max(
            1, int(rospy.get_param("~fork_mesh_selection_points", 512)))
        self.fork_mesh_points_local = self._load_fork_mesh_points(urdf_path)
        self.publish_yellow_points = _bool_param("~publish_yellow_points", False)
        self.selected_obstacle_hold_time = float(rospy.get_param("~selected_obstacle_hold_time", 0.5))

        self.eye4 = torch.eye(4, device=self.device).unsqueeze(0)
        self.q_pad2 = torch.zeros((1, 2), device=self.device)
        self.dummy_obs = torch.full((self.cbf_graph_points, 3), 100.0, device=self.device)
        self.zero_dq = torch.zeros((1, 7), device=self.device)

        self.selected_obs = self.dummy_obs.clone()
        self.selected_pts_yellow = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.selected_num_inside = 0
        self.selected_count = 0
        self.selected_min_obs_dist = float('inf')
        self.selected_obs_stamp = rospy.get_time()
        self.debug_fork_mesh_world = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.debug_score_seed = None
        self.debug_tip_min = None
        self.debug_filter_center = None

        self.last_q_safe = None
        self.q_safe_work = torch.zeros((1, 7), device=self.device)
        self.dq_safe_work = torch.zeros((1, 7), device=self.device)
        self.last_joint_q_for_velocity = None
        self.last_joint_stamp_for_velocity = None
        self.dq_real_measured = torch.zeros((1, 7), device=self.device)
        self.last_current_q_debug = None
        self._viz_q = None
        self._viz_dq_nom = None
        self._viz_dq_safe = None
        self.last_h_debug = None
        # Slots 0-38: original telemetry. Slots 39-63: diagnostics extension
        # (39:46 dq_ff, 46:53 dq_fb, 53:60 solver output, 60 repair flag,
        #  61 final constraint, 62 pre-projection constraint, 63 |grad_h|).
        self.transfer_buffer = torch.zeros(64, dtype=torch.float32, device=self.device)
        self._last_active_constraints = 0
        self._log_cuda_memory("CBF after runtime buffers")

        self.comp_times = []
        self.preprocess_times = []
        rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        rospy.Subscriber('/planner/nominal_trajectory', JointTrajectory, self.trajectory_timing_callback)
        rospy.Subscriber('/planner/nominal_joint_command', Float64MultiArray, self.nominal_command_callback)
        rospy.Subscriber(self.nominal_velocity_topic, Float64MultiArray, self._nominal_vel_cb)
        obs_topic = rospy.get_param("~obs_topic", "/perception/persistent_obstacles")
        rospy.Subscriber(obs_topic, PointCloud2, self.obs_callback)
        
        self.cmd_pub = rospy.Publisher('/franka_control/safe_joint_velocities', Float32MultiArray, queue_size=1)
        self.pos_cmd_pub = None
        if self.publish_controller_command:
            self.pos_cmd_pub = rospy.Publisher('/joint_group_position_controller/command', Float64MultiArray, queue_size=1)
        self.safe_cmd_pub = rospy.Publisher('/planner/safe_joint_command', Float64MultiArray, queue_size=1)
        self.ready_pub = rospy.Publisher('/cbf_safety/ready', Bool, queue_size=1, latch=True)
        self.h_pub = rospy.Publisher('/cbf_safety/h_value', Float32MultiArray, queue_size=1)
        self.contact_event_pub = rospy.Publisher(
            '/cbf_safety/contact_event', Float32MultiArray, queue_size=1, latch=True)
        self.diag_pub = rospy.Publisher('/cbf_safety/diagnostics', Float32MultiArray, queue_size=10)
        self.diag_layout_pub = rospy.Publisher('/cbf_safety/diagnostics_layout', String, queue_size=1, latch=True)
        self.diag_layout_pub.publish(String(data=json.dumps({
            "vec_fields": DIAG_VEC_FIELDS,
            "vec_dim": 7,
            "scalar_fields": DIAG_SCALAR_FIELDS,
            "vel_source_codes": {"topic": 0.0, "diff": 1.0, "zero": 2.0},
            "node": "multicbf",
        })))
        self.debug_input_pub = rospy.Publisher('/debug/cbf/input_joint_command_seen', Float64MultiArray, queue_size=1)
        self.debug_output_pub = rospy.Publisher('/debug/cbf/output_joint_command', Float64MultiArray, queue_size=1)
        self.debug_input_vel_pub = rospy.Publisher('/debug/cbf/input_joint_velocity', Float32MultiArray, queue_size=1)
        self.debug_output_vel_pub = rospy.Publisher('/debug/cbf/output_joint_velocity', Float32MultiArray, queue_size=1)
        self.debug_alignment_pub = rospy.Publisher('/debug/cbf/nominal_safe_alignment', Float32MultiArray, queue_size=1)
        
        # Publishers RViz (Uniquement Jaune et Rouge)
        self.pub_inside_yellow = rospy.Publisher('/viz/obs_inside_yellow', PointCloud2, queue_size=1)
        self.pub_top100_red = rospy.Publisher('/viz/obs_top100_red', PointCloud2, queue_size=1)
        self.pub_fork_mesh_debug = rospy.Publisher('/viz/cbf_fork_mesh_points', PointCloud2, queue_size=1)
        self.pub_selection_debug = rospy.Publisher('/viz/cbf_selection_debug', MarkerArray, queue_size=1)
        self.safe_traj_viz_pub = rospy.Publisher('/viz/safe_trajectory_3d', MarkerArray, queue_size=1)
        self.pub_cbf_velocity_arrows = rospy.Publisher('/viz/cbf_velocity_arrows', MarkerArray, queue_size=1)
        
        self.rate_hz = float(rospy.get_param("~rate_hz", 150.0))
        if self.rate_hz <= 0.0:
            raise ValueError("~rate_hz must be > 0")
        self.rate = rospy.Rate(self.rate_hz)
        self.last_time = rospy.get_time()
        rospy.Timer(rospy.Duration(1.0 / self.preprocess_rate_hz),
                    self.preprocess_obstacles)
        if self.publish_viz_topics:
            rospy.Timer(rospy.Duration(1.0 / self.viz_rate_hz),
                        self.publish_visualization)
        rospy.loginfo(
            "Multi-constraint CBF ready: control=%.1f Hz preprocess=%.1f Hz viz=%.1f Hz "
            "graph_points=%d active_constraints=%d h_activate=%.3fm max_inward=%.3fm/s "
            "recovery=%.3fm/s candidate_filter=%s selection=%s cluster=%s "
            "integrate_from_current=%s command_dt=%.3fs passthrough_inactive=%s "
            "yellow=%s debug=%s profile_sync=%s",
            self.rate_hz, self.preprocess_rate_hz, self.viz_rate_hz,
            self.cbf_graph_points,
            self.cbf_active_constraints,
            self.cbf_h_activate,
            self.cbf_max_inward_speed,
            self.cbf_recovery_speed,
            self.cbf_candidate_filter,
            self.cbf_selection_metric,
            self.cbf_cluster_mode,
            self.cbf_integrate_from_current,
            self.cbf_command_dt,
            self.cbf_passthrough_when_inactive,
            self.publish_yellow_points,
            self.publish_debug_topics, self.profile_sync)
        self.ready_pub.publish(Bool(data=True))

    def _log_cuda_memory(self, label):
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            return
        torch.cuda.synchronize(self.device)
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        rospy.loginfo(
            "%s CUDA memory | allocated=%.1f MiB reserved=%.1f MiB max_allocated=%.1f MiB",
            label, allocated, reserved, max_allocated)

    def _log_cuda_memory_throttle(self, label):
        if self.cuda_memory_log_period <= 0.0:
            return
        now = rospy.get_time()
        if now - self._last_cuda_memory_log >= self.cuda_memory_log_period:
            self._last_cuda_memory_log = now
            self._log_cuda_memory(label)

    def _resolve_mesh_path(self, mesh_uri):
        if not mesh_uri:
            return None
        if mesh_uri.startswith('package://'):
            relative = mesh_uri[len('package://'):]
            package_name, _, path_in_package = relative.partition('/')
            try:
                return os.path.join(rospack.get_path(package_name), path_in_package)
            except Exception as exc:
                rospy.logwarn("Could not resolve mesh package '%s': %s", package_name, exc)
                return None
        return mesh_uri

    def _origin_to_matrix(self, origin_elem):
        T = np.eye(4, dtype=np.float64)
        if origin_elem is None:
            return T

        xyz = np.fromstring(origin_elem.get('xyz', '0 0 0'), sep=' ', dtype=np.float64)
        rpy = np.fromstring(origin_elem.get('rpy', '0 0 0'), sep=' ', dtype=np.float64)
        if xyz.size != 3:
            xyz = np.zeros(3, dtype=np.float64)
        if rpy.size != 3:
            rpy = np.zeros(3, dtype=np.float64)

        r, p, y = rpy
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(r), -np.sin(r)],
                       [0.0, np.sin(r), np.cos(r)]], dtype=np.float64)
        Ry = np.array([[np.cos(p), 0.0, np.sin(p)],
                       [0.0, 1.0, 0.0],
                       [-np.sin(p), 0.0, np.cos(p)]], dtype=np.float64)
        Rz = np.array([[np.cos(y), -np.sin(y), 0.0],
                       [np.sin(y), np.cos(y), 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = xyz
        return T

    def _load_fork_mesh_points(self, urdf_path):
        empty = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        if self.cbf_selection_metric != "fork_mesh":
            return empty
        if trimesh is None:
            rospy.logwarn("trimesh is unavailable; falling back from fork_mesh to distance selection.")
            self.cbf_selection_metric = "distance"
            return empty

        try:
            root = ET.parse(urdf_path).getroot()
            fork_link = root.find("./link[@name='fork_tip']")
            if fork_link is None:
                raise ValueError("fork_tip link not found in URDF")

            visual = fork_link.find('visual')
            geometry = visual.find('geometry') if visual is not None else None
            mesh_elem = geometry.find('mesh') if geometry is not None else None
            if mesh_elem is None:
                raise ValueError("fork_tip visual mesh not found in URDF")

            mesh_path = self._resolve_mesh_path(mesh_elem.get('filename'))
            if not mesh_path or not os.path.exists(mesh_path):
                raise FileNotFoundError(mesh_path or mesh_elem.get('filename'))

            mesh = trimesh.load(mesh_path, force='mesh')
            scale = np.fromstring(mesh_elem.get('scale', '1 1 1'), sep=' ', dtype=np.float64)
            if scale.size != 3:
                scale = np.ones(3, dtype=np.float64)
            mesh.apply_scale(scale)
            mesh.apply_transform(self._origin_to_matrix(visual.find('origin')))

            points = mesh.sample(self.fork_mesh_selection_points).astype(np.float32)
            rospy.loginfo(
                "Loaded %d fork mesh selection points from %s",
                points.shape[0], mesh_path)
            return torch.as_tensor(points, dtype=torch.float32, device=self.device)
        except Exception as exc:
            rospy.logwarn(
                "Could not load fork mesh selection points (%s); falling back to distance selection.",
                exc)
            self.cbf_selection_metric = "distance"
            return empty

    def _barrier_h_value_no_grad(self, q, obs):
        _, sdf_per_link = self.bernstein_core.get_whole_body_sdf_batch(
            obs, self.eye4, q, return_per_link=True)
        sdf_min, _ = sdf_per_link.min(dim=-1, keepdim=True)
        shifted_sdf = sdf_per_link - sdf_min
        exp_terms = torch.exp(-shifted_sdf / self.barrier.alpha)
        h_per_link = (
            sdf_min.squeeze(-1)
            - self.barrier.alpha * torch.log(exp_terms.sum(dim=-1))
            - self.d_safe
        )
        return h_per_link.min(dim=1).values

    def _maybe_check_gradient_direction(self, current_q, obs, h_now, grad_h):
        if self.cbf_gradient_check_period <= 0.0:
            return
        now = rospy.get_time()
        if now - self._last_gradient_check < self.cbf_gradient_check_period:
            return
        self._last_gradient_check = now

        grad_norm = torch.norm(grad_h, dim=1, keepdim=True)
        if grad_norm.item() < 1e-8:
            rospy.logwarn_throttle(
                2.0,
                "CBF grad check skipped: gradient norm is near zero while h=%.4f",
                float(h_now.item()),
            )
            return

        eps = max(self.cbf_gradient_check_eps, 1e-6)
        direction = grad_h / (grad_norm + 1e-9)
        h_plus = self._barrier_h_value_no_grad(current_q + eps * direction, obs)
        h_minus = self._barrier_h_value_no_grad(current_q - eps * direction, obs)
        fd_grad = (h_plus - h_minus) / (2.0 * eps)
        predicted = grad_norm.squeeze(1)

        if fd_grad.item() < -1e-4:
            rospy.logwarn_throttle(
                1.0,
                "CBF GRADIENT SIGN SUSPECT | h=%.4f h+=%.4f h-=%.4f "
                "fd=%.4f predicted=%.4f",
                float(h_now.item()),
                float(h_plus.item()),
                float(h_minus.item()),
                float(fd_grad.item()),
                float(predicted.item()),
            )
        else:
            rospy.loginfo_throttle(
                2.0,
                "CBF grad check | h=%.4f h+=%.4f h-=%.4f fd=%.4f predicted=%.4f",
                float(h_now.item()),
                float(h_plus.item()),
                float(h_minus.item()),
                float(fd_grad.item()),
                float(predicted.item()),
            )

    def setup_cuda_graph(self, batch_size=1, n_points=100):
        """
        Single unified CUDA graph:
          Calculates whole-body safety over n_points selected obstacle points.
          dq_safe = dq_nom + lam * grad_h.
        """
        print(f"⚡ Initialisation Multi-CBF ({n_points} selected points)...")
        torch.cuda.empty_cache()

        self.static_q      = torch.zeros((batch_size, 7), device=self.device, requires_grad=True)
        self.static_obs    = torch.zeros((n_points, 3),   device=self.device)
        self.static_dq_nom = torch.zeros((batch_size, 7), device=self.device)
        self.static_dq_safe = torch.zeros((batch_size, 7), device=self.device)
        self.static_h      = torch.zeros((batch_size,), device=self.device)
        self.static_constr = torch.zeros((batch_size,), device=self.device)
        self.static_grad_h = torch.zeros((batch_size, 7), device=self.device)
        self.graph = None
        if self.cbf_solver_mode != "fast_tangent":
            self._log_cuda_memory("Multi-CBF runtime buffers ready")
            return

        # Capture scalars as Python floats so the CUDA graph bakes in the params.
        constraint_margin = self.cbf_constraint_margin
        h_activate = max(self.cbf_h_activate, 1e-6)
        max_inward_speed = max(0.0, self.cbf_max_inward_speed)
        recovery_speed = max(0.0, self.cbf_recovery_speed)
        recovery_depth = max(self.cbf_recovery_depth, 1e-6)
        max_correction_speed = max(0.0, self.cbf_max_correction_speed)

        def _qp_step(h, grad_h, dq_in):
            self.static_h.copy_(h)
            # Tangent-preserving fast CBF:
            # - outside activation: no correction
            # - near boundary: limit inward normal velocity only
            # - inside: request a bounded outward recovery speed
            active = h <= h_activate
            outside_bound = -max_inward_speed * torch.clamp(
                h / h_activate, min=0.0, max=1.0)
            inside_bound = recovery_speed * torch.clamp(
                (-h) / recovery_depth, min=0.0, max=1.0)
            bound = torch.where(h > 0.0, outside_bound, inside_bound)
            bound = bound + constraint_margin
            gdq = (grad_h * dq_in).sum(dim=-1)
            constr = torch.where(active, gdq - bound, torch.ones_like(gdq))
            self.static_constr.copy_(constr)
            denom  = (grad_h ** 2).sum(dim=-1)
            correction = torch.clamp(bound - gdq, min=0.0)
            dq_delta = correction.unsqueeze(-1) * grad_h / (denom.unsqueeze(-1) + 1e-4)
            if max_correction_speed > 0.0:
                delta_norm = torch.norm(dq_delta, dim=-1, keepdim=True)
                scale = torch.clamp(max_correction_speed / (delta_norm + 1e-6), max=1.0)
                dq_delta = dq_delta * scale
            dq_projected = dq_in + dq_delta

            dq_safe = torch.where(
                (active & (constr < 0.0) & (denom > 1e-8)).unsqueeze(-1),
                torch.where(
                    denom.unsqueeze(-1) < 1e-6,
                    torch.zeros_like(dq_in),
                    dq_projected,
                ),
                dq_in
            )
            return dq_safe

        # ── Warmup ────────────────────────────────────────────────────────────
        def _warmup():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    if self.static_q.grad is not None: self.static_q.grad.zero_()
                    h_w, g_w, _ = self.barrier(self.static_q, self.static_obs)
                    self.static_grad_h.copy_(g_w)
                    self.static_dq_safe.copy_(_qp_step(h_w, g_w, self.static_dq_nom))
            torch.cuda.current_stream().wait_stream(s)

        _warmup()
        torch.cuda.empty_cache()

        # ── Unified Graph ─────────────────────────────────────────────────────
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            if self.static_q.grad is not None: self.static_q.grad.zero_()
            h, g, _ = self.barrier(self.static_q, self.static_obs)
            self.static_grad_h.copy_(g)
            self.static_dq_safe.copy_(_qp_step(h, g, self.static_dq_nom))

        print(f"✅ Graphe CUDA capturé ({n_points} points) !")
        self._log_cuda_memory("CBF setup_cuda_graph captured")

    def solve_multicbf_projection(self, current_q, obs_points, dq_nom):
        """
        Tangent-preserving multi-constraint CBF.

        Each active obstacle point contributes one half-space:
            grad_h_i · dq >= bound_i(h_i)
        The sequential projection keeps dq as close as possible to dq_nom while
        removing only the unsafe inward normal component.
        """
        valid_count = min(
            int(self.selected_count),
            int(obs_points.shape[0]),
            int(self.preprocess_max_points) if self.preprocess_max_points > 0 else int(obs_points.shape[0]),
        )
        if valid_count <= 0:
            self._last_active_constraints = 0
            self.static_h.fill_(1.0)
            self.static_constr.zero_()
            self.static_grad_h.zero_()
            return dq_nom

        obs_eval = obs_points[:valid_count]
        with torch.no_grad():
            q_probe = current_q.detach()
            pose_probe = self.eye4.expand(q_probe.shape[0], 4, 4)
            sdf_probe = self.bernstein_core.get_whole_body_sdf_batch(
                obs_eval, pose_probe, q_probe, return_per_link=False)
            h_probe = sdf_probe[0] - self.d_safe

        active_window = min(valid_count, max(self.cbf_active_constraints * 3, self.cbf_active_constraints))
        _, near_idx = torch.topk(h_probe, k=active_window, largest=False, sorted=True)
        h_near = h_probe[near_idx]
        active_mask = h_near <= self.cbf_h_activate
        if not torch.any(active_mask):
            self._last_active_constraints = 0
            h_min = h_probe.min().view(1)
            self.static_h.copy_(h_min)
            self.static_constr.copy_(torch.zeros_like(h_min))
            self.static_grad_h.zero_()
            return dq_nom

        active_idx = near_idx[active_mask][:self.cbf_active_constraints]
        obs_active = obs_eval[active_idx]

        prev_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        q_eval = current_q.detach().clone().requires_grad_(True)
        pose = self.eye4.expand(q_eval.shape[0], 4, 4)
        sdf_active = self.bernstein_core.get_whole_body_sdf_batch(
            obs_active, pose, q_eval, return_per_link=False)
        h_active = sdf_active[0] - self.d_safe

        if int(h_active.numel()) <= 0:
            self._last_active_constraints = 0
            self.static_h.copy_(h_probe.min().view(1))
            self.static_constr.zero_()
            self.static_grad_h.zero_()
            torch.set_grad_enabled(prev_grad_enabled)
            return dq_nom

        # One batched vector-Jacobian product is much cheaper than K separate
        # backward passes. Fall back only for older PyTorch versions.
        try:
            grad_outputs = torch.eye(
                int(h_active.numel()), device=self.device, dtype=h_active.dtype)
            grad_batch = torch.autograd.grad(
                outputs=h_active,
                inputs=q_eval,
                grad_outputs=grad_outputs,
                is_grads_batched=True,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )[0]
            G = grad_batch[:, 0, :]
        except TypeError:
            grads = []
            for j, h_j in enumerate(h_active):
                grad_j = torch.autograd.grad(
                    outputs=h_j,
                    inputs=q_eval,
                    grad_outputs=torch.ones_like(h_j),
                    retain_graph=(j < int(h_active.numel()) - 1),
                    create_graph=False,
                    only_inputs=True,
                )[0]
                grads.append(grad_j.squeeze(0))
            G = torch.stack(grads, dim=0)
        h_det = h_active.detach()

        h_activate = max(self.cbf_h_activate, 1e-6)
        recovery_depth = max(self.cbf_recovery_depth, 1e-6)
        outside_bound = -self.cbf_max_inward_speed * torch.clamp(
            h_det / h_activate, min=0.0, max=1.0)
        inside_bound = self.cbf_recovery_speed * torch.clamp(
            (-h_det) / recovery_depth, min=0.0, max=1.0)
        bounds = torch.where(h_det > 0.0, outside_bound, inside_bound)
        bounds = bounds + self.cbf_constraint_margin

        # The projection itself is a 7-variable QP with K half-spaces: a CPU
        # problem. Solving it on GPU forced one host sync per constraint per
        # sweep (`bool(violation > 0)`); instead, pack G/bounds/h/dq into one
        # transfer and run Gauss-Seidel in numpy (microseconds at this size).
        K = int(G.shape[0])
        packed = torch.cat([
            G.reshape(-1), bounds, h_det, dq_nom.detach().view(-1),
        ]).cpu().numpy().astype(np.float64)
        G_np = packed[:7 * K].reshape(K, 7)
        b_np = packed[7 * K:8 * K]
        h_np = packed[8 * K:9 * K]
        dq_np = packed[9 * K:9 * K + 7].copy()

        row_sq = np.einsum('ij,ij->i', G_np, G_np) + 1e-6
        vmax = self.max_joint_velocity
        relax = self.cbf_projection_relaxation
        for _ in range(self.cbf_projection_iters):
            updated = False
            for i in range(K):
                violation = b_np[i] - G_np[i].dot(dq_np)
                if violation > 0.0:
                    dq_np += relax * violation * G_np[i] / row_sq[i]
                    np.clip(dq_np, -vmax, vmax, out=dq_np)
                    updated = True
            if not updated:
                break

        constr = G_np.dot(dq_np) - b_np
        min_idx = int(np.argmin(h_np))
        self._last_active_constraints = K
        self.static_h.copy_(h_det[min_idx].view(1))
        self.static_constr.fill_(float(constr.min()))
        self.static_grad_h.copy_(G[min_idx].view(1, 7))
        torch.set_grad_enabled(prev_grad_enabled)
        return torch.as_tensor(
            dq_np, dtype=torch.float32, device=self.device).view(1, 7)

    def joint_callback(self, msg):
        try:
            pos_dict = {n: p for n, p in zip(msg.name, msg.position)}
            q_list = []
            for jn in self.joint_names:
                if jn in pos_dict:
                    q_list.append(pos_dict[jn])
                else:
                    return # Ignore message if arm joints are missing
            q_now = torch.tensor(q_list, dtype=torch.float32, device=self.device).unsqueeze(0)
            stamp = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0.0 else rospy.get_time()
            if self.last_joint_q_for_velocity is not None and self.last_joint_stamp_for_velocity is not None:
                dt = stamp - self.last_joint_stamp_for_velocity
                if 1e-4 < dt < 0.5:
                    dq_real_raw = (q_now - self.last_joint_q_for_velocity) / dt
                    gamma = dt / (dt + self.real_velocity_filter_tau)
                    self.dq_real_measured = (
                        (1.0 - gamma) * self.dq_real_measured
                        + gamma * dq_real_raw
                    )
            self.last_joint_q_for_velocity = q_now.detach().clone()
            self.last_joint_stamp_for_velocity = stamp
            self.current_q = q_now
            if self.last_q_safe is None:
                self.last_q_safe = self.current_q.detach().clone()
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error in CBF joint_callback: {e}")

    def nominal_command_callback(self, msg):
        if len(msg.data) >= 7:
            self.nominal_q = torch.tensor(msg.data[:7], dtype=torch.float32,
                                          device=self.device).unsqueeze(0)
            if (
                self.log_trajectory_timing
                and self.active_plan_stamp is not None
                and not self._cbf_first_nominal_for_plan
            ):
                if self.current_q is None:
                    err = float('nan')
                else:
                    err = float(torch.norm(self.nominal_q - self.current_q).item())
                now = rospy.Time.now()
                rospy.loginfo(
                    "[TRAJ TIMING] cbf_first_nominal id=%d "
                    "publish_to_cbf_nominal=%.3fs expected_duration=%.3fs "
                    "err_to_current=%.4frad",
                    self.active_plan_id,
                    (now - self.active_plan_stamp).to_sec(),
                    self.active_plan_duration,
                    err,
                )
                self._cbf_first_nominal_for_plan = True
            if not self._logged_first_nominal:
                if self.current_q is None:
                    err = float('nan')
                else:
                    err = float(torch.norm(self.nominal_q - self.current_q).item())
                rospy.loginfo(
                    "CBF received first nominal joint command | err_to_current=%.4f rad",
                    err)
                self._logged_first_nominal = True

    def trajectory_timing_callback(self, msg):
        if not msg.points:
            return
        plan_stamp = msg.header.stamp
        if plan_stamp.to_sec() <= 0.0:
            plan_stamp = rospy.Time.now()
        plan_id = int(getattr(msg.header, 'seq', 0))
        if plan_id <= 0:
            self.active_plan_id += 1
        else:
            self.active_plan_id = plan_id
        self.active_plan_stamp = plan_stamp
        self.active_plan_duration = msg.points[-1].time_from_start.to_sec()
        self._cbf_first_nominal_for_plan = False
        self._cbf_first_safe_for_plan = False
        if self.nominal_q is not None:
            self.last_nominal_q = self.nominal_q.detach().clone()
        if self.dq_nom_filtered is not None:
            self.dq_nom_filtered.zero_()
        self.last_nominal_tangent_dir = None

    def _shape_tracking_feedback(self, dq_nom_pure, dq_fb):
        mode = self.cbf_tracking_feedback_mode
        if mode == "full":
            return dq_fb
        if mode == "none":
            return torch.zeros_like(dq_fb)
        if mode == "tangent_near" and self.last_h_value_for_feedback > self.cbf_h_activate:
            return dq_fb

        tangent_speed = torch.norm(dq_nom_pure, dim=1, keepdim=True)
        if bool(torch.all(tangent_speed > self.cbf_tangent_feedback_min_speed)):
            tangent_dir = dq_nom_pure / (tangent_speed + 1e-6)
            self.last_nominal_tangent_dir = tangent_dir.detach().clone()
        elif self.last_nominal_tangent_dir is not None:
            tangent_dir = self.last_nominal_tangent_dir
        else:
            return torch.zeros_like(dq_fb)

        forward_speed = (dq_fb * tangent_dir).sum(dim=1, keepdim=True).clamp(min=0.0)
        return forward_speed * tangent_dir

    def _nominal_vel_cb(self, msg):
        if len(msg.data) < 7:
            return
        dq = torch.tensor(msg.data[:7], dtype=torch.float32, device=self.device).unsqueeze(0)
        self.nominal_dq_ff = dq
        self.nominal_dq_stamp = rospy.get_time()

    def obs_callback(self, msg):
        try:
            points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, int(msg.point_step/4))
            self.obs_points = torch.from_numpy(points[:, :3].copy()).to(
                self.device, dtype=torch.float32)
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error unpacking obstacles in CBF: {e}")

    def create_cloud_xyzrgb(self, points_tensor, color):
        """Créer un nuage XYZRGB pour RViz (sans les verts) via C++ ultra-rapide."""
        if points_tensor.shape[0] == 0:
            return None
            
        r, g, b = color
        pts_np = points_tensor.cpu().numpy().astype(np.float32)

        msg_bytes = fast_perception_module.create_cloud_xyzrgb(pts_np, r, g, b, "world")
        
        msg = PointCloud2()
        msg.deserialize(msg_bytes)
        msg.header.stamp = rospy.Time.now()
        
        return msg

    def _sphere_marker(self, marker_id, ns, point_tensor, color, scale):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]

        p = point_tensor.detach().flatten().cpu().numpy()
        marker.pose.position.x = float(p[0])
        marker.pose.position.y = float(p[1])
        marker.pose.position.z = float(p[2])
        return marker

    def publish_selection_debug_markers(self):
        markers = MarkerArray()
        if self.debug_score_seed is not None:
            # Magenta: obstacle point used as the seed for the one-cluster selection.
            markers.markers.append(self._sphere_marker(
                0, "cbf_score_seed", self.debug_score_seed, (1.0, 0.0, 1.0), 0.025))
        if self.debug_tip_min is not None:
            # Cyan: obstacle point with minimum Euclidean distance to fork_tip.
            markers.markers.append(self._sphere_marker(
                1, "cbf_tip_min", self.debug_tip_min, (0.0, 1.0, 1.0), 0.022))
        if self.debug_filter_center is not None:
            # White: fork_tip center used for the initial tcp_filter_radius candidate sphere.
            markers.markers.append(self._sphere_marker(
                2, "cbf_filter_center", self.debug_filter_center, (1.0, 1.0, 1.0), 0.018))
        if markers.markers:
            stamp = rospy.Time.now()
            for marker in markers.markers:
                marker.header.stamp = stamp
            self.pub_selection_debug.publish(markers)

    def get_link_pose(self, q7, link_name):
        if q7.dim() == 1:
            q7 = q7.unsqueeze(0)
        q_eval = q7
        if q_eval.shape[-1] == 7:
            q_eval = torch.cat([
                q_eval,
                torch.zeros((q_eval.shape[0], 2), device=self.device)
            ], dim=-1)

        link_poses = self.robot_layer._native_forward_kinematics(q_eval)
        return link_poses.get(link_name, None)

    def get_fork_tip_pose(self, q7):
        # Use fork_tip so that it aligns with nominal_trajectory_follower
        # and the Flow Matching trajectory coordinates.
        return self.get_link_pose(q7, 'fork_tip')

    def publish_safe_trajectory_marker(self, q_start, q_goal):
        q_start = q_start.detach()
        q_goal = q_goal.detach()
        num_samples = 16
        alpha = torch.linspace(0.0, 1.0, num_samples, device=self.device).view(num_samples, 1)
        q_path = q_start + alpha * (q_goal - q_start)

        T_path = self.get_fork_tip_pose(q_path)
        if T_path is None:
            return

        marker_array = MarkerArray()

        line_marker = Marker()
        line_marker.header.frame_id = "world"
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = "safe_trajectory"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.012
        line_marker.color.a = 1.0
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0

        path_xyz = T_path[:, :3, 3].detach().cpu().numpy()
        for xyz in path_xyz:
            p = Point()
            p.x = float(xyz[0])
            p.y = float(xyz[1])
            p.z = float(xyz[2])
            line_marker.points.append(p)

        head_marker = Marker()
        head_marker.header.frame_id = "world"
        head_marker.header.stamp = line_marker.header.stamp
        head_marker.ns = "safe_trajectory_head"
        head_marker.id = 1
        head_marker.type = Marker.SPHERE
        head_marker.action = Marker.ADD
        head_marker.pose.position = line_marker.points[-1]
        head_marker.scale.x = 0.025
        head_marker.scale.y = 0.025
        head_marker.scale.z = 0.025
        head_marker.color.a = 1.0
        head_marker.color.r = 0.0
        head_marker.color.g = 1.0
        head_marker.color.b = 0.0

        marker_array.markers.append(line_marker)
        marker_array.markers.append(head_marker)
        self.safe_traj_viz_pub.publish(marker_array)

    def _dq_to_cartesian_dir(self, q7, dq7, eps=0.02):
        """Return the TCP xyz displacement for a unit step in joint direction dq7."""
        norm = torch.norm(dq7)
        if norm < 1e-6:
            return None
        dq_unit = dq7 / norm
        T0 = self.get_fork_tip_pose(q7)
        T1 = self.get_fork_tip_pose(q7 + eps * dq_unit)
        if T0 is None or T1 is None:
            return None
        p0 = T0[0, :3, 3].cpu().numpy()
        p1 = T1[0, :3, 3].cpu().numpy()
        return p0, p1 - p0  # (origin, direction)

    def _make_arrow_marker(self, mid, origin, direction, scale, r, g, b, arrow_len=0.12):
        """Build a Marker.ARROW from origin pointing along direction."""
        import numpy as np
        d_norm = np.linalg.norm(direction)
        if d_norm < 1e-6:
            return None
        tip = origin + (arrow_len / d_norm) * direction
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = rospy.Time.now()
        m.ns = "cbf_velocity"
        m.id = mid
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale.x = 0.008   # shaft diameter
        m.scale.y = 0.016   # head diameter
        m.scale.z = 0.025   # head length
        m.color.a = 0.9
        m.color.r = r
        m.color.g = g
        m.color.b = b
        from geometry_msgs.msg import Point
        tail = Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2]))
        head = Point(x=float(tip[0]),    y=float(tip[1]),    z=float(tip[2]))
        m.points = [tail, head]
        return m

    def publish_cbf_velocity_arrows(self):
        """Visualize three Cartesian arrows at the fork tip:
          Blue  — nominal dq (FM planner intent)
          Green — safe dq   (CBF output)
          Red   — grad_h    (obstacle repulsion direction)
        Seeing these together shows immediately if the CBF is fighting the planner.
        """
        q = self._viz_q
        dq_nom = self._viz_dq_nom
        dq_safe = self._viz_dq_safe
        if q is None or dq_nom is None or dq_safe is None:
            return
        grad_h = self.static_grad_h.detach()

        markers = MarkerArray()
        for mid, dq, r, g, b in [
            (10, dq_nom,  0.1, 0.4, 1.0),   # blue  = nominal
            (11, dq_safe, 0.1, 1.0, 0.2),   # green = safe
            (12, grad_h,  1.0, 0.4, 0.1),   # orange-red = grad_h (repulsion)
        ]:
            result = self._dq_to_cartesian_dir(q, dq.squeeze(0))
            if result is None:
                continue
            origin, direction = result
            m = self._make_arrow_marker(mid, origin, direction, 0.008, r, g, b)
            if m is not None:
                markers.markers.append(m)
        self.pub_cbf_velocity_arrows.publish(markers)

    def _copy_fixed_obstacles(self, obs):
        n = min(int(obs.shape[0]), self.cbf_graph_points)
        if n == self.cbf_graph_points:
            new_obs = obs[:n].clone()
        else:
            new_obs = torch.empty((self.cbf_graph_points, 3), dtype=torch.float32, device=self.device)
            if n > 0:
                new_obs[:n].copy_(obs[:n])
            new_obs[n:].fill_(100.0)

        # Atomic swap
        self.selected_obs = new_obs
        if n > 0:
            self.selected_obs_stamp = rospy.get_time()
        self.selected_count = n

    def _clear_or_hold_obstacles(self):
        if (
            self.selected_obstacle_hold_time > 0.0
            and self.selected_count > 0
            and rospy.get_time() - self.selected_obs_stamp < self.selected_obstacle_hold_time
        ):
            self.selected_pts_yellow = torch.empty((0, 3), dtype=torch.float32,
                                                   device=self.device)
            return

        self._copy_fixed_obstacles(torch.empty((0, 3), device=self.device))
        self.selected_pts_yellow = torch.empty((0, 3), dtype=torch.float32,
                                               device=self.device)
        self.selected_num_inside = 0
        self.selected_min_obs_dist = float('inf')
        self.debug_fork_mesh_world = torch.empty((0, 3), dtype=torch.float32,
                                                 device=self.device)
        self.debug_score_seed = None
        self.debug_tip_min = None
        self.debug_filter_center = None

    def _prefilter_for_sdf(self, pts, q9):
        """Cheap all-link prefilter before expensive whole-body SDF."""
        if self.cbf_sdf_prefilter_radius <= 0.0 and self.cbf_sdf_prefilter_max_points <= 0:
            return pts

        link_poses = self.robot_layer._native_forward_kinematics(q9)
        centers = []
        for link_name in self.protected_link_names:
            T_link = link_poses.get(link_name, None)
            if T_link is not None:
                centers.append(T_link[0, :3, 3])

        if not centers:
            return pts

        centers = torch.stack(centers, dim=0)
        link_dist = torch.cdist(pts.unsqueeze(0), centers.unsqueeze(0)).squeeze(0).min(dim=1).values

        if self.cbf_sdf_prefilter_radius > 0.0:
            keep_mask = link_dist <= self.cbf_sdf_prefilter_radius
            if torch.any(keep_mask):
                pts = pts[keep_mask]
                link_dist = link_dist[keep_mask]

        if self.cbf_sdf_prefilter_max_points > 0 and pts.shape[0] > self.cbf_sdf_prefilter_max_points:
            _, keep_idx = torch.topk(
                link_dist,
                k=self.cbf_sdf_prefilter_max_points,
                largest=False,
            )
            pts = pts[keep_idx]

        return pts

    def _whole_body_sdf_candidates(self, pts, q9):
        """Return points near protected links, scored by whole-body SDF.

        This is outside the CUDA graph and runs at the preprocessing rate, so
        chunking keeps the memory bounded when the persistent cloud is large.
        """
        pts = self._prefilter_for_sdf(pts, q9)
        if pts.shape[0] == 0:
            empty = torch.empty((0,), dtype=torch.float32, device=self.device)
            return torch.empty((0, 3), dtype=torch.float32, device=self.device), empty, empty, empty

        kept_pts = []
        kept_dist = []
        kept_sdf_all = []
        kept_sdf_fork = []

        for start in range(0, int(pts.shape[0]), self.cbf_sdf_prune_chunk_size):
            pts_chunk = pts[start:start + self.cbf_sdf_prune_chunk_size]
            _, sdf_per_link = self.bernstein_core.get_whole_body_sdf_batch(
                pts_chunk, self.eye4, q9, return_per_link=True)

            sdf_body = sdf_per_link[0, :5, :].min(dim=0).values
            sdf_all = sdf_per_link[0].min(dim=0).values
            sdf_fork = sdf_per_link[0, self.fork_link_index, :]
            sdf_self_filter = sdf_all if self.cbf_sdf_self_filter_all_links else sdf_body

            # Keep points close to any protected link, while dropping points
            # clearly inside already-observed robot geometry.
            keep_mask = (
                (sdf_all <= self.cbf_sdf_candidate_max_dist)
                & (sdf_self_filter > self.cbf_sdf_self_filter_margin)
            )
            if torch.any(keep_mask):
                kept_pts.append(pts_chunk[keep_mask])
                kept_dist.append(sdf_all[keep_mask])
                kept_sdf_all.append(sdf_all[keep_mask])
                kept_sdf_fork.append(sdf_fork[keep_mask])

        if not kept_pts:
            empty = torch.empty((0,), dtype=torch.float32, device=self.device)
            return torch.empty((0, 3), dtype=torch.float32, device=self.device), empty, empty, empty

        return (
            torch.cat(kept_pts, dim=0),
            torch.cat(kept_dist, dim=0),
            torch.cat(kept_sdf_all, dim=0),
            torch.cat(kept_sdf_fork, dim=0),
        )

    def preprocess_obstacles(self, event):
        if self.current_q is None:
            return

        if self.profile_sync:
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        pts = self.obs_points
        if pts is None or pts.shape[0] == 0:
            self._clear_or_hold_obstacles()
            return

        try:
            with torch.no_grad():
                T_fork_center = self.get_fork_tip_pose(self.current_q)
                if T_fork_center is None:
                    rospy.logwarn_throttle(
                        5, "CBF preprocessing cannot find fork_tip in FK tree.")
                    return

                x_now_pos = T_fork_center[:, :3, 3]
                self.debug_filter_center = x_now_pos[0].detach()
                self.debug_fork_mesh_world = torch.empty((0, 3), dtype=torch.float32,
                                                         device=self.device)
                self.debug_score_seed = None
                self.debug_tip_min = None

                sdf_all_candidates = None
                sdf_fork_candidates = None
                q9 = None

                if self.cbf_candidate_filter == "sdf":
                    q9 = torch.cat([self.current_q.detach(), self.q_pad2], dim=1)
                    pts_inside, _, sdf_all_candidates, sdf_fork_candidates = self._whole_body_sdf_candidates(
                        pts, q9)
                    dist_inside = torch.norm(pts_inside - x_now_pos, dim=1)
                else:
                    dist_tcp = torch.norm(pts - x_now_pos, dim=1)
                    inside_mask = dist_tcp < self.tcp_filter_radius
                    pts_inside = pts[inside_mask]
                    dist_inside = dist_tcp[inside_mask]

                num_inside = int(pts_inside.shape[0])

                if num_inside == 0:
                    self._clear_or_hold_obstacles()
                    return

                if self.cbf_selection_metric in ("sdf", "fork_sdf", "fork_mesh") or self.cbf_candidate_filter == "sdf":
                    if self.preprocess_max_points > 0 and num_inside > self.preprocess_max_points:
                        preselect_scores = (
                            sdf_all_candidates
                            if sdf_all_candidates is not None
                            else dist_inside
                        )
                        _, pre_idx = torch.topk(
                            preselect_scores,
                            k=self.preprocess_max_points,
                            largest=False,
                        )
                        pts_inside = pts_inside[pre_idx]
                        dist_inside = dist_inside[pre_idx]
                        if sdf_all_candidates is not None:
                            sdf_all_candidates = sdf_all_candidates[pre_idx]
                            sdf_fork_candidates = sdf_fork_candidates[pre_idx]
                        num_inside = int(pts_inside.shape[0])

                if self.cbf_selection_metric == "fork_mesh":
                    T_fork_now = self.get_fork_tip_pose(self.current_q)
                    if T_fork_now is None or self.fork_mesh_points_local.shape[0] == 0:
                        scores = dist_inside
                    else:
                        R_fork = T_fork_now[0, :3, :3]
                        t_fork = T_fork_now[0, :3, 3]
                        fork_mesh_world = (
                            self.fork_mesh_points_local @ R_fork.transpose(0, 1)
                        ) + t_fork
                        self.debug_fork_mesh_world = fork_mesh_world.detach()
                        mesh_dist = torch.cdist(
                            pts_inside.unsqueeze(0),
                            fork_mesh_world.unsqueeze(0),
                        ).squeeze(0).min(dim=1).values
                        scores = mesh_dist
                    largest = False
                elif self.cbf_selection_metric in ("sdf", "fork_sdf"):
                    if sdf_all_candidates is not None:
                        scores = (
                            sdf_fork_candidates
                            if self.cbf_selection_metric == "fork_sdf"
                            else sdf_all_candidates
                        )
                    else:
                        if q9 is None:
                            q9 = torch.cat([self.current_q.detach(), self.q_pad2], dim=1)
                        _, sdf_per_link = self.bernstein_core.get_whole_body_sdf_batch(
                            pts_inside, self.eye4, q9, return_per_link=True)

                        # Pruned self-filter: indices 0:5 now correspond to [link4, link5, link6, link7, hand]
                        sdf_body = sdf_per_link[0, :5, :].min(dim=0).values
                        sdf_all = sdf_per_link[0].min(dim=0).values
                        sdf_fork = sdf_per_link[0, self.fork_link_index, :]
                        sdf_self_filter = (
                            sdf_all if self.cbf_sdf_self_filter_all_links else sdf_body)
                        not_self = sdf_self_filter > self.cbf_sdf_self_filter_margin
                        pts_inside = pts_inside[not_self]
                        dist_inside = dist_inside[not_self]
                        if self.cbf_selection_metric == "fork_sdf":
                            scores = sdf_fork[not_self]
                        else:
                            scores = sdf_all[not_self]
                    largest = False
                    num_inside = int(pts_inside.shape[0])
                else:
                    scores = dist_inside
                    largest = False

                if num_inside == 0:
                    self._clear_or_hold_obstacles()
                    return

                score_min_idx = torch.argmin(scores)
                tip_min_idx = torch.argmin(dist_inside)
                self.debug_score_seed = pts_inside[score_min_idx].detach()
                self.debug_tip_min = pts_inside[tip_min_idx].detach()

                if num_inside > self.cbf_graph_points:
                    if self.cbf_cluster_mode == "topk":
                        # Diagnostic/global mode: protect the k lowest-score
                        # points even if they are split across the bowl/scene.
                        _, cluster_idx = torch.topk(
                            scores, k=self.cbf_graph_points, largest=largest)
                    else:
                        # Default/local mode: keep one Euclidean cluster around
                        # the single most dangerous point.
                        closest_pt = pts_inside[score_min_idx]
                        dist_to_closest = torch.norm(pts_inside - closest_pt, dim=-1)
                        _, cluster_idx = torch.topk(
                            dist_to_closest, k=self.cbf_graph_points, largest=False)
                    obs = pts_inside[cluster_idx]

                    if self.publish_yellow_points:
                        top_mask = torch.zeros(num_inside, dtype=torch.bool, device=self.device)
                        top_mask[cluster_idx] = True
                        pts_yellow = pts_inside[~top_mask]
                    else:
                        pts_yellow = torch.empty((0, 3), dtype=torch.float32, device=self.device)
                else:
                    obs = pts_inside
                    pts_yellow = torch.empty((0, 3), dtype=torch.float32, device=self.device)

                dist_tip = torch.norm(pts_inside - x_now_pos, dim=1)
                self._copy_fixed_obstacles(obs.contiguous())
                self.selected_pts_yellow = pts_yellow.detach()
                self.selected_num_inside = num_inside
                self.selected_min_obs_dist = float(dist_tip.min().item())

                if self.profile_sync:
                    torch.cuda.synchronize()
                self.preprocess_times.append(time.perf_counter() - t_start)
                if len(self.preprocess_times) >= 20:
                    avg_t = sum(self.preprocess_times) / len(self.preprocess_times)
                    rospy.loginfo_throttle(
                        5.0,
                        "⏱️ [TIMING] CBF red points: %.3f ms | "
                        "selection=%s cluster=%s candidates=%d selected=%d graph_points=%d",
                        avg_t * 1000.0,
                        self.cbf_selection_metric,
                        self.cbf_cluster_mode,
                        self.selected_num_inside,
                        self.selected_count,
                        self.cbf_graph_points,
                    )
                    self.preprocess_times.clear()
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error in CBF obstacle preprocessing: {e}")

    def publish_visualization(self, event):
        if self.current_q is None or self.last_q_safe is None:
            return

        try:
            pts_yellow = self.selected_pts_yellow
            obs = self.selected_obs[:self.selected_count]

            msg_yellow = self.create_cloud_xyzrgb(pts_yellow, (255, 255, 0))
            msg_red = self.create_cloud_xyzrgb(obs, (255, 0, 0))
            msg_mesh = self.create_cloud_xyzrgb(self.debug_fork_mesh_world, (160, 80, 255))

            if msg_yellow is not None:
                self.pub_inside_yellow.publish(msg_yellow)
            else:
                msg_yellow = PointCloud2()
                msg_yellow.header.stamp = rospy.Time.now()
                msg_yellow.header.frame_id = "world"
                self.pub_inside_yellow.publish(msg_yellow)

            if msg_red is not None:
                self.pub_top100_red.publish(msg_red)

            if msg_mesh is not None:
                self.pub_fork_mesh_debug.publish(msg_mesh)

            self.publish_selection_debug_markers()
            self.publish_safe_trajectory_marker(self.current_q, self.last_q_safe)
            self.publish_cbf_velocity_arrows()
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error publishing CBF visualization: {e}")

    def run(self):
        # print(f"🛡️  CBF NODE : Bouclier actif à {self.rate_hz}Hz. Sécurités activées.")
        dt_fixed = 1.0 / self.rate_hz

        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            dt = current_time - self.last_time
            self.last_time = current_time
            dt_measured = dt

            if dt <= 0.001: 
                self.rate.sleep()
                continue
                
            # Prevent massive jumps on initialization or lag spikes
            if dt > 3.0 * dt_fixed:
                dt = dt_fixed

            if self.current_q is not None and self.nominal_q is not None:
                with torch.no_grad():
                    current_q = self.current_q.detach()
                    nominal_q = self.nominal_q.detach()
                    direct_nominal_position = (
                        self.publish_controller_command
                        and self.cbf_direct_position_mode == "nominal_correction"
                    )
                    stateful_position_command = (
                        self.publish_controller_command
                        and self.cbf_stateful_position_command
                        and not direct_nominal_position
                    )
                    if self.last_q_safe is None:
                        self.last_q_safe = current_q.clone()
                    elif not stateful_position_command:
                        self.last_q_safe = (1.0 - self.cbf_feedback_coupling) * self.last_q_safe + self.cbf_feedback_coupling * current_q

                    # 1. Feedforward nominal velocity. Prefer the executor's
                    # retimed velocity topic (constant joint speed); fall back
                    # to position differencing only when the topic is stale.
                    velocity_age = float('inf')
                    if self.use_nominal_velocity_topic and self.nominal_dq_ff is not None:
                        velocity_age = current_time - self.nominal_dq_stamp
                    use_velocity_topic = (
                        self.use_nominal_velocity_topic
                        and self.nominal_dq_ff is not None
                        and (self.nominal_velocity_timeout <= 0.0
                             or velocity_age <= self.nominal_velocity_timeout)
                    )
                    if use_velocity_topic:
                        dq_nom_pure = self.nominal_feedforward_gain * self.nominal_dq_ff.detach()
                        diag_vel_source = 0.0
                    else:
                        last_nominal_q = self.last_nominal_q
                        if last_nominal_q is None:
                            last_nominal_q = nominal_q.clone()
                        dq_nom_pure = (nominal_q - last_nominal_q) / dt
                        diag_vel_source = 1.0
                    self.last_nominal_q = nominal_q.clone()
                    dq_fb = torch.zeros_like(dq_nom_pure)

                    if direct_nominal_position:
                        # In direct position-controller mode, the trajectory
                        # executor already gives us the timed position command.
                        # The controller output remains the timed nominal
                        # position while the CBF is inactive. The velocity fed
                        # to the QP still needs bounded pursuit, otherwise a
                        # robot held back by safety sees nearly zero nominal
                        # velocity once the timed trajectory has run ahead.
                        if self.cbf_direct_feedback_when_reactive:
                            dq_fb = self.cbf_kp * (nominal_q - current_q)
                            dq_fb = torch.clamp(
                                dq_fb,
                                min=-self.max_feedback_velocity,
                                max=self.max_feedback_velocity)
                            dq_fb = self._shape_tracking_feedback(dq_nom_pure, dq_fb)
                            dq_nom_raw = dq_nom_pure + dq_fb
                        else:
                            dq_nom_raw = dq_nom_pure
                    else:
                        # Bounded tracking error feedback for velocity/integrated
                        # command modes.
                        tracking_q = current_q if self.cbf_integrate_from_current else self.last_q_safe
                        dq_fb = self.cbf_kp * (nominal_q - tracking_q)
                        dq_fb = torch.clamp(dq_fb, min=-self.max_feedback_velocity, max=self.max_feedback_velocity)
                        dq_fb = self._shape_tracking_feedback(dq_nom_pure, dq_fb)
                        dq_nom_raw = dq_nom_pure + dq_fb

                    # 4. Low-pass filter to eliminate derivative jerk from piecewise-linear interpolation
                    if self.dq_nom_filtered is None:
                        self.dq_nom_filtered = dq_nom_raw.clone()
                    gamma = dt / (dt + self.cbf_filter_tau)
                    self.dq_nom_filtered = (1.0 - gamma) * self.dq_nom_filtered + gamma * dq_nom_raw

                    # 5. Final nominal velocity command fed to the QP solver
                    dq_nom_torch = torch.clamp(self.dq_nom_filtered, min=-self.max_joint_velocity, max=self.max_joint_velocity)

                    nominal_is_hold = torch.norm(dq_nom_torch) < self.nominal_hold_deadband
                    if nominal_is_hold:
                        dq_nom_torch = torch.zeros_like(dq_nom_torch)
                        if self.dq_nom_filtered is not None:
                            self.dq_nom_filtered.zero_()

                if self.profile_sync:
                    torch.cuda.synchronize()
                t_start = time.perf_counter()

                with torch.no_grad():
                    self.transfer_buffer[39:].zero_()
                    if self.enable_cbf:
                        local_obs = self.selected_obs # Read reference atomically
                        self.static_q.copy_(current_q)
                        self.static_obs.copy_(local_obs)
                        self.static_dq_nom.copy_(dq_nom_torch)
                        if self.cbf_solver_mode == "fast_tangent":
                            self.graph.replay()
                            self.dq_safe_work.copy_(self.static_dq_safe.detach())
                        else:
                            self.dq_safe_work.copy_(self.solve_multicbf_projection(
                                current_q, local_obs, dq_nom_torch).detach())
                    else:
                        self.dq_safe_work.copy_(dq_nom_torch)

                    solver_out = self.dq_safe_work.detach().clone()
                    if self.cbf_monitor_only:
                        # Baseline mode: keep h/diagnostics (solver_out records
                        # the counterfactual correction) but apply nothing.
                        self.dq_safe_work.copy_(dq_nom_torch)

                    self._viz_q = current_q.detach().clone()
                    self._viz_dq_nom = dq_nom_torch.detach().clone()
                    self._viz_dq_safe = self.dq_safe_work.detach().clone()

                    dot = torch.sum(dq_nom_torch * self.dq_safe_work, dim=1)
                    denom = torch.norm(dq_nom_torch, dim=1) * torch.norm(self.dq_safe_work, dim=1)
                    alignment = torch.where(denom > 1e-8, dot / denom, torch.ones_like(dot))

                    self.dq_safe_work.clamp_(min=-self.max_joint_velocity,
                                             max=self.max_joint_velocity)

                    velocity_filter_active = False
                    if self.cbf_velocity_filter_tau > 0.0:
                        if self._dq_safe_filtered is None:
                            self._dq_safe_filtered = self.dq_safe_work.detach().clone()
                        else:
                            gamma_v = dt / (dt + self.cbf_velocity_filter_tau)
                            self._dq_safe_filtered.mul_(1.0 - gamma_v).add_(
                                self.dq_safe_work, alpha=gamma_v)
                        self.dq_safe_work.copy_(self._dq_safe_filtered)
                        velocity_filter_active = True

                    if self.enable_cbf and self.cbf_enforce_final_constraint and not self.cbf_monitor_only:
                        # Component-wise velocity limits can invalidate the QP
                        # half-space projection. If that happens, project the
                        # already-filtered command again so tangential/nominal
                        # motion is preserved instead of braking to zero.
                        grad_h = self.static_grad_h.detach()
                        grad_norm_sq = (grad_h ** 2).sum(dim=1, keepdim=True)
                        kappa_eff = torch.where(
                            self.static_h.detach() < 0.0,
                            torch.full_like(self.static_h.detach(), self.cbf_recovery_kappa),
                            torch.full_like(self.static_h.detach(), self.cbf_kappa),
                        )
                        final_constr = (
                            (grad_h * self.dq_safe_work).sum(dim=1)
                            + kappa_eff * self.static_h.detach()
                            - self.cbf_constraint_margin
                        )
                        final_active = self.static_h.detach() <= self.cbf_h_activate
                        needs_repair = (
                            final_active
                            & (final_constr < 0.0)
                            & (grad_norm_sq.squeeze(1) > 1e-8)
                        ).view(-1, 1)
                        repair_step = torch.where(
                            final_active,
                            (-final_constr).clamp(min=0.0),
                            torch.zeros_like(final_constr),
                        ).view(-1, 1)
                        dq_repaired = (
                            self.dq_safe_work
                            + repair_step * grad_h / (grad_norm_sq + 1e-6)
                        )
                        dq_repaired.clamp_(min=-self.max_joint_velocity, max=self.max_joint_velocity)
                        self.dq_safe_work.copy_(torch.where(needs_repair, dq_repaired, self.dq_safe_work))
                        final_constr = (
                            (grad_h * self.dq_safe_work).sum(dim=1)
                            + kappa_eff * self.static_h.detach()
                            - self.cbf_constraint_margin
                        )
                        self.static_constr.copy_(torch.where(
                            final_active,
                            final_constr,
                            torch.ones_like(final_constr),
                        ))
                        self.transfer_buffer[60].copy_(needs_repair.float().view(-1)[0])
                        self.transfer_buffer[61].copy_(final_constr.view(-1)[0])

                    t_cbf_done = time.perf_counter()
                    cbf_delta_vel_for_command = self.dq_safe_work - dq_nom_torch
                    cbf_modified_mask = (
                        torch.norm(cbf_delta_vel_for_command)
                        > self.nominal_hold_deadband
                    )
                    if self.enable_cbf:
                        cbf_active_mask_for_command = (
                            self.static_h.detach() <= self.cbf_h_activate
                        )
                    else:
                        cbf_active_mask_for_command = torch.zeros(
                            (1,), device=self.device, dtype=torch.bool)

                    if self.enable_cbf:
                        h_now = self.static_h.detach()
                        grad_h_dbg = self.static_grad_h.detach()
                        self._maybe_check_gradient_direction(
                            current_q, local_obs, h_now, grad_h_dbg)
                        kappa_dbg = torch.where(
                            h_now < 0.0,
                            torch.full_like(h_now, self.cbf_recovery_kappa),
                            torch.full_like(h_now, self.cbf_kappa),
                        )
                        self.transfer_buffer[62].copy_((
                            (grad_h_dbg * dq_nom_torch).sum(dim=1)
                            + kappa_dbg * h_now
                            - self.cbf_constraint_margin
                        ).view(-1)[0])
                        self.transfer_buffer[63].copy_(torch.norm(grad_h_dbg))

                        dq_real = self.dq_real_measured.detach()

                        if self.last_h_debug is None:
                            h_rate = torch.zeros_like(h_now)
                        else:
                            h_rate = (h_now - self.last_h_debug) / max(dt_measured, 1e-6)
                        if self.last_h_debug is None:
                            self.last_h_debug = h_now.clone()
                        else:
                            self.last_h_debug.copy_(h_now)

                        cmd_hdot = (grad_h_dbg * self.dq_safe_work).sum(dim=1)
                        real_hdot = (grad_h_dbg * dq_real).sum(dim=1)
                        real_constr = (
                            real_hdot
                            + kappa_dbg * h_now
                            - self.cbf_constraint_margin
                        )
                        norm_dq_real = torch.norm(dq_real)
                    else:
                        cmd_hdot = torch.zeros((1,), device=self.device)
                        real_hdot = torch.zeros((1,), device=self.device)
                        real_constr = torch.zeros((1,), device=self.device)
                        norm_dq_real = torch.zeros((), device=self.device)
                        h_rate = torch.zeros((1,), device=self.device)

                    direct_nominal_position = (
                        self.publish_controller_command
                        and self.cbf_direct_position_mode == "nominal_correction"
                    )
                    stateful_position_command = (
                        self.publish_controller_command
                        and self.cbf_stateful_position_command
                        and not direct_nominal_position
                    )
                    if direct_nominal_position:
                        # Publish the timed nominal position exactly when the
                        # QP does not need to change it. If the CBF changes the
                        # velocity, apply only that correction as a small
                        # position offset. This preserves trajectory timing and
                        # makes CBF a local modifier instead of a full follower.
                        correction_dt = (
                            self.cbf_position_correction_dt
                            if self.cbf_position_correction_dt > 0.0
                            else dt
                        )
                        cbf_delta_vel = self.dq_safe_work - dq_nom_torch
                        cbf_delta_speed = torch.norm(cbf_delta_vel)
                        use_reactive_step = (
                            self.cbf_use_reactive_position_step
                            and self.enable_cbf
                            and bool(cbf_delta_speed > self.nominal_hold_deadband)
                        )
                        if use_reactive_step:
                            # When the QP is active, command a local safe step
                            # from the measured state. Adding a tiny correction
                            # to an already unsafe nominal setpoint can still
                            # let the position controller drive through the
                            # obstacle.
                            reactive_dt = (
                                self.cbf_reactive_position_dt
                                if self.cbf_reactive_position_dt > 0.0
                                else correction_dt
                            )
                            torch.add(current_q, self.dq_safe_work,
                                      alpha=reactive_dt, out=self.q_safe_work)
                            if self.cbf_max_command_lead > 0.0:
                                q_lead = self.q_safe_work - current_q
                                q_lead.clamp_(min=-self.cbf_max_command_lead,
                                              max=self.cbf_max_command_lead)
                                self.q_safe_work.copy_(current_q + q_lead)
                            if self.last_cbf_position_delta is not None:
                                self.last_cbf_position_delta.zero_()
                        else:
                            cbf_delta = cbf_delta_vel * correction_dt
                            if self.cbf_max_position_correction > 0.0:
                                delta_norm = torch.norm(cbf_delta)
                                if bool(delta_norm > self.cbf_max_position_correction):
                                    cbf_delta = cbf_delta * (
                                        self.cbf_max_position_correction / (delta_norm + 1e-6)
                                    )
                            if self.last_cbf_position_delta is None:
                                self.last_cbf_position_delta = torch.zeros_like(cbf_delta)
                            if self.cbf_position_correction_filter_tau > 0.0:
                                gamma_delta = dt / (dt + self.cbf_position_correction_filter_tau)
                                self.last_cbf_position_delta = (
                                    (1.0 - gamma_delta) * self.last_cbf_position_delta
                                    + gamma_delta * cbf_delta
                                )
                                cbf_delta = self.last_cbf_position_delta
                            else:
                                self.last_cbf_position_delta = cbf_delta
                            torch.add(nominal_q, cbf_delta, out=self.q_safe_work)
                    elif stateful_position_command:
                        # Direct JointGroupPositionController mode: integrate
                        # the velocity command into a persistent position
                        # setpoint. Rebuilding q_safe from q_current every tick
                        # can collapse the requested velocity into tiny one-step
                        # targets that the position controller never realizes.
                        # The integration step must be the real loop dt: using
                        # cbf_command_dt here (a lead gain meant for the
                        # current-anchored mode) makes the reference advance at
                        # cbf_command_dt/dt times the certified velocity.
                        integration_base = self.last_q_safe
                        integration_dt = dt
                    else:
                        integration_base = (
                            current_q if self.cbf_integrate_from_current
                            else self.last_q_safe
                        )
                        integration_dt = (
                            self.cbf_command_dt
                            if self.cbf_integrate_from_current and self.cbf_command_dt > 0.0
                            else dt
                        )
                    if not direct_nominal_position:
                        torch.add(integration_base, self.dq_safe_work, alpha=integration_dt,
                                  out=self.q_safe_work)
                        if stateful_position_command and self.cbf_max_command_lead > 0.0:
                            q_lead = self.q_safe_work - current_q
                            q_lead.clamp_(min=-self.cbf_max_command_lead,
                                          max=self.cbf_max_command_lead)
                            self.q_safe_work.copy_(current_q + q_lead)
                        if self.cbf_passthrough_when_inactive:
                            # When the QP leaves the nominal velocity unchanged,
                            # publish the timed nominal waypoint itself. The
                            # integrated velocity target is only for local CBF
                            # corrections, otherwise it makes free-space motion
                            # feel delayed and weak.
                            passthrough_mask = (
                                (~cbf_modified_mask)
                                & (~cbf_active_mask_for_command)
                            ).view(1, 1)
                            self.q_safe_work.copy_(torch.where(
                                passthrough_mask,
                                nominal_q,
                                self.q_safe_work,
                            ))
                    self.last_q_safe.copy_(self.q_safe_work)
                    q_lead_norm = torch.norm(self.q_safe_work - current_q)

                    if self.cbf_integrate_from_current:
                        dq_pub = self.dq_safe_work
                    else:
                        dq_pub = (self.q_safe_work - current_q) / dt

                    if self._contact_stopped:
                        # Latched contact stop: hold the frozen configuration.
                        self.dq_safe_work.zero_()
                        dq_pub = self.dq_safe_work
                        self.q_safe_work.copy_(self._contact_q)
                        self.last_q_safe.copy_(self._contact_q)

                    # Fill CUDA transfer buffer
                    self.transfer_buffer[0:7].copy_(dq_pub.squeeze(0))
                    self.transfer_buffer[7:14].copy_(self.q_safe_work.squeeze(0))
                    if self.enable_cbf:
                        self.transfer_buffer[14].copy_(self.static_h.squeeze(0))
                        self.transfer_buffer[15].copy_(self.static_constr.squeeze(0))
                    else:
                        self.transfer_buffer[14] = 1.0
                        self.transfer_buffer[15] = 0.0
                    self.transfer_buffer[16].copy_(alignment.squeeze(0))
                    self.transfer_buffer[17].copy_(torch.norm(dq_nom_torch))
                    self.transfer_buffer[18].copy_(torch.norm(self.dq_safe_work))
                    self.transfer_buffer[19:26].copy_(nominal_q.squeeze(0))
                    self.transfer_buffer[26:33].copy_(dq_nom_torch.squeeze(0))
                    self.transfer_buffer[33].copy_(cmd_hdot.squeeze(0))
                    self.transfer_buffer[34].copy_(real_hdot.squeeze(0))
                    self.transfer_buffer[35].copy_(real_constr.squeeze(0))
                    self.transfer_buffer[36].copy_(norm_dq_real)
                    self.transfer_buffer[37].copy_(h_rate.squeeze(0))
                    self.transfer_buffer[38].copy_(q_lead_norm)
                    self.transfer_buffer[39:46].copy_(dq_nom_pure.squeeze(0))
                    self.transfer_buffer[46:53].copy_(dq_fb.squeeze(0))
                    self.transfer_buffer[53:60].copy_(solver_out.squeeze(0))

                t_q_cmd_done = time.perf_counter()
                # Single sync transfer to CPU
                cpu_buffer = self.transfer_buffer.cpu().numpy()
                dq_pub_cpu = cpu_buffer[0:7].tolist()
                q_safe_work_cpu = cpu_buffer[7:14].tolist()
                h_val = float(cpu_buffer[14])
                self.last_h_value_for_feedback = h_val
                clearance_val = h_val + self.d_safe
                constr_val = float(cpu_buffer[15])
                alignment_val = float(cpu_buffer[16])
                norm_dq_nom_val = float(cpu_buffer[17])
                norm_dq_safe_val = float(cpu_buffer[18])
                nominal_q_cpu = cpu_buffer[19:26].tolist()
                dq_nom_torch_cpu = cpu_buffer[26:33].tolist()
                cmd_hdot_val = float(cpu_buffer[33])
                real_hdot_val = float(cpu_buffer[34])
                real_constr_val = float(cpu_buffer[35])
                norm_dq_real_val = float(cpu_buffer[36])
                h_rate_val = float(cpu_buffer[37])
                q_lead_val = float(cpu_buffer[38])
                cbf_delta_speed_val = float(
                    np.linalg.norm(
                        np.array(dq_pub_cpu, dtype=np.float32)
                        - np.array(dq_nom_torch_cpu, dtype=np.float32)
                    )
                )

                if (
                    self.contact_stop_enabled
                    and self.enable_cbf
                    and not self._contact_stopped
                ):
                    if self.cbf_solver_mode == "fast_tangent":
                        h_bias = float(self.barrier.alpha) * float(
                            np.log(max(1, int(self.selected_count))))
                    else:
                        h_bias = 0.0
                    clearance_est = h_val + h_bias + self.d_safe
                    if (
                        int(self.selected_count) > 0
                        and clearance_est <= self.contact_stop_clearance
                    ):
                        self._contact_counter += 1
                    else:
                        self._contact_counter = 0
                    if self._contact_counter >= self.contact_stop_cycles:
                        self._contact_stopped = True
                        self._contact_q = current_q.detach().clone()
                        self.contact_event_pub.publish(Float32MultiArray(
                            data=[rospy.get_time(), h_val, clearance_est]))
                        rospy.logerr(
                            "CONTACT STOP: estimated clearance %.4f m <= %.4f m "
                            "(h=%.4f). Robot frozen at current configuration.",
                            clearance_est, self.contact_stop_clearance, h_val)

                if self.enable_cbf and self.log_cbf_events:
                    if h_val < 0:
                        rospy.logwarn_throttle(0.5, f"💥 CBF COLLISION: Margin breached! (h = {h_val:.4f} < 0)")
                    elif constr_val < 0:
                        rospy.loginfo_throttle(0.5, f"🛡️ CBF REACTIVE: Correcting trajectory (h = {h_val:.4f})")

                self.cmd_pub.publish(Float32MultiArray(data=dq_pub_cpu))

                pos_msg = Float64MultiArray(data=q_safe_work_cpu)
                if self.log_cbf_events and not self._logged_first_safe_publish:
                    # Allow this one-time logging to compute on CPU array
                    safe_err = float(np.linalg.norm(np.array(q_safe_work_cpu) - current_q.squeeze(0).cpu().numpy()))
                    nominal_err = float(np.linalg.norm(np.array(nominal_q_cpu) - current_q.squeeze(0).cpu().numpy()))
                    rospy.loginfo(
                        "CBF publishing first safe joint command | "
                        "safe_err_to_current=%.4f rad nominal_err_to_current=%.4f rad",
                        safe_err, nominal_err)
                    self._logged_first_safe_publish = True
                if (
                    self.log_trajectory_timing
                    and self.active_plan_stamp is not None
                    and not self._cbf_first_safe_for_plan
                ):
                    safe_err = float(np.linalg.norm(
                        np.array(q_safe_work_cpu)
                        - current_q.squeeze(0).cpu().numpy()
                    ))
                    now = rospy.Time.now()
                    rospy.loginfo(
                        "[TRAJ TIMING] cbf_first_safe id=%d "
                        "publish_to_cbf_safe=%.3fs expected_duration=%.3fs "
                        "safe_err=%.4frad q_lead=%.4frad dq_nom=%.3frad/s "
                        "dq_safe=%.3frad/s align=%.2f",
                        self.active_plan_id,
                        (now - self.active_plan_stamp).to_sec(),
                        self.active_plan_duration,
                        safe_err,
                        q_lead_val,
                        norm_dq_nom_val,
                        norm_dq_safe_val,
                        alignment_val,
                    )
                    self._cbf_first_safe_for_plan = True
                self.safe_cmd_pub.publish(pos_msg)
                t_pub_done = time.perf_counter()

                if self.publish_diagnostics:
                    nan = float('nan')
                    self.h_pub.publish(Float32MultiArray(data=[h_val]))
                    if self.enable_cbf and self.cbf_solver_mode == "fast_tangent":
                        h_corr_val = h_val + float(self.barrier.alpha) * float(
                            np.log(max(1, int(self.selected_count))))
                    else:
                        # multi_projected uses a hard min: no soft-min bias.
                        h_corr_val = h_val
                    solver_out_v = cpu_buffer[53:60]
                    dq_base_v = cpu_buffer[26:33]
                    repair_on = self.enable_cbf and self.cbf_enforce_final_constraint
                    min_obs_d = float(self.selected_min_obs_dist)
                    scalars = [
                        h_val if self.enable_cbf else nan,
                        h_corr_val if self.enable_cbf else nan,
                        nan,  # lam
                        nan,  # cap_active
                        nan,  # recovery_used
                        nan,  # grad_degenerate
                        float(cpu_buffer[62]) if self.enable_cbf else nan,
                        float(cpu_buffer[60]) if repair_on else nan,
                        float(cpu_buffer[61]) if repair_on else nan,
                        float(cpu_buffer[63]) if self.enable_cbf else nan,
                        0.0,  # escape_active
                        0.0,  # ema_applied
                        1.0 if velocity_filter_active else 0.0,
                        diag_vel_source,
                        float(velocity_age) if np.isfinite(velocity_age) else nan,
                        float(dt_measured),
                        float(self.selected_count),
                        min_obs_d if np.isfinite(min_obs_d) else nan,
                        (t_cbf_done - t_start) * 1000.0,
                        (t_q_cmd_done - t_cbf_done) * 1000.0,
                        (t_pub_done - t_start) * 1000.0,
                        float(cpu_buffer[33]),  # cmd_hdot
                        float(cpu_buffer[34]),  # real_hdot
                        float(cpu_buffer[35]),  # real_constr
                        float(cpu_buffer[37]),  # h_rate
                        float(cpu_buffer[16]),  # alignment
                        float(cpu_buffer[38]),  # q_lead
                        float(cpu_buffer[36]),  # dq_real_norm
                        float(self._last_active_constraints
                              if self.cbf_solver_mode == "multi_projected" else 1.0),
                        0.0 if self.cbf_solver_mode == "fast_tangent" else 1.0,
                        1.0 if self.cbf_monitor_only else 0.0,
                        1.0 if self._contact_stopped else 0.0,
                    ]
                    self.diag_pub.publish(Float32MultiArray(data=(
                        cpu_buffer[39:46].tolist()                 # dq_ff
                        + cpu_buffer[46:53].tolist()               # dq_fb
                        + dq_base_v.tolist()                       # dq_base
                        + [0.0] * 7                                # dq_escape
                        + (solver_out_v - dq_base_v).tolist()      # dq_cbf_delta
                        + solver_out_v.tolist()                    # dq_pre_filter
                        + cpu_buffer[0:7].tolist()                 # dq_final
                        + scalars)))

                if self.profile_sync:
                    torch.cuda.synchronize()
                if self.log_cbf_command_timing:
                    self.comp_times.append(time.perf_counter() - t_start)
                    self._log_cuda_memory_throttle("CBF runtime")
                    if len(self.comp_times) >= 20:
                        avg_t = sum(self.comp_times) / len(self.comp_times)
                        rospy.loginfo_throttle(
                            5.0,
                            f"⏱️ [TIMING] CBF command: {avg_t*1000:.3f} ms | "
                            f"candidates={self.selected_num_inside} selected={self.selected_count} "
                            f"h={h_val:.4f} clearance={clearance_val:.4f} constr={constr_val:.4f} "
                            f"dq_nom={norm_dq_nom_val:.3f} dq_safe={norm_dq_safe_val:.3f} "
                            f"dq_delta={cbf_delta_speed_val:.3f} "
                            f"dq_real={norm_dq_real_val:.3f} q_lead={q_lead_val:.3f} "
                            f"align={alignment_val:.2f} "
                            f"hdot_cmd={cmd_hdot_val:.4f} hdot_real={real_hdot_val:.4f} "
                            f"h_rate={h_rate_val:.4f} real_constr={real_constr_val:.4f}")
                        self.comp_times.clear()

                if self.publish_debug_topics:
                    self.debug_input_pub.publish(Float64MultiArray(data=nominal_q_cpu))
                    self.debug_input_vel_pub.publish(Float32MultiArray(data=dq_nom_torch_cpu))
                    self.debug_output_pub.publish(pos_msg)
                    self.debug_output_vel_pub.publish(Float32MultiArray(data=dq_pub_cpu))
                    self.debug_alignment_pub.publish(Float32MultiArray(data=[
                        alignment_val,
                        norm_dq_nom_val,
                        norm_dq_safe_val
                    ]))

                if self.pos_cmd_pub is not None:
                    self.pos_cmd_pub.publish(pos_msg)
                
            self.rate.sleep()

if __name__ == '__main__':
    node = CBFSafetyNode()
    node.run()
