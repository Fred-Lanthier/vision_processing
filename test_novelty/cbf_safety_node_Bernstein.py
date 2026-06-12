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

try:
    from skimage.measure import marching_cubes
except ImportError:
    marching_cubes = None

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


# Layout of /cbf_safety/diagnostics (Float32MultiArray). Joint-space vectors
# come first (7 values each, in DIAG_VEC_FIELDS order), then the scalars in
# DIAG_SCALAR_FIELDS order. The full layout is also published as JSON on the
# latched topic /cbf_safety/diagnostics_layout so analysis scripts can decode
# bags without importing this file.
DIAG_VEC_FIELDS = [
    "dq_ff",          # feedforward from the nominal trajectory
    "dq_fb",          # saturated proportional tracking feedback
    "dq_base",        # filtered nominal = ff + fb after low-pass/deadband
    "dq_escape",      # escape velocity (tangent + normal), zero when inactive
    "dq_cbf_delta",   # projection output minus the velocity fed to the QP
    "dq_pre_filter",  # safe velocity before the output low-pass filter
    "dq_final",       # velocity actually integrated into the command
]
DIAG_SCALAR_FIELDS = [
    "h",                # soft-min barrier value (uncorrected)
    "h_corr",           # h + alpha*log(N) soft-min bias compensation
    "lam",              # QP multiplier (after cap)
    "cap_active",       # 1 if the safe-region lambda cap bound this cycle
    "recovery_used",    # 1 if hard recovery replaced the projection
    "grad_degenerate",  # 1 if |grad_h|^2 < 1e-6 while constraint violated
    "constr_pre",       # CBF constraint value of the velocity fed to the QP
    "repair_applied",   # 1 if post-filter repair modified the command
    "constr_final",     # constraint value after repair (NaN if repair off)
    "grad_h_norm",      # |grad_h|
    "escape_active",    # 1 while the escape hysteresis is engaged
    "ema_applied",      # 1 if boundary-band EMA smoothed dq_safe
    "velocity_filter_active",  # 1 if the output low-pass filter ran
    "vel_source",       # nominal velocity source: 0=topic 1=diff 2=zero
    "velocity_age",     # age of the nominal velocity topic sample (s)
    "dt",               # control loop dt (s)
    "selected_count",   # number of obstacle points fed to the barrier
    "min_obs_dist",     # min obstacle distance from the selection stage
    "cbf_ms",           # barrier + projection (+escape replay) time
    "q_cmd_ms",         # command construction time
    "total_ms",         # loop start to safe-command publish
]
DIAG_VEC_SOURCE_CODES = {"topic": 0.0, "diff": 1.0, "zero": 2.0}


class CBFSafetyNode:
    def __init__(self):
        rospy.init_node('cbf_safety_node')
        self.device = torch.device('cuda')

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
        barrier_alpha = float(rospy.get_param("~barrier_alpha", 0.001))
        self.barrier = BernsteinBarrier(self.bernstein_core, d_safe=self.d_safe, alpha=barrier_alpha)
        self._log_cuda_memory("CBF after Bernstein barrier")

        # kappa: class-K coefficient in the CBF constraint ∇h·dq + κh ≥ 0.
        # High kappa forces hard corrections even when h is slightly positive (safe).
        # With a jumpy gradient, high kappa amplifies oscillation near obstacles.
        # Recommended: 1.0–3.0. Must be read BEFORE setup_cuda_graph (baked into graph).
        self.cbf_kappa = float(rospy.get_param("~cbf_kappa", 2.0))
        self.cbf_recovery_kappa = float(rospy.get_param("~cbf_recovery_kappa", 3.0))
        self.cbf_constraint_margin = float(rospy.get_param("~cbf_constraint_margin", 0.0))
        self.cbf_enforce_final_constraint = _bool_param("~cbf_enforce_final_constraint", True)
        self.cbf_direct_position_mode = str(rospy.get_param(
            "~cbf_direct_position_mode", "nominal_correction")).lower()
        if self.cbf_direct_position_mode not in ("integrated", "nominal_correction"):
            raise ValueError("~cbf_direct_position_mode must be 'integrated' or 'nominal_correction'")
        self.cbf_position_correction_dt = float(rospy.get_param(
            "~cbf_position_correction_dt", 0.04))
        self.cbf_max_command_lead = float(rospy.get_param("~cbf_max_command_lead", 0.0))
        # setup_cuda_graph captures this attribute verbatim. Keeping it at
        # infinity disables the old hard recovery override without exposing the
        # removed cbf_recovery_switch_margin ROS parameter.
        self.cbf_recovery_switch_margin = float('inf')
        self.cbf_gradient_check_period = float(rospy.get_param("~cbf_gradient_check_period", 1.0))
        self.cbf_gradient_check_eps = float(rospy.get_param("~cbf_gradient_check_eps", 1e-3))
        self.log_cbf_events = _bool_param("~log_cbf_events", False)
        self.log_cbf_command_timing = _bool_param("~log_cbf_command_timing", True)
        self._last_gradient_check = 0.0

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
        self._last_q_safe = None
        self.obs_points = torch.empty((0, 3), dtype=torch.float32, device=self.device)

        self.max_joint_velocity = float(rospy.get_param("~max_joint_velocity", 0.7))
        self.cbf_kp = float(rospy.get_param("~cbf_kp", 10.0))
        self.max_feedback_velocity = float(rospy.get_param("~max_feedback_velocity", 0.05))
        self.cbf_filter_tau = float(rospy.get_param("~cbf_filter_tau", 0.05))
        self.cbf_velocity_filter_tau = float(rospy.get_param(
            "~cbf_velocity_filter_tau", 0.08))
        self._dq_safe_filtered = None
        self.use_nominal_velocity_topic = _bool_param("~use_nominal_velocity_topic", True)
        self.nominal_velocity_topic = rospy.get_param(
            "~nominal_velocity_topic", "/planner/nominal_joint_velocity")
        self.nominal_velocity_timeout = float(rospy.get_param(
            "~nominal_velocity_timeout", 0.25))
        self.nominal_feedforward_gain = float(rospy.get_param(
            "~nominal_feedforward_gain", 1.0))
        self.cbf_escape_enabled = _bool_param("~cbf_escape_enabled", False)
        self.cbf_escape_h_trigger = float(rospy.get_param("~cbf_escape_h_trigger", 0.006))
        self.cbf_escape_h_release = float(rospy.get_param("~cbf_escape_h_release", 0.012))
        self.cbf_escape_use_bias_corrected_h = _bool_param(
            "~cbf_escape_use_bias_corrected_h", True)
        self.cbf_escape_activation_cycles = max(
            1, int(rospy.get_param("~cbf_escape_activation_cycles", 2)))
        self.cbf_escape_min_cbf_delta = float(rospy.get_param(
            "~cbf_escape_min_cbf_delta", 0.02))
        self.cbf_escape_tangent_gain = float(rospy.get_param("~cbf_escape_tangent_gain", 1.0))
        self.cbf_escape_normal_gain = float(rospy.get_param("~cbf_escape_normal_gain", 0.04))
        self.cbf_escape_normal_h_trigger = float(rospy.get_param(
            "~cbf_escape_normal_h_trigger", 0.0))
        self.cbf_escape_max_velocity = float(rospy.get_param("~cbf_escape_max_velocity", 0.08))
        self._cbf_escape_active_cycles = 0
        self.last_nominal_q = None
        self.nominal_dq_ff = None
        self.nominal_dq_stamp = 0.0
        self.dq_nom_filtered = None

        self.enable_cbf = _bool_param("~enable_cbf", True)
        self.publish_controller_command = _bool_param("~publish_controller_command", False)
        self.preprocess_rate_hz = float(rospy.get_param("~preprocess_rate_hz", 30.0))
        self.viz_rate_hz = float(rospy.get_param("~viz_rate_hz", 5.0))
        self.publish_debug_topics = _bool_param("~publish_debug_topics", False)
        self.publish_diagnostics = _bool_param("~publish_diagnostics", True)
        self.publish_viz_topics = _bool_param("~publish_viz_topics", True)
        self.publish_safety_envelope = _bool_param(
            "~publish_safety_envelope", True)
        self.safety_envelope_resolution = max(
            24, int(rospy.get_param("~safety_envelope_resolution", 40)))
        self.safety_envelope_alpha = float(rospy.get_param(
            "~safety_envelope_alpha", 0.20))
        self.profile_sync = _bool_param("~profile_sync", False)
        self.cuda_memory_log_period = float(rospy.get_param("~cuda_memory_log_period", 10.0))
        self._last_cuda_memory_log = 0.0
        self.tcp_filter_radius = float(rospy.get_param("~tcp_filter_radius", 0.40))
        self.fork_filter_radius = float(rospy.get_param("~fork_filter_radius", 0.15))
        self.nominal_hold_deadband = float(rospy.get_param("~nominal_hold_deadband", 1e-4))
        # EMA smoothing on dq_safe output to reduce gradient-jump jitter from
        # changing obstacle selections. alpha=new-sample weight; lower = smoother.
        self.cbf_ema_alpha = float(rospy.get_param("~cbf_ema_alpha", 0.15))
        self._dq_safe_ema = None
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
        self.cbf_command_dt = float(rospy.get_param("~cbf_command_dt", 0.10))

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

        # Kept for unchanged visualization helpers.
        self.last_q_safe = None
        self.q_safe_work = torch.zeros((1, 7), device=self.device)
        self.dq_safe_work = torch.zeros((1, 7), device=self.device)
        self.last_current_q_debug = None
        self._viz_q = None
        self._viz_dq_nom = None
        self._viz_dq_safe = None
        self.last_h_debug = None
        self.transfer_buffer = torch.zeros(39, dtype=torch.float32, device=self.device)
        self._log_cuda_memory("CBF after runtime buffers")

        self.comp_times = []
        self.preprocess_times = []
        rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        rospy.Subscriber('/planner/nominal_joint_command', Float64MultiArray, self._nominal_cb)
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
        self.diag_pub = rospy.Publisher('/cbf_safety/diagnostics', Float32MultiArray, queue_size=10)
        self.diag_layout_pub = rospy.Publisher('/cbf_safety/diagnostics_layout', String, queue_size=1, latch=True)
        self.diag_layout_pub.publish(String(data=json.dumps({
            "vec_fields": DIAG_VEC_FIELDS,
            "vec_dim": 7,
            "scalar_fields": DIAG_SCALAR_FIELDS,
            "vel_source_codes": DIAG_VEC_SOURCE_CODES,
        })))
        self.debug_input_pub = rospy.Publisher('/debug/cbf/input_joint_command_seen', Float64MultiArray, queue_size=1)
        self.debug_output_pub = rospy.Publisher('/debug/cbf/output_joint_command', Float64MultiArray, queue_size=1)
        self.debug_input_vel_pub = rospy.Publisher('/debug/cbf/input_joint_velocity', Float32MultiArray, queue_size=1)
        self.debug_output_vel_pub = rospy.Publisher('/debug/cbf/output_joint_velocity', Float32MultiArray, queue_size=1)
        self.debug_alignment_pub = rospy.Publisher('/debug/cbf/nominal_safe_alignment', Float32MultiArray, queue_size=1)
        self.debug_timing_pub = rospy.Publisher('/debug/cbf/command_timing_ms', Float32MultiArray, queue_size=1)
        
        # Publishers RViz (Uniquement Jaune et Rouge)
        self.pub_inside_yellow = rospy.Publisher('/viz/obs_inside_yellow', PointCloud2, queue_size=1)
        self.pub_top100_red = rospy.Publisher('/viz/obs_top100_red', PointCloud2, queue_size=1)
        self.pub_fork_mesh_debug = rospy.Publisher('/viz/cbf_fork_mesh_points', PointCloud2, queue_size=1)
        self.pub_selection_debug = rospy.Publisher('/viz/cbf_selection_debug', MarkerArray, queue_size=1)
        self.safe_traj_viz_pub = rospy.Publisher('/viz/safe_trajectory_3d', MarkerArray, queue_size=1)
        self.pub_cbf_velocity_arrows = rospy.Publisher('/viz/cbf_velocity_arrows', MarkerArray, queue_size=1)
        self.pub_safety_envelope = rospy.Publisher(
            '/viz/robot_safety_envelope', MarkerArray, queue_size=1, latch=True)

        if self.publish_viz_topics and self.publish_safety_envelope:
            envelope_markers = self._build_safety_envelope_markers()
            if envelope_markers.markers:
                self.pub_safety_envelope.publish(envelope_markers)
        
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
            "Optimized CBF ready: control=%.1f Hz preprocess=%.1f Hz viz=%.1f Hz "
            "graph_points=%d preprocess_max_points=%d candidate_filter=%s selection=%s "
            "cluster=%s yellow=%s debug=%s profile_sync=%s",
            self.rate_hz, self.preprocess_rate_hz, self.viz_rate_hz,
            self.cbf_graph_points,
            self.preprocess_max_points,
            self.cbf_candidate_filter,
            self.cbf_selection_metric,
            self.cbf_cluster_mode,
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

    def _evaluate_link_sdf_local(self, link_index, points_local):
        """Evaluate one link with the same bounded Bernstein SDF as the CBF."""
        group = None
        group_index = None
        for candidate in self.bernstein_core.groups.values():
            if link_index in candidate["indices"]:
                group = candidate
                group_index = candidate["indices"].index(link_index)
                break
        if group is None:
            raise ValueError("No Bernstein group for protected link index %d" % link_index)

        offset = self.bernstein_core.offsets[link_index]
        scale = self.bernstein_core.scales[link_index]
        points_scaled = (points_local - offset) / scale
        points_bounded = torch.clamp(points_scaled, min=-0.99, max=0.99)
        residual = points_scaled - points_bounded

        phi = self.bernstein_core.build_basis_function_from_points(
            points_bounded,
            n_func=group["n_func"],
            comb=group["comb"],
            i_tensor=group["i_tensor"],
        )
        sdf = torch.matmul(phi, group["weights"][group_index])
        return (sdf + torch.norm(residual, dim=-1)) * scale

    def _find_visual_info(self, link_name):
        for info in self.robot_layer.meshes_info:
            if info['link_name'] == link_name:
                return info

        target = link_name.replace('panda_', '').replace('_w', '').replace('.pt', '')
        for info in self.robot_layer.meshes_info:
            candidate = info['link_name'].replace('panda_', '')
            if target in candidate or candidate in target:
                return info
        return None

    def _build_safety_envelope_markers(self):
        """Precompute d_robot=d_safe isosurfaces in each protected link frame."""
        marker_array = MarkerArray()
        if marching_cubes is None:
            rospy.logwarn(
                "scikit-image is unavailable; the robot safety envelope is disabled.")
            return marker_array

        resolution = self.safety_envelope_resolution
        level = self.d_safe
        started = time.perf_counter()

        try:
            with torch.no_grad():
                for link_index, link_name in enumerate(self.protected_link_names):
                    offset = self.bernstein_core.offsets[link_index]
                    scale = self.bernstein_core.scales[link_index]
                    half_extent = 0.99 * float(scale.item()) + level + 0.005
                    lower = offset - half_extent
                    upper = offset + half_extent

                    axes = [
                        torch.linspace(
                            float(lower[axis].item()),
                            float(upper[axis].item()),
                            resolution,
                            device=self.device,
                            dtype=torch.float32,
                        )
                        for axis in range(3)
                    ]
                    grid = torch.stack(
                        torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(-1, 3)

                    sdf_chunks = []
                    for points_chunk in torch.split(grid, 4096, dim=0):
                        sdf_chunks.append(
                            self._evaluate_link_sdf_local(
                                link_index, points_chunk).detach().cpu())
                    sdf_grid = torch.cat(sdf_chunks).reshape(
                        resolution, resolution, resolution).numpy()

                    sdf_min = float(sdf_grid.min())
                    sdf_max = float(sdf_grid.max())
                    if not (sdf_min <= level <= sdf_max):
                        rospy.logwarn(
                            "Safety envelope skipped for %s: level %.4f outside [%.4f, %.4f]",
                            link_name, level, sdf_min, sdf_max)
                        continue

                    spacing = tuple(
                        float((upper[axis] - lower[axis]).item()) / (resolution - 1)
                        for axis in range(3)
                    )
                    vertices, faces, _, _ = marching_cubes(
                        sdf_grid, level=level, spacing=spacing)
                    vertices += lower.detach().cpu().numpy()

                    visual_info = self._find_visual_info(link_name)
                    if visual_info is None:
                        rospy.logwarn(
                            "Safety envelope skipped for %s: visual transform not found",
                            link_name)
                        continue

                    visual_transform = visual_info['visual_offset'].detach().cpu().numpy()
                    vertices_h = np.concatenate([
                        vertices,
                        np.ones((vertices.shape[0], 1), dtype=vertices.dtype),
                    ], axis=1)
                    vertices_link = (vertices_h @ visual_transform.T)[:, :3]

                    marker = Marker()
                    marker.header.frame_id = link_name
                    marker.header.stamp = rospy.Time(0)
                    marker.ns = "robot_safety_envelope"
                    marker.id = link_index
                    marker.type = Marker.TRIANGLE_LIST
                    marker.action = Marker.ADD
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = 1.0
                    marker.scale.y = 1.0
                    marker.scale.z = 1.0
                    marker.color.r = 1.0
                    marker.color.g = 0.55
                    marker.color.b = 0.05
                    marker.color.a = max(0.0, min(1.0, self.safety_envelope_alpha))
                    marker.frame_locked = True

                    for vertex_index in faces.reshape(-1):
                        xyz = vertices_link[vertex_index]
                        marker.points.append(Point(
                            x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])))
                    marker_array.markers.append(marker)

                    rospy.loginfo(
                        "Safety envelope %s: d_robot=%.3fm, %d triangles",
                        link_name, level, int(faces.shape[0]))
        except Exception as exc:
            rospy.logerr("Could not build robot safety envelope: %s", exc)
            return MarkerArray()

        rospy.loginfo(
            "Robot safety envelope ready: d_robot=%.3fm, links=%d, %.1fms",
            level, len(marker_array.markers),
            1000.0 * (time.perf_counter() - started))
        return marker_array

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
        print(f"⚡ Capture du graphe CUDA (QP {n_points} points)...")
        torch.cuda.empty_cache()

        self.static_q      = torch.zeros((batch_size, 7), device=self.device, requires_grad=True)
        self.static_obs    = torch.zeros((n_points, 3),   device=self.device)
        self.static_dq_nom = torch.zeros((batch_size, 7), device=self.device)
        self.static_dq_safe = torch.zeros((batch_size, 7), device=self.device)
        self.static_h      = torch.zeros((batch_size,), device=self.device)
        self.static_constr = torch.zeros((batch_size,), device=self.device)
        self.static_grad_h = torch.zeros((batch_size, 7), device=self.device)
        # Diagnostic buffers: expose decisions taken inside the captured graph
        # (lambda cap, recovery mode, degenerate gradient) for logging.
        self.static_lam             = torch.zeros((batch_size,), device=self.device)
        self.static_cap_active      = torch.zeros((batch_size,), device=self.device)
        self.static_recovery_used   = torch.zeros((batch_size,), device=self.device)
        self.static_grad_degenerate = torch.zeros((batch_size,), device=self.device)

        # Capture scalars as Python floats so the CUDA graph bakes in the params.
        kappa = self.cbf_kappa
        recovery_kappa = self.cbf_recovery_kappa
        constraint_margin = self.cbf_constraint_margin
        recovery_switch_h = -max(0.0, self.cbf_recovery_switch_margin)

        def _qp_step(h, grad_h, dq_in):
            self.static_h.copy_(h)
            kappa_eff = torch.where(
                h < 0.0,
                torch.full_like(h, recovery_kappa),
                torch.full_like(h, kappa),
            )
            constr = (grad_h * dq_in).sum(dim=-1) + kappa_eff * h - constraint_margin
            self.static_constr.copy_(constr)
            denom  = (grad_h ** 2).sum(dim=-1)
            lam_uncapped = -constr / (denom + 1e-4)
            # In the safe region (h >= 0) cap the correction to nom_speed to
            # prevent a massive normal push from overwhelming tangential exit
            # motion in bowl-shaped cavities.  In recovery (h < 0) the robot is
            # already in violation — allow the full uncapped correction so the
            # constraint is actually satisfied rather than silently truncated.
            nom_speed = torch.norm(dq_in, dim=-1)
            lam_max_safe = nom_speed / (torch.sqrt(denom) + 1e-6)
            lam_max = torch.where(h < 0.0, torch.full_like(lam_max_safe, 1e6), lam_max_safe)
            lam = torch.clamp(lam_uncapped, max=lam_max)
            dq_projected = dq_in + lam.unsqueeze(-1) * grad_h

            # Hard recovery is useful only once we are meaningfully inside the
            # boundary. Near h=0, keep tangential nominal motion and project only
            # the unsafe normal component; otherwise the controller snaps at
            # d_safe due to tiny SDF/point-cloud noise.
            dot_nom_grad = (grad_h * dq_in).sum(dim=-1)
            nom_speed = torch.norm(dq_in, dim=-1)
            grad_h_unit = grad_h / (torch.sqrt(denom).unsqueeze(-1) + 1e-6)
            dq_recovery = nom_speed.unsqueeze(-1) * grad_h_unit
            use_recovery = (h < recovery_switch_h) & (dot_nom_grad < 0.0)

            proj_branch = (constr < 0.0) & (denom >= 1e-6)
            self.static_lam.copy_(lam)
            self.static_cap_active.copy_(
                (proj_branch & (~use_recovery) & (lam_uncapped > lam_max)).float())
            self.static_recovery_used.copy_((proj_branch & use_recovery).float())
            self.static_grad_degenerate.copy_(
                ((constr < 0.0) & (denom < 1e-6)).float())

            dq_safe = torch.where(
                constr.unsqueeze(-1) < 0,
                torch.where(
                    denom.unsqueeze(-1) < 1e-6,
                    torch.zeros_like(dq_in),
                    torch.where(use_recovery.unsqueeze(-1), dq_recovery, dq_projected),
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

    def joint_callback(self, msg):
        try:
            pos_dict = {n: p for n, p in zip(msg.name, msg.position)}
            q_list = []
            for jn in self.joint_names:
                if jn in pos_dict:
                    q_list.append(pos_dict[jn])
                else:
                    return # Ignore message if arm joints are missing
            self.current_q = torch.tensor(q_list, dtype=torch.float32, device=self.device).unsqueeze(0)
            if self.last_q_safe is None:
                self.last_q_safe = self.current_q.detach().clone()
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error in CBF joint_callback: {e}")

    def _nominal_cb(self, msg):
        q = torch.tensor(msg.data[:7], dtype=torch.float32, device=self.device).unsqueeze(0)
        self.nominal_q = q

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

            # Keep points close to any protected link, while dropping points
            # clearly inside the robot body mesh.
            keep_mask = (
                (sdf_all <= self.cbf_sdf_candidate_max_dist)
                & (sdf_body > self.cbf_sdf_self_filter_margin)
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
                        not_self = sdf_body > self.cbf_sdf_self_filter_margin
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
        dt_fixed = 1.0 / self.rate_hz
        last_time = rospy.get_time()

        while not rospy.is_shutdown():
            now = rospy.get_time()
            dt = now - last_time
            last_time = now

            if dt < 1e-4:
                self.rate.sleep()
                continue
            dt = min(dt, 3.0 * dt_fixed)

            if self.current_q is None or self.nominal_q is None:
                self.rate.sleep()
                continue

            t_loop_start = time.perf_counter()
            t_cbf_done = t_loop_start
            t_q_cmd_done = t_loop_start
            with torch.no_grad():
                q_cur = self.current_q.detach()
                q_nom = self.nominal_q.detach()

                # Keep the Flow-Matching/CASF tangent in the command given to
                # the CBF. Pure waypoint feedback points straight back to the
                # nominal curve after a safety deviation, which often leaves the
                # CBF with only "stop or push outward" behavior near obstacles.
                velocity_age = float('inf')
                use_velocity_topic = False
                if self.use_nominal_velocity_topic and self.nominal_dq_ff is not None:
                    velocity_age = now - self.nominal_dq_stamp
                    use_velocity_topic = (
                        self.nominal_velocity_timeout <= 0.0
                        or velocity_age <= self.nominal_velocity_timeout
                    )

                if use_velocity_topic:
                    dq_nom_pure = self.nominal_feedforward_gain * self.nominal_dq_ff.detach()
                    nominal_velocity_source = "topic"
                elif self.last_nominal_q is None:
                    dq_nom_pure = torch.zeros_like(q_nom)
                    nominal_velocity_source = "zero"
                else:
                    dq_nom_pure = (q_nom - self.last_nominal_q) / max(dt, 1e-4)
                    nominal_velocity_source = "diff"
                self.last_nominal_q = q_nom.clone()
                dq_nom_pure = torch.clamp(
                    dq_nom_pure,
                    min=-self.max_joint_velocity,
                    max=self.max_joint_velocity,
                )
                dq_feedforward = dq_nom_pure.clone()

                dq_feedback = torch.clamp(
                    self.cbf_kp * (q_nom - q_cur),
                    min=-self.max_feedback_velocity,
                    max=self.max_feedback_velocity,
                )
                dq_nom_raw = dq_nom_pure + dq_feedback

                if self.dq_nom_filtered is None:
                    self.dq_nom_filtered = dq_nom_raw.clone()
                if self.cbf_filter_tau > 0.0:
                    gamma = dt / (dt + self.cbf_filter_tau)
                    self.dq_nom_filtered = (
                        (1.0 - gamma) * self.dq_nom_filtered
                        + gamma * dq_nom_raw
                    )
                    dq_nom = self.dq_nom_filtered.clone()
                else:
                    dq_nom = dq_nom_raw

                if torch.norm(dq_nom) < self.nominal_hold_deadband:
                    dq_nom.zero_()
                    if self.dq_nom_filtered is not None:
                        self.dq_nom_filtered.zero_()

                dq_nom = torch.clamp(
                    dq_nom,
                    min=-self.max_joint_velocity,
                    max=self.max_joint_velocity,
                )
                dq_nom_base = dq_nom.clone()
                dq_escape = torch.zeros_like(dq_nom_base)
                dq_escape_tangent = torch.zeros_like(dq_nom_base)
                dq_escape_normal = torch.zeros_like(dq_nom_base)
                escape_active = False
                dq_cbf_delta = torch.zeros_like(dq_nom_base)
                constr_pre_t = None
                repair_flag_t = None
                final_constr_t = None
                ema_applied = False

                if self.enable_cbf:
                    local_obs = self.selected_obs
                    self.static_q.copy_(q_cur)
                    self.static_obs.copy_(local_obs)
                    self.static_dq_nom.copy_(dq_nom_base)
                    self.graph.replay()
                    dq_safe = self.static_dq_safe.detach().clone()

                    h_val_ema = float(self.static_h.item())
                    h_escape = h_val_ema
                    base_delta_value = float(torch.norm(dq_safe - dq_nom_base).item())
                    if self.cbf_escape_enabled:
                        grad_h = self.static_grad_h.detach()
                        grad_norm = torch.norm(grad_h, dim=1, keepdim=True)
                        h_bias_escape = float(
                            self.barrier.alpha * np.log(max(1, int(self.selected_count))))
                        h_escape = (
                            h_val_ema + h_bias_escape
                            if self.cbf_escape_use_bias_corrected_h
                            else h_val_ema
                        )
                        near_barrier = h_escape < self.cbf_escape_h_trigger
                        release_barrier = h_escape > self.cbf_escape_h_release
                        base_corrected = base_delta_value > self.cbf_escape_min_cbf_delta
                        if near_barrier and (base_corrected or h_escape < 0.0):
                            self._cbf_escape_active_cycles += 1
                        elif release_barrier or not base_corrected:
                            self._cbf_escape_active_cycles = 0
                        else:
                            self._cbf_escape_active_cycles = max(
                                0, self._cbf_escape_active_cycles - 1)

                        escape_active = (
                            self._cbf_escape_active_cycles >= self.cbf_escape_activation_cycles
                            and bool((grad_norm > 1e-7).all().item())
                        )
                        if escape_active:
                            normal_dir = grad_h / (grad_norm + 1e-6)
                            normal_speed = (dq_nom_base * normal_dir).sum(dim=1, keepdim=True)
                            dq_tangent = dq_nom_base - normal_speed * normal_dir

                            dq_escape_tangent = self.cbf_escape_tangent_gain * dq_tangent
                            dq_escape = dq_escape_tangent.clone()
                            if self.cbf_escape_normal_gain > 0.0:
                                h_trigger = max(abs(self.cbf_escape_h_trigger), 1e-6)
                                h_scale = (
                                    self.static_h.detach() + h_bias_escape
                                    if self.cbf_escape_use_bias_corrected_h
                                    else self.static_h.detach()
                                )
                                normal_scale = torch.clamp(
                                    (self.cbf_escape_normal_h_trigger - h_scale).view(-1, 1)
                                    / h_trigger,
                                    min=0.0,
                                    max=1.5,
                                )
                                dq_escape_normal = (
                                    self.cbf_escape_normal_gain * normal_scale * normal_dir
                                )
                                dq_escape = dq_escape + dq_escape_normal

                            if self.cbf_escape_max_velocity > 0.0:
                                escape_norm = torch.norm(dq_escape, dim=1, keepdim=True)
                                escape_scale = torch.clamp(
                                    self.cbf_escape_max_velocity / (escape_norm + 1e-6),
                                    max=1.0,
                                )
                                dq_escape_tangent = dq_escape_tangent * escape_scale
                                dq_escape_normal = dq_escape_normal * escape_scale
                                dq_escape = dq_escape * escape_scale

                            dq_nom = torch.clamp(
                                dq_nom_base + dq_escape,
                                min=-self.max_joint_velocity,
                                max=self.max_joint_velocity,
                            )
                            self.static_dq_nom.copy_(dq_nom)
                            self.graph.replay()
                            dq_safe = self.static_dq_safe.detach().clone()
                    else:
                        self._cbf_escape_active_cycles = 0

                    # Diagnostics: raw projection effect (before EMA/filtering)
                    # and the constraint value of the velocity fed to the QP.
                    dq_cbf_delta = dq_safe - dq_nom
                    constr_pre_t = self.static_constr.detach().clone()

                    t_cbf_done = time.perf_counter()
                    self._maybe_check_gradient_direction(
                        q_cur, local_obs, self.static_h.detach(),
                        self.static_grad_h.detach())
                    # EMA smoothing: reduces dq_safe jitter from 30 Hz gradient-
                    # direction jumps due to changing obstacle point selection.
                    # Only smooth near the boundary (|h| < 0.005); bypass and
                    # reset EMA when deeply negative so recovery corrections are
                    # not delayed by the filter's time constant (~90 ms).
                    if (
                        self._dq_safe_ema is None
                        or escape_active
                        or h_val_ema > 0.02
                        or h_val_ema < -0.005
                    ):
                        # Safe region, first iteration, escape, or deep recovery: no smoothing.
                        self._dq_safe_ema = dq_safe.clone()
                    else:
                        self._dq_safe_ema = (
                            self.cbf_ema_alpha * dq_safe
                            + (1.0 - self.cbf_ema_alpha) * self._dq_safe_ema
                        )
                        dq_safe = self._dq_safe_ema.clone()
                        ema_applied = True
                else:
                    dq_safe = dq_nom.clone()
                    self._dq_safe_ema = None
                    t_cbf_done = time.perf_counter()

                dq_safe = torch.clamp(
                    dq_safe,
                    min=-self.max_joint_velocity,
                    max=self.max_joint_velocity,
                )
                dq_safe_pre_filter = dq_safe.clone()
                velocity_filter_active = False

                if self.cbf_velocity_filter_tau > 0.0:
                    gamma_cmd = dt / (dt + self.cbf_velocity_filter_tau)
                    if self._dq_safe_filtered is None:
                        self._dq_safe_filtered = dq_safe.clone()
                    else:
                        self._dq_safe_filtered = (
                            (1.0 - gamma_cmd) * self._dq_safe_filtered
                            + gamma_cmd * dq_safe
                        )
                    dq_safe = torch.clamp(
                        self._dq_safe_filtered.clone(),
                        min=-self.max_joint_velocity,
                        max=self.max_joint_velocity,
                    )
                    velocity_filter_active = True
                else:
                    self._dq_safe_filtered = None

                if self.enable_cbf and self.cbf_enforce_final_constraint:
                    # The velocity clamp can invalidate the single half-space
                    # projection computed inside the CUDA graph. Filtering can
                    # also reintroduce an unsafe normal component. Repair once
                    # after clamping/filtering so the command still satisfies
                    # the active CBF inequality when joint limits allow it.
                    grad_h = self.static_grad_h.detach()
                    grad_norm_sq = (grad_h ** 2).sum(dim=1, keepdim=True)
                    h_now = self.static_h.detach()
                    kappa_eff = torch.where(
                        h_now < 0.0,
                        torch.full_like(h_now, self.cbf_recovery_kappa),
                        torch.full_like(h_now, self.cbf_kappa),
                    )
                    final_constr = (
                        (grad_h * dq_safe).sum(dim=1)
                        + kappa_eff * h_now
                        - self.cbf_constraint_margin
                    )
                    needs_repair = (
                        (final_constr < 0.0)
                        & (grad_norm_sq.squeeze(1) > 1e-8)
                    ).view(-1, 1)
                    repair_step = (-final_constr).clamp(min=0.0).view(-1, 1)
                    dq_repaired = (
                        dq_safe
                        + repair_step * grad_h / (grad_norm_sq + 1e-6)
                    )
                    dq_repaired = torch.clamp(
                        dq_repaired,
                        min=-self.max_joint_velocity,
                        max=self.max_joint_velocity,
                    )
                    dq_safe = torch.where(needs_repair, dq_repaired, dq_safe)
                    repaired_constr = (
                        (grad_h * dq_safe).sum(dim=1)
                        + kappa_eff * h_now
                        - self.cbf_constraint_margin
                    )
                    self.static_constr.copy_(repaired_constr)
                    repair_flag_t = needs_repair.float().view(-1)
                    final_constr_t = repaired_constr.detach().view(-1)

                if (
                    self.publish_controller_command
                    and self.cbf_direct_position_mode == "nominal_correction"
                ):
                    cbf_delta_vel = dq_safe - dq_nom
                    cbf_modified = (
                        bool((torch.norm(cbf_delta_vel) > self.nominal_hold_deadband).item())
                        or escape_active
                    )
                    if self.enable_cbf and cbf_modified:
                        correction_dt = (
                            self.cbf_position_correction_dt
                            if self.cbf_position_correction_dt > 0.0
                            else self.cbf_command_dt
                        )
                        q_cmd = q_cur + dq_safe * correction_dt
                        if self.cbf_max_command_lead > 0.0:
                            q_lead = q_cmd - q_cur
                            q_lead = torch.clamp(
                                q_lead,
                                min=-self.cbf_max_command_lead,
                                max=self.cbf_max_command_lead,
                            )
                            q_cmd = q_cur + q_lead
                    else:
                        q_cmd = q_nom.clone()
                else:
                    q_cmd = q_cur + dq_safe * self.cbf_command_dt
                    if self.cbf_max_command_lead > 0.0:
                        q_lead = q_cmd - q_cur
                        q_lead = torch.clamp(
                            q_lead,
                            min=-self.cbf_max_command_lead,
                            max=self.cbf_max_command_lead,
                        )
                        q_cmd = q_cur + q_lead

                self.q_safe_work.copy_(q_cmd)
                self.dq_safe_work.copy_(dq_safe)
                self._last_q_safe = q_cmd.clone()
                if self.last_q_safe is None:
                    self.last_q_safe = q_cmd.clone()
                else:
                    self.last_q_safe.copy_(q_cmd)

                self._viz_q = q_cur.detach().clone()
                self._viz_dq_nom = dq_nom.detach().clone()
                self._viz_dq_safe = dq_safe.detach().clone()
                t_q_cmd_done = time.perf_counter()

            q_cmd_cpu = q_cmd.squeeze(0).cpu().numpy().tolist()
            t_cpu_ready = time.perf_counter()
            pos_msg = Float64MultiArray(data=q_cmd_cpu)
            t_msg_ready = time.perf_counter()
            t_before_controller_pub = time.perf_counter()
            t_after_controller_pub = t_before_controller_pub
            if self.pos_cmd_pub is not None:
                self.pos_cmd_pub.publish(pos_msg)
                t_after_controller_pub = time.perf_counter()
            t_before_safe_pub = time.perf_counter()
            self.safe_cmd_pub.publish(pos_msg)
            t_after_safe_pub = time.perf_counter()

            if self.publish_diagnostics:
                nan_t = torch.full((1,), float('nan'), device=self.device)
                if self.enable_cbf:
                    h_t = self.static_h.detach().view(-1)
                    lam_t = self.static_lam.detach().view(-1)
                    cap_t = self.static_cap_active.detach().view(-1)
                    rec_t = self.static_recovery_used.detach().view(-1)
                    deg_t = self.static_grad_degenerate.detach().view(-1)
                    grad_norm_t = torch.norm(
                        self.static_grad_h.detach(), dim=1).view(-1)
                else:
                    h_t = lam_t = cap_t = rec_t = deg_t = grad_norm_t = nan_t
                # Single GPU->CPU transfer: 7 joint vectors then the GPU scalars
                # in the order unpacked below.
                diag_gpu = torch.cat([
                    dq_feedforward.view(-1),
                    dq_feedback.view(-1),
                    dq_nom_base.view(-1),
                    dq_escape.view(-1),
                    dq_cbf_delta.view(-1),
                    dq_safe_pre_filter.view(-1),
                    dq_safe.view(-1),
                    h_t, lam_t, cap_t, rec_t, deg_t,
                    constr_pre_t.view(-1) if constr_pre_t is not None else nan_t,
                    repair_flag_t if repair_flag_t is not None else nan_t,
                    final_constr_t if final_constr_t is not None else nan_t,
                    grad_norm_t,
                ]).float().cpu().numpy()
                n_vec = 7 * len(DIAG_VEC_FIELDS)
                (h_d, lam_d, cap_d, rec_d, deg_d,
                 cpre_d, rep_d, cfin_d, gnorm_d) = diag_gpu[n_vec:].tolist()
                if self.enable_cbf:
                    h_corr_d = h_d + float(self.barrier.alpha) * float(
                        np.log(max(1, int(self.selected_count))))
                else:
                    h_corr_d = float('nan')
                min_obs_d = float(self.selected_min_obs_dist)
                scalars = [
                    h_d,
                    h_corr_d,
                    lam_d,
                    cap_d,
                    rec_d,
                    deg_d,
                    cpre_d,
                    rep_d,
                    cfin_d,
                    gnorm_d,
                    1.0 if escape_active else 0.0,
                    1.0 if ema_applied else 0.0,
                    1.0 if velocity_filter_active else 0.0,
                    DIAG_VEC_SOURCE_CODES.get(
                        nominal_velocity_source, float('nan')),
                    float(velocity_age) if np.isfinite(velocity_age) else float('nan'),
                    float(dt),
                    float(self.selected_count),
                    min_obs_d if np.isfinite(min_obs_d) else float('nan'),
                    (t_cbf_done - t_loop_start) * 1000.0,
                    (t_q_cmd_done - t_cbf_done) * 1000.0,
                    (t_after_safe_pub - t_loop_start) * 1000.0,
                ]
                self.diag_pub.publish(Float32MultiArray(
                    data=diag_gpu[:n_vec].tolist() + scalars))

            if self.publish_debug_topics:
                self.debug_input_pub.publish(
                    Float64MultiArray(data=q_nom.squeeze(0).cpu().numpy().tolist()))
                self.debug_input_vel_pub.publish(
                    Float32MultiArray(data=dq_nom.squeeze(0).cpu().numpy().tolist()))
                self.debug_output_pub.publish(pos_msg)
                self.debug_output_vel_pub.publish(
                    Float32MultiArray(data=dq_safe.squeeze(0).cpu().numpy().tolist()))

            if self.enable_cbf:
                h_val = float(self.static_h.cpu().item())
                h_bias = float(self.barrier.alpha * np.log(max(1, int(self.selected_count))))
                h_bias_corrected = h_val + h_bias
                self.h_pub.publish(Float32MultiArray(data=[h_val]))
                if self.log_cbf_command_timing:
                    corrected = (
                        float(torch.norm(dq_safe - dq_nom_base).cpu())
                        > self.nominal_hold_deadband
                    )
                    cbf_submit_ms = (t_cbf_done - t_loop_start) * 1000.0
                    q_cmd_submit_ms = (t_q_cmd_done - t_cbf_done) * 1000.0
                    gpu_cpu_ms = (t_cpu_ready - t_q_cmd_done) * 1000.0
                    msg_ms = (t_msg_ready - t_cpu_ready) * 1000.0
                    safe_pub_ms = (t_after_safe_pub - t_before_safe_pub) * 1000.0
                    safe_total_ms = (t_after_safe_pub - t_loop_start) * 1000.0
                    if self.pos_cmd_pub is not None:
                        controller_pub_ms = (t_after_controller_pub - t_before_controller_pub) * 1000.0
                        corrected_to_controller_ms = (t_after_controller_pub - t_cpu_ready) * 1000.0
                        total_to_controller_ms = (t_after_controller_pub - t_loop_start) * 1000.0
                    else:
                        controller_pub_ms = float("nan")
                        corrected_to_controller_ms = float("nan")
                        total_to_controller_ms = float("nan")
                    self.debug_timing_pub.publish(Float32MultiArray(data=[
                        float(total_to_controller_ms),
                        float(corrected_to_controller_ms),
                        float(controller_pub_ms),
                        float(safe_total_ms),
                        float(safe_pub_ms),
                        float(gpu_cpu_ms),
                        float(cbf_submit_ms),
                        1.0 if corrected else 0.0,
                    ]))
                    rospy.loginfo_throttle(
                        1.0,
                        "[CBF TIMING] corrected=%s escape=%s h_soft=%.4f h_bias_corr=%.4f "
                        "total_to_controller=%.3fms corrected_to_controller=%.3fms "
                        "controller_publish=%.3fms safe_publish=%.3fms "
                        "gpu_cpu=%.3fms cbf_submit=%.3fms q_cmd=%.3fms msg=%.3fms",
                        corrected,
                        escape_active,
                        h_val,
                        h_bias_corrected,
                        total_to_controller_ms,
                        corrected_to_controller_ms,
                        controller_pub_ms,
                        safe_pub_ms,
                        gpu_cpu_ms,
                        cbf_submit_ms,
                        q_cmd_submit_ms,
                        msg_ms,
                    )
                    rospy.loginfo_throttle(
                        1.0,
                        "[CBF] h_soft=%.4f h_bias_corr=%.4f bias=%.4f "
                        "h_gate=%.4f vel_src=%s escape=%s cycles=%d "
                        "|dq_ff|=%.3f |dq_base|=%.3f |dq_cbf_delta|=%.3f "
                        "|dq_esc_t|=%.3f |dq_esc_n|=%.3f |dq_escape|=%.3f "
                        "|dq_des|=%.3f vel_filt=%s |dq_pre_filt|=%.3f |dq_safe|=%.3f",
                        h_val,
                        h_bias_corrected,
                        h_bias,
                        h_escape,
                        nominal_velocity_source,
                        escape_active,
                        self._cbf_escape_active_cycles,
                        float(torch.norm(dq_feedforward).cpu()),
                        float(torch.norm(dq_nom_base).cpu()),
                        base_delta_value,
                        float(torch.norm(dq_escape_tangent).cpu()),
                        float(torch.norm(dq_escape_normal).cpu()),
                        float(torch.norm(dq_escape).cpu()),
                        float(torch.norm(dq_nom).cpu()),
                        velocity_filter_active,
                        float(torch.norm(dq_safe_pre_filter).cpu()),
                        float(torch.norm(dq_safe).cpu()),
                    )
                    self._log_cuda_memory_throttle("CBF runtime")

            self.rate.sleep()

if __name__ == '__main__':
    node = CBFSafetyNode()
    node.run()
