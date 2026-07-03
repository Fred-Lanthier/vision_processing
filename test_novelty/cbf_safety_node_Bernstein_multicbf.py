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
from pipeline_timing import TimingPublisher

from third_party.SafeFlowMatcher.diffuser.models.rdf_cbf import RDF_CBF # Le solveur CBF est conservé
from third_party.RDF.urdf_layer import URDFLayer # Le module de cinématique est conservé pour l'Autograd

# import depuis le nouveau projet Bernstein
from third_party.SDF_Bernstein_Basis.src.rdf_weights import RDF_Weights
from third_party.SDF_Bernstein_Basis.bernstein_core import BernsteinCore
from third_party.SDF_Bernstein_Basis.bernstein_barrier import BernsteinBarrier
from vision_processing.analytical_bernstein import (
    AnalyticalBernsteinSoftmin,
    project_halfspaces_sequential,
)

from vision_processing import fast_perception_module


def _bool_param(name, default=False):
    value = rospy.get_param(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


# Layout of /cbf_safety/diagnostics (Float32MultiArray). Same base fields as
# cbf_safety_node_Bernstein.py so analysis scripts work across both nodes;
# concepts this node does not implement (output EMA, single QP multiplier) are
# published as 0/NaN. The multicbf extensions are appended after the base
# scalars. Full layout is published as JSON on the latched topic
# /cbf_safety/diagnostics_layout.
DIAG_VEC_FIELDS = [
    "dq_ff",          # feedforward from nominal position differencing
    "dq_fb",          # shaped tracking feedback (see _shape_tracking_feedback)
    "dq_base",        # filtered nominal velocity fed to the solver
    "dq_escape",      # local tangent/normal escape velocity before CBF solve
    "dq_cbf_delta",   # solver output minus dq_base (before repair)
    "dq_pre_filter",  # solver output before clamp/repair
    "dq_post_filter", # clamped solver output after the output low-pass, pre-repair
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
    "escape_active",    # 1 while local tangent/normal escape is active
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
    "h_hard",           # hard-min barrier on the same selected set
    "softmin_gap",      # h_hard - h; near 0 when alpha is effectively hard-min
    "h_food",           # grasped-target group barrier (its own grasp_d_safe);
                        # NaN unless two-group (grasp separate constraint) is on.
                        # The base "h" field is then the robot/fork group barrier.
]


class CBFSafetyNode:
    def __init__(self):
        rospy.init_node('cbf_safety_node')
        self.device = torch.device('cuda')
        # Per-stage timing -> /pipeline/timing/* (recorded in the bag for section 5.6)
        self.timing = TimingPublisher(enabled=rospy.get_param("~publish_timing", True))
        self._logged_first_nominal = False
        self._logged_first_safe_publish = False

        # Default = feeding fork robot (unchanged). Pick-and-place overrides this
        # with ~urdf_path:=.../urdf/panda_pickplace.xacro so the CBF builds the
        # gripper (no-fork) kinematics that match the pp Gazebo robot.
        urdf_path_raw = str(rospy.get_param('~urdf_path',
                                            pkg_path + '/urdf/panda_camera.xacro'))

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
        # Dynamic: ~cbf_link_names lets pick-and-place pass the gripper set (no fork);
        # default = the feeding fork set (unchanged). ~controlled_link is the link the
        # FM trajectory is generated about (fork_tip feeding, panda_TCP pick-place).
        link_names = list(rospy.get_param(
            '~cbf_link_names',
            ['panda_link4', 'panda_link5', 'panda_link6', 'panda_link7',
             'panda_hand', 'panda_rightfinger', 'panda_leftfinger', 'fork_tip']))
        self.protected_link_names = link_names
        self.controlled_link = str(rospy.get_param('~controlled_link', 'fork_tip'))
        # The TOOL link is the protected SDF link whose SDF drives the TCP/grasp
        # obstacle filtering. Use the controlled link if it is itself a protected SDF
        # link (feeding: fork_tip); else the most distal protected link = the tool
        # body (pick-place: panda_hand, since panda_TCP is a massless frame with no SDF).
        if self.controlled_link in link_names:
            self.fork_link_index = link_names.index(self.controlled_link)
        elif 'panda_hand' in link_names:
            self.fork_link_index = link_names.index('panda_hand')
        else:
            self.fork_link_index = len(link_names) - 1
        self.tool_link = link_names[self.fork_link_index]
        self.protected_fk_tree = self.robot_layer.make_fk_subset(link_names)
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
        self.barrier_alpha = float(rospy.get_param("~barrier_alpha", 0.005))
        self.barrier = BernsteinBarrier(
            self.bernstein_core, d_safe=self.d_safe, alpha=self.barrier_alpha)
        self._log_cuda_memory("CBF after Bernstein barrier")

        # Grasped object: when the grasp node welds food to fork_tip it publishes
        # the target's point cloud (in fork_tip frame); the core then treats it as
        # an extra protected link (distance-to-nearest-grasped-point SDF) AND
        # self-filters it. Configured BEFORE setup_cuda_graph so the grasp term is
        # captured in the CUDA graph (fixed-size cloud buffer).
        self.cbf_grasp_enabled = bool(rospy.get_param("~cbf_grasp_box_enabled", True))
        # Apply the grasped object as its OWN QP half-space (food group) separate
        # from the robot+fork group, each with its own d_safe, instead of folding
        # it into the single min-over-links gradient (where the food's low d_safe
        # lets it out-compete and starve the robot/fork correction).
        self.cbf_grasp_separate_constraint = bool(rospy.get_param(
            "~cbf_grasp_separate_constraint", True))
        self.grasp_attach_link = str(rospy.get_param("~grasp_box_attach_link", "fork_tip"))
        self.grasp_npts = int(rospy.get_param("~grasp_npts", 256))
        self.grasp_point_radius = float(rospy.get_param("~grasp_point_radius", 0.012))
        self.grasp_softmin_beta = float(rospy.get_param("~grasp_softmin_beta", 0.003))
        # Split the selected-obstacle budget so the food (smaller d_safe, closest)
        # cannot crowd out the robot/fork's nearby points: reserve this many of
        # cbf_graph_points for food-nearest points, the rest for robot/fork-nearest.
        self.cbf_grasp_obstacle_points = int(rospy.get_param(
            "~cbf_grasp_obstacle_points", 25))
        # Per-link d_safe for the grasped object (<0 => use the global d_safe).
        # Lower than the global d_safe so the carried food can pass close
        # obstacles (e.g. the container rim on exit) without braking.
        grasp_d_safe = float(rospy.get_param("~grasp_d_safe", -1.0))
        grasp_d_safe = None if grasp_d_safe < 0.0 else grasp_d_safe
        if self.cbf_grasp_enabled:
            self.bernstein_core.configure_grasp_cloud(
                self.grasp_attach_link, self.grasp_npts, self.grasp_point_radius,
                softmin_beta=self.grasp_softmin_beta,
                d_safe_global=self.d_safe, grasp_d_safe=grasp_d_safe)

        # Exact explicit RDF/Jacobian chain.  This evaluator never enables
        # gradients and never calls torch.autograd; it returns one smooth row per
        # protected link (plus the optional rigid grasp cloud).
        self.analytical_barrier = AnalyticalBernsteinSoftmin(
            self.bernstein_core,
            temperature=self.barrier_alpha,
            d_safe=self.d_safe,
        )

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
        if self.cbf_solver_mode not in ("fast_tangent", "multi_projected", "multi_graphed"):
            raise ValueError("~cbf_solver_mode must be 'fast_tangent', "
                             "'multi_projected', or 'multi_graphed'")
        self.cbf_active_constraints = max(1, int(rospy.get_param(
            "~cbf_active_constraints", 4)))
        # Phase profiling for the multi_projected solver (eager path). When on,
        # inserts cuda.synchronize() around evaluate / selection / POCS to see
        # where the latency goes. Adds syncs, so leave OFF in production.
        self.cbf_profile_multicbf = _bool_param("~cbf_profile_multicbf", False)
        self.cbf_multi_gradient_mode = str(rospy.get_param(
            "~cbf_multi_gradient_mode", "analytical")).lower()
        if self.cbf_multi_gradient_mode not in (
                "analytical", "finite_difference", "autograd"):
            raise ValueError(
                "~cbf_multi_gradient_mode must be 'analytical', "
                "'finite_difference', or 'autograd'")
        self.cbf_multi_fd_eps = max(
            1e-5, float(rospy.get_param("~cbf_multi_fd_eps", 1e-3)))
        self.cbf_h_activate = float(rospy.get_param("~cbf_h_activate", 0.006))
        self.cbf_max_inward_speed = float(rospy.get_param(
            "~cbf_max_inward_speed", 0.20))
        self.cbf_recovery_speed = float(rospy.get_param(
            "~cbf_recovery_speed", 0.0))
        self.cbf_recovery_depth = float(rospy.get_param(
            "~cbf_recovery_depth", 0.01))
        self.cbf_max_correction_speed = float(rospy.get_param(
            "~cbf_max_correction_speed", 0.35))
        self.cbf_recovery_max_correction_speed = max(
            0.0, float(rospy.get_param("~cbf_recovery_max_correction_speed", 0.0)))
        # Per-group overrides for the grasped-TARGET half-space. The values above
        # govern the fork+robot group; these govern the target group so its
        # smaller d_safe can have its own braking band / recovery authority. Each
        # defaults to the fork+robot value, so omitting them keeps both groups
        # identical (current behavior).
        self.cbf_grasp_h_activate = float(rospy.get_param(
            "~cbf_grasp_h_activate", self.cbf_h_activate))
        self.cbf_grasp_max_inward_speed = float(rospy.get_param(
            "~cbf_grasp_max_inward_speed", self.cbf_max_inward_speed))
        self.cbf_grasp_recovery_speed = float(rospy.get_param(
            "~cbf_grasp_recovery_speed", self.cbf_recovery_speed))
        self.cbf_grasp_recovery_depth = float(rospy.get_param(
            "~cbf_grasp_recovery_depth", self.cbf_recovery_depth))
        self.cbf_grasp_max_correction_speed = float(rospy.get_param(
            "~cbf_grasp_max_correction_speed", self.cbf_max_correction_speed))
        # Repair gradient gate. When |grad_h| falls below this the obstacle
        # geometry is degenerate/conflicting (e.g. the fork wedged between both
        # walls of a slim container, where opposing per-point gradients cancel):
        # the repair direction grad_h/|grad_h| is unreliable, so the repair is
        # ramped to zero between grad_min and 2*grad_min and the nominal (which
        # enters/exits the container vertically) is trusted instead.
        self.cbf_repair_grad_min = float(rospy.get_param(
            "~cbf_repair_grad_min", 0.1))
        # Metric for the QP min-norm projection. "identity" is the Euclidean
        # joint-space projection (legacy: dq_delta ∝ grad_h, so the correction
        # lands on whichever joint has the largest moment arm on the barrier,
        # regardless of what that does to the task trajectory). "task" projects
        # in the metric W = Jp^T Jp + w_rot Jo^T Jo + lambda I of the controlled
        # link (TCP/fork_tip): among all dq satisfying the same half-space, pick
        # the one that moves the task frame least, so body-link constraints
        # (e.g. the elbow near an obstacle) resolve through the arm's null space
        # instead of fighting the nominal. The half-space itself is unchanged:
        # safety is identical, only the correction direction differs.
        self.cbf_projection_metric = str(rospy.get_param(
            "~cbf_projection_metric", "identity")).lower()
        if self.cbf_projection_metric not in ("identity", "task"):
            raise ValueError(
                "~cbf_projection_metric must be 'identity' or 'task'")
        if (self.cbf_projection_metric == "task"
                and self.cbf_solver_mode == "multi_projected"):
            raise ValueError(
                "~cbf_projection_metric 'task' is only wired into the graphed "
                "solver modes (fast_tangent / multi_graphed)")
        self.cbf_task_metric_enabled = self.cbf_projection_metric == "task"
        # lambda: joint-space cost of null-space motion relative to the ~sigma^2
        # (~0.1-0.5) task-space cost. Smaller = freer null-space swivel.
        self.cbf_task_metric_lambda = float(rospy.get_param(
            "~cbf_task_metric_lambda", 0.01))
        # Orientation-vs-position trade in W: 0.01 treats 1 rad of task-frame
        # rotation like 10 cm of translation. Matters for the fork/TCP attitude.
        self.cbf_task_metric_rot_weight = float(rospy.get_param(
            "~cbf_task_metric_rot_weight", 0.01))
        self.cbf_task_metric_update_every = max(1, int(rospy.get_param(
            "~cbf_task_metric_update_every", 1)))
        self._task_metric_cycle = 0
        # Active null-space clearance ("deform while following"). The QP
        # correction, in ANY metric, only cancels inward velocity at the
        # barrier: nothing ever seeks MORE clearance than the bound demands, so
        # a nominal that keeps pushing a body link at an obstacle parks the arm
        # at h~0 indefinitely. This term adds dq_clear = gain * proximity *
        # N grad_h, with N the null-space projector of the controlled-link
        # 6D Jacobian: the posture actively deforms away from the critical
        # obstacle (elbow swivel) while the TCP stays exactly on the FM path.
        # It is injected BEFORE the QP, so every per-link half-space still
        # filters it. 0.0 disables (legacy). Requires the task metric (uses
        # its per-cycle Jacobian).
        self.cbf_nullspace_clearance_gain = float(rospy.get_param(
            "~cbf_nullspace_clearance_gain", 0.0))
        self.cbf_nullspace_clearance_h_activate = float(rospy.get_param(
            "~cbf_nullspace_clearance_h_activate", self.cbf_h_activate))
        if (self.cbf_nullspace_clearance_gain > 0.0
                and not self.cbf_task_metric_enabled):
            # The identity ablation arm must stay pure legacy: disable the
            # clearance drive (it needs the task Jacobian anyway) rather than
            # refusing to start.
            rospy.logwarn(
                "cbf_nullspace_clearance_gain=%.3g ignored: requires "
                "cbf_projection_metric 'task' (running pure legacy identity)",
                self.cbf_nullspace_clearance_gain)
            self.cbf_nullspace_clearance_gain = 0.0
        rospy.loginfo(
            "CBF projection metric: %s (lambda=%.3g, rot_weight=%.3g, "
            "nullspace_clearance_gain=%.3g)",
            self.cbf_projection_metric, self.cbf_task_metric_lambda,
            self.cbf_task_metric_rot_weight, self.cbf_nullspace_clearance_gain)
        self.cbf_final_constraint_mode = str(rospy.get_param(
            "~cbf_final_constraint_mode", "legacy_kappa")).lower()
        if self.cbf_final_constraint_mode not in ("legacy_kappa", "solver_bound"):
            raise ValueError(
                "~cbf_final_constraint_mode must be 'legacy_kappa' or 'solver_bound'")
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
        self.log_cbf_preprocess_breakdown = _bool_param("~log_cbf_preprocess_breakdown", False)
        self.cbf_preprocess_breakdown_period = float(rospy.get_param(
            "~cbf_preprocess_breakdown_period", 2.0))
        self._last_preprocess_breakdown_log = 0.0
        self.log_trajectory_timing = _bool_param("~log_trajectory_timing", True)
        self.active_plan_id = 0
        self.active_plan_stamp = None
        self.active_plan_duration = 0.0
        self._cbf_first_nominal_for_plan = False
        self._cbf_first_safe_for_plan = False
        self._last_gradient_check = 0.0
        self.last_cbf_position_delta = None

        self.cbf_graph_points = max(1, int(rospy.get_param("~cbf_graph_points", 100)))
        # Point partition for the two-group barrier: the first n_obs_robot of the
        # cbf_graph_points selected points feed the robot/fork constraint, the
        # remaining cbf_grasp_obstacle_points feed the target constraint.
        # _split_select_obstacles orders the selection as [robot... | food...] to
        # match this boundary, baked into the CUDA graph below.
        self.n_obs_robot_split = max(
            1, self.cbf_graph_points - max(0, self.cbf_grasp_obstacle_points))
        # Number of POCS sweeps over the active half-space constraints. Each
        # sweep projects dq onto every active constraint in turn; projecting onto
        # one can re-violate a previously satisfied one, so 1 sweep only
        # guarantees the last constraint while >1 iterates to a dq that satisfies
        # them all at convergence. Covers both the per-link constraints
        # (multi_graphed) and the two-group robot/grasp split. Baked into the
        # CUDA graph below.
        self.cbf_constraint_sweeps = max(
            1, int(rospy.get_param("~cbf_constraint_sweeps", 6)))
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
        self.cbf_escape_enabled = _bool_param("~cbf_escape_enabled", False)
        self.cbf_escape_h_trigger = float(rospy.get_param(
            "~cbf_escape_h_trigger", 0.006))
        self.cbf_escape_h_release = float(rospy.get_param(
            "~cbf_escape_h_release", 0.012))
        self.cbf_escape_use_bias_corrected_h = _bool_param(
            "~cbf_escape_use_bias_corrected_h", True)
        self.cbf_escape_activation_cycles = max(
            1, int(rospy.get_param("~cbf_escape_activation_cycles", 2)))
        self.cbf_escape_min_cbf_delta = float(rospy.get_param(
            "~cbf_escape_min_cbf_delta", 0.02))
        self.cbf_escape_tangent_gain = float(rospy.get_param(
            "~cbf_escape_tangent_gain", 1.0))
        self.cbf_escape_normal_gain = float(rospy.get_param(
            "~cbf_escape_normal_gain", 0.04))
        self.cbf_escape_normal_h_trigger = float(rospy.get_param(
            "~cbf_escape_normal_h_trigger", 0.0))
        self.cbf_escape_max_velocity = float(rospy.get_param(
            "~cbf_escape_max_velocity", 0.08))
        # Directed lift escape for head-on TCP-path blocks (runPP14: hand parked
        # at the prism wall during cube->box transport). The stock escape
        # amplifies the NOMINAL's tangential component, which is ~zero when the
        # nominal points straight into the wall; and the null-space clearance
        # has no authority on a hand-borne constraint. This term instead adds a
        # velocity that raises the controlled link while staying TANGENT to the
        # barrier (component along grad_h projected out): g^T dq_lift = 0, so it
        # never trades safety, and the QP still filters it against the other
        # link half-spaces. Requires the task metric (uses its Jacobian). Gated
        # to constraints WITHOUT null-space room, so it complements (not fights)
        # the clearance drive. 0.0 disables.
        self.cbf_escape_lift_gain = float(rospy.get_param(
            "~cbf_escape_lift_gain", 0.0))
        if self.cbf_escape_lift_gain > 0.0 and not self.cbf_task_metric_enabled:
            rospy.logwarn(
                "cbf_escape_lift_gain=%.3g ignored: requires "
                "cbf_projection_metric 'task' (needs the task Jacobian)",
                self.cbf_escape_lift_gain)
            self.cbf_escape_lift_gain = 0.0
        self._cbf_escape_active_cycles = 0

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
        self.publish_velocity_controller_command = _bool_param(
            "~publish_velocity_controller_command", False)
        if (self.publish_controller_command
                and self.publish_velocity_controller_command):
            raise ValueError(
                "Position and velocity controller command publication are "
                "mutually exclusive")
        self.preprocess_rate_hz = float(rospy.get_param("~preprocess_rate_hz", 30.0))
        self.viz_rate_hz = float(rospy.get_param("~viz_rate_hz", 5.0))
        self.publish_debug_topics = _bool_param("~publish_debug_topics", False)
        self.publish_diagnostics = _bool_param("~publish_diagnostics", True)
        self.cbf_diagnostics_decimation = max(
            1, int(rospy.get_param("~cbf_diagnostics_decimation", 1)))
        self._diagnostics_cycle = 0
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
        self.cbf_candidate_filter = str(rospy.get_param("~cbf_candidate_filter", "sphere")).lower()
        if self.cbf_candidate_filter not in ("sphere", "sdf", "link_spheres", "prism"):
            raise ValueError(
                "~cbf_candidate_filter must be 'sphere', 'link_spheres', 'prism', or 'sdf'")
        # 'prism' prefilter margin (world metres). The per-link box is the tight
        # per-axis AABB of {sdf <= margin} -- i.e. every point within `margin` of
        # the link surface -- so it is a compact rectangular prism, not a cube.
        # d_safe is 1.5 cm, so the default 3 cm keeps every point that could ever
        # be active in the QP while discarding the rest with a few matmuls.
        self.cbf_prism_margin = float(rospy.get_param("~cbf_prism_margin", 0.03))
        # Tight per-link boxes (local-frame AABB corners), filled lazily below.
        self.prism_lo = None
        self.prism_hi = None
        if self.cbf_candidate_filter == "prism":
            self._precompute_prism_boxes()
        # Debug radius for the RViz link spheres. The live 'link_spheres'
        # prefilter keeps a fixed number of nearest points, rather than a hard
        # radius, so the CBF is never starved when the cloud is sparse or the
        # nearest obstacle is just outside this visual radius.
        self.cbf_link_filter_radius = float(
            rospy.get_param("~cbf_link_filter_radius", 0.25))
        self.cbf_link_prefilter_points = int(rospy.get_param(
            "~cbf_link_prefilter_points", 512))
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
        self.cbf_selection_sticky_points = max(
            0, int(rospy.get_param("~cbf_selection_sticky_points", 0)))
        self.cbf_selection_sticky_radius = float(rospy.get_param(
            "~cbf_selection_sticky_radius", 0.03))
        self.cbf_selection_sticky_score_margin = float(rospy.get_param(
            "~cbf_selection_sticky_score_margin", 0.002))
        self.cbf_selection_duplicate_radius = float(rospy.get_param(
            "~cbf_selection_duplicate_radius", 0.003))
        self.cbf_selection_new_point_penalty = max(
            0.0, float(rospy.get_param("~cbf_selection_new_point_penalty", 0.0)))
        self.cbf_selection_sticky_force_keep = _bool_param(
            "~cbf_selection_sticky_force_keep", True)
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
        self.debug_link_centers = None
        # ('prism' filter) per-link oriented boxes for RViz: (centers[K,3],
        # rotations[K,3,3], full_side[K]) in the world frame.
        self.debug_prism_boxes = None
        self.debug_candidate_points = None   # the candidate pool (~512) the SDF ranks
        self.debug_sdf_prefilter_counts = None

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
        # Slots 0-38: original telemetry. Slots 39-70: diagnostics extension
        # (39:46 dq_ff, 46:53 dq_fb, 53:60 solver output, 60 repair flag,
        #  61 final constraint, 62 pre-projection constraint, 63 |grad_h|,
        #  64:71 dq_escape).
        self.transfer_buffer = torch.zeros(79, dtype=torch.float32, device=self.device)
        # Hot-path command buffer: dq_final, q_safe, h, constr. Keep diagnostics
        # out of this copy so command publication is not blocked by telemetry.
        self.command_buffer = torch.zeros(16, dtype=torch.float32, device=self.device)
        self._last_active_constraints = 0
        self._graphed_active_constraints = (
            max(1, min(self.cbf_active_constraints, self.bernstein_core.K))
            if self.cbf_solver_mode == "multi_graphed" else 0)
        self._log_cuda_memory("CBF after runtime buffers")

        self.comp_times = []
        self.preprocess_times = []
        rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        rospy.Subscriber('/planner/nominal_trajectory', JointTrajectory, self.trajectory_timing_callback)
        rospy.Subscriber('/planner/nominal_joint_command', Float64MultiArray, self.nominal_command_callback)
        rospy.Subscriber(self.nominal_velocity_topic, Float64MultiArray, self._nominal_vel_cb)
        obs_topic = rospy.get_param("~obs_topic", "/perception/persistent_obstacles")
        rospy.Subscriber(obs_topic, PointCloud2, self.obs_callback)
        if self.cbf_grasp_enabled:
            grasp_cloud_topic = rospy.get_param("~grasp_cloud_topic", "/fork_grasp/grasped_cloud")
            rospy.Subscriber(grasp_cloud_topic, PointCloud2, self._grasp_cloud_cb,
                             queue_size=1)

        self.cmd_pub = rospy.Publisher('/franka_control/safe_joint_velocities', Float32MultiArray, queue_size=1)
        self.pos_cmd_pub = None
        if self.publish_controller_command:
            self.pos_cmd_pub = rospy.Publisher('/joint_group_position_controller/command', Float64MultiArray, queue_size=1)
        self.velocity_cmd_pub = None
        if self.publish_velocity_controller_command:
            self.velocity_cmd_pub = rospy.Publisher(
                '/joint_group_velocity_controller/command',
                Float64MultiArray,
                queue_size=1)
            rospy.on_shutdown(self._stop_velocity_controller)
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
        self.pub_candidates = rospy.Publisher('/viz/cbf_candidates', PointCloud2, queue_size=1)
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
            "Multi-constraint CBF ready: control=%.1f Hz preprocess=%.1f Hz viz=%.1f Hz "
            "graph_points=%d active_constraints=%d multi_grad=%s fd_eps=%.1e "
            "h_activate=%.3fm max_inward=%.3fm/s "
            "recovery=%.3fm/s candidate_filter=%s selection=%s cluster=%s "
            "sticky=%d sticky_radius=%.3fm sticky_margin=%.3fm sticky_force=%s "
            "integrate_from_current=%s command_dt=%.3fs passthrough_inactive=%s "
            "yellow=%s debug=%s profile_sync=%s",
            self.rate_hz, self.preprocess_rate_hz, self.viz_rate_hz,
            self.cbf_graph_points,
            self.cbf_active_constraints,
            self.cbf_multi_gradient_mode,
            self.cbf_multi_fd_eps,
            self.cbf_h_activate,
            self.cbf_max_inward_speed,
            self.cbf_recovery_speed,
            self.cbf_candidate_filter,
            self.cbf_selection_metric,
            self.cbf_cluster_mode,
            self.cbf_selection_sticky_points,
            self.cbf_selection_sticky_radius,
            self.cbf_selection_sticky_score_margin,
            self.cbf_selection_sticky_force_keep,
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

    def _hardmin_h_value_no_grad(self, q, obs):
        if obs is None or int(obs.shape[0]) == 0:
            return torch.ones((q.shape[0],), dtype=torch.float32, device=self.device)
        pose = self.eye4.expand(q.shape[0], 4, 4)
        sdf_value = self.bernstein_core.get_whole_body_sdf_batch(
            obs, pose, q, return_per_link=False)
        return sdf_value.min(dim=1).values - self.d_safe

    def _robot_constraint_bound(self, h):
        h_activate = max(self.cbf_h_activate, 1e-6)
        recovery_depth = max(self.cbf_recovery_depth, 1e-6)
        outside_bound = -self.cbf_max_inward_speed * torch.clamp(
            h / h_activate, min=0.0, max=1.0)
        inside_bound = self.cbf_recovery_speed * torch.clamp(
            (-h) / recovery_depth, min=0.0, max=1.0)
        return torch.where(h > 0.0, outside_bound, inside_bound) \
            + self.cbf_constraint_margin

    def _constraint_residual(self, h, grad_h, dq):
        gdq = (grad_h * dq).sum(dim=1)
        if self.cbf_final_constraint_mode == "solver_bound":
            return gdq - self._robot_constraint_bound(h)

        kappa_eff = torch.where(
            h < 0.0,
            torch.full_like(h, self.cbf_recovery_kappa),
            torch.full_like(h, self.cbf_kappa),
        )
        return gdq + kappa_eff * h - self.cbf_constraint_margin

    def _diagnostic_active_constraints(self):
        if self.cbf_solver_mode == "multi_graphed":
            return self._graphed_active_constraints if self.selected_count > 0 else 0
        if self.cbf_solver_mode == "multi_projected":
            return self._last_active_constraints
        return 1 if self.selected_count > 0 else 0

    def _evaluate_link_sdf_local(self, link_index, points_local):
        """Evaluate one protected link SDF in that link's local frame."""
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

    def _precompute_prism_boxes(self):
        """Tight per-link oriented box for the 'prism' candidate filter.

        For each protected link, sample its Bernstein SDF on a grid in the link's
        local frame and take the per-axis bounds of ``{sdf <= cbf_prism_margin}``
        -- the set of points that could ever fall within the prefilter margin of
        the link surface. Stored as local-frame AABB corners
        (``prism_lo``/``prism_hi``, world metres). The box is constant in the
        link frame, so this runs once, and it is a compact rectangular prism that
        is tight on each axis independently (unlike the isotropic SDF cube).
        """
        resolution = self.safety_envelope_resolution
        level = self.cbf_prism_margin
        K = self.bernstein_core.K
        lo = torch.zeros((K, 3), device=self.device)
        hi = torch.zeros((K, 3), device=self.device)
        started = time.perf_counter()
        with torch.no_grad():
            for link_index in range(K):
                offset = self.bernstein_core.offsets[link_index]
                scale = float(self.bernstein_core.scales[link_index].item())
                # The surface lives within `scale` of the centroid; {sdf<=level}
                # extends ~level beyond it. Pad a little so the grid encloses it.
                half_extent = scale + level + 0.01
                spacing = 2.0 * half_extent / (resolution - 1)
                axes = [
                    torch.linspace(
                        float(offset[a].item()) - half_extent,
                        float(offset[a].item()) + half_extent,
                        resolution, device=self.device, dtype=torch.float32)
                    for a in range(3)
                ]
                grid = torch.stack(
                    torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(-1, 3)
                sdf_chunks = [
                    self._evaluate_link_sdf_local(link_index, chunk)
                    for chunk in torch.split(grid, 4096, dim=0)
                ]
                sdf = torch.cat(sdf_chunks)                       # [G]
                inside = sdf <= level
                if not bool(inside.any()):
                    # Degenerate: fall back to the isotropic cube.
                    lo[link_index] = offset - (scale + level)
                    hi[link_index] = offset + (scale + level)
                    rospy.logwarn(
                        "prism box: link %s has empty {sdf<=%.3f}; using cube.",
                        self.protected_link_names[link_index], level)
                    continue
                pts_in = grid[inside]
                # Pad by one grid cell so discretisation never clips a real point.
                lo[link_index] = pts_in.min(dim=0).values - spacing
                hi[link_index] = pts_in.max(dim=0).values + spacing
        self.prism_lo = lo
        self.prism_hi = hi
        rospy.loginfo(
            "Precomputed %d tight prism boxes (margin=%.3f m, grid=%d^3, %.0f ms).",
            K, level, resolution, (time.perf_counter() - started) * 1000.0)

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

    def _grasp_cloud_cb(self, msg):
        """Grasped target cloud from the grasp node, in the attach-link (fork_tip)
        frame. Sampled/padded to the fixed buffer size and written into the core;
        the CUDA graph picks it up on the next replay. Empty cloud => released."""
        core = self.bernstein_core
        if core.grasp_points is None:
            return
        try:
            pts = np.frombuffer(msg.data, dtype=np.float32).reshape(
                -1, int(msg.point_step / 4))[:, :3]
        except Exception:
            return
        M = core.grasp_npts
        n = len(pts)
        if n == 0:
            core.grasp_active.zero_()
            return
        idx = np.random.choice(n, M, replace=(n < M))
        core.grasp_points.copy_(
            torch.from_numpy(pts[idx].copy()).to(self.device, dtype=torch.float32))
        core.grasp_active.fill_(1.0)

    def setup_cuda_graph(self, batch_size=1, n_points=100):
        """
        Single unified CUDA graph:
          Calculates whole-body safety over n_points selected obstacle points.
          dq_safe = dq_nom + lam * grad_h.
        """
        print(
            f"⚡ Initialisation CBF ({n_points} selected points, "
            f"solver={self.cbf_solver_mode}, gradient={self.cbf_multi_gradient_mode})...")
        torch.cuda.empty_cache()

        self.static_q = torch.zeros(
            (batch_size, 7), device=self.device,
            requires_grad=(
                self.cbf_solver_mode == "fast_tangent"
                and self.cbf_multi_gradient_mode != "analytical"))
        self.static_obs    = torch.zeros((n_points, 3),   device=self.device)
        self.static_dq_nom = torch.zeros((batch_size, 7), device=self.device)
        self.static_dq_safe = torch.zeros((batch_size, 7), device=self.device)
        self.static_h      = torch.zeros((batch_size,), device=self.device)
        # Target (grasped-food) group barrier, retained for diagnostics only. The
        # two-group solve computes it but otherwise overwrites static_h with the
        # robot/fork value; keep a copy here so both can be plotted. NaN until the
        # two-group path writes it (so the diagnostic reads NaN when no grasp).
        self.static_h_food = torch.full((batch_size,), float('nan'), device=self.device)
        self.static_constr = torch.zeros((batch_size,), device=self.device)
        self.static_grad_h = torch.zeros((batch_size, 7), device=self.device)
        # W^-1 of the weighted min-norm projection. Identity = legacy Euclidean
        # projection (bit-compatible); with cbf_projection_metric == "task" the
        # control loop refreshes it every cycle OUTSIDE the graph and the graphed
        # _qp_step just reads it, so the capture stays static-shape.
        self.static_Winv = torch.eye(7, device=self.device)
        self._task_metric_eye7 = torch.eye(7, device=self.device)
        # Null-space projector of the controlled-link 6D Jacobian and its +z
        # position row, refreshed by _update_task_metric; None until the first
        # refresh.
        self._task_metric_N = None
        self._task_metric_Jz = None
        # Escape-trigger scalars [h, |dq_safe-dq_nom|, |grad_h|]: async D2H
        # into pinned memory, read one cycle late via event query (no sync).
        # Init = far-from-barrier so the escape stays off until real data.
        self._esc_pin = torch.tensor([1.0, 0.0, 1.0]).pin_memory()
        self._esc_vals = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        self._esc_event = torch.cuda.Event()
        self.graph = None
        if self.cbf_solver_mode not in ("fast_tangent", "multi_graphed"):
            self._log_cuda_memory("Multi-CBF runtime buffers ready")
            return

        # Capture scalars as Python floats so the CUDA graph bakes in the params.
        constraint_margin = self.cbf_constraint_margin
        # Per-group band/recovery constants, baked at capture. The fork+robot set
        # and the grasped-target set are passed explicitly to _qp_step so each
        # half-space brakes/recovers with its own authority.
        robot_band = (
            max(self.cbf_h_activate, 1e-6),
            max(0.0, self.cbf_max_inward_speed),
            max(0.0, self.cbf_recovery_speed),
            max(self.cbf_recovery_depth, 1e-6),
            max(0.0, self.cbf_max_correction_speed),
        )
        food_band = (
            max(self.cbf_grasp_h_activate, 1e-6),
            max(0.0, self.cbf_grasp_max_inward_speed),
            max(0.0, self.cbf_grasp_recovery_speed),
            max(self.cbf_grasp_recovery_depth, 1e-6),
            max(0.0, self.cbf_grasp_max_correction_speed),
        )

        def _qp_step(h, grad_h, dq_in, h_activate, max_inward_speed,
                     recovery_speed, recovery_depth, max_correction_speed):
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
            # Weighted min-norm step: dq_delta = c * W^-1 g / (g^T W^-1 g).
            # static_Winv == I reproduces the legacy Euclidean projection
            # exactly; either way grad_h^T dq_delta == c, so the half-space is
            # satisfied identically and only the correction direction changes.
            w_grad = grad_h @ self.static_Winv
            denom  = (grad_h * w_grad).sum(dim=-1)
            correction = torch.clamp(bound - gdq, min=0.0)
            dq_delta = correction.unsqueeze(-1) * w_grad / (denom.unsqueeze(-1) + 1e-4)
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

        # Two-group mode: robot/fork and the grasped object are separate QP
        # half-spaces (each its own d_safe). A single food-then-robot pass only
        # guarantees the LAST (robot) constraint; alternating the two projections
        # (POCS) with the gradients fixed at the current q converges to a dq that
        # satisfies BOTH half-spaces. Ending on the robot step leaves static_h /
        # static_constr / static_grad_h set to the robot group for the repair.
        two_group = (self.cbf_grasp_enabled and self.cbf_grasp_separate_constraint)
        n_robot = self.bernstein_core.K
        constraint_sweeps = self.cbf_constraint_sweeps

        analytical_point_mask = None
        if self.cbf_multi_gradient_mode == "analytical" and two_group:
            # Selected obstacles are ordered [robot points | food points].  Keep
            # the original two-group partition while evaluating all analytical
            # rows in one pass and one CUDA graph.
            analytical_point_mask = torch.zeros(
                (n_robot + 1, n_points), dtype=torch.bool, device=self.device)
            analytical_point_mask[:n_robot, :self.n_obs_robot_split] = True
            analytical_point_mask[n_robot, self.n_obs_robot_split:] = True

        def _analytical_constraints():
            result = self.analytical_barrier.evaluate(
                self.static_q,
                self.static_obs,
                point_mask=analytical_point_mask,
            )
            h_links = result.h
            g_links = result.grad_q
            if two_group:
                h_robot_links = h_links[:, :n_robot]
                robot_index = h_robot_links.argmin(dim=1)
                h_robot = h_robot_links.gather(
                    1, robot_index[:, None]).squeeze(1)
                g_robot = g_links[:, :n_robot, :].gather(
                    1,
                    robot_index[:, None, None].expand(-1, 1, 7),
                ).squeeze(1)
                return h_robot, g_robot, h_links[:, n_robot], g_links[:, n_robot, :]

            link_index = h_links.argmin(dim=1)
            h = h_links.gather(1, link_index[:, None]).squeeze(1)
            gradient = g_links.gather(
                1, link_index[:, None, None].expand(-1, 1, 7)).squeeze(1)
            return h, gradient

        # multi_graphed: fixed-shape topk(K_max) selection + a fixed POCS loop,
        # so the entire multi-constraint solve is CUDA-graph-capturable like
        # fast_tangent (no dynamic boolean masking, no GPU->CPU syncs). The proven
        # graph-safe _qp_step is reused; inactive (far) links no-op inside it
        # (active = h <= h_activate), so padding the set to a fixed K_max is safe.
        K_max = max(1, min(self.cbf_active_constraints, n_robot))
        if self.cbf_solver_mode == "multi_graphed":
            self._last_active_constraints = K_max

        def _multi_constraints():
            result = self.analytical_barrier.evaluate(
                self.static_q, self.static_obs, point_mask=analytical_point_mask)
            h_links = result.h            # [B, n_robot(+1 food)]
            g_links = result.grad_q       # [B, n_robot(+1), 7]
            h_robot = h_links[:, :n_robot]
            g_robot = g_links[:, :n_robot, :]
            h_vals, idx = torch.topk(h_robot, k=K_max, dim=1, largest=False)
            g_vals = g_robot.gather(1, idx[:, :, None].expand(-1, -1, 7))
            if two_group:
                return h_vals, g_vals, h_links[:, n_robot], g_links[:, n_robot, :]
            return h_vals, g_vals, None, None

        def _solve_multi(dq_nom):
            h_vals, g_vals, h_f, g_f = _multi_constraints()
            if two_group:
                self.static_h_food.copy_(torch.clamp(h_f, max=0.25))
            dq = dq_nom
            for _ in range(constraint_sweeps):
                if two_group:
                    dq = _qp_step(h_f, g_f, dq, *food_band)       # target half-space
                for k in range(K_max):                            # robot links (last)
                    dq = _qp_step(h_vals[:, k], g_vals[:, k, :], dq, *robot_band)
            # Leave static_h / static_grad_h on the most-critical robot link for
            # the diagnostics / repair (the loop ends on the least-critical row).
            min_k = h_vals.argmin(dim=1)
            self.static_h.copy_(h_vals.gather(1, min_k[:, None]).squeeze(1))
            self.static_grad_h.copy_(
                g_vals.gather(1, min_k[:, None, None].expand(-1, 1, 7)).squeeze(1))
            return dq

        def _solve(dq_nom):
            if self.cbf_solver_mode == "multi_graphed":
                return _solve_multi(dq_nom)
            if two_group:
                if self.cbf_multi_gradient_mode == "analytical":
                    h_r, g_r, h_f, g_f = _analytical_constraints()
                else:
                    h_r, g_r, h_f, g_f = self.barrier.forward_two_group(
                        self.static_q, self.static_obs, n_robot,
                        n_obs_robot=self.n_obs_robot_split)
                # Retain the target barrier for diagnostics before _qp_step
                # overwrites static_h with each group's value in turn. Capped at
                # 0.25 for the diagnostic only (the QP still uses the true h_f
                # below); far-from-obstacle spikes otherwise squash the plot.
                self.static_h_food.copy_(torch.clamp(h_f, max=0.25))
                dq = dq_nom
                for _ in range(constraint_sweeps):
                    dq = _qp_step(h_f, g_f, dq, *food_band)    # target half-space
                    dq = _qp_step(h_r, g_r, dq, *robot_band)   # fork+robot (last)
                self.static_grad_h.copy_(g_r)
                return dq
            if self.cbf_multi_gradient_mode == "analytical":
                h, g = _analytical_constraints()
            else:
                h, g, _ = self.barrier(self.static_q, self.static_obs)
            self.static_grad_h.copy_(g)
            return _qp_step(h, g, dq_nom, *robot_band)

        # ── Warmup ────────────────────────────────────────────────────────────
        def _warmup():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    if self.static_q.grad is not None: self.static_q.grad.zero_()
                    self.static_dq_safe.copy_(_solve(self.static_dq_nom))
            torch.cuda.current_stream().wait_stream(s)

        _warmup()
        torch.cuda.empty_cache()

        # ── Unified Graph ─────────────────────────────────────────────────────
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            if self.static_q.grad is not None: self.static_q.grad.zero_()
            self.static_dq_safe.copy_(_solve(self.static_dq_nom))

        print(
            f"✅ Graphe CUDA capturé ({n_points} points, "
            f"gradient={self.cbf_multi_gradient_mode}) !")
        self._log_cuda_memory("CBF setup_cuda_graph captured")

    def solve_multicbf_projection(self, current_q, obs_points, dq_nom):
        """
        Tangent-preserving multi-constraint CBF.

        In analytical mode, each active robot link contributes one smooth
        Softmin half-space (the optional grasp cloud is one additional row):
            grad_h_k · dq >= bound_k(h_k)
        Legacy finite-difference/autograd modes retain one row per obstacle point.
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
        q_probe = current_q.detach()

        prof = self.cbf_profile_multicbf
        _t = []
        if prof:
            torch.cuda.synchronize(self.device)
            _t.append(time.perf_counter())

        def _lap():
            if prof:
                torch.cuda.synchronize(self.device)
                _t.append(time.perf_counter())

        if self.cbf_multi_gradient_mode == "analytical":
            # One vectorized evaluation produces every per-link h_k and the full
            # Kx7 constraint Jacobian.  evaluate() is inference-mode internally:
            # no requires_grad tensors, dynamic graph, or autograd calls.
            analytical = self.analytical_barrier.evaluate(q_probe, obs_eval)
            h_probe = analytical.h[0]
            G_probe = analytical.grad_q[0]
            _lap()  # t_eval: analytical barrier (per-link h + Kx7 Jacobian)
            if int(h_probe.numel()) > self.bernstein_core.K:
                self.static_h_food.copy_(torch.clamp(h_probe[-1], max=0.25).view(1))

            active_window = min(
                int(h_probe.numel()),
                max(self.cbf_active_constraints * 3, self.cbf_active_constraints),
            )
            _, near_idx = torch.topk(
                h_probe, k=active_window, largest=False, sorted=True)
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
            h_det = h_probe[active_idx]
            G = G_probe[active_idx].contiguous()
            _lap()  # t_select: topk + any + boolean-mask indexing (sync points)
        else:
            # Legacy formulation: hard-min over links at each obstacle point.
            with torch.no_grad():
                pose_probe = self.eye4.expand(q_probe.shape[0], 4, 4)
                sdf_probe = self.bernstein_core.get_whole_body_sdf_batch(
                    obs_eval, pose_probe, q_probe, return_per_link=False)
                h_probe = sdf_probe[0] - self.d_safe

            active_window = min(
                valid_count,
                max(self.cbf_active_constraints * 3, self.cbf_active_constraints),
            )
            _, near_idx = torch.topk(
                h_probe, k=active_window, largest=False, sorted=True)
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

            if self.cbf_multi_gradient_mode == "finite_difference":
                with torch.no_grad():
                    eps = self.cbf_multi_fd_eps
                    q_base = current_q.detach().view(1, 7)
                    q_eval = q_base.expand(8, 7).clone()
                    q_eval[1:, :].add_(eps * torch.eye(
                        7, device=self.device, dtype=q_eval.dtype))
                    pose = self.eye4.expand(q_eval.shape[0], 4, 4)
                    sdf_fd = self.bernstein_core.get_whole_body_sdf_batch(
                        obs_active, pose, q_eval, return_per_link=False)
                    h_fd = sdf_fd - self.d_safe
                    h_det = h_fd[0].detach()
                    G = ((h_fd[1:] - h_fd[0:1]) / eps).transpose(
                        0, 1).contiguous()
            else:
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
                torch.set_grad_enabled(prev_grad_enabled)

        if int(h_det.numel()) <= 0:
            self._last_active_constraints = 0
            self.static_h.copy_(h_probe.min().view(1))
            self.static_constr.zero_()
            self.static_grad_h.zero_()
            return dq_nom

        h_activate = max(self.cbf_h_activate, 1e-6)
        recovery_depth = max(self.cbf_recovery_depth, 1e-6)
        outside_bound = -self.cbf_max_inward_speed * torch.clamp(
            h_det / h_activate, min=0.0, max=1.0)
        inside_bound = self.cbf_recovery_speed * torch.clamp(
            (-h_det) / recovery_depth, min=0.0, max=1.0)
        bounds = torch.where(h_det > 0.0, outside_bound, inside_bound)
        bounds = bounds + self.cbf_constraint_margin

        # Fixed-iteration on-device POCS for the 7-variable, K-row QP.  Tensor
        # conditions replace Python scalar reads, so there is no intermediate
        # GPU->CPU synchronization.  The only D2H copy left in the control path
        # is the final command_buffer transfer required by rospy publication.
        K = int(G.shape[0])
        dq_projected = project_halfspaces_sequential(
            dq_nom.detach().view(-1),
            G,
            bounds,
            iterations=self.cbf_projection_iters,
            relaxation=self.cbf_projection_relaxation,
            max_velocity=self.max_joint_velocity,
        )
        _lap()  # t_pocs: project_halfspaces_sequential
        if prof and len(_t) >= 4:
            ev = (_t[1] - _t[0]) * 1e3
            sel = (_t[2] - _t[1]) * 1e3
            pocs = (_t[3] - _t[2]) * 1e3
            rospy.loginfo_throttle(
                1.0,
                "⏱️ [multicbf] K=%d  evaluate=%.2f ms | select(sync)=%.2f ms | "
                "POCS=%.2f ms | sum=%.2f ms" % (
                    int(G.shape[0]), ev, sel, pocs, ev + sel + pocs))

        constr = torch.mv(G, dq_projected) - bounds
        min_idx = torch.argmin(h_det)
        self._last_active_constraints = K
        self.static_h.copy_(h_det[min_idx].view(1))
        self.static_constr.copy_(constr.min().view(1))
        self.static_grad_h.copy_(G[min_idx].view(1, 7))
        return dq_projected.view(1, 7)

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
        # NOTE: do NOT zero self.dq_nom_filtered here. The planner streams a
        # fresh plan at the waypoint rate (~10 Hz), so zeroing the velocity
        # low-pass on every plan reset it ~10x/s and never let it reach steady
        # state: the EMA fed to the QP became a 10 Hz sawtooth averaging ~55% of
        # the intended feedforward speed. The low-pass is meant to persist
        # across the continuously-streamed nominal; the finite-diff fallback is
        # already protected by the last_nominal_q reset above, and the default
        # velocity-topic feedforward does not use differencing at all.
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

    def _update_task_metric(self, current_q):
        """Refresh static_Winv (+ null-space projector / lift row) from the
        controlled-link Jacobian at the current q, via one batched
        finite-difference FK. Runs in the PREPROCESS timer thread (~30 Hz), NOT
        the 100 Hz control loop: putting it in the control path cost ~25 ms of
        GPU contention per cycle (runPP14: cbf_ms p50 4.8 -> 29.8 ms). The
        metric is a slowly-varying preconditioner, so <=33 ms staleness
        (<0.025 rad of joint motion) is irrelevant; the control thread only
        reads the buffers. All linear algebra is numpy on the 8 downloaded
        poses -- cuSolver on 7x7/6x6 is ~13x slower than numpy and adds sync
        points."""
        self._task_metric_cycle += 1
        if (self._task_metric_cycle - 1) % self.cbf_task_metric_update_every:
            return
        eps = 1e-3
        q_batch = current_q.detach().view(1, 7).expand(8, 7).clone()
        q_batch[1:].add_(eps * self._task_metric_eye7)
        T = self.get_link_pose(q_batch, self.controlled_link)
        if T is None:
            return
        Tc = T.detach().cpu().numpy().astype(np.float64)            # [8,4,4]
        # Rows of dP are dp/dq_i, so Jp^T Jp == dP @ dP^T (same for omega).
        dP = (Tc[1:, :3, 3] - Tc[0, :3, 3]) / eps                   # [7,3]
        dR = (Tc[1:, :3, :3] @ Tc[0, :3, :3].T - np.eye(3)) / eps   # ~skew(w_i)
        omega = 0.5 * np.stack([
            dR[:, 2, 1] - dR[:, 1, 2],
            dR[:, 0, 2] - dR[:, 2, 0],
            dR[:, 1, 0] - dR[:, 0, 1],
        ], axis=1)                                                   # [7,3]
        W = (dP @ dP.T
             + self.cbf_task_metric_rot_weight * (omega @ omega.T)
             + self.cbf_task_metric_lambda * np.eye(7))
        self.static_Winv.copy_(
            torch.from_numpy(np.linalg.inv(W).astype(np.float32)))
        if self.cbf_nullspace_clearance_gain > 0.0 or self.cbf_escape_lift_gain > 0.0:
            # Damped null-space projector N = I - J^+ J of the full 6D task
            # (position + orientation): motions in range(N) leave the TCP pose
            # unchanged to first order (the Panda's 1-DOF elbow swivel).
            J = np.concatenate([dP.T, omega.T], axis=0)              # 6x7
            N = np.eye(7) - J.T @ np.linalg.solve(
                J @ J.T + 1e-4 * np.eye(6), J)
            # Reference swap is atomic under the GIL; the control thread reads
            # whichever full projector is current.
            self._task_metric_N = torch.from_numpy(
                N.astype(np.float32)).to(self.device)
        if self.cbf_escape_lift_gain > 0.0:
            # dz_TCP/dq row of the position Jacobian: the joint direction that
            # raises the controlled link, used by the lift escape.
            self._task_metric_Jz = torch.from_numpy(
                dP[:, 2].astype(np.float32)).view(1, 7).to(self.device)

    def _nullspace_clearance_velocity(self):
        """Posture deformation velocity: push the body away from the critical
        obstacle inside the task null space, ramped by barrier proximity.
        Reads the previous cycle's h / grad_h (one 100 Hz cycle stale). All
        tensor ops: no GPU->CPU sync in the control path. The result is added
        to the nominal BEFORE the QP, so every half-space still filters it
        (e.g. a swivel toward another obstacle gets clipped by that link's
        constraint)."""
        if self.cbf_nullspace_clearance_gain <= 0.0 or self._task_metric_N is None:
            return None
        g = self.static_grad_h.detach()                       # [1,7], last cycle
        n_g = g @ self._task_metric_N                         # N symmetric
        n_norm = torch.norm(n_g, dim=1, keepdim=True)
        g_norm = torch.norm(g, dim=1, keepdim=True)
        h_act = max(self.cbf_nullspace_clearance_h_activate, 1e-6)
        proximity = torch.clamp(
            (h_act - self.static_h.detach().view(-1, 1)) / h_act,
            min=0.0, max=1.0)
        # Authority ramps with the fraction of grad_h that lies in the null
        # space: a constraint acting on the TCP itself (n_g ~ 0) has no
        # deformation room and gets no spurious push.
        align = torch.clamp(n_norm / (0.3 * g_norm + 1e-6), max=1.0)
        gate = (g_norm > 1e-3).float()
        return (self.cbf_nullspace_clearance_gain * proximity * align * gate
                * n_g / (n_norm + 1e-6))

    def _run_velocity_cbf_solver(self, current_q, local_obs, dq_nom):
        self.static_q.copy_(current_q)
        self.static_obs.copy_(local_obs)
        self.static_dq_nom.copy_(dq_nom)
        if self.cbf_solver_mode in ("fast_tangent", "multi_graphed"):
            self.graph.replay()
            return self.static_dq_safe.detach()
        return self.solve_multicbf_projection(
            current_q, local_obs, dq_nom).detach()

    def _queue_escape_state(self, dq_nom_cbf):
        """Enqueue an async D2H copy of the escape-trigger scalars (h,
        |dq_safe - dq_nom|, |grad_h|) into pinned memory after the solve.
        _compute_escape_velocity reads them NEXT cycle if the copy has
        completed (event query, never blocks). The old synchronous .cpu()
        here drained the contended GPU queue every 100 Hz cycle."""
        if not self.cbf_escape_enabled:
            return
        vals = torch.stack((
            self.static_h.detach().view(-1)[0],
            torch.norm(self.dq_safe_work - dq_nom_cbf),
            torch.norm(self.static_grad_h.detach()),
        ))
        self._esc_pin.copy_(vals, non_blocking=True)
        self._esc_event.record()

    def _compute_escape_velocity(self, dq_nom_base):
        """Escape decision from the PREVIOUS cycle's solve state (one 10 ms
        cycle stale -- the trigger is hysteresis/counter based, so this is
        equivalent in behavior but allows a single graph replay per cycle and
        no mid-loop sync)."""
        if not self.cbf_escape_enabled:
            self._cbf_escape_active_cycles = 0
            return torch.zeros_like(dq_nom_base), False

        grad_h = self.static_grad_h.detach()
        grad_norm = torch.norm(grad_h, dim=1, keepdim=True)
        h_bias = 0.0
        if self.cbf_solver_mode in ("fast_tangent", "multi_graphed"):
            h_bias = float(self.barrier.alpha) * float(
                np.log(max(1, int(self.selected_count))))
        if self._esc_event.query():
            self._esc_vals = self._esc_pin.numpy().copy()
        h_escape = float(self._esc_vals[0]) + (
            h_bias if self.cbf_escape_use_bias_corrected_h else 0.0)
        base_delta = float(self._esc_vals[1])
        grad_norm_value = float(self._esc_vals[2])

        near_barrier = h_escape < self.cbf_escape_h_trigger
        release_barrier = h_escape > self.cbf_escape_h_release
        base_corrected = base_delta > self.cbf_escape_min_cbf_delta
        if near_barrier and (base_corrected or h_escape < 0.0):
            self._cbf_escape_active_cycles += 1
        elif release_barrier or not base_corrected:
            self._cbf_escape_active_cycles = 0
        else:
            self._cbf_escape_active_cycles = max(
                0, self._cbf_escape_active_cycles - 1)

        escape_active = (
            self._cbf_escape_active_cycles >= self.cbf_escape_activation_cycles
            and grad_norm_value > 1e-7
        )
        if not escape_active:
            return torch.zeros_like(dq_nom_base), False

        normal_dir = grad_h / (grad_norm + 1e-6)
        normal_speed = (dq_nom_base * normal_dir).sum(dim=1, keepdim=True)
        dq_tangent = dq_nom_base - normal_speed * normal_dir
        dq_escape_tangent = self.cbf_escape_tangent_gain * dq_tangent
        dq_escape = dq_escape_tangent

        if self.cbf_escape_lift_gain > 0.0 and self._task_metric_Jz is not None:
            # Directed lift for head-on blocks: raise the controlled link along
            # the barrier's tangent plane (grad_h component projected out, so
            # g^T dq_lift = 0 and safety is untouched). Authority ramps up as
            # the null-space fraction of grad_h vanishes, i.e. exactly when the
            # constraint is TCP-borne and the clearance drive can do nothing.
            lift = self._task_metric_Jz
            lift_t = lift - (lift * normal_dir).sum(
                dim=1, keepdim=True) * normal_dir
            lift_norm = torch.norm(lift_t, dim=1, keepdim=True)
            lift_gate = torch.ones_like(lift_norm)
            if self._task_metric_N is not None:
                ns_frac = (
                    torch.norm(grad_h @ self._task_metric_N, dim=1, keepdim=True)
                    / (grad_norm + 1e-6))
                lift_gate = torch.clamp(1.0 - ns_frac / 0.3, min=0.0, max=1.0)
            dq_escape = dq_escape + torch.where(
                lift_norm > 1e-3,
                self.cbf_escape_lift_gain * lift_gate * lift_t
                / (lift_norm + 1e-6),
                torch.zeros_like(lift_t))

        if self.cbf_escape_normal_gain > 0.0:
            h_trigger = max(abs(self.cbf_escape_h_trigger), 1e-6)
            h_scale = (
                self.static_h.detach() + h_bias
                if self.cbf_escape_use_bias_corrected_h
                else self.static_h.detach()
            )
            normal_scale = torch.clamp(
                (self.cbf_escape_normal_h_trigger - h_scale).view(-1, 1)
                / h_trigger,
                min=0.0,
                max=1.5,
            )
            dq_escape = (
                dq_escape
                + self.cbf_escape_normal_gain * normal_scale * normal_dir
            )

        if self.cbf_escape_max_velocity > 0.0:
            escape_norm = torch.norm(dq_escape, dim=1, keepdim=True)
            escape_scale = torch.clamp(
                self.cbf_escape_max_velocity / (escape_norm + 1e-6),
                max=1.0,
            )
            dq_escape = dq_escape * escape_scale

        return dq_escape, True

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

    def _sphere_marker(self, marker_id, ns, point_tensor, color, scale, alpha=1.0):
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
        marker.color.a = alpha
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]

        p = point_tensor.detach().flatten().cpu().numpy()
        marker.pose.position.x = float(p[0])
        marker.pose.position.y = float(p[1])
        marker.pose.position.z = float(p[2])
        return marker

    @staticmethod
    def _matrix_to_quat(R):
        """3x3 rotation matrix (numpy) -> (x, y, z, w) quaternion."""
        m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
        tr = m00 + m11 + m22
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif m00 > m11 and m00 > m22:
            s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif m11 > m22:
            s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return x, y, z, w

    def _box_marker(self, marker_id, ns, center, rot, sides, color, alpha=1.0):
        """Oriented CUBE marker. center [3], rot [3,3], sides [3] (full edges)."""
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        s = sides.detach().flatten().cpu().numpy()
        marker.scale.x = float(s[0])
        marker.scale.y = float(s[1])
        marker.scale.z = float(s[2])
        marker.color.a = alpha
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        c = center.detach().flatten().cpu().numpy()
        marker.pose.position.x = float(c[0])
        marker.pose.position.y = float(c[1])
        marker.pose.position.z = float(c[2])
        qx, qy, qz, qw = self._matrix_to_quat(rot.detach().cpu().numpy())
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
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
        if self.debug_link_centers is not None:
            # Translucent green: the per-link candidate spheres of the
            # 'link_spheres' filter (diameter = 2 * cbf_link_filter_radius).
            diam = 2.0 * self.cbf_link_filter_radius
            centers = self.debug_link_centers
            for i in range(int(centers.shape[0])):
                markers.markers.append(self._sphere_marker(
                    100 + i, "cbf_link_spheres", centers[i],
                    (0.2, 0.85, 0.3), diam, alpha=0.15))
        if self.debug_prism_boxes is not None:
            # Translucent orange: the tight per-link oriented boxes of the
            # 'prism' filter (per-axis AABB of {sdf <= cbf_prism_margin}).
            centers_w, rots, sides = self.debug_prism_boxes
            for i in range(int(centers_w.shape[0])):
                markers.markers.append(self._box_marker(
                    200 + i, "cbf_prism_boxes", centers_w[i], rots[i],
                    sides[i], (1.0, 0.55, 0.1), alpha=0.12))
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
        # Pose of the controlled link (fork_tip feeding / panda_TCP pick-place) so it
        # aligns with nominal_trajectory_follower and the Flow Matching trajectory
        # coordinates. Full FK so frames like panda_TCP (not a protected SDF link)
        # still resolve.
        return self.get_link_pose(q7, self.controlled_link)

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

    def _split_select_obstacles(self, pts, q9, link_poses):
        """Split the cbf_graph_points budget between the robot/fork group and the
        grasped-food group, ranking each by ITS OWN per-link SDF. Prevents the
        food (smaller d_safe, closest) from taking all the slots and starving the
        robot/fork constraint. Returns the chosen obstacle points [M<=graph_points, 3]."""
        n_pts = int(pts.shape[0])
        n_food = max(0, min(self.cbf_grasp_obstacle_points, self.cbf_graph_points))
        n_robot = self.cbf_graph_points - n_food
        if q9 is None:
            q9 = torch.cat([self.current_q.detach(), self.q_pad2], dim=1)
        _, sdf_pl = self.bernstein_core.get_whole_body_sdf_batch(
            pts, self.eye4, q9, return_per_link=True, link_poses=link_poses)
        K = self.bernstein_core.K
        sdf_robot = sdf_pl[0, :K, :].min(dim=0).values            # nearest robot/fork link
        if sdf_pl.shape[1] <= K:                                  # no food link → no split
            k = min(self.cbf_graph_points, n_pts)
            _, idx = torch.topk(sdf_robot, k=k, largest=False)
            return pts[idx]
        sdf_food = sdf_pl[0, K, :]                                # grasped-food link
        n_robot = min(n_robot, n_pts)
        _, idx_robot = torch.topk(sdf_robot, k=max(n_robot, 1), largest=False)
        # Food-nearest points that are not already chosen for the robot group.
        keep = torch.ones(n_pts, dtype=torch.bool, device=self.device)
        keep[idx_robot] = False
        n_food = min(n_food, int(keep.sum()))
        if n_food > 0:
            food_scores = torch.where(keep, sdf_food,
                                      torch.full_like(sdf_food, float('inf')))
            _, idx_food = torch.topk(food_scores, k=n_food, largest=False)
            chosen = torch.cat([idx_robot, idx_food])
        else:
            chosen = idx_robot
        return pts[chosen]

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
        self.debug_link_centers = None
        self.debug_prism_boxes = None
        self.debug_candidate_points = None   # the candidate pool (~512) the SDF ranks

    def _prefilter_for_sdf(self, pts, q9, link_poses=None):
        """Cheap all-link prefilter before expensive whole-body SDF."""
        n_input = int(pts.shape[0])
        self.debug_sdf_prefilter_counts = (n_input, n_input, n_input)
        if self.cbf_candidate_filter == "prism":
            # Tight per-link oriented boxes instead of the radius ball: narrows
            # the cloud more aggressively before the whole-body SDF ranks it.
            kept = self._prism_prefilter(pts, q9, link_poses=link_poses)
            self.debug_sdf_prefilter_counts = (n_input, n_input, int(kept.shape[0]))
            return kept
        if self.cbf_sdf_prefilter_radius <= 0.0 and self.cbf_sdf_prefilter_max_points <= 0:
            return pts

        if link_poses is None:
            link_poses = self.robot_layer._native_forward_kinematics_subset(
                q9, self.protected_fk_tree)
        centers = []
        for link_name in self.protected_link_names:
            T_link = link_poses.get(link_name, None)
            if T_link is not None:
                centers.append(T_link[0, :3, 3])

        if not centers:
            return pts

        centers = torch.stack(centers, dim=0)
        link_dist = torch.cdist(pts.unsqueeze(0), centers.unsqueeze(0)).squeeze(0).min(dim=1).values

        # Sync-free prefilter: push out-of-radius points to +inf, then keep the
        # nearest max_points via topk (a fixed-size index gather — no boolean
        # mask and no torch.any, both of which force a GPU->CPU sync that, under
        # FM contention, blocks for tens of ms). Out-of-radius points only slip
        # through when fewer than max_points are in range, and the whole-body SDF
        # self-filter downstream drops them anyway.
        if self.cbf_sdf_prefilter_radius > 0.0:
            link_dist = torch.where(
                link_dist <= self.cbf_sdf_prefilter_radius,
                link_dist,
                torch.full_like(link_dist, float('inf')),
            )

        if self.cbf_sdf_prefilter_max_points > 0 and pts.shape[0] > self.cbf_sdf_prefilter_max_points:
            _, keep_idx = torch.topk(
                link_dist,
                k=self.cbf_sdf_prefilter_max_points,
                largest=False,
            )
            pts = pts[keep_idx]

        self.debug_sdf_prefilter_counts = (
            n_input, n_input, int(pts.shape[0]))
        return pts

    def _prism_prefilter(self, pts, q9, link_poses=None):
        """Cheap per-link oriented-box (prism) prefilter for the SDF candidate path.

        Each protected link carries a tight rectangular box (the per-axis AABB of
        ``{sdf <= cbf_prism_margin}``, precomputed in ``_precompute_prism_boxes``)
        in its local frame -- an oriented prism in the world, compact on every
        axis independently. This keeps only points inside SOME link's prism with a
        handful of matmuls plus a comparison, far cheaper than a Bernstein SDF
        evaluation and far tighter than a union-of-spheres.

        It REPLACES the radius-based ``_prefilter_for_sdf`` step: the survivors are
        then handed to the whole-body SDF in ``_whole_body_sdf_candidates``, which
        does the real surface-distance ranking AND the robot self-filter. The
        prism only narrows the set cheaply; it never decides clearance itself
        (ranking by box membership / link-centre distance is a poor proxy for an
        elongated link, which is why the standalone prism path was removed).

        Returns the reduced point set. The boolean mask is the same kind the
        ``sdf`` path already uses in this preprocessing stage (outside the live
        CUDA graph).
        """
        if int(pts.shape[0]) == 0:
            return pts

        core = self.bernstein_core
        # [1, K, 4, 4] visual-frame transforms -- the exact frames the boxes were
        # precomputed in, so prism_lo/prism_hi align with the links by index.
        trans = core._stack_used_link_transforms(self.eye4, q9, link_poses=link_poses)
        fk = trans.reshape(-1, 4, 4)                     # [K, 4, 4]
        R = fk[:, :3, :3].contiguous()                  # [K, 3, 3]
        t_vec = fk[:, :3, 3].contiguous()               # [K, 3]

        diff = pts.unsqueeze(0) - t_vec.unsqueeze(1)    # [K, N, 3]
        # torch.bmm(diff, R) == R^T @ diff (world->local), as in the SDF batch.
        x_local = torch.bmm(diff, R)                     # [K, N, 3]

        # Tight per-axis box test in the link frame (margin already baked in).
        lo = self.prism_lo.unsqueeze(1)                  # [K, 1, 3]
        hi = self.prism_hi.unsqueeze(1)                  # [K, 1, 3]
        inside_per_link = ((x_local >= lo) & (x_local <= hi)).all(dim=-1)  # [K, N]
        inside_any = inside_per_link.any(dim=0)                            # [N]

        # RViz boxes: oriented CUBE per link, centre = R @ centre_local + t,
        # orientation = R, per-axis side = hi - lo. Built only when subscribed.
        if self.pub_selection_debug.get_num_connections() > 0:
            centre_local = 0.5 * (self.prism_lo + self.prism_hi)          # [K, 3]
            centers_w = torch.bmm(centre_local.unsqueeze(1), R.transpose(1, 2)
                                  ).squeeze(1) + t_vec                     # [K, 3]
            sides = (self.prism_hi - self.prism_lo)                       # [K, 3]
            self.debug_prism_boxes = (
                centers_w.detach(), R.detach(), sides.detach())

        return pts[inside_any]

    def _whole_body_sdf_candidates(self, pts, q9, link_poses=None):
        """Return points near protected links, scored by whole-body SDF.

        This is outside the CUDA graph and runs at the preprocessing rate, so
        chunking keeps the memory bounded when the persistent cloud is large.
        """
        pts = self._prefilter_for_sdf(pts, q9, link_poses=link_poses)
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
                pts_chunk, self.eye4, q9, return_per_link=True, link_poses=link_poses)

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
            # No `if torch.any(keep_mask)` guard: that reduction is its own
            # GPU->CPU sync. Appending an empty masked tensor is harmless (the
            # cat below and the `if not kept_pts` fallback handle it), so drop
            # the guard and keep only the unavoidable mask-index.
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

    def _score_previous_selection(self, prev_pts, x_now_pos, q9, link_poses_current):
        """Score old selected points with the current selection metric."""
        if prev_pts.shape[0] == 0:
            empty = torch.empty((0,), dtype=torch.float32, device=self.device)
            return prev_pts, empty, empty

        dist_prev = torch.norm(prev_pts - x_now_pos, dim=1)
        if self.cbf_selection_metric == "fork_mesh":
            fork_mesh_world = self.debug_fork_mesh_world
            if fork_mesh_world is None or fork_mesh_world.shape[0] == 0:
                return prev_pts, dist_prev, dist_prev
            scores_prev = torch.cdist(
                prev_pts.unsqueeze(0),
                fork_mesh_world.unsqueeze(0),
            ).squeeze(0).min(dim=1).values
            return prev_pts, scores_prev, dist_prev

        if self.cbf_selection_metric in ("sdf", "fork_sdf"):
            _, sdf_per_link = self.bernstein_core.get_whole_body_sdf_batch(
                prev_pts, self.eye4, q9, return_per_link=True,
                link_poses=link_poses_current)
            sdf_body = sdf_per_link[0, :5, :].min(dim=0).values
            sdf_all = sdf_per_link[0].min(dim=0).values
            sdf_fork = sdf_per_link[0, self.fork_link_index, :]
            sdf_self_filter = (
                sdf_all if self.cbf_sdf_self_filter_all_links else sdf_body)
            not_self = sdf_self_filter > self.cbf_sdf_self_filter_margin
            if not torch.any(not_self):
                empty = torch.empty((0,), dtype=torch.float32, device=self.device)
                return prev_pts[:0], empty, empty
            prev_pts = prev_pts[not_self]
            dist_prev = dist_prev[not_self]
            scores_prev = sdf_fork[not_self] if self.cbf_selection_metric == "fork_sdf" else sdf_all[not_self]
            return prev_pts, scores_prev, dist_prev

        return prev_pts, dist_prev, dist_prev

    def _apply_selection_stickiness(self, pts_inside, scores, dist_inside,
                                    x_now_pos, q9, link_poses_current,
                                    largest=False):
        """Add hysteresis to obstacle selection to avoid red-point flicker."""
        sticky_max = min(
            self.cbf_selection_sticky_points,
            self.cbf_graph_points,
            int(self.selected_count),
        )
        if sticky_max <= 0 or pts_inside.shape[0] == 0:
            return pts_inside, scores, dist_inside

        prev_pts = self.selected_obs[:int(self.selected_count)].detach()
        if prev_pts.shape[0] == 0:
            return pts_inside, scores, dist_inside

        # Map each previously-selected point to its nearest current candidate.
        nearest_idx = None
        if self.cbf_selection_sticky_radius > 0.0:
            dmin = torch.cdist(
                prev_pts.unsqueeze(0),
                pts_inside.unsqueeze(0),
            ).squeeze(0).min(dim=1)
            still_present = dmin.values <= self.cbf_selection_sticky_radius
            if not torch.any(still_present):
                return pts_inside, scores, dist_inside
            prev_pts = prev_pts[still_present]
            nearest_idx = dmin.indices[still_present]

        if self.cbf_selection_metric in ("sdf", "fork_sdf"):
            # Reuse the nearest current candidate's already-computed SDF score
            # instead of recomputing a whole-body SDF on prev_pts (saves one
            # ~5 ms fixed-overhead call per cycle). The points are within
            # sticky_radius, so the score and self-filter decision match: a
            # point now inside the robot body has no nearby candidate (those
            # were self-filtered) and is already dropped by still_present above —
            # the same effect as _score_previous_selection's self-filter.
            if nearest_idx is None:
                nearest_idx = torch.cdist(
                    prev_pts.unsqueeze(0),
                    pts_inside.unsqueeze(0),
                ).squeeze(0).min(dim=1).indices
            prev_scores = scores[nearest_idx]
            prev_dist = torch.norm(prev_pts - x_now_pos, dim=1)
        else:
            prev_pts, prev_scores, prev_dist = self._score_previous_selection(
                prev_pts, x_now_pos, q9, link_poses_current)
            if prev_pts.shape[0] == 0:
                return pts_inside, scores, dist_inside

        if prev_pts.shape[0] > sticky_max:
            _, prev_idx = torch.topk(prev_scores, k=sticky_max, largest=largest)
            prev_pts = prev_pts[prev_idx]
            prev_scores = prev_scores[prev_idx]
            prev_dist = prev_dist[prev_idx]

        margin = abs(self.cbf_selection_sticky_score_margin)
        if self.cbf_selection_sticky_force_keep:
            anchor = scores.max() if largest else scores.min()
            prev_scores = (
                torch.full_like(prev_scores, anchor + margin)
                if largest else
                torch.full_like(prev_scores, anchor - margin)
            )
        else:
            prev_scores = prev_scores + margin if largest else prev_scores - margin

        if self.cbf_selection_duplicate_radius > 0.0:
            d_new_to_prev = torch.cdist(
                pts_inside.unsqueeze(0),
                prev_pts.unsqueeze(0),
            ).squeeze(0).min(dim=1).values
            new_mask = d_new_to_prev > self.cbf_selection_duplicate_radius
            pts_inside = pts_inside[new_mask]
            scores = scores[new_mask]
            dist_inside = dist_inside[new_mask]

        if self.cbf_selection_new_point_penalty > 0.0 and scores.numel() > 0:
            penalty = self.cbf_selection_new_point_penalty
            scores = scores - penalty if largest else scores + penalty

        pts_inside = torch.cat([prev_pts, pts_inside], dim=0)
        scores = torch.cat([prev_scores, scores], dim=0)
        dist_inside = torch.cat([prev_dist, dist_inside], dim=0)
        return pts_inside, scores, dist_inside

    def preprocess_obstacles(self, event):
        if self.current_q is None:
            return

        # Task-metric refresh lives here (preprocess thread, ~30 Hz), off the
        # 100 Hz control path. Before the obstacle early-return so the metric
        # stays current even while the cloud is empty.
        if self.cbf_task_metric_enabled:
            try:
                self._update_task_metric(self.current_q.detach())
            except Exception as e:
                rospy.logwarn_throttle(
                    5.0, "task-metric update failed: %s", e)

        if self.profile_sync:
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        stage_times = []
        last_stage_t = t_start

        def mark_stage(name):
            nonlocal last_stage_t
            if not self.log_cbf_preprocess_breakdown:
                return
            if self.profile_sync:
                torch.cuda.synchronize()
            now = time.perf_counter()
            stage_times.append((name, (now - last_stage_t) * 1000.0))
            last_stage_t = now

        pts = self.obs_points
        if pts is None or pts.shape[0] == 0:
            self._clear_or_hold_obstacles()
            return

        try:
            with torch.no_grad():
                q9_current = torch.cat([self.current_q.detach(), self.q_pad2], dim=1)
                link_poses_current = self.robot_layer._native_forward_kinematics_subset(
                    q9_current, self.protected_fk_tree)
                T_fork_center = link_poses_current.get(self.tool_link, None)
                if T_fork_center is None:
                    rospy.logwarn_throttle(
                        5, "CBF preprocessing cannot find tool link '%s' in FK tree." % self.tool_link)
                    return
                mark_stage("fork_fk")

                x_now_pos = T_fork_center[:, :3, 3]
                self.debug_filter_center = x_now_pos[0].detach()
                self.debug_fork_mesh_world = torch.empty((0, 3), dtype=torch.float32,
                                                         device=self.device)
                self.debug_score_seed = None
                self.debug_tip_min = None

                sdf_all_candidates = None
                sdf_fork_candidates = None
                q9 = q9_current

                if self.cbf_candidate_filter in ("sdf", "prism"):
                    # 'prism' uses the oriented-box prefilter inside
                    # _prefilter_for_sdf; both then rank survivors by the
                    # whole-body SDF (true surface clearance) + self-filter.
                    pts_inside, _, sdf_all_candidates, sdf_fork_candidates = self._whole_body_sdf_candidates(
                        pts, q9, link_poses=link_poses_current)
                    dist_inside = torch.norm(pts_inside - x_now_pos, dim=1)
                elif self.cbf_candidate_filter == "link_spheres":
                    # Cheap union-of-spheres over ALL protected links (one FK
                    # call, no voxel SDF), so obstacles near the elbow/wrist are
                    # kept, not just those near the fork. dist_inside is the
                    # nearest-link distance so the distance pre-top-k below also
                    # respects the whole body before the SDF selection runs.
                    centers = [T[0, :3, 3] for T in
                               (link_poses_current.get(n) for n in self.protected_link_names)
                               if T is not None]
                    if centers:
                        centers = torch.stack(centers, dim=0)
                        self.debug_link_centers = centers.detach()
                        diff = pts.unsqueeze(1) - centers.unsqueeze(0)
                        d_links_sq = (diff * diff).sum(dim=-1).min(dim=1).values
                        max_candidates = int(pts.shape[0])
                        if self.preprocess_max_points > 0:
                            max_candidates = min(max_candidates, self.preprocess_max_points)
                        if self.cbf_link_prefilter_points > 0:
                            max_candidates = min(max_candidates, self.cbf_link_prefilter_points)
                        max_candidates = max(
                            min(int(pts.shape[0]), self.cbf_graph_points),
                            max_candidates,
                        )
                        if int(pts.shape[0]) > max_candidates:
                            _, pre_idx = torch.topk(
                                d_links_sq, k=max_candidates, largest=False)
                            pts_inside = pts[pre_idx]
                            dist_inside = torch.sqrt(d_links_sq[pre_idx].clamp_min(0.0))
                        else:
                            pts_inside = pts
                            dist_inside = torch.sqrt(d_links_sq.clamp_min(0.0))
                    else:
                        dist_tcp = torch.norm(pts - x_now_pos, dim=1)
                        inside_mask = dist_tcp < self.tcp_filter_radius
                        pts_inside = pts[inside_mask]
                        dist_inside = dist_tcp[inside_mask]
                else:
                    dist_tcp = torch.norm(pts - x_now_pos, dim=1)
                    inside_mask = dist_tcp < self.tcp_filter_radius
                    pts_inside = pts[inside_mask]
                    dist_inside = dist_tcp[inside_mask]
                mark_stage("candidate_filter")

                num_inside = int(pts_inside.shape[0])

                if num_inside == 0:
                    self._clear_or_hold_obstacles()
                    return

                if self.cbf_selection_metric in ("sdf", "fork_sdf", "fork_mesh") or self.cbf_candidate_filter in ("sdf", "prism"):
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
                mark_stage("preselect")

                needs_score = (
                    num_inside > self.cbf_graph_points
                    or sdf_all_candidates is not None
                    or self.cbf_candidate_filter in ("sdf", "prism")
                )

                if not needs_score:
                    scores = dist_inside
                    largest = False
                elif self.cbf_selection_metric == "fork_mesh":
                    T_fork_now = T_fork_center
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
                            pts_inside, self.eye4, q9, return_per_link=True,
                            link_poses=link_poses_current)

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
                mark_stage("score")

                pts_inside, scores, dist_inside = self._apply_selection_stickiness(
                    pts_inside, scores, dist_inside, x_now_pos, q9,
                    link_poses_current, largest=largest)
                num_inside = int(pts_inside.shape[0])
                mark_stage("sticky")

                if num_inside == 0:
                    self._clear_or_hold_obstacles()
                    return

                # The candidate pool the SDF actually ranks (sphere-preselected,
                # distance-capped to preprocess_max_points): the set the top-100
                # are SDF-selected from. Published for visualization.
                self.debug_candidate_points = pts_inside.detach()

                score_min_idx = torch.argmin(scores)
                tip_min_idx = torch.argmin(dist_inside)
                self.debug_score_seed = pts_inside[score_min_idx].detach()
                self.debug_tip_min = pts_inside[tip_min_idx].detach()
                mark_stage("debug_seed")

                split_select = (
                    self.cbf_grasp_enabled
                    and self.cbf_grasp_separate_constraint
                    and self.cbf_grasp_obstacle_points > 0
                    and self.cbf_selection_metric in ("sdf", "fork_sdf")
                    and float(self.bernstein_core.grasp_active) > 0.5
                    and num_inside > self.cbf_graph_points
                )
                if split_select:
                    # Per-group budget: robot/fork-nearest + food-nearest, so the
                    # food cannot crowd the robot/fork points out of the QP set.
                    obs = self._split_select_obstacles(
                        pts_inside, q9, link_poses_current)
                    pts_yellow = torch.empty((0, 3), dtype=torch.float32, device=self.device)
                elif num_inside > self.cbf_graph_points:
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
                mark_stage("topk_copy")
                # Keep this on the GPU (no .item() here): a .item() in the
                # preprocess thread is a hard GPU sync that blocks until the whole
                # device queue drains — including the FM planner's 60-78 ms
                # predict sharing the GPU. It is converted to a Python float in
                # the command loop at L~2083, which runs *after* the existing
                # transfer_buffer.cpu() sync, so the read is then free.
                self.selected_min_obs_dist = dist_tip.min().detach()
                mark_stage("min_dist")

                if self.profile_sync:
                    torch.cuda.synchronize()
                self.preprocess_times.append(time.perf_counter() - t_start)
                self.timing.publish('critical_point_selection',
                                    (time.perf_counter() - t_start) * 1000.0)
                if self.log_cbf_preprocess_breakdown:
                    now_ros = rospy.get_time()
                    if now_ros - self._last_preprocess_breakdown_log >= self.cbf_preprocess_breakdown_period:
                        self._last_preprocess_breakdown_log = now_ros
                        total_ms = (time.perf_counter() - t_start) * 1000.0
                        stage_msg = " ".join(
                            f"{name}={dt_ms:.2f}ms" for name, dt_ms in stage_times)
                        prefilter_msg = ""
                        if self.debug_sdf_prefilter_counts is not None:
                            prefilter_msg = " prefilter=%d->%d->%d" % (
                                self.debug_sdf_prefilter_counts[0],
                                self.debug_sdf_prefilter_counts[1],
                                self.debug_sdf_prefilter_counts[2],
                            )
                        rospy.loginfo(
                            "[TIMING] CBF red breakdown: total=%.3fms obs=%d "
                            "candidates=%d selected=%d sync=%s%s | %s",
                            total_ms,
                            int(pts.shape[0]),
                            self.selected_num_inside,
                            self.selected_count,
                            self.profile_sync,
                            prefilter_msg,
                            stage_msg,
                        )
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

            # Blue: the full candidate pool the SDF ranks (red 100 are a subset).
            if self.debug_candidate_points is not None:
                msg_cand = self.create_cloud_xyzrgb(
                    self.debug_candidate_points, (60, 120, 255))
                if msg_cand is not None:
                    self.pub_candidates.publish(msg_cand)

            if msg_mesh is not None:
                self.pub_fork_mesh_debug.publish(msg_mesh)

            self.publish_selection_debug_markers()
            self.publish_safe_trajectory_marker(self.current_q, self.last_q_safe)
            self.publish_cbf_velocity_arrows()
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error publishing CBF visualization: {e}")

    def _stop_velocity_controller(self):
        """Best-effort stop; the controller watchdog is the hard backstop."""
        if self.velocity_cmd_pub is not None:
            try:
                self.velocity_cmd_pub.publish(
                    Float64MultiArray(data=[0.0] * 7))
            except rospy.ROSException:
                pass

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
                    dq_nom_base = dq_nom_torch.clone()
                    dq_escape = torch.zeros_like(dq_nom_base)
                    escape_active = False

                if self.profile_sync:
                    torch.cuda.synchronize()
                t_start = time.perf_counter()

                with torch.no_grad():
                    self.transfer_buffer[39:].zero_()
                    if self.enable_cbf:
                        # (task-metric buffers are refreshed by the preprocess
                        # timer thread; this loop only reads them)
                        # Null-space clearance (posture deformation) and the
                        # deadlock escape are both computed BEFORE the solve
                        # from the previous cycle's h/grad_h, then filtered by
                        # the SINGLE graph replay below. The old flow solved,
                        # synced to evaluate the escape, and re-solved when
                        # escaping -- two GPU-queue drains per cycle under
                        # contention (runPP14: cbf_ms p50 17-26 ms vs 4.8
                        # baseline). Both terms share the dq_escape diag slot.
                        dq_clear = self._nullspace_clearance_velocity()
                        dq_escape_t, escape_active = self._compute_escape_velocity(
                            dq_nom_base)
                        extra = dq_escape_t if escape_active else None
                        if dq_clear is not None:
                            extra = dq_clear if extra is None else extra + dq_clear
                        dq_nom_cbf = dq_nom_base
                        if extra is not None:
                            dq_escape = extra
                            dq_nom_cbf = torch.clamp(
                                dq_nom_base + extra,
                                min=-self.max_joint_velocity,
                                max=self.max_joint_velocity)
                        local_obs = self.selected_obs # Read reference atomically
                        self.dq_safe_work.copy_(self._run_velocity_cbf_solver(
                            current_q, local_obs, dq_nom_cbf))
                        self._queue_escape_state(dq_nom_cbf)
                        dq_nom_torch = dq_nom_cbf
                    else:
                        self.dq_safe_work.copy_(dq_nom_torch)

                    solver_out = self.dq_safe_work.detach().clone()
                    if self.cbf_monitor_only:
                        # Baseline mode: keep h/diagnostics (solver_out records
                        # the counterfactual correction) but apply nothing.
                        self.dq_safe_work.copy_(dq_nom_base)

                    self._viz_q = current_q.detach().clone()
                    self._viz_dq_nom = dq_nom_base.detach().clone()
                    self._viz_dq_safe = self.dq_safe_work.detach().clone()

                    dot = torch.sum(dq_nom_base * self.dq_safe_work, dim=1)
                    denom = torch.norm(dq_nom_base, dim=1) * torch.norm(self.dq_safe_work, dim=1)
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

                    # Pure output-filter command (clamped solver output through
                    # F_tau_safe), captured before the final-constraint repair so
                    # the diagnostic logs the filter's effect in isolation. With
                    # cbf_velocity_filter_tau == 0 this equals the clamped solver
                    # output and overlaps dq_pre_filter, as expected.
                    self.transfer_buffer[72:79].copy_(self.dq_safe_work.squeeze(0))

                    if self.enable_cbf and self.cbf_enforce_final_constraint and not self.cbf_monitor_only:
                        # Component-wise velocity limits can invalidate the QP
                        # half-space projection. If that happens, project the
                        # already-filtered command again so tangential/nominal
                        # motion is preserved instead of braking to zero.
                        grad_h = self.static_grad_h.detach()
                        grad_norm_sq = (grad_h ** 2).sum(dim=1, keepdim=True)
                        h_final = self.static_h.detach()
                        final_constr = self._constraint_residual(
                            h_final, grad_h, self.dq_safe_work)
                        final_active = h_final <= self.cbf_h_activate
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
                        # Repair velocity = repair_step * grad_h / |grad_h|^2, so
                        # its norm is repair_step / |grad_h|. When the soft-min
                        # gradient collapses (thin/slim obstacles: opposing
                        # per-point gradients on the two close faces cancel), this
                        # explodes and shoves the arm THROUGH the wall in an
                        # ill-defined direction. Bound the repair velocity norm to
                        # cbf_max_correction_speed so a degenerate gradient cannot
                        # produce a violent command; the residual violation is
                        # handled on later cycles as the geometry de-degenerates.
                        # Same weighted projection as _qp_step (static_Winv == I
                        # in identity mode): the repair keeps satisfying the raw
                        # half-space, but distributes the fix in the task metric
                        # so it does not undo the null-space-preferring solve.
                        # The degeneracy gates below stay on the RAW |grad_h|.
                        w_grad_final = grad_h @ self.static_Winv
                        w_denom_final = (grad_h * w_grad_final).sum(
                            dim=1, keepdim=True)
                        repair_delta = repair_step * w_grad_final / (
                            w_denom_final + 1e-6)
                        if (
                            self.cbf_max_correction_speed > 0.0
                            or self.cbf_recovery_max_correction_speed > 0.0
                        ):
                            rd_norm = torch.norm(repair_delta, dim=1, keepdim=True)
                            if self.cbf_max_correction_speed > 0.0:
                                repair_limit = torch.full_like(
                                    rd_norm, self.cbf_max_correction_speed)
                            else:
                                repair_limit = torch.full_like(rd_norm, float("inf"))
                            if self.cbf_recovery_max_correction_speed > 0.0:
                                recovery_limit = torch.full_like(
                                    rd_norm, self.cbf_recovery_max_correction_speed)
                                repair_limit = torch.where(
                                    self.static_h.detach().view(-1, 1) < 0.0,
                                    recovery_limit,
                                    repair_limit,
                                )
                            rd_scale = torch.clamp(
                                repair_limit / (rd_norm + 1e-9),
                                max=1.0)
                            repair_delta = repair_delta * rd_scale
                        # Gradient-reliability gate: ramp the repair from 0 at
                        # |grad_h| = grad_min to full at 3*grad_min, so a
                        # degenerate/conflicting gradient (fork wedged in a slim
                        # container, where |grad_h| collapses well below its ~0.6
                        # single-wall value) does not push the arm sideways along
                        # a meaningless direction. Trust the nominal there instead.
                        if self.cbf_repair_grad_min > 0.0:
                            grad_norm = torch.sqrt(grad_norm_sq)
                            grad_gate = torch.clamp(
                                (grad_norm - self.cbf_repair_grad_min)
                                / (2.0 * self.cbf_repair_grad_min + 1e-9),
                                min=0.0, max=1.0)
                            repair_delta = repair_delta * grad_gate
                        dq_repaired = self.dq_safe_work + repair_delta
                        dq_repaired.clamp_(min=-self.max_joint_velocity, max=self.max_joint_velocity)
                        self.dq_safe_work.copy_(torch.where(needs_repair, dq_repaired, self.dq_safe_work))
                        final_constr = self._constraint_residual(
                            h_final, grad_h, self.dq_safe_work)
                        self.static_constr.copy_(torch.where(
                            final_active,
                            final_constr,
                            torch.ones_like(final_constr),
                        ))
                        self.transfer_buffer[60].copy_(needs_repair.float().view(-1)[0])
                        self.transfer_buffer[61].copy_(final_constr.view(-1)[0])

                    if self.profile_sync:
                        torch.cuda.synchronize()
                    t_cbf_done = time.perf_counter()
                    cbf_delta_vel_for_command = self.dq_safe_work - dq_nom_base
                    cbf_modified_mask = (
                        torch.norm(cbf_delta_vel_for_command)
                        > self.nominal_hold_deadband
                    )
                    if self.enable_cbf:
                        h_now = self.static_h.detach()
                        cbf_active_mask_for_command = (
                            h_now <= self.cbf_h_activate
                        )
                    else:
                        h_now = None
                        cbf_active_mask_for_command = torch.zeros(
                            (1,), device=self.device, dtype=torch.bool)

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
                        cbf_delta_vel = self.dq_safe_work - dq_nom_base
                        cbf_delta_speed = torch.norm(cbf_delta_vel)
                        use_reactive_step = (
                            self.cbf_use_reactive_position_step
                            and self.enable_cbf
                            and (
                                bool(cbf_delta_speed > self.nominal_hold_deadband)
                                or escape_active
                            )
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

                    # Minimal hot-path copy for command publication.
                    self.command_buffer[0:7].copy_(dq_pub.squeeze(0))
                    self.command_buffer[7:14].copy_(self.q_safe_work.squeeze(0))
                    if self.enable_cbf:
                        self.command_buffer[14].copy_(self.static_h.squeeze(0))
                        self.command_buffer[15].copy_(self.static_constr.squeeze(0))
                    else:
                        self.command_buffer[14] = 1.0
                        self.command_buffer[15] = 0.0

                if self.profile_sync:
                    torch.cuda.synchronize()
                t_q_cmd_done = time.perf_counter()
                # Single small sync transfer to CPU for the actual command.
                command_cpu = self.command_buffer.cpu().numpy()
                dq_pub_cpu = command_cpu[0:7].tolist()
                q_safe_work_cpu = command_cpu[7:14].tolist()
                h_val = float(command_cpu[14])
                self.last_h_value_for_feedback = h_val
                clearance_val = h_val + self.d_safe
                constr_val = float(command_cpu[15])

                if (
                    self.contact_stop_enabled
                    and self.enable_cbf
                    and not self._contact_stopped
                ):
                    # Test the SAME conservative soft-min clearance that is
                    # reported/logged (clearance_val = h + d_safe). The previous
                    # de-bias term h_bias = alpha*log(N) (~4.6 mm with N=100)
                    # made this test that much more optimistic than the displayed
                    # clearance, so contact never latched even when the reported
                    # clearance reached ~0. The soft-min pessimism also offsets
                    # the perception underestimate of penetration, so clearance_val
                    # actually crosses 0 at true contact.
                    clearance_est = clearance_val
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
                if self.velocity_cmd_pub is not None:
                    # dq_safe is the actuator command in velocity mode.  Publish
                    # it before diagnostics so telemetry cannot delay control.
                    self.velocity_cmd_pub.publish(
                        Float64MultiArray(data=dq_pub_cpu))

                pos_msg = Float64MultiArray(data=q_safe_work_cpu)
                self.safe_cmd_pub.publish(pos_msg)
                t_pub_done = time.perf_counter()
                self.timing.publish('cbf_correction', (t_cbf_done - t_start) * 1000.0)
                self.timing.publish('cbf_command', (t_q_cmd_done - t_cbf_done) * 1000.0)
                self.timing.publish('cbf_total', (t_pub_done - t_start) * 1000.0)

                diag_due = (
                    self.publish_diagnostics
                    and (self._diagnostics_cycle % self.cbf_diagnostics_decimation == 0)
                )
                self._diagnostics_cycle += 1
                need_telemetry = (
                    diag_due
                    or self.log_cbf_events
                    or self.log_trajectory_timing
                    or self.publish_debug_topics
                )
                cpu_buffer = None
                if need_telemetry:
                    with torch.no_grad():
                        self.transfer_buffer[0:7].copy_(dq_pub.squeeze(0))
                        self.transfer_buffer[7:14].copy_(self.q_safe_work.squeeze(0))
                        if self.enable_cbf:
                            h_now = self.static_h.detach()
                            grad_h_dbg = self.static_grad_h.detach()
                            self._maybe_check_gradient_direction(
                                current_q, local_obs, h_now, grad_h_dbg)
                            self.transfer_buffer[14].copy_(h_now.squeeze(0))
                            self.transfer_buffer[15].copy_(self.static_constr.squeeze(0))
                            self.transfer_buffer[71].copy_(
                                self.static_h_food.detach().view(-1)[0])
                            self.transfer_buffer[62].copy_(
                                self._constraint_residual(
                                    h_now, grad_h_dbg, dq_nom_base).view(-1)[0])
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
                            real_constr = self._constraint_residual(
                                h_now, grad_h_dbg, dq_real)
                            norm_dq_real = torch.norm(dq_real)
                        else:
                            self.transfer_buffer[14] = 1.0
                            self.transfer_buffer[15] = 0.0
                            cmd_hdot = torch.zeros((1,), device=self.device)
                            real_hdot = torch.zeros((1,), device=self.device)
                            real_constr = torch.zeros((1,), device=self.device)
                            norm_dq_real = torch.zeros((), device=self.device)
                            h_rate = torch.zeros((1,), device=self.device)

                        self.transfer_buffer[16].copy_(alignment.squeeze(0))
                        self.transfer_buffer[17].copy_(torch.norm(dq_nom_base))
                        self.transfer_buffer[18].copy_(torch.norm(self.dq_safe_work))
                        self.transfer_buffer[19:26].copy_(nominal_q.squeeze(0))
                        self.transfer_buffer[26:33].copy_(dq_nom_base.squeeze(0))
                        self.transfer_buffer[33].copy_(cmd_hdot.squeeze(0))
                        self.transfer_buffer[34].copy_(real_hdot.squeeze(0))
                        self.transfer_buffer[35].copy_(real_constr.squeeze(0))
                        self.transfer_buffer[36].copy_(norm_dq_real)
                        self.transfer_buffer[37].copy_(h_rate.squeeze(0))
                        self.transfer_buffer[38].copy_(torch.norm(self.q_safe_work - current_q))
                        self.transfer_buffer[39:46].copy_(dq_nom_pure.squeeze(0))
                        self.transfer_buffer[46:53].copy_(dq_fb.squeeze(0))
                        self.transfer_buffer[53:60].copy_(solver_out.squeeze(0))
                        self.transfer_buffer[64:71].copy_(dq_escape.squeeze(0))

                    cpu_buffer = self.transfer_buffer.cpu().numpy()
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
                    dq_escape_cpu = cpu_buffer[64:71]
                    norm_dq_escape_val = float(np.linalg.norm(dq_escape_cpu))
                    cbf_delta_speed_val = float(
                        np.linalg.norm(
                            np.array(dq_pub_cpu, dtype=np.float32)
                            - np.array(dq_nom_torch_cpu, dtype=np.float32)
                        )
                    )

                if cpu_buffer is not None and self.log_cbf_events and not self._logged_first_safe_publish:
                    safe_err = float(np.linalg.norm(np.array(q_safe_work_cpu) - current_q.squeeze(0).cpu().numpy()))
                    nominal_err = float(np.linalg.norm(np.array(nominal_q_cpu) - current_q.squeeze(0).cpu().numpy()))
                    rospy.loginfo(
                        "CBF publishing first safe joint command | "
                        "safe_err_to_current=%.4f rad nominal_err_to_current=%.4f rad",
                        safe_err, nominal_err)
                    self._logged_first_safe_publish = True

                if (
                    cpu_buffer is not None
                    and self.log_trajectory_timing
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

                if diag_due and cpu_buffer is not None:
                    nan = float('nan')
                    self.h_pub.publish(Float32MultiArray(data=[h_val]))
                    if self.enable_cbf:
                        valid_obs_count = max(0, min(
                            int(self.selected_count), int(local_obs.shape[0])))
                        with torch.no_grad():
                            h_hard_tensor = self._hardmin_h_value_no_grad(
                                current_q.detach(), local_obs[:valid_obs_count])
                        h_hard_val = float(h_hard_tensor.detach().view(-1)[0].cpu())
                        softmin_gap_val = h_hard_val - h_val
                    else:
                        h_hard_val = nan
                        softmin_gap_val = nan
                    if self.enable_cbf and self.cbf_solver_mode in ("fast_tangent", "multi_graphed"):
                        h_corr_val = h_val + float(self.barrier.alpha) * float(
                            np.log(max(1, int(self.selected_count))))
                    else:
                        # Keep the conservative per-link Softmin value in the
                        # analytical multi_projected path; legacy multi modes are
                        # hard-min constraints and need no correction either.
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
                        1.0 if escape_active else 0.0,
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
                        float(self._diagnostic_active_constraints()),
                        0.0 if self.cbf_solver_mode == "fast_tangent" else 1.0,
                        1.0 if self.cbf_monitor_only else 0.0,
                        1.0 if self._contact_stopped else 0.0,
                        h_hard_val,
                        softmin_gap_val,
                        float(cpu_buffer[71]) if self.enable_cbf else nan,  # h_food
                    ]
                    self.diag_pub.publish(Float32MultiArray(data=(
                        cpu_buffer[39:46].tolist()                 # dq_ff
                        + cpu_buffer[46:53].tolist()               # dq_fb
                        + dq_base_v.tolist()                       # dq_base
                        + cpu_buffer[64:71].tolist()               # dq_escape
                        + (solver_out_v - dq_base_v).tolist()      # dq_cbf_delta
                        + solver_out_v.tolist()                    # dq_pre_filter
                        + cpu_buffer[72:79].tolist()               # dq_post_filter
                        + cpu_buffer[0:7].tolist()                 # dq_final
                        + scalars)))

                if self.profile_sync:
                    torch.cuda.synchronize()
                if self.log_cbf_command_timing:
                    self.comp_times.append(t_pub_done - t_start)
                    self._log_cuda_memory_throttle("CBF runtime")
                    if len(self.comp_times) >= 20:
                        avg_t = sum(self.comp_times) / len(self.comp_times)
                        if cpu_buffer is not None:
                            rospy.loginfo_throttle(
                                5.0,
                                f"⏱️ [TIMING] CBF publish latency: {avg_t*1000:.3f} ms | "
                                f"candidates={self.selected_num_inside} selected={self.selected_count} "
                                f"h={h_val:.4f} clearance={clearance_val:.4f} constr={constr_val:.4f} "
                                f"dq_nom={norm_dq_nom_val:.3f} dq_safe={norm_dq_safe_val:.3f} "
                                f"dq_delta={cbf_delta_speed_val:.3f} "
                                f"esc={int(escape_active)} dq_esc={norm_dq_escape_val:.3f} "
                                f"dq_real={norm_dq_real_val:.3f} q_lead={q_lead_val:.3f} "
                                f"align={alignment_val:.2f} "
                                f"hdot_cmd={cmd_hdot_val:.4f} hdot_real={real_hdot_val:.4f} "
                                f"h_rate={h_rate_val:.4f} real_constr={real_constr_val:.4f}")
                        else:
                            rospy.loginfo_throttle(
                                5.0,
                                f"⏱️ [TIMING] CBF publish latency: {avg_t*1000:.3f} ms | "
                                f"candidates={self.selected_num_inside} selected={self.selected_count} "
                                f"h={h_val:.4f} clearance={clearance_val:.4f} constr={constr_val:.4f}")
                        self.comp_times.clear()

                if self.publish_debug_topics and cpu_buffer is not None:
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
