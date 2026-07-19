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
from std_msgs.msg import Bool, Float32, Float32MultiArray, Float64MultiArray, Header, String
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
    "escape_active",    # proportional escape gate [0,1] (0 = off)
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
    "sampled_margin",   # state-dependent sampled-data QP margin m(dq) [m/s]
                        # (0 when ~cbf_sampled_data_margin is off)
]


class CBFSafetyNode:
    def __init__(self):
        rospy.init_node('cbf_safety_node')
        self.device = torch.device('cuda')
        # All heavy math runs on the GPU; the default intra-op pool (one
        # thread per core) only adds OpenMP spin-wait thrash against the SAM /
        # FM processes, inflating every eager-op dispatch in the 100 Hz loop
        # (runPP101: ~80 us/op vs the ~15 us normal).
        torch.set_num_threads(1)
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

        # Voxelized link meshes for URDFLayer's mesh resolution (matched by
        # basename). Default = panda; the xArm7 launch passes voxel_128_xarm7
        # (hand/finger meshes fall back to franka_description either way).
        voxel_dir = rospy.get_param(
            '~voxel_dir', pkg_path + '/third_party/RDF/panda_layer/meshes/voxel_128')
        self.robot_layer = URDFLayer(
            urdf_path=urdf_path,
            device=self.device,
            package_dir=pkg_path,
            voxel_dir=voxel_dir
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
            ['panda_link3','panda_link4', 'panda_link5', 'panda_link6', 'panda_link7',
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
        # Optional per-link candidate models (same mechanism as the SDF
        # visualizer): ~model_override_dir + ~model_override_links swap in
        # candidate <link>_w.pt files (e.g. the n_func=12 finger retrains in
        # panda_test/Models/nfunc12_candidate) without touching production.
        # BernsteinCore below re-groups by n_func, so mixed orders are fine.
        override_dir = str(rospy.get_param('~model_override_dir', '')).strip()
        if override_dir:
            from third_party.SDF_Bernstein_Basis.src.core.assets.load_model_wrapper \
                import load_link_weight_model
            override_links = rospy.get_param('~model_override_links', [])
            if isinstance(override_links, str):
                override_links = override_links.replace(',', ' ').split()
            for link in override_links:
                if link not in link_names:
                    rospy.logwarn("model override: %s not a protected link, skipped", link)
                    continue
                candidate_path = os.path.join(override_dir, f"{link}_w.pt")
                if not os.path.isfile(candidate_path):
                    raise FileNotFoundError(
                        f"model override for {link}: {candidate_path} not found")
                candidate = load_link_weight_model(
                    torch.load(candidate_path, map_location=self.device,
                               weights_only=False),
                    device=self.device, dtype=torch.float32)
                setattr(weight_handler, link + weight_handler.model_extension,
                        candidate)
                rospy.loginfo("model override: %s <- %s (n_func=%d)",
                              link, candidate_path, int(candidate.n_func))
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

        # Measured gripper tail for whole-body FK. The URDF carries 9 dof
        # (7 arm + 2 prismatic fingers) but only the arm is controlled; every
        # padding site used to assume closed fingers (zeros) while the real
        # gripper opens to 0.04 m per finger. This buffer holds the measured
        # finger positions, is updated in-place once per control cycle, and is
        # attached to both evaluators so all FK paths (including inside the
        # captured CUDA graph) read the live values. Must exist BEFORE
        # setup_cuda_graph so the capture records views of this memory.
        self.q_pad2 = torch.zeros((1, 2), device=self.device)
        self.bernstein_core.q_extra = self.q_pad2
        self.analytical_barrier.q_extra = self.q_pad2

        # kappa: class-K coefficient in the CBF constraint ∇h·dq + κh ≥ 0.
        # High kappa forces hard corrections even when h is slightly positive (safe).
        # With a jumpy gradient, high kappa amplifies oscillation near obstacles.
        # Recommended: 1.0–3.0. Must be read BEFORE setup_cuda_graph (baked into graph).
        self.cbf_kappa = float(rospy.get_param("~cbf_kappa", 2.0))
        self.cbf_recovery_kappa = float(rospy.get_param("~cbf_recovery_kappa", 3.0))
        self.cbf_constraint_margin = float(rospy.get_param("~cbf_constraint_margin", 0.0))
        # Sampled-data (ZOH) margin: replace the CONSTANT standoff heuristic
        # with the state-dependent term of the sampled-data CBF condition
        # (Breeden, Garg & Panagou, IEEE L-CSS 2022) that the certified reflex
        # already enforces post-hoc. Over one hold of period T the barrier can
        # dip by c = (L/2)||dq||^2 T^2 below its linear extrapolation, so
        # keeping h >= 0 across the WHOLE hold (not just at sample instants)
        # needs  hdot >= -kappa h + m(dq)  with  m(dq) = (L/2)||dq||^2 T [m/s].
        # The margin is thus EARNED by the certificate, scales with the speed
        # the QP itself allows (fast slide -> standoff covering its own
        # curvature dip; at rest -> 0), and makes the QP and the reflex agree
        # at the boundary instead of the reflex vetoing the QP every tick
        # (runPP1013: h parked at m_const/kappa = 0.2 mm < dip c, alpha 0.38,
        # braking 100% of ticks). The quadratic constraint is linearized per
        # cycle by evaluating ||dq||^2 at max(previous published command,
        # current nominal) — both slowly varying at 100 Hz.
        self.cbf_sampled_data_margin = _bool_param("~cbf_sampled_data_margin", False)
        # L: gradient-Lipschitz bound [m/rad^2], same constant family as the
        # reflex ~grad_lipschitz (scalar; default matches the launch's 20).
        self.cbf_sampled_data_L = float(rospy.get_param("~cbf_sampled_data_L", 20.0))
        # T: hold period [s]; 0 = auto (1/rate_hz). Read the param directly:
        # self.rate_hz is only set after finalize_cuda_graph() bakes the
        # coefficient into the graphed control step.
        self.cbf_sampled_data_T = float(rospy.get_param("~cbf_sampled_data_T", 0.0))
        # Cap [m/s] against runaway tightening at velocity spikes; at the
        # 0.7 rad/s clamp with L=20, T=10 ms, m = 0.049 m/s sits below it.
        self.cbf_sampled_data_margin_max = float(rospy.get_param(
            "~cbf_sampled_data_margin_max", 0.08))
        _sd_T = (self.cbf_sampled_data_T if self.cbf_sampled_data_T > 0.0
                 else 1.0 / max(float(rospy.get_param("~rate_hz", 150.0)), 1e-3))
        self._sampled_margin_T = _sd_T
        self._sampled_margin_coeff = (
            0.5 * self.cbf_sampled_data_L * _sd_T
            if self.cbf_sampled_data_margin else 0.0)
        # Linearization point of the margin's ||dq||^2 (and of the speed-
        # dependent ISSf bound below). "command" (legacy) = max(previous
        # published command, current nominal). That point is wrong twice
        # (runPP1023): the published command includes the repair, so a
        # repair episode self-inflates the margin to its cap (0.1*0.898^2 =
        # 0.081 -> bound infeasible -> worst_final_constraint -0.0455 while
        # h was +5 mm and RISING), and at a parked local minimum the
        # NOMINAL keeps pushing 0.46 rad/s while the arm executes 0.06, so
        # the standoff charges for a dip along a path never travelled
        # (~60x overcharge in v^2). "executed" = min(command-based value,
        # (||dq_measured|| + vel_headroom)^2): the dip is a property of the
        # EXECUTED ZOH segment, the executed speed is measured, and the
        # headroom is the identified per-hold speed INCREASE bound
        # (braking cannot deepen the dip; accel-only p95 per >=33 ms:
        # 0.075-0.138 rad/s over runPP1022/23). min() of two valid upper
        # bounds is a valid upper bound, and it breaks the repair->margin
        # feedback loop structurally.
        self.cbf_sampled_data_linearization = str(rospy.get_param(
            "~cbf_sampled_data_linearization", "command")).strip().lower()
        if self.cbf_sampled_data_linearization not in ("command", "executed"):
            rospy.logwarn("~cbf_sampled_data_linearization must be 'command' "
                          "or 'executed'; using 'command'")
            self.cbf_sampled_data_linearization = "command"
        self.cbf_sampled_data_vel_headroom = float(rospy.get_param(
            "~cbf_sampled_data_vel_headroom", 0.15))
        # Per-link gradient-Lipschitz constants for the margin, in
        # cbf_link_names order (same family as the reflex ~grad_lipschitz;
        # sdf_gradient_experiments/lipschitz_bound_study.py). The margin
        # row for link l then uses (L_l/2)*T*v^2 instead of the global
        # worst case: arm links carry L ~ 1.4-10 vs the global 20, so
        # their standoff halves for free. multi_graphed rows only; empty
        # string = scalar ~cbf_sampled_data_L everywhere (legacy).
        _ll = str(rospy.get_param("~cbf_sampled_data_L_links", "")).strip()
        self.cbf_sampled_data_L_links = (
            [float(x) for x in _ll.replace(",", " ").split()] if _ll else None)
        if self.cbf_sampled_data_margin:
            rospy.loginfo(
                "CBF sampled-data margin ON: m(dq) = %.3f*||dq||^2 m/s "
                "(L=%.1f, T=%.4f s, cap %.3f m/s)",
                self._sampled_margin_coeff, self.cbf_sampled_data_L,
                _sd_T, self.cbf_sampled_data_margin_max)
        # ISSf (input-to-state safety) tracking-error margin, per QP row.
        # The plant executes dq = u + d, not the command u (velocity-
        # controller lag, output low-pass memory, reflex alpha-scaling), so
        # a constraint certified on u says nothing about the true flow. With
        # an identified matched-disturbance bound ||d|| <= epsilon [rad/s],
        # tightening every row to
        #   grad_h_i . u >= bound_i + ||grad_h_i|| * epsilon
        # makes the TRUE velocity satisfy grad_h_i . (u + d) >= bound_i
        # (Kolathaya & Ames, "Input-to-State Safety with Control Barrier
        # Functions", IEEE L-CSS 2019). Without it the certificate silently
        # assumes perfect tracking: runPP1022 dipped to h = -1.1 mm while
        # the QP commanded OUTWARD every cycle (cmd_hdot +0.02 vs measured
        # -0.02). epsilon is identified from the logged mismatch
        # ||dq_real - dq_published|| (dedup + ZOH-averaged command:
        # runPP1022 rms 0.034, p99 0.12, dip-window p95 0.16). Standoff
        # cost |grad_h| eps / kappa (~7 mm at eps 0.15, |g| 0.22, kappa 5).
        # 0 = off (legacy bound, bit-identical). No new diagnostic needed:
        # the margin reconstructs offline as eps * grad_h_norm (logged).
        self.cbf_issf_epsilon = float(rospy.get_param(
            "~cbf_issf_epsilon", 0.0))
        # Speed-dependent disturbance bound eps(v) = eps0 + rho * v, with v
        # the SAME executed-envelope linearization speed as the sampled
        # margin. Identified from the tracking mismatch binned by measured
        # speed (runPP1022/23 p95 envelope: ~0.04 parked, 0.06 at 0.1,
        # 0.09 at 0.25, 0.16+ at 0.5 -> eps0 = 0.05, rho = 0.35). All
        # margins then shrink to their identified floor at rest — the
        # parked standoff stays certified but near-minimal — and grow with
        # actual activity. rho = 0 reproduces the constant-eps ISSf bound.
        self.cbf_issf_rho = float(rospy.get_param("~cbf_issf_rho", 0.0))
        self._issf_on = (self.cbf_issf_epsilon > 0.0
                         or self.cbf_issf_rho > 0.0)
        if self._issf_on:
            rospy.loginfo(
                "CBF ISSf margin ON: bound += ||grad_h_row|| * (%.3f + "
                "%.3f * v) rad/s (linearization: %s)",
                self.cbf_issf_epsilon, self.cbf_issf_rho,
                self.cbf_sampled_data_linearization)
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
        # Repair scope. The final-constraint repair re-validates the POST-
        # low-pass command; "critical" (legacy, False) projects it onto the
        # min-h row's half-space only. That is wrong whenever SEVERAL rows
        # are active with opposing gradients: runPP1022 parked pinched
        # between link4/link5 (cos(g4,g5) = -0.30, both h ~ -0.1 mm); the
        # filter pulled the command off the feasible set, the repair fixed
        # link4 to residual ~0 and thereby pushed INTO link5 (-0.028 m/s
        # violated every cycle), and the reflex rightly vetoed the result
        # (alpha = 0, arm frozen at h = -0.2 mm) even though a direction
        # improving BOTH rows existed (bisector: +0.037 m/s at |dq| 0.3).
        # True = re-run the Dykstra sweeps over ALL K solve rows (h/grad/
        # env frozen from this cycle's evaluate, exported by _solve_multi;
        # pure fixed-shape in-graph vector ops). multi_graphed + graphed
        # control step only; other paths warn and keep the legacy repair.
        self.cbf_repair_all_rows = _bool_param("~cbf_repair_all_rows", False)
        # Sweep count for the all-rows repair, decoupled from the solve's
        # cbf_constraint_sweeps. 0 = use cbf_constraint_sweeps (legacy).
        # Each sweep is K tiny [7]-vector ops inside the CUDA graph, so a
        # high count is essentially free; runPP1023 v6's constant -0.003
        # residual was the geometric tail of only 6 sweeps over task-
        # metric-skewed opposing rows, re-paid every cycle.
        self.cbf_repair_sweeps = int(rospy.get_param("~cbf_repair_sweeps", 0))
        # Inner algorithm of the all-rows repair. "dykstra" (default) =
        # correction-memory sweeps converging to the EXACT min-W-norm
        # projection onto the row intersection (same algorithm family the
        # solve validated in scripts/solver_study.py; smallest deviation
        # from the near-optimal post-filter command, order-independent).
        # Dykstra REQUIRES exact per-row projections, so under it the
        # grad-min gate is BINARY (a row with ||g|| < cbf_repair_grad_min
        # is excluded outright): damping the step instead parks the fixed
        # point at gate*projection, permanently short of the half-space —
        # the runPP1023 v6 -0.0012 block. "pocs" = plain under-relaxed
        # cyclic projections, which tolerate the legacy damped ramp
        # (feasibility only, possible outward over-correction).
        self.cbf_repair_qp_algorithm = str(rospy.get_param(
            "~cbf_repair_qp_algorithm", "dykstra")).strip().lower()
        if self.cbf_repair_qp_algorithm not in ("dykstra", "pocs"):
            rospy.logwarn("~cbf_repair_qp_algorithm must be 'dykstra' or "
                          "'pocs'; using 'dykstra'")
            self.cbf_repair_qp_algorithm = "dykstra"
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
        # Barrier VALUE mode. "soft": legacy per-link Softmin value, which
        # under-reports clearance by the crowding bias tau*log(sum e^{-d/tau})
        # -- ~15 mm (= d_safe!) when a dense synthetic cloud (moving sphere,
        # 1000 pts) parks many points within tau of the minimum, so the
        # reported h dips below 0 while the true clearance stays positive.
        # "hard": use the exact per-link hard-min for the VALUE fed to the QP
        # band and diagnostics, keeping the Softmin only for the (smooth)
        # gradient direction. Analytical gradient mode only.
        self.cbf_barrier_value_mode = str(rospy.get_param(
            "~cbf_barrier_value_mode", "soft")).lower()
        if self.cbf_barrier_value_mode not in ("soft", "hard"):
            raise ValueError("~cbf_barrier_value_mode must be 'soft' or 'hard'")
        if (self.cbf_barrier_value_mode == "hard"
                and self.cbf_multi_gradient_mode != "analytical"):
            rospy.logwarn(
                "cbf_barrier_value_mode 'hard' requires analytical gradients; "
                "falling back to 'soft'")
            self.cbf_barrier_value_mode = "soft"
        self._barrier_value_hard = self.cbf_barrier_value_mode == "hard"
        # Time-varying CBF term for NON-static scenes: the QP bound assumes
        # dh/dt = 0, so an obstacle approaching at v eats the whole braking
        # band regardless of what the robot does. Estimate the environment
        # rate d_env = hdot_measured - grad_h . dq_robot (model-free, GPU-only,
        # no sync), EMA-filter it, and tighten every active half-space by
        # clamp(-d_env - deadband, 0, max): the robot yields proportionally to
        # the approach speed while still outside d_safe. Static scenes give
        # d_env ~ 0 and the deadband zeroes the term.
        self.cbf_dynamic_hdot_enabled = _bool_param(
            "~cbf_dynamic_hdot_enabled", False)
        self.cbf_dynamic_hdot_tau = float(rospy.get_param(
            "~cbf_dynamic_hdot_tau", 0.1))
        self.cbf_dynamic_hdot_deadband = float(rospy.get_param(
            "~cbf_dynamic_hdot_deadband", 0.05))
        self.cbf_dynamic_hdot_max = float(rospy.get_param(
            "~cbf_dynamic_hdot_max", 0.5))
        # Estimator gates: ignore h jumps from selection/critical-link switches
        # (implausible rates) and don't estimate far from obstacles.
        self.cbf_dynamic_hdot_h_range = float(rospy.get_param(
            "~cbf_dynamic_hdot_h_range", 0.3))
        self.cbf_dynamic_hdot_jump = float(rospy.get_param(
            "~cbf_dynamic_hdot_jump", 1.5))
        if self.cbf_dynamic_hdot_enabled and (
                self.cbf_multi_gradient_mode != "analytical"
                or self.cbf_solver_mode not in ("fast_tangent", "multi_graphed")):
            rospy.logwarn(
                "cbf_dynamic_hdot disabled: needs analytical gradients and a "
                "graphed solver mode (per-link h/grad buffers)")
            self.cbf_dynamic_hdot_enabled = False
        rospy.loginfo(
            "CBF projection metric: %s (lambda=%.3g, rot_weight=%.3g, "
            "nullspace_clearance_gain=%.3g) | barrier value: %s | "
            "dynamic hdot: %s",
            self.cbf_projection_metric, self.cbf_task_metric_lambda,
            self.cbf_task_metric_rot_weight, self.cbf_nullspace_clearance_gain,
            self.cbf_barrier_value_mode,
            "on" if self.cbf_dynamic_hdot_enabled else "off")
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
        # Inner solver over the active half-spaces (multi_graphed loop).
        # "pocs" (legacy): cyclic projections converge to SOME feasible dq --
        # once a projection over-corrects, the excess is never given back, so
        # with several simultaneously active rows the output deviates more
        # from the nominal than the QP optimum requires, and depends on row
        # order. "dykstra": Dykstra's corrected cyclic projections (for
        # half-spaces equivalent to Hildreth's dual method) -- each row keeps
        # a correction-memory vector that is added back before its next
        # projection, which provably converges to the EXACT projection of the
        # nominal onto the intersection in the W metric (the true QP
        # solution): minimal deviation, order-independent limit, identical to
        # pocs whenever <= 1 constraint is active. Same _qp_step projections,
        # ~2 extra vector ops per row-step (graph-capturable, static shapes).
        # NOTE the per-step max_correction_speed clamp makes projections
        # inexact when it engages, so Dykstra's optimality is approximate
        # during hard-clamped corrections (still safe: feasibility logic and
        # the final repair are unchanged).
        # Refs: R. L. Dykstra, "An Algorithm for Restricted Least Squares
        # Regression," JASA 78(384), 1983; J. P. Boyle, R. L. Dykstra, "A
        # Method for Finding Projections onto the Intersection of Convex Sets
        # in Hilbert Spaces," 1986; C. Hildreth, "A Quadratic Programming
        # Procedure," Naval Res. Logist. Q. 4, 1957.
        self.cbf_qp_algorithm = str(rospy.get_param(
            "~cbf_qp_algorithm", "pocs")).strip().lower()
        if self.cbf_qp_algorithm not in ("pocs", "dykstra"):
            rospy.logwarn("Unknown ~cbf_qp_algorithm '%s', using 'pocs'",
                          self.cbf_qp_algorithm)
            self.cbf_qp_algorithm = "pocs"
        # Cross-cycle Dykstra warm start: carry the dual multipliers lambda_l
        # PER LINK IDENTITY across control cycles and rebuild the correction
        # memories in the CURRENT constraint geometry at the head of each
        # solve. The physical constellation persists across 100 Hz cycles, so
        # this accumulates sweeps across cycles: the ill-conditioned tail
        # (near-parallel gradients, rho -> 1) converges within a few cycles
        # instead of restarting from scratch (offline prototype:
        # scripts/solver_study.py -- carrying raw z vectors instead DIVERGES
        # under scene drift, the lambda form is the correct one). Off by
        # default until validated in sim; inert diff when false.
        self.cbf_dykstra_warmstart = bool(rospy.get_param(
            "~cbf_dykstra_warmstart", False))
        preprocess_max_points = int(rospy.get_param("~preprocess_max_points", 512))
        self.preprocess_max_points = (
            max(self.cbf_graph_points, preprocess_max_points)
            if preprocess_max_points > 0 else 0
        )

        # Setup the new fast CUDA graph
        self.setup_cuda_graph(batch_size=1, n_points=self.cbf_graph_points)
        self._log_cuda_memory("CBF after CUDA graph")
        
        # Arm joint names for /joint_states lookup and command ordering.
        # Default = panda; the xArm7 cross-embodiment launch passes xarm7_joint1..7.
        self.joint_names = list(rospy.get_param('~arm_joint_names', [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
            'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]))
        # Uncontrolled gripper joints, in the URDF order that follows the arm
        # joints. Their measured positions feed the q_pad2 FK tail; missing
        # names (e.g. on the xArm7) leave the tail at zero, the old behavior.
        self.finger_joint_names = list(rospy.get_param('~finger_joint_names', [
            'panda_finger_joint1', 'panda_finger_joint2'
        ]))
        self._finger_staging = None
        self._finger_uploaded = None
        self.current_q = None
        self.nominal_q = None
        self.target_x = None
        self.obs_points = torch.empty((0, 3), dtype=torch.float32, device=self.device)

        # Soft joint-limit barrier: per-joint velocity damping toward each
        # limit, dq_i in [kappa*(q_lo+m - q_i), kappa*(q_hi-m - q_i)],
        # applied to the FINAL commanded velocity (after the safety repair)
        # in both the graphed and eager paths. On the Panda this doubles as
        # SINGULARITY avoidance: elbow-straight is q4 at its upper limit and
        # wrist-aligned is q6 at its lower limit, so braking margin_rad away
        # from the limits keeps the arm out of both. Once past the margin
        # band the bound turns positive and pushes back inside, capped at
        # limit_push [rad/s]. margin 0 = disabled (legacy).
        self.cbf_joint_limit_margin = float(rospy.get_param(
            "~cbf_joint_limit_margin", 0.0))
        self.cbf_joint_limit_kappa = float(rospy.get_param(
            "~cbf_joint_limit_kappa", 3.0))
        self.cbf_joint_limit_push = float(rospy.get_param(
            "~cbf_joint_limit_push", 0.05))
        lim_lo, lim_hi = [-1e9] * 7, [1e9] * 7
        if self.cbf_joint_limit_margin > 0.0:
            try:
                import xml.etree.ElementTree as _ET
                _lims = {}
                for _j in _ET.parse(urdf_path).getroot().iter("joint"):
                    _l = _j.find("limit")
                    if _l is not None and "lower" in _l.attrib:
                        _lims[_j.attrib.get("name", "")] = (
                            float(_l.attrib["lower"]),
                            float(_l.attrib["upper"]))
                lim_lo = [_lims[n][0] for n in self.joint_names]
                lim_hi = [_lims[n][1] for n in self.joint_names]
                rospy.loginfo(
                    "CBF joint-limit barrier: margin %.2f rad, kappa %.1f "
                    "(lo %s, hi %s)", self.cbf_joint_limit_margin,
                    self.cbf_joint_limit_kappa,
                    ["%.2f" % v for v in lim_lo],
                    ["%.2f" % v for v in lim_hi])
            except Exception as exc:
                rospy.logwarn("joint-limit parse failed (%s): "
                              "limit barrier DISABLED", exc)
                self.cbf_joint_limit_margin = 0.0
        self._q_lim_lo = torch.tensor(
            lim_lo, dtype=torch.float32, device=self.device).view(1, 7)
        self._q_lim_hi = torch.tensor(
            lim_hi, dtype=torch.float32, device=self.device).view(1, 7)

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
        # Sampled-clearance escape DIRECTION (generalizes the fixed +z lift).
        # The lift always goes UP, which is wrong for tall/overhanging
        # obstacles. Probe mode instead samples ~num_dirs unit directions in
        # the tangent plane of the TCP-to-obstacle normal, virtually steps the
        # TCP by ~probe_radii along each, and scores: clearance against the
        # obstacle cloud (capped so far-field ties break on the other terms)
        # + goal alignment (nominal task velocity) + a commitment bonus
        # (direction hysteresis -- prevents side-flip limit cycles, the same
        # multimodal-commit failure the temporal ensembler fixes for the FM
        # policy) - a workspace-floor penalty (the table is NOT in the
        # obstacle cloud, so "down" would otherwise look free). Runs beside
        # the task metric in the preprocess thread; the 100 Hz graph
        # consumes the staged [1,7] joint row exactly as it consumed the +z
        # lift row, so the control-loop cost is unchanged.
        # cbf_escape_lift_gain remains the authority [rad/s]. The committed
        # direction FREEZES while the escape trigger is active so a maneuver
        # finishes on the side it started.
        self.cbf_escape_probe_enabled = _bool_param(
            "~cbf_escape_probe_enabled", False)
        self.cbf_escape_probe_num_dirs = max(4, int(rospy.get_param(
            "~cbf_escape_probe_num_dirs", 16)))
        self.cbf_escape_probe_radii = sorted(
            float(v) for v in str(rospy.get_param(
                "~cbf_escape_probe_radii", "0.05 0.10")).split())
        # Probe elevations [deg] toward the obstacle normal. 0 = the legacy
        # tangent ring. A concave pocket (crescent bite) has its ONLY exit
        # along +normal (back out through the mouth), which no tangent
        # direction can reach; extra rings tilted toward +n (plus the pure
        # retreat +n itself, always appended when any elevation > 0) put that
        # exit in the candidate set. Outward directions cost nothing in
        # safety (they increase h), and the 100 Hz consumer keeps their
        # outward component (it only strips the inward part).
        self.cbf_escape_probe_elevations = [
            float(v) for v in str(rospy.get_param(
                "~cbf_escape_probe_elevations", "0")).split()]
        # Clearance-GROWTH reward: a corridor (the crescent's vertical groove)
        # has clear(r_max) ~ clear(r_min) -- riding it never opens clearance;
        # a genuine exit has clearance increasing with radius. 0 = legacy
        # (min-over-radii clearance only).
        self.cbf_escape_probe_growth_weight = float(rospy.get_param(
            "~cbf_escape_probe_growth_weight", 0.0))
        self.cbf_escape_probe_clearance_cap = float(rospy.get_param(
            "~cbf_escape_probe_clearance_cap", 0.20))
        self.cbf_escape_probe_goal_weight = float(rospy.get_param(
            "~cbf_escape_probe_goal_weight", 0.02))
        self.cbf_escape_probe_commit_weight = float(rospy.get_param(
            "~cbf_escape_probe_commit_weight", 0.02))
        self.cbf_escape_probe_min_z = float(rospy.get_param(
            "~cbf_escape_probe_min_z", 0.03))
        self.cbf_escape_probe_ema = float(rospy.get_param(
            "~cbf_escape_probe_ema", 0.3))
        # Tabu re-vote: a greedy one-step probe cannot see past a concave
        # obstacle (crescent bite: clearance looks fine in directions that
        # lead nowhere). If a committed escape direction produces less than
        # ~tabu_min_progress of TCP motion along itself over ~tabu_timeout
        # seconds (the QP is clipping it: wedged), that direction is
        # penalized for ~tabu_ttl seconds and the vote re-opens. Progress is
        # measured as displacement along e, NOT as h improvement: a
        # SUCCESSFUL tangential escape slides at h~trigger the whole way and
        # h only rises once past the obstacle.
        self.cbf_escape_probe_tabu_timeout = float(rospy.get_param(
            "~cbf_escape_probe_tabu_timeout", 1.5))
        self.cbf_escape_probe_tabu_min_progress = float(rospy.get_param(
            "~cbf_escape_probe_tabu_min_progress", 0.03))
        self.cbf_escape_probe_tabu_weight = float(rospy.get_param(
            "~cbf_escape_probe_tabu_weight", 0.1))
        self.cbf_escape_probe_tabu_ttl = float(rospy.get_param(
            "~cbf_escape_probe_tabu_ttl", 8.0))
        # Re-entry tabu: the stall tabu never fires on a maneuver that
        # retreats fine but resolves nothing (runPP1001: retreat -> release
        # at h_release -> nominal dives back -> re-trigger, an identical
        # ~1.7 s pump forever, because the re-vote picks the same winner).
        # If the escape RE-triggers within this window [s] after a release,
        # the direction of the finished maneuver is tabu'd, so successive
        # pumps explore different directions (lateral, around the horns)
        # instead of repeating one. 0 = off (legacy).
        self.cbf_escape_probe_reentry_window = float(rospy.get_param(
            "~cbf_escape_probe_reentry_window", 0.0))
        # Resolution timeout [s]: tabu a committed direction that has been
        # held this long while the escape is STILL engaged, even if it makes
        # displacement progress. The stall tabu misses directions that move
        # but resolve nothing (runPP1002.3ag: "up" climbed a 1.0 m wall to
        # the kinematic ceiling and parked there 30+ s at h~0, because
        # climbing counts as progress and the conflict never cleared).
        # 0 = off.
        self.cbf_escape_probe_resolve_timeout = float(rospy.get_param(
            "~cbf_escape_probe_resolve_timeout", 0.0))
        self._probe_commit_e_t0 = None  # committed-direction hold start
        self._escape_was_active = False
        self._escape_release_t = None
        self._escape_release_e = None
        # Proportional escape gate EMA time constant [s]: how fast the escape
        # authority ramps in/out of conflicts (see _escape_trigger_update).
        self.cbf_escape_gate_tau = float(rospy.get_param(
            "~cbf_escape_gate_tau", 0.3))
        self._escape_gate = 0.0
        self._probe_tabu = []          # [(unit dir [3], expiry wall-time)]
        self._probe_commit_t0 = None   # progress-window start (wall-time)
        self._probe_commit_p0 = None   # TCP position at window start
        # While escape is active, cancel this fraction of the nominal's
        # INWARD (toward-obstacle) velocity component before the QP. The QP
        # alone only zeroes net inward motion at the barrier, so an escape
        # capped at cbf_escape_max_velocity can never out-push a nominal that
        # keeps demanding into a pocket: the arm pins at h ~ h_trigger
        # instead of retreating. 0 = legacy (escape fights the full nominal).
        self.cbf_escape_suppress_nominal = float(rospy.get_param(
            "~cbf_escape_suppress_nominal", 0.0))
        if self.cbf_escape_probe_enabled and self.cbf_escape_lift_gain <= 0.0:
            rospy.logwarn(
                "cbf_escape_probe_enabled ignored: needs "
                "cbf_escape_lift_gain > 0 (it sets the escape authority)")
            self.cbf_escape_probe_enabled = False
        self._escape_probe_e = None   # committed task-space direction [3]
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
        # Critical-point selection path.
        #
        # 'legacy': cdist link-centre prefilter -> chunked eager whole-body SDF
        #   -> boolean keep_mask (a D2H SYNC: masked indexing must read the
        #   survivor count to size its output) -> cat -> topk. Three passes
        #   over the cloud at dynamic shapes, so nothing can be graphed.
        #
        # 'graphed': ONE fixed-size CUDA-graph replay does the whole job --
        #   FK, whole-body SDF, self-filter, top-k, dummy fill. The cloud is
        #   padded to ~cbf_sdf_graph_points with far dummies, rejects are
        #   pushed to +inf instead of masked (same trick _prefilter_for_sdf
        #   already uses), so every shape is static and there is no sync.
        #   Measured on the RTX 3090 with the deployed 9-link PP model, real
        #   2000-point cloud: 3.82 -> 0.90 ms median, 6.31 -> 1.15 ms p99.
        #   Bit-exact vs the eager path over random q and cloud sizes.
        #
        # The prefilter it drops was pure waste in PP anyway: the cloud (2000)
        # never exceeded ~cbf_sdf_prefilter_max_points (4096), so the cdist was
        # computed and discarded every cycle.
        self.cbf_selection_mode = str(
            rospy.get_param("~cbf_selection_mode", "legacy")).lower()
        if self.cbf_selection_mode not in ("legacy", "graphed"):
            raise ValueError("~cbf_selection_mode must be 'legacy' or 'graphed'")
        # Fixed graph width. MUST cover the whole cloud or the broad phase
        # below has to pick which points survive. Observed clouds: feeding
        # /perception/persistent_obstacles 2573 max, PP 2000 (1000 obstacle +
        # 1000 moving sphere), so 4096 covers both with headroom and no
        # truncation. Cost scales with the padded width, not the real cloud:
        # 2048 -> 0.68 ms, 4096 -> 0.90 ms, 10000 -> 1.72 ms. Sizing it far
        # above the real cloud just buys dummy SDF evaluations.
        self.cbf_sdf_graph_points = max(
            1, int(rospy.get_param("~cbf_sdf_graph_points", 4096)))
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
        # self.q_pad2 created earlier (before setup_cuda_graph) and attached
        # to the evaluators; it now carries the measured finger positions.
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

        # Graphed selection state (see ~cbf_selection_mode). Built lazily on
        # the first preprocess tick, so capture happens on the preprocess
        # stream with the rest of the node already constructed.
        self._sel_graph = None
        self._sel_static_pts = None
        self._sel_static_q9 = None
        self._sel_out_obs = None
        self._sel_out_min_dist = None
        self._sel_out_n_real = None
        # Pinned landing pad for the real selected-point count (see
        # _preprocess_obstacles_graphed): written by an async D2H, read by the
        # control thread at adoption, when the pack event guarantees it has
        # already landed. Never read with .item() from the preprocess thread.
        self._sel_n_pin = torch.zeros(1, dtype=torch.long).pin_memory()
        self._sel_graph_failed = False
        self._sel_broad_phase_hits = 0
        self.debug_sdf_prefilter_counts = None

        # --- CUDA stream split: control loop vs preprocess -------------------
        # Both threads used to share the default stream, so every control-loop
        # D2H (h/command readback) waited behind the whole queued preprocess
        # batch (whole-body SDF over the protected links x up to 4k points),
        # and under FM/SAM GPU contention the process launch queue backed up
        # until even kernel ENQUEUE blocked (runPP100: cbf_ms p50 16 ms, loop
        # at ~33 Hz instead of 100). The control loop now runs on a
        # high-priority stream; preprocess enqueues on its own stream and
        # publishes results as an immutable (seq, event, refs...) pack that
        # the control loop adopts only once the event reports completion.
        # Adoption never blocks: until then the previous selection stays live
        # (adds <= 1 control cycle to the selection latency, which is already
        # ~33 ms by design; h itself is recomputed in-graph every cycle).
        # Default OFF: the feeding launches share this node and keep the
        # legacy single-stream behavior; the PP launch opts in explicitly.
        # NOTE: the grasp-cloud subscriber writes bernstein_core grasp buffers
        # from its own callback thread (default stream); with streams ON that
        # write is no longer stream-ordered against the graph replay. Keep
        # this off when cbf_grasp_box_enabled is true until that path stages
        # its updates like the task metric does.
        self._use_streams = _bool_param("~cbf_separate_streams", False)
        if self._use_streams:
            self._ctrl_stream = torch.cuda.Stream(priority=-1)
            self._prep_stream = torch.cuda.Stream()
        else:
            self._ctrl_stream = None
            self._prep_stream = None
        self._prep_done_event = torch.cuda.Event()
        self._prep_pack = None
        self._prep_seq = 0
        self._adopted_seq = 0
        self._prep_skips = 0
        # Task-metric results are staged as fresh CPU tensors by the preprocess
        # thread and copied H2D on the control stream at adoption, so the
        # graph-referenced static_Winv is never written concurrently with a
        # replay that reads it.
        self._staged_winv_cpu = None
        self._staged_N_cpu = None
        self._staged_Jz_cpu = None
        # Monotone counter bumped at every critical-point selection swap.
        # Appended to /cbf_safety/reflex_state so the reflex node's empirical
        # Lipschitz monitor can skip gradient pairs straddling a selection
        # update (a barrier-function change, not curvature). Kept within
        # float32 exact-integer range for the Float32MultiArray transport.
        self._selection_generation = 0
        self._adopted_winv_src = None
        self._adopted_N_src = None
        self._adopted_Jz_src = None
        self._task_metric_N_buf = None
        self._task_metric_Jz_buf = None
        # Control-thread view of the preprocess outputs (event-complete only).
        self._adopted_obs = self.selected_obs
        self._adopted_count = 0
        self._adopted_min_dist = float('inf')

        self.last_q_safe = None
        self.q_safe_work = torch.zeros((1, 7), device=self.device)
        self.dq_safe_work = torch.zeros((1, 7), device=self.device)
        # Callback staging: subscribers write NumPy only; the control loop
        # uploads the latest samples to the GPU once per cycle (see run()).
        self._joint_prev_np = None
        self._joint_prev_stamp = None
        self._dq_real_np = np.zeros(7, dtype=np.float32)
        self._joint_staging = None
        self._joint_uploaded_stamp = None
        # Stale-data watchdog: a barrier evaluated on frozen inputs certifies
        # nothing. Timeouts are generous multiples of the nominal rates
        # (joints ~1 kHz, obstacle cloud ~30 Hz); on staleness the loop
        # commands a zero-velocity hold instead of solving. `require_obstacles`
        # also gates /cbf_safety/ready on the first cloud; set it false for
        # obstacle-free ablations.
        self.watchdog_enabled = _bool_param("~watchdog_enabled", True)
        self.watchdog_joint_timeout = float(
            rospy.get_param("~watchdog_joint_timeout", 0.3))
        self.watchdog_obstacle_timeout = float(
            rospy.get_param("~watchdog_obstacle_timeout", 1.0))
        self.watchdog_require_obstacles = _bool_param(
            "~watchdog_require_obstacles", True)
        self.obs_stamp = None
        self._ready_announced = False
        self._joint_uploaded_np = None  # CPU copy of the q the solve used (reflex export)
        self._nominal_q_staging = None
        self._nominal_q_uploaded = None
        self._nominal_dq_staging = None
        self._nominal_dq_uploaded = None
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
        self.transfer_buffer = torch.zeros(80, dtype=torch.float32, device=self.device)
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
        # Barrier-state export for the optional 1 kHz reflex brake layer
        # (cbf_reflex_node.py). Layout per message:
        #   [t_solve, n_rows, q0(7), h(n), env_hdot(n), grad_h(n*7 row-major)]
        # where q0 is the measured q the solve used, so the consumer can
        # extrapolate h(t) ~= h + grad_h^T (q - q0) between QP solves.
        self.cbf_reflex_publish = _bool_param("~cbf_reflex_publish", False)
        self.reflex_state_pub = None
        if self.cbf_reflex_publish:
            self.reflex_state_pub = rospy.Publisher(
                '/cbf_safety/reflex_state', Float32MultiArray, queue_size=1)
        self.contact_event_pub = rospy.Publisher(
            '/cbf_safety/contact_event', Float32MultiArray, queue_size=1, latch=True)
        # ISSf assumption monitor: publishes per cycle the realized
        # disturbance of the kinematic plant (q_dot = u + d) projected on the
        # active barrier gradients -- the direction cbf_issf_epsilon actually
        # bounds. [t, eps_proj, ||d||, exceeded, running max]. The reflex
        # consumes `exceeded` as a monitored-assumption veto; the bags feed
        # the ch. 5 epsilon-validation panel. Piggybacks on the reflex-state
        # snapshot, so it requires cbf_reflex_publish.
        self.issf_monitor_enabled = _bool_param("~issf_monitor_enabled", True)
        self.issf_monitor_pub = None
        if self.issf_monitor_enabled and self.cbf_reflex_publish:
            self.issf_monitor_pub = rospy.Publisher(
                '/cbf_safety/issf_monitor', Float32MultiArray, queue_size=1)
            rospy.Subscriber('/cbf_reflex/status', Float32MultiArray,
                             self._reflex_status_cb, queue_size=1)
        self._issf_prev_cmd_np = None
        self._issf_monitor_max = 0.0
        self._reflex_alpha_seen = 1.0
        self.diag_pub = rospy.Publisher('/cbf_safety/diagnostics', Float32MultiArray, queue_size=10)
        # Estimated environment closing speed (max per-link dynamic-hdot
        # tightening, m/s; 0 in static scenes by deadband design). Consumed by
        # the planner's dynamic-scene ensembler gate.
        self.env_hdot_pub = rospy.Publisher('/cbf_safety/env_hdot', Float32, queue_size=1)
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
        
        # Capture the CUDA graph now that every parameter above is read (the
        # graphed control step bakes many of them in).
        self.finalize_cuda_graph()

        self.rate_hz = float(rospy.get_param("~rate_hz", 150.0))
        if self.rate_hz <= 0.0:
            raise ValueError("~rate_hz must be > 0")
        _dt_nom = 1.0 / self.rate_hz
        self._escape_gate_beta = _dt_nom / (
            _dt_nom + max(self.cbf_escape_gate_tau, 1e-3))
        self.rate = rospy.Rate(self.rate_hz)
        self.last_time = rospy.get_time()
        # Decide the selection path ONCE, at startup, so an unsupported config
        # is a loud line in the log rather than a silent per-tick fallback.
        if self.cbf_selection_mode == "graphed":
            unsupported = self._selection_graph_supported()
            if unsupported:
                self.cbf_selection_mode = "legacy"
                rospy.logwarn(
                    "~cbf_selection_mode:=graphed ignored -- this config is "
                    "not the policy the graph implements (%s). Using legacy.",
                    "; ".join(unsupported))
            else:
                # Capture HERE, for the same reason finalize_cuda_graph does:
                # before the timers below start, so no preprocess/control tick
                # can run concurrently with capture. q is read from the static
                # buffer on every replay, so capturing at q=0 is fine.
                try:
                    self._build_selection_graph(
                        torch.zeros((1, 9), device=self.device))
                except Exception as e:
                    self.cbf_selection_mode = "legacy"
                    self._sel_graph_failed = True
                    self._sel_graph = None
                    rospy.logerr(
                        "CBF selection graph capture failed (%s) -- using the "
                        "legacy path.", e)
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
            "selection_mode=%s sdf_graph_points=%d "
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
            self.cbf_selection_mode,
            self.cbf_sdf_graph_points,
            self.cbf_selection_sticky_points,
            self.cbf_selection_sticky_radius,
            self.cbf_selection_sticky_score_margin,
            self.cbf_selection_sticky_force_keep,
            self.cbf_integrate_from_current,
            self.cbf_command_dt,
            self.cbf_passthrough_when_inactive,
            self.publish_yellow_points,
            self.publish_debug_topics, self.profile_sync)
        # /cbf_safety/ready is announced from the control loop once verified
        # joint states (and, unless waived, a first obstacle cloud) have been
        # received -- not here at the end of initialization.

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
        # Same time-varying tightening the graphed solve applied to the
        # critical row: the repair / residual must re-validate against the
        # bound that was actually enforced, not the weaker static one.
        bound = torch.where(h > 0.0, outside_bound, inside_bound) \
            + self.cbf_constraint_margin + self.static_hdot_env_min \
            + self.static_sampled_margin
        if self._issf_on:
            # ISSf tightening of the critical row (see ~cbf_issf_epsilon).
            bound = bound + self.static_issf_eps * torch.norm(
                self.static_grad_h, dim=1)
        return bound

    def _constraint_residual(self, h, grad_h, dq):
        gdq = (grad_h * dq).sum(dim=1)
        if self.cbf_final_constraint_mode == "solver_bound":
            return gdq - self._robot_constraint_bound(h)

        kappa_eff = torch.where(
            h < 0.0,
            torch.full_like(h, self.cbf_recovery_kappa),
            torch.full_like(h, self.cbf_kappa),
        )
        res = (gdq + kappa_eff * h - self.cbf_constraint_margin
               - self.static_hdot_env_min - self.static_sampled_margin)
        if self._issf_on:
            res = res - self.static_issf_eps * torch.norm(grad_h, dim=1)
        return res

    def _diagnostic_active_constraints(self):
        if self.cbf_solver_mode == "multi_graphed":
            return self._graphed_active_constraints if self._adopted_count > 0 else 0
        if self.cbf_solver_mode == "multi_projected":
            return self._last_active_constraints
        return 1 if self._adopted_count > 0 else 0

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
        # Escape-trigger scalars [h, |dq_safe-dq_nom_cbf|, |grad_h|,
        # |dq_safe-dq_nom_RAW|]: async D2H into pinned memory, read one cycle
        # late via event query (no sync). The trigger's base_corrected uses
        # the RAW-nominal delta [3]: the composed delta [1] goes to ~0 as
        # soon as the escape makes the command safe, so gating on it made
        # the trigger reset itself the moment the escape worked (runPP1000:
        # ~29% duty cycle, 100+ sub-100ms activations, probe commit/tabu
        # never engaged). Against the raw nominal the delta includes the
        # escape itself, so the trigger stays latched until h > h_release.
        # Init = far-from-barrier so the escape stays off until real data.
        self._esc_pin = torch.tensor([1.0, 0.0, 1.0, 0.0]).pin_memory()
        self._esc_vals = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        self._esc_event = torch.cuda.Event()
        # PER-ROW environment hdot tightening (>= 0) for the time-varying CBF
        # bound: one slot per robot link (+1 grasp row), gathered per QP row
        # inside the graph (0 = static-scene legacy bound). Per-link is
        # essential: a scalar min-h estimator is blind to an approaching
        # obstacle while a DIFFERENT link pins the min (fork parked 0.2 mm
        # from the food), and a scalar tightening would shove that other link
        # (the fork) away from its legitimate near-contact. static_h_links /
        # static_grad_links are written by the graphed constraint closures so
        # the estimator can difference per link with stable row identity.
        n_hdot_rows = int(self.bernstein_core.K) + 1
        self.static_hdot_env_links = torch.zeros(n_hdot_rows, device=self.device)
        # Critical-row tightening, exported by the solve for the OUT-OF-GRAPH
        # final-constraint repair / residual diagnostics: the repair must
        # re-validate against the same time-varying bound the solver enforced,
        # or the output low-pass can eat the retreat without triggering repair.
        self.static_hdot_env_min = torch.zeros((), device=self.device)
        # Sampled-data margin m(dq) [m/s], 0-dim: every constraint consumer
        # (_qp_step, eager POCS bounds, repair/residual) adds it
        # unconditionally; it stays 0 when ~cbf_sampled_data_margin is off.
        # Written once per cycle — in-graph for the graphed control step,
        # eagerly in _run_velocity_cbf_solver otherwise. With per-link L
        # (~cbf_sampled_data_L_links) this buffer is the CRITICAL ROW's
        # margin (exported by _solve_multi like static_hdot_env_min, so the
        # repair/residual re-validate the enforced bound); the scalar-L
        # value used by the food row / single-row paths lives in
        # static_sampled_scalar, and the linearized v^2 in static_v2_lin.
        self.static_sampled_margin = torch.zeros((), device=self.device)
        self.static_sampled_scalar = torch.zeros((), device=self.device)
        self.static_v2_lin = torch.zeros((), device=self.device)
        # ISSf eps(v) [rad/s], 0-dim, initialized at the floor so paths
        # that never refresh it still charge eps0.
        self.static_issf_eps = torch.full(
            (), self.cbf_issf_epsilon, device=self.device)
        self.static_h_links = torch.ones(n_hdot_rows, device=self.device)
        self.static_grad_links = torch.zeros((n_hdot_rows, 7), device=self.device)
        # Reflex export staging: [h_rows | env_hdot_rows | grad_rows(flat)]
        # for the N robot links only. The food row is deliberately excluded:
        # braking on the target barrier would fight the intentional grasp
        # approach. One GPU buffer so the hot path pays 3 device copies plus
        # a single tiny D2H, and only when ~cbf_reflex_publish is set.
        self._reflex_n_rows = int(self.bernstein_core.K)
        self._reflex_gpu = torch.zeros(
            self._reflex_n_rows * 9, device=self.device)
        self._hdot_zero = torch.zeros((), device=self.device)
        self._hdot_ema = torch.zeros(n_hdot_rows, device=self.device)
        self._hdot_prev_h = None
        self._hdot_prev_q = None
        self.graph = None
        self._solve_fn = None
        if self.cbf_solver_mode not in ("fast_tangent", "multi_graphed"):
            self._log_cuda_memory("Multi-CBF runtime buffers ready")
            return

        # Capture scalars as Python floats so the CUDA graph bakes in the params.
        constraint_margin = self.cbf_constraint_margin
        issf_on = self._issf_on
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
                     recovery_speed, recovery_depth, max_correction_speed,
                     hdot_env, exact_denom=False, sampled_margin=None):
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
            # Time-varying CBF: grad_h . dq >= bound - dh_env/dt. hdot_env is
            # THIS row's clamp(-dh_env/dt - deadband, 0, max) >= 0, gathered
            # from static_hdot_env_links; 0 when cbf_dynamic_hdot is disabled
            # or this link's environment is static.
            bound = bound + constraint_margin + hdot_env \
                + (self.static_sampled_scalar if sampled_margin is None
                   else sampled_margin)
            if issf_on:
                # ISSf: certify the row for the TRUE velocity u + d,
                # ||d|| <= eps(v) (see ~cbf_issf_epsilon / ~cbf_issf_rho).
                bound = bound + self.static_issf_eps * torch.norm(
                    grad_h, dim=-1)
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
            # Legacy: +1e-4 regularization slightly under-corrects each
            # projection (POCS re-projects the residual on later sweeps, so
            # it converges anyway). Dykstra REQUIRES exact projections: the
            # correction memory re-applies the same step each sweep, so a
            # regularized (under-shooting) projection freezes short of the
            # half-space forever. The denom > 1e-8 gate below still routes
            # degenerate gradients away from the division.
            denom_reg = (torch.clamp(denom, min=1e-8) if exact_denom
                         else denom + 1e-4)
            dq_delta = correction.unsqueeze(-1) * w_grad / denom_reg.unsqueeze(-1)
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

        # Barrier value mode: with "hard", the QP-band VALUE is the exact
        # per-link hard-min (no Softmin crowding bias -- a dense cloud parked
        # near one link otherwise under-reports clearance by tau*log(M_eff)),
        # while the gradient stays the smooth Softmin blend. result.sdf is the
        # raw [B,K,M] per-point field, so the hard value is one masked min.
        value_mode_hard = self._barrier_value_hard

        def _link_h_values(result):
            if not value_mode_hard:
                return result.h
            sdf = result.sdf
            if analytical_point_mask is not None:
                sdf = sdf.masked_fill(
                    ~analytical_point_mask.unsqueeze(0), float("inf"))
            return sdf.min(dim=-1).values - self.analytical_barrier.d_safe

        def _export_link_rows(h_links, g_links):
            # Fixed-identity per-link h/grad snapshot for the dynamic-hdot
            # estimator (rows beyond the current row count keep their far
            # init and estimate 0).
            n_valid = h_links.shape[1]
            self.static_h_links[:n_valid].copy_(h_links[0])
            self.static_grad_links[:n_valid].copy_(g_links[0])

        def _analytical_constraints():
            result = self.analytical_barrier.evaluate(
                self.static_q,
                self.static_obs,
                point_mask=analytical_point_mask,
            )
            h_links = _link_h_values(result)
            g_links = result.grad_q
            _export_link_rows(h_links, g_links)
            if two_group:
                h_robot_links = h_links[:, :n_robot]
                robot_index = h_robot_links.argmin(dim=1)
                h_robot = h_robot_links.gather(
                    1, robot_index[:, None]).squeeze(1)
                g_robot = g_links[:, :n_robot, :].gather(
                    1,
                    robot_index[:, None, None].expand(-1, 1, 7),
                ).squeeze(1)
                env_robot = self.static_hdot_env_links.index_select(
                    0, robot_index)
                env_food = self.static_hdot_env_links[n_robot:n_robot + 1]
                return (h_robot, g_robot, env_robot,
                        h_links[:, n_robot], g_links[:, n_robot, :], env_food)

            link_index = h_links.argmin(dim=1)
            h = h_links.gather(1, link_index[:, None]).squeeze(1)
            gradient = g_links.gather(
                1, link_index[:, None, None].expand(-1, 1, 7)).squeeze(1)
            env = self.static_hdot_env_links.index_select(0, link_index)
            return h, gradient, env

        # multi_graphed: fixed-shape topk(K_max) selection + a fixed POCS loop,
        # so the entire multi-constraint solve is CUDA-graph-capturable like
        # fast_tangent (no dynamic boolean masking, no GPU->CPU syncs). The proven
        # graph-safe _qp_step is reused; inactive (far) links no-op inside it
        # (active = h <= h_activate), so padding the set to a fixed K_max is safe.
        K_max = max(1, min(self.cbf_active_constraints, n_robot))
        if self.cbf_solver_mode == "multi_graphed":
            self._last_active_constraints = K_max
        # Dykstra correction memory: one row per constraint (row 0 = food,
        # 1..K_max = robot rows), zeroed at the start of every solve. Lives
        # outside the closure so the CUDA graph captures a stable pointer.
        use_dykstra = (self.cbf_qp_algorithm == "dykstra")
        dykstra_z = torch.zeros((K_max + 1, 7), device=self.device)
        # Warm-start state (see ~cbf_dykstra_warmstart): one multiplier per
        # ROBOT LINK (topk reshuffles rows every cycle, so row-indexed state
        # would attach to the wrong physical constraint) + one for the food
        # group. Zero-then-scatter each cycle: a link that drops out of the
        # top-K loses its multiplier; only continuously selected constraints
        # accumulate sweeps.
        warm_dykstra = use_dykstra and self.cbf_dykstra_warmstart
        dykstra_lam_links = torch.zeros(max(n_robot, 1), device=self.device)
        dykstra_lam_food = torch.zeros(1, device=self.device)
        # Row export for the all-rows final repair (~cbf_repair_all_rows):
        # h / grad / env of every robot solve row, frozen from this cycle's
        # evaluate, so the repair can re-project the post-filter command
        # onto the SAME constraint set the solve enforced. Persistent self
        # buffers: the repair closure lives in finalize_cuda_graph and both
        # sides are captured in the same CUDA graph.
        self._repair_K = K_max
        self.static_repair_h = torch.zeros((K_max,), device=self.device)
        self.static_repair_g = torch.zeros((K_max, 7), device=self.device)
        self.static_repair_env = torch.zeros((K_max,), device=self.device)
        self.static_repair_sampled = torch.zeros((K_max,), device=self.device)
        self._repair_dykstra_z = torch.zeros((K_max, 7), device=self.device)
        # Per-row sampled-margin coefficients 0.5 * L_l * T, gathered by link
        # identity in _solve_multi (~cbf_sampled_data_L_links). Uniform
        # scalar-L values when unset/malformed (legacy behavior).
        _coeffs = [self._sampled_margin_coeff] * n_robot
        self._sampled_per_row_L = False
        if (self.cbf_sampled_data_margin
                and self.cbf_sampled_data_L_links is not None):
            if len(self.cbf_sampled_data_L_links) == n_robot:
                _coeffs = [0.5 * l * self._sampled_margin_T
                           for l in self.cbf_sampled_data_L_links]
                self._sampled_per_row_L = True
            else:
                rospy.logwarn(
                    "cbf_sampled_data_L_links has %d values but %d protected "
                    "links; falling back to scalar L=%.1f",
                    len(self.cbf_sampled_data_L_links), n_robot,
                    self.cbf_sampled_data_L)
        self._sampled_coeff_links = torch.tensor(
            _coeffs, dtype=torch.float32, device=self.device)

        def _multi_constraints():
            result = self.analytical_barrier.evaluate(
                self.static_q, self.static_obs, point_mask=analytical_point_mask)
            h_links = _link_h_values(result)  # [B, n_robot(+1 food)]
            g_links = result.grad_q           # [B, n_robot(+1), 7]
            _export_link_rows(h_links, g_links)
            h_robot = h_links[:, :n_robot]
            g_robot = g_links[:, :n_robot, :]
            h_vals, idx = torch.topk(h_robot, k=K_max, dim=1, largest=False)
            g_vals = g_robot.gather(1, idx[:, :, None].expand(-1, -1, 7))
            # Per-row env tightening, gathered by LINK identity (idx), so an
            # approaching obstacle tightens only the link it approaches.
            env_vals = self.static_hdot_env_links[:n_robot].index_select(
                0, idx[0])                    # [K_max] (batch is 1)
            if two_group:
                return (h_vals, g_vals, env_vals, h_links[:, n_robot],
                        g_links[:, n_robot, :],
                        self.static_hdot_env_links[n_robot:n_robot + 1], idx)
            return h_vals, g_vals, env_vals, None, None, None, idx

        def _solve_multi(dq_nom):
            (h_vals, g_vals, env_vals, h_f, g_f, env_f,
             idx) = _multi_constraints()
            # Per-row sampled margin: (L_link/2)*T*v_lin^2, gathered by the
            # same link identity as env_vals. Falls back to the scalar-L
            # value (broadcast view) when per-link L is unset.
            if self._sampled_per_row_L:
                sampled_rows = torch.clamp(
                    self._sampled_coeff_links.index_select(0, idx[0])
                    * self.static_v2_lin,
                    max=self.cbf_sampled_data_margin_max)
            else:
                sampled_rows = self.static_sampled_scalar.expand(K_max)
            if self.cbf_repair_all_rows:
                self.static_repair_h.copy_(h_vals[0])
                self.static_repair_g.copy_(g_vals[0])
                self.static_repair_env.copy_(env_vals)
                self.static_repair_sampled.copy_(sampled_rows)
            if two_group:
                self.static_h_food.copy_(torch.clamp(h_f, max=0.25))
            dq = dq_nom
            if use_dykstra:
                # Dykstra / Hildreth: add back this row's previous correction
                # before re-projecting, store the new one. Converges to the
                # exact min-W-norm projection onto the intersection (see
                # ~cbf_qp_algorithm); reduces to plain POCS when z stays 0,
                # i.e. whenever <= 1 constraint is active.
                if warm_dykstra:
                    # z_l = -lambda_l * W^-1 g_l, rebuilt in the CURRENT
                    # geometry from the carried per-link multipliers.
                    wg_rows = g_vals[0] @ self.static_Winv          # [K,7]
                    lam_rows = dykstra_lam_links.index_select(0, idx[0])
                    dykstra_z[1:].copy_(-lam_rows.unsqueeze(1) * wg_rows)
                    if two_group:
                        dykstra_z[0].copy_(
                            -dykstra_lam_food * (g_f[0] @ self.static_Winv))
                    else:
                        dykstra_z[0].zero_()
                    # The dual state is TWO things: the memories z_l AND the
                    # primal they imply, dq = dq_nom + sum_l lambda_l W^-1 g_l
                    # = dq_nom - sum_l z_l. Injecting z without shifting the
                    # primal makes the first projection see dq_nom - lambda_l
                    # W^-1 g_l (the nominal pushed BACKWARDS along grad_h);
                    # an inactive row then passes that offset straight
                    # through, and the carried multipliers silently cancel
                    # whatever pushes toward the barrier -- including the
                    # clearance drive and the escape. That is the diverging
                    # "naive z" arm of scripts/solver_study.py.
                    dq = dq - dykstra_z.sum(dim=0, keepdim=True)
                else:
                    dykstra_z.zero_()
                for _ in range(constraint_sweeps):
                    if two_group:
                        dq_tmp = dq + dykstra_z[0]
                        dq = _qp_step(h_f, g_f, dq_tmp, *food_band, env_f,
                                      exact_denom=True)
                        dykstra_z[0].copy_((dq_tmp - dq)[0])
                    for k in range(K_max):                # robot links (last)
                        dq_tmp = dq + dykstra_z[k + 1]
                        dq = _qp_step(h_vals[:, k], g_vals[:, k, :], dq_tmp,
                                      *robot_band, env_vals[k],
                                      exact_denom=True,
                                      sampled_margin=sampled_rows[k])
                        dykstra_z[k + 1].copy_((dq_tmp - dq)[0])
                if warm_dykstra:
                    # lambda_l = -(g_l . z_l) / (g_l . W^-1 g_l) >= 0, stored
                    # by link identity for the next cycle.
                    wg_rows = g_vals[0] @ self.static_Winv
                    denom = (g_vals[0] * wg_rows).sum(dim=-1).clamp(min=1e-8)
                    lam_new = torch.clamp(
                        -(g_vals[0] * dykstra_z[1:]).sum(dim=-1) / denom,
                        min=0.0)
                    dykstra_lam_links.zero_()
                    dykstra_lam_links.scatter_(0, idx[0], lam_new)
                    if two_group:
                        gf = g_f[0]
                        wgf = gf @ self.static_Winv
                        dykstra_lam_food.copy_(torch.clamp(
                            -(gf * dykstra_z[0]).sum()
                            / (gf * wgf).sum().clamp(min=1e-8),
                            min=0.0).reshape(1))
            else:
                for _ in range(constraint_sweeps):
                    if two_group:
                        dq = _qp_step(h_f, g_f, dq, *food_band, env_f)  # target half-space
                    for k in range(K_max):                          # robot links (last)
                        dq = _qp_step(h_vals[:, k], g_vals[:, k, :], dq,
                                      *robot_band, env_vals[k],
                                      sampled_margin=sampled_rows[k])
            # Leave static_h / static_grad_h / static_hdot_env_min on the
            # most-critical robot link for the diagnostics / repair (the loop
            # ends on the least-critical row).
            min_k = h_vals.argmin(dim=1)
            self.static_h.copy_(h_vals.gather(1, min_k[:, None]).squeeze(1))
            self.static_grad_h.copy_(
                g_vals.gather(1, min_k[:, None, None].expand(-1, 1, 7)).squeeze(1))
            self.static_hdot_env_min.copy_(
                env_vals.index_select(0, min_k).squeeze(0))
            if self._sampled_per_row_L:
                # Critical row's margin for the repair/residual bound.
                self.static_sampled_margin.copy_(
                    sampled_rows.index_select(0, min_k).squeeze(0))
            return dq

        def _solve(dq_nom):
            if self.cbf_solver_mode == "multi_graphed":
                return _solve_multi(dq_nom)
            if two_group:
                if self.cbf_multi_gradient_mode == "analytical":
                    h_r, g_r, env_r, h_f, g_f, env_f = _analytical_constraints()
                else:
                    h_r, g_r, h_f, g_f = self.barrier.forward_two_group(
                        self.static_q, self.static_obs, n_robot,
                        n_obs_robot=self.n_obs_robot_split)
                    env_r = env_f = self._hdot_zero
                # Retain the target barrier for diagnostics before _qp_step
                # overwrites static_h with each group's value in turn. Capped at
                # 0.25 for the diagnostic only (the QP still uses the true h_f
                # below); far-from-obstacle spikes otherwise squash the plot.
                self.static_h_food.copy_(torch.clamp(h_f, max=0.25))
                dq = dq_nom
                for _ in range(constraint_sweeps):
                    dq = _qp_step(h_f, g_f, dq, *food_band, env_f)   # target half-space
                    dq = _qp_step(h_r, g_r, dq, *robot_band, env_r)  # fork+robot (last)
                self.static_grad_h.copy_(g_r)
                self.static_hdot_env_min.copy_(env_r.reshape(-1)[0])
                return dq
            if self.cbf_multi_gradient_mode == "analytical":
                h, g, env = _analytical_constraints()
            else:
                h, g, _ = self.barrier(self.static_q, self.static_obs)
                env = self._hdot_zero
            self.static_grad_h.copy_(g)
            self.static_hdot_env_min.copy_(env.reshape(-1)[0])
            return _qp_step(h, g, dq_nom, *robot_band, env)

        # Capture is DEFERRED to finalize_cuda_graph(): it runs at the end of
        # __init__, once every parameter is read, so the graph can optionally
        # swallow the whole control step (escape/clearance compose, dynamic
        # hdot, repair, integration, command fill) and not just the solve.
        self._solve_fn = _solve
        self._graph_n_points = n_points
        self.graph = None

    def finalize_cuda_graph(self):
        """Warm up and capture the CUDA graph (called after all params).

        Legacy mode captures exactly the solve, as before. With
        ~cbf_graphed_control_step (and a compatible configuration), the graph
        instead captures the ENTIRE per-cycle correction+command pipeline:
        the 100 Hz loop then replays ONE graph instead of dispatching ~90
        eager ops, each of which pays GIL/dispatcher contention against the
        preprocess thread and the SAM/FM processes (runPP102: ~80 us/op,
        cbf_correction p50 16 ms with the GPU idle at readback)."""
        if self._solve_fn is None:
            self._graph_control_step = False
            return
        _solve = self._solve_fn

        self._graph_control_step = (
            _bool_param("~cbf_graphed_control_step", False)
            and self.enable_cbf
            and not self.cbf_monitor_only
            and self.cbf_integrate_from_current
            and self.cbf_command_dt > 0.0
            and not self.publish_controller_command
            and self.cbf_enforce_final_constraint
        )
        if _bool_param("~cbf_graphed_control_step", False) \
                and not self._graph_control_step:
            rospy.logwarn(
                "cbf_graphed_control_step requested but the configuration "
                "is incompatible (needs enable_cbf, integrated velocity "
                "mode with cbf_command_dt>0, final-constraint repair, no "
                "monitor-only / position-controller mode); falling back to "
                "the eager control step.")
        # All-rows repair needs the exported multi_graphed solve rows AND the
        # graphed control step (the eager control loop keeps its legacy
        # single-row repair). Decided here, baked into the capture below.
        self._repair_all_rows_active = (
            self.cbf_repair_all_rows
            and self.cbf_solver_mode == "multi_graphed"
            and self._graph_control_step)
        if self.cbf_repair_all_rows and not self._repair_all_rows_active:
            rospy.logwarn(
                "cbf_repair_all_rows requested but unavailable (needs "
                "cbf_solver_mode multi_graphed + graphed control step); "
                "keeping the legacy critical-row repair.")

        if not self._graph_control_step:
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
            self.graph = torch.cuda.CUDAGraph()
            # thread_local: capture now happens at the END of __init__, when
            # the ROS subscribers are already live; the default (global)
            # capture mode invalidates the capture if ANY thread touches
            # CUDA meanwhile (obs_callback H2D killed the node here).
            with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
                if self.static_q.grad is not None: self.static_q.grad.zero_()
                self.static_dq_safe.copy_(_solve(self.static_dq_nom))
            print(
                f"✅ Graphe CUDA capturé ({self._graph_n_points} points, "
                f"gradient={self.cbf_multi_gradient_mode}) !")
            self._log_cuda_memory("CBF setup_cuda_graph captured")
            return

        # ── Graphed control step ─────────────────────────────────────────────
        # Static inputs filled per cycle before replay. static_dq_nom now
        # carries the BASE nominal (pre escape/clearance); the graph composes
        # the extra terms itself from the previous cycle's h/grad_h, exactly
        # like the eager path did.
        dev = self.device
        self.static_nominal_q = torch.zeros((1, 7), device=dev)
        self.static_dq_nom_cbf = torch.zeros((1, 7), device=dev)
        self.static_dq_escape_out = torch.zeros((1, 7), device=dev)
        # Per-cycle scalars: [0] escape flag, [1] velocity-filter gamma,
        # [2] 1/dt, [3] dynamic-hdot gamma, [4] escape h bias,
        # [5] measured joint speed ||dq_real|| (executed linearization).
        self.static_scalars = torch.zeros(8, device=dev)
        self._scalar_pin = torch.zeros(8, dtype=torch.float32).pin_memory()
        self._scalar_np = self._scalar_pin.numpy()
        self._graph_dq_filtered = torch.zeros((1, 7), device=dev)
        if self._task_metric_N_buf is None:
            self._task_metric_N_buf = torch.zeros((7, 7), device=dev)
        if self._task_metric_Jz_buf is None:
            self._task_metric_Jz_buf = torch.zeros((1, 7), device=dev)
        self._hdot_prev_h = self.static_h_links.clone()
        self._hdot_prev_q = torch.zeros(7, device=dev)
        # last_q_safe must be a FIXED buffer (the graph writes it in place).
        self.last_q_safe = torch.zeros((1, 7), device=dev)
        self._last_q_safe_seeded = False

        mjv = self.max_joint_velocity
        vel_gamma_on = self.cbf_velocity_filter_tau > 0.0

        def _control_extra():
            """Escape + null-space clearance from the PREVIOUS cycle's
            h/grad_h (same one-cycle-stale semantics as the eager path); the
            escape trigger stays on the CPU and arrives as scalars[0]."""
            sc = self.static_scalars
            g_prev = self.static_grad_h
            h_prev = self.static_h
            g_norm = torch.norm(g_prev, dim=1, keepdim=True)
            extra = torch.zeros_like(self.static_dq_nom)
            if self.cbf_nullspace_clearance_gain > 0.0:
                n_g = g_prev @ self._task_metric_N_buf
                n_norm = torch.norm(n_g, dim=1, keepdim=True)
                h_act = max(self.cbf_nullspace_clearance_h_activate, 1e-6)
                proximity = torch.clamp(
                    (h_act - h_prev.view(-1, 1)) / h_act, min=0.0, max=1.0)
                align = torch.clamp(n_norm / (0.3 * g_norm + 1e-6), max=1.0)
                gate = (g_norm > 1e-3).float()
                extra = extra + (self.cbf_nullspace_clearance_gain * proximity
                                 * align * gate * n_g / (n_norm + 1e-6))
            if self.cbf_escape_enabled:
                normal_dir = g_prev / (g_norm + 1e-6)
                normal_speed = (self.static_dq_nom * normal_dir).sum(
                    dim=1, keepdim=True)
                dq_esc = self.cbf_escape_tangent_gain * (
                    self.static_dq_nom - normal_speed * normal_dir)
                if self.cbf_escape_lift_gain > 0.0:
                    lift = self._task_metric_Jz_buf
                    # Strip only the INWARD component: an outward (grad_h-
                    # aligned) escape row increases h and must survive, or a
                    # retreat direction voted by the hemisphere probe (the
                    # only exit from a concave pocket) is deleted right here.
                    lift_n = (lift * normal_dir).sum(dim=1, keepdim=True)
                    lift_t = lift - torch.clamp(lift_n, max=0.0) * normal_dir
                    lift_norm = torch.norm(lift_t, dim=1, keepdim=True)
                    ns_frac = (torch.norm(g_prev @ self._task_metric_N_buf,
                                          dim=1, keepdim=True)
                               / (g_norm + 1e-6))
                    lift_gate = torch.clamp(1.0 - ns_frac / 0.3,
                                            min=0.0, max=1.0)
                    dq_esc = dq_esc + torch.where(
                        lift_norm > 1e-3,
                        self.cbf_escape_lift_gain * lift_gate * lift_t
                        / (lift_norm + 1e-6),
                        torch.zeros_like(lift_t))
                if self.cbf_escape_normal_gain > 0.0:
                    h_trig = max(abs(self.cbf_escape_h_trigger), 1e-6)
                    normal_scale = torch.clamp(
                        (self.cbf_escape_normal_h_trigger
                         - (h_prev + sc[4])).view(-1, 1) / h_trig,
                        min=0.0, max=1.5)
                    dq_esc = dq_esc + (self.cbf_escape_normal_gain
                                       * normal_scale * normal_dir)
                if self.cbf_escape_max_velocity > 0.0:
                    en = torch.norm(dq_esc, dim=1, keepdim=True)
                    dq_esc = dq_esc * torch.clamp(
                        self.cbf_escape_max_velocity / (en + 1e-6), max=1.0)
                if self.cbf_escape_suppress_nominal > 0.0:
                    # Cancel the nominal's inward push AFTER the authority
                    # clamp: it is a cancellation, not extra escape budget.
                    # Without it the nominal out-pushes the capped escape and
                    # the QP pins the arm at h ~ h_trigger instead of letting
                    # it retreat.
                    dq_esc = dq_esc - (self.cbf_escape_suppress_nominal
                                       * torch.clamp(normal_speed, max=0.0)
                                       * normal_dir)
                extra = extra + sc[0] * dq_esc
            return extra

        def _control_step():
            sc = self.static_scalars
            if self.static_q.grad is not None:
                self.static_q.grad.zero_()
            extra = _control_extra()
            self.static_dq_escape_out.copy_(extra)
            dq_nom_cbf = torch.clamp(self.static_dq_nom + extra,
                                     min=-mjv, max=mjv)
            self.static_dq_nom_cbf.copy_(dq_nom_cbf)
            # Sampled-data margin m(dq) = (L/2) T ||dq||^2 and ISSf eps(v).
            # Linearization v^2: legacy "command" = max(previous published
            # command (dq_safe_work still holds last cycle's final value
            # here), current nominal); "executed" additionally caps it by
            # (||dq_measured|| + headroom)^2 — min of two valid upper
            # bounds, breaking the repair->margin feedback loop (see
            # ~cbf_sampled_data_linearization). Runs IN-GRAPH: pure tensor
            # ops on persistent buffers, no sync. sc[5] = ||dq_real||
            # staged per cycle by the control loop.
            if self.cbf_sampled_data_margin or self._issf_on:
                v2 = torch.maximum(
                    (self.dq_safe_work * self.dq_safe_work).sum(),
                    (dq_nom_cbf * dq_nom_cbf).sum())
                if self.cbf_sampled_data_linearization == "executed":
                    v_exec = sc[5] + self.cbf_sampled_data_vel_headroom
                    v2 = torch.minimum(v2, v_exec * v_exec)
                self.static_v2_lin.copy_(v2)
                if self.cbf_sampled_data_margin:
                    m_scalar = torch.clamp(
                        self._sampled_margin_coeff * v2,
                        max=self.cbf_sampled_data_margin_max)
                    self.static_sampled_scalar.copy_(m_scalar)
                    # Critical-row export default; _solve_multi overwrites
                    # it with the per-row value when per-link L is active.
                    self.static_sampled_margin.copy_(m_scalar)
                if self._issf_on:
                    # eps(v) at the MEASURED speed (executed mode): the
                    # bound was identified by binning the mismatch against
                    # the measured speed of the same interval, so ε(v_meas)
                    # IS the identified relationship; the headroom belongs
                    # only to the curvature-dip term above (a future speed
                    # increase steepens the path, but the disturbance
                    # statistics already contain within-hold acceleration).
                    v_issf = (sc[5] if self.cbf_sampled_data_linearization
                              == "executed" else torch.sqrt(v2))
                    self.static_issf_eps.copy_(
                        self.cbf_issf_epsilon
                        + self.cbf_issf_rho * v_issf)
            # Solve (writes static_h / static_grad_h / static_constr fresh).
            dq = _solve(dq_nom_cbf)
            self.static_dq_safe.copy_(dq)
            self.dq_safe_work.copy_(dq)
            # Dynamic-hdot estimator (identical math to _update_dynamic_hdot,
            # gammas arrive as scalars; runs AFTER the solve like the eager
            # path, so the solve used the previous cycle's tightening).
            if self.cbf_dynamic_hdot_enabled:
                h_now_l = self.static_h_links
                dh = (h_now_l - self._hdot_prev_h) * sc[2]
                robot_part = (self.static_grad_links
                              @ (self.static_q.view(-1) - self._hdot_prev_q)
                              ) * sc[2]
                env_raw = dh - robot_part
                valid = (h_now_l < self.cbf_dynamic_hdot_h_range) & (
                    env_raw.abs() < self.cbf_dynamic_hdot_jump)
                self._hdot_ema.copy_(torch.where(
                    valid,
                    self._hdot_ema + sc[3] * (env_raw - self._hdot_ema),
                    torch.zeros_like(env_raw)))
                self.static_hdot_env_links.copy_(torch.clamp(
                    -self._hdot_ema - self.cbf_dynamic_hdot_deadband,
                    min=0.0, max=self.cbf_dynamic_hdot_max))
                self._hdot_prev_h.copy_(h_now_l)
                self._hdot_prev_q.copy_(self.static_q.view(-1))
            # Clamp + output low-pass (gamma_v in scalars[1]).
            self.dq_safe_work.clamp_(min=-mjv, max=mjv)
            if vel_gamma_on:
                self._graph_dq_filtered.copy_(
                    self._graph_dq_filtered
                    + sc[1] * (self.dq_safe_work - self._graph_dq_filtered))
                self.dq_safe_work.copy_(self._graph_dq_filtered)
            self.transfer_buffer[72:79].copy_(self.dq_safe_work.squeeze(0))
            # Final-constraint repair: exact port of the eager block.
            grad_h = self.static_grad_h
            grad_norm_sq = (grad_h ** 2).sum(dim=1, keepdim=True)
            h_final = self.static_h
            final_constr = self._constraint_residual(
                h_final, grad_h, self.dq_safe_work)
            final_active = h_final <= self.cbf_h_activate
            if self._repair_all_rows_active:
                # All-rows repair (~cbf_repair_all_rows): Dykstra sweeps of
                # exact projections onto EVERY exported solve row (gradients
                # frozen from this cycle's evaluate), so the post-filter
                # command returns to the intersection the solve certified.
                # The critical-row projection alone can push INTO a second
                # active row when gradients oppose (runPP1022: link4/link5,
                # cos = -0.30 -> alpha=0 reflex freeze at h = -0.2 mm).
                rg = self.static_repair_g                       # [K,7]
                rh = self.static_repair_h                       # [K]
                r_out = -self.cbf_max_inward_speed * torch.clamp(
                    rh / max(self.cbf_h_activate, 1e-6), min=0.0, max=1.0)
                r_in = self.cbf_recovery_speed * torch.clamp(
                    (-rh) / max(self.cbf_recovery_depth, 1e-6),
                    min=0.0, max=1.0)
                r_bound = torch.where(rh > 0.0, r_out, r_in) \
                    + self.cbf_constraint_margin + self.static_repair_env \
                    + self.static_repair_sampled
                r_gnorm = torch.norm(rg, dim=1)
                if self._issf_on:
                    r_bound = r_bound + self.static_issf_eps * r_gnorm
                # Degenerate-gradient gate. Dykstra needs exact projections,
                # so its gate is a binary row exclusion at grad_min; the
                # POCS branch keeps the legacy damped ramp (converges, just
                # slower on half-trusted rows).
                repair_dykstra = (self.cbf_repair_qp_algorithm == "dykstra")
                if self.cbf_repair_grad_min > 0.0:
                    if repair_dykstra:
                        r_gate = (r_gnorm
                                  >= self.cbf_repair_grad_min).float()
                    else:
                        r_gate = torch.clamp(
                            (r_gnorm - self.cbf_repair_grad_min)
                            / (2.0 * self.cbf_repair_grad_min + 1e-9),
                            min=0.0, max=1.0)
                else:
                    r_gate = torch.ones_like(r_gnorm)
                r_scale = (rh <= self.cbf_h_activate).float() * r_gate
                dq_before = self.dq_safe_work.view(-1)
                needs_repair = (
                    ((rg @ dq_before) < r_bound - 1e-9)
                    & (rh <= self.cbf_h_activate)
                    & (r_gnorm > 1e-4)).any().view(1, 1)
                w_rg = rg @ self.static_Winv                    # [K,7]
                r_denom = torch.clamp((rg * w_rg).sum(dim=1), min=1e-8)
                dq_r = dq_before
                n_sweeps = (self.cbf_repair_sweeps
                            if self.cbf_repair_sweeps > 0
                            else self.cbf_constraint_sweeps)
                if repair_dykstra:
                    # Dykstra: exact projections (binary gate above) +
                    # correction memory -> min-W-norm point of the
                    # intersection (~cbf_repair_qp_algorithm).
                    self._repair_dykstra_z.zero_()
                    for _s in range(n_sweeps):
                        for k in range(self._repair_K):
                            dq_tmp = dq_r + self._repair_dykstra_z[k]
                            corr = torch.clamp(
                                r_bound[k] - (rg[k] * dq_tmp).sum(),
                                min=0.0)
                            dq_r = dq_tmp + (r_scale[k] * corr
                                             / r_denom[k]) * w_rg[k]
                            self._repair_dykstra_z[k].copy_(dq_tmp - dq_r)
                else:
                    # POCS: plain cyclic projections, damped-ramp-tolerant.
                    for _s in range(n_sweeps):
                        for k in range(self._repair_K):
                            corr = torch.clamp(
                                r_bound[k] - (rg[k] * dq_r).sum(), min=0.0)
                            dq_r = dq_r + (r_scale[k] * corr
                                           / r_denom[k]) * w_rg[k]
                repair_delta = (dq_r - dq_before).view(1, 7)
            else:
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
                w_grad_final = grad_h @ self.static_Winv
                w_denom_final = (grad_h * w_grad_final).sum(dim=1, keepdim=True)
                repair_delta = repair_step * w_grad_final / (w_denom_final + 1e-6)
            if (self.cbf_max_correction_speed > 0.0
                    or self.cbf_recovery_max_correction_speed > 0.0):
                rd_norm = torch.norm(repair_delta, dim=1, keepdim=True)
                if self.cbf_max_correction_speed > 0.0:
                    repair_limit = torch.full_like(
                        rd_norm, self.cbf_max_correction_speed)
                else:
                    repair_limit = torch.full_like(rd_norm, float("inf"))
                if self.cbf_recovery_max_correction_speed > 0.0:
                    repair_limit = torch.where(
                        h_final.view(-1, 1) < 0.0,
                        torch.full_like(
                            rd_norm, self.cbf_recovery_max_correction_speed),
                        repair_limit)
                repair_delta = repair_delta * torch.clamp(
                    repair_limit / (rd_norm + 1e-9), max=1.0)
            if not self._repair_all_rows_active \
                    and self.cbf_repair_grad_min > 0.0:
                grad_norm = torch.sqrt(grad_norm_sq)
                grad_gate = torch.clamp(
                    (grad_norm - self.cbf_repair_grad_min)
                    / (2.0 * self.cbf_repair_grad_min + 1e-9),
                    min=0.0, max=1.0)
                repair_delta = repair_delta * grad_gate
            dq_repaired = (self.dq_safe_work + repair_delta).clamp(
                min=-mjv, max=mjv)
            self.dq_safe_work.copy_(torch.where(
                needs_repair, dq_repaired, self.dq_safe_work))
            # Joint-limit barrier LAST so nothing downstream re-opens it;
            # the recomputed final_constr below reports the clamped command.
            self._joint_limit_clamp_(self.dq_safe_work)
            final_constr = self._constraint_residual(
                h_final, grad_h, self.dq_safe_work)
            if self._repair_all_rows_active:
                # Worst residual over EVERY exported solve row, evaluated on
                # the FINAL command (post filter, saturation, repair and
                # joint-limit clamp). The critical row alone under-reports
                # multi-contact violations; this makes static_constr (and the
                # bagged constr diagnostic) the honest certificate residual,
                # matching what the eager path already reports. Rows outside
                # the activation band or with degenerate gradients are
                # certified-inactive (+1).
                dq_fin = self.dq_safe_work.view(-1)
                all_res = rg @ dq_fin - r_bound
                row_live = (rh <= self.cbf_h_activate) & (r_gnorm > 1e-6)
                worst_all = torch.where(
                    row_live, all_res, torch.ones_like(all_res)).min()
                self.static_constr.copy_(worst_all.view(1))
            else:
                self.static_constr.copy_(torch.where(
                    final_active, final_constr, torch.ones_like(final_constr)))
            self.transfer_buffer[60].copy_(needs_repair.float().view(-1)[0])
            self.transfer_buffer[61].copy_(final_constr.view(-1)[0])
            # Integrated-velocity position command + passthrough.
            torch.add(self.static_q, self.dq_safe_work,
                      alpha=self.cbf_command_dt, out=self.q_safe_work)
            if self.cbf_passthrough_when_inactive:
                modified = (torch.norm(self.dq_safe_work - self.static_dq_nom)
                            > self.nominal_hold_deadband)
                active_m = self.static_h <= self.cbf_h_activate
                passthrough_mask = ((~modified) & (~active_m)).view(1, 1)
                self.q_safe_work.copy_(torch.where(
                    passthrough_mask, self.static_nominal_q, self.q_safe_work))
            self.last_q_safe.copy_(self.q_safe_work)
            # Command buffer for the single D2H readback.
            self.command_buffer[0:7].copy_(self.dq_safe_work.squeeze(0))
            self.command_buffer[7:14].copy_(self.q_safe_work.squeeze(0))
            self.command_buffer[14].copy_(self.static_h.squeeze(0))
            self.command_buffer[15].copy_(self.static_constr.squeeze(0))

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _control_step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.empty_cache()
        self.graph = torch.cuda.CUDAGraph()
        # thread_local: see the solve-only capture above -- subscriber
        # threads (obs/joint callbacks) legitimately touch CUDA during this
        # capture and must not invalidate it.
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            _control_step()
        # Reset the state the warmup polluted; the first replays re-derive
        # everything from live inputs.
        for buf in (self.static_h, self.static_grad_h, self.static_constr,
                    self.static_hdot_env_links, self._hdot_ema,
                    self._graph_dq_filtered, self.dq_safe_work,
                    self.q_safe_work, self.last_q_safe,
                    self.static_dq_escape_out, self.static_sampled_margin):
            buf.zero_()
        self.static_h.fill_(1.0)
        self.static_h_links.fill_(1.0)
        self._hdot_prev_h.copy_(self.static_h_links)
        self._hdot_prev_q.zero_()
        print(
            f"✅ Graphe CUDA capturé (control step + {self._graph_n_points} "
            f"points, gradient={self.cbf_multi_gradient_mode}) !")
        self._log_cuda_memory("CBF control-step graph captured")

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
            int(self._adopted_count),
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
        bounds = bounds + self.cbf_constraint_margin + self.static_sampled_margin
        if self._issf_on:
            bounds = bounds + self.static_issf_eps * G.norm(dim=-1)

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
        # NumPy only, no CUDA here: this fires at the joint-state rate in its
        # own thread and used to enqueue ~10 CUDA ops per message, competing
        # with the 100 Hz control loop for the GIL/dispatcher. The control
        # loop uploads the latest staged sample once per cycle instead.
        try:
            pos_dict = {n: p for n, p in zip(msg.name, msg.position)}
            # Fingers first: a gripper-only message must still update them
            # even though it lacks the arm joints and returns below.
            if len(self.finger_joint_names) == 2:
                f1 = pos_dict.get(self.finger_joint_names[0])
                f2 = pos_dict.get(self.finger_joint_names[1])
                if f1 is not None and f2 is not None:
                    self._finger_staging = np.asarray(
                        [f1, f2], dtype=np.float32)
            q_list = []
            for jn in self.joint_names:
                if jn in pos_dict:
                    q_list.append(pos_dict[jn])
                else:
                    return # Ignore message if arm joints are missing
            q_np = np.asarray(q_list, dtype=np.float32)
            stamp = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0.0 else rospy.get_time()
            if self._joint_prev_np is not None:
                dt = stamp - self._joint_prev_stamp
                if 1e-4 < dt < 0.5:
                    dq_real_raw = (q_np - self._joint_prev_np) / dt
                    gamma = dt / (dt + self.real_velocity_filter_tau)
                    self._dq_real_np = (
                        (1.0 - gamma) * self._dq_real_np + gamma * dq_real_raw
                    ).astype(np.float32)
            self._joint_prev_np = q_np
            self._joint_prev_stamp = stamp
            # Single-ref swap = atomic under the GIL.
            self._joint_staging = (q_np, self._dq_real_np, stamp)
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error in CBF joint_callback: {e}")

    def nominal_command_callback(self, msg):
        if len(msg.data) >= 7:
            nom_np = np.asarray(msg.data[:7], dtype=np.float32)
            self._nominal_q_staging = nom_np  # uploaded by the control loop
            if (
                self.log_trajectory_timing
                and self.active_plan_stamp is not None
                and not self._cbf_first_nominal_for_plan
            ):
                if self._joint_prev_np is None:
                    err = float('nan')
                else:
                    err = float(np.linalg.norm(nom_np - self._joint_prev_np))
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
                if self._joint_prev_np is None:
                    err = float('nan')
                else:
                    err = float(np.linalg.norm(nom_np - self._joint_prev_np))
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
        finite-difference FK. Runs in the PREPROCESS timer thread, NOT the
        100 Hz control loop: putting it in the control path cost ~25 ms of
        GPU contention per cycle (runPP14: cbf_ms p50 4.8 -> 29.8 ms). The
        metric is a slowly-varying preconditioner, so <=33 ms staleness
        (<0.025 rad of joint motion) is irrelevant; the control thread only
        reads the buffers. All linear algebra is numpy on the 8 downloaded
        poses -- cuSolver on 7x7/6x6 is ~13x slower than numpy and adds sync
        points.

        REFRESH RATE = ~preprocess_rate_hz / ~cbf_task_metric_update_every,
        because the decimation below counts TICKS, not seconds. Keep that
        quotient near 30 Hz when retuning either one: the T.cpu() below is a
        hard D2H sync (the cost the graphed selection was built to remove),
        and it lands BEFORE the critical_point_selection timer starts, so it
        is invisible in that stat. 30 Hz tick + every=1, or 100 Hz tick +
        every=3, both give the ~33 ms staleness this is budgeted for."""
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
        # Stage on the CPU only: the control thread copies these H2D on ITS
        # stream at adoption (fresh tensors each refresh, so an in-flight
        # adoption copy can never read a half-rewritten buffer). Writing
        # static_Winv directly from this thread would race the graph replay
        # now that control and preprocess run on different streams.
        self._staged_winv_cpu = torch.from_numpy(
            np.linalg.inv(W).astype(np.float32))
        if self.cbf_nullspace_clearance_gain > 0.0 or self.cbf_escape_lift_gain > 0.0:
            # Damped null-space projector N = I - J^+ J of the full 6D task
            # (position + orientation): motions in range(N) leave the TCP pose
            # unchanged to first order (the Panda's 1-DOF elbow swivel).
            J = np.concatenate([dP.T, omega.T], axis=0)              # 6x7
            N = np.eye(7) - J.T @ np.linalg.solve(
                J @ J.T + 1e-4 * np.eye(6), J)
            self._staged_N_cpu = torch.from_numpy(N.astype(np.float32))
        if self.cbf_escape_lift_gain > 0.0:
            # Joint row of the directed escape: dP @ e maps the chosen
            # task-space escape direction e onto joint space. Legacy is
            # e = +z (dP @ z == dP[:, 2], the lift); probe mode picks e by
            # sampled clearance. The buffer keeps its historical _Jz name --
            # the 100 Hz consumers (graphed _control_extra and the eager
            # _compute_escape_velocity) are untouched by the generalization.
            e_esc = np.array([0.0, 0.0, 1.0])
            if self.cbf_escape_probe_enabled:
                try:
                    e_esc = self._probe_escape_direction(dP, Tc[0, :3, 3])
                except Exception as exc:
                    rospy.logwarn_throttle(
                        5.0, "escape probe failed (falling back to +z): %s",
                        exc)
            self._staged_Jz_cpu = torch.from_numpy(
                (dP @ e_esc).astype(np.float32)).view(1, 7)

    def _probe_escape_direction(self, dP, p_tcp):
        """Pick the task-space escape direction e by sampled tangential
        clearance. Preprocess thread only (~30 Hz, beside the task metric):
        one tiny cdist against the raw obstacle cloud + numpy scoring, so the
        100 Hz control path never sees it -- it only reads the staged dP @ e
        row. Returns a unit [3] float64 vector (world frame).

        The normal is taken TCP-to-nearest-obstacle-point rather than from
        grad_h: it needs no control-thread graph buffers (no race with graph
        replay), and exactness is irrelevant because the 100 Hz consumer
        re-projects the row tangent to the live grad_h anyway."""
        e_prev = self._escape_probe_e
        now = time.time()
        # Freeze the committed direction for the whole escape maneuver: the
        # trigger's hysteresis decides when it ends, re-opening the vote.
        # Tabu re-vote: if the frozen direction stops producing TCP motion
        # along itself (the QP is clipping it: wedged in a concavity), give
        # up early, penalize it for tabu_ttl seconds and re-vote below.
        if (e_prev is not None and self._cbf_escape_active_cycles
                >= self.cbf_escape_activation_cycles):
            if self._probe_commit_t0 is None:
                self._probe_commit_t0 = now
                self._probe_commit_p0 = p_tcp.copy()
            if self._probe_commit_e_t0 is None:
                self._probe_commit_e_t0 = now
            if (float((p_tcp - self._probe_commit_p0) @ e_prev)
                    >= self.cbf_escape_probe_tabu_min_progress):
                # Headway: restart the window, requiring CONTINUED motion.
                self._probe_commit_t0 = now
                self._probe_commit_p0 = p_tcp.copy()
            # Resolution timeout: a direction that keeps "progressing" but
            # never clears the conflict (climbing a wall taller than the
            # reach) is as wedged as a stall -- give another direction a
            # turn. The stall window below cannot catch it.
            unresolved = (
                self.cbf_escape_probe_resolve_timeout > 0.0
                and now - self._probe_commit_e_t0
                > self.cbf_escape_probe_resolve_timeout)
            if (not unresolved and now - self._probe_commit_t0
                    < self.cbf_escape_probe_tabu_timeout):
                rospy.loginfo_throttle(
                    2.0, "escape probe: committed dir [%.2f %.2f %.2f]",
                    e_prev[0], e_prev[1], e_prev[2])
                return e_prev
            self._probe_tabu.append(
                (e_prev.copy(), now + self.cbf_escape_probe_tabu_ttl))
            rospy.logwarn(
                "escape probe: dir [%.2f %.2f %.2f] %s -> tabu, re-voting",
                e_prev[0], e_prev[1], e_prev[2],
                ("unresolved after %.2g s"
                 % self.cbf_escape_probe_resolve_timeout) if unresolved
                else ("stalled for %.2g s"
                      % self.cbf_escape_probe_tabu_timeout))
            self._escape_probe_e = None
            self._probe_commit_t0 = None
            self._probe_commit_e_t0 = None
            e_prev = None
        else:
            self._probe_commit_t0 = None
            self._probe_commit_e_t0 = None
        fallback = e_prev if e_prev is not None else np.array([0.0, 0.0, 1.0])
        pts = self.obs_points
        if pts is None or int(pts.shape[0]) == 0:
            return fallback
        with torch.no_grad():
            p = torch.from_numpy(
                p_tcp.astype(np.float32)).to(pts.device).view(1, 3)
            nearest = pts[torch.argmin(
                torch.norm(pts - p, dim=1))].cpu().numpy().astype(np.float64)
        n = p_tcp - nearest
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-6:
            return fallback
        n = n / n_norm
        # Tangent basis of the plane orthogonal to the obstacle normal.
        a = (np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9
             else np.array([1.0, 0.0, 0.0]))
        t1 = np.cross(n, a)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        K = self.cbf_escape_probe_num_dirs
        th = 2.0 * np.pi * np.arange(K) / K
        ring = np.cos(th)[:, None] * t1 + np.sin(th)[:, None] * t2  # [K,3]
        # Hemisphere toward +n: one ring per elevation (0 deg = the legacy
        # tangent ring) plus the pure retreat +n. In a concave pocket the
        # only exit is +n; a tangent-only candidate set cannot represent it.
        elevs = np.deg2rad(np.asarray(self.cbf_escape_probe_elevations))
        dirs = np.concatenate(
            [np.cos(e) * ring + np.sin(e) * n[None, :] for e in elevs]
            + ([n[None, :]] if (elevs > 1e-6).any() else []), axis=0)  # [D,3]
        radii = np.asarray(self.cbf_escape_probe_radii)             # [R] asc.
        probes = (p_tcp[None, None, :]
                  + radii[:, None, None] * dirs[None, :, :])        # [R,D,3]
        with torch.no_grad():
            X = torch.from_numpy(
                probes.reshape(-1, 3).astype(np.float32)).to(pts.device)
            clear_rd = torch.cdist(X, pts).min(dim=1).values.view(
                len(radii), dirs.shape[0]).cpu().numpy()            # [R,D]
        cap = self.cbf_escape_probe_clearance_cap
        # Cap the clearance so directions past all obstacles tie, letting the
        # goal/commit terms break the tie instead of "whatever is emptiest".
        score = np.minimum(
            clear_rd.min(axis=0).astype(np.float64), cap)
        if len(radii) >= 2 and self.cbf_escape_probe_growth_weight > 0.0:
            # Corridors keep clearance constant with radius; exits open up.
            score += self.cbf_escape_probe_growth_weight * np.clip(
                (clear_rd[-1] - clear_rd[0]).astype(np.float64), -cap, cap)
        # Workspace floor: the table is not an obstacle point, so a downward
        # probe would otherwise look maximally clear.
        below = (probes[..., 2] < self.cbf_escape_probe_min_z).any(axis=0)
        score[below] -= 1.0
        dq_nom = self._nominal_dq_staging
        if dq_nom is not None and self.cbf_escape_probe_goal_weight > 0.0:
            v = dP.T @ dq_nom.astype(np.float64)   # nominal task velocity [3]
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-3:
                score += (self.cbf_escape_probe_goal_weight
                          * (dirs @ (v / v_norm)))
        if e_prev is not None:
            score += self.cbf_escape_probe_commit_weight * (dirs @ e_prev)
        if self._probe_tabu:
            self._probe_tabu = [
                (d, t_exp) for d, t_exp in self._probe_tabu if t_exp > now]
            for d, _ in self._probe_tabu:
                score -= (self.cbf_escape_probe_tabu_weight
                          * np.clip(dirs @ d, 0.0, None))
        e_best = dirs[int(np.argmax(score))]
        if e_prev is None:
            e = e_best
        else:
            beta = self.cbf_escape_probe_ema
            e = (1.0 - beta) * e_prev + beta * e_best
            e_norm = np.linalg.norm(e)
            e = e_best if e_norm < 1e-6 else e / e_norm
        self._escape_probe_e = e
        return e

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

    def _adopt_preprocess_results(self):
        """Adopt the latest completed preprocess pack (control thread only).

        Never blocks: event.query() is a host-side flag check. Until the pack's
        GPU work has finished, the control loop keeps using the previously
        adopted selection, so a slow preprocess batch degrades selection
        freshness instead of control latency. Task-metric tensors are copied
        H2D here, on the control stream, so the graphed static_Winv read can
        never race a write from the preprocess thread."""
        pack = self._prep_pack
        if pack is None or pack[0] == self._adopted_seq or not pack[1].query():
            return
        (seq, _, obs, count, min_dist,
         winv_cpu, n_cpu, jz_cpu, n_real_pin) = pack
        self._adopted_obs = obs
        if n_real_pin is not None:
            # Graphed path: swap the fixed graph width for the real count now
            # that the pack's event has fired (so the async D2H has landed and
            # this is a pinned-memory read, not a GPU sync). Restores exactly
            # the legacy semantics -- top-k sorts the +inf rejects last, so
            # rows [0, count) are the real points and the rest is dummy pad --
            # which is what the Softmin alpha*log(N) de-bias needs to match.
            count = min(int(n_real_pin[0]), int(count))
        self._adopted_count = count
        self._adopted_min_dist = min_dist
        if winv_cpu is not None and winv_cpu is not self._adopted_winv_src:
            self.static_Winv.copy_(winv_cpu)
            self._adopted_winv_src = winv_cpu
        if n_cpu is not None and n_cpu is not self._adopted_N_src:
            if self._task_metric_N_buf is None:
                self._task_metric_N_buf = torch.empty(
                    (7, 7), device=self.device)
            self._task_metric_N_buf.copy_(n_cpu)
            self._task_metric_N = self._task_metric_N_buf
            self._adopted_N_src = n_cpu
        if jz_cpu is not None and jz_cpu is not self._adopted_Jz_src:
            if self._task_metric_Jz_buf is None:
                self._task_metric_Jz_buf = torch.empty(
                    (1, 7), device=self.device)
            self._task_metric_Jz_buf.copy_(jz_cpu)
            self._task_metric_Jz = self._task_metric_Jz_buf
            self._adopted_Jz_src = jz_cpu
        self._adopted_seq = seq

    def _run_velocity_cbf_solver(self, current_q, local_obs, dq_nom):
        self.static_q.copy_(current_q)
        self.static_obs.copy_(local_obs)
        self.static_dq_nom.copy_(dq_nom)
        # Sampled-data margin + ISSf eps(v), eager twin of the in-graph
        # update in _control_step (dq_safe_work still holds last cycle's
        # final command at solver entry).
        if self.cbf_sampled_data_margin or self._issf_on:
            v2 = torch.maximum(
                (self.dq_safe_work * self.dq_safe_work).sum(),
                (dq_nom * dq_nom).sum())
            if self.cbf_sampled_data_linearization == "executed":
                v_exec = (torch.norm(self.dq_real_measured)
                          + self.cbf_sampled_data_vel_headroom)
                v2 = torch.minimum(v2, v_exec * v_exec)
            self.static_v2_lin.copy_(v2)
            if self.cbf_sampled_data_margin:
                m_scalar = torch.clamp(
                    self._sampled_margin_coeff * v2,
                    max=self.cbf_sampled_data_margin_max)
                self.static_sampled_scalar.copy_(m_scalar)
                self.static_sampled_margin.copy_(m_scalar)
            if self._issf_on:
                # eps(v) at the measured speed in executed mode (see the
                # in-graph twin for the identification argument).
                v_issf = (torch.norm(self.dq_real_measured)
                          if self.cbf_sampled_data_linearization == "executed"
                          else torch.sqrt(v2))
                self.static_issf_eps.copy_(
                    self.cbf_issf_epsilon + self.cbf_issf_rho * v_issf)
        if self.cbf_solver_mode in ("fast_tangent", "multi_graphed"):
            self.graph.replay()
            return self.static_dq_safe.detach()
        return self.solve_multicbf_projection(
            current_q, local_obs, dq_nom).detach()

    def _joint_limit_clamp_(self, dq):
        """In-place soft joint-limit velocity barrier on the final command
        (reads self.static_q; pure tensor ops, CUDA-graph safe). Bounds:
        dq_i <= kappa*(q_hi - margin - q_i)  (floored at -limit_push)
        dq_i >= kappa*(q_lo + margin - q_i)  (capped  at +limit_push)
        i.e. per-joint braking into the margin band and a small bounded
        push back out of it. On the Panda the band doubles as singularity
        avoidance (q4 upper = elbow straight, q6 lower = wrist aligned)."""
        if self.cbf_joint_limit_margin <= 0.0:
            return dq
        k = self.cbf_joint_limit_kappa
        m = self.cbf_joint_limit_margin
        b_hi = torch.clamp(k * (self._q_lim_hi - m - self.static_q),
                           min=-self.cbf_joint_limit_push)
        b_lo = torch.clamp(k * (self._q_lim_lo + m - self.static_q),
                           max=self.cbf_joint_limit_push)
        dq.copy_(torch.minimum(torch.maximum(dq, b_lo), b_hi))
        return dq

    def _queue_escape_state(self, dq_nom_cbf, dq_out=None, dq_nom_raw=None):
        """Enqueue an async D2H copy of the escape-trigger scalars (h,
        |dq_safe - dq_nom_cbf|, |grad_h|, |dq_safe - dq_nom_raw|) into pinned
        memory after the solve. _compute_escape_velocity reads them NEXT
        cycle if the copy has completed (event query, never blocks). The old
        synchronous .cpu() here drained the contended GPU queue every 100 Hz
        cycle. dq_nom_raw = the pre-escape nominal; the trigger latches on
        that delta (see _esc_pin comment)."""
        if not self.cbf_escape_enabled:
            return
        if dq_out is None:
            dq_out = self.dq_safe_work
        if dq_nom_raw is None:
            dq_nom_raw = dq_nom_cbf
        vals = torch.stack((
            self.static_h.detach().view(-1)[0],
            torch.norm(dq_out - dq_nom_cbf),
            torch.norm(self.static_grad_h.detach()),
            torch.norm(dq_out - dq_nom_raw),
        ))
        self._esc_pin.copy_(vals, non_blocking=True)
        self._esc_event.record()

    def _update_dynamic_hdot(self, current_q, dt):
        """Model-free PER-LINK environment-rate estimator for the time-varying
        CBF term: d_env[k] = (h_k_now - h_k_prev)/dt - grad_h_k . dq_robot,
        i.e. each link's measured barrier rate minus the robot's own
        contribution. Per link (not on the min) so an obstacle approaching one
        link is detected even while another link (fork at the food) pins the
        min at ~0, and so the tightening lands only on the approached link's
        rows. EMA-filtered, gated per link against selection jumps and far
        field, clamped to the approaching side, written into
        static_hdot_env_links for the NEXT cycle's graph replay. All tensor
        ops on-device: no sync."""
        if not self.cbf_dynamic_hdot_enabled:
            return
        h_now = self.static_h_links.detach()                  # [R]
        q_now = current_q.detach().view(-1)                   # [7]
        if self._hdot_prev_h is None:
            self._hdot_prev_h = h_now.clone()
            self._hdot_prev_q = q_now.clone()
            return
        inv_dt = 1.0 / max(dt, 1e-3)
        dh = (h_now - self._hdot_prev_h) * inv_dt             # [R]
        robot_part = (self.static_grad_links.detach()
                      @ (q_now - self._hdot_prev_q)) * inv_dt  # [R]
        env_raw = dh - robot_part
        valid = (h_now < self.cbf_dynamic_hdot_h_range) & (
            env_raw.abs() < self.cbf_dynamic_hdot_jump)
        gamma = dt / (dt + self.cbf_dynamic_hdot_tau)
        # Invalid samples (selection switch, far field) hard-reset that link's
        # filter instead of decaying: a stale tightening must not outlive its
        # cause.
        self._hdot_ema = torch.where(
            valid,
            (1.0 - gamma) * self._hdot_ema + gamma * env_raw,
            torch.zeros_like(env_raw))
        self.static_hdot_env_links.copy_(torch.clamp(
            -self._hdot_ema - self.cbf_dynamic_hdot_deadband,
            min=0.0, max=self.cbf_dynamic_hdot_max))
        self._hdot_prev_h.copy_(h_now)
        self._hdot_prev_q.copy_(q_now)

    def _escape_trigger_update(self):
        """CPU-only PROPORTIONAL escape gate on the pinned async readback of
        (h, |dq_safe-dq_nom_cbf|, |grad_h|). Shared by the eager and graphed
        control steps. Returns (gate in [0,1], h_bias).

        History: the original binary trigger FLICKERED (the composed-nominal
        delta drops to ~0 the moment the escape works -> instant reset,
        runPP1000), which accidentally pulse-width-modulated the escape into
        a gentle average push -- smooth, but too weak for a concave pocket.
        The latch that replaced it (raw-nominal delta) had unbounded memory
        and counted the null-space clearance as conflict: runPP1002 latched
        30 s at full authority, wrestling the nominal (wiggle, h < 0). This
        gate makes the duty-cycling EXPLICIT and smooth: an h-proximity ramp
        gated by conflict, EMA-filtered. The same feedback (conflict vanishes
        as the escape succeeds) now settles the gate at a partial-authority
        equilibrium instead of chattering full-on/full-off; magnitudes ramp
        continuously, so no command steps."""
        if not self.cbf_escape_enabled:
            self._cbf_escape_active_cycles = 0
            self._escape_gate = 0.0
            return 0.0, 0.0
        h_bias = 0.0
        if (self.cbf_solver_mode in ("fast_tangent", "multi_graphed")
                and not self._barrier_value_hard):
            h_bias = float(self.barrier.alpha) * float(
                np.log(max(1, int(self._adopted_count))))
        if self._esc_event.query():
            self._esc_vals = self._esc_pin.numpy().copy()
        h_escape = float(self._esc_vals[0]) + (
            h_bias if self.cbf_escape_use_bias_corrected_h else 0.0)
        # Composed-nominal delta: the self-regulation feedback (goes to ~0
        # while the escape fully absorbs the conflict).
        base_delta = float(self._esc_vals[1])
        grad_norm_value = float(self._esc_vals[2])

        # Proximity ramp: 1 below h_trigger, 0 above h_release, linear
        # between -- replaces the binary near/release hysteresis.
        span = max(self.cbf_escape_h_release - self.cbf_escape_h_trigger,
                   1e-6)
        ramp = min(max(
            (self.cbf_escape_h_release - h_escape) / span, 0.0), 1.0)
        conflict = (base_delta > self.cbf_escape_min_cbf_delta
                    or h_escape < 0.0)
        target = ramp if (conflict and grad_norm_value > 1e-7) else 0.0
        gate = self._escape_gate + self._escape_gate_beta * (
            target - self._escape_gate)
        self._escape_gate = gate

        # Discrete activity for the probe commit/tabu bookkeeping only:
        # Schmitt thresholds so the equilibrium gate doesn't flap it.
        if self._escape_was_active:
            escape_active = gate > 0.25
        else:
            escape_active = gate > 0.60
        self._cbf_escape_active_cycles = (
            self._cbf_escape_active_cycles + 1 if escape_active else 0)
        if self.cbf_escape_probe_reentry_window > 0.0:
            now = time.time()
            if self._escape_was_active and not escape_active:
                # Maneuver ended (usually h > h_release): remember what it
                # tried. List append/read is GIL-safe vs the preprocess
                # thread, same as _escape_probe_e itself.
                self._escape_release_t = now
                self._escape_release_e = (
                    None if self._escape_probe_e is None
                    else self._escape_probe_e.copy())
            elif (escape_active and not self._escape_was_active
                    and self._escape_release_e is not None
                    and now - (self._escape_release_t or 0.0)
                    < self.cbf_escape_probe_reentry_window):
                # Re-trigger right after a release: that direction escaped
                # but resolved nothing -- tabu it so the next vote explores.
                self._probe_tabu.append(
                    (self._escape_release_e,
                     now + self.cbf_escape_probe_tabu_ttl))
                rospy.logwarn(
                    "escape probe: re-entry %.2gs after release -> tabu dir "
                    "[%.2f %.2f %.2f], exploring",
                    now - self._escape_release_t,
                    self._escape_release_e[0], self._escape_release_e[1],
                    self._escape_release_e[2])
                self._escape_release_e = None
                self._escape_probe_e = None  # force a fresh vote
        self._escape_was_active = escape_active
        return gate, h_bias

    def _compute_escape_velocity(self, dq_nom_base):
        """Escape decision from the PREVIOUS cycle's solve state (one 10 ms
        cycle stale -- the gate is EMA-filtered, so this is equivalent in
        behavior but allows a single graph replay per cycle and no mid-loop
        sync). The whole composed escape scales by the proportional gate."""
        gate, h_bias = self._escape_trigger_update()
        if gate < 0.01:
            return self.zero_dq, False

        grad_h = self.static_grad_h.detach()
        grad_norm = torch.norm(grad_h, dim=1, keepdim=True)

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
            # Keep the outward component (see the graphed twin): retreat
            # directions from the hemisphere probe must survive projection.
            lift_n = (lift * normal_dir).sum(dim=1, keepdim=True)
            lift_t = lift - torch.clamp(lift_n, max=0.0) * normal_dir
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

        if self.cbf_escape_suppress_nominal > 0.0:
            # Post-clamp cancellation of the nominal's inward push (see the
            # graphed twin in _control_extra).
            dq_escape = dq_escape - (self.cbf_escape_suppress_nominal
                                     * torch.clamp(normal_speed, max=0.0)
                                     * normal_dir)

        return gate * dq_escape, True

    def _nominal_vel_cb(self, msg):
        if len(msg.data) < 7:
            return
        self._nominal_dq_staging = np.asarray(msg.data[:7], dtype=np.float32)
        self.nominal_dq_stamp = rospy.get_time()

    def _reflex_status_cb(self, msg):
        # Latest reflex brake scale: its braking is a certified modification
        # of the command, not plant disturbance, so the ISSf monitor compares
        # the measured velocity against alpha * u, not u.
        if len(msg.data) >= 1:
            self._reflex_alpha_seen = float(msg.data[0])

    def obs_callback(self, msg):
        try:
            points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, int(msg.point_step/4))
            self.obs_points = torch.from_numpy(points[:, :3].copy()).to(
                self.device, dtype=torch.float32)
            self.obs_stamp = rospy.get_time()
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
        self._selection_generation = (self._selection_generation + 1) % (1 << 24)

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

    # ── graphed critical-point selection (~cbf_selection_mode:=graphed) ────

    def _selection_graph_supported(self):
        """The graphed path implements exactly ONE selection policy.

        It fuses the 'sdf' candidate filter with the 'sdf' + 'topk' selection:
        rank the cloud by whole-body SDF, drop self-hits and anything past
        cbf_sdf_candidate_max_dist, keep the k lowest. Any config whose
        selection differs (fork_mesh/fork_sdf scoring, 'local' Euclidean
        clustering, stickiness memory, the grasp split budget) is NOT this
        policy, so it stays on the legacy path rather than being silently
        approximated. Reported once at startup.
        """
        why = []
        if self.cbf_candidate_filter != "sdf":
            why.append("candidate_filter=%s (need sdf)" % self.cbf_candidate_filter)
        if self.cbf_selection_metric != "sdf":
            why.append("selection_metric=%s (need sdf)" % self.cbf_selection_metric)
        if self.cbf_cluster_mode != "topk":
            why.append("cluster_mode=%s (need topk)" % self.cbf_cluster_mode)
        if self.cbf_selection_sticky_points > 0:
            why.append("selection stickiness is on")
        if self.cbf_grasp_enabled and self.cbf_grasp_separate_constraint \
                and self.cbf_grasp_obstacle_points > 0:
            why.append("grasp split-select is on")
        # NOTE: soft barrier value is supported. The graph emits a fixed k rows
        # padded with far dummies, so selected_count is k rather than the real
        # count; that is inert for the barrier (a dummy at 100 m contributes
        # exp(-100/alpha) = 0 to the Softmin, so the VALUE is unchanged), but
        # soft mode's alpha*log(N) de-bias reads the count and log(k) would
        # over-correct when the pool is mostly dummies (runPP1010: 44% were).
        # Resolved without a sync: the graph emits n_real, an async D2H lands
        # it in a pinned buffer, and adoption reads it once the pack event has
        # fired. See _preprocess_obstacles_graphed / _adopt_preprocess_results.
        return why

    def _build_selection_graph(self, q9):
        """Capture FK + whole-body SDF + self-filter + top-k as one graph.

        Called once from __init__ (before the timers start), not lazily: see
        the call site.

        FK is captured INSIDE the graph on purpose: link_poses depends on q,
        and _native_forward_kinematics_subset allocates fresh tensors per
        call, so passing a dict in from outside would freeze stale pointers
        and filter the cloud against where the arm USED to be. Capturing it
        keeps the replay live in q (verified: replaying after restaging q
        changes the selection, and matches the eager path bit-exactly).
        """
        n = self.cbf_sdf_graph_points
        self._sel_static_pts = torch.full((n, 3), 100.0, dtype=torch.float32,
                                          device=self.device)
        self._sel_static_q9 = q9.detach().clone()
        # MUST be an attribute, not a local. A CUDA graph records raw device
        # pointers: any tensor allocated OUTSIDE the capture and referenced
        # inside it must outlive the graph, or the allocator hands its block
        # to someone else and every replay reads that stranger's bytes. As a
        # local this dangled onto _sel_static_q9's memory, so the dummy pad
        # came back as q instead of (100,100,100) -- invisible whenever the
        # selection was full (torch.where never picks the dummy branch), and
        # a phantom obstacle at the robot's own joint values the moment fewer
        # than k points were valid. Tensors created INSIDE the capture are
        # fine: they live in the graph's private pool.
        self._sel_dummy_row = torch.full((1, 3), 100.0, device=self.device)

        def _select():
            link_poses = self.robot_layer._native_forward_kinematics_subset(
                self._sel_static_q9, self.protected_fk_tree)
            _, sdf_per_link = self.bernstein_core.get_whole_body_sdf_batch(
                self._sel_static_pts, self.eye4, self._sel_static_q9,
                return_per_link=True, link_poses=link_poses)
            sdf_body = sdf_per_link[0, :5, :].min(dim=0).values
            sdf_all = sdf_per_link[0].min(dim=0).values
            sdf_self_filter = (sdf_all if self.cbf_sdf_self_filter_all_links
                               else sdf_body)
            # Same keep rule as _whole_body_sdf_candidates, expressed as a
            # score instead of a boolean mask: rejects go to +inf so top-k
            # can never pick them, and every shape stays static. This is what
            # removes the mask-index sync.
            valid = (
                (sdf_all <= self.cbf_sdf_candidate_max_dist)
                & (sdf_self_filter > self.cbf_sdf_self_filter_margin)
            )
            score = torch.where(valid, sdf_all,
                                torch.full_like(sdf_all, float('inf')))
            # sorted=True (default) => scores ascend and the +inf rejects sort
            # LAST, so the real points land in rows [0, n_real) and the dummy
            # pad in [n_real, k): exactly the legacy _copy_fixed_obstacles
            # layout, which is what lets selected_count keep its meaning.
            sc, idx = torch.topk(score, k=self.cbf_graph_points, largest=False)
            sel = self._sel_static_pts[idx]
            finite = torch.isfinite(sc)
            # Fewer than k valid points: the tail of the top-k holds rejects.
            # Overwrite those with the far dummy, matching the legacy
            # _copy_fixed_obstacles pad (and keeping pct_dummy_selection
            # meaningful) instead of admitting a self-hit as an obstacle.
            obs = torch.where(finite.unsqueeze(1), sel, self._sel_dummy_row)
            n_real = finite.sum()

            # min distance from the TOOL to the candidate pool, over the same
            # valid set the legacy path uses (not just the selected k), kept
            # on-device: a .item() here would sync the preprocess thread.
            T_tool = link_poses.get(self.tool_link)
            x_now = T_tool[:, :3, 3]
            dist = torch.norm(self._sel_static_pts - x_now, dim=1)
            min_dist = torch.where(
                valid, dist, torch.full_like(dist, float('inf'))).min()
            return obs, min_dist, n_real

        # Warm up on a side stream (allocator + any lazy init must not be
        # captured), then capture. capture_error_mode="thread_local" matches
        # finalize_cuda_graph: the default global mode errors if ANY other
        # thread touches CUDA during capture, and this node has an FM planner
        # and a control loop alongside it.
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _select()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize(self.device)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            (self._sel_out_obs, self._sel_out_min_dist,
             self._sel_out_n_real) = _select()
        self._sel_graph = g
        rospy.loginfo(
            "CBF selection: graphed path ready (width=%d pts, k=%d, "
            "max_dist=%.3f, self_margin=%.3f)",
            self.cbf_sdf_graph_points, self.cbf_graph_points,
            self.cbf_sdf_candidate_max_dist, self.cbf_sdf_self_filter_margin)

    def _selection_broad_phase(self, pts, q9, n_keep):
        """Cloud wider than the graph: keep the n_keep nearest by link centre.

        Only fires when the cloud exceeds ~cbf_sdf_graph_points, which no
        currently-published cloud does (feeding 2573, PP 2000 vs 4096). It
        exists so that growing the cloud degrades into a cheap broad phase
        (~0.16 ms) instead of silently truncating it -- dropping obstacle
        points is a safety hole, not a slow path. Sync-free: out-of-range
        goes to +inf, then a fixed-size top-k (no boolean mask).
        """
        link_poses = self.robot_layer._native_forward_kinematics_subset(
            q9, self.protected_fk_tree)
        centers = [link_poses[n][0, :3, 3] for n in self.protected_link_names
                   if link_poses.get(n) is not None]
        if not centers:
            return pts[:n_keep]
        centers = torch.stack(centers, dim=0)
        d = torch.cdist(pts.unsqueeze(0), centers.unsqueeze(0)
                        ).squeeze(0).min(dim=1).values
        _, keep = torch.topk(d, k=n_keep, largest=False)
        return pts[keep]

    def _preprocess_obstacles_graphed(self, pts, q9, t_start):
        """One fixed-size replay: FK + SDF + self-filter + top-k + dummies."""
        n_cloud = int(pts.shape[0])
        if n_cloud > self.cbf_sdf_graph_points:
            self._sel_broad_phase_hits += 1
            rospy.logwarn_throttle(
                10.0,
                "CBF selection: cloud %d > graph width %d -- broad phase "
                "engaged (%d ticks). Raise ~cbf_sdf_graph_points to keep the "
                "whole cloud in the exact path.",
                n_cloud, self.cbf_sdf_graph_points, self._sel_broad_phase_hits)
            pts = self._selection_broad_phase(
                pts, q9, self.cbf_sdf_graph_points)
            n_cloud = int(pts.shape[0])

        # Stage: pad the tail with far dummies so retired points from a
        # shorter cloud cannot linger in the buffer as phantom obstacles.
        self._sel_static_pts.fill_(100.0)
        self._sel_static_pts[:n_cloud].copy_(pts[:n_cloud])
        self._sel_static_q9.copy_(q9)
        self._sel_graph.replay()

        # _copy_fixed_obstacles clones into a fresh tensor, which is what
        # keeps preprocess_obstacles' pack invariant intact: the control
        # thread must never hold a ref to the graph's static output, or the
        # next replay would mutate the selection under a pending adoption.
        # It sets selected_count = k (the fixed width); the real count lands
        # asynchronously below and replaces it at adoption.
        self._copy_fixed_obstacles(self._sel_out_obs)
        # Real-count readback, the _esc_pin trick: an ASYNC D2H into pinned
        # memory, never awaited here. A blocking read (.item()) would wait on
        # the replay behind whatever SAM2/FM has queued on the GPU -- that is
        # precisely the 38 ms stall this path exists to remove. The copy is
        # issued on the preprocess stream BEFORE preprocess_obstacles records
        # _prep_done_event, so the event covers it and the control thread can
        # read the value at adoption without ever blocking.
        self._sel_n_pin.copy_(self._sel_out_n_real, non_blocking=True)
        self.selected_num_inside = n_cloud
        self.selected_min_obs_dist = self._sel_out_min_dist.clone()
        # Candidate-pool viz would need the valid mask = a sync. The selected
        # points still publish; only the yellow "considered but not chosen"
        # cloud is unavailable on this path.
        self.selected_pts_yellow = torch.empty((0, 3), dtype=torch.float32,
                                               device=self.device)
        self.debug_candidate_points = None
        self.debug_score_seed = None
        self.debug_tip_min = None

        if self.profile_sync:
            torch.cuda.synchronize()
        self.preprocess_times.append(time.perf_counter() - t_start)
        self.timing.publish('critical_point_selection',
                            (time.perf_counter() - t_start) * 1000.0)

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
        # Skip-if-busy: if the previous batch's GPU work has not finished,
        # enqueueing another on top only deepens the launch queue that the
        # control thread then blocks behind (the enqueue-side stalls seen in
        # runPP100). Dropping the tick keeps at most one preprocess batch in
        # flight; the selection simply refreshes at the rate the GPU sustains.
        if self._prep_pack is not None and not self._prep_done_event.query():
            self._prep_skips += 1
            rospy.loginfo_throttle(
                10.0, "CBF preprocess: skipping ticks, GPU behind (%d skips)",
                self._prep_skips)
            return
        if self._use_streams:
            with torch.cuda.stream(self._prep_stream):
                self._preprocess_obstacles_impl()
            record_stream = self._prep_stream
        else:
            self._preprocess_obstacles_impl()
            record_stream = torch.cuda.current_stream()
        # Publish an immutable pack for the control thread. The refs are
        # snapshot here (every batch allocates fresh tensors), so a later
        # batch can never mutate what a pending adoption will read. Reusing
        # one event is safe: the skip gate above guarantees it has completed
        # before it is re-recorded.
        self._prep_done_event.record(record_stream)
        self._prep_seq += 1
        self._prep_pack = (
            self._prep_seq,
            self._prep_done_event,
            self.selected_obs,
            self.selected_count,
            self.selected_min_obs_dist,
            self._staged_winv_cpu,
            self._staged_N_cpu,
            self._staged_Jz_cpu,
            # Graphed path only: selected_count above is the fixed graph width
            # (dummy pad included). The true count is still in flight in this
            # pinned buffer; adoption resolves it once the event says it has
            # landed. None on the legacy path, whose count is already exact.
            self._sel_n_pin if self.cbf_selection_mode == "graphed" else None,
        )

    def _preprocess_obstacles_impl(self):
        # Task-metric refresh lives here (preprocess thread), off the 100 Hz
        # control path. Before the obstacle early-return so the metric stays
        # current even while the cloud is empty -- and therefore OUTSIDE the
        # critical_point_selection timing below, which starts at t_start.
        # Its own rate is preprocess_rate_hz / cbf_task_metric_update_every;
        # see _update_task_metric.
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

        if self.cbf_selection_mode == "graphed" and not self._sel_graph_failed:
            try:
                with torch.no_grad():
                    q9_current = torch.cat(
                        [self.current_q.detach(), self.q_pad2], dim=1)
                    self._preprocess_obstacles_graphed(pts, q9_current, t_start)
                return
            except Exception as e:
                # Never take the node down over an optimization: fall back to
                # the legacy path for the rest of the run and say so loudly.
                self._sel_graph_failed = True
                self._sel_graph = None
                rospy.logerr(
                    "CBF graphed selection failed (%s) -- falling back to the "
                    "legacy path for the rest of this run.", e)

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
            # Control and preprocess now run on their own CUDA streams; this
            # timer thread reads tensors produced on both. A device sync here
            # (viz thread only, cosmetic rate) guarantees complete data
            # without adding any wait to the control path.
            torch.cuda.synchronize()
            pts_yellow = self.selected_pts_yellow
            obs = self._adopted_obs[:self._adopted_count]

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

        if self._use_streams:
            # Current stream is thread-local: everything this loop enqueues
            # (graph replay, command build, D2H readbacks) lands on the
            # high-priority control stream and no longer orders behind the
            # preprocess batches. Callback threads keep the default stream.
            torch.cuda.set_stream(self._ctrl_stream)

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

            # Upload the latest callback samples (staged as NumPy by the
            # subscriber threads) once per cycle. Pageable H2D is
            # host-synchronous, so every later reader on any stream sees
            # complete data. 2-3 tiny copies here replace the per-message
            # CUDA traffic the callbacks used to generate.
            staging = self._joint_staging
            if staging is not None and staging[2] != self._joint_uploaded_stamp:
                q_np, dq_np, stamp = staging
                self.current_q = torch.from_numpy(q_np).to(
                    self.device).unsqueeze(0)
                self.dq_real_measured = torch.from_numpy(dq_np).to(
                    self.device).unsqueeze(0)
                self._joint_uploaded_stamp = stamp
                self._joint_uploaded_np = q_np
            fingers_np = self._finger_staging
            if fingers_np is not None and fingers_np is not self._finger_uploaded:
                # In-place copy: the evaluators (and the captured CUDA graph)
                # hold views of q_pad2, so the new tail is visible everywhere.
                self.q_pad2.copy_(
                    torch.from_numpy(fingers_np).view(1, 2))
                self._finger_uploaded = fingers_np
            nom_np = self._nominal_q_staging
            if nom_np is not None and nom_np is not self._nominal_q_uploaded:
                self.nominal_q = torch.from_numpy(nom_np).to(
                    self.device).unsqueeze(0)
                self._nominal_q_uploaded = nom_np
            ndq_np = self._nominal_dq_staging
            if ndq_np is not None and ndq_np is not self._nominal_dq_uploaded:
                self.nominal_dq_ff = torch.from_numpy(ndq_np).to(
                    self.device).unsqueeze(0)
                self._nominal_dq_uploaded = ndq_np

            # Announce readiness only once verified safety inputs have been
            # received (joints always; a first obstacle cloud unless waived).
            if not self._ready_announced:
                if self._joint_uploaded_stamp is not None and (
                        self.obs_stamp is not None
                        or not self.watchdog_require_obstacles):
                    self.ready_pub.publish(Bool(data=True))
                    self._ready_announced = True

            # Stale-data watchdog: with a dead publisher upstream the barrier
            # would keep certifying against a frozen world. Command a zero-
            # velocity hold instead of solving; the low-level controller's own
            # command timeout is the hard backstop if this node dies too.
            if self.watchdog_enabled:
                stale = None
                if (self._joint_uploaded_stamp is not None
                        and current_time - self._joint_uploaded_stamp
                        > self.watchdog_joint_timeout):
                    stale = 'joint states'
                elif (self.obs_stamp is not None
                        and current_time - self.obs_stamp
                        > self.watchdog_obstacle_timeout):
                    stale = 'obstacle cloud'
                if stale is not None:
                    rospy.logwarn_throttle(
                        1.0, "CBF watchdog: %s stale -- commanding safe hold."
                        % stale)
                    self._stop_velocity_controller()
                    self.rate.sleep()
                    continue

            if self.current_q is not None and self.nominal_q is not None:
                self._adopt_preprocess_results()
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
                    elif self._graph_control_step and not self._last_q_safe_seeded:
                        # Graph mode: last_q_safe is a FIXED buffer captured by
                        # the graph; seed it in place on the first cycle.
                        self.last_q_safe.copy_(current_q)
                        self._last_q_safe_seeded = True
                    elif not stateful_position_command:
                        if self._graph_control_step:
                            # In-place: rebinding would detach the buffer the
                            # captured graph writes.
                            self.last_q_safe.mul_(
                                1.0 - self.cbf_feedback_coupling
                            ).add_(current_q, alpha=self.cbf_feedback_coupling)
                        else:
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
                    dq_fb = self.zero_dq

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

                    # Masked (sync-free) hold: `if bool(tensor)` here forced a
                    # full stream drain every cycle, stalling the control
                    # thread behind whatever the GPU was chewing on.
                    nominal_hold_keep = (
                        torch.norm(dq_nom_torch) >= self.nominal_hold_deadband
                    ).to(dq_nom_torch.dtype)
                    dq_nom_torch = dq_nom_torch * nominal_hold_keep
                    if self.dq_nom_filtered is not None:
                        self.dq_nom_filtered.mul_(nominal_hold_keep)
                    dq_nom_base = dq_nom_torch.clone()
                    dq_escape = self.zero_dq
                    escape_active = False

                # Telemetry gating decided BEFORE the hot section so all
                # diagnostics-only tensor math (~25 eager dispatches) is
                # skipped on non-diagnostic cycles. Under CPU oversubscription
                # each dispatch costs ~50-100 us, which is where the
                # cbf_correction time actually went (cbf_d2h ~0.1 ms showed
                # the GPU itself was never the bottleneck).
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

                if self.profile_sync:
                    torch.cuda.synchronize()
                t_start = time.perf_counter()

                with torch.no_grad():
                    if need_telemetry:
                        self.transfer_buffer[39:].zero_()
                    if self._graph_control_step:
                        # ONE replay covers escape/clearance compose, solve,
                        # dynamic-hdot, repair, integration and the command
                        # buffer: ~15 eager dispatches per cycle instead of
                        # ~90, each of which paid GIL/dispatcher contention
                        # (runPP102: ~80 us/op, cbf_correction p50 16 ms with
                        # the GPU idle at readback).
                        local_obs = self._adopted_obs
                        esc_gate, esc_h_bias = self._escape_trigger_update()
                        escape_active = esc_gate > 0.01
                        self.static_q.copy_(current_q)
                        self.static_obs.copy_(local_obs)
                        self.static_dq_nom.copy_(dq_nom_torch)
                        self.static_nominal_q.copy_(nominal_q)
                        sn = self._scalar_np
                        # Proportional gate: the graph multiplies the whole
                        # composed escape (incl. suppression) by sc[0], so a
                        # fractional value ramps the authority smoothly.
                        sn[0] = esc_gate
                        sn[1] = (dt / (dt + self.cbf_velocity_filter_tau)
                                 if self.cbf_velocity_filter_tau > 0.0 else 0.0)
                        sn[2] = 1.0 / max(dt, 1e-3)
                        sn[3] = dt / (dt + self.cbf_dynamic_hdot_tau)
                        sn[4] = (esc_h_bias
                                 if self.cbf_escape_use_bias_corrected_h
                                 else 0.0)
                        # Measured joint speed for the executed-velocity
                        # margin linearization (EMA'd FD from joint_callback;
                        # atomic numpy ref grab, callback rebinds).
                        sn[5] = float(np.linalg.norm(self._dq_real_np))
                        self.static_scalars.copy_(self._scalar_pin,
                                                  non_blocking=True)
                        self.graph.replay()
                        self._queue_escape_state(self.static_dq_nom_cbf,
                                                 dq_out=self.static_dq_safe,
                                                 dq_nom_raw=self.static_dq_nom)
                        dq_escape = self.static_dq_escape_out
                    elif self.enable_cbf:
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
                        local_obs = self._adopted_obs  # event-complete selection
                        self.dq_safe_work.copy_(self._run_velocity_cbf_solver(
                            current_q, local_obs, dq_nom_cbf))
                        self._joint_limit_clamp_(self.dq_safe_work)
                        self._queue_escape_state(dq_nom_cbf,
                                                 dq_nom_raw=dq_nom_base)
                        self._update_dynamic_hdot(current_q, dt)
                        dq_nom_torch = dq_nom_cbf
                    else:
                        self.dq_safe_work.copy_(dq_nom_torch)

                    solver_out = None
                    if need_telemetry or self.cbf_monitor_only:
                        # In graph mode dq_safe_work is already the FINAL
                        # command; the raw solver output was exported to
                        # static_dq_safe inside the graph.
                        solver_out = (
                            self.static_dq_safe.detach().clone()
                            if self._graph_control_step
                            else self.dq_safe_work.detach().clone())
                    if self.cbf_monitor_only:
                        # Baseline mode: keep h/diagnostics (solver_out records
                        # the counterfactual correction) but apply nothing.
                        self.dq_safe_work.copy_(dq_nom_base)

                    if self.publish_viz_topics:
                        self._viz_q = current_q.detach().clone()
                        self._viz_dq_nom = dq_nom_base.detach().clone()
                        self._viz_dq_safe = self.dq_safe_work.detach().clone()

                    alignment = None
                    if need_telemetry:
                        dot = torch.sum(dq_nom_base * self.dq_safe_work, dim=1)
                        denom = torch.norm(dq_nom_base, dim=1) * torch.norm(self.dq_safe_work, dim=1)
                        alignment = torch.where(denom > 1e-8, dot / denom, torch.ones_like(dot))

                    if self._graph_control_step:
                        # Clamp, output filter, repair, integration, command
                        # buffer: all already applied inside the graph.
                        velocity_filter_active = self.cbf_velocity_filter_tau > 0.0
                    else:
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
                        if need_telemetry:
                            self.transfer_buffer[72:79].copy_(
                                self.dq_safe_work.squeeze(0))

                    if (self.enable_cbf and self.cbf_enforce_final_constraint
                            and not self.cbf_monitor_only
                            and not self._graph_control_step):
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
                        if need_telemetry:
                            self.transfer_buffer[60].copy_(
                                needs_repair.float().view(-1)[0])
                            self.transfer_buffer[61].copy_(
                                final_constr.view(-1)[0])

                    if self.profile_sync:
                        torch.cuda.synchronize()
                    t_cbf_done = time.perf_counter()
                    if self._graph_control_step:
                        # Integration, passthrough and the command buffer were all
                        # produced inside the graph replay.
                        dq_pub = self.dq_safe_work
                        h_now = self.static_h.detach() if self.enable_cbf else None
                        if self._contact_stopped:
                            # Latched contact stop: override the graphed command.
                            self.dq_safe_work.zero_()
                            dq_pub = self.dq_safe_work
                            self.q_safe_work.copy_(self._contact_q)
                            self.last_q_safe.copy_(self._contact_q)
                            self.command_buffer[0:7].zero_()
                            self.command_buffer[7:14].copy_(self._contact_q.squeeze(0))
                    else:
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
                # This is the ONE hard GPU wait of the control cycle, so
                # cbf_d2h below is the honest measure of GPU drain time
                # (solver execution + cross-process GPU contention).
                command_cpu = self.command_buffer.cpu().numpy()
                t_d2h_done = time.perf_counter()
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
                        int(self._adopted_count) > 0
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
                    elif constr_val < -1e-3:
                        # With cbf_repair_all_rows, constr_val is the worst
                        # residual across ALL active rows on the final
                        # published command: negative here means at least one
                        # row left violated after repair and clamping.
                        rospy.logwarn_throttle(
                            0.5, f"🛡️ CBF RESIDUAL: worst final row residual "
                            f"{constr_val:.4f} < 0 (h = {h_val:.4f})")
                    elif constr_val < 0:
                        rospy.loginfo_throttle(0.5, f"🛡️ CBF REACTIVE: Correcting trajectory (h = {h_val:.4f})")

                self.cmd_pub.publish(Float32MultiArray(data=dq_pub_cpu))
                # ZOH command of this cycle, for the next cycle's ISSf
                # disturbance monitor.
                self._issf_prev_cmd_np = np.asarray(
                    dq_pub_cpu[:7], dtype=np.float64)
                if self.velocity_cmd_pub is not None:
                    # dq_safe is the actuator command in velocity mode.  Publish
                    # it before diagnostics so telemetry cannot delay control.
                    self.velocity_cmd_pub.publish(
                        Float64MultiArray(data=dq_pub_cpu))

                pos_msg = Float64MultiArray(data=q_safe_work_cpu)
                self.safe_cmd_pub.publish(pos_msg)

                if (self.reflex_state_pub is not None and self.enable_cbf
                        and self._joint_uploaded_np is not None):
                    # Per-link (h, env_hdot, grad_h) snapshot for the 1 kHz
                    # reflex brake. The command readback above already drained
                    # the control stream, so this extra small D2H costs only
                    # the transfer itself. Rows never touched by the solver
                    # keep their far init (h=1, grad=0) and stay inert
                    # downstream (the reflex only acts on h < h_activate).
                    nr = self._reflex_n_rows
                    self._reflex_gpu[0:nr].copy_(self.static_h_links[:nr])
                    self._reflex_gpu[nr:2 * nr].copy_(
                        self.static_hdot_env_links[:nr])
                    self._reflex_gpu[2 * nr:].copy_(
                        self.static_grad_links[:nr].reshape(-1))
                    reflex_cpu = self._reflex_gpu.cpu().numpy()
                    # Trailing selection generation (backward-compatible:
                    # legacy parsers slice exact ranges and ignore it). Read
                    # without a lock: a swap racing this read misattributes
                    # at most one monitor pair, which the gate then skips or
                    # keeps as the legacy behavior did.
                    self.reflex_state_pub.publish(Float32MultiArray(
                        data=[current_time, float(nr)]
                        + self._joint_uploaded_np.tolist()
                        + reflex_cpu.tolist()
                        + [float(self._selection_generation)]))

                    # ISSf assumption monitor (see __init__): realized
                    # disturbance projected on the active barrier gradients,
                    # reusing the reflex snapshot already on the CPU.
                    if (self.issf_monitor_pub is not None
                            and self._issf_prev_cmd_np is not None):
                        u_eff = (self._issf_prev_cmd_np
                                 * self._reflex_alpha_seen)
                        d_vec = self._dq_real_np.astype(np.float64) - u_eff
                        h_rows = reflex_cpu[0:nr]
                        g_rows = reflex_cpu[2 * nr:].reshape(nr, 7)
                        g_norm = np.linalg.norm(g_rows, axis=1)
                        act = (h_rows < self.cbf_h_activate) & (g_norm > 1e-6)
                        eps_proj = float(np.max(
                            -(g_rows[act] @ d_vec) / g_norm[act])) \
                            if np.any(act) else 0.0
                        eps_norm = float(np.linalg.norm(d_vec))
                        # Exceedance only when the command actually executes
                        # (reflex not braking): during a brake, servo lag
                        # guarantees measured != alpha*u and the veto would
                        # re-trigger itself. The reflex's own certificate
                        # covers braked ticks. Statistics keep flowing either
                        # way for the ch. 5 panel.
                        exceeded = float(
                            self.cbf_issf_epsilon > 0.0
                            and self._reflex_alpha_seen > 0.9
                            and eps_proj > self.cbf_issf_epsilon)
                        if eps_proj > self._issf_monitor_max:
                            self._issf_monitor_max = eps_proj
                        self.issf_monitor_pub.publish(Float32MultiArray(
                            data=[current_time, eps_proj, eps_norm,
                                  exceeded, self._issf_monitor_max]))
                        if exceeded:
                            rospy.logwarn_throttle(
                                1.0,
                                "ISSf monitor: realized disturbance %.3f "
                                "rad/s exceeds epsilon %.3f on an active "
                                "row -- reflex will veto.",
                                eps_proj, self.cbf_issf_epsilon)
                t_pub_done = time.perf_counter()
                self.timing.publish('cbf_correction', (t_cbf_done - t_start) * 1000.0)
                self.timing.publish('cbf_command', (t_q_cmd_done - t_cbf_done) * 1000.0)
                self.timing.publish('cbf_d2h', (t_d2h_done - t_q_cmd_done) * 1000.0)
                self.timing.publish('cbf_total', (t_pub_done - t_start) * 1000.0)

                # diag_due / need_telemetry were decided before the hot
                # section (see above) so diagnostics-only tensor math could be
                # skipped there too.
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
                        self.transfer_buffer[79].copy_(self.static_sampled_margin)

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
                    env_hdot_val = 0.0
                    if self.enable_cbf and self.cbf_dynamic_hdot_enabled:
                        env_links = getattr(self, 'static_hdot_env_links', None)
                        if env_links is not None:
                            # Cheap: the transfer_buffer.cpu() above already
                            # drained the queue this cycle.
                            env_hdot_val = float(env_links.max().detach().cpu())
                    self.env_hdot_pub.publish(Float32(data=env_hdot_val))
                    if self.enable_cbf:
                        valid_obs_count = max(0, min(
                            int(self._adopted_count), int(local_obs.shape[0])))
                        with torch.no_grad():
                            h_hard_tensor = self._hardmin_h_value_no_grad(
                                current_q.detach(), local_obs[:valid_obs_count])
                        h_hard_val = float(h_hard_tensor.detach().view(-1)[0].cpu())
                        softmin_gap_val = h_hard_val - h_val
                    else:
                        h_hard_val = nan
                        softmin_gap_val = nan
                    if (self.enable_cbf
                            and self.cbf_solver_mode in ("fast_tangent", "multi_graphed")
                            and not self._barrier_value_hard):
                        h_corr_val = h_val + float(self.barrier.alpha) * float(
                            np.log(max(1, int(self._adopted_count))))
                    else:
                        # Keep the conservative per-link Softmin value in the
                        # analytical multi_projected path; legacy multi modes are
                        # hard-min constraints and need no correction either.
                        h_corr_val = h_val
                    solver_out_v = cpu_buffer[53:60]
                    dq_base_v = cpu_buffer[26:33]
                    repair_on = self.enable_cbf and self.cbf_enforce_final_constraint
                    min_obs_d = float(self._adopted_min_dist)
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
                        float(self._escape_gate),  # proportional gate [0,1]
                        0.0,  # ema_applied
                        1.0 if velocity_filter_active else 0.0,
                        diag_vel_source,
                        float(velocity_age) if np.isfinite(velocity_age) else nan,
                        float(dt_measured),
                        float(self._adopted_count),
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
                        float(cpu_buffer[79]) if self.enable_cbf else nan,  # sampled_margin
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
