#!/usr/bin/env python3

import rospy
import rospkg
import sys
import time
import torch
import numpy as np
import struct
import xacro
import tempfile
from std_msgs.msg import Float32MultiArray, Float64MultiArray, Header
from sensor_msgs.msg import JointState, PointCloud2, PointField
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs.point_cloud2 as pc2

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
        link_names = ['panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 'panda_hand', 'panda_leftfinger', 'panda_rightfinger', 'fork_tip']
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
        # Recommended range: 0.003–0.005 for N=100 (trigger ≈ 18–28 mm).
        barrier_alpha = float(rospy.get_param("~barrier_alpha", 0.005))
        self.barrier = BernsteinBarrier(self.bernstein_core, d_safe=self.d_safe, alpha=barrier_alpha)
        self._log_cuda_memory("CBF after Bernstein barrier")

        # kappa: class-K coefficient in the CBF constraint ∇h·dq + κh ≥ 0.
        # High kappa forces hard corrections even when h is slightly positive (safe).
        # With a jumpy gradient, high kappa amplifies oscillation near obstacles.
        # Recommended: 1.0–3.0. Must be read BEFORE setup_cuda_graph (baked into graph).
        self.cbf_kappa = float(rospy.get_param("~cbf_kappa", 2.0))

        # Setup the new fast CUDA graph
        self.setup_cuda_graph(batch_size=1, n_points=100)
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
        self.enable_cbf = rospy.get_param("~enable_cbf", True)
        self.publish_controller_command = rospy.get_param("~publish_controller_command", False)
        self.preprocess_rate_hz = float(rospy.get_param("~preprocess_rate_hz", 30.0))
        self.viz_rate_hz = float(rospy.get_param("~viz_rate_hz", 5.0))
        self.publish_debug_topics = bool(rospy.get_param("~publish_debug_topics", False))
        self.publish_viz_topics = bool(rospy.get_param("~publish_viz_topics", True))
        self.profile_sync = bool(rospy.get_param("~profile_sync", False))
        self.cuda_memory_log_period = float(rospy.get_param("~cuda_memory_log_period", 10.0))
        self._last_cuda_memory_log = 0.0
        self.tcp_filter_radius = float(rospy.get_param("~tcp_filter_radius", 0.40))
        self.fork_filter_radius = float(rospy.get_param("~fork_filter_radius", 0.15))
        self.nominal_hold_deadband = float(rospy.get_param("~nominal_hold_deadband", 1e-4))

        self.eye4 = torch.eye(4, device=self.device).unsqueeze(0)
        self.q_pad2 = torch.zeros((1, 2), device=self.device)
        self.dummy100 = torch.full((100, 3), 100.0, device=self.device)
        self.zero_dq = torch.zeros((1, 7), device=self.device)

        self.selected_obs = self.dummy100.clone()
        self.selected_pts_yellow = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.selected_num_inside = 0
        self.selected_count = 0
        self.selected_min_obs_dist = float('inf')

        self.last_q_safe = None
        self.q_safe_work = torch.zeros((1, 7), device=self.device)
        self.dq_safe_work = torch.zeros((1, 7), device=self.device)
        self._log_cuda_memory("CBF after runtime buffers")

        self.comp_times = []
        rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        rospy.Subscriber('/planner/nominal_joint_command', Float64MultiArray, self.nominal_command_callback)
        obs_topic = rospy.get_param("~obs_topic", "/perception/persistent_obstacles")
        rospy.Subscriber(obs_topic, PointCloud2, self.obs_callback)
        
        self.cmd_pub = rospy.Publisher('/franka_control/safe_joint_velocities', Float32MultiArray, queue_size=1)
        self.pos_cmd_pub = None
        if self.publish_controller_command:
            self.pos_cmd_pub = rospy.Publisher('/joint_group_position_controller/command', Float64MultiArray, queue_size=1)
        self.safe_cmd_pub = rospy.Publisher('/planner/safe_joint_command', Float64MultiArray, queue_size=1)
        self.debug_input_pub = rospy.Publisher('/debug/cbf/input_joint_command_seen', Float64MultiArray, queue_size=1)
        self.debug_output_pub = rospy.Publisher('/debug/cbf/output_joint_command', Float64MultiArray, queue_size=1)
        self.debug_input_vel_pub = rospy.Publisher('/debug/cbf/input_joint_velocity', Float32MultiArray, queue_size=1)
        self.debug_output_vel_pub = rospy.Publisher('/debug/cbf/output_joint_velocity', Float32MultiArray, queue_size=1)
        self.debug_alignment_pub = rospy.Publisher('/debug/cbf/nominal_safe_alignment', Float32MultiArray, queue_size=1)
        
        # Publishers RViz (Uniquement Jaune et Rouge)
        self.pub_inside_yellow = rospy.Publisher('/viz/obs_inside_yellow', PointCloud2, queue_size=1)
        self.pub_top100_red = rospy.Publisher('/viz/obs_top100_red', PointCloud2, queue_size=1)
        self.safe_traj_viz_pub = rospy.Publisher('/viz/safe_trajectory_3d', MarkerArray, queue_size=1)
        
        # FRÉQUENCE STABLE À 150 Hz
        self.rate_hz = 150
        self.rate = rospy.Rate(self.rate_hz)
        self.last_time = rospy.get_time()
        rospy.Timer(rospy.Duration(1.0 / self.preprocess_rate_hz),
                    self.preprocess_obstacles)
        if self.publish_viz_topics:
            rospy.Timer(rospy.Duration(1.0 / self.viz_rate_hz),
                        self.publish_visualization)
        rospy.loginfo(
            "Optimized CBF ready: control=%.1f Hz preprocess=%.1f Hz viz=%.1f Hz "
            "debug=%s profile_sync=%s",
            self.rate_hz, self.preprocess_rate_hz, self.viz_rate_hz,
            self.publish_debug_topics, self.profile_sync)

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

    def setup_cuda_graph(self, batch_size=1, n_points=100):
        """
        Single unified CUDA graph:
          Calculates whole-body safety over 100 points.
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

        # Capture kappa as a Python float so the CUDA graph bakes in the param value.
        kappa = self.cbf_kappa

        def _qp_step(h, grad_h, dq_in):
            self.static_h.copy_(h)
            constr = (grad_h * dq_in).sum(dim=-1) + kappa * h
            self.static_constr.copy_(constr)
            denom  = (grad_h ** 2).sum(dim=-1).clamp(min=1e-8)
            lam    = torch.where(constr < 0, -constr / denom, torch.zeros_like(constr))
            return dq_in + lam.unsqueeze(-1) * grad_h

        # ── Warmup ────────────────────────────────────────────────────────────
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                if self.static_q.grad is not None: self.static_q.grad.zero_()
                h, g, _ = self.barrier(self.static_q, self.static_obs)
                self.static_dq_safe.copy_(_qp_step(h, g, self.static_dq_nom))
        torch.cuda.current_stream().wait_stream(s)

        # ── Unified Graph ─────────────────────────────────────────────────────
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            if self.static_q.grad is not None: self.static_q.grad.zero_()
            h, g, _ = self.barrier(self.static_q, self.static_obs)
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

    def nominal_command_callback(self, msg):
        if len(msg.data) >= 7:
            self.nominal_q = torch.tensor(msg.data[:7], dtype=torch.float32,
                                          device=self.device).unsqueeze(0)
            if not self._logged_first_nominal:
                if self.current_q is None:
                    err = float('nan')
                else:
                    err = float(torch.norm(self.nominal_q - self.current_q).item())
                rospy.loginfo(
                    "CBF received first nominal joint command | err_to_current=%.4f rad",
                    err)
                self._logged_first_nominal = True

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

    def get_fork_tip_pose(self, q7):
        if q7.dim() == 1:
            q7 = q7.unsqueeze(0)
        q_eval = q7
        if q_eval.shape[-1] == 7:
            q_eval = torch.cat([
                q_eval,
                torch.zeros((q_eval.shape[0], 2), device=self.device)
            ], dim=-1)

        link_poses = self.robot_layer._native_forward_kinematics(q_eval)
        # Use fork_tine_tip (physical tine centroid) so the green marker
        # appears at the actual tine tips, not the fork attachment point.
        return link_poses.get('fork_tine_tip', link_poses.get('fork_tip', None))

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

    def _copy_fixed_obstacles(self, obs):
        self.selected_obs.copy_(self.dummy100)
        n = min(int(obs.shape[0]), 100)
        if n > 0:
            self.selected_obs[:n].copy_(obs[:n])
        self.selected_count = n

    def preprocess_obstacles(self, event):
        if self.current_q is None:
            return

        pts = self.obs_points
        if pts is None or pts.shape[0] == 0:
            self._copy_fixed_obstacles(torch.empty((0, 3), device=self.device))
            self.selected_pts_yellow = torch.empty((0, 3), dtype=torch.float32,
                                                   device=self.device)
            self.selected_num_inside = 0
            self.selected_min_obs_dist = float('inf')
            return

        try:
            with torch.no_grad():
                T_fork_now = self.get_fork_tip_pose(self.current_q)
                if T_fork_now is None:
                    rospy.logwarn_throttle(
                        5, "CBF preprocessing cannot find fork_tine_tip/fork_tip in FK tree.")
                    return

                x_now_pos = T_fork_now[:, :3, 3]
                dist_tcp = torch.norm(pts - x_now_pos, dim=1)
                pts_inside = pts[dist_tcp < self.tcp_filter_radius]
                num_inside = int(pts_inside.shape[0])

                if num_inside == 0:
                    self._copy_fixed_obstacles(torch.empty((0, 3), device=self.device))
                    self.selected_pts_yellow = torch.empty((0, 3), dtype=torch.float32,
                                                           device=self.device)
                    self.selected_num_inside = 0
                    self.selected_min_obs_dist = float('inf')
                    return

                q9 = torch.cat([self.current_q.detach(), self.q_pad2], dim=1)
                _, sdf_per_link = self.bernstein_core.get_whole_body_sdf_batch(
                    pts_inside, self.eye4, q9, return_per_link=True)

                # Pruned self-filter: indices 0:5 now correspond to [link4, link5, link6, link7, hand]
                sdf_body = sdf_per_link[0, :5, :].min(dim=0).values
                sdf_all = sdf_per_link[0].min(dim=0).values
                not_self = sdf_body > -0.003
                pts_inside = pts_inside[not_self]
                sdf_filtered = sdf_all[not_self]
                num_inside = int(pts_inside.shape[0])

                if num_inside == 0:
                    self._copy_fixed_obstacles(torch.empty((0, 3), device=self.device))
                    self.selected_pts_yellow = torch.empty((0, 3), dtype=torch.float32,
                                                           device=self.device)
                    self.selected_num_inside = 0
                    self.selected_min_obs_dist = float('inf')
                    return

                if num_inside >= 100:
                    _, top_idx = torch.topk(sdf_filtered, k=100, largest=False)
                    obs = pts_inside[top_idx]
                    top_mask = torch.zeros(num_inside, dtype=torch.bool,
                                           device=self.device)
                    top_mask[top_idx] = True
                    pts_yellow = pts_inside[~top_mask]
                else:
                    obs = pts_inside
                    pts_yellow = torch.empty((0, 3), dtype=torch.float32,
                                             device=self.device)

                dist_tip = torch.norm(pts_inside - x_now_pos, dim=1)
                self._copy_fixed_obstacles(obs.contiguous())
                self.selected_pts_yellow = pts_yellow.detach()
                self.selected_num_inside = num_inside
                self.selected_min_obs_dist = float(dist_tip.min().item())
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

            if msg_yellow is not None:
                self.pub_inside_yellow.publish(msg_yellow)
            else:
                msg_yellow = PointCloud2()
                msg_yellow.header.stamp = rospy.Time.now()
                msg_yellow.header.frame_id = "world"
                self.pub_inside_yellow.publish(msg_yellow)

            if msg_red is not None:
                self.pub_top100_red.publish(msg_red)

            self.publish_safe_trajectory_marker(self.current_q, self.last_q_safe)
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error publishing CBF visualization: {e}")

    def run(self):
        # print(f"🛡️  CBF NODE : Bouclier actif à {self.rate_hz}Hz. Sécurités activées.")
        dt_fixed = 1.0 / self.rate_hz

        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            dt = current_time - self.last_time
            self.last_time = current_time

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
                    if self.last_q_safe is None:
                        self.last_q_safe = current_q.clone()

                    dq_nom_torch = (nominal_q - self.last_q_safe) / dt
                    nominal_is_hold = torch.norm(dq_nom_torch) < self.nominal_hold_deadband
                if self.publish_debug_topics:
                    self.debug_input_pub.publish(Float64MultiArray(
                        data=nominal_q.squeeze(0).cpu().numpy().tolist()))
                    self.debug_input_vel_pub.publish(Float32MultiArray(
                        data=dq_nom_torch.squeeze(0).cpu().numpy().tolist()))

                if self.profile_sync:
                    torch.cuda.synchronize()
                t_start = time.perf_counter()

                with torch.no_grad():
                    if nominal_is_hold:
                        self.q_safe_work.copy_(nominal_q)
                        self.last_q_safe.copy_(self.q_safe_work)
                        self.dq_safe_work.zero_()
                    elif self.enable_cbf:
                        self.static_q.copy_(current_q)
                        self.static_obs.copy_(self.selected_obs)
                        self.static_dq_nom.copy_(dq_nom_torch)

                        self.graph.replay()
                        self.dq_safe_work.copy_(self.static_dq_safe.detach())
                        
                        # Logging CBF violations
                        h_val = float(self.static_h.item())
                        constr_val = float(self.static_constr.item())
                        if h_val < 0:
                            rospy.logwarn_throttle(0.5, f"💥 CBF COLLISION: Margin breached! (h = {h_val:.4f} < 0)")
                        elif constr_val < 0:
                            rospy.loginfo_throttle(0.5, f"🛡️ CBF REACTIVE: Correcting trajectory (h = {h_val:.4f})")
                    else:
                        self.dq_safe_work.copy_(dq_nom_torch)

                    if nominal_is_hold:
                        alignment = torch.ones((1,), device=self.device)
                    else:
                        dot = torch.sum(dq_nom_torch * self.dq_safe_work, dim=1)
                        denom = torch.norm(dq_nom_torch, dim=1) * torch.norm(self.dq_safe_work, dim=1)
                        alignment = torch.where(denom > 1e-8, dot / denom, torch.ones_like(dot))

                        reverse = (alignment < 0.0).view(1, 1)
                        self.dq_safe_work.copy_(torch.where(reverse, self.zero_dq, self.dq_safe_work))
                        self.dq_safe_work.clamp_(min=-self.max_joint_velocity,
                                                 max=self.max_joint_velocity)

                        torch.add(self.last_q_safe, self.dq_safe_work, alpha=dt,
                                  out=self.q_safe_work)
                        self.last_q_safe.copy_(self.q_safe_work)


                if self.profile_sync:
                    torch.cuda.synchronize()
                self.comp_times.append(time.perf_counter() - t_start)
                self._log_cuda_memory_throttle("CBF runtime")
                if len(self.comp_times) >= 20:
                    avg_t = sum(self.comp_times) / len(self.comp_times)
                    rospy.loginfo_throttle(
                        5.0,
                        f"⏱️ [TIMING] CBF hot loop: {avg_t*1000:.3f} ms | "
                        f"selected_obs={self.selected_num_inside}")
                    self.comp_times.clear()

                dq_pub = (self.q_safe_work - current_q) / dt
                self.cmd_pub.publish(Float32MultiArray(
                    data=dq_pub.detach().squeeze(0).cpu().numpy().tolist()))

                pos_msg = Float64MultiArray(
                    data=self.q_safe_work.detach().squeeze(0).cpu().numpy().tolist())
                if not self._logged_first_safe_publish:
                    safe_err = float(torch.norm(self.q_safe_work - current_q).item())
                    nominal_err = float(torch.norm(nominal_q - current_q).item())
                    rospy.loginfo(
                        "CBF publishing first safe joint command | "
                        "safe_err_to_current=%.4f rad nominal_err_to_current=%.4f rad",
                        safe_err, nominal_err)
                    self._logged_first_safe_publish = True
                self.safe_cmd_pub.publish(pos_msg)

                if self.publish_debug_topics:
                    self.debug_output_pub.publish(pos_msg)
                    self.debug_output_vel_pub.publish(Float32MultiArray(
                        data=dq_pub.detach().squeeze(0).cpu().numpy().tolist()))
                    self.debug_alignment_pub.publish(Float32MultiArray(data=[
                        float(alignment.item()),
                        float(torch.norm(dq_nom_torch).item()),
                        float(torch.norm(self.dq_safe_work).item())
                    ]))

                if self.pos_cmd_pub is not None:
                    self.pos_cmd_pub.publish(pos_msg)
                
            self.rate.sleep()

if __name__ == '__main__':
    node = CBFSafetyNode()
    node.run()
