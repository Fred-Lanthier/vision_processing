#!/usr/bin/env python3
"""
Time-based joint trajectory executor.

Subscribes to /planner/nominal_trajectory (JointTrajectory).
On each new trajectory, compensates for planning latency by finding the
waypoint closest to the current robot state and anchoring the clock there.
Publishes linearly-interpolated joint positions at a fixed rate.
A monotone guard prevents backward jumps within a trajectory.
"""
import rospy
import numpy as np
import threading
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Float64MultiArray, Float32MultiArray

JOINT_NAMES = [f'panda_joint{i}' for i in range(1, 8)]


class TrajectoryExecutor:
    def __init__(self):
        rospy.init_node('trajectory_executor')

        # Arm joint names for the /joint_states lookup. Default = panda; the
        # xArm7 cross-embodiment launch passes xarm7_joint1..7.
        self.joint_names = list(rospy.get_param('~joint_names', JOINT_NAMES))
        self.rate_hz      = rospy.get_param('~rate_hz',      50.0)
        self.hold_current_until_trajectory = rospy.get_param(
            '~hold_current_until_trajectory', False)
        self.log_events = bool(rospy.get_param('~log_events', True))
        self.log_timing = bool(rospy.get_param('~log_timing', True))
        self.motion_start_threshold_rad = float(rospy.get_param(
            '~motion_start_threshold_rad', 0.003))
        self.goal_reached_threshold_rad = float(rospy.get_param(
            '~goal_reached_threshold_rad', 0.025))
        self.goal_reached_speed_threshold = float(rospy.get_param(
            '~goal_reached_speed_threshold', 0.02))
        self.use_progress_clock = bool(rospy.get_param(
            '~use_progress_clock', True))
        self.progress_lag_limit_rad = float(rospy.get_param(
            '~progress_lag_limit_rad', 0.08))
        self.command_lookahead_time = float(rospy.get_param(
            '~command_lookahead_time', 0.0))
        self.anchor_mode = rospy.get_param('~anchor_mode', 'closest').lower()
        if self.anchor_mode not in ('closest', 'start'):
            raise ValueError("~anchor_mode must be 'closest' or 'start'")
        # Topic that receives the interpolated nominal command.
        # Set to /planner/nominal_joint_command to route through the CBF,
        # or to /joint_group_position_controller/command to bypass it.
        self.out_topic    = rospy.get_param('~out_topic',
                                            '/joint_group_position_controller/command')
        self.velocity_out_topic = rospy.get_param(
            '~velocity_out_topic', '/planner/nominal_joint_velocity')

        self.traj_pts  = None   # list of (t_sec, q_7d)
        self.t_start   = None
        self.q_current = None
        self.q_hold    = None   # latched startup position
        self._min_t    = 0.0   # monotone guard: never search before this time
        self._holding_logged = False
        self._logged_first_traj = False
        self._logged_first_publish = False
        self._local_plan_seq = 0
        self._active_plan = None
        self._last_joint_q = None
        self._last_joint_stamp = None
        self._exec_t = 0.0
        self._last_control_stamp = None
        self._state_lock = threading.Lock()

        # CBF h-value freeze: when h < cbf_hold_threshold, freeze the progress
        # clock regardless of lag error so the nominal does not race ahead while
        # the CBF is recovering from a constraint violation.
        self.cbf_hold_threshold = float(rospy.get_param('~cbf_hold_threshold', 0.0))
        self._cbf_h_value = float('inf')  # safe default until first message

        rospy.Subscriber('/planner/nominal_trajectory', JointTrajectory,
                         self._traj_cb, queue_size=1)
        rospy.Subscriber('/joint_states', JointState,
                         self._joint_cb, queue_size=1)
        rospy.Subscriber('/cbf_safety/h_value', Float32MultiArray,
                         self._cbf_h_cb, queue_size=1)

        self.pub = rospy.Publisher(self.out_topic, Float64MultiArray, queue_size=1)
        self.vel_pub = rospy.Publisher(
            self.velocity_out_topic, Float64MultiArray, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._control_loop)

        if self.log_events:
            rospy.loginfo('trajectory_executor ready → %s and %s at %.0f Hz',
                          self.out_topic, self.velocity_out_topic, self.rate_hz)

    # ── callbacks ──────────────────────────────────────────────────────────

    def _cbf_h_cb(self, msg):
        if msg.data:
            self._cbf_h_value = float(msg.data[0])

    def _joint_cb(self, msg):
        now = rospy.Time.now()
        pos = {n: p for n, p in zip(msg.name, msg.position)}
        q = [pos.get(j) for j in self.joint_names]
        if None not in q:
            q_current = np.array(q, dtype=np.float64)
            with self._state_lock:
                if self._last_joint_q is not None and self._last_joint_stamp is not None:
                    dt = max((now - self._last_joint_stamp).to_sec(), 1e-6)
                    q_speed = float(np.linalg.norm(q_current - self._last_joint_q) / dt)
                else:
                    q_speed = 0.0
                self._last_joint_q = q_current.copy()
                self._last_joint_stamp = now
                self.q_current = q_current
                if self.q_hold is None:
                    self.q_hold = q_current.copy()
                active = self._active_plan

                if self.log_timing and active is not None:
                    if not active['actual_start_logged']:
                        moved = float(np.linalg.norm(q_current - active['q_start']))
                        if moved >= self.motion_start_threshold_rad:
                            active['actual_start_logged'] = True
                            active['actual_start_time'] = now
                            since_publish = (now - active['publish_stamp']).to_sec()
                            since_receive = (now - active['receive_time']).to_sec()
                            if active['first_nominal_time'] is None:
                                since_first_nominal = float('nan')
                            else:
                                since_first_nominal = (
                                    now - active['first_nominal_time']).to_sec()
                            rospy.loginfo(
                                "[TRAJ TIMING] actual_start id=%d "
                                "publish_to_start=%.3fs receive_to_start=%.3fs "
                                "first_nominal_to_start=%.3fs moved=%.4frad speed=%.4frad/s",
                                active['id'],
                                since_publish,
                                since_receive,
                                since_first_nominal,
                                moved,
                                q_speed,
                            )

                    if active['actual_start_logged'] and not active['actual_done_logged']:
                        final_err = float(np.linalg.norm(q_current - active['q_final']))
                        if (
                            final_err <= self.goal_reached_threshold_rad
                            and q_speed <= self.goal_reached_speed_threshold
                        ):
                            active['actual_done_logged'] = True
                            since_publish = (now - active['publish_stamp']).to_sec()
                            motion_time = (
                                now - active['actual_start_time']).to_sec()
                            start_delay = max(0.0, since_publish - motion_time)
                            speed_ratio = (
                                motion_time / active['duration']
                                if active['duration'] > 1e-6 else float('nan')
                            )
                            rospy.loginfo(
                                "[TRAJ TIMING] actual_done id=%d "
                                "publish_to_done=%.3fs actual_motion_duration=%.3fs "
                                "expected_duration=%.3fs start_delay=%.3fs "
                                "actual_over_expected=%.2f final_err=%.4frad speed=%.4frad/s",
                                active['id'],
                                since_publish,
                                motion_time,
                                active['duration'],
                                start_delay,
                                speed_ratio,
                                final_err,
                                q_speed,
                            )

    def _traj_cb(self, msg):
        if not msg.points:
            return
        pts = [(p.time_from_start.to_sec(),
                np.array(p.positions[:7], dtype=np.float64))
               for p in msg.points]
        pts.sort(key=lambda x: x[0])
        receive_time = rospy.Time.now()
        publish_stamp = msg.header.stamp
        if publish_stamp.to_sec() <= 0.0:
            publish_stamp = receive_time
        plan_id = int(getattr(msg.header, 'seq', 0))
        if plan_id <= 0:
            self._local_plan_seq += 1
            plan_id = self._local_plan_seq

        with self._state_lock:
            q_current = None if self.q_current is None else self.q_current.copy()

        # In reactive mode the planner already starts trajectories from the
        # measured state. Closest-waypoint anchoring can jump onto a later or
        # opposite branch when the CBF has deviated around an obstacle, creating
        # a backwards nominal tangent.
        if q_current is not None and self.anchor_mode == 'closest':
            qs = np.array([q for _, q in pts])
            dists = np.linalg.norm(qs - q_current, axis=1)
            closest_idx = int(np.argmin(dists))
            closest_t = pts[closest_idx][0]
            t_start = receive_time - rospy.Duration(closest_t)
            closest_err = float(dists[closest_idx])
        elif q_current is not None:
            closest_idx = 0
            closest_t = pts[0][0]
            t_start = receive_time - rospy.Duration(closest_t)
            closest_err = float(np.linalg.norm(pts[0][1] - q_current))
        else:
            t_start = receive_time
            closest_idx = -1
            closest_t = 0.0
            closest_err = float('nan')

        with self._state_lock:
            self.t_start = t_start
            self.traj_pts = pts
            self._min_t = 0.0   # new trajectory resets the monotone guard
            self._exec_t = float(closest_t)
            self._last_control_stamp = receive_time
            q_start = q_current.copy() if q_current is not None else pts[0][1].copy()
            self._active_plan = {
                'id': plan_id,
                'publish_stamp': publish_stamp,
                'receive_time': receive_time,
                'duration': float(pts[-1][0]),
                'exec_t_start': float(closest_t),
                'q_start': q_start,
                'q_final': pts[-1][1].copy(),
                'first_nominal_logged': False,
                'first_nominal_time': None,
                'nominal_end_logged': False,
                'actual_start_logged': False,
                'actual_start_time': None,
                'actual_done_logged': False,
            }

        if self.log_timing:
            rospy.loginfo(
                "[TRAJ TIMING] received id=%d points=%d expected_duration=%.3fs "
                "publish_to_receive=%.3fs closest_idx=%d closest_t=%.3fs "
                "closest_err=%.4frad anchor=%s progress_clock=%s lag_limit=%.3frad",
                plan_id,
                len(pts),
                pts[-1][0],
                (receive_time - publish_stamp).to_sec(),
                closest_idx,
                closest_t,
                closest_err,
                self.anchor_mode,
                self.use_progress_clock,
                self.progress_lag_limit_rad,
            )

        if self.log_events and not self._logged_first_traj:
            rospy.loginfo(
                'trajectory_executor received first trajectory: points=%d '
                'duration=%.3fs closest_idx=%d closest_t=%.3fs closest_err=%.4f rad',
                len(pts), pts[-1][0], closest_idx, closest_t, closest_err)
            self._logged_first_traj = True

    # ── interpolation ──────────────────────────────────────────────────────

    def _interpolate(self, t, pts):
        if t <= pts[0][0]:
            return pts[0][1]
        if t >= pts[-1][0]:
            return pts[-1][1]
        for i in range(len(pts) - 1):
            t0, q0 = pts[i]
            t1, q1 = pts[i + 1]
            if t0 <= t <= t1:
                alpha = (t - t0) / max(t1 - t0, 1e-9)
                return q0 + alpha * (q1 - q0)
        return pts[-1][1]

    def _velocity_at(self, t, pts):
        if len(pts) < 2 or t >= pts[-1][0]:
            return np.zeros(7, dtype=np.float64)
        if t <= pts[0][0]:
            t0, q0 = pts[0]
            t1, q1 = pts[1]
            return (q1 - q0) / max(t1 - t0, 1e-9)
        for i in range(len(pts) - 1):
            t0, q0 = pts[i]
            t1, q1 = pts[i + 1]
            if t0 <= t <= t1:
                return (q1 - q0) / max(t1 - t0, 1e-9)
        return np.zeros(7, dtype=np.float64)

    # ── control loop ───────────────────────────────────────────────────────

    def _control_loop(self, event):
        with self._state_lock:
            q_current = None if self.q_current is None else self.q_current.copy()
            q_hold = None if self.q_hold is None else self.q_hold.copy()
            traj_pts = self.traj_pts
            t_start = self.t_start
            min_t = self._min_t
            exec_t = self._exec_t
            last_control_stamp = self._last_control_stamp

        if q_current is None or q_hold is None:
            return

        if traj_pts is None or t_start is None:
            if self.hold_current_until_trajectory:
                self.pub.publish(Float64MultiArray(data=q_hold.tolist()))
                self.vel_pub.publish(Float64MultiArray(data=[0.0] * 7))
                if not self._holding_logged:
                    rospy.logdebug(
                        'trajectory_executor holding latched joints until first trajectory')
                    self._holding_logged = True
            return

        now = rospy.Time.now()
        if self.use_progress_clock:
            if last_control_stamp is None:
                dt_exec = 1.0 / max(self.rate_hz, 1e-6)
            else:
                dt_exec = max((now - last_control_stamp).to_sec(), 0.0)
            dt_exec = min(dt_exec, 3.0 / max(self.rate_hz, 1e-6))

            q_progress = self._interpolate(exec_t, traj_pts)
            lag_err = float(np.linalg.norm(q_progress - q_current))
            cbf_hold = self._cbf_h_value < self.cbf_hold_threshold
            if lag_err <= self.progress_lag_limit_rad and not cbf_hold:
                t = min(exec_t + dt_exec, traj_pts[-1][0])
            else:
                t = exec_t
            t_cmd = min(t + self.command_lookahead_time, traj_pts[-1][0])
            q_des = self._interpolate(t_cmd, traj_pts)
            dq_des = self._velocity_at(t_cmd, traj_pts)
            with self._state_lock:
                if traj_pts is self.traj_pts:
                    self._exec_t = t
                    self._last_control_stamp = now
                    self._min_t = t
        else:
            t = (now - t_start).to_sec()

            # Monotone guard: never go backward in the trajectory.
            # Protects against the closest-waypoint search landing on a later section
            # of a new trajectory that briefly passes near the current joint config.
            t = max(t, min_t)
            lag_err = float(np.linalg.norm(self._interpolate(t, traj_pts) - q_current))
            with self._state_lock:
                if traj_pts is self.traj_pts:
                    self._min_t = t
            t_cmd = min(t + self.command_lookahead_time, traj_pts[-1][0])
            q_des = self._interpolate(t_cmd, traj_pts)
            dq_des = self._velocity_at(t_cmd, traj_pts)

        active = None
        if self.log_timing:
            with self._state_lock:
                active = self._active_plan
                if active is not None and not active['first_nominal_logged']:
                    now = rospy.Time.now()
                    active['first_nominal_logged'] = True
                    active['first_nominal_time'] = now
                    err = float(np.linalg.norm(q_des - q_current))
                    rospy.loginfo(
                        "[TRAJ TIMING] first_nominal id=%d "
                        "publish_to_first_nominal=%.3fs receive_to_first_nominal=%.3fs "
                        "traj_t=%.3fs cmd_t=%.3fs expected_duration=%.3fs err_to_current=%.4frad "
                        "progress_clock=%s lag_err=%.4frad",
                        active['id'],
                        (now - active['publish_stamp']).to_sec(),
                        (now - active['receive_time']).to_sec(),
                        t,
                        t_cmd,
                        active['duration'],
                        err,
                        self.use_progress_clock,
                        lag_err,
                    )
                if (
                    active is not None
                    and not active['nominal_end_logged']
                    and t >= active['duration']
                ):
                    now = rospy.Time.now()
                    active['nominal_end_logged'] = True
                    rospy.loginfo(
                        "[TRAJ TIMING] nominal_end id=%d "
                        "publish_to_nominal_end=%.3fs receive_to_nominal_end=%.3fs "
                        "expected_duration=%.3fs",
                        active['id'],
                        (now - active['publish_stamp']).to_sec(),
                        (now - active['receive_time']).to_sec(),
                        active['duration'],
                    )
        if self.log_events and not self._logged_first_publish:
            err = float(np.linalg.norm(q_des - q_current))
            rospy.loginfo(
                'trajectory_executor publishing first nominal command | '
                't=%.3fs cmd_t=%.3fs err_to_current=%.4f rad topic=%s',
                t, t_cmd, err, self.out_topic)
            self._logged_first_publish = True
        self.pub.publish(Float64MultiArray(data=q_des.tolist()))
        self.vel_pub.publish(Float64MultiArray(data=dq_des.tolist()))

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    TrajectoryExecutor().run()
