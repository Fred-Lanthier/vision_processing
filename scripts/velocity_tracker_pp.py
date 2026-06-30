#!/usr/bin/env python3
"""
velocity_tracker_pp.py — closed-loop joint-velocity tracker for PICK-AND-PLACE.

Replicates the *tracking* law the CBF node runs in green_cube_feeding_casf.launch
(feedforward nominal velocity + position-error feedback + low-pass filter), WITHOUT
the fork-coupled CBF safety machinery. This replaces the open-loop bypass that fed
the executor's raw velocity straight to the controller — open loop never corrects
lag, so any disturbance (e.g. the welded cube's load after grasp) accumulated and
the arm fell behind the nominal path.

Command law (matches cbf_safety_node_Bernstein_multicbf.py):
    dq_ff  = ff_gain * nominal_dq                       (from /planner/nominal_joint_velocity)
    dq_fb  = clamp(kp * (nominal_q - current_q), ±vfb)  (position feedback)
    dq_raw = dq_ff + dq_fb
    dq     = lowpass(dq_raw, tau);  clamp(±max_joint_velocity)
-> Float64MultiArray on /joint_group_velocity_controller/command.

When no nominal command arrives within command_timeout, it commands zero (hold).
This is a stop-gap until the CBF `_pp` adaptation (obstacle stage) takes over both
tracking AND safety.
"""
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

ARM_JOINTS = ['panda_joint1', 'panda_joint2', 'panda_joint3',
              'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']


class VelocityTrackerPP:
    def __init__(self):
        rospy.init_node('velocity_tracker_pp')
        self.rate_hz = float(rospy.get_param('~rate_hz', 100.0))
        self.kp = float(rospy.get_param('~cbf_kp', 5.0))
        self.max_feedback_velocity = float(rospy.get_param('~max_feedback_velocity', 0.5))
        self.ff_gain = float(rospy.get_param('~nominal_feedforward_gain', 1.0))
        self.filter_tau = float(rospy.get_param('~filter_tau', 0.05))
        self.max_joint_velocity = float(rospy.get_param('~max_joint_velocity', 0.7))
        self.command_timeout = float(rospy.get_param('~command_timeout', 0.25))
        self.nominal_cmd_topic = rospy.get_param('~nominal_command_topic', '/planner/nominal_joint_command')
        self.nominal_vel_topic = rospy.get_param('~nominal_velocity_topic', '/planner/nominal_joint_velocity')
        self.out_topic = rospy.get_param('~out_topic', '/joint_group_velocity_controller/command')

        self.nominal_q = None
        self.nominal_q_stamp = 0.0
        self.nominal_dq = None
        self.nominal_dq_stamp = 0.0
        self.current_q = None
        self.dq_filtered = None

        rospy.Subscriber(self.nominal_cmd_topic, Float64MultiArray, self._cmd_cb, queue_size=1)
        rospy.Subscriber(self.nominal_vel_topic, Float64MultiArray, self._vel_cb, queue_size=1)
        rospy.Subscriber('/joint_states', JointState, self._js_cb, queue_size=1)
        self.pub = rospy.Publisher(self.out_topic, Float64MultiArray, queue_size=1)

        rospy.loginfo("velocity_tracker_pp ready: kp=%.2f vfb=%.2f ff=%.2f tau=%.3f -> %s",
                      self.kp, self.max_feedback_velocity, self.ff_gain,
                      self.filter_tau, self.out_topic)
        rospy.Timer(rospy.Duration(1.0 / max(self.rate_hz, 1.0)), self._tick)

    def _cmd_cb(self, m):
        if len(m.data) >= 7:
            self.nominal_q = np.asarray(m.data[:7], dtype=np.float64)
            self.nominal_q_stamp = rospy.get_time()

    def _vel_cb(self, m):
        if len(m.data) >= 7:
            self.nominal_dq = np.asarray(m.data[:7], dtype=np.float64)
            self.nominal_dq_stamp = rospy.get_time()

    def _js_cb(self, m):
        idx = {n: i for i, n in enumerate(m.name)}
        if all(j in idx for j in ARM_JOINTS):
            self.current_q = np.array([m.position[idx[j]] for j in ARM_JOINTS], dtype=np.float64)

    def _publish(self, dq):
        self.pub.publish(Float64MultiArray(data=[float(v) for v in dq]))

    def _tick(self, _evt):
        now = rospy.get_time()
        if self.current_q is None or self.nominal_q is None:
            return
        # Hold (zero) if the nominal command is stale (planner stopped / lost target).
        if self.command_timeout > 0.0 and (now - self.nominal_q_stamp) > self.command_timeout:
            self._publish(np.zeros(7))
            self.dq_filtered = None
            return

        dt = 1.0 / self.rate_hz
        # Feedforward from the executor's retimed velocity (fall back to zero).
        if self.nominal_dq is not None and (now - self.nominal_dq_stamp) <= self.command_timeout:
            dq_ff = self.ff_gain * self.nominal_dq
        else:
            dq_ff = np.zeros(7)
        # Position-error feedback (this is what corrects load-induced lag).
        dq_fb = np.clip(self.kp * (self.nominal_q - self.current_q),
                        -self.max_feedback_velocity, self.max_feedback_velocity)
        dq_raw = dq_ff + dq_fb
        # Low-pass filter to remove piecewise-linear jerk.
        if self.dq_filtered is None:
            self.dq_filtered = dq_raw.copy()
        gamma = dt / (dt + self.filter_tau)
        self.dq_filtered = (1.0 - gamma) * self.dq_filtered + gamma * dq_raw
        dq = np.clip(self.dq_filtered, -self.max_joint_velocity, self.max_joint_velocity)
        self._publish(dq)


if __name__ == '__main__':
    VelocityTrackerPP()
    rospy.spin()
