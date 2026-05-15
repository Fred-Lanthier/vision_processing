#!/usr/bin/env python3
"""
Bridge CBF safe commands into rolling JointTrajectory chunks.

Sits at the end of the FM-CASF-CBF chain:
  /planner/safe_joint_command  (Float64MultiArray, ~150 Hz from CBF)
      -> rolling JointTrajectory horizon
      -> /position_joint_trajectory_controller/command

franka_gazebo's position_joint_trajectory_controller holds gravity correctly
and interpolates smoothly between the horizon points.  The controller is
updated at rate_hz (default 20 Hz); each chunk starts slightly in the future
and spans horizon_points * point_dt seconds from the current robot state to the
latest CBF-safe target.

Before the first safe command arrives, the bridge publishes a hold trajectory at
the measured joint state. This closes the startup gap between Gazebo spawn and
the planner/CBF pipeline becoming ready.
"""
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINT_NAMES = [f'panda_joint{i}' for i in range(1, 8)]


class TrajectoryControllerBridge:
    def __init__(self):
        rospy.init_node('trajectory_controller_bridge')

        self.rate_hz      = float(rospy.get_param('~rate_hz',       20.0))
        self.out_topic    = rospy.get_param('~out_topic',
                                            '/position_joint_trajectory_controller/command')
        self.horizon_points = max(int(rospy.get_param('~horizon_points', 3)), 2)
        self.point_dt     = float(rospy.get_param('~point_dt',      0.05))
        self.lead_time    = float(rospy.get_param('~lead_time',     0.06))
        self.command_timeout = float(rospy.get_param('~command_timeout', 0.5))
        self.hold_current_until_command = bool(rospy.get_param(
            '~hold_current_until_command', True))
        self.hold_when_stale = bool(rospy.get_param('~hold_when_stale', True))
        self.hold_duration = float(rospy.get_param('~hold_duration', 0.25))

        self.q_current  = None   # latest robot state from /joint_states
        self.q_safe     = None   # latest CBF-safe target from /planner/safe_joint_command
        self.q_hold     = None   # latched startup position
        self.t_safe     = None
        self._last_log  = 0.0
        self._stale_logged = False
        self._holding_logged = False
        self._logged_first_safe = False

        rospy.Subscriber('/joint_states', JointState,
                         self._joint_cb, queue_size=1)
        self.in_topic = rospy.get_param('~in_topic', '/planner/safe_joint_command')
        rospy.Subscriber(self.in_topic, Float64MultiArray,
                         self._safe_cb, queue_size=1)
        self.pub = rospy.Publisher(self.out_topic, JointTrajectory, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._loop)

        rospy.loginfo(
            'trajectory_controller_bridge ready: %s -> %s at %.1f Hz '
            'horizon=%d point_dt=%.3fs lead_time=%.3fs timeout=%.2fs '
            'startup_hold=%s stale_hold=%s',
            self.in_topic, self.out_topic, self.rate_hz, self.horizon_points,
            self.point_dt, self.lead_time, self.command_timeout,
            self.hold_current_until_command, self.hold_when_stale)

    def _joint_cb(self, msg):
        pos = {n: p for n, p in zip(msg.name, msg.position)}
        q = [pos.get(j) for j in JOINT_NAMES]
        if None not in q:
            self.q_current = np.array(q, dtype=np.float64)
            if self.q_hold is None:
                self.q_hold = self.q_current.copy()

    def _safe_cb(self, msg):
        if len(msg.data) >= 7:
            self.q_safe = np.array(msg.data[:7], dtype=np.float64)
            self.t_safe = rospy.Time.now()
            self._stale_logged = False
            if not self._logged_first_safe:
                if self.q_current is None:
                    err = float('nan')
                else:
                    err = float(np.linalg.norm(self.q_safe - self.q_current))
                rospy.loginfo(
                    'trajectory_controller_bridge received first safe command | '
                    'err_to_current=%.4f rad',
                    err)
                self._logged_first_safe = True

    def _loop(self, event):
        if self.q_current is None or self.q_hold is None:
            return

        if self.q_safe is None:
            if self.hold_current_until_command:
                self._publish_hold('waiting for first safe command')
            return

        if self.t_safe is not None:
            age = (rospy.Time.now() - self.t_safe).to_sec()
            if age > self.command_timeout:
                if not self._stale_logged:
                    rospy.logwarn(
                        'trajectory_controller_bridge stopping: last safe command is %.3fs old',
                        age)
                    self._stale_logged = True
                if self.hold_when_stale:
                    self._publish_hold('safe command stale')
                return

        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now() + rospy.Duration(self.lead_time)
        msg.joint_names  = JOINT_NAMES

        # Point 0 is scheduled in the near future, avoiding "point occurs before
        # current time" drops while keeping the controller anchored at the
        # measured state.
        p0 = JointTrajectoryPoint()
        p0.positions       = self.q_current.tolist()
        p0.time_from_start = rospy.Duration(0.0)
        msg.points.append(p0)

        # Points 1..horizon: linearly interpolate from current state to CBF target.
        # Spreading the motion over horizon_points * point_dt seconds gives the
        # controller a smooth path rather than a single hard step.
        q0 = self.q_current
        q1 = self.q_safe
        for i in range(1, self.horizon_points + 1):
            alpha = i / self.horizon_points
            pt = JointTrajectoryPoint()
            pt.positions       = (q0 + alpha * (q1 - q0)).tolist()
            pt.time_from_start = rospy.Duration(i * self.point_dt)
            msg.points.append(pt)

        self.pub.publish(msg)
        self._holding_logged = False

        now = rospy.get_time()
        if now - self._last_log > 1.0:
            self._last_log = now
            err = float(np.linalg.norm(q1 - q0))
            rospy.logdebug('bridge | q_err_norm=%.4f rad horizon=%.2fs lead=%.3fs',
                          err, self.horizon_points * self.point_dt, self.lead_time)

    def _publish_hold(self, reason):
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now() + rospy.Duration(self.lead_time)
        msg.joint_names = JOINT_NAMES

        p0 = JointTrajectoryPoint()
        p0.positions = self.q_hold.tolist()
        p0.time_from_start = rospy.Duration(0.0)
        msg.points.append(p0)

        p1 = JointTrajectoryPoint()
        p1.positions = self.q_hold.tolist()
        p1.time_from_start = rospy.Duration(self.hold_duration)
        msg.points.append(p1)

        self.pub.publish(msg)

        if not self._holding_logged:
            rospy.logdebug('trajectory_controller_bridge holding latched joints: %s', reason)
            self._holding_logged = True

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    TrajectoryControllerBridge().run()
