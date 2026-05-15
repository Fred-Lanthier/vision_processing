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
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Float64MultiArray

JOINT_NAMES = [f'panda_joint{i}' for i in range(1, 8)]


class TrajectoryExecutor:
    def __init__(self):
        rospy.init_node('trajectory_executor')

        self.rate_hz      = rospy.get_param('~rate_hz',      50.0)
        self.hold_current_until_trajectory = rospy.get_param(
            '~hold_current_until_trajectory', False)
        # Topic that receives the interpolated nominal command.
        # Set to /planner/nominal_joint_command to route through the CBF,
        # or to /joint_group_position_controller/command to bypass it.
        self.out_topic    = rospy.get_param('~out_topic',
                                            '/joint_group_position_controller/command')

        self.traj_pts  = None   # list of (t_sec, q_7d)
        self.t_start   = None
        self.q_current = None
        self.q_hold    = None   # latched startup position
        self._min_t    = 0.0   # monotone guard: never search before this time
        self._holding_logged = False
        self._logged_first_traj = False
        self._logged_first_publish = False

        rospy.Subscriber('/planner/nominal_trajectory', JointTrajectory,
                         self._traj_cb, queue_size=1)
        rospy.Subscriber('/joint_states', JointState,
                         self._joint_cb, queue_size=1)

        self.pub = rospy.Publisher(self.out_topic, Float64MultiArray, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._control_loop)

        rospy.loginfo('trajectory_executor ready → %s at %.0f Hz',
                      self.out_topic, self.rate_hz)

    # ── callbacks ──────────────────────────────────────────────────────────

    def _joint_cb(self, msg):
        pos = {n: p for n, p in zip(msg.name, msg.position)}
        q = [pos.get(j) for j in JOINT_NAMES]
        if None not in q:
            self.q_current = np.array(q, dtype=np.float64)
            if self.q_hold is None:
                self.q_hold = self.q_current.copy()

    def _traj_cb(self, msg):
        if not msg.points:
            return
        pts = [(p.time_from_start.to_sec(),
                np.array(p.positions[:7], dtype=np.float64))
               for p in msg.points]
        pts.sort(key=lambda x: x[0])
        self.traj_pts = pts
        self._min_t   = 0.0   # new trajectory resets the monotone guard

        # Compensate for planning latency: find the waypoint closest to the
        # current robot state and start interpolation from there. If the planner
        # took 200ms to compute, the robot has already advanced ~2 waypoints
        # past wps[0] by the time this trajectory arrives.  Resetting t_start=now
        # would command wps[0] (stale) and pull the robot backward.
        if self.q_current is not None:
            qs = np.array([q for _, q in pts])
            dists = np.linalg.norm(qs - self.q_current, axis=1)
            closest_idx = int(np.argmin(dists))
            closest_t = pts[closest_idx][0]
            self.t_start = rospy.Time.now() - rospy.Duration(closest_t)
            closest_err = float(dists[closest_idx])
        else:
            self.t_start = rospy.Time.now()
            closest_idx = -1
            closest_t = 0.0
            closest_err = float('nan')

        if not self._logged_first_traj:
            rospy.loginfo(
                'trajectory_executor received first trajectory: points=%d '
                'duration=%.3fs closest_idx=%d closest_t=%.3fs closest_err=%.4f rad',
                len(pts), pts[-1][0], closest_idx, closest_t, closest_err)
            self._logged_first_traj = True

    # ── interpolation ──────────────────────────────────────────────────────

    def _interpolate(self, t):
        pts = self.traj_pts
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

    # ── control loop ───────────────────────────────────────────────────────

    def _control_loop(self, event):
        if self.q_current is None or self.q_hold is None:
            return

        if self.traj_pts is None:
            if self.hold_current_until_trajectory:
                self.pub.publish(Float64MultiArray(data=self.q_hold.tolist()))
                if not self._holding_logged:
                    rospy.logdebug(
                        'trajectory_executor holding latched joints until first trajectory')
                    self._holding_logged = True
            return

        t = (rospy.Time.now() - self.t_start).to_sec()

        # Monotone guard: never go backward in the trajectory.
        # Protects against the closest-waypoint search landing on a later section
        # of a new trajectory that briefly passes near the current joint config.
        t = max(t, self._min_t)
        self._min_t = t

        q_des = self._interpolate(t)
        if not self._logged_first_publish:
            err = float(np.linalg.norm(q_des - self.q_current))
            rospy.loginfo(
                'trajectory_executor publishing first nominal command | '
                't=%.3fs err_to_current=%.4f rad topic=%s',
                t, err, self.out_topic)
            self._logged_first_publish = True
        self.pub.publish(Float64MultiArray(data=q_des.tolist()))

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    TrajectoryExecutor().run()
