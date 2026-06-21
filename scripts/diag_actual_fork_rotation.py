#!/usr/bin/env python3
"""Test 3 — where in the pipeline is the fork rotation lost?

diag_fork_rotation.py reads /planner/nominal_fork_trajectory, which is the model
SE(3) published BEFORE the IK solve. It proves the model COMMANDS rotation; it
does NOT prove the robot ACHIEVES it. This node closes that gap by tracking the
ACTUAL fork orientation from TF (world -> fork_tip) over the whole approach and
comparing it to the latest commanded target.

It reports, continuously:
  - actual cumulative fork rotation since the node started, decomposed in the
    fork-start frame (X / Y / Z) and as a total angle. If fork-X climbs toward
    ~90 deg over the approach, rotation IS executing. If it stays near 0 while
    the commanded chunks each show ~12 deg, the IK/executor/CBF chain is
    dropping it.
  - instantaneous actual rotation rate (deg/s).
  - the latest commanded net rotation (wp0 -> wpN of the published fork
    trajectory), so you can see commanded vs achieved side by side.

Run in the live system (needs TF + the planner running):
    source venv_sam3/bin/activate
    python3 scripts/diag_actual_fork_rotation.py
"""
import numpy as np
import rospy
import tf
import tf.transformations as tft
from geometry_msgs.msg import PoseArray


def quat_to_R(q):
    return tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]


def rotvec_deg(R):
    """(3,3) -> rotvec in degrees (axis*angle)."""
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(tr)
    if ang < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]]) / (2.0 * np.sin(ang))
    return np.degrees(w * ang)


class ActualForkRotation:
    def __init__(self):
        rospy.init_node('diag_actual_fork_rotation')
        self.world = rospy.get_param('~world_frame', 'world')
        self.fork = rospy.get_param('~fork_frame', 'fork_tip')
        self.rate_hz = float(rospy.get_param('~rate_hz', 10.0))
        self.listener = tf.TransformListener()

        self.R_start = None          # fork orientation at first reading
        self.R_prev = None
        self.t_prev = None
        self.cmd_net = None          # latest commanded wp0->wpN rotvec (deg, fork0)

        rospy.Subscriber('/planner/nominal_fork_trajectory', PoseArray,
                         self._cmd_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._tick)
        rospy.loginfo('diag_actual_fork_rotation: tracking %s -> %s',
                      self.world, self.fork)

    def _cmd_cb(self, msg):
        if len(msg.poses) < 2:
            return
        R0 = quat_to_R(msg.poses[0].orientation)
        Rn = quat_to_R(msg.poses[-1].orientation)
        self.cmd_net = rotvec_deg(R0.T @ Rn)

    def _tick(self, _):
        try:
            (_, rot) = self.listener.lookupTransform(
                self.world, self.fork, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException):
            return
        R_now = tft.quaternion_matrix(rot)[:3, :3]
        now = rospy.get_time()

        if self.R_start is None:
            self.R_start = R_now
            self.R_prev = R_now
            self.t_prev = now
            return

        # cumulative actual rotation since start, expressed in the start frame
        cum = rotvec_deg(self.R_start.T @ R_now)
        cum_tot = float(np.linalg.norm(cum))

        # instantaneous rate
        dt = max(now - self.t_prev, 1e-6)
        step = rotvec_deg(self.R_prev.T @ R_now)
        rate = float(np.linalg.norm(step)) / dt
        self.R_prev = R_now
        self.t_prev = now

        cmd = self.cmd_net if self.cmd_net is not None else np.zeros(3)
        cmd_tot = float(np.linalg.norm(cmd))

        rospy.loginfo_throttle(
            0.5,
            '[ACTUAL FORK] cum since start: tot=%.1f deg  X=%+.1f Y=%+.1f Z=%+.1f'
            '  | rate=%.1f deg/s  || commanded chunk wp0->wpN: tot=%.1f '
            '(X=%+.1f Y=%+.1f Z=%+.1f)',
            cum_tot, cum[0], cum[1], cum[2], rate,
            cmd_tot, cmd[0], cmd[1], cmd[2])


if __name__ == '__main__':
    ActualForkRotation()
    rospy.spin()
