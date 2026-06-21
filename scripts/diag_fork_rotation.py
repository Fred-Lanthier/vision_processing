#!/usr/bin/env python3
"""Test 1 — does the planner actually emit fork rotation along the horizon?

Subscribes to /planner/nominal_fork_trajectory (PoseArray, the per-waypoint fork
SE(3) the planner publishes BEFORE IK) and, for every received plan, reports the
rotation of each waypoint relative to waypoint 0:

  - total geodesic angle wp0 -> wpN  (how much rotation the model commands)
  - the rotation axis of wp0 -> wpN expressed in the wp0 fork frame
    (should be ~ (0, +/-1, 0) if the fork rotates about its own Y axis)
  - the angle about each fork-0 axis (X / Y / Z) so you can see it is Y-dominant
  - a compact per-waypoint profile of the Y-angle, to see if rotation is
    front-loaded or back-loaded across the 16-step horizon

Run with the venv python (geometry_msgs + scipy):
    source venv_sam3/bin/activate
    rosrun vision_processing diag_fork_rotation.py
or directly:
    python3 scripts/diag_fork_rotation.py
"""
import numpy as np
import rospy
from geometry_msgs.msg import PoseArray
from scipy.spatial.transform import Rotation as R


def quat_to_R(q):
    # geometry_msgs quaternion is (x, y, z, w); scipy wants (x, y, z, w)
    return R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()


class ForkRotationDiag:
    def __init__(self):
        rospy.init_node('diag_fork_rotation')
        self.profile_points = int(rospy.get_param('~profile_points', 8))
        rospy.Subscriber('/planner/nominal_fork_trajectory', PoseArray,
                         self._cb, queue_size=1)
        rospy.loginfo('diag_fork_rotation: waiting for '
                      '/planner/nominal_fork_trajectory ...')

    def _cb(self, msg):
        n = len(msg.poses)
        if n < 2:
            rospy.logwarn('plan has %d poses, need >=2', n)
            return

        R0 = quat_to_R(msg.poses[0].orientation)
        Rlast = quat_to_R(msg.poses[-1].orientation)

        # Relative rotation wp0 -> wpN, expressed in the wp0 (fork-start) frame.
        R_rel = R0.T @ Rlast
        rot = R.from_matrix(R_rel)
        rotvec = rot.as_rotvec()                      # axis * angle, in fork-0 frame
        total_deg = float(np.degrees(np.linalg.norm(rotvec)))
        axis = rotvec / (np.linalg.norm(rotvec) + 1e-12)

        # Signed angle accumulated about each fork-0 axis along the full horizon
        # (intrinsic xyz Euler in the start frame is a good readout when the
        # motion is dominated by one axis).
        ex, ey, ez = rot.as_euler('xyz', degrees=True)

        # Per-waypoint Y-angle profile: rotation of wp_k relative to wp0,
        # projected onto the fork-0 Y axis, to expose front/back-loading.
        ys = []
        idxs = np.linspace(0, n - 1, min(self.profile_points, n)).astype(int)
        for k in idxs:
            Rk = quat_to_R(msg.poses[k].orientation)
            rv = R.from_matrix(R0.T @ Rk).as_rotvec()
            ys.append(float(np.degrees(rv[1])))       # Y component (deg)

        profile = '  '.join(f'{i:>2}:{y:+6.1f}' for i, y in zip(idxs, ys))

        rospy.loginfo(
            '\n[FORK ROT] n=%d  total(wp0->wpN)=%.1f deg  axis(fork0)=[%+.2f %+.2f %+.2f]'
            '\n           about X=%+.1f  Y=%+.1f  Z=%+.1f  (deg, fork0 frame)'
            '\n           Y-angle profile (wp:deg): %s',
            n, total_deg, axis[0], axis[1], axis[2],
            ex, ey, ez, profile)


if __name__ == '__main__':
    ForkRotationDiag()
    rospy.spin()
