#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

class JointErrorMonitor:
    def __init__(self):
        rospy.init_node('joint_error_monitor')

        self.joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
            'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]

        self.current_q = None
        self.target_q = None # Safe command (final)
        self.nominal_q = None # Planner output (intent)

        # Subscribers
        rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        rospy.Subscriber('/position_joint_trajectory_controller/command', JointTrajectory, self.target_callback)
        rospy.Subscriber('/planner/nominal_joint_command', Float64MultiArray, self.nominal_callback)

        # Timer for reporting at 1Hz
        rospy.Timer(rospy.Duration(1.0), self.report_callback)

        rospy.loginfo("🔍 Joint Error Monitor initialized at 1Hz")
        rospy.loginfo(f"Monitoring Intent vs Safe vs Actual")

    def joint_callback(self, msg):
        try:
            pos_dict = {n: p for n, p in zip(msg.name, msg.position)}
            q = [pos_dict[name] for name in self.joint_names if name in pos_dict]
            if len(q) == 7:
                self.current_q = np.array(q)
        except Exception:
            pass

    def nominal_callback(self, msg):
        if len(msg.data) >= 7:
            self.nominal_q = np.array(msg.data[:7])

    def target_callback(self, msg):
        if len(msg.points) > 0:
            try:
                target_dict = {n: p for n, p in zip(msg.joint_names, msg.points[0].positions)}
                q = [target_dict[name] for name in self.joint_names if name in target_dict]
                if len(q) == 7:
                    self.target_q = np.array(q)
            except Exception:
                pass

    def report_callback(self, event):
        if self.current_q is None:
            rospy.logwarn_throttle(5, "Waiting for /joint_states...")
            return
        if self.target_q is None:
            rospy.logwarn_throttle(5, "Waiting for controller command...")
            return

        # Error is final tracking performance
        tracking_error = self.target_q - self.current_q

        # Shield correction is Nominal vs Safe
        shield_delta = np.zeros(7)
        if self.nominal_q is not None:
            shield_delta = self.target_q - self.nominal_q

        total_err = np.linalg.norm(tracking_error)

        print("\n" + "="*110)
        print(f" Joint State Monitor (rad) | Tracking L2: {total_err:.6f}")
        print("-" * 110)
        print(f" {'Joint Name':<15} | {'Nominal Q':>12} | {'Safe Q':>12} | {'Actual Q':>12} | {'Correction':>12}")
        print("-" * 110)
        for i, name in enumerate(self.joint_names):
            nom = f"{self.nominal_q[i]:>12.6f}" if self.nominal_q is not None else "     N/A    "
            print(f" {name:<15} | {nom} | {self.target_q[i]:>12.6f} | {self.current_q[i]:>12.6f} | {shield_delta[i]:>12.6f}")
        print("="*110)

if __name__ == '__main__':
    try:
        monitor = JointErrorMonitor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
