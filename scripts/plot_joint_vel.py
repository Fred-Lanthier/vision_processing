#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
import sys

class JointVelPlotter:
    def __init__(self):
        rospy.init_node('plot_joint_vel', anonymous=True)
        self.sub = rospy.Subscriber('/franka_control/safe_joint_velocities', Float32MultiArray, self.callback)
        self.max_vel = 0.8  # Max velocity for scaling the bar chart
        self.bar_width = 30 # Number of characters for each half of the bar chart (positive and negative)
        self.first = True
        print("\033[?25l") # Hide cursor

    def callback(self, msg):
        if len(msg.data) < 7:
            return
            
        lines = []
        lines.append("="*75)
        lines.append(" Live Joint Velocities (rad/s) | /franka_control/safe_joint_velocities")
        lines.append("="*75)
        
        for idx, vel in enumerate(msg.data[:7]):
            # Scale to self.bar_width
            num_chars = int(abs(vel) / self.max_vel * self.bar_width)
            num_chars = min(num_chars, self.bar_width)
            
            if vel >= 0:
                bar = " " * self.bar_width + "|" + "=" * num_chars + " " * (self.bar_width - num_chars)
            else:
                bar = " " * (self.bar_width - num_chars) + "=" * num_chars + "|" + " " * self.bar_width
                
            lines.append(f"Joint {idx+1}: [{bar}] {vel:6.3f} rad/s")
            
        lines.append("="*75)
        lines.append("Press Ctrl+C to exit.")
        
        # Rewrite terminal block in-place
        if not self.first:
            # Move cursor up by len(lines) lines
            sys.stdout.write(f"\033[{len(lines)}A")
        else:
            self.first = False
            
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    def cleanup(self):
        print("\033[?25h") # Show cursor again

if __name__ == '__main__':
    plotter = JointVelPlotter()
    try:
        rospy.spin()
    finally:
        plotter.cleanup()
