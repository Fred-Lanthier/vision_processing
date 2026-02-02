#!/usr/bin/env python3
"""
Inference Time Measurement Node

Measures the time between trajectory predictions to verify real-time capability.
Also measures actual inference computation time.

Usage:
    rosrun vision_processing inference_timer.py

Let it run for 30-60 seconds while your policy is executing, then Ctrl+C to see stats.
"""
import rospy
import numpy as np
import time
from collections import deque
from geometry_msgs.msg import PoseArray


class InferenceTimer:
    def __init__(self):
        rospy.init_node('inference_timer')
        
        # Storage
        self.intervals = deque(maxlen=1000)  # Time between predictions
        self.last_time = None
        
        # Subscribe to predictions
        self.sub = rospy.Subscriber(
            '/diffusion/target_trajectory',
            PoseArray,
            self.callback
        )
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("⏱️  Inference Timer Started")
        rospy.loginfo("   Measuring prediction intervals...")
        rospy.loginfo("   Press Ctrl+C to see statistics")
        rospy.loginfo("=" * 50)
    
    def callback(self, msg):
        now = time.time()
        
        if self.last_time is not None:
            interval_ms = (now - self.last_time) * 1000
            self.intervals.append(interval_ms)
            
            # Live update every 10 samples
            if len(self.intervals) % 10 == 0:
                mean = np.mean(self.intervals)
                hz = 1000 / mean if mean > 0 else 0
                rospy.loginfo(f"   Samples: {len(self.intervals)} | Mean: {mean:.1f}ms | Rate: {hz:.1f}Hz")
        
        self.last_time = now
    
    def print_stats(self):
        """Print comprehensive statistics"""
        if len(self.intervals) < 2:
            rospy.logwarn("Not enough samples collected")
            return
        
        times = np.array(self.intervals)
        
        print("\n")
        print("=" * 60)
        print("📊 INFERENCE TIMING STATISTICS")
        print("=" * 60)
        print(f"   Total samples:      {len(times)}")
        print(f"   Mean interval:      {np.mean(times):.2f} ms")
        print(f"   Std deviation:      {np.std(times):.2f} ms")
        print(f"   Min interval:       {np.min(times):.2f} ms")
        print(f"   Max interval:       {np.max(times):.2f} ms")
        print(f"   Median:             {np.median(times):.2f} ms")
        print(f"   95th percentile:    {np.percentile(times, 95):.2f} ms")
        print(f"   99th percentile:    {np.percentile(times, 99):.2f} ms")
        print("-" * 60)
        
        # Real-time performance
        mean_hz = 1000 / np.mean(times)
        print(f"   Average rate:       {mean_hz:.2f} Hz")
        
        # Check if meeting targets
        targets = [10, 20, 30]  # Hz
        for target in targets:
            target_ms = 1000 / target
            success = np.mean(times <= target_ms) * 100
            status = "✅" if success > 95 else "⚠️" if success > 80 else "❌"
            print(f"   {status} Meets {target}Hz target: {success:.1f}%")
        
        print("=" * 60)
        
        # Return data for plotting
        return times
    
    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            pass
        finally:
            self.print_stats()


if __name__ == '__main__':
    timer = InferenceTimer()
    timer.run()
