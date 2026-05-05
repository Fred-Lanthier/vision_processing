#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray, Header
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def generate_noisy_sphere_cloud(center, radius, num_points=1000):
    """Génère un nuage de 1000 points répartis uniformément avec un bruit de capteur."""
    # Sphère de Fibonacci pour une belle répartition
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    base_sphere = np.vstack((x, y, z)).T
    
    # Bruit de capteur (jitter) de 3mm
    noise = np.random.normal(0, 0.003, base_sphere.shape)
    return base_sphere + center + noise

def run_environment():
    rospy.init_node('mock_environment_node')
    
    # Publishers de données (Pour le CBF)
    target_pub = rospy.Publisher('/planning/target_pose', Float32MultiArray, queue_size=1)
    obs_pub = rospy.Publisher('/perception/obstacles', PointCloud2, queue_size=1)
    
    # Publishers RViz (Ce que j'avais supprimé par erreur !)
    viz_target_pub = rospy.Publisher('/viz/target_pose', Marker, queue_size=1)
    viz_obs_pub = rospy.Publisher('/viz/raw_obstacles', PointCloud2, queue_size=1)
    
    rate = rospy.Rate(100) 
    
    circle_center = np.array([0.4, 0.0, 0.4])
    circle_radius = 0.15
    omega_circle = 0.5 
    
    # obs_start = np.array([0.4, 1.0, 0.4]) # On le fait partir plus proche pour le voir entrer dans la CropBox
    # obs_velocity = np.array([0.0, -0.05, 0.0]) 
    # obs_radius = 0.08
    
    start_time = rospy.get_time()
    print("🌍 Démarrage Environnement (1000 pts + Bruit + RViz)...")
    
    while not rospy.is_shutdown():
        t = rospy.get_time() - start_time
        
        # 1. Cible
        target_x = circle_center[0] + circle_radius * np.cos(omega_circle * t)
        target_y = circle_center[1] + circle_radius * np.sin(omega_circle * t)
        target_z = circle_center[2] 
        target_pose = np.array([target_x, target_y, target_z], dtype=np.float32)
        
        # 2. Obstacle
        # current_obs_center = obs_start + obs_velocity * t
        # obs_points = generate_noisy_sphere_cloud(current_obs_center, obs_radius, num_points=1000)
        
        # 3. Publication aux contrôleurs
        # target_pub.publish(Float32MultiArray(data=target_pose.tolist()))
        
        # Nuage Brut (Blanc par défaut dans RViz)
        # header = Header()
        # header.stamp = rospy.Time.now()
        # header.frame_id = "world"
        # cloud_msg = pc2.create_cloud_xyz32(header, obs_points.tolist())
        # obs_pub.publish(cloud_msg)
        # viz_obs_pub.publish(cloud_msg)
        
        rate.sleep()

if __name__ == '__main__':
    run_environment()