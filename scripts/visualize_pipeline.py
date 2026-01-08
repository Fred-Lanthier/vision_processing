#!/usr/bin/env python3
import rospy
import numpy as np
import open3d as o3d
import message_filters
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import tf.transformations as tft
import rospkg
import os
import sys

# --- IMPORT DU LOADER ---
# Ajoutez le chemin vers votre dossier contenant Compute_3D_point_cloud_from_mesh.py
# Adaptez 'vision_processing' et le chemin si nécessaire
rospack = rospkg.RosPack()
sys.path.append(os.path.join(rospack.get_path('vision_processing'), 'scripts'))

try:
    from Compute_3D_point_cloud_from_mesh import RobotMeshLoaderOptimized
except ImportError:
    print("❌ ERREUR: Impossible d'importer RobotMeshLoaderOptimized.")
    print("Vérifiez que Compute_3D_point_cloud_from_mesh.py est bien accessible.")
    sys.exit(1)

class InferenceVisualizer:
    def __init__(self):
        rospy.init_node('inference_visualizer', anonymous=True)
        
        # 1. Chargement du Robot (URDF)
        print("⏳ Chargement du modèle URDF...")
        # On cherche le fichier combiné
        pkg_path = rospack.get_path('vision_processing') # Ou fl_read_pose selon votre setup
        # Tente de trouver le fichier xacro/urdf
        urdf_path = os.path.join(rospack.get_path('fl_read_pose'), 'scripts', 'panda_arm_hand_combined.urdf.xacro')
        if not os.path.exists(urdf_path):
             # Fallback sur une valeur par défaut ou paramètre ROS
             urdf_path = rospy.get_param('/robot_description_file', '')
        
        if not os.path.exists(urdf_path):
            rospy.logerr(f"URDF introuvable: {urdf_path}")
            # NOTE: Assurez-vous que le chemin est bon ou hardcodez-le pour le test
        
        self.mesh_loader = RobotMeshLoaderOptimized(urdf_path)
        print("✅ Modèle Robot chargé.")

        # 2. Variables d'état
        self.food_pcd_world = None # Ce sera le nuage fixe
        self.bridge = CvBridge()
        self.is_initialized = False

        # Paramètres Caméra (À ajuster selon votre calibration réelle ou CameraInfo)
        self.fx = 607.18
        self.fy = 606.91
        self.cx = 320.85
        self.cy = 243.40
        
        # Transform EE -> Camera (Offset physique)
        # D'après votre xacro : <origin xyz="-0.052 0.035 -0.045" rpy="${-pi/2} 0 ${-pi/2}"/>
        # C'est la transfo TCP -> Camera Link
        # Attention : C'est souvent TCP -> Optical Frame qu'on veut.
        # Vérifions votre xacro :
        # wrist_link -> optical_frame : rpy="${-pi/2} 0 ${-pi/2}" (Standard ROS to CV)
        # TCP -> wrist_link : xyz="-0.052 0.035 -0.045" rpy="${-pi/2} 0 ${-pi/2}"
        
        # Calcul de T_tcp_cam (Matrice 4x4)
        # NOTE : Simplification pour le test, ajustez si le nuage food est décalé
        T_tcp_wrist = tft.compose_matrix(
            translate=[-0.052, 0.035, -0.045],
            angles=tft.euler_from_quaternion(tft.quaternion_from_euler(-np.pi/2, 0, -np.pi/2))
        )
        # Optical frame offset souvent nécessaire
        T_wrist_optical = tft.compose_matrix(translate=[0,0,0], angles=tft.euler_from_quaternion(tft.quaternion_from_euler(-np.pi/2, 0, -np.pi/2)))
        
        self.T_tcp_optical = np.dot(T_tcp_wrist, T_wrist_optical)

        # 3. Subscribers Synchronisés
        sub_depth = message_filters.Subscriber("/synced/camera_wrist/depth", Image)
        sub_pose = message_filters.Subscriber("/synced/ee_pose", PoseStamped)
        sub_joints = message_filters.Subscriber("/synced/joint_states", JointState)
        
        # On n'a pas besoin de RGB pour le dummy test, mais nécessaire plus tard pour SAM3
        # sub_rgb = ...

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_depth, sub_pose, sub_joints], queue_size=5, slop=0.1
        )
        self.ts.registerCallback(self.callback)

        # 4. Visualisation Open3D
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Inference Input Debug", width=1280, height=720)
        
        # Création des géométries vides
        self.pcd_robot = o3d.geometry.PointCloud()
        self.pcd_food = o3d.geometry.PointCloud()
        
        self.vis.add_geometry(self.pcd_robot)
        self.vis.add_geometry(self.pcd_food)
        
        # Repère visuel (origine)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        self.vis.add_geometry(axis)

        print("🚀 Visualizer prêt. En attente de données...")

    def pose_to_matrix(self, pose_msg):
        p = pose_msg.position
        q = pose_msg.orientation
        trans = [p.x, p.y, p.z]
        rot = [q.x, q.y, q.z, q.w]
        return tft.compose_matrix(translate=trans, angles=tft.euler_from_quaternion(rot))

    def process_depth_to_cloud(self, depth_img, mask=None):
        # Conversion profondeur -> Nuage (Camera Frame)
        # Vectorized implementation
        h, w = depth_img.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        z = depth_img / 1000.0 # mm to meters
        
        # Filtre basique : ignorer ce qui est trop loin (>1m) ou vide
        valid = (z > 0.1) & (z < 1.0)
        
        if mask is not None:
            valid = valid & mask
            
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        # Stack (N, 3)
        points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
        return points

    def callback(self, depth_msg, ee_pose_msg, joint_msg):
        # 1. Update Robot State
        joint_map = {}
        for i, name in enumerate(joint_msg.name):
            if "panda" in name:
                joint_map[name] = joint_msg.position[i]
        
        robot_points = self.mesh_loader.create_point_cloud(joint_map)
        self.pcd_robot.points = o3d.utility.Vector3dVector(robot_points)
        self.pcd_robot.paint_uniform_color([0.5, 0.5, 0.5])

        # 2. Gestion Food (Debug amélioré)
        if self.food_pcd_world is None:
            try:
                # Convertir en float32 (mètres ou millimètres)
                cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
                
                # Conversion mm -> mètres si nécessaire (Gazebo sort souvent du float32 en mètres directement, ou uint16 en mm)
                if cv_depth.dtype == np.uint16:
                    z_img = cv_depth / 1000.0
                else:
                    z_img = cv_depth # Supposons mètres pour float32
                
                # Stats de débogage
                valid_mask = ~np.isnan(z_img) & (z_img > 0.1) & (z_img < 3.0)
                valid_count = np.sum(valid_mask)
                
                if valid_count > 0:
                    min_dist = np.min(z_img[valid_mask])
                    max_dist = np.max(z_img[valid_mask])
                    print(f"👀 Vision active ! {valid_count} points valides. Distances: [{min_dist:.2f}m - {max_dist:.2f}m]")
                    
                    # On prend TOUT ce qui est valide (pas de crop central pour l'instant)
                    points_cam = self.process_depth_to_cloud(cv_depth, mask=None) # Mask None = Tout prendre
                    
                    # Transformation vers World
                    T_world_tcp = self.pose_to_matrix(ee_pose_msg.pose)
                    T_world_cam = np.dot(T_world_tcp, self.T_tcp_optical)
                    
                    ones = np.ones((points_cam.shape[0], 1))
                    points_hom = np.hstack([points_cam, ones])
                    points_world = np.dot(T_world_cam, points_hom.T).T[:, :3]
                    
                    self.food_pcd_world = points_world
                    self.pcd_food.points = o3d.utility.Vector3dVector(self.food_pcd_world)
                    self.pcd_food.paint_uniform_color([0, 1, 0]) # Vert pour voir que ça a marché
                    print("✅ SNAPSHOT RÉUSSI ! Nuage fixé.")
                
                else:
                    # Affiche ce qui ne va pas pour comprendre
                    print("⚠️ Image DEPTH reçue mais vide (NaN ou hors range).")
                    print(f"   Valeurs brutes -> Min: {np.nanmin(z_img):.2f}, Max: {np.nanmax(z_img):.2f}")
                    print("   👉 BOUGEZ LE ROBOT pour viser le cube/sol.")

            except Exception as e:
                print(f"Erreur processing: {e}")

        # 3. Render
        self.vis.update_geometry(self.pcd_robot)
        if self.food_pcd_world is not None:
            self.vis.update_geometry(self.pcd_food)
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self):
        rospy.spin()
        self.vis.destroy_window()

if __name__ == '__main__':
    viz = InferenceVisualizer()
    viz.run()