#!/usr/bin/env python3
"""
Robot Mesh to Point Cloud Converter - Clean Optimized Version
Optimizations: Vectorized transforms + Pre-computed surface areas

Author: Frederick Lanthier (optimized)
"""

import numpy as np
import xml.etree.ElementTree as ET
import rospkg
import subprocess
import os
import tempfile
import time
import yaml
import json

import importlib_resources
import sys
import types
importlib_resources_module = types.SimpleNamespace(files=importlib_resources.files)
sys.modules['importlib.resources'] = importlib_resources_module

import trimesh


class RobotMeshLoaderOptimized:
    def __init__(self, urdf_path):
        """Initialize the optimized mesh loader"""
        self.urdf_path = urdf_path
        
        # Store mesh data as numpy arrays for fast operations
        self.mesh_vertices = {}      # Nx3 vertex arrays
        self.mesh_faces = {}         # Mx3 face arrays
        self.mesh_face_areas = {}    # OPTIMIZATION: Pre-computed areas
        
        # Convert XACRO if needed
        if urdf_path.endswith('.xacro'):
            print("Converting XACRO to URDF...")
            self.urdf_path = self._convert_xacro_to_urdf(urdf_path)
        
        self._parse_urdf()
        self._load_all_meshes()
    
    def _convert_xacro_to_urdf(self, xacro_path):
        """Convert XACRO to URDF"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            urdf_path = f.name
        
        cmd = ['rosrun', 'xacro', 'xacro', xacro_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("\n" + "="*60)
            print("ERROR: XACRO conversion failed!")
            print("="*60)
            print(f"Command: {' '.join(cmd)}")
            print(f"\nSTDOUT:\n{result.stdout}")
            print(f"\nSTDERR:\n{result.stderr}")
            print("="*60)
            raise Exception("XACRO conversion failed")
        
        with open(urdf_path, 'w') as f:
            f.write(result.stdout)
        
        print(f"✓ URDF generated")
        return urdf_path
    
    def _resolve_package_path(self, uri):
        """Resolve package:// URIs to file paths"""
        if not uri.startswith('package://'):
            return uri
        
        uri = uri.replace('package://', '')
        parts = uri.split('/', 1)
        package_name = parts[0]
        relative_path = parts[1] if len(parts) > 1 else ''
        
        rospack = rospkg.RosPack()
        try:
            package_path = rospack.get_path(package_name)
            return os.path.join(package_path, relative_path)
        except Exception as e:
            print(f"Warning: Could not resolve package {package_name}: {e}")
            return None
    
    def _parse_urdf(self):
        """Parse URDF and extract mesh information"""
        print("Parsing URDF...")
        
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        
        self.links_info = {}
        self.joints_info = {}
        
        # Parse links
        for link in root.findall('link'):
            link_name = link.get('name')
            visual = link.find('visual')
            mesh_path = None
            visual_origin = np.eye(4)
            scale = [1, 1, 1]
            
            if visual is not None:
                geometry = visual.find('geometry')
                if geometry is not None:
                    mesh_elem = geometry.find('mesh')
                    if mesh_elem is not None:
                        mesh_uri = mesh_elem.get('filename')
                        mesh_path = self._resolve_package_path(mesh_uri)
                        scale_str = mesh_elem.get('scale', '1 1 1')
                        scale = [float(x) for x in scale_str.split()]
                
                origin = visual.find('origin')
                if origin is not None:
                    xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
                    rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()]
                    visual_origin = self._create_transform(xyz, rpy)
            
            self.links_info[link_name] = {
                'name': link_name,
                'mesh_path': mesh_path,
                'visual_origin': visual_origin,
                'scale': scale
            }
        
        # Parse joints
        for joint in root.findall('joint'):
            joint_name = joint.get('name')
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            
            origin = joint.find('origin')
            xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()] if origin is not None else [0, 0, 0]
            rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()] if origin is not None else [0, 0, 0]
            
            axis_elem = joint.find('axis')
            axis = [float(x) for x in axis_elem.get('xyz', '0 0 1').split()] if axis_elem is not None else [0, 0, 1]
            
            self.joints_info[joint_name] = {
                'parent': parent,
                'child': child,
                'xyz': xyz,
                'rpy': rpy,
                'axis': axis,
                'type': joint.get('type')
            }
        
        print(f"✓ Found {len(self.links_info)} links")
    
    def _load_all_meshes(self):
        """Load all meshes once and extract numpy arrays"""
        print("\n" + "="*60)
        print("Loading meshes (happens only once)...")
        print("="*60)
        self.static_point_clouds = {} # On stocke les points locaux ici
        for link_name, link_info in self.links_info.items():
            mesh_path = link_info['mesh_path']
            
            if mesh_path and os.path.exists(mesh_path):
                print(f"  Loading: {link_name} from {os.path.basename(mesh_path)}")
                
                try:
                    # Load mesh with trimesh
                    mesh = trimesh.load(mesh_path, force='mesh')
                    
                    # # Apply scale
                    scale = link_info['scale']
                    if scale != [1, 1, 1]:
                        mesh.apply_scale(scale)
                    
                    # # Apply visual origin
                    mesh.apply_transform(link_info['visual_origin'])
                    
                    # # Extract numpy arrays for fast operations
                    # self.mesh_vertices[link_name] = np.array(mesh.vertices, dtype=np.float64)
                    # self.mesh_faces[link_name] = np.array(mesh.faces, dtype=np.int32)
                    
                    # # OPTIMIZATION: Pre-compute face areas for sampling
                    # self.mesh_face_areas[link_name] = mesh.area_faces
                    
                    # mesh = trimesh.load(mesh_path, force='mesh')
        
                    # On définit combien de points par lien (ex: au prorata de la surface)
                    # Pour ton test, disons 5000 points par lien pour arriver à ~50k total
                    num_samples = 5000 
                    self.static_point_clouds[link_name] = mesh.sample(num_samples)
                except Exception as e:
                    print(f"    Warning: Could not load {link_name}: {e}")
        
        print(f"\n✓ Successfully loaded {len(self.static_point_clouds)} meshes")
        print("="*60)

    def _create_transform(self, xyz, rpy):
        """Create 4x4 transformation matrix"""
        r, p, y = rpy
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(r), -np.sin(r)],
                       [0, np.sin(r), np.cos(r)]])
        Ry = np.array([[np.cos(p), 0, np.sin(p)],
                       [0, 1, 0],
                       [-np.sin(p), 0, np.cos(p)]])
        Rz = np.array([[np.cos(y), -np.sin(y), 0],
                       [np.sin(y), np.cos(y), 0],
                       [0, 0, 1]])
        
        R = Rz @ Ry @ Rx
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = xyz
        
        return T
    
    def compute_link_transforms(self, joint_angles=None):
        """Compute transformation matrix for each link"""
        if joint_angles is None:
            joint_angles = {}
        
        transforms = {}
        
        # Find base link
        base_names = ['panda_link0', 'base_link', 'world']
        for base_name in base_names:
            if base_name in self.links_info:
                transforms[base_name] = np.eye(4)
                break
        
        def get_transform(link_name):
            if link_name in transforms:
                return transforms[link_name]
            
            # Find parent joint
            for joint_name, joint in self.joints_info.items():
                if joint['child'] == link_name:
                    parent_transform = get_transform(joint['parent'])
                    joint_angle = joint_angles.get(joint_name, 0.0)
                    
                    # Create joint transform
                    joint_T = self._create_transform(joint['xyz'], joint['rpy'])
                    
                    # Apply rotation about joint axis
                    if joint['type'] in ['revolute', 'continuous']:
                        axis = np.array(joint['axis'])
                        axis = axis / np.linalg.norm(axis)
                        
                        # Rodrigues' rotation formula
                        K = np.array([[0, -axis[2], axis[1]],
                                     [axis[2], 0, -axis[0]],
                                     [-axis[1], axis[0], 0]])
                        
                        R_joint = np.eye(3) + np.sin(joint_angle) * K + (1 - np.cos(joint_angle)) * K @ K
                        joint_T[:3, :3] = joint_T[:3, :3] @ R_joint
                    
                    transforms[link_name] = parent_transform @ joint_T
                    return transforms[link_name]
            
            transforms[link_name] = np.eye(4)
            return transforms[link_name]
        
        # Compute all transforms
        for link_name in self.links_info:
            get_transform(link_name)
        
        return transforms
    
    # 
    def create_point_cloud(self, joint_angles=None):
        transforms = self.compute_link_transforms(joint_angles)
        all_transformed_points = []

        for link_name, local_points in self.static_point_clouds.items():
            if (link_name in transforms):
                T = transforms[link_name]
            
                # Transformation vectorisée (NumPy)
                # On ajoute une colonne de 1 pour la multiplication matricielle
                ones = np.ones((len(local_points), 1))
                points_homo = np.hstack([local_points, ones])
                
                # Application de la matrice T (4x4) sur tous les points (Nx4)
                transformed = (T @ points_homo.T).T[:, :3]
                all_transformed_points.append(transformed)
                # if link_name == 'fork_tip':
                    # all_transformed_points.append(transformed)

        return np.vstack(all_transformed_points)
    
    def _sample_surface_weighted(self, vertices, faces, face_areas, num_points):
        """
        Sample points from mesh surface weighted by face area
        
        This is what trimesh.sample.sample_surface does internally,
        but we do it directly using our pre-computed areas
        """
        # Normalize areas to get probabilities
        area_cumsum = np.cumsum(face_areas)
        area_total = area_cumsum[-1]
        
        # Sample face indices based on area
        random_samples = np.random.uniform(0, area_total, num_points)
        face_indices = np.searchsorted(area_cumsum, random_samples)
        
        # Get triangle vertices for sampled faces
        triangles = vertices[faces[face_indices]]
        
        # Generate random barycentric coordinates
        # This ensures uniform distribution within each triangle
        r1 = np.random.random(num_points)
        r2 = np.random.random(num_points)
        
        # Ensure points are inside triangle
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        # Barycentric interpolation: P = A + r1*(B-A) + r2*(C-A)
        points = (triangles[:, 0] + 
                 r1[:, np.newaxis] * (triangles[:, 1] - triangles[:, 0]) +
                 r2[:, np.newaxis] * (triangles[:, 2] - triangles[:, 0]))
        
        return points


def benchmark_point_cloud_generation(loader, configurations, num_points=10000):
    """
    Benchmark point cloud generation for multiple configurations
    
    Args:
        loader: RobotMeshLoaderOptimized instance
        configurations: List of joint angle dictionaries
        num_points: Number of points per cloud
    
    Returns:
        times: List of computation times
        all_point_clouds: List of (points, colors) tuples
    """
    print("\n" + "="*60)
    print(f"Benchmarking {len(configurations)} configurations")
    print(f"Points per cloud: {num_points}")
    print("="*60)
    
    times = []
    all_point_clouds = []
    
    for i, joint_angles in enumerate(configurations):
        print(f"\nConfiguration {i+1}/{len(configurations)}")
        
        start_time = time.time()
        points = loader.create_point_cloud(joint_angles)
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        all_point_clouds.append(points)
        
        print(f"  Time: {elapsed:.4f} seconds")
    
    print("\n" + "="*60)
    print("Benchmark Results")
    print("="*60)
    print(f"Total configurations: {len(configurations)}")
    print(f"Average time: {np.mean(times):.4f} seconds")
    print(f"Min time: {np.min(times):.4f} seconds")
    print(f"Max time: {np.max(times):.4f} seconds")
    print(f"Std dev: {np.std(times):.4f} seconds")
    print("="*60)
    
    return times, all_point_clouds


def main():
    """Main function"""
    print("="*60)
    print("Optimized Robot Mesh Loader")
    print("Optimizations: Vectorized transforms + Pre-computed areas")
    print("="*60)
    
    # Get URDF path
    rospack = rospkg.RosPack()
    franka_path = rospack.get_path('franka_description')
    package_path_fl = rospack.get_path('fl_read_pose')
    package_path_vision_processing = rospack.get_path('vision_processing')
    # First, try to find existing combined file
    urdf_options = [
        os.path.join(package_path_fl, 'scripts', 'panda_arm_hand_combined.urdf.xacro'),
    ]
    
    urdf_path = None
    for path in urdf_options:
        if os.path.exists(path):
            urdf_path = path
            print(f"\n✓ Found URDF: {os.path.basename(path)}")
            break
    
    if urdf_path is None:
        print("\nERROR: Could not find URDF file!")
        print("\nYou need to copy panda_arm_hand_combined.urdf.xacro to:")
        print(f"  {os.path.join(package_path_fl, 'scripts')}/")
        return
    
    # Create loader (loads meshes once)
    loader = RobotMeshLoaderOptimized(urdf_path)
    
    # Load trajectory data
    
    config_file = os.path.join(package_path_fl, 'config', 'trajectory_recreation_config.yaml')
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        idx = config['trajectory_number']
        write_folder = config['trajectory_folder_write']
        read_folder = config['trajectory_folder_read']
    total_traj = 45
    for traj_id in range(1, total_traj+1):
        json_file = os.path.join(package_path_vision_processing, write_folder, 
                                f'Trajectory_{traj_id}', f'trajectory_{traj_id}.json')
        
        images_folder = os.path.join(package_path_vision_processing, write_folder,
                                    f'Trajectory_{traj_id}', f'images_Trajectory_{traj_id}')
        
        with open(json_file, 'r') as file:
            datas = json.load(file)
            states = datas['states']
        
        # Create test configurations
        configurations = []
        sample_indices = np.linspace(0, len(states), len(states)+1).tolist()
        print(len(states))
        # for idx in range(1,2):
        for idx in sample_indices:
            if idx < len(states):
                joint_pos = states[int(idx)]['joint_positions']
                joint_angles = {
                    'panda_joint1': joint_pos[0],
                    'panda_joint2': joint_pos[1],
                    'panda_joint3': joint_pos[2],
                    'panda_joint4': joint_pos[3],
                    'panda_joint5': joint_pos[4],
                    'panda_joint6': joint_pos[5],
                    'panda_joint7': joint_pos[6],
                    'panda_finger_joint1': 0.04,
                    'panda_finger_joint2': 0.04,
                }
                configurations.append(joint_angles)
        
        # Run benchmark
        times, point_clouds = benchmark_point_cloud_generation(
            loader, 
            configurations, 
            num_points=50000
        )
        pcd = np.array(point_clouds)
        print(pcd.shape)
        # Saving point cloud
        print('=' * 60)
        print('Saving point cloud as NPY files')
        print('=' * 60)

        for idx in range(0, len(pcd)):
            num_0 = 4 - len(str(idx+1))
            id = '0'*num_0 + str(idx+1)
            # Define the filename for your .npy file
            # filename = f'Fork_point_cloud_{id}.npy'
            filename = f'Robot_point_cloud_{id}.npy'
            save_file = os.path.join(images_folder, filename)
            # Save the NumPy array to the .npy file
            np.save(save_file, pcd[idx])

    # Visualize one point cloud if desired
    # print("\nWould you like to visualize one of the point clouds? (y/n)")
    # choice = input().strip().lower()
    
    # if choice == 'y':
    #     print(f"Which configuration? (1-{len(configurations)})")
    #     idx = int(input().strip()) - 1
        
    #     if 0 <= idx < len(point_clouds):
    #         import matplotlib
    #         matplotlib.use("TkAgg")
    #         import matplotlib.pyplot as plt
            
    #         points, colors = point_clouds[idx]
            
    #         fig = plt.figure(figsize=(12, 10))
    #         ax = fig.add_subplot(111, projection='3d')
            
    #         ax.scatter(points[::10, 0], points[::10, 1], points[::10, 2],
    #                   c=colors[::10], s=1, alpha=0.6)
            
    #         ax.set_xlabel('X (m)')
    #         ax.set_ylabel('Y (m)')
    #         ax.set_zlabel('Z (m)')
    #         ax.set_title(f'Robot Point Cloud ({points.shape[0]} points)')
            
    #         # Equal aspect ratio
    #         max_range = np.array([points[:, 0].max() - points[:, 0].min(),
    #                             points[:, 1].max() - points[:, 1].min(),
    #                             points[:, 2].max() - points[:, 2].min()]).max() / 2.0
            
    #         mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    #         mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    #         mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
            
    #         ax.set_xlim(mid_x - max_range, mid_x + max_range)
    #         ax.set_ylim(mid_y - max_range, mid_y + max_range)
    #         ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
    #         plt.show()


if __name__ == '__main__':
    main()
