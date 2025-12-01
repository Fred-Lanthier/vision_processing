import rospkg
import os
import glob
import json
# Add the load of ONLY the end effector

def load_datas_path():
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vision_processing')
    datas_path = os.path.join(package_path, 'datas')
    return datas_path

def load_datas():
    # Load Trajectories paths
    datas_path = load_datas_path()
    Trajectories_path = glob.glob(os.path.join(datas_path, 'Trajectories_record', 'Trajectory*'))
    Trajectories_path = sorted(Trajectories_path, key=lambda x: int(x.split('_')[-1]))
    
    # Load all the files
    for i, trajectory_path in enumerate(Trajectories_path):
        ee_rgb = glob.glob(os.path.join(trajectory_path, 'images_Trajectory_{i}', 'ee_rgb_step_*.png'))
        ee_depth = glob.glob(os.path.join(trajectory_path, 'images_Trajectory_{i}', 'ee_depth_step_*.npy'))
        static_rgb = glob.glob(os.path.join(trajectory_path, 'images_Trajectory_{i}', 'static_rgb_step_*.png'))
        static_depth = glob.glob(os.path.join(trajectory_path, 'images_Trajectory_{i}', 'static_depth_step_*.npy'))
        Robot_urdf = glob.glob(os.path.join(trajectory_path, 'images_Trajectory_{i}', 'Robot_point_cloud_*.npy'))
        
    # Load the json file
    for i, trajectory_path in enumerate(Trajectories_path):
        json_path = os.path.join(trajectory_path, 'trajectory_{i}.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for step in data['steps']:
            
        

    return Trajectories_path

def main():
    datas = load_datas()
    print(datas)

if __name__ == '__main__':
    main()
    
    
    
