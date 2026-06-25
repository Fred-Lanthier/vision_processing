import os
import re

def numerical_sort_key(s):
    """Sorts strings containing numbers numerically (e.g., Trajectory_2 before Trajectory_10)"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def print_truncated_tree(base_path):
    if not os.path.exists(base_path):
        print(f"Directory '{base_path}' not found.")
        return

    print(f" {os.path.basename(base_path)}/")
    
    # Get and sort all Trajectory_X folders
    all_items = os.listdir(base_path)
    traj_folders = [d for d in all_items if os.path.isdir(os.path.join(base_path, d)) and d.startswith("Trajectory_")]
    traj_folders.sort(key=numerical_sort_key)
    
    # Limit to the first 3 Trajectory folders
    for traj in traj_folders[:3]:
        print(f"├── {traj}/")
        traj_path = os.path.join(base_path, traj)
        
        # Look for the .json file and the images subdirectory
        traj_items = os.listdir(traj_path)
        
        # Print the json file if it exists
        json_files = [f for f in traj_items if f.endswith('.json')]
        for json_file in json_files:
            print(f"│   ├── {json_file}")
            
        # Process the image subdirectory
        img_dir_name = f"images_{traj}"
        img_dir_path = os.path.join(traj_path, img_dir_name)
        
        if os.path.exists(img_dir_path) and os.path.isdir(img_dir_path):
            print(f"│   └── {img_dir_name}/")
            
            img_items = os.listdir(img_dir_path)
            
            # Filter and sort .png and .npy files
            png_files = [f for f in img_items if f.endswith('.png')]
            png_files.sort(key=numerical_sort_key)
            
            npy_files = [f for f in img_items if f.endswith('.npy')]
            npy_files.sort(key=numerical_sort_key)
            
            # Print first 3 PNGs
            for png in png_files[:3]:
                print(f"│       ├── {png}")
            
            jpg_files = [f for f in img_items if f.endswith('.jpg')]
            jpg_files.sort(key=numerical_sort_key)
            
            # Print first 3 JPGs
            for jpg in jpg_files[:3]:
                print(f"│       ├── {jpg}")
            
            # Print first 3 NPYs
            for npy in npy_files[:3]:
                # Adjust connector if it's the absolute last item
                connector = "└──" if npy == npy_files[:3][-1] else "├──"
                print(f"│       {connector} {npy}")

        # Process the image subdirectory
        img_dir_name = f"Merged_Fork_{traj}"
        img_dir_path = os.path.join(traj_path, img_dir_name)
        
        if os.path.exists(img_dir_path) and os.path.isdir(img_dir_path):
            print(f"│   └── {img_dir_name}/")
            
            img_items = os.listdir(img_dir_path)
            
            # Filter and sort .png and .npy files
            png_files = [f for f in img_items if f.endswith('.png')]
            png_files.sort(key=numerical_sort_key)
            
            npy_files = [f for f in img_items if f.endswith('.npy')]
            npy_files.sort(key=numerical_sort_key)
            
            # Print first 3 PNGs
            for png in png_files[:3]:
                print(f"│       ├── {png}")
            
            jpg_files = [f for f in img_items if f.endswith('.jpg')]
            jpg_files.sort(key=numerical_sort_key)
            
            # Print first 3 JPGs
            for jpg in jpg_files[:3]:
                print(f"│       ├── {jpg}")
            
            # Print first 3 NPYs
            for npy in npy_files[:3]:
                # Adjust connector if it's the absolute last item
                connector = "└──" if npy == npy_files[:3][-1] else "├──"
                print(f"│       {connector} {npy}")


if __name__ == "__main__":
    # Assumes the script is run from the directory containing 'PickPlace_record_TEST'
    print_truncated_tree("datas/Trajectories_preprocess")