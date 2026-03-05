import rospkg
import json
import matplotlib.pyplot as plt
import numpy as np
import os

rospack = rospkg.RosPack()
package_path = rospack.get_path('vision_processing')
json_path = os.path.join(package_path, "datas", "Trajectories_preprocess_TEST", "Trajectory_46", "trajectory_46.json")

with open(json_path) as f:
    datas = json.load(f)
    states = datas["states"]
    
fork_tip_np = np.zeros((len(states), 3))
step_np = np.zeros((len(states), 1))
for i, state in enumerate(states):
    fork_tip_np[i] = state["fork_tip_position"]
    step_np[i] = state["time_step"]


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(fork_tip_np[:, 0], fork_tip_np[:, 1], fork_tip_np[:, 2], label='Fork Tip Trajectory', color='blue')

# Add start and end points
ax.scatter(fork_tip_np[0, 0], fork_tip_np[0, 1], fork_tip_np[0, 2], color='green', s=100, label='Start')
ax.scatter(fork_tip_np[-1, 0], fork_tip_np[-1, 1], fork_tip_np[-1, 2], color='red', s=100, label='End')

# Find the first local minimum with respect to the z position
z_pos = fork_tip_np[:, 2]
local_minima = np.where((z_pos[1:-1] < z_pos[:-2]) & (z_pos[1:-1] < z_pos[2:]))[0] + 1
if len(local_minima) > 0:
    first_min_idx = local_minima[0]
    print(f"First local minimum in Z found at index: {first_min_idx}, time step: {step_np[first_min_idx][0]:.4f} seconds")
    ax.scatter(fork_tip_np[first_min_idx, 0], fork_tip_np[first_min_idx, 1], fork_tip_np[first_min_idx, 2], 
               color='orange', s=150, marker='*', label='First Z Local Minimum')

ax.set_title('3D Trajectory of the Fork Tip')
ax.set_xlabel('X Position (meters)')
ax.set_ylabel('Y Position (meters)')
ax.set_zlabel('Z Position (meters)')

plt.legend()
plt.show()