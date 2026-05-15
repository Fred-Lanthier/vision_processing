# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **ROS Noetic catkin package** (`vision_processing`) implementing a safety-aware generative planning pipeline for robotic assisted feeding. A Franka Panda arm with a wrist-mounted RealSense camera picks up food items and delivers them to a person. The pipeline layers perception, Flow Matching trajectory generation, and a real-time CBF safety shield.

## Build

This is a catkin package. Build from the catkin workspace root (parent of `src/`):

```bash
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

The build compiles two pybind11 C++ modules:
- `fast_ik_module` — pinocchio-based IK (`src/vision_processing/fast_ik.cpp`)
- `fast_perception_module` — PCL-based point cloud filtering (`src/vision_processing/fast_perception.cpp`)

Both modules are output to `devel/lib/python3/dist-packages/vision_processing/` and imported as `from vision_processing import fast_ik_module`.

The CMakeLists.txt hardcodes the venv Python path for pybind11 discovery:
```
PYTHON_EXECUTABLE = venv_sam3/bin/python3
```

## Python Environment

All nodes use the venv at `venv_sam3/` (package root). SAM3 model loading requires this environment. Activate before running nodes manually:

```bash
source venv_sam3/bin/activate
```

## Running the System

**Full Gazebo simulation + green cube feeding pipeline:**
```bash
roslaunch vision_processing green_cube_feeding.launch
```

**CASF variant (Riemannian Metric Warping instead of post-hoc CBF):**
```bash
roslaunch vision_processing green_cube_feeding_casf.launch
```

**Lightweight mock pipeline (no Gazebo, uses dummy nodes):**
```bash
roslaunch vision_processing safe_generative_planner.launch
```

**Gazebo environment only:**
```bash
roslaunch vision_processing feeding_simulation.launch
```

## Architecture: Node Pipeline

The system runs as a chain of ROS nodes at different frequencies:

```
Camera (~30fps)
    └─► sam3_sam2_node.py          — SAM3 (zero-shot detect) + SAM2/Muggled SAM (track)
            ├─► /perception/target          (target food point cloud)
            └─► /perception/obstacles       (raw obstacle cloud)

/perception/obstacles
    └─► perception_processing_node.py  [30Hz] — CropBox + robot self-filter (Bernstein SDF) + ROR
            └─► /perception/cleaned_obstacles

/perception/target + /joint_states
    └─► condition_pcd_from_perception.py   — merges target PCD + fork mesh PCD
            └─► /vision/merged_cloud        (1024 pts, model input)

/vision/merged_cloud + /joint_states
    └─► safe_generative_node.py        [~8Hz] — Flow Matching inference (27D trajectory)
        OR casf_generative_node_PC.py  [~8Hz] — CASF with Riemannian metric warping
            └─► /vision/nominal_trajectory

/vision/nominal_trajectory + /perception/cleaned_obstacles + /joint_states
    └─► cbf_safety_node_Bernstein.py   [150Hz] — Bernstein-Basis CBF safety filter
            └─► /joint_group_position_controller/command
```

The **CASF pipeline** (`green_cube_feeding_casf.launch`) inserts a `point_cloud_projector_node.py` between SAM2 mask output and the obstacle cloud, and replaces the CBF post-filter with inline Riemannian metric warping inside the planner.

## Key Concepts

**Action Space (9D SE(3))**: Trajectories are represented as 9D vectors — 3D position + 6D rotation (two column vectors, Gram-Schmidt orthogonalized via `decode_9d_to_se3_gpu`). The 27D variant adds velocity and acceleration (9D × 3) for smooth dynamics.

**Flow Matching Model**: A 1D U-Net conditioned on a global point cloud embedding. Input: robot state + point cloud. Output: trajectory sequence in 9D/27D. Trained in `diffusion_model_train/Train_Fork_FlowMP.py`. Checkpoints in `models/`.

**Temporal Ensembler**: Both planner nodes (`safe_generative_node.py`, `casf_generative_node_PC.py`) maintain a ring buffer of recent predictions and blend them with exponential weighting to smooth outputs.

**Bernstein-Basis CBF**: The safety shield uses robot SDF approximations (Bernstein polynomial basis via `third_party/SDF_Bernstein_Basis`) to enforce minimum clearance from obstacles. It solves a QP at 150Hz to filter the nominal joint velocity command. Uses `fast_perception_module` for GPU-accelerated PCL operations.

**RDF / URDFLayer**: `third_party/RDF` provides differentiable robot kinematics (`URDFLayer`) used by both the perception filter and the CBF. Voxelized link meshes are in `third_party/RDF/panda_layer/meshes/voxel_128/`.

## Submodules

Initialize all submodules after cloning:
```bash
git submodule update --init --recursive
```

| Submodule | Purpose |
|---|---|
| `third_party/RDF` | Differentiable robot kinematics (URDFLayer) |
| `third_party/SDF_Bernstein_Basis` | Bernstein-basis robot SDF for CBF |
| `third_party/SafeFlowMatcher` | RDF_CBF solver (legacy, partially used) |
| `third_party/flow_mp` | Flow matching utilities |
| `third_party/muggled_sam` | SAM2/Samurai video tracker |
| `third_party/tapnet` | Point tracking network |

## Training

Training scripts are in `src/vision_processing/diffusion_model_train/`. Run directly with the venv Python:

```bash
cd src/vision_processing/diffusion_model_train/
python Train_Fork_FlowMP.py        # 27D dynamics model (recommended)
python Train_Fork_FlowMP_9D.py     # 9D pose-only model
python Train_Fork_FM.py            # basic flow matching
```

Normalization stats are saved as `normalization_stats_*.json` at the package root and must match the model checkpoint used at inference.

## ROS Services and Messages

Custom services in `srv/`:
- `ProcessImage.srv` — food segmentation from RGB image
- `ProcessRobot.srv` — robot segmentation from RGBD
- `Sam3Segment.srv` — SAM3 zero-shot segmentation (used by `sam3_server.py`)

`sam3_server.py` loads the SAM3 model once at startup (30-60s) and serves `/sam3/segment`. All other nodes call this service instead of loading the model themselves.

## URDF / Robot Description

- `urdf/panda_camera.xacro` — Panda arm with wrist camera and fork attachment (primary URDF)
- `urdf/panda_fork.urdf` — Fork-only description

Several nodes call `xacro.process_file()` directly at startup to produce a temp URDF file for pinocchio/URDFLayer.

## ROS Network Configuration

For real robot operation, set in `config_vision.sh`:
```bash
export ROS_MASTER_URI=http://132.207.24.13:11311
export ROS_IP=132.207.24.13
```
