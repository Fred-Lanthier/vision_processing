import gymnasium as gym
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import sapien
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.agents.robots.panda import Panda
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig

import os
import glob
import json
import numpy as np
from PIL import Image
import rospkg
import time

rospack = rospkg.RosPack()
VP_PATH = rospack.get_path('vision_processing')
os.makedirs(os.path.join(VP_PATH, "datas", "Trajectories_record_TEST"), exist_ok=True)
BASE_RECORD_DIR = os.path.join(VP_PATH, "datas", "Trajectories_record_TEST")
BASE_PREPROCESS_DIR = os.path.join(VP_PATH, "datas", "Trajectories_preprocess")


# =================================================================
# FILE I/O HELPERS
# =================================================================

def get_next_trajectory_id(base_directories):
    json_files = []
    for base_directory in base_directories:
        json_files.extend(glob.glob(os.path.join(base_directory, "Trajectory_*")))
    if not json_files:
        return 1
    trajectory_ids = []
    for filename in json_files:
        try:
            base_name = os.path.basename(filename)
            id_str = base_name.replace("Trajectory_", "")
            trajectory_ids.append(int(id_str))
        except ValueError:
            continue
    for i in range(1, len(trajectory_ids) + 1):
        if i not in trajectory_ids:
            return i
    if trajectory_ids:
        return max(trajectory_ids) + 1
    else:
        return 1

def get_next_n_trajectory_ids(base_directories, n):
    first_id = get_next_trajectory_id(base_directories)
    return list(range(first_id, first_id + n))

def create_trajectory_directories(base_directory, trajectory_id):
    trajectory_folder_name = f"Trajectory_{trajectory_id}"
    trajectory_path = os.path.join(base_directory, trajectory_folder_name)
    os.makedirs(trajectory_path, exist_ok=True)
    image_directory = os.path.join(trajectory_path, f"images_Trajectory_{trajectory_id}")
    os.makedirs(image_directory, exist_ok=True)
    print(f"Created trajectory directories: {trajectory_path}")
    return trajectory_path, image_directory

def save_states_to_json(trajectory_path, trajectory_id, states, current_task, food_type, plate_type, camera_pose):
    filename = os.path.join(trajectory_path, f'trajectory_{trajectory_id}.json')
    data = {
        'trajectory_id': trajectory_id,
        'states': states,
        'current_task': current_task,
        'food_type': food_type,
        'plate_type': plate_type,
        'T_static_s': camera_pose["T_static_s"],
        'T_ee_s': camera_pose["T_ee_s"]
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved trajectory data: {filename}")


# =================================================================
# 1. CUSTOM PANDA FORK AGENT
# =================================================================

@register_agent()
class PandaFork(Panda):
    uid = "panda_fork"
    urdf_path = os.path.join(VP_PATH, "urdf", "panda_fork.urdf")

    @property
    def _ee_link_name(self):
        return "fork_tip"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="ee",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=640, height=480, fov=np.deg2rad(42), near=0.01, far=2.0,
                mount=self.robot.links_map["camera_wrist_link"],
            ),
            CameraConfig(
                uid="static",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=640, height=480, fov=np.deg2rad(42), near=0.01, far=5.0,
                mount=self.robot.links_map["camera_static_link"],
            )
        ]


# =================================================================
# 2. FOOD CONFIGURATION
# =================================================================
#
# FOOD_BUILD_CONFIGS replaces the old if/elif chain in _load_scene.
# All mesh/scale/pose info for building each food actor lives here.
#
#   half_z:      the "food radius" used to compute spiking depth
#   yaw_offset:  if True, base_yaw gets +90° (organic shapes like broccoli/chicken)

FOOD_BUILD_CONFIGS = {
    "patate": {
        "mesh_path": "object/simple_potato.glb",
        "scale": [0.0075, 0.0075, 0.0075],
        "half_z": 0.0075,
        "local_p": [0.0, 0.0, 0.0],
        "local_euler_xyz_deg": [0, 0, 0],
        "color": [0.15, 0.1, 0.05, 1.0],
        "yaw_offset": False,
    },
    "chicken": {
        "mesh_path": "object/chicken_breast_grilled.glb",
        "scale": [0.05, 0.025, 0.025],
        "half_z": 0.0025,
        "local_p": [0.0, 0.0, 0.0],
        "local_euler_xyz_deg": [90, 0, 0],
        "color": [0.25, 0.2, 0.15, 1.0],
        "yaw_offset": True,
    },
    "broccoli": {
        "mesh_path": "object/game-ready_free_broccoli.glb",
        "scale": [0.005, 0.005, 0.005],
        "half_z": 0.0015,
        "local_p": [0.0, 0.0, 0.005],
        "local_euler_xyz_deg": [15, 0, 0],
        "color": [0.05, 0.05, 0.05, 1.0],
        "yaw_offset": True,
    },
    "sushi": {
        "mesh_path": "object/sushi.glb",
        "scale": [0.4, 0.4, 0.4],
        "half_z": 0.00,
        "local_p": [0.0, 0.0, 0.0],
        "local_euler_xyz_deg": [90, 0, 0],
        "color": [0.05, 0.05, 0.05, 1.0],
        "yaw_offset": False,
    },
    "sushi_roll": {
        "mesh_path": "object/sushi_roll.glb",
        "scale": [0.4, 0.4, 0.4],
        "half_z": 0.0075,
        "local_p": [0.0, 0.0, 0.0],
        "local_euler_xyz_deg": [90, 0, 0],
        "color": [0.05, 0.05, 0.05, 1.0],
        "yaw_offset": False,
    },
    "sushi_cube": {
        "mesh_path": "object/salmon_sushi_-_free_giveaway.glb",
        "scale": [0.03, 0.03, 0.03],
        "half_z": 0.0135,
        "local_p": [0.0, 0.0, 0.0],
        "local_euler_xyz_deg": [90, 0, 0],
        "color": [0.05, 0.05, 0.05, 1.0],
        "yaw_offset": False,
    },
    "sausage": {
        "mesh_path": "object/low_poly_sausage.glb",
        "scale": [0.01, 0.01, 0.01],
        "half_z": 0.0075,
        "local_p": [0.0, -0.016, 0.0],
        "local_euler_xyz_deg": [0, -50, 0],
        "color": [0.05, 0.05, 0.05, 1.0],
        "yaw_offset": False,
    },
}

ALL_FOOD_TYPES = list(FOOD_BUILD_CONFIGS.keys())


# =================================================================
# 3. ENVIRONMENT
# =================================================================
#
# KEY IDEA: We build ALL 7 food actors in every scene.
# Each env gets a random food type per episode.
# The "active" actor sits on the pedestal; the other 6 are hidden at (99, 99, -10).
#
# Two helper methods handle the routing:
#   get_active_food_poses()      -> (p, q)   reads from the correct actor per env
#   set_active_food_poses(p, q)              writes to the correct actor per env

@register_env("FoodSpiking-v1", max_episode_steps=750)
class FoodSpikingEnv(PickCubeEnv):

    def __init__(self, *args, **kwargs):
        self.per_env_food_type = []
        self.per_env_food_half_z = None
        self.per_env_yaw_offset = None
        self.food_actors = {}
        self.food_masks = {}
        self.principal_axis_world = None
        super().__init__(*args, **kwargs)

    # ---------------------------------------------------------
    # SCENE LOADING: build ALL 7 food actors + pedestal
    # ---------------------------------------------------------
    def _load_scene(self, options: dict):
        super()._load_scene(options)

        self.food_actors = {}
        for food_type, cfg in FOOD_BUILD_CONFIGS.items():
            mat = sapien.render.RenderMaterial()
            mat.base_color = cfg["color"]
            mat.roughness = 0.8
            mat.metallic = 0.0

            local_q_xyzw = R.from_euler('xyz', cfg["local_euler_xyz_deg"], degrees=True).as_quat()
            local_q_wxyz = [local_q_xyzw[3], local_q_xyzw[0], local_q_xyzw[1], local_q_xyzw[2]]
            local_pose = sapien.Pose(p=cfg["local_p"], q=local_q_wxyz)

            builder = self.scene.create_actor_builder()
            builder.add_visual_from_file(
                filename=cfg["mesh_path"],
                scale=cfg["scale"],
                material=mat,
                pose=local_pose
            )
            self.food_actors[food_type] = builder.build_kinematic(name=f"food_{food_type}")

        ped_builder = self.scene.create_actor_builder()
        self.pedestal_half_z = 0.20
        ped_builder.add_box_collision(half_size=[0.05, 0.05, self.pedestal_half_z])
        ped_builder.add_box_visual(half_size=[0.05, 0.05, self.pedestal_half_z])
        self.pedestal = ped_builder.build_kinematic(name="pedestal")

    # ---------------------------------------------------------
    # EPISODE INIT: random food type per env, place actors
    # ---------------------------------------------------------
    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        b = len(env_idx)

        homing_qpos = torch.tensor(
            [[-0.000059, -0.125928, 0.000117, -2.193312,
              -0.000251, 2.064780, 0.785511, 0.04, 0.04]],
            device=self.device
        ).repeat(b, 1)
        self.agent.reset(homing_qpos)

        far_away = torch.tensor([[99.0, 99.0, -10.0]], device=self.device).repeat(b, 1)
        self.cube.set_pose(Pose.create_from_pq(p=far_away))

        # =========================================================
        # RANDOM FOOD TYPE PER ENV
        # =========================================================
        self.per_env_food_type = np.random.choice(ALL_FOOD_TYPES, size=b).tolist()

        self.per_env_food_half_z = torch.tensor(
            [FOOD_BUILD_CONFIGS[ft]["half_z"] for ft in self.per_env_food_type],
            device=self.device, dtype=torch.float32
        )
        self.per_env_yaw_offset = torch.tensor(
            [FOOD_BUILD_CONFIGS[ft]["yaw_offset"] for ft in self.per_env_food_type],
            device=self.device, dtype=torch.bool
        )

        # Pre-compute masks (which envs use which food type)
        # so we don't rebuild them every step
        self.food_masks = {}
        for food_type in ALL_FOOD_TYPES:
            self.food_masks[food_type] = torch.tensor(
                [ft == food_type for ft in self.per_env_food_type],
                dtype=torch.bool, device=self.device
            )

        # =========================================================
        # RANDOM POSITIONS AND ORIENTATIONS
        # =========================================================
        rand_x = -0.23 + torch.rand(b, device=self.device) * 0.12
        rand_y = -0.14 + torch.rand(b, device=self.device) * 0.16
        random_z_offset = torch.rand(b, device=self.device) * 0.08

        self.target_yaw = (torch.rand(b, device=self.device) * 2 * np.pi) - np.pi
        yaw_np = self.target_yaw.cpu().numpy()

        food_q = R.from_euler("z", yaw_np).as_quat()
        food_q_wxyz = np.concatenate((food_q[:, 3:4], food_q[:, 0:3]), axis=1)
        food_q_tensor = torch.tensor(food_q_wxyz, device=self.device, dtype=torch.float32)

        local_axis = np.array([1., 0., 0.])
        c, s = np.cos(yaw_np), np.sin(yaw_np)
        principal_axis_np = np.stack([
            c * local_axis[0] - s * local_axis[1],
            s * local_axis[0] + c * local_axis[1],
            np.zeros(b)
        ], axis=-1)
        self.principal_axis_world = torch.tensor(principal_axis_np, device=self.device, dtype=torch.float32)

        food_pos = torch.zeros((b, 3), device=self.device)
        food_pos[:, 0] = rand_x
        food_pos[:, 1] = rand_y
        food_pos[:, 2] = random_z_offset + self.per_env_food_half_z

        # =========================================================
        # PLACE ACTORS: active one on pedestal, others hidden
        # =========================================================
        far_p = torch.tensor([[99.0, 99.0, -10.0]], device=self.device).repeat(b, 1)
        far_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(b, 1)

        for food_type, actor in self.food_actors.items():
            mask = self.food_masks[food_type]
            p = far_p.clone()
            q = far_q.clone()
            if mask.any():
                p[mask] = food_pos[mask]
                q[mask] = food_q_tensor[mask]
            actor.set_pose(Pose.create_from_pq(p=p, q=q))

        ped_pos = food_pos.clone()
        ped_pos[:, 2] = random_z_offset - self.pedestal_half_z
        self.pedestal.set_pose(Pose.create_from_pq(p=ped_pos))

    # ---------------------------------------------------------
    # HELPERS: read/write food poses through the correct actors
    # ---------------------------------------------------------
    def get_active_food_poses(self):
        """Gather (p [B,3], q [B,4]) from each env's active food actor."""
        B = self.num_envs
        p = torch.zeros((B, 3), device=self.device)
        q = torch.zeros((B, 4), device=self.device)
        q[:, 0] = 1.0

        for food_type, actor in self.food_actors.items():
            mask = self.food_masks[food_type]
            if mask.any():
                pose = actor.pose
                p[mask] = pose.p[mask]
                q[mask] = pose.q[mask]
        return p, q

    def set_active_food_poses(self, p, q):
        """Scatter poses to the correct actors. Inactive envs stay hidden."""
        B = self.num_envs
        FAR_P = torch.tensor([[99.0, 99.0, -10.0]], device=self.device).expand(B, 3)
        FAR_Q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(B, 4)

        for food_type, actor in self.food_actors.items():
            mask = self.food_masks[food_type]
            new_p = FAR_P.clone()
            new_q = FAR_Q.clone()
            if mask.any():
                new_p[mask] = p[mask]
                new_q[mask] = q[mask]
            actor.set_pose(Pose.create_from_pq(p=new_p.contiguous(), q=new_q.contiguous()))

    def get_desired_fork_yaw(self):
        yaws_rad = torch.atan2(self.principal_axis_world[:, 1], self.principal_axis_world[:, 0])
        return torch.rad2deg(yaws_rad)


# =================================================================
# 4. BATCHED DATA COLLECTION
# =================================================================

if __name__ == "__main__":

    num_envs = 2
    env = gym.make(
        "FoodSpiking-v1",
        num_envs=num_envs,
        robot_uids="panda_fork",
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
    )

    NUM_EPISODES = 1
    os.makedirs(BASE_RECORD_DIR, exist_ok=True)
    starts = time.time()

    uw = env.unwrapped
    B = uw.num_envs
    device = env.device

    sapien2opencv = torch.tensor([
            [ 0.0,  0.0,  1.0,  0.0],
            [-1.0,  0.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0,  0.0],
            [ 0.0,  0.0,  0.0,  1.0]
        ], dtype=torch.float32, device=device)
    
    for ep in range(NUM_EPISODES):

        obs, _ = env.reset(seed=ep, options=dict(reconfigure=True))
        robot = uw.agent.robot
        links = {l.name: l for l in robot.get_links()}

        # ==============================================================
        # PER-ENV SETUP
        # ==============================================================
        traj_ids = get_next_n_trajectory_ids([BASE_RECORD_DIR, BASE_PREPROCESS_DIR], B)
        traj_paths = []
        img_dirs = []
        for b_idx in range(B):
            tp, imd = create_trajectory_directories(BASE_RECORD_DIR, traj_ids[b_idx])
            traj_paths.append(tp)
            img_dirs.append(imd)

        base_pose = robot.pose
        cam_static_pose = links["camera_static_link"].pose
        rel_static_pose = base_pose.inv() * cam_static_pose

        cam_wrist_pose = links["camera_wrist_link"].pose
        rel_wrist_pose = base_pose.inv() * cam_wrist_pose
        
        # 1. On extrait la matrice SAPIEN relative à la base [B, 4, 4]
        T_ee_s_sapien = rel_wrist_pose.to_transformation_matrix()
        T_static_s_sapien = rel_static_pose.to_transformation_matrix()

        # 2. On la convertit en OpenCV via une multiplication matricielle
        T_ee_s_cv = torch.matmul(T_ee_s_sapien, sapien2opencv)
        T_static_s_cv = torch.matmul(T_static_s_sapien, sapien2opencv)

        # 3. ON ENREGISTRE LES MATRICES CORRIGÉES (Et non plus les anciennes !)
        camera_poses_batch = []
        for b_idx in range(B):
            camera_poses_batch.append({
                "T_static_s": T_static_s_cv[b_idx].tolist(),
                "T_ee_s": T_ee_s_cv[b_idx].tolist()
            })

        states_batch = [[] for _ in range(B)]
        rgb_static_bufs = [[] for _ in range(B)]
        depth_static_bufs = [[] for _ in range(B)]
        rgb_ee_bufs = [[] for _ in range(B)]
        depth_ee_bufs = [[] for _ in range(B)]
        record_step_count = torch.zeros(B, dtype=torch.int32, device=device)

        null_action = torch.zeros((B, 7), dtype=torch.float32, device=device)
        for _ in range(25):
            env.step(null_action)

        food_names = uw.per_env_food_type
        print(f"\n--- Ep {ep+1}/{NUM_EPISODES} | Foods: {food_names} | Batch: {B} ---")

        # ==============================================================
        # VECTORIZED STATE TENSORS
        # ==============================================================
        current_phase = torch.ones(B, dtype=torch.long, device=device)
        hover_reached = torch.zeros(B, dtype=torch.bool, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        step_count = torch.zeros(B, dtype=torch.int32, device=device)
        hold_steps = torch.zeros(B, dtype=torch.int32, device=device)

        locked_fork_pos = torch.zeros((B, 3), dtype=torch.float32, device=device)
        pitch_start = torch.zeros(B, dtype=torch.float32, device=device)
        locked_relative_p = torch.zeros((B, 3), dtype=torch.float32, device=device)
        locked_relative_q = torch.zeros((B, 4), dtype=torch.float32, device=device)
        locked_relative_q[:, 0] = 1.0
        has_locked_pose = torch.zeros(B, dtype=torch.bool, device=device)

        USER_MOUTH_POS = torch.tensor([-0.2, 0.3, 0.16], dtype=torch.float32, device=device).unsqueeze(0).expand(B, 3)

        # ==============================================================
        # BASE YAW: per-env offset via uw.per_env_yaw_offset
        # ==============================================================
        base_yaw = uw.get_desired_fork_yaw()
        if not isinstance(base_yaw, torch.Tensor):
            base_yaw = torch.tensor(base_yaw, dtype=torch.float32, device=device)

        # Only broccoli/chicken envs get +90° (per_env_yaw_offset is a [B] bool tensor)
        base_yaw[uw.per_env_yaw_offset] += 90.0
        base_yaw = (base_yaw + 180) % 360 - 180

        desired_approach_yaw = base_yaw.clone()
        front_mask = (base_yaw > -90.0) & (base_yaw < 90.0)
        desired_approach_yaw[front_mask & (base_yaw >= 0.0)] -= 180.0
        desired_approach_yaw[front_mask & (base_yaw < 0.0)] += 180.0

        desired_approach_pitch = -10.0
        R_scoop = 0.035

        RECORD_INTERVAL = 0.1
        steps_per_record = max(1, round(RECORD_INTERVAL / uw.control_timestep))

        # ==============================================================
        # MAIN SIMULATION LOOP
        # ==============================================================
        while not done.all():
            links = {l.name: l for l in uw.agent.robot.get_links()}
            panda_tcp = links["panda_TCP"]
            fork_tip = links["fork_tip"]

            fork_pose = fork_tip.pose
            fork_pos = fork_pose.p

            # Read food poses through per-env routing
            food_pos, food_q = uw.get_active_food_poses()

            q_wxyz = fork_pose.q.cpu().numpy()
            q_xyzw = np.concatenate((q_wxyz[:, 1:], q_wxyz[:, 0:1]), axis=1)
            r = R.from_quat(q_xyzw)
            euler = r.as_euler("xyz", degrees=True)
            roll = torch.tensor(euler[:, 0], dtype=torch.float32, device=device)
            pitch = torch.tensor(euler[:, 1], dtype=torch.float32, device=device)
            yaw = torch.tensor(euler[:, 2], dtype=torch.float32, device=device)

            action = torch.zeros((B, 7), dtype=torch.float32, device=device)
            action[:, 6] = 1.0

            active_mask = ~done
            phase1_mask = (current_phase == 1) & active_mask
            phase2_mask = (current_phase == 2) & active_mask
            phase3_mask = (current_phase == 3) & active_mask

            # =========================================================
            # RECORDING
            # =========================================================
            base_pose = uw.agent.robot.pose
            rel_tcp_pose = base_pose.inv() * panda_tcp.pose

            for b_idx in range(B):
                if done[b_idx]:
                    continue
                if int(step_count[b_idx]) % steps_per_record != 0:
                    continue

                rec_idx = int(record_step_count[b_idx])
                sensors = obs["sensor_data"]

                # --- SAUVEGARDE DES IMAGES (C'est de retour !) ---
                rgb_static_bufs[b_idx].append(sensors["static"]["rgb"][b_idx].cpu().numpy())
                depth_static_bufs[b_idx].append(sensors["static"]["depth"][b_idx].cpu().numpy())
                rgb_ee_bufs[b_idx].append(sensors["ee"]["rgb"][b_idx].cpu().numpy())
                depth_ee_bufs[b_idx].append(sensors["ee"]["depth"][b_idx].cpu().numpy())
                # -------------------------------------------------

                time_step = int(step_count[b_idx]) * uw.control_timestep
                states_batch[b_idx].append({
                    "step_number": rec_idx,
                    "simulation_step": int(step_count[b_idx]),
                    "time_step": round(time_step, 4),
                    "joint_positions": robot.get_qpos()[b_idx].cpu().numpy().tolist(),
                    "joint_velocities": robot.get_qvel()[b_idx].cpu().numpy().tolist(),
                    
                    # CORRECTION DU RÉFÉRENTIEL : rel_tcp_pose au lieu de panda_tcp.pose
                    "end_effector_position": rel_tcp_pose.p[b_idx].cpu().numpy().tolist(),
                    "end_effector_orientation": rel_tcp_pose.q[b_idx].cpu().numpy().tolist(),
                    
                    "end_effector_velocity": panda_tcp.linear_velocity[b_idx].cpu().numpy().tolist() +
                                             panda_tcp.angular_velocity[b_idx].cpu().numpy().tolist(),
                    "phase": int(current_phase[b_idx]),
                    "camera_static": {
                        "rgb": f"static_rgb_step_{rec_idx:04d}.png",
                        "depth": None,
                        "depth_npy": f"static_depth_step_{rec_idx:04d}.npy"
                    },
                    "camera_ee": {
                        "rgb": f"ee_rgb_step_{rec_idx:04d}.png",
                        "depth": None,
                        "depth_npy": f"ee_depth_step_{rec_idx:04d}.npy"
                    }
                })
                # TRÈS IMPORTANT : Ne pas oublier de l'incrémenter !
                record_step_count[b_idx] += 1

            # ---------------------------------------------------------
            # PHASE 1: APPROACH + YAW ALIGNMENT
            # ---------------------------------------------------------
            if phase1_mask.any():
                hover_pos = food_pos.clone()
                hover_pos[:, 2] += 0.03

                yaw_err = desired_approach_yaw - yaw
                yaw_err = torch.where(yaw_err > 180, yaw_err - 360, yaw_err)
                yaw_err = torch.where(yaw_err < -180, yaw_err + 360, yaw_err)
                yaw_update_mask = phase1_mask & (torch.abs(yaw_err) > 2.0)
                action[yaw_update_mask, 5] = -torch.clamp(torch.deg2rad(yaw_err[yaw_update_mask]) * 0.7, -0.7, 0.7)

                pitch_err = desired_approach_pitch - pitch
                pitch_err = torch.where(pitch_err > 180, pitch_err - 360, pitch_err)
                pitch_err = torch.where(pitch_err < -180, pitch_err + 360, pitch_err)
                pitch_update_mask = phase1_mask & (torch.abs(pitch_err) > 2.0)
                action[pitch_update_mask, 4] = torch.clamp(torch.deg2rad(pitch_err[pitch_update_mask]) * 0.7, -0.7, 0.7)

                roll_err = 0 - roll
                roll_err = torch.where(roll_err > 180, roll_err - 360, roll_err)
                roll_err = torch.where(roll_err < -180, roll_err + 360, roll_err)
                roll_update_mask = phase1_mask & (torch.abs(roll_err) > 2.0)
                action[roll_update_mask, 3] = torch.clamp(torch.deg2rad(roll_err[roll_update_mask]) * 0.7, -0.7, 0.7)

                moving_mask = phase1_mask & ~hover_reached
                if moving_mask.any():
                    direction = hover_pos[moving_mask] - fork_pos[moving_mask]
                    dist = torch.norm(direction, dim=1, keepdim=True)
                    arrived = (dist.squeeze(-1) < 0.01)
                    arrived_indices = moving_mask.nonzero(as_tuple=True)[0][arrived]
                    hover_reached[arrived_indices] = True
                    still_far = ~arrived
                    far_indices = moving_mask.nonzero(as_tuple=True)[0][still_far]
                    if len(far_indices) > 0:
                        dir_norm = direction[still_far] / dist[still_far]
                        speed = torch.clamp(dist[still_far] * 5.0, max=0.05)
                        action[far_indices, :3] = dir_norm * speed

                spiking_mask = phase1_mask & hover_reached
                if spiking_mask.any():
                    # Per-env food_half_z from config
                    food_half_z = uw.per_env_food_half_z
                    z_top = food_pos[:, 2] + food_half_z
                    theoretical_penetration = torch.clamp(2.0 * food_half_z - 0.002, max=0.015)
                    target_z = z_top - theoretical_penetration
                    pedestal_surface_z = uw.pedestal.pose.p[:, 2] + uw.pedestal_half_z
                    target_z = torch.max(target_z, pedestal_surface_z + 0.004)

                    above_mask = spiking_mask & (fork_pos[:, 2] > target_z + 0.001)
                    if above_mask.any():
                        target_spike_pos = food_pos[above_mask].clone()
                        target_spike_pos[:, 2] = target_z[above_mask]
                        direction = target_spike_pos - fork_pos[above_mask]
                        action[above_mask, :3] = torch.clamp(direction * 3.0, min=-0.02, max=0.02)

                    spiked_mask = spiking_mask & ~above_mask
                    if spiked_mask.any():
                        idx = spiked_mask.nonzero(as_tuple=True)[0]
                        current_phase[idx] = 2
                        locked_fork_pos[idx] = fork_pos[idx].clone()
                        pitch_start[idx] = pitch[idx].clone()

                        f_pose_iso = Pose.create_from_pq(p=fork_pos[idx].clone(), q=fork_pose.q[idx].clone())
                        fd_pose_iso = Pose.create_from_pq(p=food_pos[idx].clone(), q=food_q[idx].clone())
                        rel_pose_iso = f_pose_iso.inv() * fd_pose_iso

                        locked_relative_p[idx] = rel_pose_iso.p.clone()
                        locked_relative_q[idx] = rel_pose_iso.q.clone()
                        has_locked_pose[idx] = True

            # ---------------------------------------------------------
            # PHASE 2: WRIST ROTATION
            # ---------------------------------------------------------
            if phase2_mask.any():
                angle_diff = torch.abs(pitch - pitch_start)
                alpha = torch.clamp(angle_diff / 90.0, 0.0, 1.0)

                rad_yaw = torch.deg2rad(yaw)
                forward_xy = torch.stack([torch.cos(rad_yaw), torch.sin(rad_yaw), torch.zeros_like(rad_yaw)], dim=1)
                up_z = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0).expand(B, 3)

                target_arc_pos = locked_fork_pos + \
                                 forward_xy * (R_scoop * (1 - torch.cos(alpha * np.pi / 2)).unsqueeze(1)) + \
                                 up_z * (R_scoop * torch.sin(alpha * np.pi / 2).unsqueeze(1))

                pos_err = target_arc_pos - fork_pos
                action[phase2_mask, :3] = torch.clamp(pos_err[phase2_mask] * 5.0, min=-0.05, max=0.05)

                local_y_axis = np.array([[0.0, 1.0, 0.0]] * B)
                world_rotation_axis = torch.tensor(r.apply(local_y_axis), dtype=torch.float32, device=device)
                action[phase2_mask, 3:6] = world_rotation_axis[phase2_mask] * 0.25

                done_scoop = phase2_mask & (torch.abs(pitch) >= 80.0)
                if done_scoop.any():
                    current_phase[done_scoop] = 3

            # ---------------------------------------------------------
            # PHASE 3: DELIVER
            # ---------------------------------------------------------
            if phase3_mask.any():
                absolute_target_yaw = 90.0
                dir_to_user = USER_MOUTH_POS - fork_pos
                dist = torch.norm(dir_to_user, dim=1)

                yaw_err = absolute_target_yaw - yaw
                yaw_err = torch.where(yaw_err > 180, yaw_err - 360, yaw_err)
                yaw_err = torch.where(yaw_err < -180, yaw_err + 360, yaw_err)

                speed_factor = torch.clamp(1.0 - (torch.abs(yaw_err) / 60.0), 0.1, 1.0)
                max_trans_speed = 0.075 * speed_factor

                trans_mask = phase3_mask & (dist > 0.01)
                if trans_mask.any():
                    action_trans = dir_to_user[trans_mask] * 3.0 * speed_factor[trans_mask].unsqueeze(1)
                    max_spd = max_trans_speed[trans_mask].unsqueeze(1)
                    action[trans_mask, :3] = torch.max(torch.min(action_trans, max_spd), -max_spd)

                rot_mask = phase3_mask & (torch.abs(yaw_err) > 1.5)
                if rot_mask.any():
                    action[rot_mask, 5] = -torch.clamp(torch.deg2rad(yaw_err[rot_mask]) * 0.7, -0.5, 0.5)

                hold_mask = phase3_mask & (dist <= 0.01) & (torch.abs(yaw_err) <= 1.5)
                if hold_mask.any():
                    hold_steps[hold_mask] += 1
                    done_mask = hold_mask & (hold_steps > 15)
                    done[done_mask] = True

            # ---------------------------------------------------------
            # GHOST KINEMATIC + STEP
            # ---------------------------------------------------------
            step_count[~done] += 1

            if has_locked_pose.any():
                idx = has_locked_pose.nonzero(as_tuple=True)[0]
                f_pose_active = Pose.create_from_pq(p=fork_pos[idx].clone(), q=fork_pose.q[idx].clone())
                rel_pose_active = Pose.create_from_pq(p=locked_relative_p[idx].clone(), q=locked_relative_q[idx].clone())
                target_pose_active = f_pose_active * rel_pose_active

                new_food_p = food_pos.clone()
                new_food_q = food_q.clone()
                new_food_p[idx] = target_pose_active.p.clone()
                new_food_q[idx] = target_pose_active.q.clone()

                uw.set_active_food_poses(new_food_p, new_food_q)
                uw.scene._gpu_apply_all()

            obs, reward, terminated, truncated, info = env.step(action)

            if truncated.any():
                done[truncated] = True

        # ==============================================================
        # END OF EPISODE: Flush B buffers to disk
        # ==============================================================
        print(f"    Episode done. Writing {B} trajectories to disk...")

        for b_idx in range(B):
            n_frames = len(rgb_static_bufs[b_idx])
            print(f"      Env {b_idx} ({food_names[b_idx]}): {n_frames} frames")

            for i in range(n_frames):
                Image.fromarray(rgb_static_bufs[b_idx][i]).save(
                    os.path.join(img_dirs[b_idx], f"static_rgb_step_{i:04d}.png"))
                np.save(
                    os.path.join(img_dirs[b_idx], f"static_depth_step_{i:04d}.npy"),
                    depth_static_bufs[b_idx][i])
                Image.fromarray(rgb_ee_bufs[b_idx][i]).save(
                    os.path.join(img_dirs[b_idx], f"ee_rgb_step_{i:04d}.png"))
                np.save(
                    os.path.join(img_dirs[b_idx], f"ee_depth_step_{i:04d}.npy"),
                    depth_ee_bufs[b_idx][i])

            save_states_to_json(
                trajectory_path=traj_paths[b_idx],
                trajectory_id=traj_ids[b_idx],
                states=states_batch[b_idx],
                current_task="FoodSpiking",
                food_type=food_names[b_idx],
                plate_type="pedestal",
                camera_pose=camera_poses_batch[b_idx]
            )

    end = time.time()
    print(f"\nData collection completed in {(end - starts):.3f} seconds.")
    print(f"Total trajectories: {NUM_EPISODES * B}")
    env.close()