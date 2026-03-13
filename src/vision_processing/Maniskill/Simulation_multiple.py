import gymnasium as gym
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import sapien
import cv2
import json
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.agents.robots.panda import Panda
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.wrappers.record import RecordEpisode

# =================================================================
# 1. CUSTOM PANDA FORK AGENT (same as before)
# =================================================================

@register_agent()
class PandaFork(Panda):
    uid = "panda_fork"
    urdf_path = "/home/flanthier/Github/src/vision_processing/urdf/panda_fork.urdf"

    @property
    def _ee_link_name(self):
        return "fork_tip"

    @property
    def _sensor_configs(self):
        return [
            # Caméra Poignet (End-Effector)
            CameraConfig(
                uid="ee",  # Sera appelé 'ee' dans notre dictionnaire d'observations
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=640, height=480, fov=np.deg2rad(42), near=0.01, far=2.0,
                mount=self.robot.links_map["camera_wrist_link"],
            ),
            # Caméra Statique
            CameraConfig(
                uid="static", # Sera appelé 'static'
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=640, height=480, fov=np.deg2rad(42), near=0.01, far=5.0,
                mount=self.robot.links_map["camera_static_link"],
            )
        ]

# =================================================================
# 2. FOOD TARGET CONFIGURATION
# =================================================================
# Each shape type mimics a real food item.
# All sizes in METERS (SI units for SAPIEN).
#
# The "principal axis" = longest dimension in the XY plane.
# The fork tines should align with this when spiking.
#
#   BOX     = tofu, brownie, cheese cube
#   SPHERE  = meatball, grape, cherry tomato
#   CAPSULE = sausage, carrot stick, spring roll

FOOD_BUILD_CONFIGS = {
    "broccoli": {
        "mesh_path": "object/game-ready_free_broccoli.glb",
        "scale": [0.005, 0.005, 0.005],
        "half_z": 0.0015,
        "local_p": [0.0, 0.0, 0.005],
        "local_euler_xyz_deg": [15, 0, 0],
        "color": [0.05, 0.05, 0.05, 1.0],
        "yaw_offset": True,
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
}

# FOOD_BUILD_CONFIGS = {
#     "patate": {
#         "mesh_path": "object/simple_potato.glb",
#         "scale": [0.0075, 0.0075, 0.0075],
#         "half_z": 0.0075,
#         "local_p": [0.0, 0.0, 0.0],
#         "local_euler_xyz_deg": [0, 0, 0],
#         "color": [0.15, 0.1, 0.05, 1.0],
#         "yaw_offset": False,
#     },
#     "chicken": {
#         "mesh_path": "object/chicken_breast_grilled.glb",
#         "scale": [0.05, 0.025, 0.025],
#         "half_z": 0.0025,
#         "local_p": [0.0, 0.0, 0.0],
#         "local_euler_xyz_deg": [90, 0, 0],
#         "color": [0.25, 0.2, 0.15, 1.0],
#         "yaw_offset": True,
#     },
#     "broccoli": {
#         "mesh_path": "object/game-ready_free_broccoli.glb",
#         "scale": [0.005, 0.005, 0.005],
#         "half_z": 0.0015,
#         "local_p": [0.0, 0.0, 0.005],
#         "local_euler_xyz_deg": [15, 0, 0],
#         "color": [0.05, 0.05, 0.05, 1.0],
#         "yaw_offset": True,
#     },
#     "sushi": {
#         "mesh_path": "object/sushi.glb",
#         "scale": [0.4, 0.4, 0.4],
#         "half_z": 0.00,
#         "local_p": [0.0, 0.0, 0.0],
#         "local_euler_xyz_deg": [90, 0, 0],
#         "color": [0.05, 0.05, 0.05, 1.0],
#         "yaw_offset": False,
#     },
#     "sushi_roll": {
#         "mesh_path": "object/sushi_roll.glb",
#         "scale": [0.4, 0.4, 0.4],
#         "half_z": 0.0075,
#         "local_p": [0.0, 0.0, 0.0],
#         "local_euler_xyz_deg": [90, 0, 0],
#         "color": [0.05, 0.05, 0.05, 1.0],
#         "yaw_offset": False,
#     },
#     "sushi_cube": {
#         "mesh_path": "object/salmon_sushi_-_free_giveaway.glb",
#         "scale": [0.03, 0.03, 0.03],
#         "half_z": 0.0135,
#         "local_p": [0.0, 0.0, 0.0],
#         "local_euler_xyz_deg": [90, 0, 0],
#         "color": [0.05, 0.05, 0.05, 1.0],
#         "yaw_offset": False,
#     },
#     "sausage": {
#         "mesh_path": "object/low_poly_sausage.glb",
#         "scale": [0.01, 0.01, 0.01],
#         "half_z": 0.0075,
#         "local_p": [0.0, -0.016, 0.0],
#         "local_euler_xyz_deg": [0, -50, 0],
#         "color": [0.05, 0.05, 0.05, 1.0],
#         "yaw_offset": False,
#     },
# }

ALL_FOOD_TYPES = list(FOOD_BUILD_CONFIGS.keys())


# =================================================================
# 3. ENVIRONMENT
# =================================================================

@register_env("FoodSpiking-v1", max_episode_steps=1000)
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

        # --- Création du piédestal ---
        ped_builder = self.scene.create_actor_builder()
        
        # 1. On réduit drastiquement la hauteur ! (8 cm au total au lieu de 40 cm)
        self.pedestal_half_z = 0.04 
        
        ped_builder.add_box_collision(half_size=[0.05, 0.05, self.pedestal_half_z])
        
        # 2. On utilise l'attribut 'color' natif (plus robuste que le material)
        ped_builder.add_box_visual(
            half_size=[0.05, 0.05, self.pedestal_half_z],
        )
        
        self.pedestal = ped_builder.build_kinematic(name="pedestal")

    # ---------------------------------------------------------
    # EPISODE INIT: random food type per env, place actors
    # ---------------------------------------------------------
    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        b = len(env_idx)

        # 1. Homing du robot
        homing_qpos = torch.tensor(
            [[-0.000059, -0.125928, 0.000117, -2.193312,
              -0.000251, 2.064780, 0.785511, 0.04, 0.04]],
            device=self.device
        ).repeat(b, 1)
        self.agent.reset(homing_qpos)

        # 2. Cacher le cube par défaut de PickCubeEnv
        far_away_p = torch.tensor([[99.0, 99.0, -10.0]], device=self.device).repeat(b, 1)
        self.cube.set_pose(Pose.create_from_pq(p=far_away_p))

        # =========================================================
        # 3. TIRAGE AU SORT DES ALIMENTS (Doit être fait à chaque reset)
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

        # Pré-calcul des masques pour savoir quel env utilise quel aliment
        self.food_masks = {}
        for food_type in ALL_FOOD_TYPES:
            self.food_masks[food_type] = torch.tensor(
                [ft == food_type for ft in self.per_env_food_type],
                dtype=torch.bool, device=self.device
            )

        # =========================================================
        # 4. GÉNÉRATION DES COORDONNÉES DE BASE
        # =========================================================
        rand_x = -0.23 + torch.rand(b, device=self.device) * 0.12
        rand_y = -0.14 + torch.rand(b, device=self.device) * 0.16

        # =========================================================
        # 5. INITIALISATION DU PIÉDESTAL (EN PREMIER)
        # =========================================================
        # 50% de chance d'avoir un piédestal
        self.per_env_has_pedestal = torch.rand(b, device=self.device) > 0.5
        pedestal_heights = 0.04 + torch.rand(b, device=self.device) * 0.06
        
        # Hauteur du sommet (Z) : hauteur du piédestal SI True, SINON 0.0 (la table)
        random_z_offset = torch.where(
            self.per_env_has_pedestal,
            pedestal_heights,
            torch.zeros(b, device=self.device)
        )

        # Création des coordonnées du piédestal en mémoire séparée
        ped_p = torch.zeros((b, 3), device=self.device)
        ped_p[:, 0] = rand_x
        ped_p[:, 1] = rand_y
        ped_p[:, 2] = random_z_offset - self.pedestal_half_z
        
        # Téléportation des piédestals non désirés sous la carte
        ped_p[~self.per_env_has_pedestal] = torch.tensor([99.0, 99.0, -10.0], device=self.device)
        ped_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(b, 1)
        
        # Application immédiate
        self.pedestal.set_pose(Pose.create_from_pq(p=ped_p.contiguous(), q=ped_q.contiguous()))

        # =========================================================
        # 6. INITIALISATION DE LA NOURRITURE (EN SECOND)
        # =========================================================
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

        # Création des coordonnées de la nourriture
        food_pos = torch.zeros((b, 3), device=self.device)
        food_pos[:, 0] = rand_x
        food_pos[:, 1] = rand_y
        food_pos[:, 2] = random_z_offset + self.per_env_food_half_z

        # Placement des acteurs
        far_p = torch.tensor([[99.0, 99.0, -10.0]], device=self.device).repeat(b, 1)
        far_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(b, 1)

        for food_type, actor in self.food_actors.items():
            mask = self.food_masks[food_type]
            new_p = far_p.clone()
            new_q = far_q.clone()
            if mask.any():
                new_p[mask] = food_pos[mask]
                new_q[mask] = food_q_tensor[mask]
                
            actor.set_pose(Pose.create_from_pq(p=new_p.contiguous(), q=new_q.contiguous()))

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
# 4. DATA COLLECTION
# =================================================================

if __name__ == "__main__":

    # =========================================================================
    # 1. INITIALISATION DE L'ENVIRONNEMENT VECTORISÉ (B = 2)
    # =========================================================================
    num_envs = 3
    env = gym.make(
        "FoodSpiking-v1",
        num_envs=num_envs,
        robot_uids="panda_fork",
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        render_mode="human", # Ouvre la fenêtre SAPIEN pour le env 0
    )

    # env = RecordEpisode(
    #     env, 
    #     output_dir="./videos_maniskill", # Dossier de destination
    #     save_trajectory=False,           # Tu fais déjà ton propre JSON
    #     info_on_video=True,              # Ajoute du texte utile sur la vidéo
    #     save_video=True
    # )

    NUM_EPISODES = 50
    all_trajectories = []
    uw = env.unwrapped
    B = uw.num_envs
    device = env.device

    for ep in range(NUM_EPISODES):

        # Reconfigure = rebuild scene
        obs, _ = env.reset(seed=ep)

        # --- Phase 0 : Stabilisation initiale ---
        null_action = torch.zeros((B, 7), dtype=torch.float32, device=device)
        for _ in range(25): # 0.5 secondes (ajuste au besoin)
            env.step(null_action)
            env.render()

        print(f"\n--- Ep {ep+1}/{NUM_EPISODES} | {uw.per_env_food_type} | Batch: {B} ---")

        # --- Initialisation des données par environnement ---
        # On crée une liste de dictionnaires distincte pour chaque robot !
        batch_trajectories = []
        for b_idx in range(B):
            batch_trajectories.append({
                "shape_type": uw.per_env_food_type[b_idx],
                # Note: Dans une version 100% vectorisée, ces paramètres pourraient différer par index. 
                # On assume ici qu'ils sont partagés ou extraits du env 0 pour l'exemple.
                "target_yaw": round(float(uw.target_yaw) if np.isscalar(uw.target_yaw) else float(uw.target_yaw[b_idx]), 4),
                "waypoints": [],
                "completed": False,
                "num_steps": 0
            })

        # =========================================================================
        # 2. DÉCLARATION DES TENSEURS D'ÉTAT (Le cœur de la vectorisation)
        # =========================================================================
        current_phase = torch.ones(B, dtype=torch.long, device=device) 
        hover_reached = torch.zeros(B, dtype=torch.bool, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        step_count = torch.zeros(B, dtype=torch.int32, device=device)
        hold_steps = torch.zeros(B, dtype=torch.int32, device=device)

        # Variables sauvegardées lors des transitions de phase
        locked_fork_pos = torch.zeros((B, 3), dtype=torch.float32, device=device)
        pitch_start = torch.zeros(B, dtype=torch.float32, device=device)
        
        # Le fantôme cinématique (Stick)
        locked_relative_p = torch.zeros((B, 3), dtype=torch.float32, device=device)
        locked_relative_q = torch.zeros((B, 4), dtype=torch.float32, device=device)
        locked_relative_q[:, 0] = 1.0 # Le W à 1.0 !
        has_locked_pose = torch.zeros(B, dtype=torch.bool, device=device)

        # Cibles 
        # (Si USER_MOUTH_POS est statique, on l'étend en tenseur [B, 3])
        USER_MOUTH_POS = torch.tensor([-0.2, 0.3, 0.16], dtype=torch.float32, device=device).unsqueeze(0).expand(B, 3)

        # --- Paramètres géométriques ---
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
        
        # Le pitch reste constant pour tout le monde
        desired_approach_pitch = -10.0

        # R_scoop partagé (Rayon du quart de cercle)
        R_scoop = 0.035 # Valeur de sécurité, tu peux la lier à uw.target_params ici

        # =========================================================================
        # 3. BOUCLE DE SIMULATION VECTORISÉE
        # =========================================================================
        while not done.all():
            links = {l.name: l for l in uw.agent.robot.get_links()}
            fork_tip = links["fork_tip"]

            # --- Extraction par lot (Batch Extraction) ---
            fork_pose = fork_tip.pose    # Pose object [B
            
            fork_pos = fork_pose.p # [B, 3]
            
            food_pos, food_q = uw.get_active_food_poses()
            # Extraction des angles avec SciPy vectorisé
            q_wxyz = fork_pose.q.cpu().numpy() # [B, 4]
            q_xyzw = np.concatenate((q_wxyz[:, 1:], q_wxyz[:, 0:1]), axis=1)
            r = R.from_quat(q_xyzw) 
            euler = r.as_euler("xyz", degrees=True) # [B, 3]
            
            roll = torch.tensor(euler[:, 0], dtype=torch.float32, device=device)
            pitch = torch.tensor(euler[:, 1], dtype=torch.float32, device=device)
            yaw = torch.tensor(euler[:, 2], dtype=torch.float32, device=device)

            # Tenseur d'action global de ce step
            action = torch.zeros((B, 7), dtype=torch.float32, device=device)
            action[:, 6] = 1.0 # Force de préhension (ouverte/fermée)

            # Création des masques primaires (On ignore les envs qui ont fini)
            active_mask = ~done
            phase1_mask = (current_phase == 1) & active_mask
            phase2_mask = (current_phase == 2) & active_mask
            phase3_mask = (current_phase == 3) & active_mask

            # ---------------------------------------------------------
            # PHASE 1: APPROACH + YAW ALIGNMENT
            # ---------------------------------------------------------
            if phase1_mask.any():
                
                # =======================================================
                # 1. CALCUL DU VRAI CENTRE CIBLE (LA "TÊTE" DU BROCOLI)
                # =======================================================
                local_offsets_np = np.zeros((B, 3), dtype=np.float32)
                for i in range(B):
                    ft = uw.per_env_food_type[i]
                    if ft in ["broccoli", "chicken"]:
                        # L'offset que tu avais dans ton ancien script !
                        local_offsets_np[i] = [0.0, 0.015, 0.0] 
                
                # Rotation de cet offset en fonction de l'orientation de chaque aliment
                food_q_wxyz_np = food_q.cpu().numpy()
                food_q_xyzw_np = np.concatenate((food_q_wxyz_np[:, 1:], food_q_wxyz_np[:, 0:1]), axis=1)
                world_offsets_np = R.from_quat(food_q_xyzw_np).apply(local_offsets_np)
                world_offsets = torch.tensor(world_offsets_np, dtype=torch.float32, device=device)
                
                # La VRAIE position cible à piquer !
                true_food_pos = food_pos + world_offsets

                # Le point de survol est basé sur la vraie position
                hover_pos = true_food_pos.clone()
                hover_pos[:, 2] += 0.03 # +3 cm Z
                
                # --- Asservissement Angulaire Vectorisé ---
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
                
                # --- Déplacement (Hover) ---
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

                # =======================================================
                # 2. PLONGEON (SPIKE) AVEC LES VRAIES DONNÉES Z
                # =======================================================
                spiking_mask = phase1_mask & hover_reached
                if spiking_mask.any():
                    food_half_z = uw.per_env_food_half_z 
                    z_top = true_food_pos[:, 2] + food_half_z
                    
                    theoretical_penetration = torch.clamp(2.0 * food_half_z - 0.002, max=0.015)
                    target_z = z_top - theoretical_penetration
                    
                    # --- NOUVEAU : Garde-fou intelligent (Piédestal vs Table) ---
                    pedestal_surface_z = uw.pedestal.pose.p[:, 2] + uw.pedestal_half_z
                    
                    # On empêche la fourchette de traverser le piédestal OU la table (Z=0)
                    safe_bottom_z = torch.max(pedestal_surface_z + 0.004, torch.tensor(0.004, device=device))
                    target_z = torch.max(target_z, safe_bottom_z)

                    above_mask = spiking_mask & (fork_pos[:, 2] > target_z + 0.001)
                    if above_mask.any():
                        target_spike_pos = true_food_pos[above_mask].clone()
                        target_spike_pos[:, 2] = target_z[above_mask]
                        direction = target_spike_pos - fork_pos[above_mask]
                        action[above_mask, :3] = torch.clamp(direction * 3.0, min=-0.02, max=0.02)
                    
                    # Transition vers Phase 2
                    spiked_mask = spiking_mask & ~above_mask
                    if spiked_mask.any():
                        idx = spiked_mask.nonzero(as_tuple=True)[0]
                        current_phase[idx] = 2
                        
                        locked_fork_pos[idx] = fork_pos[idx].clone()
                        pitch_start[idx] = pitch[idx].clone()
                        
                        # --- ISOLATION MATHÉMATIQUE ---
                        f_p_spike = fork_pos[idx].clone()
                        f_q_spike = fork_pose.q[idx].clone()
                        
                        # TRÈS IMPORTANT : On lie la fourchette à l'origine réelle (food_pos) 
                        # pour que le modèle 3D entier suive correctement la fourchette !
                        fd_p_spike = food_pos[idx].clone() 
                        fd_q_spike = food_q[idx].clone()
                        
                        f_pose_iso = Pose.create_from_pq(p=f_p_spike, q=f_q_spike)
                        fd_pose_iso = Pose.create_from_pq(p=fd_p_spike, q=fd_q_spike)
                        
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

                # Transition vers Phase 3
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

                # --- LE RETOUR DU WRAP (CHEMIN LE PLUS COURT) ---
                yaw_err = absolute_target_yaw - yaw
                
                # Ces deux lignes autorisent le robot à traverser la frontière -180/180 
                # si c'est le chemin le plus direct et le moins contraignant pour son poignet.
                yaw_err = torch.where(yaw_err > 180, yaw_err - 360, yaw_err)
                yaw_err = torch.where(yaw_err < -180, yaw_err + 360, yaw_err)

                # Facteur de vitesse (l'erreur maximale possible est maintenant de 180°)
                speed_factor = torch.clamp(1.0 - (torch.abs(yaw_err) / 180.0), 0.1, 1.0)
                max_trans_speed = 0.075 * speed_factor

                # Translation
                trans_mask = phase3_mask & (dist > 0.01)
                if trans_mask.any():
                    action_trans = dir_to_user[trans_mask] * 3.0 * speed_factor[trans_mask].unsqueeze(1)
                    max_spd = max_trans_speed[trans_mask].unsqueeze(1)
                    action[trans_mask, :3] = torch.max(torch.min(action_trans, max_spd), -max_spd)

                # Rotation
                rot_mask = phase3_mask & (torch.abs(yaw_err) > 1.5)
                if rot_mask.any():
                    # On applique l'action de rotation avec le nouveau calcul
                    action[rot_mask, 5] = -torch.clamp(torch.deg2rad(yaw_err[rot_mask]) * 0.7, -0.5, 0.5)

                # Stabilisation et fin
                hold_mask = phase3_mask & (dist <= 0.01) & (torch.abs(yaw_err) <= 1.5)
                if hold_mask.any():
                    hold_steps[hold_mask] += 1
                    done_mask = hold_mask & (hold_steps > 15)
                    done[done_mask] = True

            # ---------------------------------------------------------
            # ENREGISTREMENT ET GESTION MOTEUR
            # ---------------------------------------------------------
            # 1. On sauvegarde le waypoint uniquement pour les envs non-terminés
            active_indices = (~done).nonzero(as_tuple=True)[0]
            for idx in active_indices.cpu().numpy():
                batch_trajectories[idx]["waypoints"].append({
                    "pos": fork_pos[idx].cpu().numpy().tolist(),
                    "quat_wxyz": fork_pose.q[idx].cpu().numpy().tolist(),
                    "phase": int(current_phase[idx].item()),
                })
                step_count[idx] += 1

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
                if num_envs > 1:    
                    uw.scene._gpu_apply_all()

            obs, reward, terminated, truncated, info = env.step(action)

            if truncated.any():
                done[truncated] = True
            # =========================================================

            # 3. Étape de simulation
            env.render()
            
            # Timeout de sécurité
            if truncated.any():
                done[truncated] = True

            # --- Affichage OpenCV Vectorisé ---
            sensors = obs["sensor_data"]
            # On extrait tout le batch d'un coup : shape [B, 480, 640, 3]
            rgb_ee_batch = sensors["ee"]["rgb"].cpu().numpy() 
            
            # On colle les B images horizontalement (axis=1 correspond à la largeur)
            img_concat = np.concatenate(rgb_ee_batch, axis=1) 
            
            img_bgr = cv2.cvtColor(img_concat, cv2.COLOR_RGB2BGR)
            cv2.imshow(f"Live Feed - {B} Envs", img_bgr)
            cv2.waitKey(1)

        # Fin du While (Tous les envs de ce batch ont fini)
        for idx in range(B):
            batch_trajectories[idx]["num_steps"] = int(step_count[idx].item())
            batch_trajectories[idx]["completed"] = (current_phase[idx].item() == 3)
            all_trajectories.append(batch_trajectories[idx])

    # Fin des épisodes
    cv2.destroyAllWindows()
    env.close()

    with open("food_spiking_trajectories.json", "w") as f:
        json.dump(all_trajectories, f, indent=2)

    completed = sum(1 for t in all_trajectories if t["completed"])
    shapes = {}
    for t in all_trajectories:
        shapes[t["shape_type"]] = shapes.get(t["shape_type"], 0) + 1
    print(f"\nDone! {completed}/{NUM_EPISODES * B} completed | Shapes: {shapes}")