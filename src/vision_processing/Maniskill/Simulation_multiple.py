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

FOOD_CONFIGS = {
    "box": {
        "half_x": (0.005, 0.020),   # 2-6 cm wide
        "half_y": (0.005, 0.020),   # 2-6 cm deep
        "half_z": (0.005, 0.015),   # 1-3 cm thick
        "color": [0.2, 0.2, 0.2, 1.0],
    },
    "sphere": {
        "radius": (0.008, 0.020),   # 1.6-4 cm diameter
        "color": [0.1, 0.1, 0.1, 1.0],
    },
    "capsule": {
        # SAPIEN capsule: length axis = local X
        "radius":      (0.006, 0.012),  # 1.2-2.4 cm diameter
        "half_length": (0.020, 0.050),  # 4-10 cm long (caps excluded)
        "color": [0.05, 0.05, 0.05, 1.0],
    },
    "patate": {
        "color": [0.15, 0.1, 0.05, 1.0],
    },
    "chicken": {
        "color": [0.25, 0.2, 0.15, 1.0],
    },
    "broccoli": {
        "color": [0.05, 0.05, 0.05, 1.0],
    },
    "sushi": {
        "color": [0.05, 0.05, 0.05, 1.0],
    },
    "sushi_roll": {
        "color": [0.05, 0.05, 0.05, 1.0],
    },
    "sushi_cube": {
        "color": [0.05, 0.05, 0.05, 1.0],
    },
    "sausage": {
        "color": [0.05, 0.05, 0.05, 1.0],
    },
}


# =================================================================
# 3. ENVIRONMENT
# =================================================================

@register_env("FoodSpiking-v1", max_episode_steps=1000)
class FoodSpikingEnv(PickCubeEnv):

    def __init__(self, *args, **kwargs):
        self.target_shape_type = None
        self.target_params = {}
        self.target_yaw = 0.0
        self.principal_axis_world = np.array([1.0, 0.0, 0.0])
        super().__init__(*args, **kwargs)

    # ---------------------------------------------------------
    # SCENE LOADING (called on each reconfiguration)
    # ---------------------------------------------------------
    def _load_scene(self, options: dict):
        # Parent creates the table + default cube
        super()._load_scene(options)

        # Pick a random food shape and size
        # self.target_shape_type = np.random.choice(["patate", "chicken", "broccoli", "sushi", "sushi_roll", "sushi_cube", "sausage"])
        self.target_shape_type = np.random.choice(["broccoli"])
        cfg = FOOD_CONFIGS[self.target_shape_type]
        obj_color = cfg.get("color", [0.2, 0.2, 0.2, 1.0])
        # 1. Création du matériau visuel (SAPIEN 3)
        mat = sapien.render.RenderMaterial()
        mat.base_color = obj_color
        mat.roughness = 0.8  # Rend l'objet un peu mat (moins plastique)
        mat.metallic = 0.0   # La nourriture n'est pas en métal !
        builder = self.scene.create_actor_builder()
        
        if self.target_shape_type == "patate":
            mesh_path = "object/simple_potato.glb"
            scale = [0.0075, 0.0075, 0.0075]
            self.target_params = {"radius": 0.0075}

            z_offset_local = 0.0 
            local_p = [0.0, 0.0, z_offset_local]
            local_q_xyzw = R.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()

        elif self.target_shape_type == "chicken":
            mesh_path = "object/chicken_breast_grilled.glb"
            scale = [0.05, 0.025, 0.025]
            self.target_params = {"radius": 0.0025}

            z_offset_local = 0.0 
            local_p = [0.0, 0.0, z_offset_local]
            local_q_xyzw = R.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()

        elif self.target_shape_type == "broccoli":
            mesh_path = "object/game-ready_free_broccoli.glb"
            scale = [0.005, 0.005, 0.005]
            self.target_params = {"radius": 0.0015}

            z_offset_local = 0.005 
            local_p = [0.0, -0.015, z_offset_local]
            local_q_xyzw = R.from_euler('xyz', [15, 0, 0], degrees=True).as_quat()

        elif self.target_shape_type == "sushi":
            mesh_path = "object/sushi.glb"
            scale = [0.4, 0.4, 0.4]
            self.target_params = {"radius": 0.00}

            z_offset_local = -0.0 
            local_p = [0.0, -0.0, z_offset_local]
            local_q_xyzw = R.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
        
        elif self.target_shape_type == "sushi_roll":
            mesh_path = "object/sushi_roll.glb"
            scale = [0.4, 0.4, 0.4]
            self.target_params = {"radius": 0.0075}

            z_offset_local = -0.0 
            local_p = [0.0, -0.0, z_offset_local]
            local_q_xyzw = R.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
        
        elif self.target_shape_type == "sushi_cube":
            mesh_path = "object/salmon_sushi_-_free_giveaway.glb"
            scale = [0.03, 0.03, 0.03]
            self.target_params = {"radius": 0.0135}

            z_offset_local = -0.0 
            local_p = [0.0, -0.0, z_offset_local]
            local_q_xyzw = R.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
        
        elif self.target_shape_type == "sausage":
            mesh_path = "object/low_poly_sausage.glb"
            scale = [0.01, 0.01, 0.01]
            self.target_params = {"radius": 0.0075}

            z_offset_local = 0.00
            local_p = [0.0, -0.016, z_offset_local]
            local_q_xyzw = R.from_euler('xyz', [0, -50, 0], degrees=True).as_quat()

        local_q_wxyz = [local_q_xyzw[3], local_q_xyzw[0], local_q_xyzw[1], local_q_xyzw[2]]
        local_pose = sapien.Pose(p=local_p, q=local_q_wxyz)
        builder.add_visual_from_file(
            filename=mesh_path, 
            scale=scale, 
            material=mat,
            pose=local_pose
        )
        # self.food_target = builder.build(name="food_target")
        self.food_target = builder.build_kinematic(name="food_target")
        # Pedestal (kinematic = immune to gravity)
        ped_builder = self.scene.create_actor_builder()
        self.pedestal_half_z = 0.20
        ped_builder.add_box_collision(half_size=[0.05, 0.05, self.pedestal_half_z])
        ped_builder.add_box_visual(
            half_size=[0.05, 0.05, self.pedestal_half_z]
        )
        self.pedestal = ped_builder.build_kinematic(name="pedestal")

    # ---------------------------------------------------------
    # EPISODE INIT (called on every reset)
    # ---------------------------------------------------------
    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        b = len(env_idx)

        # Robot homing
        homing_qpos = torch.tensor(
            [[-0.000059, -0.125928, 0.000117, -2.193312,
              -0.000251, 2.064780, 0.785511, 0.04, 0.04]],
            device=self.device
        ).repeat(b, 1)
        self.agent.reset(homing_qpos)

        # Hide the default cube from PickCubeEnv
        far_away = torch.tensor([[99.0, 99.0, -10.0]], device=self.device).repeat(b, 1)
        self.cube.set_pose(Pose.create_from_pq(p=far_away))

        # Random XY
        rand_x = -0.23 + torch.rand(b, device=self.device) * 0.12
        rand_y = -0.14 + torch.rand(b, device=self.device) * 0.16

        # Random Z (pedestal height)
        random_z_offset = torch.rand(b, device=self.device) * 0.08

        # =========================================================
        # GESTION VECTORISÉE DU YAW DE LA NOURRITURE
        # =========================================================
        # 1. Génération des B angles
        self.target_yaw = (torch.rand(b, device=self.device) * 2 * np.pi) - np.pi
        yaw_np = self.target_yaw.cpu().numpy()
        
        # 2. Calcul des quaternions via SciPy
        food_q = R.from_euler("z", yaw_np).as_quat() # Shape: [b, 4] (x,y,z,w)
        
        # 3. Réarrangement (x,y,z,w) -> (w,x,y,z)
        food_q_wxyz = np.concatenate((food_q[:, 3:4], food_q[:, 0:3]), axis=1)
        food_q_tensor = torch.tensor(food_q_wxyz, device=self.device, dtype=torch.float32)

        # 4. Mise à jour de l'axe principal
        local_axis = np.array([1., 0., 0.])
        c, s = np.cos(yaw_np), np.sin(yaw_np)
        
        principal_axis_np = np.stack([
            c * local_axis[0] - s * local_axis[1],
            s * local_axis[0] + c * local_axis[1],
            np.zeros(b)
        ], axis=-1)
        
        self.principal_axis_world = torch.tensor(principal_axis_np, device=self.device, dtype=torch.float32)

        # =========================================================
        # PLACEMENT FINAL DES OBJETS
        # =========================================================
        if self.target_shape_type == "box":
            food_half_z = self.target_params["half_size"][2]
        else:
            food_half_z = self.target_params.get("radius", 0.01)

        # On crée le tenseur des positions
        food_pos = torch.zeros((b, 3), device=self.device)
        food_pos[:, 0] = rand_x
        food_pos[:, 1] = rand_y
        food_pos[:, 2] = random_z_offset + food_half_z

        # On assigne la Pose vectorisée complète (PLUS DE VIEUX CODE ICI !)
        self.food_target.set_pose(Pose.create_from_pq(p=food_pos, q=food_q_tensor))

        # --- Pedestal ---
        ped_pos = food_pos.clone()
        ped_pos[:, 2] = random_z_offset - self.pedestal_half_z
        self.pedestal.set_pose(Pose.create_from_pq(p=ped_pos))

    def get_desired_fork_yaw(self):
        """Desired yaw (degrees) so fork tines align with food long axis."""
        # On utilise torch.atan2 pour calculer l'angle de tous les robots instantanément
        yaws_rad = torch.atan2(self.principal_axis_world[:, 1], self.principal_axis_world[:, 0])
        return torch.rad2deg(yaws_rad)


# =================================================================
# 4. DATA COLLECTION
# =================================================================

if __name__ == "__main__":

    # =========================================================================
    # 1. INITIALISATION DE L'ENVIRONNEMENT VECTORISÉ (B = 2)
    # =========================================================================
    num_envs = 1
    env = gym.make(
        "FoodSpiking-v1",
        num_envs=num_envs,
        robot_uids="panda_fork",
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array", # Ouvre la fenêtre SAPIEN pour le env 0
    )

    env = RecordEpisode(
        env, 
        output_dir="./videos_maniskill", # Dossier de destination
        save_trajectory=False,           # Tu fais déjà ton propre JSON
        info_on_video=True,              # Ajoute du texte utile sur la vidéo
        save_video=True
    )

    NUM_EPISODES = 50
    all_trajectories = []
    uw = env.unwrapped
    B = uw.num_envs
    device = env.device

    for ep in range(NUM_EPISODES):

        # Reconfigure = rebuild scene
        obs, _ = env.reset(seed=ep, options=dict(reconfigure=True))

        # --- Phase 0 : Stabilisation initiale ---
        null_action = torch.zeros((B, 7), dtype=torch.float32, device=device)
        for _ in range(25): # 0.5 secondes (ajuste au besoin)
            env.step(null_action)
            env.render()

        print(f"\n--- Ep {ep+1}/{NUM_EPISODES} | {uw.target_shape_type} | Batch: {B} ---")

        # --- Initialisation des données par environnement ---
        # On crée une liste de dictionnaires distincte pour chaque robot !
        batch_trajectories = []
        for b_idx in range(B):
            batch_trajectories.append({
                "shape_type": uw.target_shape_type,
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
            # Si ta fonction renvoie une liste ou un array numpy, on le convertit
            base_yaw = torch.tensor(base_yaw, dtype=torch.float32, device=device)

        # 2. Ajout de l'offset pour les formes organiques
        if uw.target_shape_type in ["broccoli", "chicken"]:
            base_yaw += 90.0
            # On a SUPPRIMÉ les vieux "if > 180" ici !
            
        # 3. Normalisation stricte entre -180 et 180 pour tous les robots
        # (Cette seule ligne remplace tes anciens if/else et gère les dépassements)
        base_yaw = (base_yaw + 180) % 360 - 180 

        # 4. LOGIQUE DE REPOUSSEMENT (Évite de piquer face à l'utilisateur)
        desired_approach_yaw = base_yaw.clone()

        # Masque principal : On identifie les robots qui font face à l'avant (-90 à 90)
        front_mask = (base_yaw > -90.0) & (base_yaw < 90.0)

        # Sous-masques : Parmi ceux à l'avant, lesquels sont positifs / négatifs ?
        pos_mask = front_mask & (base_yaw >= 0.0)
        neg_mask = front_mask & (base_yaw < 0.0)

        # On applique la correction (-180 ou +180) UNIQUEMENT à ces robots-là
        desired_approach_yaw[pos_mask] -= 180.0
        desired_approach_yaw[neg_mask] += 180.0
        
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
            fork_pose = fork_tip.pose    # Pose object [B]
            food_pose = uw.food_target.pose # Pose object [B]
            
            fork_pos = fork_pose.p # [B, 3]
            food_pos = food_pose.p # [B, 3]

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
                hover_pos = food_pos.clone()
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
                    
                    # Mise à jour du drapeau global pour ceux qui sont arrivés
                    arrived_indices = moving_mask.nonzero(as_tuple=True)[0][arrived]
                    hover_reached[arrived_indices] = True
                    
                    # Calcul de vitesse pour ceux qui bougent encore
                    still_far = ~arrived
                    far_indices = moving_mask.nonzero(as_tuple=True)[0][still_far]
                    if len(far_indices) > 0:
                        dir_norm = direction[still_far] / dist[still_far]
                        speed = torch.clamp(dist[still_far] * 5.0, max=0.05)
                        action[far_indices, :3] = dir_norm * speed

                # --- Plongeon (Spike) ---
                spiking_mask = phase1_mask & hover_reached
                if spiking_mask.any():
                    food_half_z = 0.015 # Assume constant, or extract from batched parameters
                    z_top = food_pos[:, 2] + food_half_z
                    theoretical_penetration = min(0.015, 2.0 * food_half_z - 0.002)
                    target_z = z_top - theoretical_penetration
                    
                    pedestal_surface_z = uw.pedestal.pose.p[:, 2] + uw.pedestal_half_z
                    target_z = torch.max(target_z, pedestal_surface_z + 0.004) # Garde-fou

                    above_mask = spiking_mask & (fork_pos[:, 2] > target_z + 0.001)
                    if above_mask.any():
                        target_spike_pos = food_pos[above_mask].clone()
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
                        # On extrait UNIQUEMENT les fourchettes et aliments qui se touchent
                        f_p_spike = fork_pos[idx].clone()
                        f_q_spike = fork_pose.q[idx].clone()
                        fd_p_spike = food_pos[idx].clone()
                        fd_q_spike = food_pose.q[idx].clone()
                        
                        # On crée des mini-objets Pose (Taille K au lieu de B)
                        f_pose_iso = Pose.create_from_pq(p=f_p_spike, q=f_q_spike)
                        fd_pose_iso = Pose.create_from_pq(p=fd_p_spike, q=fd_q_spike)
                        
                        # Le calcul ne peut plus être corrompu par les autres robots
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

                yaw_err = absolute_target_yaw - yaw
                yaw_err = torch.where(yaw_err > 180, yaw_err - 360, yaw_err)
                yaw_err = torch.where(yaw_err < -180, yaw_err + 360, yaw_err)

                speed_factor = torch.clamp(1.0 - (torch.abs(yaw_err) / 60.0), 0.1, 1.0)
                max_trans_speed = 0.075 * speed_factor

                # Translation
                trans_mask = phase3_mask & (dist > 0.01)
                if trans_mask.any():
                    action_trans = dir_to_user[trans_mask] * 3.0 * speed_factor[trans_mask].unsqueeze(1)
                    max_spd = max_trans_speed[trans_mask].unsqueeze(1)
                    # Astuce PyTorch pour cliper avec un tenseur variable
                    action[trans_mask, :3] = torch.max(torch.min(action_trans, max_spd), -max_spd)

                # Rotation
                rot_mask = phase3_mask & (torch.abs(yaw_err) > 1.5)
                if rot_mask.any():
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

            # 2. Application du Ghost Kinematic (Téléportation Vectorisée)
            if has_locked_pose.any():
                idx = has_locked_pose.nonzero(as_tuple=True)[0]
                
                # 1. On extrait uniquement les robots actifs pour le calcul
                f_p_active = fork_pos[idx].clone()
                f_q_active = fork_pose.q[idx].clone()
                r_p_active = locked_relative_p[idx].clone()
                r_q_active = locked_relative_q[idx].clone()
                
                f_pose_active = Pose.create_from_pq(p=f_p_active, q=f_q_active)
                rel_pose_active = Pose.create_from_pq(p=r_p_active, q=r_q_active)
                
                # 2. Multiplication mathématique protégée
                target_pose_active = f_pose_active * rel_pose_active
                
                # 3. Création de tenseurs globaux 100% vierges
                new_food_p = torch.empty_like(food_pos)
                new_food_q = torch.empty_like(food_pose.q)
                
                # 4. On copie les anciennes positions pour TOUT LE MONDE
                new_food_p.copy_(food_pos)
                new_food_q.copy_(food_pose.q)
                
                # 5. On écrase CHIRURGICALEMENT la position de ceux qui ont piqué
                new_food_p[idx] = target_pose_active.p.clone()
                new_food_q[idx] = target_pose_active.q.clone()
                
                # 6. Envoi au moteur physique avec garantie de mémoire lisse
                uw.food_target.set_pose(Pose.create_from_pq(
                    p=new_food_p.contiguous(), 
                    q=new_food_q.contiguous()
                ))
                if num_envs > 1:
                    uw.scene._gpu_apply_all()
            # =========================================================

            # 3. Étape de simulation
            obs, reward, terminated, truncated, info = env.step(action)
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