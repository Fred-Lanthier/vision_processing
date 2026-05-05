import gymnasium as gym
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import sapien
import cv2
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.agents.robots.panda import Panda
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig

import mediapipe as mp
import math

# =================================================================
# 1. CUSTOM PANDA FORK AGENT (same as before)
# =================================================================
COLOR_CLOSED = (50, 255, 150)
COLOR_OPEN = (50, 150, 255)
COLOR_TARGET = (0, 255, 255)
COLOR_HUD_BG = (30, 30, 30)
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
            local_p = [0.0, 0.0, z_offset_local]
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

        # --- AJOUT DU MANNEQUIN TEXTURÉ (TRELLIS.2) ---
        mannequin_builder = self.scene.create_actor_builder()
        
        # Le chemin de ton fichier haute résolution
        mannequin_mesh_path = "object/handicap_high_res_2.glb"
        
        # 1. Échelle (Scale)
        mannequin_scale = [1, 1, 1] 

        # 2. Position et Rotation par rapport au robot (qui est généralement à l'origine 0,0,0)
        pos_y_gauche = 0.1 
        pos_z_table = -0.22 # Remplace par la hauteur de la table si elle n'est pas à 0
        pos_x = -0.25
        mannequin_p = [pos_x, pos_y_gauche, pos_z_table]

        mannequin_q_xyzw = R.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()
        mannequin_q_wxyz = [mannequin_q_xyzw[3], mannequin_q_xyzw[0], mannequin_q_xyzw[1], mannequin_q_xyzw[2]]
        
        mannequin_pose = sapien.Pose(p=mannequin_p, q=mannequin_q_wxyz)

        # 3. Chargement SANS MATÉRIAU PERSONNALISÉ pour conserver les textures
        mannequin_builder.add_visual_from_file(
            filename=mannequin_mesh_path, 
            scale=mannequin_scale, 
            # material=mat, <-- TRÈS IMPORTANT : On ne met pas cette ligne !
            pose=mannequin_pose
        )
        
        # Optionnel : Ajouter une collision simple (Boîte) pour que le robot ne passe pas au travers
        # mannequin_builder.add_box_collision(
        #     half_size=[0.25, 0.25, 0.25], 
        #     pose=mannequin_pose
        # )

        # On le construit en tant qu'objet cinématique (fixe) pour qu'il ne tombe pas
        self.mannequin = mannequin_builder.build_kinematic(name="mannequin_patient")

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
        far_away = torch.tensor([[99.0, 99.0, -10.0]],
                                device=self.device).repeat(b, 1)
        self.cube.set_pose(Pose.create_from_pq(p=far_away))

        # Random XY
        rand_x = -0.23 + torch.rand(b, device=self.device) * 0.12
        rand_y = -0.14 + torch.rand(b, device=self.device) * 0.16

        # Random Z (pedestal height)
        random_z_offset = torch.rand(b, device=self.device) * 0.08

        # Random food yaw
        self.target_yaw = np.random.uniform(-np.pi, np.pi)

        if self.target_shape_type == "box":
            hx, hy, _ = self.target_params["half_size"]
            local_axis = np.array([1., 0., 0.]) if hx >= hy else np.array([0., 1., 0.])
        elif self.target_shape_type == "capsule":
            local_axis = np.array([1., 0., 0.])
        else:
            local_axis = np.array([1., 0., 0.])

        c, s = np.cos(self.target_yaw), np.sin(self.target_yaw)
        self.principal_axis_world = np.array([
            c * local_axis[0] - s * local_axis[1],
            s * local_axis[0] + c * local_axis[1],
            0.0,
        ])

        # --- Food pose ---
        food_q = R.from_euler("z", self.target_yaw).as_quat()  # scipy: [x,y,z,w]
        food_q_wxyz = [food_q[3], food_q[0], food_q[1], food_q[2]]

        if self.target_shape_type == "box":
            food_half_z = self.target_params["half_size"][2]
        else:
            food_half_z = self.target_params.get("radius", 0.01)

        food_pos = torch.zeros(b, 3, device=self.device)
        food_pos[:, 0] = rand_x
        food_pos[:, 1] = rand_y
        food_pos[:, 2] = random_z_offset + food_half_z

        food_q_tensor = torch.tensor(
            [food_q_wxyz], device=self.device, dtype=torch.float32
        ).repeat(b, 1)
        self.food_target.set_pose(Pose.create_from_pq(p=food_pos, q=food_q_tensor))

        # --- Pedestal ---
        ped_pos = food_pos.clone()
        ped_pos[:, 2] = random_z_offset - self.pedestal_half_z
        self.pedestal.set_pose(Pose.create_from_pq(p=ped_pos))

    def get_desired_fork_yaw(self):
        """Desired yaw (degrees) so fork tines align with food long axis."""
        return np.degrees(
            np.arctan2(self.principal_axis_world[1], self.principal_axis_world[0])
        )

# =================================================================
# 4. MEDIAPIPE MOUTH DETECTION
# =================================================================

class CameraGeometry:
    def __init__(self, width, height):
        # Paramètres intrinsèques (Matrice K)
        # cx, cy : Centre optique de l'image
        self.cx = width / 2
        self.cy = height / 2
        
        # --- ESTIMATION DES FOCALES (fx, fy) ---
        # Sans calibration (checkerboard), on estime fx et fy.
        # Pour une webcam standard (FOV ~60°), fx est souvent proche de la largeur.
        # NOTE : Sur les capteurs modernes, les pixels sont carrés, donc souvent fx ~= fy.
        # Mais nous les séparons ici pour respecter la formule mathématique.
        
        self.fx = width  # Approximation standard
        self.fy = width  # On assume des pixels carrés. Si non, fy serait différent.
        
        # Constante physique : Ecart moyen entre les coins des yeux (en cm)
        self.REAL_EYE_DIST_CM = 6.4 

    def get_metric_coordinates(self, pixel_point, eye_distance_pixels):
        """
        Convertit un point (u, v) en (X, Y, Z) métrique (cm) en utilisant fx et fy séparément.
        """
        if eye_distance_pixels == 0: return (0, 0, 0)

        # 1. Calcul de Z (Profondeur)
        # On utilise fx car les yeux sont alignés horizontalement
        Z_cm = (self.fx * self.REAL_EYE_DIST_CM) / eye_distance_pixels

        # 2. Récupération des coordonnées pixels (u, v)
        u, v = pixel_point

        # 3. Calcul de X (Horizontal) avec fx
        # Formule : X = (Pixel - CentreX) * Z / fx
        X_cm = (u - self.cx) * Z_cm / self.fx

        # 4. Calcul de Y (Vertical) avec fy
        # Formule : Y = (Pixel - CentreY) * Z / fy
        Y_cm = (v - self.cy) * Z_cm / self.fy

        return (X_cm, Y_cm, Z_cm)


def draw_complete_interface(image, distance, status, face_landmarks, center_point, metric_coords):
    h, w, _ = image.shape
    overlay = image.copy()
    main_color = COLOR_CLOSED if status == "Fermee" else COLOR_OPEN

    if face_landmarks:
        # A. Moulage
        for connection in mp_face_mesh.FACEMESH_LIPS:
            pt1 = face_landmarks.landmark[connection[0]]
            pt2 = face_landmarks.landmark[connection[1]]
            cv2.line(image, (int(pt1.x*w), int(pt1.y*h)), (int(pt2.x*w), int(pt2.y*h)), main_color, 1, cv2.LINE_AA)

        # B. Viseur
        if center_point:
            cx, cy = center_point
            cv2.line(image, (cx-10, cy), (cx+10, cy), COLOR_TARGET, 1)
            cv2.line(image, (cx, cy-10), (cx, cy+10), COLOR_TARGET, 1)

    # --- HUD LATÉRAL ---
    cv2.rectangle(overlay, (20, 20), (280, 280), COLOR_HUD_BG, -1)
    
    cv2.putText(image, "COORD PIXELS (u, v)", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    if center_point:
        cv2.putText(image, f"u: {center_point[0]}  v: {center_point[1]}", (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TARGET, 1)

    # --- AFFICHAGE MÉTRIQUE ---
    cv2.putText(image, "COORD REELLES (X, Y)", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    X, Y, Z = metric_coords
    
    # Affichage clair X et Y
    cv2.putText(image, f"X: {X:.1f} cm", (40, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    cv2.putText(image, f"Y: {Y:.1f} cm", (40, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    
    # Affichage Profondeur
    cv2.putText(image, f"Dist (Z): {Z:.1f} cm", (40, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

    # Statut
    cv2.putText(image, status.upper(), (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, main_color, 2)

    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    return image

# =================================================================
# 4. DATA COLLECTION
# =================================================================

if __name__ == "__main__":

    # env = gym.make(
    #     "FoodSpiking-v1",
    #     num_envs=1,
    #     robot_uids="panda_fork",
    #     obs_mode="rgbd",
    #     control_mode="pd_ee_delta_pose",
    #     render_mode="human",
    # )
    env = gym.make(
        "FoodSpiking-v1",
        num_envs=1,
        robot_uids="panda_fork",
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array", # <-- Modification ici
    )

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # --- COULEURS HUD ---
    COLOR_CLOSED = (50, 255, 150)
    COLOR_OPEN = (50, 150, 255)
    COLOR_TARGET = (0, 255, 255)
    
    # --- INITIALISATION DE LA GÉOMÉTRIE (Basée sur la caméra de ManiSkill) ---
    # Récupérer la résolution de la caméra (par défaut souvent 128x128 ou 256x256 dans ManiSkill)
    # Tu pourras ajuster les focales (fx, fy) si tu connais le FOV de la caméra simulée
    cam_width, cam_height = 256, 256 # Valeurs par défaut typiques, à ajuster si besoin
    geom = CameraGeometry(cam_width, cam_height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec standard pour .mp4
    video_writer = None

    NUM_EPISODES = 1
    all_trajectories = []

    for ep in range(NUM_EPISODES):

        # Reconfigure = rebuild scene with new random shape+size
        obs, _ = env.reset(seed=ep, options=dict(reconfigure=True))

        uw = env.unwrapped
        null_action = np.zeros(7, dtype=np.float32)
        for _ in range(25*4):
            env.step(null_action)
            env.render()

        print(f"\n--- Ep {ep+1}/{NUM_EPISODES} | {uw.target_shape_type} ---")

        trajectory = {
            "shape_type": uw.target_shape_type,
            "target_params": uw.target_params,
            "target_yaw": round(uw.target_yaw, 4),
            "principal_axis": [round(x, 4) for x in uw.principal_axis_world],
            "waypoints": [],
        }

        current_phase = 1
        locked_fork_pos = None
        delivery_target_yaw = None

        DYNAMIC_MOUTH_POS = None

        if uw.target_shape_type == "broccoli" or uw.target_shape_type == "chicken":
            base_yaw = uw.get_desired_fork_yaw() + 90
            if base_yaw > 180: base_yaw -= 360
            if base_yaw < -180: base_yaw += 360
        else:
            base_yaw = uw.get_desired_fork_yaw()

        if -90.0 < base_yaw < 90.0:
            # On le repousse de l'autre côté du cercle
            if base_yaw >= 0:
                desired_approach_yaw = base_yaw - 180.0
            else:
                desired_approach_yaw = base_yaw + 180.0
        else:
            desired_approach_yaw = base_yaw

        desired_approach_pitch = -10

        step_count = 0
        done = False
        hover_reached = False
        hold_steps = 0

        locked_relative_pose = None

        while not done:
            links = {l.name: l for l in uw.agent.robot.get_links()}
            fork_tip = links["fork_tip"]

            # --- CALCUL DE L'ACTION PHYSIQUE ---
            fork_pos = fork_tip.pose.p[0]
            food_pos = uw.food_target.pose.p[0]

            # --- EXTRACTION DES ANGLES ---
            q_wxyz = fork_tip.pose.q[0].cpu().numpy()
            q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
            
            # 1. On crée et on SAUVEGARDE l'objet de rotation dans 'r'
            r = R.from_quat(q_xyzw) 
            
            # 2. On extrait les angles d'Euler à partir de 'r'
            euler = r.as_euler("xyz", degrees=True)
            roll, pitch, yaw = euler

            action = np.zeros(7, dtype=np.float32)
            action[6] = 1.0

            # ---- PHASE 1: APPROACH + YAW ALIGNMENT ----
            if current_phase == 1:
                # 1. On définit la position de survol (10 cm au-dessus de la cible)
                fork_pos_np = fork_pos.cpu().numpy()
                food_pos_np = food_pos.cpu().numpy().copy()

                if uw.target_shape_type in ["broccoli"]:
                    # Ex: S'il faut piquer 3 cm plus loin sur l'axe X de l'objet
                    local_offset = np.array([0.0, 0.015, 0.0]) 
                    
                    # On récupère l'orientation exacte de l'aliment dans l'espace
                    food_q_wxyz = uw.food_target.pose.q[0].cpu().numpy()
                    food_q_xyzw = [food_q_wxyz[1], food_q_wxyz[2], food_q_wxyz[3], food_q_wxyz[0]]
                    food_rot = R.from_quat(food_q_xyzw)
                    
                    # On convertit le décalage local en décalage dans le monde 3D
                    world_offset = food_rot.apply(local_offset)
                    
                    # On applique la correction à notre cible !
                    food_pos_np += world_offset
                
                hover_pos = food_pos_np.copy()
                hover_pos[2] += 0.03  # +10 cm en Z
                
                # On calcule les distances séparément (Plan XY et axe Z)
                xy_dist = np.linalg.norm(hover_pos[:2] - fork_pos_np[:2])
                z_dist = abs(hover_pos[2] - fork_pos_np[2])

                # 2. Gestion continue du Yaw (Orientation)
                yaw_err = desired_approach_yaw - yaw
                if yaw_err > 180: yaw_err -= 360
                if yaw_err < -180: yaw_err += 360
                if abs(yaw_err) > 2.0:
                    action[5] = -np.clip(np.deg2rad(yaw_err) * 0.7, -0.7, 0.7)

                pitch_err = desired_approach_pitch - pitch
                if pitch_err > 180: pitch_err -= 360
                if pitch_err < -180: pitch_err += 360
                print(pitch_err)
                if abs(pitch_err) > 2.0:
                    action[4] = np.clip(np.deg2rad(pitch_err) * 0.7, -0.7, 0.7)
                
                roll_err = 0 - roll
                if roll_err > 180: roll_err -= 360
                if roll_err < -180: roll_err += 360
                if abs(roll_err) > 2.0:
                    action[3] = np.clip(np.deg2rad(roll_err) * 0.7, -0.7, 0.7)

                # 3. Logique de déplacement (Survol puis Plongeon)
                # Tant qu'on n'est pas à 1 cm près du point de survol...
                if not hover_reached:
                    # Calcul de la distance 3D totale vers le point de survol
                    dist_to_hover = np.linalg.norm(hover_pos - fork_pos_np)
                    
                    if dist_to_hover > 0.01: # Tant qu'on n'est pas à 1 cm du but
                        # Calcul du vecteur unitaire (direction pure en diagonale)
                        direction = hover_pos - fork_pos_np
                        direction_normalized = direction / dist_to_hover
                        
                        # Calcul de la vitesse globale (Proportionnelle, max 0.05)
                        speed = min(0.05, dist_to_hover * 5.0)
                        
                        # Application de la vitesse sur le vecteur parfait
                        action[:3] = direction_normalized * speed
                    else:
                        # On active le drapeau de façon permanente pour cet épisode
                        hover_reached = True
                if hover_reached:
                    # =====================================================
                    # CALCUL DE LA CIBLE DE PIQUAGE (Spike Target Z)
                    # =====================================================
                    # 1. On récupère la demi-hauteur (Z) de l'objet selon sa forme
                    if uw.target_shape_type == "box":
                        food_half_z = uw.target_params["half_size"][2]
                    elif uw.target_shape_type in ["sphere", "capsule"]:
                        # Pour la sphère et la capsule couchée, la demi-hauteur est le rayon
                        food_half_z = uw.target_params["radius"]
                    else:
                        food_half_z = 0.01
                    
                    # 2. On calcule le Z de la surface supérieure (Top Z)
                    z_top = food_pos_np[2] + food_half_z
                    
                    # 3. Logique de pénétration théorique
                    max_penetration = 0.015
                    food_thickness = 2.0 * food_half_z
                    theoretical_penetration = min(max_penetration, food_thickness - 0.002)
                    target_z = z_top - theoretical_penetration
                    
                    pedestal_center_z = uw.pedestal.pose.p[0][2].cpu().numpy()
                    pedestal_surface_z = pedestal_center_z + uw.pedestal_half_z
                    
                    # La limite de sécurité (ex: 4 mm au-dessus du piédestal)
                    safe_bottom_z = pedestal_surface_z + 0.00
                    
                    # Si notre cible théorique traverse le piédestal, on la bloque !
                    if target_z < safe_bottom_z:
                        target_z = safe_bottom_z
                        
                    # Recalcul de la pénétration réelle pour l'affichage console
                    actual_penetration = z_top - target_z

                    # =====================================================
                    # EXÉCUTION DU PLONGEON
                    # =====================================================
                    # Tant que la pointe de la fourchette est au-dessus de notre Target Z
                    if fork_pos_np[2] > target_z+0.001:
                        # On crée un point 3D qui vise notre cible de profondeur exacte
                        target_spike_pos = food_pos_np.copy()
                        target_spike_pos[2] = target_z
                        
                        direction = target_spike_pos - fork_pos_np
                        
                        # On limite la vitesse pour un piqué précis
                        action[:3] = np.clip(direction * 3.0, -0.02, 0.02) 
                    else:
                        print(f"    Spiked Z={fork_pos[2].item():.3f} (Penetration: {actual_penetration*100:.1f} cm)")
                        current_phase = 2
                        
                        # On sauvegarde les états pour l'arc de cercle de la Phase 2
                        locked_fork_pos = fork_pos.cpu().numpy().copy()
                        pitch_start = pitch

                        fork_pose_sp = fork_tip.pose
                        food_pose_sp = uw.food_target.pose
                        
                        # On sauvegarde l'offset relatif de la nourriture 
                        # par rapport au bout de la fourchette.
                        locked_relative_pose = fork_pose_sp.inv() * food_pose_sp

                        # Calcul du rayon de l'arc de cercle (Turning Radius)
                        if uw.target_shape_type == "box":
                            R_scoop = max(uw.target_params["half_size"][0], uw.target_params["half_size"][1])
                        elif uw.target_shape_type == "sphere":
                            R_scoop = uw.target_params["radius"]
                        elif uw.target_shape_type == "capsule":
                            R_scoop = uw.target_params["half_length"]
                        else:
                            R_scoop = 0.02
                        R_scoop += 0.015 # Marge de sécurité pour bien dégager le piédestal

            # ---- PHASE 2: WRIST ROTATION ----
            elif current_phase == 2:
                # 1. Calcul de la progression de la rotation (alpha = 0.0 à 1.0)
                # On assume que le poignet va tourner d'environ 90 degrés au total
                angle_diff = abs(pitch - pitch_start)
                alpha = np.clip(angle_diff / 90.0, 0.0, 1.0)
                
                # 2. Définition des axes locaux dans le monde (Avant et Haut)
                rad_yaw = np.deg2rad(yaw)
                forward_xy = np.array([np.cos(rad_yaw), np.sin(rad_yaw), 0.0]) # Direction vers laquelle pointe la fourchette
                up_z = np.array([0.0, 0.0, 1.0])
                
                # 3. ÉQUATION PARAMÉTRIQUE DU QUART DE CERCLE
                # À alpha=0 : Offset nul (On est dans la nourriture)
                # À alpha=1 : On a avancé de R_scoop et on est monté de R_scoop
                # La combinaison (1-cos) et (sin) permet de monter rapidement au début pour dégager, puis d'avancer à la fin.
                target_arc_pos = locked_fork_pos + \
                                 forward_xy * (R_scoop * (1 - np.cos(alpha * np.pi / 2))) + \
                                 up_z * (R_scoop * np.sin(alpha * np.pi / 2))
                
                # 4. Asservissement cartésien sur la cible mobile
                pos_err = target_arc_pos - fork_pos.cpu().numpy()
                action[:3] = np.clip(pos_err * 5.0, -0.05, 0.05)
                
                # 5. Rotation dynamique autour de l'axe LOCAL Y (Pitch)
                local_y_axis = np.array([0.0, 1.0, 0.0])
                world_rotation_axis = r.apply(local_y_axis)
                action[3:6] = world_rotation_axis * (0.25)

                # Condition de sortie (ajuste le 80 si ton horizontale absolue est un peu différente)
                print(pitch)
                if abs(pitch) >= 80:
                    print(f"    Wrist done (pitch={pitch:.1f}, lift={R_scoop:.3f}m)")
                    current_phase = 3
                    delivery_target_yaw = yaw + 90.0
                    if delivery_target_yaw > 180: delivery_target_yaw -= 360
                    if delivery_target_yaw < -180: delivery_target_yaw += 360
            
            # ---- PHASE 3: DELIVER ----
            elif current_phase == 3:
                
                # 1. Mise à jour de la mémoire dynamique SI on voit la bouche
                if mouth_detected_this_frame:
                    if DYNAMIC_MOUTH_POS is None:
                        DYNAMIC_MOUTH_POS = current_mouth_world.copy()
                        print("    Bouche acquise par la caméra ! Passage en asservissement visuel.")
                    else:
                        # Lissage pour éviter les tremblements
                        DYNAMIC_MOUTH_POS = 0.8 * DYNAMIC_MOUTH_POS + 0.2 * current_mouth_world

                # 2. SÉLECTION DE LA CIBLE (Le cœur de la logique hybride)
                if DYNAMIC_MOUTH_POS is not None:
                    # MODE TRACKING : Objectif = Z de la bouche, X de la bouche, Y - 5cm
                    target_fork_pos = DYNAMIC_MOUTH_POS + np.array([0.0, -0.05, 0.0])
                else:
                    # MODE AVEUGLE : On s'approche de la zone de recherche probable
                    # (On conserve x=-0.2 basé sur ton ancien code pour rester cohérent avec le robot)
                    target_fork_pos = np.array([-0.20, 0.30, 0.20])
                
                # 3. Calcul de l'erreur spatiale (Translation)
                dir_to_user = target_fork_pos - fork_pos.cpu().numpy() 
                dist = np.linalg.norm(dir_to_user)

                # 4. Calcul de l'erreur angulaire (Rotation Yaw vers 90°)
                absolute_target_yaw = 90.0 
                yaw_err = absolute_target_yaw - yaw
                if yaw_err > 180: yaw_err -= 360
                if yaw_err < -180: yaw_err += 360

                # =========================================================
                # EXÉCUTION DU CONTRÔLE (Couplage Vitesse / Orientation)
                # =========================================================
                speed_factor = np.clip(1.0 - (abs(yaw_err) / 60.0), 0.1, 1.0)
                max_trans_speed = 0.075 * speed_factor

                # --- Asservissement en position ---
                if dist > 0.01:
                    action[:3] = np.clip(dir_to_user * 3.0 * speed_factor, -max_trans_speed, max_trans_speed)
                else:
                    action[:3] = 0.0 # Frein absolu

                # --- Asservissement en orientation ---
                if abs(yaw_err) > 1.5:
                    action[5] = -np.clip(np.deg2rad(yaw_err) * 0.7, -0.5, 0.5)
                else:
                    action[5] = 0.0 # Frein absolu

                # 5. Phase de Stabilisation (Hold)
                if dist <= 0.01 and abs(yaw_err) <= 1.5:
                    hold_steps += 1
                    
                    if hold_steps == 1:
                        if DYNAMIC_MOUTH_POS is None:
                            print("    Target atteinte en mode aveugle (Bouche non trouvée).")
                        else:
                            print("    Target visuelle atteinte ! Stabilisation...")
                        
                    if hold_steps > 15: 
                        print("    Delivered successfully!")
                        done = True
                        continue

            trajectory["waypoints"].append({
                "pos": fork_pos.cpu().numpy().tolist(),
                "quat_wxyz": q_wxyz.tolist(),
                "phase": current_phase,
            })
            if current_phase >= 2 and locked_relative_pose is not None:
                # On calcule la nouvelle position absolue de l'aliment
                new_food_pose = fork_tip.pose * locked_relative_pose
                
                # On téléporte l'aliment (Zéro vibration car zéro gravité !)
                uw.food_target.set_pose(new_food_pose)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            # --- 1. RÉCUPÉRATION DE L'IMAGE ---
            sensors = obs["sensor_data"]
            rgb_ee_array = sensors["ee"]["rgb"][0].cpu().numpy()
            
            # --- 2. TRAITEMENT MEDIAPIPE (Sur l'image RGB native) ---
            results = face_mesh.process(rgb_ee_array)
            
            # Conversion en BGR pour l'affichage avec OpenCV
            img_bgr = cv2.cvtColor(rgb_ee_array, cv2.COLOR_RGB2BGR)
            h, w, _ = img_bgr.shape
            
            # Si c'est la première frame, on met à jour la géométrie avec la vraie taille
            if geom.cx != w/2:
                geom = CameraGeometry(w, h)

            status = "Recherche..."
            center_px = None
            metric_coords = (0, 0, 0)
            
            # --- 3. ANALYSE DES RÉSULTATS MEDIAPIPE ET ASSERVISSEMENT ---
            mouth_detected_this_frame = False
            current_mouth_world = None

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0]
                
                # Coordonnées pixel
                top, bottom = lm.landmark[13], lm.landmark[14]
                cx, cy = int((top.x + bottom.x)/2 * w), int((top.y + bottom.y)/2 * h)
                center_px = (cx, cy)
                
                # Distance des yeux
                eye_left, eye_right = lm.landmark[33], lm.landmark[263]
                eye_dist_px = math.sqrt((eye_left.x*w - eye_right.x*w)**2 + (eye_left.y*h - eye_right.y*h)**2)
                
                # Résultats
                metric_coords = geom.get_metric_coordinates(center_px, eye_dist_px)
                mouth_dist = math.sqrt((top.x - bottom.x)**2 + (top.y - bottom.y)**2)
                status = "Ouverte" if mouth_dist > 0.04 else "Fermee"
                
                # =========================================================
                # MAGIE : PROJECTION DANS LE MONDE RÉEL (PBVS)
                # =========================================================
                try:
                    # 1. On récupère la matrice World -> Camera de ManiSkill3 (3x4)
                    extrinsic_3x4 = obs["sensor_param"]["ee"]["extrinsic_cv"][0].cpu().numpy()
                    
                    # 2. On la convertit en matrice homogène carrée (4x4)
                    world2cam = np.eye(4)
                    world2cam[:3, :4] = extrinsic_3x4
                    
                    # 3. Maintenant on peut l'inverser pour faire Camera -> World !
                    cam2world = np.linalg.inv(world2cam) 
                    
                    # 4. On crée le point homogène OpenCV (en mètres !)
                    p_cam = np.array([metric_coords[0]/100.0, metric_coords[1]/100.0, metric_coords[2]/100.0, 1.0])
                    
                    # 5. On multiplie pour obtenir le (X, Y, Z) global du robot
                    p_world = cam2world @ p_cam
                    current_mouth_world = p_world[:3] / p_world[3]
                    
                    mouth_detected_this_frame = True
                except KeyError:
                    # Sécurité si la structure du dictionnaire ManiSkill change
                    print("Attention: Matrice de caméra non trouvée.")
            
           # --- 4. AFFICHAGE DE L'INTERFACE ---
            img_bgr = draw_complete_interface(
                img_bgr, 0, status, 
                results.multi_face_landmarks[0] if results.multi_face_landmarks else None, 
                center_px, metric_coords
            )
            
            # --- ÉCRITURE DANS LE FICHIER MP4 AU LIEU DE L'AFFICHAGE ---
            if video_writer is None:
                # Initialisation dynamique basée sur la taille réelle de l'image
                h_img, w_img, _ = img_bgr.shape
                # 25.0 correspond aux FPS de la vidéo. Tu peux ajuster si ça te semble trop rapide/lent.
                video_writer = cv2.VideoWriter('mouth_tracking_output_3.mp4', fourcc, 25.0, (w_img, h_img))
            
            video_writer.write(img_bgr)
            
            # (Supprime ou commente ces deux lignes)
            # cv2.imshow("Live Feed - Camera EE avec Tracking", img_bgr)
            # cv2.waitKey(1)
            
            step_count += 1

            if truncated:
                print(f"    Timeout ({step_count} steps)")
                done = True

        trajectory["num_steps"] = step_count
        trajectory["completed"] = (current_phase == 3)
        all_trajectories.append(trajectory)
    if video_writer is not None:
        video_writer.release()
    # cv2.destroyAllWindows()
    env.close()

    import json
    with open("food_spiking_trajectories.json", "w") as f:
        json.dump(all_trajectories, f, indent=2)

    completed = sum(1 for t in all_trajectories if t["completed"])
    shapes = {}
    for t in all_trajectories:
        shapes[t["shape_type"]] = shapes.get(t["shape_type"], 0) + 1
    print(f"\nDone! {completed}/{NUM_EPISODES} completed | Shapes: {shapes}")