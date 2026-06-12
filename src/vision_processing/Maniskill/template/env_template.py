"""
env_template.py
===============
Environnement de collecte GENERIQUE, agnostique a la tache.

Reprend la structure eprouvee de `FoodSpikingEnv` mais parametree par
`object_configs` (dict[str, ObjectConfig]) et `randomization` (RandomizationConfig)
passes a `gym.make(...)`. La logique :

  * `_load_scene`  : appele UNE fois -> charge tous les acteurs + le piedestal.
  * `_initialize_episode` : appele a chaque reset -> DOMAIN RANDOMIZATION
    (type d'objet, position, yaw, piedestal) de maniere vectorisee (batch).

On sous-classe `PickCubeEnv` pour heriter gratuitement de la table, de la
lumiere et du robot ; le cube par defaut est simplement cache hors champ.
Pour une scene radicalement differente, remplacer la classe de base par une
autre tache ManiSkill ou par `BaseEnv` + `TableSceneBuilder`.
"""

import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation as R

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from scene_config import RandomizationConfig

FAR_P = [99.0, 99.0, -10.0]
FAR_Q = [1.0, 0.0, 0.0, 0.0]


@register_env("GenericCollect-v0", max_episode_steps=750)
class GenericCollectEnv(PickCubeEnv):

    def __init__(self, *args, object_configs=None, randomization=None, **kwargs):
        assert object_configs, "object_configs (dict[str, ObjectConfig]) est requis."
        self.object_configs = object_configs
        self.object_names = list(object_configs.keys())
        self.rand = randomization or RandomizationConfig()

        # Etat par-env rempli a l'initialisation de l'episode.
        self.object_actors = {}
        self.object_masks = {}
        self.per_env_object_type = []
        self.per_env_half_z = None
        self.per_env_has_pedestal = None
        self.principal_axis_world = None
        self.target_yaw = None
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Chargement de la scene (une seule fois)
    # ------------------------------------------------------------------ #
    def _load_scene(self, options: dict):
        super()._load_scene(options)

        self.object_actors = {}
        for name, cfg in self.object_configs.items():
            mat = sapien.render.RenderMaterial()
            mat.base_color = cfg.color
            mat.roughness = 0.8
            mat.metallic = 0.0

            q_xyzw = R.from_euler("xyz", cfg.local_euler_xyz_deg, degrees=True).as_quat()
            local_pose = sapien.Pose(
                p=list(cfg.local_p),
                q=[q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
            )

            builder = self.scene.create_actor_builder()
            if cfg.type == "mesh":
                builder.add_visual_from_file(
                    filename=cfg.mesh_path, scale=cfg.scale,
                    material=mat, pose=local_pose,
                )
            elif cfg.type == "box":
                builder.add_box_visual(half_size=cfg.scale, material=mat, pose=local_pose)
            elif cfg.type == "sphere":
                builder.add_sphere_visual(radius=cfg.scale[0], material=mat, pose=local_pose)
            else:
                raise ValueError(f"Type d'objet inconnu: {cfg.type}")
            self.object_actors[name] = builder.build_kinematic(name=f"obj_{name}")

        # Piedestal (boite kinematique) pour varier la hauteur de la surface.
        self.pedestal_half_z = self.rand.pedestal_half_z
        ped = self.scene.create_actor_builder()
        hs = [self.rand.pedestal_half_xy, self.rand.pedestal_half_xy, self.pedestal_half_z]
        ped.add_box_collision(half_size=hs)
        ped.add_box_visual(half_size=hs)
        self.pedestal = ped.build_kinematic(name="pedestal")

    # ------------------------------------------------------------------ #
    # Initialisation d'episode = DOMAIN RANDOMIZATION (vectorisee)
    # ------------------------------------------------------------------ #
    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        b = len(env_idx)
        dev = self.device

        # 1) Homing du robot sur le keyframe de repos.
        home = self.agent.keyframes["rest"].qpos
        home_q = torch.tensor(home, device=dev, dtype=torch.float32).unsqueeze(0).repeat(b, 1)
        self.agent.reset(home_q)

        # 2) Cacher le cube par defaut de PickCubeEnv.
        far = torch.tensor([FAR_P], device=dev).repeat(b, 1)
        self.cube.set_pose(Pose.create_from_pq(p=far))

        # 3) Tirage du type d'objet par env + masques booleens.
        self.per_env_object_type = np.random.choice(self.object_names, size=b).tolist()
        self.per_env_half_z = torch.tensor(
            [self.object_configs[t].half_z for t in self.per_env_object_type],
            device=dev, dtype=torch.float32,
        )
        self.object_masks = {
            name: torch.tensor(
                [t == name for t in self.per_env_object_type], dtype=torch.bool, device=dev
            )
            for name in self.object_names
        }

        # 4) Position de base aleatoire.
        xr, yr = self.rand.x_range, self.rand.y_range
        rand_x = xr[0] + torch.rand(b, device=dev) * (xr[1] - xr[0])
        rand_y = yr[0] + torch.rand(b, device=dev) * (yr[1] - yr[0])

        # 5) Piedestal aleatoire (en premier, definit la hauteur de surface).
        if self.rand.use_pedestal:
            self.per_env_has_pedestal = torch.rand(b, device=dev) < self.rand.pedestal_prob
            hmin, hmax = self.rand.pedestal_height_range
            heights = hmin + torch.rand(b, device=dev) * (hmax - hmin)
        else:
            self.per_env_has_pedestal = torch.zeros(b, dtype=torch.bool, device=dev)
            heights = torch.zeros(b, device=dev)

        z_off = torch.where(self.per_env_has_pedestal, heights, torch.zeros(b, device=dev))

        ped_p = torch.zeros((b, 3), device=dev)
        ped_p[:, 0], ped_p[:, 1] = rand_x, rand_y
        ped_p[:, 2] = z_off - self.pedestal_half_z
        ped_p[~self.per_env_has_pedestal] = torch.tensor(FAR_P, device=dev)
        ped_q = torch.tensor([FAR_Q], device=dev).repeat(b, 1)
        self.pedestal.set_pose(Pose.create_from_pq(p=ped_p.contiguous(), q=ped_q.contiguous()))

        # 6) Orientation (yaw) + axe principal monde de l'objet.
        if self.rand.random_yaw:
            self.target_yaw = (torch.rand(b, device=dev) * 2 * np.pi) - np.pi
        else:
            self.target_yaw = torch.zeros(b, device=dev)
        yaw_np = self.target_yaw.cpu().numpy()

        q = R.from_euler("z", yaw_np).as_quat()  # xyzw
        obj_q = torch.tensor(
            np.concatenate((q[:, 3:4], q[:, 0:3]), axis=1), device=dev, dtype=torch.float32
        )
        c, s = np.cos(yaw_np), np.sin(yaw_np)
        self.principal_axis_world = torch.tensor(
            np.stack([c, s, np.zeros(b)], axis=-1), device=dev, dtype=torch.float32
        )

        obj_p = torch.zeros((b, 3), device=dev)
        obj_p[:, 0], obj_p[:, 1] = rand_x, rand_y
        obj_p[:, 2] = z_off + self.per_env_half_z

        # 7) Placer l'objet tire, cacher les autres.
        far_p = torch.tensor([FAR_P], device=dev).repeat(b, 1)
        far_q = torch.tensor([FAR_Q], device=dev).repeat(b, 1)
        for name, actor in self.object_actors.items():
            mask = self.object_masks[name]
            new_p, new_q = far_p.clone(), far_q.clone()
            if mask.any():
                new_p[mask] = obj_p[mask]
                new_q[mask] = obj_q[mask]
            actor.set_pose(Pose.create_from_pq(p=new_p.contiguous(), q=new_q.contiguous()))

    # ------------------------------------------------------------------ #
    # Helpers de pose objet (lecture/ecriture via masques)
    # ------------------------------------------------------------------ #
    def get_active_object_poses(self):
        B = self.num_envs
        p = torch.zeros((B, 3), device=self.device)
        q = torch.zeros((B, 4), device=self.device)
        q[:, 0] = 1.0
        for name, actor in self.object_actors.items():
            mask = self.object_masks[name]
            if mask.any():
                pose = actor.pose
                p[mask] = pose.p[mask]
                q[mask] = pose.q[mask]
        return p, q

    def set_active_object_poses(self, p, q):
        B = self.num_envs
        far_p = torch.tensor([FAR_P], device=self.device).expand(B, 3)
        far_q = torch.tensor([FAR_Q], device=self.device).expand(B, 4)
        for name, actor in self.object_actors.items():
            mask = self.object_masks[name]
            new_p, new_q = far_p.clone(), far_q.clone()
            if mask.any():
                new_p[mask] = p[mask]
                new_q[mask] = q[mask]
            actor.set_pose(Pose.create_from_pq(p=new_p.contiguous(), q=new_q.contiguous()))

    def get_desired_yaw_deg(self):
        yaws = torch.atan2(self.principal_axis_world[:, 1], self.principal_axis_world[:, 0])
        return torch.rad2deg(yaws)
