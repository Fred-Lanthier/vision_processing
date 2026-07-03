"""
env_pick_place.py
=================
Environnement PICK-AND-PLACE : aller chercher un cube au sol et le poser sur une
tablette (etagere) situee a la DROITE du robot.

Differences clefs avec l'env generique fourchette :
  * On sous-classe `PickCubeEnv` pour reutiliser son cube DYNAMIQUE et
    prehensible (`self.cube`) -> la prehension est une vraie interaction
    physique avec le gripper (pas un "ghost kinematic").
  * On ajoute une etagere kinematique fixe a droite (y < 0, le robot regarde +x).
  * `place_target` expose la pose de depot (centre du dessus de l'etagere).

Repere : base robot a l'origine, le robot regarde +x ; gauche = +y, droite = -y.
"""

import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation as R

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("PickPlaceShelf-v0", max_episode_steps=400)
class PickPlaceShelfEnv(PickCubeEnv):

    # Geometrie (m), en coordonnees MONDE. ATTENTION : le TableSceneBuilder place
    # la base du Panda a world (-0.615, 0, 0). Au homing l'effecteur est donc a
    # world ~ (0, 0, 0.169) et l'empreinte de la camera EE au sol est ~ (-0.05, -0.035).
    # CUBE_HALF sert de repli ; on prefere self.cube_half_size si disponible.
    CUBE_HALF = 0.02
    # Boite TRES BASSE : 1 cm de haut (demi-hauteur 0.005), posee sur la table.
    # Le depot est donc quasi au sol -> la trajectoire pick->place est un seul arc
    # propre, sans grand lift vertical.
    SHELF_HALF = (0.08, 0.08, 0.005)       # demi-tailles : boite 16x16x1 cm
    SHELF_POS = (-0.10, -0.35, 0.005)      # pose nominale (z fixe, sur la table)
    # Region de spawn ALEATOIRE de l'etagere (x/y par episode, z fixe). Une pose
    # fixe rendait la politique aveugle a tout deplacement de la boite au
    # deploiement (40% du nuage hors distribution -> biais constant). La plage y
    # couvre les positions deja utilisees en deploiement (base-rel -0.28 a -0.50,
    # soit monde ManiSkill -0.28 a -0.50 ; x base-rel 0.465..0.565). Bord de
    # l'etagere a y >= -0.20 : reste loin de la zone de spawn du cube (y >= -0.085).
    SHELF_X_RANGE = (-0.15, -0.05)
    SHELF_Y_RANGE = (-0.50, -0.28)
    # Region de spawn du cube : RECENTREE sous l'effecteur au homing (empreinte
    # camera EE ~ (-0.05, -0.035)) pour qu'il soit toujours visible par la camera
    # poignet sur la frame initiale, tout en variant a chaque episode.
    CUBE_X_RANGE = (-0.10, 0.00)
    CUBE_Y_RANGE = (-0.085, 0.015)

    def _load_scene(self, options: dict):
        super()._load_scene(options)  # construit table + cube dynamique

        mat = sapien.render.RenderMaterial()
        mat.base_color = [0.55, 0.4, 0.25, 1.0]
        mat.roughness = 0.9
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=list(self.SHELF_HALF))
        builder.add_box_visual(half_size=list(self.SHELF_HALF), material=mat)
        self.shelf = builder.build_kinematic(name="shelf")

        self.shelf_top_z = self.SHELF_POS[2] + self.SHELF_HALF[2]

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        b = len(env_idx)
        dev = self.device
        cube_half = getattr(self, "cube_half_size", self.CUBE_HALF)

        # 1) Homing du robot sur le keyframe de repos (gripper pointant vers le bas).
        home = self.agent.keyframes["rest"].qpos
        home_q = torch.tensor(home, device=dev, dtype=torch.float32).unsqueeze(0).repeat(b, 1)
        self.agent.reset(home_q)

        # 2) Etagere a droite, position x/y ALEATOIRE par episode (z fixe).
        sxr, syr = self.SHELF_X_RANGE, self.SHELF_Y_RANGE
        sx = sxr[0] + torch.rand(b, device=dev) * (sxr[1] - sxr[0])
        sy = syr[0] + torch.rand(b, device=dev) * (syr[1] - syr[0])
        shelf_p = torch.zeros((b, 3), device=dev)
        shelf_p[:, 0], shelf_p[:, 1] = sx, sy
        shelf_p[:, 2] = self.SHELF_POS[2]
        shelf_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev).repeat(b, 1)
        self.shelf.set_pose(Pose.create_from_pq(p=shelf_p.contiguous(), q=shelf_q.contiguous()))

        # 3) Cube au sol, position + yaw aleatoires devant le robot.
        xr, yr = self.CUBE_X_RANGE, self.CUBE_Y_RANGE
        cx = xr[0] + torch.rand(b, device=dev) * (xr[1] - xr[0])
        cy = yr[0] + torch.rand(b, device=dev) * (yr[1] - yr[0])
        cube_p = torch.zeros((b, 3), device=dev)
        cube_p[:, 0], cube_p[:, 1] = cx, cy
        cube_p[:, 2] = cube_half

        yaw = (torch.rand(b, device=dev) * 2 * np.pi) - np.pi
        q = R.from_euler("z", yaw.cpu().numpy()).as_quat()  # xyzw
        cube_q = torch.tensor(
            np.concatenate((q[:, 3:4], q[:, 0:3]), axis=1), device=dev, dtype=torch.float32
        )
        self.cube.set_pose(Pose.create_from_pq(p=cube_p.contiguous(), q=cube_q.contiguous()))

        # 4) Cible de depot : centre du dessus de l'etagere + demi-cube
        # (suit la position aleatoire de l'etagere de chaque env).
        self.place_target = torch.stack(
            [sx, sy, torch.full((b,), self.shelf_top_z + cube_half, device=dev)],
            dim=1,
        ).to(dtype=torch.float32)

    # ------------------------------------------------------------------ #
    def get_cube_pose(self):
        return self.cube.pose.p, self.cube.pose.q
