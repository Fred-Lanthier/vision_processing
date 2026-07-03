"""
collect_pick_place.py
=====================
Collecte de demonstrations PICK-AND-PLACE : le Panda va chercher un cube au sol
et le pose sur une etagere a sa droite. Deux cameras enregistrees :

  * "ee"   : montee sur le poignet (panda_hand), regarde l'axe de prehension.
  * "left" : montee sur la base FIXE (panda_link0), placee EN FACE du robot
             (devant, centree en y) et inclinee vers le bas pour voir le robot,
             le cube ET l'etagere de face. (uid "left" conserve pour l'aval.)

  python -m vision_processing.Maniskill.template.collect_pick_place
"""

import os
import time

import gymnasium as gym
import rospkg
import torch

from mani_skill import PACKAGE_ASSET_DIR

import numpy as np

from cam_utils import look_at_quat_wxyz, rpy_to_quat_wxyz
from data_collector import TrajectoryRecorder
from env_pick_place import PickPlaceShelfEnv  # noqa: F401 (enregistre l'env)
from policy_pick_place import PickPlacePolicy
from robot_spec import CameraSpec, RobotSpec, build_agent

VP_PATH = rospkg.RosPack().get_path("vision_processing")

# Pose de homing (gripper pointant vers le bas) calculee par IK. Le TCP est a
# world z = 0.35 (base-rel (0.615, 0, 0.35)), orientation top-down exacte
# [0,0,-1]. Abaisse de 0.15 m par rapport a l'ancienne pose (z=0.50) : depart
# plus bas, plus proche du cube. Seuls j2/j4/j6 changent (j1=j3=j5=0 -> bras dans
# le plan x-z). Pour remonter/descendre, refaire l'IK planaire sur z.
# Ancienne (z=0.50): [0.0, 0.198794, 0.0, -1.368571, 0.0, 1.567364, 0.785399, ...]
PANDA_HOME_QPOS = [0.0, 0.188029, 0.0, -1.767649, 0.0, 1.955677, 0.785399, 0.04, 0.04]


def make_robot_spec() -> RobotSpec:
    # --- Camera poignet : REPLIQUE exactement la camera wrist de panda_fork.urdf ---
    # Dans le URDF : montee sur panda_TCP (== panda_hand_tcp du Panda standard),
    # origin xyz=(-0.052, 0.035, -0.045) rpy=(0, -pi/2, 0). Son +x (axe optique
    # SAPIEN) pointe alors selon +z du TCP = direction de prehension.
    ee_cam = CameraSpec(
        uid="ee",
        mount_link="panda_hand_tcp",
        local_p=(-0.052, 0.035, -0.045),
        local_q_wxyz=rpy_to_quat_wxyz(0.0, -np.pi / 2, 0.0),
        fov_deg=42.0, far=2.0,
    )

    # --- Camera statique EN FACE du robot (devant, +x ; centree en y) ------------
    # panda_link0 est fixe et aligne au repere base (z vers le haut), donc local_*
    # est directement exprime dans le repere monde de la base.
    # NB: coordonnees dans le repere de panda_link0 (= base, world (-0.615,0,0)).
    # On centre la camera sur la ligne de travail (y ~ -0.12, entre le robot y=0 et
    # l'etagere y=-0.35) et on la fait regarder droit vers -x : vue frontale du
    # robot (axe optique dans le plan x-z), au lieu d'une vue de trois-quarts gauche.
    cam_pos = (0.90, -0.12, 0.60)         # devant (grand x), centre, en hauteur
    look_target = (0.30, -0.12, 0.10)     # robot + cube + etagere, meme y -> vue frontale
    left_cam = CameraSpec(
        uid="left",                       # uid conserve pour ne pas casser l'aval
        mount_link="panda_link0",
        local_p=cam_pos,
        local_q_wxyz=look_at_quat_wxyz(cam_pos, look_target),
        width=640, height=480, fov_deg=55.0, far=5.0,
    )

    return RobotSpec(
        uid="panda_pickplace",
        robot_urdf_path=f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v2.urdf",
        home_qpos=PANDA_HOME_QPOS,
        tool=None,                         # gripper standard, pas d'outil fusionne
        cameras=[ee_cam, left_cam],
        record_links=["panda_hand_tcp"],   # pose EE enregistree (cartesien + quat)
        control_ee_link="panda_hand_tcp",
        control_mode="pd_ee_delta_pose",
    )


def main(num_envs=4, num_episodes=5, arc_mode="parabola", render=False,
         lift_angle_deg=60.0, transport_start_angle_deg=60.0,
         transport_end_angle_deg=-60.0):
    # Nombre TOTAL de trajectoires = num_envs * num_episodes (envs en parallele).
    base_record_dir = os.path.join(VP_PATH, "datas", "PickPlace_record_TEST")

    spec = make_robot_spec()
    build_agent(spec)

    env_kwargs = dict(
        num_envs=num_envs,
        robot_uids=spec.uid,
        obs_mode="rgbd",
        control_mode=spec.control_mode,
    )
    if render:
        env_kwargs["render_mode"] = "human"

    env = gym.make("PickPlaceShelf-v0", **env_kwargs)

    uw = env.unwrapped

    recorder = TrajectoryRecorder(
        base_record_dir=base_record_dir,
        record_links=spec.record_links,
        cameras=spec.cameras,
        record_interval=0.1,
        # Poses GT enregistrees a chaque step -> le preprocess reconstruit leur
        # nuage sans segmentation (pas de SAM). On garde les DEUX objets utiles a
        # la politique : le cube (cible a saisir) ET l'etagere/boite (cible de
        # depot a droite). Sans l'etagere, le modele ne "voit" pas ou poser.
        record_actors={"cube": uw.cube, "shelf": uw.shelf},
    )
    policy = PickPlacePolicy(
        arc_mode=arc_mode,
        departure_angle_min=lift_angle_deg,
        departure_angle_max=lift_angle_deg,
        transport_start_angle_deg=transport_start_angle_deg,
        transport_end_angle_deg=transport_end_angle_deg,
    )
    device = env.device
    t0 = time.time()

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        recorder.start_episode(env)
        policy.reset(env)

        # Stabilisation physique avant enregistrement.
        null = torch.zeros((uw.num_envs, 7), device=device)
        for _ in range(25):
            obs, *_ = env.step(null)
            if render:
                env.render()

        step_count = torch.zeros(uw.num_envs, dtype=torch.int32, device=device)
        done = torch.zeros(uw.num_envs, dtype=torch.bool, device=device)
        print(f"\n--- Ep {ep+1}/{num_episodes} ---")

        while not done.all():
            recorder.record_step(env, obs, step_count, done)
            action, pol_done = policy.act(env, obs)
            done = done | pol_done

            step_count[~done] += 1
            obs, _, _, truncated, _ = env.step(action)
            if render:
                env.render()
            done = done | truncated

        per_env_meta = [{"object_type": "cube", "surface_type": "shelf"}
                        for _ in range(uw.num_envs)]
        recorder.finish_episode(task_name="PickPlaceShelf", per_env_meta=per_env_meta)

    print(f"\nTermine en {time.time() - t0:.1f}s | {num_episodes * num_envs} trajectoires.")
    env.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Collecte PICK-AND-PLACE (trajectoires = envs x episodes).")
    p.add_argument("--num-envs", type=int, default=4, help="envs simules en parallele")
    p.add_argument("--num-episodes", type=int, default=10, help="episodes (resets) par env")
    p.add_argument("--num-trajectories", type=int, default=None,
                   help="raccourci : fixe le TOTAL (lance dans 1 seul env, num_episodes = total)")
    p.add_argument("--arc-mode", choices=["parabola", "circle"], default="parabola",
                   help="forme de l'arc de transport")
    p.add_argument("--render", action="store_true",
                   help="ouvre la fenetre ManiSkill/SAPIEN (utile avec --num-envs 1)")
    p.add_argument("--lift-angle-deg", type=float, default=60.0,
                   help="angle de depart post-grasp par rapport a la table")
    p.add_argument("--transport-start-angle-deg", type=float, default=60.0,
                   help="pente initiale de l'arc de transport par rapport a la table")
    p.add_argument("--transport-end-angle-deg", type=float, default=-60.0,
                   help="pente finale de l'arc de transport par rapport a la table")
    args = p.parse_args()

    if args.num_trajectories is not None:
        main(num_envs=1, num_episodes=args.num_trajectories,
             arc_mode=args.arc_mode, render=args.render,
             lift_angle_deg=args.lift_angle_deg,
             transport_start_angle_deg=args.transport_start_angle_deg,
             transport_end_angle_deg=args.transport_end_angle_deg)
    else:
        main(num_envs=args.num_envs, num_episodes=args.num_episodes,
             arc_mode=args.arc_mode, render=args.render,
             lift_angle_deg=args.lift_angle_deg,
             transport_start_angle_deg=args.transport_start_angle_deg,
             transport_end_angle_deg=args.transport_end_angle_deg)
