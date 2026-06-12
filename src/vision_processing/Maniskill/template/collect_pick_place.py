"""
collect_pick_place.py
=====================
Collecte de demonstrations PICK-AND-PLACE : le Panda va chercher un cube au sol
et le pose sur une etagere a sa droite. Deux cameras enregistrees :

  * "ee"   : montee sur le poignet (panda_hand), regarde l'axe de prehension.
  * "left" : montee sur la base FIXE (panda_link0), placee a GAUCHE du robot,
             inclinee ~45 deg vers le bas pour voir le robot ET l'etagere.

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

# Pose de homing (gripper pointant vers le bas) calculee par IK pour placer le
# TCP a 0.50 m au-dessus de la table (world z = 0.50, base-rel (0.615, 0, 0.50)).
PANDA_HOME_QPOS = [0.0, 0.198794, 0.0, -1.368571, 0.0, 1.567364, 0.785399, 0.04, 0.04]


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

    # --- Camera statique a GAUCHE (+y), plus EN AVANT (grand x) pour voir le robot --
    # panda_link0 est fixe et aligne au repere base (z vers le haut), donc local_*
    # est directement exprime dans le repere monde de la base.
    # NB: coordonnees dans le repere de panda_link0 (= base, world (-0.615,0,0)).
    cam_pos = (0.90, 0.65, 0.60)          # devant (grand x), a gauche, en hauteur
    look_target = (0.40, -0.12, 0.10)     # centre de la scene (robot + cube + etagere)
    left_cam = CameraSpec(
        uid="left",
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


def main():
    num_envs = 4
    num_episodes = 5
    base_record_dir = os.path.join(VP_PATH, "datas", "PickPlace_record_TEST")

    spec = make_robot_spec()
    build_agent(spec)

    env = gym.make(
        "PickPlaceShelf-v0",
        num_envs=num_envs,
        robot_uids=spec.uid,
        obs_mode="rgbd",
        control_mode=spec.control_mode,
    )

    recorder = TrajectoryRecorder(
        base_record_dir=base_record_dir,
        record_links=spec.record_links,
        cameras=spec.cameras,
        record_interval=0.1,
    )
    policy = PickPlacePolicy()

    uw = env.unwrapped
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

        step_count = torch.zeros(uw.num_envs, dtype=torch.int32, device=device)
        done = torch.zeros(uw.num_envs, dtype=torch.bool, device=device)
        print(f"\n--- Ep {ep+1}/{num_episodes} ---")

        while not done.all():
            recorder.record_step(env, obs, step_count, done)
            action, pol_done = policy.act(env, obs)
            done = done | pol_done

            step_count[~done] += 1
            obs, _, _, truncated, _ = env.step(action)
            done = done | truncated

        per_env_meta = [{"object_type": "cube", "surface_type": "shelf"}
                        for _ in range(uw.num_envs)]
        recorder.finish_episode(task_name="PickPlaceShelf", per_env_meta=per_env_meta)

    print(f"\nTermine en {time.time() - t0:.1f}s | {num_episodes * num_envs} trajectoires.")
    env.close()


if __name__ == "__main__":
    main()
