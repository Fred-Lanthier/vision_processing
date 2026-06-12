"""
collect.py
==========
Point d'entree : assemble RobotSpec + Env generique + Politique + Recorder pour
collecter un dataset de demonstrations vectorise.

  python -m vision_processing.Maniskill.template.collect

C'est le fichier a copier/adapter pour une nouvelle tache : on y declare le
robot (URDF + outil + cameras + liens enregistres), les objets de la scene, la
randomisation, et on branche une politique.
"""

import os
import time

import gymnasium as gym
import rospkg
import torch

from data_collector import TrajectoryRecorder
from env_template import GenericCollectEnv  # noqa: F401 (enregistre "GenericCollect-v0")
from policy_base import ReachHoldPolicy
from robot_spec import CameraSpec, RobotSpec, ToolSpec, build_agent
from scene_config import EXAMPLE_OBJECT_CONFIGS, RandomizationConfig

VP_PATH = rospkg.RosPack().get_path("vision_processing")


# --------------------------------------------------------------------------- #
# 1) Declaration du robot (exemple : Panda + fourchette fusionnee + 2 cameras)
# --------------------------------------------------------------------------- #
def make_robot_spec() -> RobotSpec:
    # Exemple A : l'outil est DEJA fusionne dans l'URDF -> tool=None.
    return RobotSpec(
        uid="panda_template",
        robot_urdf_path=os.path.join(VP_PATH, "urdf", "panda_fork.urdf"),
        home_qpos=[-0.000059, -0.125928, 0.000117, -2.193312,
                   -0.000251, 2.064780, 0.785511, 0.04, 0.04],
        tool=None,  # voir Exemple B ci-dessous pour fusionner un outil separe
        cameras=[
            CameraSpec(uid="static", mount_link="camera_static_link", far=5.0),
            CameraSpec(uid="ee", mount_link="camera_wrist_link", far=2.0),
        ],
        record_links=["fork_tip", "panda_TCP"],
        control_ee_link="panda_hand_tcp",
        control_mode="pd_ee_delta_pose",
    )
    # Exemple B : robot nu + outil dans un URDF separe (approche B, fusion auto) :
    # return RobotSpec(
    #     uid="panda_custom_tool",
    #     robot_urdf_path=".../panda_v2.urdf",
    #     tool=ToolSpec(urdf_path=".../my_tool.urdf", parent_link="panda_hand",
    #                   attach_xyz=(0, 0, 0.05)),
    #     cameras=[CameraSpec(uid="ee", mount_link="tool_camera_link")],
    #     record_links=["my_tool_tip"],
    #     home_qpos=[...],
    # )


def main():
    num_envs = 3
    num_episodes = 5
    base_record_dir = os.path.join(VP_PATH, "datas", "Template_record_TEST")

    # Construire + enregistrer l'agent a partir de la spec.
    spec = make_robot_spec()
    build_agent(spec)

    env = gym.make(
        "GenericCollect-v0",
        num_envs=num_envs,
        robot_uids=spec.uid,
        obs_mode="rgbd",
        control_mode=spec.control_mode,
        object_configs=EXAMPLE_OBJECT_CONFIGS,
        randomization=RandomizationConfig(),
    )

    recorder = TrajectoryRecorder(
        base_record_dir=base_record_dir,
        record_links=spec.record_links,
        cameras=spec.cameras,
        record_interval=0.1,
    )
    policy = ReachHoldPolicy()

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
        print(f"\n--- Ep {ep+1}/{num_episodes} | objets: {uw.per_env_object_type} ---")

        while not done.all():
            recorder.record_step(env, obs, step_count, done)
            action, pol_done = policy.act(env, obs)
            done = done | pol_done

            step_count[~done] += 1
            obs, _, _, truncated, _ = env.step(action)
            done = done | truncated

        per_env_meta = [
            {
                "object_type": uw.per_env_object_type[b],
                "surface_type": "pedestal" if bool(uw.per_env_has_pedestal[b]) else "table",
            }
            for b in range(uw.num_envs)
        ]
        recorder.finish_episode(task_name="GenericCollect", per_env_meta=per_env_meta)

    print(f"\nTermine en {time.time() - t0:.1f}s | {num_episodes * num_envs} trajectoires.")
    env.close()


if __name__ == "__main__":
    main()
