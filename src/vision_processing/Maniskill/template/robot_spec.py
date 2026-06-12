"""
robot_spec.py
=============
Specification declarative d'un robot + outil + capteurs, et construction
dynamique de l'agent ManiSkill3 correspondant.

Idee directrice (issue de la discussion de design) :
  * Le ou les URDF sont la source de verite pour la cinematique et le montage
    des capteurs.
  * On garde une petite config Python pour ce que l'URDF ne peut PAS porter
    dans ManiSkill : mode de controle, keyframe de repos, lien EE controle,
    et la liste explicite des liens dont on enregistre la pose.

On sous-classe `Panda` : il fournit deja `_controller_configs` (dont
`pd_ee_delta_pose`) et le gripper. On ne surcharge que l'URDF, le lien EE, les
capteurs et le keyframe de homing.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sapien

from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda import Panda
from mani_skill.sensors.camera import CameraConfig

from urdf_compose import compose_robot_tool


# --------------------------------------------------------------------------- #
# Dataclasses de specification
# --------------------------------------------------------------------------- #
@dataclass
class ToolSpec:
    """Outil a fusionner dans l'URDF du robot (approche B). None = pas d'outil."""
    urdf_path: str
    parent_link: str                       # lien du robot ou attacher l'outil
    attach_xyz: tuple = (0.0, 0.0, 0.0)     # translation du joint fixe (m)
    attach_rpy: tuple = (0.0, 0.0, 0.0)     # rotation du joint fixe (rad)


@dataclass
class CameraSpec:
    """Camera montee explicitement sur un lien (du robot OU de l'outil fusionne)."""
    uid: str
    mount_link: str
    width: int = 640
    height: int = 480
    fov_deg: float = 42.0
    near: float = 0.01
    far: float = 2.0
    # Pose locale de la camera par rapport au lien de montage (optique souvent identite).
    local_p: tuple = (0.0, 0.0, 0.0)
    local_q_wxyz: tuple = (1.0, 0.0, 0.0, 0.0)


@dataclass
class RobotSpec:
    """Specification complete d'un robot pour la collecte de donnees."""
    uid: str
    robot_urdf_path: str
    home_qpos: list                                  # keyframe de repos (qpos complet)
    tool: Optional[ToolSpec] = None                  # None = robot nu
    cameras: list = field(default_factory=list)      # list[CameraSpec]
    # Liens dont on enregistre la pose (cartesien + quaternion). Remplace le
    # `fork_tip` code en dur du script original.
    record_links: list = field(default_factory=list)
    # Lien EE *controle* par pd_ee_delta_pose (souvent le TCP du robot, pas l'outil).
    control_ee_link: str = "panda_hand_tcp"
    control_mode: str = "pd_ee_delta_pose"

    # Chemin de l'URDF effectivement charge (rempli par build_agent).
    _resolved_urdf_path: Optional[str] = field(default=None, repr=False)


# --------------------------------------------------------------------------- #
# Construction dynamique de l'agent
# --------------------------------------------------------------------------- #
def build_agent(spec: RobotSpec):
    """Construit, enregistre et retourne la classe d'agent ManiSkill pour `spec`.

    - Fusionne robot + outil si `spec.tool` est defini (URDF unique).
    - Genere une sous-classe de `Panda` avec le bon URDF, lien EE, capteurs
      et keyframe de homing.
    - Enregistre l'agent (override=True pour etre re-executable dans un notebook).
    """
    # 1) URDF effectif : fusionne si outil, sinon URDF brut.
    if spec.tool is not None:
        urdf_path = compose_robot_tool(
            robot_urdf_path=spec.robot_urdf_path,
            tool_urdf_path=spec.tool.urdf_path,
            parent_link=spec.tool.parent_link,
            attach_xyz=spec.tool.attach_xyz,
            attach_rpy=spec.tool.attach_rpy,
        )
    else:
        urdf_path = os.path.abspath(spec.robot_urdf_path)
    spec._resolved_urdf_path = urdf_path

    cameras = list(spec.cameras)
    home_qpos = np.asarray(spec.home_qpos, dtype=np.float32)

    def _sensor_configs(self):
        configs = []
        for cam in cameras:
            configs.append(
                CameraConfig(
                    uid=cam.uid,
                    pose=sapien.Pose(p=list(cam.local_p), q=list(cam.local_q_wxyz)),
                    width=cam.width,
                    height=cam.height,
                    fov=np.deg2rad(cam.fov_deg),
                    near=cam.near,
                    far=cam.far,
                    mount=self.robot.links_map[cam.mount_link],
                )
            )
        return configs

    cls = type(
        f"Agent_{spec.uid}",
        (Panda,),
        {
            "uid": spec.uid,
            "urdf_path": urdf_path,
            "ee_link_name": spec.control_ee_link,
            "keyframes": dict(rest=Keyframe(qpos=home_qpos, pose=sapien.Pose())),
            "_sensor_configs": property(_sensor_configs),
        },
    )

    register_agent(override=True)(cls)
    return cls
