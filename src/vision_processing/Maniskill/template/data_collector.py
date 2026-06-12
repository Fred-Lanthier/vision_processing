"""
data_collector.py
=================
Infrastructure d'ENREGISTREMENT reutilisable, agnostique a la tache et au robot.

Reprend la logique du script original (buffers RAM -> flush disque, IDs de
trajectoire reutilisables, conversion de repere SAPIEN -> OpenCV) mais :
  * enregistre la pose de N `record_links` arbitraires (au lieu de `fork_tip`
    code en dur) ;
  * lit les cameras par liste d'uids ;
  * recoit les metadonnees par-env (type d'objet, type de surface...) du caller.

Format de sortie identique a l'original :
  Trajectory_<id>/
    trajectory_<id>.json
    images_Trajectory_<id>/{static,ee}_{rgb,depth}_step_XXXX.{png,npy}
"""

import glob
import json
import os

import numpy as np
import torch
from PIL import Image

from mani_skill.utils.structs.pose import Pose

# Conversion repere camera SAPIEN -> OpenCV (identique au script original).
SAPIEN2OPENCV = torch.tensor(
    [[0.0, 0.0, 1.0, 0.0],
     [-1.0, 0.0, 0.0, 0.0],
     [0.0, -1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=torch.float32,
)


# --------------------------------------------------------------------------- #
# Gestion des IDs / repertoires
# --------------------------------------------------------------------------- #
def get_next_trajectory_id(base_dirs):
    ids = []
    for base in base_dirs:
        for fn in glob.glob(os.path.join(base, "Trajectory_*")):
            try:
                ids.append(int(os.path.basename(fn).replace("Trajectory_", "")))
            except ValueError:
                continue
    for i in range(1, len(ids) + 1):
        if i not in ids:
            return i
    return max(ids) + 1 if ids else 1


def get_next_n_trajectory_ids(base_dirs, n):
    first = get_next_trajectory_id(base_dirs)
    return list(range(first, first + n))


# --------------------------------------------------------------------------- #
# Enregistreur
# --------------------------------------------------------------------------- #
class TrajectoryRecorder:
    def __init__(self, base_record_dir, record_links, cameras,
                 record_interval=0.1, extra_id_dirs=None):
        """
        Args:
            base_record_dir: dossier de sortie des trajectoires.
            record_links:    list[str] liens dont on enregistre la pose (rel. base).
            cameras:         list[CameraSpec] (ou objets avec .uid, .mount_link,
                             .local_p, .local_q_wxyz). Sert a la fois a savoir
                             quelles cles lire dans obs["sensor_data"] et a
                             calculer la pose REELLE de chaque camera
                             (mount_link.pose compose avec l'offset local).
            record_interval: intervalle d'enregistrement en secondes simulees.
            extra_id_dirs:   dossiers additionnels a scanner pour eviter les
                             collisions d'IDs (ex: dossier de preprocess).
        """
        os.makedirs(base_record_dir, exist_ok=True)
        self.base_record_dir = base_record_dir
        self.record_links = record_links
        self.cameras = cameras
        self.camera_uids = [c.uid for c in cameras]
        self.record_interval = record_interval
        self.id_scan_dirs = [base_record_dir] + (extra_id_dirs or [])

    # -- debut d'episode --------------------------------------------------- #
    def start_episode(self, env):
        uw = env.unwrapped
        self.B = uw.num_envs
        self.dev = env.device
        self.steps_per_record = max(1, round(self.record_interval / uw.control_timestep))

        robot = uw.agent.robot
        links = {l.name: l for l in robot.get_links()}
        base_pose = robot.pose

        # Poses des cameras (rel. base), converties SAPIEN -> OpenCV.
        # Pose camera = base_inv * mount_link.pose * offset_local (la vue rendue
        # inclut l'offset local, donc les extrinseques enregistrees aussi).
        s2o = SAPIEN2OPENCV.to(self.dev)
        self.camera_poses_batch = [dict() for _ in range(self.B)]
        for cam in self.cameras:
            if cam.mount_link not in links:
                raise KeyError(
                    f"Lien de montage '{cam.mount_link}' (camera '{cam.uid}') absent "
                    f"du robot. Liens: {sorted(links)}")
            local = Pose.create_from_pq(
                p=torch.tensor(cam.local_p, dtype=torch.float32, device=self.dev),
                q=torch.tensor(cam.local_q_wxyz, dtype=torch.float32, device=self.dev),
            )
            cam_rel = (base_pose.inv() * links[cam.mount_link].pose) * local
            T_cv = torch.matmul(cam_rel.to_transformation_matrix(), s2o)
            for b in range(self.B):
                self.camera_poses_batch[b][f"T_{cam.uid}_s"] = T_cv[b].tolist()

        # IDs + repertoires.
        self.traj_ids = get_next_n_trajectory_ids(self.id_scan_dirs, self.B)
        self.traj_paths, self.img_dirs = [], []
        for tid in self.traj_ids:
            tp = os.path.join(self.base_record_dir, f"Trajectory_{tid}")
            imd = os.path.join(tp, f"images_Trajectory_{tid}")
            os.makedirs(imd, exist_ok=True)
            self.traj_paths.append(tp)
            self.img_dirs.append(imd)

        # Buffers RAM par env.
        self.states = [[] for _ in range(self.B)]
        self.rgb_bufs = {c: [[] for _ in range(self.B)] for c in self.camera_uids}
        self.depth_bufs = {c: [[] for _ in range(self.B)] for c in self.camera_uids}
        self.rec_count = torch.zeros(self.B, dtype=torch.int32, device=self.dev)
        return self.traj_ids

    # -- enregistrement d'un step ----------------------------------------- #
    def record_step(self, env, obs, step_count, done_mask):
        uw = env.unwrapped
        robot = uw.agent.robot
        links = {l.name: l for l in robot.get_links()}
        base_inv = robot.pose.inv()
        sensors = obs["sensor_data"]

        # Poses (rel. base) de chaque lien suivi, pre-calculees pour le batch.
        link_rel = {}
        for ln in self.record_links:
            rel = base_inv * links[ln].pose
            link_rel[ln] = (rel.p, rel.q,
                            links[ln].linear_velocity, links[ln].angular_velocity)

        for b in range(self.B):
            if done_mask[b] or int(step_count[b]) % self.steps_per_record != 0:
                continue
            rec_idx = int(self.rec_count[b])

            for c in self.camera_uids:
                self.rgb_bufs[c][b].append(sensors[c]["rgb"][b].cpu().numpy())
                self.depth_bufs[c][b].append(sensors[c]["depth"][b].cpu().numpy())

            link_states = {}
            for ln, (p, q, lv, av) in link_rel.items():
                link_states[ln] = {
                    "position": p[b].cpu().numpy().tolist(),
                    "orientation": q[b].cpu().numpy().tolist(),
                    "velocity": lv[b].cpu().numpy().tolist() + av[b].cpu().numpy().tolist(),
                }

            self.states[b].append({
                "step_number": rec_idx,
                "simulation_step": int(step_count[b]),
                "time_step": round(int(step_count[b]) * uw.control_timestep, 4),
                "joint_positions": robot.get_qpos()[b].cpu().numpy().tolist(),
                "joint_velocities": robot.get_qvel()[b].cpu().numpy().tolist(),
                "links": link_states,
                "cameras": {
                    c: {"rgb": f"{c}_rgb_step_{rec_idx:04d}.png",
                        "depth_npy": f"{c}_depth_step_{rec_idx:04d}.npy"}
                    for c in self.camera_uids
                },
            })
            self.rec_count[b] += 1

    # -- flush disque ------------------------------------------------------ #
    def finish_episode(self, task_name, per_env_meta):
        """per_env_meta : list[dict] de metadonnees libres par env (object_type...)."""
        for b in range(self.B):
            n = len(self.rgb_bufs[self.camera_uids[0]][b])
            for i in range(n):
                for c in self.camera_uids:
                    Image.fromarray(self.rgb_bufs[c][b][i]).save(
                        os.path.join(self.img_dirs[b], f"{c}_rgb_step_{i:04d}.png"))
                    np.save(
                        os.path.join(self.img_dirs[b], f"{c}_depth_step_{i:04d}.npy"),
                        self.depth_bufs[c][b][i])

            data = {
                "trajectory_id": self.traj_ids[b],
                "current_task": task_name,
                "states": self.states[b],
                "camera_poses": self.camera_poses_batch[b],
                **per_env_meta[b],
            }
            out = os.path.join(self.traj_paths[b], f"trajectory_{self.traj_ids[b]}.json")
            with open(out, "w") as f:
                json.dump(data, f, indent=4)
            print(f"  [recorder] {n} frames -> {out}")
