#!/usr/bin/env python3
"""Section 5.3 — Processus de génération (Flow Matching) dans la scène réelle.

Visualise le « débruitage » du Flow Matching : l'intégration de l'EDO qui
transporte une trajectoire de bruit gaussien pur (t=0) vers la trajectoire
nominale finale (t=1), superposée à la SCÈNE CAPTÉE (nuage de points caméra :
aliment + assiette) ET au ROBOT Franka rendu par cinématique directe à partir
des angles articulaires enregistrés (green_cube_feeding_casf.launch).

Tout vit dans le repère centré sur la pointe de la fourchette (translation
seule, orientation monde préservée) : le nuage Merged_Fork est déjà dans ce
repère, et la scène monde + le robot monde y sont ramenés en soustrayant la
position monde de la pointe — exactement le centrage du Data_Loader. La pointe
FK retombe sur la pointe enregistrée (base panda_link0 à l'origine, non tournée),
donc l'alignement robot/scène/trajectoire est exact.

Produit UNE figure 2x3 (plans XZ et YZ, trois étapes du flux) au style
thesis_style partagé :
  fm_denoising_scene et fm_denoising_scene_<trajectoire>_<état>
      — t=0 (bruit) -> ... -> t=1 (trajectoire générée).

Lancer avec le python du venv depuis ce dossier :
    python3 generate_fm_denoising_figure.py
Figures écrites dans ./thesis_fm_figures/ en .svg (thèse) + .png (aperçu).
"""
import os
import sys
import json
import tempfile

import numpy as np
import torch

import rospkg

# --- shared thesis style (same module as the SDF/CBF/FM figures) -------------
_PKG = rospkg.RosPack().get_path('vision_processing')
sys.path.insert(0, os.path.join(_PKG, 'scripts'))
import thesis_style as ts          # noqa: E402
ts.apply()
import matplotlib.pyplot as plt    # noqa: E402  (after ts.apply so rcParams stick)
from matplotlib.lines import Line2D               # noqa: E402
from matplotlib.ticker import MaxNLocator          # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

# --- 9D dynamics model + data (same modules as generate_fm_thesis_figures) ---
from Train_Fork_FlowMP_9D import FlowMatchingAgent       # noqa: E402
from Data_Loader_Fork_FlowMP_9D import rotation_matrix_to_ortho6d  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thesis_fm_figures')
CKPT_NAME = 'best_fm_model_9D_dynamics_1024.ckpt'

# --- scene selection (self-contained: robot joints + scene + trajectory all
#     come from the SAME recorded timestep) ------------------------------------
TRAJ_ID = 60           # recorded trajectory folder (datas/Trajectories_preprocess)
STATE_IDX = 25        # timestep: fork descending toward the cube (approach phase)
OBS_HORIZON = 2
PRED_HORIZON = 16
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thesis_fm_figures')
VIZ_STEPS = 5         # ODE integration steps (= deployed n_pred budget)
DISPLAY_TIMES = (0.0, 0.5, 1.0)
FM_SEED = 7           # noise seed (matches launch fm_noise_seed default)
ARROW_FRAC = 0.095    # flow arrow length as a fraction of the view extent
VIEW_ZOOM = 1.06      # >1 pads the view box around the content
ROBOT_VIEW_RADIUS = 0.18  # include robot points within this radius (m) of the
                          # fork tip in the view box -> shows the reaching
                          # end-effector without zooming out to the whole arm
SHOW_ROBOT = True
SHOW_SCENE = False    # raw camera cloud (table/plate fragments) — off: too sparse
                      # in this tight box to read as anything but noise
ROBOT_MAX_PTS = 16000  # robot points kept after clipping (rasterized layer)

# colours
ROBOT_GREY = '#8a929c'   # steel
FOOD_GREEN = '#2e8b57'   # the green cube (its real colour)
SCENE_TAN = '#b9a684'    # plate / captured surroundings
GT_COLOR = '#1a1a1a'     # ground truth: dark, distinct from the steel robot
STAR_COLOR = ts.GREY   # fork-tip origin marker (distinct from the green food)
AXES_FACE = '#f2f4f5'  # contrast for white-edged trajectory markers
FOOD_FORK_SEP = 0.012    # m: Merged_Fork points farther than this from the FK
                         # robot are the food target (the rest is the fork)

_ROBOT = {}              # lazy FK-mesh loader cache


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def ortho6d_to_rotation_matrix(d6):
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt (inverse of the loader's encode)."""
    x_raw = d6[..., 0:3]
    y_raw = d6[..., 3:6]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


# ==============================================================================
# Self-contained sample (mirrors Data_Loader_Fork_FlowMP_9D centering)
# ==============================================================================

def _pose9(pos, quat):
    mat = R.from_quat(quat).as_matrix()
    return np.concatenate([pos, rotation_matrix_to_ortho6d(mat)]).astype(np.float32)


def _sample_to(pc, n):
    if pc.shape[0] == n:
        return pc
    idx = np.random.choice(pc.shape[0], n, replace=pc.shape[0] < n)
    return pc[idx]


def build_sample(traj_id, state_idx, num_points=1024,
                 obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON):
    """One recorded planner instant: model conditioning + GT + scene + joints.

    Centering is translation-only on the world fork-tip position (curr_pos), so
    the returned agent_pos/action live in the same fork-tip frame as the stored
    Merged_Fork conditioning cloud, and scene_world/curr_pos let the figure bring
    the captured scene and the FK robot into that frame too."""
    base = os.path.join(_PKG, 'datas', 'Trajectories_preprocess', f'Trajectory_{traj_id}')
    states = json.load(open(os.path.join(base, f'trajectory_{traj_id}.json')))['states']
    assert state_idx - obs_horizon + 1 >= 0 and state_idx + pred_horizon <= len(states), \
        f'state_idx {state_idx} out of range for {len(states)} states'

    def pose_at(i):
        s = states[i]
        return _pose9(np.array(s['fork_tip_position'], dtype=np.float32),
                      s['fork_tip_orientation'])

    curr = np.array(states[state_idx]['fork_tip_position'], dtype=np.float32)
    obs = np.stack([pose_at(i) for i in range(state_idx - obs_horizon + 1, state_idx + 1)])
    act = np.stack([pose_at(i) for i in range(state_idx, state_idx + pred_horizon)])
    obs[:, :3] -= curr
    act[:, :3] -= curr

    st = states[state_idx]
    pcd = np.load(os.path.join(base, f'Merged_Fork_Trajectory_{traj_id}',
                               st['Merged_Fork_point_cloud'])).astype(np.float32)
    pcd = _sample_to(pcd, num_points)
    scene_world = np.load(os.path.join(base, f'Merged_pcd_Trajectory_{traj_id}',
                                       st['Merged_point_cloud'])).astype(np.float32)
    return {
        'obs': {'point_cloud': torch.from_numpy(pcd).float(),
                'agent_pos': torch.from_numpy(obs).float()},
        'action': torch.from_numpy(act).float(),
        'scene_world': scene_world[:, :3],
        'curr_pos': curr,
        'joint_positions': np.array(st['joint_positions'], dtype=np.float64),
    }


def robot_points_fig(joint_positions, curr_pos):
    """Franka surface point cloud (FK from joint angles) in the fork-tip frame.

    Uses the project's RobotMeshLoaderOptimized (full URDF FK + mesh sampling).
    panda_link0 is at the world origin unrotated, so the sampled cloud is in
    world coordinates; subtracting curr_pos (world fork tip) yields the figure
    frame, matching the trajectory exactly."""
    loader = _ROBOT.get('loader')
    if loader is None:
        import xacro
        from Compute_3D_point_cloud_from_mesh import RobotMeshLoaderOptimized
        urdf_xml = xacro.process_file(os.path.join(_PKG, 'urdf', 'panda_camera.xacro')).toxml()
        tf = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
        tf.write(urdf_xml)
        tf.close()
        loader = RobotMeshLoaderOptimized(tf.name)
        _ROBOT['loader'] = loader
    jp = joint_positions
    ja = {f'panda_joint{i + 1}': float(jp[i]) for i in range(7)}
    ja.update({'panda_finger_joint1': 0.04, 'panda_finger_joint2': 0.04})
    rc = loader.create_point_cloud(ja).astype(np.float32)   # world frame
    return rc - curr_pos[None, :]


# ==============================================================================
# Flow Matching ODE integration with the learned velocity field
# ==============================================================================

def integrate_with_intermediate_steps(model, sample, device, num_steps, seed=None):
    """Euler-integrate the FM ODE, returning the state AND the flow at every step.

    frames[k] : [H,9] metric state; frames[0]=noise (t=0), frames[-1]=generated.
    flows[k]  : [H,3] learned velocity v(x_k,t_k) in metric position units;
                flows[-1]=0 (flow terminated at t=1)."""
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    obs = {
        'point_cloud': sample['obs']['point_cloud'].unsqueeze(0).to(device),
        'agent_pos':   model.normalizer.normalize_obs(
            sample['obs']['agent_pos'].unsqueeze(0).to(device)),
    }
    half_range = ((model.normalizer.act_max[:3] - model.normalizer.act_min[:3])
                  / 2.0).cpu().numpy()
    B = 1
    x = torch.randn(B, model.pred_horizon, model.action_dim, device=device)
    dt = 1.0 / num_steps
    frames = [model.normalizer.unnormalize_act(x).cpu().numpy()[0]]
    flows = []
    with torch.no_grad():
        for i in range(num_steps):
            t = torch.ones(B, device=device) * (i / num_steps)
            v = model.forward(obs, x, t)
            flows.append(v.cpu().numpy()[0, :, :3] * half_range)
            x = x + v * dt
            frames.append(model.normalizer.unnormalize_act(x).cpu().numpy()[0])
    flows.append(np.zeros_like(flows[-1]))
    return frames, flows


# ==============================================================================
# Figure
# ==============================================================================

def fig_denoising_scene(model, sample, device, robot_fig=None,
                        viz_steps=VIZ_STEPS, display_times=DISPLAY_TIMES,
                        seed=FM_SEED):
    print(f"[5.3] processus de génération FM (Trajectory {TRAJ_ID}, état {STATE_IDX}) ...")
    gt = sample['action'].numpy()
    hist = sample['obs']['agent_pos'].numpy()
    scene = sample['scene_world'] - sample['curr_pos'][None, :]    # captured scene -> fig frame

    frames, flows = integrate_with_intermediate_steps(model, sample, device,
                                                      viz_steps, seed=seed)

    # Keep the deployed five-step integration. The requested t=0.5 state lies
    # halfway between the stored Euler states at t=0.4 and t=0.6, so it is
    # obtained by linear interpolation along that integration segment.
    display_frames, display_flows = [], []
    for t in display_times:
        u = float(t) * (len(frames) - 1)
        k0 = int(np.floor(u))
        k1 = min(k0 + 1, len(frames) - 1)
        alpha = u - k0
        display_frames.append((1.0 - alpha) * frames[k0] + alpha * frames[k1])
        display_flows.append((1.0 - alpha) * flows[k0] + alpha * flows[k1])
    n_panels = len(display_times)

    # food target = the Merged_Fork conditioning points that are NOT the fork,
    # i.e. far from the FK robot surface (dense, ~150 pts on the cube). The fork
    # portion of Merged_Fork is already shown by the robot render.
    mf = sample['obs']['point_cloud'].numpy()[:, :3]
    food = mf
    if robot_fig is not None and len(robot_fig):
        from scipy.spatial import cKDTree
        d, _ = cKDTree(robot_fig).query(mf, k=1)
        food = mf[d > FOOD_FORK_SEP]

    # Shared limits contain every displayed trajectory frame (including t=0),
    # the target, and the reaching part of the robot. Per-axis limits use more
    # of the available 2D area than the cubic box required by the former 3D view.
    core = ([gt[:, :3], hist[:, :3], food]
            + [frame[:, :3] for frame in display_frames])
    if robot_fig is not None:
        near = robot_fig[np.linalg.norm(robot_fig, axis=1) < ROBOT_VIEW_RADIUS]
        if len(near):
            core.append(near)
    core = np.concatenate(core, axis=0)
    ctr = (core.min(0) + core.max(0)) / 2.0
    half_span = np.maximum((core.max(0) - core.min(0)) * VIEW_ZOOM / 2.0,
                           0.035)
    lo, hi = ctr - half_span, ctr + half_span

    def clip(p):
        m = np.all((p >= lo) & (p <= hi), axis=1)
        return p[m]

    scene_v = clip(scene)          # captured plate/surroundings (faint context)
    food_v = clip(food)            # dense green cube

    robot_v = None
    if robot_fig is not None:
        robot_v = clip(robot_fig)
        if len(robot_v) > ROBOT_MAX_PTS:
            robot_v = robot_v[np.random.choice(len(robot_v), ROBOT_MAX_PTS, replace=False)]

    projections = [(0, 2, 'X', 'Z'), (1, 2, 'Y', 'Z')]

    fig, axes = plt.subplots(2, n_panels, figsize=(ts.TEXTWIDTH, 3.55),
                             squeeze=False)
    for p, (traj, flow, t) in enumerate(
            zip(display_frames, display_flows, display_times)):
        col = ts.BLUE
        for row, (i, j, x_name, y_name) in enumerate(projections):
            ax = axes[row, p]
            ax.set_facecolor(AXES_FACE)

            # Static scene context, projected on XY or XZ. Dense layers remain
            # rasterized so the thesis SVG stays compact.
            if robot_v is not None and len(robot_v):
                ax.scatter(robot_v[:, i], robot_v[:, j], s=0.25,
                           c=ROBOT_GREY, alpha=0.22, linewidths=0, zorder=0,
                           rasterized=True)
            if SHOW_SCENE and len(scene_v):
                ax.scatter(scene_v[:, i], scene_v[:, j], s=1.2,
                           c=SCENE_TAN, alpha=0.35, linewidths=0, zorder=1,
                           rasterized=True)
            if len(food_v):
                ax.scatter(food_v[:, i], food_v[:, j], s=3.0,
                           c=FOOD_GREEN, alpha=0.95, linewidths=0, zorder=2,
                           rasterized=True)

            ax.plot(gt[:, i], gt[:, j], color=GT_COLOR, lw=1.3,
                    alpha=0.90, zorder=3)
            ax.scatter(hist[:, i], hist[:, j], marker='s', s=20,
                       c=ts.ORANGE, edgecolors='white', linewidths=0.4,
                       zorder=5)

            # Learned FM velocity, projected and normalized to uniform arrow
            # length: the arrows show denoising direction, not magnitude.
            flow_2d = flow[:, [i, j]]
            flow_mag = np.linalg.norm(flow_2d, axis=1)
            valid = np.flatnonzero(flow_mag > 1e-8)
            if len(valid):
                arrow_len = ARROW_FRAC * max(hi[i] - lo[i], hi[j] - lo[j])
                flow_dir = flow_2d[valid] / flow_mag[valid, None] * arrow_len
                ax.quiver(traj[valid, i], traj[valid, j],
                          flow_dir[:, 0], flow_dir[:, 1], color=col,
                          angles='xy', scale_units='xy', scale=1.0,
                          width=0.007, headwidth=3.4, headlength=4.5,
                          headaxislength=3.8, minshaft=1.0,
                          alpha=0.88, zorder=4)

            # As in the original 3D design, waypoints are deliberately not
            # connected: the arrows encode the learned flow direction.
            ax.scatter(traj[:, i], traj[:, j], c=[col], s=14,
                       edgecolors='white', linewidths=0.35, zorder=5)

            ax.scatter([0], [0], marker='*', s=42, c=[STAR_COLOR],
                       edgecolors='white', linewidths=0.35, zorder=6)
            ax.set_xlim(lo[i], hi[i])
            ax.set_ylim(lo[j], hi[j])
            ax.set_aspect('equal', adjustable='datalim')
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(labelsize=6.5, pad=1.5, length=2.5)
            if p == 0:
                ax.set_ylabel(f'{y_name} (m)', fontsize=8, labelpad=4)
            else:
                ax.tick_params(labelleft=False)
            ax.set_xlabel(f'{x_name} (m)', fontsize=8, labelpad=2)

        label = (r'$t=0$ (bruit)' if p == 0 else
                 r'$t=1$ (généré)' if p == n_panels - 1 else fr'$t={t:g}$')
        axes[0, p].set_title(fr'$\mathbf{{({chr(97 + p)})}}$  {label}',
                             fontsize=8.2, pad=4)

    # Same categorical legend as the original 3D figure. FM states and vectors
    # use the fixed semantic model colour (blue) in every time-labelled panel.
    handles = [
        Line2D([], [], ls='none', marker='o', ms=4, mfc=ROBOT_GREY, mec='none'),
        Line2D([], [], ls='none', marker='o', ms=4, mfc=FOOD_GREEN, mec='none'),
        Line2D([], [], ls='none', marker='s', ms=4, mfc=ts.ORANGE,
               mec='white', mew=0.4),
        Line2D([], [], ls='none', marker='*', ms=8, mfc=STAR_COLOR,
               mec='white', mew=0.4),
        Line2D([], [], ls='-', color=GT_COLOR, lw=1.3),
    ]
    labels = ['Robot (Franka)', 'Aliment (cible)', 'Observations',
              'Pointe de la fourchette', 'Vérité terrain']
    if SHOW_SCENE:
        handles.insert(1, Line2D([], [], ls='none', marker='o', ms=4,
                                 mfc=SCENE_TAN, mec='none'))
        labels.insert(1, 'Scène captée')
    fig.legend(handles=handles, labels=labels, loc='lower center',
               bbox_to_anchor=(0.5, 0.86), ncol=len(handles),
               columnspacing=0.9, handletextpad=0.4, borderaxespad=0.0,
               fontsize=7.2)

    fig.subplots_adjust(left=0.09, right=0.995, top=0.75, bottom=0.09,
                        wspace=0.10, hspace=0.28)

    # Keep a stable thesis filename and a state-specific copy for traceability.
    canonical_name = 'fm_denoising_scene'
    figure_name = f'{canonical_name}_{TRAJ_ID}_{STATE_IDX}'
    fig.savefig(os.path.join(OUTDIR, canonical_name + '.svg'))
    fig.savefig(os.path.join(OUTDIR, canonical_name + '.png'))
    ts.save(fig, OUTDIR, figure_name)
    print(f"  -> {os.path.join(OUTDIR, canonical_name)}.svg / .png")
    print(f"  -> {os.path.join(OUTDIR, figure_name)}.svg / .png")


def main():
    seed_everything(42)
    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  | figure -> {OUTDIR}")

    ckpt_path = os.path.join(_PKG, 'models', CKPT_NAME)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    stats = ckpt.get('stats', None)
    model = FlowMatchingAgent(
        obs_dim=9, action_dim=9, obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON,
        encoder_output_dim=64, diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024], kernel_size=5, n_groups=8, stats=stats).to(device)
    weights = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(weights, strict=False)
    model.eval()
    print(f"Weights loaded ({CKPT_NAME}). Normalizer init: "
          f"{bool(model.normalizer.is_initialized)}")

    sample = build_sample(TRAJ_ID, STATE_IDX)
    robot_fig = robot_points_fig(sample['joint_positions'], sample['curr_pos']) \
        if SHOW_ROBOT else None
    fig_denoising_scene(model, sample, device, robot_fig=robot_fig)
    print("Done.")


if __name__ == '__main__':
    main()
