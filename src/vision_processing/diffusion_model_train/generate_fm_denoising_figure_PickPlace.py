#!/usr/bin/env python3
"""Processus de generation (Flow Matching) — scene PICK-AND-PLACE.

Variante PICK-AND-PLACE de `generate_fm_denoising_figure.py`. Visualise le
« debruitage » du Flow Matching : l'integration de l'EDO qui transporte une
trajectoire de bruit gaussien pur (t=0) vers la trajectoire nominale (t=1),
superposee a la SCENE.

Difference clef avec la version fourchette : le preprocess pick-and-place ne
produit QUE le nuage `Merged_Fork` (pince + cube + etagere), deja centre sur le
TCP (translation seule, orientation monde gardee). Il n'y a ni nuage monde
`Merged_pcd` separe, ni rendu FK du bras (le rig fourchette panda_camera.xacro
ne s'applique pas). On utilise donc directement `Merged_Fork` comme scene :
  * points proches/au-dessus de l'origine -> pince + cube,
  * points loin en -y (etagere) -> cible de DEPOT, mis en evidence.

Produit UNE figure 2x3 (plans XZ et YZ, t=0 -> 0.5 -> 1) au style these partage.
Sorties dans ./thesis_pickplace_figures/ en .svg + .png (prefixe pp_).

Lancer avec le python du venv depuis ce dossier :
    python3 generate_fm_denoising_figure_PickPlace.py
"""
import os
import sys
import json

import numpy as np
import torch

import rospkg

# --- shared thesis style ------------------------------------------------------
_PKG = rospkg.RosPack().get_path('vision_processing')
sys.path.insert(0, os.path.join(_PKG, 'scripts'))
import thesis_style as ts          # noqa: E402
ts.apply()
import matplotlib.pyplot as plt    # noqa: E402  (after ts.apply so rcParams stick)
from matplotlib.lines import Line2D               # noqa: E402
from matplotlib.ticker import MaxNLocator          # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

# --- 9D model + data (PICK-AND-PLACE) ----------------------------------------
from Train_PickPlace import FlowMatchingAgent       # noqa: E402
from Data_Loader_PickPlace import rotation_matrix_to_ortho6d  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thesis_pickplace_figures')
CKPT_NAME = 'best_fm_model_pickplace_9D_1024.ckpt'

# --- scene selection ----------------------------------------------------------
TRAJ_ID = 1            # recorded trajectory folder (datas/PickAndPlace_preprocess)
STATE_IDX = 30         # timestep (transport phase). Clamped to a valid range in main.
OBS_HORIZON = 2
PRED_HORIZON = 16
VIZ_STEPS = 5          # ODE integration steps (= deployed n_pred budget)
DISPLAY_TIMES = (0.0, 0.5, 1.0)
FM_SEED = 7            # noise seed
ARROW_FRAC = 0.095     # flow arrow length as a fraction of the view extent
VIEW_ZOOM = 1.06       # >1 pads the view box around the content
SHELF_Y_THRESH = -0.20  # Merged_Fork points with y < this are the shelf (place target)

# colours
TOOL_GREY = '#8a929c'   # gripper + cube cluster (near origin)
SHELF_TAN = '#b9a684'   # shelf / place target
GT_COLOR = '#1a1a1a'    # ground truth
STAR_COLOR = ts.GREY    # fork-tip (TCP) origin marker
AXES_FACE = '#f2f4f5'


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ==============================================================================
# Self-contained sample (mirrors Data_Loader_PickPlace centering)
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
    """One recorded planner instant: model conditioning + GT, in the TCP frame.

    Translation-only centering on the world TCP position (curr), so agent_pos /
    action live in the same TCP frame as the stored Merged_Fork conditioning
    cloud (which is already TCP-centered at this step)."""
    base = os.path.join(_PKG, 'datas', 'PickAndPlace_preprocess', f'Trajectory_{traj_id}')
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
    return {
        'obs': {'point_cloud': torch.from_numpy(pcd[:, :3]).float(),
                'agent_pos': torch.from_numpy(obs).float()},
        'action': torch.from_numpy(act).float(),
    }


# ==============================================================================
# Flow Matching ODE integration with the learned velocity field
# ==============================================================================

def integrate_with_intermediate_steps(model, sample, device, num_steps, seed=None):
    """Euler-integrate the FM ODE, returning the state AND the flow at every step.

    frames[k] : [H,9] metric state; frames[0]=noise (t=0), frames[-1]=generated.
    flows[k]  : [H,3] learned velocity in metric position units; flows[-1]=0."""
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

def fig_denoising_scene(model, sample, device,
                        viz_steps=VIZ_STEPS, display_times=DISPLAY_TIMES,
                        seed=FM_SEED):
    print(f"[pp] processus de génération FM (Trajectory {TRAJ_ID}, état {STATE_IDX}) ...")
    gt = sample['action'].numpy()
    hist = sample['obs']['agent_pos'].numpy()

    # Scene = Merged_Fork (deja centre sur le TCP). On separe l'etagere (loin en
    # -y, cible de depot) du reste (pince + cube proches de l'origine).
    mf = sample['obs']['point_cloud'].numpy()[:, :3]
    shelf = mf[mf[:, 1] < SHELF_Y_THRESH]
    toolcube = mf[mf[:, 1] >= SHELF_Y_THRESH]

    frames, flows = integrate_with_intermediate_steps(model, sample, device,
                                                      viz_steps, seed=seed)

    # t=0.5 lies between stored Euler states -> linear interpolation.
    display_frames, display_flows = [], []
    for t in display_times:
        u = float(t) * (len(frames) - 1)
        k0 = int(np.floor(u))
        k1 = min(k0 + 1, len(frames) - 1)
        alpha = u - k0
        display_frames.append((1.0 - alpha) * frames[k0] + alpha * frames[k1])
        display_flows.append((1.0 - alpha) * flows[k0] + alpha * flows[k1])
    n_panels = len(display_times)

    # Shared limits contain trajectory frames, the targets and the tool cluster.
    core = ([gt[:, :3], hist[:, :3], mf]
            + [frame[:, :3] for frame in display_frames])
    core = np.concatenate(core, axis=0)
    ctr = (core.min(0) + core.max(0)) / 2.0
    half_span = np.maximum((core.max(0) - core.min(0)) * VIEW_ZOOM / 2.0, 0.035)
    lo, hi = ctr - half_span, ctr + half_span

    def clip(p):
        if len(p) == 0:
            return p
        m = np.all((p >= lo) & (p <= hi), axis=1)
        return p[m]

    shelf_v = clip(shelf)
    toolcube_v = clip(toolcube)

    projections = [(0, 2, 'X', 'Z'), (1, 2, 'Y', 'Z')]

    fig, axes = plt.subplots(2, n_panels, figsize=(ts.TEXTWIDTH, 3.55),
                             squeeze=False)
    for p, (traj, flow, t) in enumerate(
            zip(display_frames, display_flows, display_times)):
        col = ts.BLUE
        for row, (i, j, x_name, y_name) in enumerate(projections):
            ax = axes[row, p]
            ax.set_facecolor(AXES_FACE)

            if len(toolcube_v):
                ax.scatter(toolcube_v[:, i], toolcube_v[:, j], s=1.5,
                           c=TOOL_GREY, alpha=0.35, linewidths=0, zorder=1,
                           rasterized=True)
            if len(shelf_v):
                ax.scatter(shelf_v[:, i], shelf_v[:, j], s=2.0,
                           c=SHELF_TAN, alpha=0.8, linewidths=0, zorder=2,
                           rasterized=True)

            ax.plot(gt[:, i], gt[:, j], color=GT_COLOR, lw=1.3,
                    alpha=0.90, zorder=3)
            ax.scatter(hist[:, i], hist[:, j], marker='s', s=20,
                       c=ts.ORANGE, edgecolors='white', linewidths=0.4,
                       zorder=5)

            # Learned FM velocity, normalized to uniform arrow length (direction).
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

            ax.scatter(traj[:, i], traj[:, j], c=[col], s=14,
                       edgecolors='white', linewidths=0.35, zorder=5)
            ax.scatter([0], [0], marker='*', s=42, c=[STAR_COLOR],
                       edgecolors='white', linewidths=0.35, zorder=6)
            ax.set_xlim(lo[i], hi[i])
            ax.set_ylim(lo[j], hi[j])
            # adjustable='box' : honore les memes limites sur les 3 colonnes
            # (sinon equal+datalim re-zoome chaque panneau differemment).
            ax.set_aspect('equal', adjustable='box')
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

    handles = [
        Line2D([], [], ls='none', marker='o', ms=4, mfc=TOOL_GREY, mec='none'),
        Line2D([], [], ls='none', marker='o', ms=4, mfc=SHELF_TAN, mec='none'),
        Line2D([], [], ls='none', marker='s', ms=4, mfc=ts.ORANGE,
               mec='white', mew=0.4),
        Line2D([], [], ls='none', marker='*', ms=8, mfc=STAR_COLOR,
               mec='white', mew=0.4),
        Line2D([], [], ls='-', color=GT_COLOR, lw=1.3),
        Line2D([], [], ls='none', marker='o', ms=4, mfc=ts.BLUE, mec='none'),
    ]
    labels = ['Pince + cube', 'Étagère (dépôt)', 'Observations',
              'Pointe (TCP)', 'Vérité terrain', 'État FM']
    fig.legend(handles=handles, labels=labels, loc='lower center',
               bbox_to_anchor=(0.5, 0.86), ncol=len(handles),
               columnspacing=0.9, handletextpad=0.4, borderaxespad=0.0,
               fontsize=7.2)

    fig.subplots_adjust(left=0.09, right=0.995, top=0.75, bottom=0.09,
                        wspace=0.10, hspace=0.28)

    canonical_name = 'pp_fm_denoising_scene'
    figure_name = f'{canonical_name}_{TRAJ_ID}_{STATE_IDX}'
    fig.savefig(os.path.join(OUTDIR, canonical_name + '.svg'))
    fig.savefig(os.path.join(OUTDIR, canonical_name + '.png'))
    ts.save(fig, OUTDIR, figure_name)
    print(f"  -> {os.path.join(OUTDIR, canonical_name)}.svg / .png")
    print(f"  -> {os.path.join(OUTDIR, figure_name)}.svg / .png")


def main():
    global STATE_IDX
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

    # Clamp STATE_IDX to a valid window for the chosen trajectory.
    base = os.path.join(_PKG, 'datas', 'PickAndPlace_preprocess', f'Trajectory_{TRAJ_ID}')
    n_states = len(json.load(open(os.path.join(base, f'trajectory_{TRAJ_ID}.json')))['states'])
    STATE_IDX = int(np.clip(STATE_IDX, OBS_HORIZON - 1, n_states - PRED_HORIZON - 1))
    print(f"Using Trajectory {TRAJ_ID}, state {STATE_IDX} / {n_states}")

    sample = build_sample(TRAJ_ID, STATE_IDX)
    fig_denoising_scene(model, sample, device)
    print("Done.")


if __name__ == '__main__':
    main()
