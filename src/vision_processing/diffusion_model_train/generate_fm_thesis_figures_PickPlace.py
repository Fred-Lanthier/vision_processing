#!/usr/bin/env python3
"""Figures these — Qualite de la generation nominale (Flow Matching) PICK-AND-PLACE.

Variante PICK-AND-PLACE de `generate_fm_thesis_figures.py` : memes figures et
metriques (multimodalite, erreur le long de l'horizon, distributions d'erreur),
mais sur le modele/donnees pick-and-place :
  * modele   : Train_PickPlace.FlowMatchingAgent
  * donnees  : datas/PickAndPlace_preprocess (Data_Loader_PickPlace)
  * ckpt     : best_fm_model_pickplace_9D_1024.ckpt
Sorties dans ./thesis_pickplace_figures/ (prefixe pp_) pour ne pas ecraser les
figures de la fourchette.

Lancer avec le python du venv depuis ce dossier :
    python3 generate_fm_thesis_figures_PickPlace.py
"""
import os
import sys
import time

import numpy as np
import torch

import rospkg
from dtw import dtw

# --- shared thesis style (same module as the SDF/CBF figures) ----------------
_PKG = rospkg.RosPack().get_path('vision_processing')
sys.path.insert(0, os.path.join(_PKG, 'scripts'))
import thesis_style as ts          # noqa: E402
ts.apply()
import matplotlib.pyplot as plt    # noqa: E402  (after ts.apply so rcParams stick)

# --- model + data infrastructure (PICK-AND-PLACE) ---------------------------
from Train_PickPlace import FlowMatchingAgent     # noqa: E402
from Data_Loader_PickPlace import Robot3DDataset   # noqa: E402

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thesis_pickplace_figures')
NUM_STEPS = 5            # deployed inference budget
CKPT_NAME = 'best_fm_model_pickplace_9D_1024.ckpt'
MULTIMODAL_SAMPLES = 40  # intentionally unchanged: qualitative figure
EVAL_WINDOWS = 200       # validation windows sampled across the full set
EVAL_SAMPLES = 20        # stochastic predictions per validation window


# ==============================================================================
# Minimal inference / geometry helpers (self-contained, no side effects)
# ==============================================================================

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def ortho6d_to_rotation_matrix(d6):
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt."""
    x_raw = d6[..., 0:3]
    y_raw = d6[..., 3:6]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def geodesic_angle_deg(R1, R2):
    """Geodesic angle [deg] between rotation matrices, batched over (...,3,3)."""
    r_diff = np.matmul(R1, np.swapaxes(R2, -1, -2))
    trace = np.trace(r_diff, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def compute_dtw_position_error(pred_pos, gt_pos, normalize=True):
    """Normalized DTW distance between two 3D position trajectories (shape fidelity)."""
    pred_pos = np.asarray(pred_pos, dtype=np.float64)
    gt_pos = np.asarray(gt_pos, dtype=np.float64)
    if len(pred_pos) == 0 or len(gt_pos) == 0:
        return np.nan
    alignment = dtw(pred_pos, gt_pos, dist_method='euclidean', keep_internals=False)
    if not normalize:
        return alignment.distance
    return alignment.distance / max(len(alignment.index1), 1)


def infer_single(model, sample, device, num_steps=NUM_STEPS, method='euler'):
    """One FM rollout -> (pred_action [H,9], final_dist, final_angle_deg, dt_s)."""
    model.eval()
    obs = {
        'point_cloud': sample['obs']['point_cloud'].unsqueeze(0).to(device),
        'agent_pos':   sample['obs']['agent_pos'].unsqueeze(0).to(device),
    }
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.predict_action(obs, num_steps=num_steps, method=method)
        pred = out['action_pred'].cpu().numpy()[0]
    dt = time.perf_counter() - t0

    gt = sample['action'].numpy()
    final_dist = float(np.linalg.norm(pred[-1, :3] - gt[-1, :3]))
    gt_rot = ortho6d_to_rotation_matrix(gt[-1, 3:][None, None, :])[0, 0]
    pr_rot = ortho6d_to_rotation_matrix(pred[-1, 3:][None, None, :])[0, 0]
    final_angle = float(geodesic_angle_deg(gt_rot, pr_rot))
    return pred, final_dist, final_angle, dt


def infer_many(model, sample, device, n_samples, num_steps=NUM_STEPS,
               method='euler'):
    """Batched stochastic FM rollouts for one conditioning sample."""
    model.eval()
    obs = {
        'point_cloud': sample['obs']['point_cloud'].unsqueeze(0).expand(
            n_samples, -1, -1).to(device),
        'agent_pos': sample['obs']['agent_pos'].unsqueeze(0).expand(
            n_samples, -1, -1).to(device),
    }
    with torch.no_grad():
        out = model.predict_action(obs, num_steps=num_steps, method=method)
    return out['action_pred'].cpu().numpy()


# ==============================================================================
# Multimodalite de la trajectoire generee
# ==============================================================================

def fig_trajectory_multimodality(model, sample, device, idx,
                                 n_samples=MULTIMODAL_SAMPLES,
                                 num_steps=NUM_STEPS):
    print(f"[pp] multimodalité (seq {idx}) ...")
    gt = sample['action'].numpy()
    hist = sample['obs']['agent_pos'].numpy()

    preds = np.array([infer_single(model, sample, device, num_steps)[0]
                      for _ in range(n_samples)])
    mean_pred = preds.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(ts.TEXTWIDTH, 3.0))
    for ax, (i, j, li, lj) in zip(axes, [(0, 1, 'X', 'Y'), (0, 2, 'X', 'Z')]):
        for k in range(len(preds)):
            ax.plot(preds[k, :, i], preds[k, :, j], color=ts.BLUE, alpha=0.10,
                    lw=0.8, zorder=2,
                    label='Prédictions (FM)' if k == 0 else None)
        ax.plot(mean_pred[:, i], mean_pred[:, j], color=ts.BLUE, lw=2.0,
                zorder=4, label='Prédiction moyenne')
        ax.plot(gt[:, i], gt[:, j], color=ts.GREY, ls='--', lw=1.6, zorder=5,
                label='Vérité terrain')
        ax.plot(hist[:, i], hist[:, j], color=ts.ORANGE, marker='o', ms=3,
                lw=1.2, zorder=6, label='Historique')
        ax.set_xlabel(f'{li} (m)')
        ax.set_ylabel(f'{lj} (m)')
        ax.set_aspect('equal', adjustable='datalim')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.94),
               ncol=len(labels), columnspacing=1.1, handletextpad=0.5,
               borderaxespad=0.0)
    ts.panel_label(axes[0], '(a)')
    ts.panel_label(axes[1], '(b)')
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    ts.save(fig, OUTDIR, 'pp_fm_trajectoire_multimodale')


# ==============================================================================
# Erreur le long de l'horizon (coherence avec la cible)
# ==============================================================================

def fig_error_along_horizon(model, dataset, device,
                            n_windows=EVAL_WINDOWS, n_samples=EVAL_SAMPLES,
                            num_steps=NUM_STEPS):
    n_windows = min(n_windows, len(dataset))
    print(f"[pp] erreur le long de l'horizon "
          f"({n_windows} fenêtres x {n_samples} prédictions) ...")
    idxs = np.linspace(0, len(dataset) - 1, n_windows, dtype=int)
    pos_errs, rot_errs = [], []
    for idx in idxs:
        sample = dataset[int(idx)]
        gt = sample['action'].numpy()
        gt_rot = ortho6d_to_rotation_matrix(gt[None, :, 3:])[0]
        preds = infer_many(model, sample, device, n_samples, num_steps)
        pos_errs.append(np.linalg.norm(
            preds[:, :, :3] - gt[None, :, :3], axis=2) * 100.0)
        pr_rot = ortho6d_to_rotation_matrix(preds[:, :, 3:])
        rot_errs.append(geodesic_angle_deg(pr_rot, gt_rot[None, ...]))
    pos_errs = np.concatenate(pos_errs, axis=0)
    rot_errs = np.concatenate(rot_errs, axis=0)
    steps = np.arange(pos_errs.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(ts.TEXTWIDTH, 3.0))
    for ax, data, ylab in [
        (axes[0], pos_errs, 'Erreur de position (cm)'),
        (axes[1], rot_errs, 'Erreur de rotation ($^\\circ$)'),
    ]:
        med = np.median(data, axis=0)
        mean = np.mean(data, axis=0)
        q25, q75 = np.percentile(data, [25, 75], axis=0)
        p10, p90 = np.percentile(data, [10, 90], axis=0)
        ax.fill_between(steps, p10, p90, color=ts.BLUE, alpha=0.12,
                        label='Percentiles 10–90')
        ax.fill_between(steps, q25, q75, color=ts.BLUE, alpha=0.28,
                        label='Écart interquartile')
        ax.plot(steps, med, color=ts.BLUE, lw=2.0, label='Médiane')
        ax.plot(steps, mean, color=ts.GREEN, lw=1.6, ls='--', label='Moyenne')
        ax.set_xlabel("Point de l'horizon de prédiction ($k$)")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, steps[-1])
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.94),
               ncol=len(labels), columnspacing=1.1, handletextpad=0.5,
               borderaxespad=0.0)
    ts.panel_label(axes[0], '(a)')
    ts.panel_label(axes[1], '(b)')
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    ts.save(fig, OUTDIR, 'pp_fm_erreur_horizon')


# ==============================================================================
# Distributions d'erreur sur l'ensemble de validation
# ==============================================================================

def fig_error_distributions(model, dataset, device,
                            n_windows=EVAL_WINDOWS, n_samples=EVAL_SAMPLES,
                            num_steps=NUM_STEPS):
    n_windows = min(n_windows, len(dataset))
    print(f"[pp] distributions d'erreur "
          f"({n_windows} fenêtres x {n_samples} prédictions) ...")
    idxs = np.linspace(0, len(dataset) - 1, n_windows, dtype=int)
    dtw_err, rot_final = [], []
    for idx in idxs:
        sample = dataset[int(idx)]
        gt = sample['action'].numpy()
        preds = infer_many(model, sample, device, n_samples, num_steps)
        for pred in preds:
            dtw_err.append(compute_dtw_position_error(
                pred[:, :3], gt[:, :3], normalize=True) * 100.0)
        gt_rot = ortho6d_to_rotation_matrix(gt[-1, 3:])
        pr_rot = ortho6d_to_rotation_matrix(preds[:, -1, 3:])
        rot_final.extend(geodesic_angle_deg(pr_rot, gt_rot[None, ...]))
    dtw_err = np.asarray(dtw_err)
    rot_final = np.asarray(rot_final)

    fig, axes = plt.subplots(1, 2, figsize=(ts.TEXTWIDTH, 3.0))
    for ax, data, xlab in [
        (axes[0], dtw_err, 'DTW de position (cm)'),
        (axes[1], rot_final, 'Erreur de rotation finale ($^\\circ$)'),
    ]:
        ax.hist(data, bins=40, color=ts.BLUE, alpha=0.85)
        ax.axvline(np.median(data), color=ts.ORANGE, lw=1.6,
                   label=f'Médiane = {np.median(data):.2f}')
        ax.axvline(np.mean(data), color=ts.GREEN, lw=1.6, ls='--',
                   label=f'Moyenne = {np.mean(data):.2f}')
        ax.set_xlabel(xlab)
        ax.set_ylabel('Compte')
        ts.legend_top(ax)
    ts.panel_label(axes[0], '(a)')
    ts.panel_label(axes[1], '(b)')
    fig.tight_layout()
    ts.save(fig, OUTDIR, 'pp_fm_distribution_erreur')


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    seed_everything(42)
    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  | figures -> {OUTDIR}")

    data_path = os.path.join(_PKG, 'datas', 'PickAndPlace_preprocess')
    ckpt_path = os.path.join(_PKG, 'models', CKPT_NAME)

    val_dataset = Robot3DDataset(
        data_path, mode='val', val_ratio=0.2, seed=42,
        num_points=1024, obs_horizon=2, pred_horizon=16, data_source='all')
    print(f"Validation set: {len(val_dataset)} sequences")

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location=device)

    stats = ckpt.get('stats', None)
    if stats is None:
        print("WARNING: checkpoint has no 'stats' — relying on normalizer buffers.")

    model = FlowMatchingAgent(
        obs_dim=9, action_dim=9, obs_horizon=2, pred_horizon=16,
        encoder_output_dim=64, diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024], kernel_size=5, n_groups=8,
        stats=stats).to(device)

    weights = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(weights, strict=False)
    model.eval()
    print(f"Weights loaded ({CKPT_NAME}). Normalizer initialized: "
          f"{bool(model.normalizer.is_initialized)}")

    sample_idx = 25 if len(val_dataset) > 25 else 0
    fig_trajectory_multimodality(model, val_dataset[sample_idx], device, sample_idx)
    fig_error_along_horizon(model, val_dataset, device)
    fig_error_distributions(model, val_dataset, device)

    print("Done. PickPlace FM figures written (.svg + .png).")


if __name__ == '__main__':
    main()
