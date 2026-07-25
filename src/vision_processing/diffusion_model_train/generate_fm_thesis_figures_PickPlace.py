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
import json

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
# Modèle 10D : pose 9D (position 3 + ortho6d 6) + commande gripper (dim 9).
CKPT_NAME = 'best_fm_model_pickplace_10D_1024_rdn.ckpt'
ROT = slice(3, 9)        # rotation ortho6d ; dim 9 = gripper (ouvert/fermé)
GRIP = 9
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
    gt_rot = ortho6d_to_rotation_matrix(gt[-1, ROT][None, None, :])[0, 0]
    pr_rot = ortho6d_to_rotation_matrix(pred[-1, ROT][None, None, :])[0, 0]
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
        ax.plot(mean_pred[:, i], mean_pred[:, j], color=ts.GREEN, lw=2.0,
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
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    ts.save(fig, OUTDIR, 'pp_fm_trajectoire_multimodale')


# ==============================================================================
# Erreur le long de l'horizon (coherence avec la cible)
# ==============================================================================

def _dump_horizon_json(outdir, name, steps, med, mean, q25, q75, p10, p90,
                       n_windows, n_samples, num_steps):
    """Chiffres de la figure d'erreur le long de l'horizon, à côté du SVG.

    Contient aussi l'OUVERTURE de la bande interquartile, q75-q25, et les
    premiers indices k où elle franchit des seuils explicites : c'est la
    grandeur qui dit à partir d'où la prédiction cesse d'être contrainte par
    l'observation et où les modes se séparent."""
    iqr = np.asarray(q75) - np.asarray(q25)

    def first_above(thr):
        hit = np.flatnonzero(iqr > thr)
        return int(hit[0]) if hit.size else None

    payload = {
        'figure': name,
        'fenetres_validation': int(n_windows),
        'predictions_par_fenetre': int(n_samples),
        'pas_euler': int(num_steps),
        'par_k': [
            {'k': int(k), 'mediane': float(med[i]), 'moyenne': float(mean[i]),
             'q25': float(q25[i]), 'q75': float(q75[i]),
             'p10': float(p10[i]), 'p90': float(p90[i]),
             'largeur_iqr': float(iqr[i])}
            for i, k in enumerate(steps)
        ],
        'ouverture_bande_interquartile': {
            'largeur_a_k0': float(iqr[0]),
            'largeur_finale': float(iqr[-1]),
            'premier_k_iqr_sup_0p5': first_above(0.5),
            'premier_k_iqr_sup_1p0': first_above(1.0),
            'premier_k_iqr_sup_moitie_finale': first_above(iqr[-1] / 2.0),
        },
    }
    out = os.path.join(outdir, name + '.json')
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  -> {out}")


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
        gt_rot = ortho6d_to_rotation_matrix(gt[None, :, ROT])[0]
        preds = infer_many(model, sample, device, n_samples, num_steps)
        pos_errs.append(np.linalg.norm(
            preds[:, :, :3] - gt[None, :, :3], axis=2) * 100.0)
        pr_rot = ortho6d_to_rotation_matrix(preds[:, :, ROT])
        rot_errs.append(geodesic_angle_deg(pr_rot, gt_rot[None, ...]))
    pos_errs = np.concatenate(pos_errs, axis=0)
    rot_errs = np.concatenate(rot_errs, axis=0)
    steps = np.arange(pos_errs.shape[1])

    # Une figure à un seul graphique par métrique (position / rotation).
    for data, ylab, name in [
        (pos_errs, 'Erreur de position (cm)', 'pp_fm_erreur_horizon_position'),
        (rot_errs, 'Erreur de rotation ($^\\circ$)', 'pp_fm_erreur_horizon_rotation'),
    ]:
        med = np.median(data, axis=0)
        mean = np.mean(data, axis=0)
        q25, q75 = np.percentile(data, [25, 75], axis=0)
        p10, p90 = np.percentile(data, [10, 90], axis=0)
        _dump_horizon_json(OUTDIR, name, steps, med, mean, q25, q75, p10, p90,
                           n_windows, n_samples, num_steps)
        fig, ax = plt.subplots(figsize=(ts.width(0.5), 3.3))
        # Bandes ombrées (sens précisé en légende de la figure, pas dans le
        # cadre) : écart interquartile (foncé) et percentiles 10–90 (clair).
        ax.fill_between(steps, p10, p90, color=ts.BLUE, alpha=0.12)
        ax.fill_between(steps, q25, q75, color=ts.BLUE, alpha=0.28)
        ax.plot(steps, med, color=ts.ORANGE, lw=2.0, label='Médiane')
        ax.plot(steps, mean, color=ts.GREEN, lw=1.6, ls='--', label='Moyenne')
        ax.set_xlabel("Point de l'horizon de prédiction ($k$)")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, steps[-1])
        ax.set_ylim(bottom=0)
        ts.legend_top(ax)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        ts.save(fig, OUTDIR, name)


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
        gt_rot = ortho6d_to_rotation_matrix(gt[-1, ROT])
        pr_rot = ortho6d_to_rotation_matrix(preds[:, -1, ROT])
        rot_final.extend(geodesic_angle_deg(pr_rot, gt_rot[None, ...]))
    dtw_err = np.asarray(dtw_err)
    rot_final = np.asarray(rot_final)

    # Une figure à un seul histogramme par métrique (DTW / rotation finale).
    for data, xlab, name in [
        (dtw_err, 'DTW de position (cm)', 'pp_fm_distribution_erreur_dtw'),
        (rot_final, 'Erreur de rotation finale ($^\\circ$)',
         'pp_fm_distribution_erreur_rotation'),
    ]:
        fig, ax = plt.subplots(figsize=(ts.width(0.5), 3.3))
        ax.hist(data, bins=40, color=ts.BLUE, alpha=0.85)
        ax.axvline(np.median(data), color=ts.ORANGE, lw=1.6,
                   label=f'Médiane = {np.median(data):.2f}')
        ax.axvline(np.mean(data), color=ts.GREEN, lw=1.6, ls='--',
                   label=f'Moyenne = {np.mean(data):.2f}')
        ax.set_xlabel(xlab)
        ax.set_ylabel('Compte')
        ts.legend_top(ax)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        ts.save(fig, OUTDIR, name)


# ==============================================================================
# Annexe I — Convergence de l'entraînement (courbes de perte)
# ==============================================================================

def _ema(x, beta=0.9):
    """Exponential moving average with bias correction (smoothing des courbes)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    m = 0.0
    for i, v in enumerate(x):
        m = beta * m + (1.0 - beta) * v
        out[i] = m / (1.0 - beta ** (i + 1))
    return out


def fig_loss_curves(history, outdir=OUTDIR, name='pp_fm_loss_curves'):
    """Annexe I — courbes de perte L_CFM entraînement/validation + meilleur ckpt."""
    tr = np.asarray(history['train_loss'], dtype=np.float64)
    va = np.asarray(history['val_loss'], dtype=np.float64)
    ep = np.arange(1, len(tr) + 1)
    best = int(np.argmin(va))
    print(f"[Annexe I] courbes de perte ({len(ep)} époques, "
          f"meilleur ckpt à l'époque {ep[best]}) ...")

    fig, ax = plt.subplots(figsize=(ts.width(0.74), 3.1))
    ax.plot(ep, tr, color=ts.GREY, alpha=0.25, lw=0.8)
    ax.plot(ep, va, color=ts.BLUE, alpha=0.25, lw=0.8)
    ax.plot(ep, _ema(tr), color=ts.GREY, lw=1.8, label='Entraînement (EMA)')
    ax.plot(ep, _ema(va), color=ts.BLUE, lw=1.8, label='Validation (EMA)')
    ax.axvline(ep[best], color=ts.ORANGE, ls='--', lw=1.4,
               label=f'Meilleur point de contrôle (époque {ep[best]})')
    ax.scatter([ep[best]], [va[best]], color=ts.ORANGE, s=26, zorder=6)
    ax.set_yscale('log')
    ax.set_xlabel('Époque')
    ax.set_ylabel(r'Perte $\mathcal{L}_{\mathrm{CFM}}$')
    ax.set_xlim(1, ep[-1])
    ts.legend_top(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    ts.save(fig, outdir, name)


# ==============================================================================
# Annexe I — Étude d'ablation sur le nombre de pas d'Euler N
# ==============================================================================

def _time_inference(model, sample, device, num_steps, warmup=8, reps=50,
                    compiled=True):
    """Temps de génération FM [ms] à N pas, DANS LA CONFIGURATION DÉPLOYÉE.

    La version précédente chronométrait `predict_action`, ce qui surestimait
    d'un facteur ~4,6 le coût réellement payé en ligne, pour deux raisons :

      1. `predict_action` appelle `self.forward`, donc RÉ-ENCODE le nuage de
         points à chaque pas d'Euler. Le nœud déployé appelle `encode_obs` UNE
         SEULE FOIS par replanification puis boucle sur `velocity_net` seul ;
         l'encodage est d'ailleurs publié séparément (`fm_encode`), il ne fait
         pas partie de l'étage `fm_generation` de la §5.6.
      2. `predict_action` s'exécute en mode « eager ». Le nœud déployé compile
         `velocity_net` en `torch.compile(mode='reduce-overhead')`, donc le
         capture en graphe CUDA : à batch 1 le coût est dominé par le lancement
         des noyaux, que le graphe supprime.

    Ce chronomètre reproduit donc la boucle du nœud : encodage hors mesure,
    puis N passes `velocity_net` sous `inference_mode`, batch 1, fp32.
    Retourne (t_generation_ms, t_encode_ms) — même découpage que la §5.6."""
    B, H = 1, model.pred_horizon
    obs = {
        'point_cloud': sample['obs']['point_cloud'].unsqueeze(0).to(device),
        'agent_pos':   model.normalizer.normalize_obs(
            sample['obs']['agent_pos'].unsqueeze(0).to(device)),
    }
    dt = 1.0 / num_steps
    sync = (lambda: torch.cuda.synchronize()) if device.type == 'cuda' else (lambda: None)

    net = _compiled_velocity_net(model, device) if compiled else model.velocity_net

    with torch.inference_mode():
        # Rodage : compilation + capture du graphe CUDA aux formes de production.
        for _ in range(warmup):
            gc = model.encode_obs(obs)
            A = torch.randn(B, H, model.action_dim, device=device)
            for i in range(num_steps):
                A = A + net(A, torch.full((B,), i * dt, device=device), gc) * dt
        sync()

        t0 = time.perf_counter()
        for _ in range(reps):
            gc = model.encode_obs(obs)
        sync()
        t_encode = (time.perf_counter() - t0) / reps * 1000.0

        gc = model.encode_obs(obs)
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            A = torch.randn(B, H, model.action_dim, device=device)
            for i in range(num_steps):
                A = A + net(A, torch.full((B,), i * dt, device=device), gc) * dt
        sync()
        t_gen = (time.perf_counter() - t0) / reps * 1000.0

        # Variante « telle qu'instrumentée dans le nœud » : le compteur
        # `fm_generation` de la §5.6 encadre les passes velocity_net SANS
        # torch.cuda.synchronize(), il ne mesure donc que le temps CPU de mise
        # en file (replay du graphe), pas l'achèvement GPU. On la reproduit ici
        # — file vide au départ, comme entre deux replanifications — pour que le
        # chiffre de la §5.6 et celui de cette figure soient réconciliables.
        t_cpu = 0.0
        for _ in range(reps):
            sync()
            t0 = time.perf_counter()
            A = torch.randn(B, H, model.action_dim, device=device)
            for i in range(num_steps):
                A = A + net(A, torch.full((B,), i * dt, device=device), gc) * dt
            t_cpu += time.perf_counter() - t0
        t_cpu = t_cpu / reps * 1000.0

    return t_gen, t_encode, t_cpu


def _compiled_velocity_net(model, device, _cache={}):
    """`velocity_net` compilé en `reduce-overhead` (graphe CUDA), comme déployé."""
    if 'net' in _cache:
        return _cache['net']
    net = model.velocity_net
    if device.type == 'cuda':
        try:
            net = torch.compile(model.velocity_net, mode='reduce-overhead')
        except Exception as e:                                   # pragma: no cover
            print(f"    (torch.compile indisponible : {e} — mesure en mode eager)")
            net = model.velocity_net
    _cache['net'] = net
    return net


def fig_euler_steps_ablation(model, dataset, device, outdir=OUTDIR,
                             steps_list=(1, 2, 3, 5, 10, 20),
                             n_windows=60, n_samples=3, operating_point=5,
                             name='pp_fm_euler_steps_ablation'):
    """Annexe I — (a) DTW moyenne vs N ; (b) temps de génération [ms] vs N.

    Le temps rapporté est celui de l'étage `fm_generation` du nœud déployé
    (encodage exclu, `velocity_net` capturé en graphe CUDA) : c'est la grandeur
    que chiffre la §5.6, contrairement à la version antérieure de cette figure.
    Un JSON de chiffres est écrit à côté des SVG."""
    n_windows = min(n_windows, len(dataset))
    print(f"[Annexe I] ablation pas d'Euler N∈{tuple(steps_list)} "
          f"({n_windows} fenêtres x {n_samples} préd.) ...")
    idxs = np.linspace(0, len(dataset) - 1, n_windows, dtype=int)
    samples = [dataset[int(k)] for k in idxs]
    timing_sample = samples[len(samples) // 2]

    rows = []
    dtw_mean, dtw_std, time_ms = [], [], []
    for N in steps_list:
        errs = []
        for s in samples:
            gt = s['action'].numpy()
            preds = infer_many(model, s, device, n_samples, num_steps=N)
            for pred in preds:
                errs.append(compute_dtw_position_error(
                    pred[:, :3], gt[:, :3], normalize=True) * 100.0)
        errs = np.asarray(errs)
        t_gen, t_enc, t_cpu = _time_inference(model, timing_sample, device, N)
        dtw_mean.append(errs.mean())
        dtw_std.append(errs.std())
        time_ms.append(t_gen)
        rows.append({
            'N': int(N),
            'dtw_moyenne_cm': float(errs.mean()),
            'dtw_ecart_type_cm': float(errs.std()),
            'dtw_mediane_cm': float(np.median(errs)),
            'dtw_q25_cm': float(np.percentile(errs, 25)),
            'dtw_q75_cm': float(np.percentile(errs, 75)),
            'fm_generation_ms': float(t_gen),
            'fm_generation_par_pas_ms': float(t_gen / N),
            'fm_generation_cpu_sans_sync_ms': float(t_cpu),
            'fm_encode_ms': float(t_enc),
            'n_echantillons': int(errs.size),
        })
        print(f"    N={N:2d} : DTW={errs.mean():.3f} cm  |  "
              f"fm_generation={t_gen:.2f} ms ({t_gen / N:.2f} ms/pas)  |  "
              f"CPU sans sync={t_cpu:.2f} ms  |  fm_encode={t_enc:.2f} ms")

    steps_arr = np.asarray(steps_list, dtype=float)
    dtw_mean = np.asarray(dtw_mean)
    dtw_std = np.asarray(dtw_std)
    time_ms = np.asarray(time_ms)

    # Une figure à un seul graphique par métrique (précision / latence).
    # (a) DTW moyenne — la moyenne est en vert (convention mémoire).
    fig, ax = plt.subplots(figsize=(ts.width(0.5), 3.3))
    ax.fill_between(steps_arr, dtw_mean - dtw_std, dtw_mean + dtw_std,
                    color=ts.GREEN, alpha=0.15)
    ax.plot(steps_arr, dtw_mean, color=ts.GREEN, lw=1.8, marker='o', ms=5)
    i_op = list(steps_list).index(operating_point)
    ts.stats_box(ax, f'$N=1$ : moyenne = {dtw_mean[0]:.2f} cm\n'
                     f'$N={operating_point}$ : moyenne = {dtw_mean[i_op]:.2f} cm\n'
                     f'$N=20$ : moyenne = {dtw_mean[-1]:.2f} cm', loc='upper left')
    ax.set_xlabel("Nombre de pas d'Euler $N$")
    ax.set_ylabel('Erreur DTW moyenne (cm)')
    ax.set_xticks(steps_arr)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    ts.save(fig, outdir, name + '_dtw')

    # (b) temps de génération — croissance ~linéaire en N.
    fig, ax = plt.subplots(figsize=(ts.width(0.5), 3.3))
    ax.plot(steps_arr, time_ms, color=ts.BLUE, lw=1.8, marker='s', ms=5)
    ts.stats_box(ax, f'$N={operating_point}$ : {time_ms[i_op]:.2f} ms\n'
                     f'Pente = {time_ms[i_op] / operating_point:.2f} ms/pas',
                 loc='upper left')
    ax.set_xlabel("Nombre de pas d'Euler $N$")
    ax.set_ylabel("Temps de génération FM (ms)")
    ax.set_xticks(steps_arr)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    ts.save(fig, outdir, name + '_temps')

    payload = {
        'configuration': {
            'checkpoint': CKPT_NAME,
            'appareil': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
            'lot': 1, 'horizon_prediction': int(model.pred_horizon),
            'precision': 'fp32',
            'velocity_net': "torch.compile(mode='reduce-overhead') — graphe CUDA",
            'encodage': "encode_obs appele une seule fois par rollout (etage fm_encode)",
            'etage_mesure': "fm_generation = somme des N passes velocity_net",
            'note_synchronisation': (
                "fm_generation_ms est synchronise (cout GPU reel, trace dans la "
                "figure). fm_generation_cpu_sans_sync_ms reproduit le compteur du "
                "noeud, qui n'appelle pas torch.cuda.synchronize() et ne mesure "
                "donc que la mise en file cote CPU : c'est la grandeur rapportee "
                "par la §5.6."),
            'fenetres_validation': int(n_windows),
            'predictions_par_fenetre': int(n_samples),
        },
        'par_N': rows,
    }
    out = os.path.join(outdir, name + '.json')
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  -> {out}")


# ==============================================================================
# Annexe I — Histogramme du ratio de rectitude des flots (Rectified Flow)
# ==============================================================================

def _straightness_ratios(model, sample, device, n_samples, n_fine=50):
    """Ratio longueur du chemin d'intégration / longueur de la corde, par échantillon.

    Intégration fine (n_fine pas d'Euler) dans l'ESPACE NORMALISÉ où le flot est
    défini (x_0 ~ N(0,I), champ v = forward). Le chemin ∫|Ẋ_t|dt ≈ Σ_k ||v_k|| dt
    et la corde = ||x_1 - x_0|| sont mesurés sur le vecteur d'action aplati, donc
    un ratio ≈ 1 traduit un transport quasi rectiligne (Rectified Flow)."""
    obs = {
        'point_cloud': sample['obs']['point_cloud'].unsqueeze(0).expand(
            n_samples, -1, -1).to(device),
        'agent_pos': model.normalizer.normalize_obs(
            sample['obs']['agent_pos'].unsqueeze(0).expand(
                n_samples, -1, -1).to(device)),
    }
    dt = 1.0 / n_fine
    x = torch.randn(n_samples, model.pred_horizon, model.action_dim, device=device)
    x0 = x.clone()
    path = torch.zeros(n_samples, device=device)
    with torch.no_grad():
        for i in range(n_fine):
            t = torch.ones(n_samples, device=device) * (i / n_fine)
            v = model.forward(obs, x, t)
            path += v.flatten(1).norm(dim=1) * dt
            x = x + v * dt
    chord = (x - x0).flatten(1).norm(dim=1)
    ratio = (path / chord.clamp_min(1e-8)).cpu().numpy()
    return ratio


def fig_straightness_ratio(model, dataset, device, outdir=OUTDIR,
                           n_windows=40, n_samples=8, n_fine=50,
                           name='pp_fm_straightness_ratio'):
    """Annexe I — distribution du ratio de rectitude des flots (≈ 1 = rectiligne)."""
    n_windows = min(n_windows, len(dataset))
    print(f"[Annexe I] ratio de rectitude "
          f"({n_windows} fenêtres x {n_samples} préd., {n_fine} pas fins) ...")
    idxs = np.linspace(0, len(dataset) - 1, n_windows, dtype=int)
    ratios = []
    for k in idxs:
        ratios.append(_straightness_ratios(model, dataset[int(k)], device,
                                           n_samples, n_fine=n_fine))
    ratios = np.concatenate(ratios)

    fig, ax = plt.subplots(figsize=(ts.width(0.74), 3.1))
    ax.hist(ratios, bins=40, color=ts.BLUE, alpha=0.85)
    ax.axvline(1.0, color=ts.GREY, ls=':', lw=1.4, label='Idéal ($=1$)')
    ax.axvline(np.median(ratios), color=ts.ORANGE, lw=1.6,
               label=f'Médiane = {np.median(ratios):.3f}')
    ax.axvline(np.mean(ratios), color=ts.GREEN, lw=1.6, ls='--',
               label=f'Moyenne = {np.mean(ratios):.3f}')
    ax.set_xlabel(r'Ratio de rectitude  $\int_0^1\!\|\dot{X}_t\|\,dt \,/\, \|X_1-X_0\|$')
    ax.set_ylabel('Compte')
    ts.legend_top(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    ts.save(fig, outdir, name)


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
        obs_dim=10, action_dim=10, obs_horizon=2, pred_horizon=16,
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

    # Annexe I — figures de convergence / d'ablation du modèle FM (pick-and-place).
    history = ckpt.get('history', None)
    if history is not None and 'val_loss' in history:
        fig_loss_curves(history)
    else:
        print("  (pas d'historique de perte dans le checkpoint — "
              "pp_fm_loss_curves ignorée)")
    fig_euler_steps_ablation(model, val_dataset, device)
    fig_straightness_ratio(model, val_dataset, device)

    print("Done. PickPlace FM + Annexe I figures written (.svg + .png).")


if __name__ == '__main__':
    main()
