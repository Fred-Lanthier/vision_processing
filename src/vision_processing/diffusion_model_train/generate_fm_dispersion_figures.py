#!/usr/bin/env python3
"""Annexe I — Dispersion des générations en fonction du nombre de pas d'Euler N.

L'ablation d'Euler classique (`fm_euler_steps_ablation_dtw`) ne mesure qu'un
PREMIER MOMENT : l'erreur DTW moyenne à la vérité terrain. Une telle métrique
est aveugle à l'effondrement des modes — si N=1 produit une politique quasi
déterministe tout en gardant la même erreur moyenne, la courbe ne bouge pas.
Or c'est précisément la multimodalité qui justifie d'avoir choisi un modèle
génératif plutôt qu'une régression.

Ce module produit les deux livraisons complémentaires :

  B1  fm_dispersion_multimodalite_xy
        Générations multiples depuis UNE MÊME observation, plan XY, un panneau
        par N ∈ {1,2,3,5,10,20} (grille 2x3). Même scène ET MÊMES GRAINES de
        bruit initial X_0 dans tous les panneaux : seul N change, donc les
        différences entre panneaux sont imputables au seul budget d'intégration.

  B2  fm_dispersion_vs_pas
        Mesure scalaire. Pour chaque scène de validation, G générations depuis
        la même observation en ne variant que X_0, puis distance DTW moyenne
        entre PAIRES de générations (cm). Médiane + écart interquartile sur les
        scènes, en fonction de N.

Une dispersion de référence des DÉMONSTRATIONS EXPERTES est encore calculée et
déposée dans le JSON, mais elle n'est PLUS tracée : superposée à la courbe du
modèle elle se confondait avec l'erreur DTW des figures 5.3, qui est une tout
autre grandeur (une génération contre SA PROPRE vérité terrain, nulle pour un
modèle parfait), alors qu'il s'agit ici d'un écart entre deux démonstrations
distinctes de configurations voisines.

Contraintes de figure : thesis_style.py, SVG, Okabe-Ito, français, sans titre,
statistiques annotées dans la figure. Un JSON de chiffres est déposé à côté de
chaque figure (analyse écrite depuis des nombres, pas depuis des pixels).

Lancer avec le python du venv depuis ce dossier :
    python3 generate_fm_dispersion_figures.py                 # alimentation
    python3 generate_fm_dispersion_figures.py --task pickplace
"""
import os
import sys
import json
import argparse
import itertools

import numpy as np
import torch

import rospkg
from dtw import dtw

# --- shared thesis style ------------------------------------------------------
_PKG = rospkg.RosPack().get_path('vision_processing')
sys.path.insert(0, os.path.join(_PKG, 'scripts'))
import thesis_style as ts          # noqa: E402
ts.apply()
import matplotlib.pyplot as plt    # noqa: E402  (after ts.apply so rcParams stick)
from matplotlib.lines import Line2D  # noqa: E402


# ==============================================================================
# Configuration par tâche
# ==============================================================================

STEPS_LIST = (1, 2, 3, 5, 10, 20)   # budgets d'intégration comparés
OPERATING_POINT = 5                  # N déployé (§5.6)

N_SCENES = 40        # fenêtres de validation échantillonnées pour B2
N_GEN = 24           # générations par scène (et segments experts par scène)
SEED_NOISE = 1234    # graine du bruit initial X_0, partagée par tous les N


TASKS = {
    'feeding': dict(
        prefix='',
        outdir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'thesis_fm_figures'),
        data_path='/data/datas/Trajectories_preprocess',
        ckpt='best_fm_model_9D_dynamics_1024.ckpt',
        dim=9,
        train_module='Train_Fork_FlowMP_9D',
        loader_module='Data_Loader_Fork_FlowMP_9D',
        scene_idx=25,   # scène de démonstration pour B1 (même que 5.3)
    ),
    'pickplace': dict(
        prefix='pp_',
        outdir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'thesis_pickplace_figures'),
        data_path=os.path.join(_PKG, 'datas', 'PickAndPlace_preprocess'),
        ckpt='best_fm_model_pickplace_10D_1024_rdn.ckpt',
        dim=10,
        train_module='Train_PickPlace',
        loader_module='Data_Loader_PickPlace',
        scene_idx=25,
    ),
}


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ==============================================================================
# Métrique : DTW de position entre deux segments (cm)
# ==============================================================================

def dtw_cm(a, b):
    """DTW normalisée entre deux trajectoires de position 3D, en centimètres.

    Strictement la même métrique que `compute_dtw_position_error` des scripts de
    figures 5.3 (distance DTW / longueur du chemin d'alignement), convertie en
    cm. Appliquée ici entre DEUX GÉNÉRATIONS (dispersion) plutôt qu'entre une
    génération et la vérité terrain (erreur)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    al = dtw(a, b, dist_method='euclidean', keep_internals=False)
    return al.distance / max(len(al.index1), 1) * 100.0


def mean_pairwise_dtw_cm(segs):
    """DTW moyenne entre toutes les paires d'un jeu de segments (G,H,3) -> cm."""
    vals = [dtw_cm(segs[a], segs[b])
            for a, b in itertools.combinations(range(len(segs)), 2)]
    return float(np.mean(vals)), np.asarray(vals)


def _as_batch(segs):
    """(H,3) ou (G,H,3) -> (G,H,3). `np.atleast_3d` ajoute l'axe À LA FIN et
    donnerait (H,3,1) pour un segment isolé : l'horizon ne serait plus l'axe 1."""
    segs = np.asarray(segs, dtype=np.float64)
    return segs[None, ...] if segs.ndim == 2 else segs


def roughness_cm(segs):
    """Rugosité d'un jeu de segments (…,H,3) : ‖différence seconde‖ moyenne, cm.

    Discrimine le BRUIT résiduel de la DIVERSITÉ de trajectoire. Une dispersion
    inter-générations peut venir de deux sources très différentes : des chemins
    lisses mais distincts (multimodalité réelle), ou un même chemin bruité pas à
    pas (sous-intégration). La différence seconde ne voit que la seconde."""
    segs = _as_batch(segs)
    d2 = segs[:, 2:, :] - 2.0 * segs[:, 1:-1, :] + segs[:, :-2, :]
    return float(np.linalg.norm(d2, axis=-1).mean() * 100.0)


def path_length_cm(segs):
    """Longueur d'arc moyenne d'un jeu de segments (…,H,3), en cm.

    Un pas d'Euler trop grossier SOUS-TRANSPORTE : l'EDO n'atteint pas t=1 et le
    chemin produit est plus court que la démonstration. Comparée à la longueur
    de la vérité terrain, cette grandeur mesure ce défaut directement."""
    segs = _as_batch(segs)
    return float(np.linalg.norm(np.diff(segs, axis=1), axis=-1).sum(1).mean() * 100.0)


# ==============================================================================
# Inférence : intégration d'Euler depuis un X_0 IMPOSÉ
# ==============================================================================

@torch.no_grad()
def rollout_from_noise(model, sample, device, x0, num_steps):
    """G rollouts FM depuis un bruit initial imposé -> actions métriques (G,H,D).

    Reproduit exactement `predict_action(method='euler')` — même champ, même
    schéma, même dénormalisation — mais avec X_0 fourni par l'appelant, ce qui
    permet de rejouer LE MÊME bruit pour chaque valeur de N. L'encodeur
    d'observation est appelé une seule fois par rollout (il est déterministe et
    l'observation est constante), comme dans le nœud déployé."""
    G = x0.shape[0]
    obs = {
        'point_cloud': sample['obs']['point_cloud'].unsqueeze(0).expand(
            G, -1, -1).to(device),
        'agent_pos': model.normalizer.normalize_obs(
            sample['obs']['agent_pos'].unsqueeze(0).expand(G, -1, -1).to(device)),
    }
    global_cond = model.encode_obs(obs)
    x = x0.clone().to(device)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((G,), i / num_steps, device=device)
        x = x + model.velocity_net(x, t, global_cond) * dt
    return model.normalizer.unnormalize_act(x).cpu().numpy()


def fixed_noise(model, device, n_gen, seed=SEED_NOISE):
    """Bruit initial X_0 partagé par tous les panneaux / tous les N."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randn(n_gen, model.pred_horizon, model.action_dim, generator=g)


# ==============================================================================
# Dispersion experte : voisinage dans l'espace de conditionnement
# ==============================================================================

@torch.no_grad()
def encode_dataset(model, dataset, device, batch=64):
    """global_cond de CHAQUE fenêtre de validation -> (M, 128) standardisé."""
    feats = []
    for start in range(0, len(dataset), batch):
        chunk = [dataset[i] for i in range(start, min(start + batch, len(dataset)))]
        obs = {
            'point_cloud': torch.stack([c['obs']['point_cloud'] for c in chunk]).to(device),
            'agent_pos': model.normalizer.normalize_obs(
                torch.stack([c['obs']['agent_pos'] for c in chunk]).to(device)),
        }
        feats.append(model.encode_obs(obs).cpu().numpy())
    f = np.concatenate(feats, axis=0)
    # Standardisation par dimension : les 64 traits du nuage et les 64 traits
    # proprioceptifs n'ont pas la même échelle brute et l'un écraserait l'autre
    # dans la distance L2 du voisinage.
    return (f - f.mean(0)) / (f.std(0) + 1e-8)


def trajectory_groups(dataset):
    """Indice de trajectoire de chaque fenêtre, reconstruit par continuité.

    Le Robot3DDataset empile les fenêtres glissantes trajectoire par trajectoire
    mais ne conserve pas l'origine de chacune. Deux fenêtres consécutives d'une
    MÊME trajectoire partagent un pas de temps : l'observation la plus ancienne
    de la fenêtre i est l'observation la plus récente de la fenêtre i-1. La part
    ROTATION de la pose (ortho6d) n'étant pas recentrée, l'égalité y est exacte
    et sert de test de continuité ; une rupture marque un changement de
    trajectoire."""
    groups = np.zeros(len(dataset), dtype=int)
    prev = dataset[0]['obs']['agent_pos'].numpy()
    g = 0
    for i in range(1, len(dataset)):
        cur = dataset[i]['obs']['agent_pos'].numpy()
        if not np.allclose(cur[0, 3:9], prev[-1, 3:9], atol=1e-6):
            g += 1
        groups[i] = g
        prev = cur
    return groups


def expert_dispersion(dataset, feats, groups, scene_idx, n_gen):
    """DTW moyenne par paires sur G segments experts de configurations proches.

    Voisins retenus dans l'espace `global_cond` standardisé, en excluant toute
    fenêtre de la même trajectoire que la scène (sinon les voisines seraient les
    pas de temps adjacents de la même démonstration)."""
    d = np.linalg.norm(feats - feats[scene_idx][None, :], axis=1)
    d[groups == groups[scene_idx]] = np.inf
    order = np.argsort(d)
    neigh = [int(k) for k in order[:n_gen - 1] if np.isfinite(d[k])]
    if len(neigh) < n_gen - 1:
        return None, None, None
    picked = [int(scene_idx)] + neigh
    segs = np.stack([dataset[k]['action'].numpy()[:, :3] for k in picked])
    mean_d, vals = mean_pairwise_dtw_cm(segs)
    return mean_d, vals, picked


# ==============================================================================
# B1 — Grille 2x3 : générations dans le plan XY, un panneau par N
# ==============================================================================

def fig_multimodality_grid(model, sample, device, cfg, x0, results_b1,
                           steps_list=STEPS_LIST):
    """Générations multiples, plan XY, un panneau par N. Bruit initial partagé."""
    name = cfg['prefix'] + 'fm_dispersion_multimodalite_xy'
    print(f"[Annexe I] B1 — multimodalité XY par N ({len(x0)} générations) ...")

    gt = sample['action'].numpy()
    hist = sample['obs']['agent_pos'].numpy()

    preds = {N: rollout_from_noise(model, sample, device, x0, N) for N in steps_list}

    # Limites communes à TOUS les panneaux : sans cela chaque panneau se
    # re-cadre sur sa propre dispersion et l'effondrement des modes devient
    # invisible — c'est exactement ce que la figure doit montrer.
    allpts = np.concatenate([p[:, :, :2].reshape(-1, 2) for p in preds.values()]
                            + [gt[:, :2], hist[:, :2]], axis=0)
    ctr = (allpts.min(0) + allpts.max(0)) / 2.0
    half = np.maximum((allpts.max(0) - allpts.min(0)) * 1.10 / 2.0, 0.01)
    half[:] = half.max()          # cadre carré : l'échelle X et Y est identique
    lo, hi = ctr - half, ctr + half
    # Marge supérieure pour l'encadré de statistiques : appliquée à l'identique
    # aux six panneaux, elle n'ajoute que du vide et laisse les distances
    # comparables d'un panneau à l'autre (aspect toujours égal).
    hi[1] += 0.30 * (hi[1] - lo[1])

    fig, axes = plt.subplots(2, 3, figsize=(ts.TEXTWIDTH, 4.5), squeeze=False)
    for k, N in enumerate(steps_list):
        ax = axes[k // 3, k % 3]
        p = preds[N]
        for g in range(len(p)):
            ax.plot(p[g, :, 0], p[g, :, 1], color=ts.BLUE, alpha=0.28, lw=0.8,
                    zorder=2, label='Générations (FM)' if (g == 0 and k == 0) else None)
        ax.plot(gt[:, 0], gt[:, 1], color=ts.GREY, ls='--', lw=1.5, zorder=4,
                label='Vérité terrain' if k == 0 else None)
        ax.plot(hist[:, 0], hist[:, 1], color=ts.ORANGE, marker='o', ms=3, lw=1.2,
                zorder=5, label='Historique' if k == 0 else None)

        mean_d, vals = mean_pairwise_dtw_cm(p[:, :, :3])
        results_b1[str(N)] = {
            'dispersion_dtw_moyenne_cm': mean_d,
            'dispersion_dtw_mediane_cm': float(np.median(vals)),
            'n_paires': int(vals.size),
        }
        ts.stats_box(ax, f'Médiane = {np.median(vals):.2f} cm\n'
                         f'Moyenne = {mean_d:.2f} cm')

        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(fr'$N = {N}$', fontsize=11, pad=3)
        if k // 3 == 1:
            ax.set_xlabel('X (m)')
        else:
            ax.tick_params(labelbottom=False)
        if k % 3 == 0:
            ax.set_ylabel('Y (m)')
        else:
            ax.tick_params(labelleft=False)

    handles = [Line2D([], [], color=ts.BLUE, lw=1.2),
               Line2D([], [], color=ts.GREY, ls='--', lw=1.5),
               Line2D([], [], color=ts.ORANGE, marker='o', ms=3, lw=1.2)]
    labels = [f'Générations (FM, $G={len(x0)}$, même $X_0$)', 'Vérité terrain',
              'Historique']
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.02, 0.955),
               ncol=3, borderaxespad=0.0, columnspacing=1.2, handlelength=1.5,
               handletextpad=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    ts.save(fig, cfg['outdir'], name)
    print(f"  -> {os.path.join(cfg['outdir'], name)}.svg / .png")


# ==============================================================================
# B2 — Dispersion (DTW inter-générations) en fonction de N
# ==============================================================================

def fig_dispersion_vs_steps(model, dataset, device, cfg, x0, results,
                            steps_list=STEPS_LIST, n_scenes=N_SCENES):
    """Médiane + IQR de la dispersion inter-générations vs N, ligne experte."""
    name = cfg['prefix'] + 'fm_dispersion_vs_pas'
    n_scenes = min(n_scenes, len(dataset))
    idxs = np.linspace(0, len(dataset) - 1, n_scenes, dtype=int)
    print(f"[Annexe I] B2 — dispersion vs N ({n_scenes} scènes x {len(x0)} "
          f"générations x {len(steps_list)} valeurs de N) ...")

    # --- dispersion des générations, scène par scène -------------------------
    # On relève au passage deux grandeurs de STRUCTURE (rugosité, longueur
    # d'arc) : elles séparent une dispersion faite de chemins lisses distincts
    # d'une dispersion faite d'un même chemin bruité.
    disp = {N: [] for N in steps_list}
    rough = {N: [] for N in steps_list}
    lenrat = {N: [] for N in steps_list}
    gt_rough, gt_len = [], []
    for c, k in enumerate(idxs):
        s = dataset[int(k)]
        gt = s['action'].numpy()[:, :3]
        gt_rough.append(roughness_cm(gt))
        gt_len.append(path_length_cm(gt))
        for N in steps_list:
            p = rollout_from_noise(model, s, device, x0, N)[:, :, :3]
            disp[N].append(mean_pairwise_dtw_cm(p)[0])
            rough[N].append(roughness_cm(p))
            lenrat[N].append(path_length_cm(p) / max(gt_len[-1], 1e-9))
        if (c + 1) % 10 == 0:
            print(f"    {c + 1}/{n_scenes} scènes")

    # --- dispersion experte sur configurations comparables -------------------
    print("[Annexe I] B2 — dispersion experte (voisinage dans global_cond) ...")
    feats = encode_dataset(model, dataset, device)
    groups = trajectory_groups(dataset)
    print(f"    {groups.max() + 1} trajectoires reconstruites "
          f"({len(dataset.traj_folders)} dossiers de validation)")
    exp_vals = []
    for k in idxs:
        m, _, _ = expert_dispersion(dataset, feats, groups, int(k), len(x0))
        if m is not None:
            exp_vals.append(m)
    exp_vals = np.asarray(exp_vals)

    steps_arr = np.asarray(steps_list, dtype=float)
    med = np.array([np.median(disp[N]) for N in steps_list])
    q25 = np.array([np.percentile(disp[N], 25) for N in steps_list])
    q75 = np.array([np.percentile(disp[N], 75) for N in steps_list])
    exp_med = float(np.median(exp_vals))
    exp_q25, exp_q75 = np.percentile(exp_vals, [25, 75])

    i_op = list(steps_list).index(OPERATING_POINT)

    def _steps_axis(ax):
        ax.set_xlabel("Nombre de pas d'Euler $N$")
        ax.set_xscale('log')
        ax.set_xticks(steps_arr)
        ax.set_xticklabels([f'{int(n)}' for n in steps_arr])
        ax.minorticks_off()

    fig, ax = plt.subplots(figsize=(ts.width(0.8), 3.5))
    ax.fill_between(steps_arr, q25, q75, color=ts.BLUE, alpha=0.25,
                    label='Écart interquartile')
    ax.plot(steps_arr, med, color=ts.ORANGE, lw=2.0, marker='o', ms=5,
            label='Médiane')
    _steps_axis(ax)
    ax.set_ylabel('Dispersion inter-générations\n(DTW par paires, cm)')
    ax.set_ylim(0, q75.max() * 1.30)
    ts.stats_box(ax,
                 f'$N=1$ : médiane = {med[0]:.2f} cm\n'
                 f'$N={OPERATING_POINT}$ : médiane = {med[i_op]:.2f} cm',
                 loc='upper left')
    ts.legend_top(ax, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    ts.save(fig, cfg['outdir'], name)
    print(f"  -> {os.path.join(cfg['outdir'], name)}.svg / .png")

    # --- figure compagnon : de quoi la dispersion est-elle faite ? -----------
    # Sans elle, la courbe ci-dessus est ambiguë : la dispersion ne s'effondre
    # PAS aux petits N, elle y est même maximale. Ces deux panneaux montrent que
    # cette dispersion-là n'est pas de la multimodalité mais du bruit résiduel
    # sur un chemin sous-transporté.
    name2 = cfg['prefix'] + 'fm_dispersion_structure'
    r_med = np.array([np.median(rough[N]) for N in steps_list])
    r_q25 = np.array([np.percentile(rough[N], 25) for N in steps_list])
    r_q75 = np.array([np.percentile(rough[N], 75) for N in steps_list])
    l_med = np.array([np.median(lenrat[N]) for N in steps_list])
    l_q25 = np.array([np.percentile(lenrat[N], 25) for N in steps_list])
    l_q75 = np.array([np.percentile(lenrat[N], 75) for N in steps_list])
    gt_r = float(np.median(gt_rough))

    fig, axes = plt.subplots(1, 2, figsize=(ts.TEXTWIDTH, 3.2))
    ax = axes[0]
    ax.fill_between(steps_arr, r_q25, r_q75, color=ts.BLUE, alpha=0.25)
    ax.plot(steps_arr, r_med, color=ts.BLUE, lw=1.8, marker='o', ms=5,
            label='Générations')
    ax.axhline(gt_r, color=ts.GREY, ls='--', lw=1.5, label='Démonstrations')
    _steps_axis(ax)
    ax.set_ylabel('Rugosité du chemin\n(‖différence seconde‖, cm)')
    ax.set_ylim(0, max(r_q75.max(), gt_r) * 1.35)   # place pour l'encadré
    ts.stats_box(ax, f'$N=1$ : {r_med[0]:.2f} cm\n'
                     f'$N={OPERATING_POINT}$ : {r_med[i_op]:.2f} cm\n'
                     f'Démos : {gt_r:.2f} cm', loc='upper right')
    ts.legend_top(ax)

    ax = axes[1]
    ax.fill_between(steps_arr, l_q25, l_q75, color=ts.BLUE, alpha=0.25)
    ax.plot(steps_arr, l_med, color=ts.BLUE, lw=1.8, marker='o', ms=5,
            label='Générations')
    ax.axhline(1.0, color=ts.GREY, ls='--', lw=1.5, label='Démonstrations')
    _steps_axis(ax)
    ax.set_ylabel("Longueur d'arc générée\n/ longueur de la démonstration")
    ax.set_ylim(0, max(l_q75.max(), 1.0) * 1.35)    # place pour l'encadré
    ts.stats_box(ax, f'$N=1$ : {l_med[0]:.2f}\n'
                     f'$N={OPERATING_POINT}$ : {l_med[i_op]:.2f}\n'
                     f'$N=20$ : {l_med[-1]:.2f}', loc='upper right')
    ts.legend_top(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    ts.save(fig, cfg['outdir'], name2)
    print(f"  -> {os.path.join(cfg['outdir'], name2)}.svg / .png")

    results['dispersion_vs_pas'] = {
        'definition': ("DTW moyenne par paires entre G generations issues de la "
                       "meme observation, seule X_0 variant ; en cm"),
        'n_scenes': int(n_scenes),
        'n_generations_par_scene': int(len(x0)),
        'n_paires_par_scene': int(len(x0) * (len(x0) - 1) // 2),
        'par_N': {
            str(N): {
                'mediane_cm': float(np.median(disp[N])),
                'moyenne_cm': float(np.mean(disp[N])),
                'q25_cm': float(np.percentile(disp[N], 25)),
                'q75_cm': float(np.percentile(disp[N], 75)),
                'min_cm': float(np.min(disp[N])),
                'max_cm': float(np.max(disp[N])),
                'ratio_a_expert': float(np.median(disp[N]) / exp_med),
                'rugosite_mediane_cm': float(np.median(rough[N])),
                'rugosite_q25_cm': float(np.percentile(rough[N], 25)),
                'rugosite_q75_cm': float(np.percentile(rough[N], 75)),
                'rugosite_ratio_aux_demos': float(np.median(rough[N]) / np.median(gt_rough)),
                'longueur_relative_mediane': float(np.median(lenrat[N])),
                'longueur_relative_q25': float(np.percentile(lenrat[N], 25)),
                'longueur_relative_q75': float(np.percentile(lenrat[N], 75)),
            } for N in steps_list
        },
        'structure': {
            'definition': ("rugosite = ||p[k+1]-2p[k]+p[k-1]|| moyenne le long de "
                           "l'horizon (cm) ; longueur_relative = longueur d'arc "
                           "generee / longueur d'arc de la demonstration"),
            'rugosite_demonstrations_mediane_cm': float(np.median(gt_rough)),
            'longueur_demonstrations_mediane_cm': float(np.median(gt_len)),
        },
        'experts': {
            'definition': ("DTW moyenne par paires sur G segments de demonstration "
                           "de configurations comparables (G-1 plus proches voisins "
                           "dans global_cond standardise, trajectoire d'origine "
                           "exclue) ; en cm"),
            'n_scenes': int(exp_vals.size),
            'mediane_cm': exp_med,
            'moyenne_cm': float(np.mean(exp_vals)),
            'q25_cm': float(exp_q25),
            'q75_cm': float(exp_q75),
        },
        'n_trajectoires_validation': int(groups.max() + 1),
    }


# ==============================================================================
# MAIN
# ==============================================================================

def load_task(task, device):
    cfg = TASKS[task]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    train_mod = __import__(cfg['train_module'], fromlist=['FlowMatchingAgent'])
    loader_mod = __import__(cfg['loader_module'], fromlist=['Robot3DDataset'])

    dataset = loader_mod.Robot3DDataset(
        cfg['data_path'], mode='val', val_ratio=0.2, seed=42,
        num_points=1024, obs_horizon=2, pred_horizon=16, data_source='all')
    print(f"Validation set: {len(dataset)} sequences")

    ckpt_path = os.path.join(_PKG, 'models', cfg['ckpt'])
    ckpt = torch.load(ckpt_path, map_location=device)
    model = train_mod.FlowMatchingAgent(
        obs_dim=cfg['dim'], action_dim=cfg['dim'], obs_horizon=2, pred_horizon=16,
        encoder_output_dim=64, diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024], kernel_size=5, n_groups=8,
        stats=ckpt.get('stats', None)).to(device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)),
                          strict=False)
    model.eval()
    print(f"Weights loaded ({cfg['ckpt']}). Normalizer initialized: "
          f"{bool(model.normalizer.is_initialized)}")
    return cfg, model, dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=sorted(TASKS), default='feeding')
    ap.add_argument('--scenes', type=int, default=N_SCENES)
    ap.add_argument('--gen', type=int, default=N_GEN)
    args = ap.parse_args()

    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg, model, dataset = load_task(args.task, device)
    os.makedirs(cfg['outdir'], exist_ok=True)
    print(f"Device: {device}  | figures -> {cfg['outdir']}")

    x0 = fixed_noise(model, device, args.gen)

    results = {
        'tache': args.task,
        'checkpoint': cfg['ckpt'],
        'pas_euler_compares': list(STEPS_LIST),
        'point_de_fonctionnement': OPERATING_POINT,
        'graine_bruit_initial': SEED_NOISE,
        'multimodalite_xy': {},
    }

    scene_idx = cfg['scene_idx'] if len(dataset) > cfg['scene_idx'] else 0
    results['multimodalite_xy']['scene_validation'] = int(scene_idx)
    fig_multimodality_grid(model, dataset[scene_idx], device, cfg, x0,
                           results['multimodalite_xy'])

    fig_dispersion_vs_steps(model, dataset, device, cfg, x0, results,
                            n_scenes=args.scenes)

    out = os.path.join(cfg['outdir'], cfg['prefix'] + 'fm_dispersion.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  -> {out}")
    print("Done.")


if __name__ == '__main__':
    main()
