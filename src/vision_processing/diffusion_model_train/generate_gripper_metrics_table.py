#!/usr/bin/env python3
"""Annexe I — Table de la métrique de succès de la PINCE (modèle FM 10D pick-and-place).

Complète les figures FM (`generate_fm_thesis_figures_PickPlace.py`) par la
caractérisation de la 10e dimension : la commande binaire d'ouverture/fermeture
de la pince. On évalue, au point de fonctionnement DÉPLOYÉ (N=5 pas d'Euler,
méthode euler), la prédiction de la pince image-par-image sur l'ensemble de
validation, en pooling toutes les étapes de l'horizon x plusieurs tirages
stochastiques par fenêtre.

La pince est un classifieur binaire (seuil 0.5) :
  * classe « ouverte »  : gripper >= 0.5,
  * classe « fermée »   : gripper <  0.5  (= saisie/transport du cube).
Métriques par classe : support, MAE de la commande, précision, rappel, F1 ;
ligne « Global » : support total, MAE globale, précision/rappel/F1 pondérés
(le rappel pondéré = exactitude globale).

Sorties (style thèse, booktabs) dans ./thesis_pickplace_figures/ :
    pp_gripper_metrics.tex   — tableau LaTeX (\\begin{tabular} ... )
    pp_gripper_metrics.csv   — mêmes chiffres, séparés par des virgules

Lancer avec le python du venv depuis ce dossier :
    python3 generate_gripper_metrics_table.py
"""
import os
import sys

import numpy as np
import torch

import rospkg

_PKG = rospkg.RosPack().get_path('vision_processing')
sys.path.insert(0, os.path.join(_PKG, 'scripts'))

from Train_PickPlace import FlowMatchingAgent, GRIPPER_BINARY_THRESHOLD  # noqa: E402
from Data_Loader_PickPlace import Robot3DDataset                          # noqa: E402

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thesis_pickplace_figures')
CKPT_NAME = 'best_fm_model_pickplace_10D_1024_rdn.ckpt'
GRIP = 9                 # index of the gripper (open/close) dim in the 10D action
NUM_STEPS = 5            # deployed inference budget (euler)
EVAL_WINDOWS = 200       # validation windows sampled across the full set
EVAL_SAMPLES = 20        # stochastic predictions per validation window


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def infer_many(model, sample, device, n_samples, num_steps=NUM_STEPS,
               method='euler'):
    """Batched stochastic FM rollouts for one conditioning sample -> [n,H,10]."""
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


def collect_gripper(model, dataset, device, n_windows=EVAL_WINDOWS,
                    n_samples=EVAL_SAMPLES, num_steps=NUM_STEPS):
    """Pool frame-level (pred, gt) gripper commands over the validation set."""
    n_windows = min(n_windows, len(dataset))
    print(f"[table pince] collecte ({n_windows} fenêtres x {n_samples} préd., "
          f"N={num_steps}) ...")
    idxs = np.linspace(0, len(dataset) - 1, n_windows, dtype=int)
    grip_pred, grip_gt = [], []
    for idx in idxs:
        sample = dataset[int(idx)]
        gt = sample['action'].numpy()
        preds = infer_many(model, sample, device, n_samples, num_steps)
        # broadcast GT over the stochastic samples, pool every horizon frame
        grip_pred.append(np.clip(preds[:, :, GRIP], 0.0, 1.0).reshape(-1))
        grip_gt.append(np.broadcast_to(gt[None, :, GRIP],
                                       preds[:, :, GRIP].shape).reshape(-1))
    return np.concatenate(grip_pred), np.concatenate(grip_gt)


def _prf(tp, fp, fn):
    """Précision, rappel, F1 (en %) à partir des comptes."""
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec * 100.0, rec * 100.0, f1 * 100.0


def compute_metrics(grip_pred, grip_gt, thr=GRIPPER_BINARY_THRESHOLD):
    """Métriques binaires par classe + globale (pondérée)."""
    pred_open = grip_pred >= thr
    gt_open = grip_gt >= thr
    gt_closed = ~gt_open
    pred_closed = ~pred_open
    abs_err = np.abs(grip_pred - grip_gt)

    # « ouverte » comme classe positive
    tp_o = int(np.sum(pred_open & gt_open))
    fp_o = int(np.sum(pred_open & gt_closed))
    fn_o = int(np.sum(pred_closed & gt_open))
    # « fermée » (saisie) comme classe positive
    tp_c = int(np.sum(pred_closed & gt_closed))
    fp_c = int(np.sum(pred_closed & gt_open))
    fn_c = int(np.sum(pred_open & gt_closed))

    n_open = int(np.sum(gt_open))
    n_closed = int(np.sum(gt_closed))
    n_tot = n_open + n_closed

    po, ro, fo = _prf(tp_o, fp_o, fn_o)
    pc, rc, fc = _prf(tp_c, fp_c, fn_c)
    mae_o = float(abs_err[gt_open].mean()) if n_open else 0.0
    mae_c = float(abs_err[gt_closed].mean()) if n_closed else 0.0
    mae_all = float(abs_err.mean())

    # global : moyennes pondérées par le support (rappel pondéré = exactitude)
    w_o, w_c = n_open / n_tot, n_closed / n_tot
    acc = float(np.mean(pred_open == gt_open)) * 100.0
    prec_w = w_o * po + w_c * pc
    f1_w = w_o * fo + w_c * fc

    return {
        'rows': [
            ('Pince ouverte',        n_open,   mae_o, po, ro, fo),
            ('Pince fermée (saisie)', n_closed, mae_c, pc, rc, fc),
        ],
        'global': ('\\textbf{Global}', n_tot, mae_all, prec_w, acc, f1_w),
        'accuracy': acc,
        'mae': mae_all,
        'thr': thr,
    }


# ==============================================================================
# Ecriture des tableaux (style thèse : booktabs)
# ==============================================================================

_HEADER = ('Classe', 'Support', 'MAE', 'Précision [%]', 'Rappel [%]', 'F1 [%]')


def _fmt_row(name, support, mae, prec, rec, f1):
    return f'{name} & {support} & {mae:.4f} & {prec:.2f} & {rec:.2f} & {f1:.2f} \\\\'


def write_tables(m, outdir=OUTDIR, name='pp_gripper_metrics'):
    os.makedirs(outdir, exist_ok=True)
    rows = list(m['rows']) + [m['global']]

    # --- CSV ---
    csv_path = os.path.join(outdir, name + '.csv')
    with open(csv_path, 'w') as f:
        f.write(','.join(('classe', 'support', 'mae', 'precision_pct',
                          'rappel_pct', 'f1_pct')) + '\n')
        for name_r, sup, mae, p, r, f1 in rows:
            clean = name_r.replace('\\textbf{', '').replace('}', '')
            f.write(f'{clean},{sup},{mae:.6f},{p:.4f},{r:.4f},{f1:.4f}\n')

    # --- LaTeX (booktabs) ---
    tex_path = os.path.join(outdir, name + '.tex')
    with open(tex_path, 'w') as f:
        f.write('\\begin{tabular}{lrrrrr}\n\\toprule\n')
        f.write(' & '.join(_HEADER) + ' \\\\\n\\midrule\n')
        for name_r, sup, mae, p, r, f1 in m['rows']:
            f.write(_fmt_row(name_r, sup, mae, p, r, f1) + '\n')
        f.write('\\midrule\n')
        gname, gsup, gmae, gp, gr, gf1 = m['global']
        f.write(_fmt_row(gname, gsup, gmae, gp, gr, gf1) + '\n')
        f.write('\\bottomrule\n\\end{tabular}\n')

    return tex_path, csv_path


def print_console(m):
    print('\n' + '=' * 60)
    print('  MÉTRIQUE DE SUCCÈS DE LA PINCE — modèle FM 10D (pick-and-place)')
    print(f'  seuil binaire = {m["thr"]:.2f}  |  N = {NUM_STEPS} pas (euler)')
    print('=' * 60)
    hdr = f'{"Classe":<24}{"Support":>9}{"MAE":>9}{"Préc.%":>9}{"Rappel%":>9}{"F1%":>9}'
    print(hdr)
    print('-' * 60)
    for name_r, sup, mae, p, r, f1 in m['rows']:
        print(f'{name_r:<24}{sup:>9}{mae:>9.4f}{p:>9.2f}{r:>9.2f}{f1:>9.2f}')
    print('-' * 60)
    gname, gsup, gmae, gp, gr, gf1 = m['global']
    print(f'{"Global":<24}{gsup:>9}{gmae:>9.4f}{gp:>9.2f}{gr:>9.2f}{gf1:>9.2f}')
    print('=' * 60)
    print(f'  Exactitude globale : {m["accuracy"]:.2f} %   |   MAE globale : {m["mae"]:.4f}')
    print('=' * 60)


def main():
    seed_everything(42)
    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  | table -> {OUTDIR}")

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

    grip_pred, grip_gt = collect_gripper(model, val_dataset, device)
    m = compute_metrics(grip_pred, grip_gt)
    print_console(m)
    tex_path, csv_path = write_tables(m)
    print(f"\n  -> {tex_path}")
    print(f"  -> {csv_path}")
    print("Done.")


if __name__ == '__main__':
    main()
