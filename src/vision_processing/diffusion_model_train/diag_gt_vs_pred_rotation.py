#!/usr/bin/env python3
"""Disambiguate the deployment 'no rotation' finding against ground truth.

The live diagnostic (scripts/diag_fork_rotation.py) showed the deployed planner
commands ~12 deg of rotation per 16-step chunk, dominated by the fork-frame X
axis (drifting into Z), with only ~2-4 deg about Y. This script decodes GROUND
TRUTH dataset windows AND model predictions on the SAME conditioning, with the
identical wp0->wpN relative-rotation / axis math, so we can tell whether:

  * GT itself is X-dominant  -> the model is faithful; "rotation about Y" was a
    mislabel and the physical 'teeth parallel to table' motion is fork-X here.
  * GT is Y-dominant but the prediction is X-dominant -> a real generation or
    rotation-convention bug, not just execution starvation.

It also prints the per-waypoint TOTAL-angle profile so you can see whether the
rotation is front- or back-loaded across the horizon (the live tool only showed
the Y component).

Run with the venv python from this directory:
    python3 diag_gt_vs_pred_rotation.py
"""
import os
import sys

import numpy as np
import torch
import rospkg

_PKG = rospkg.RosPack().get_path('vision_processing')
sys.path.insert(0, os.path.join(_PKG, 'scripts'))

from Train_Fork_FlowMP_9D import FlowMatchingAgent      # noqa: E402
from Data_Loader_Fork_FlowMP_9D import Robot3DDataset    # noqa: E402

CKPT_NAME = 'best_fm_model_9D_dynamics_1024.ckpt'
NUM_STEPS = 5
N_WINDOWS = 12          # windows sampled across the val set
PROFILE_PTS = 8


def ortho6d_to_R(d6):
    """(...,6)->(...,3,3) Gram-Schmidt, matching decode_9d_to_se3_gpu."""
    x = d6[..., 0:3]
    y = d6[..., 3:6]
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def rotvec(R):
    """(3,3)->axis*angle via scipy-free Rodrigues log."""
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(tr)
    if ang < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]]) / (2.0 * np.sin(ang))
    return w * ang


def analyze(action_9d, label):
    """action_9d: [H,9] in the same local frame the model/GT live in."""
    H = action_9d.shape[0]
    Rs = ortho6d_to_R(action_9d[:, 3:])          # [H,3,3]
    R0 = Rs[0]
    R_rel_last = R0.T @ Rs[-1]                    # fork0_R_forkN -> axis in FORK frame
    rv = rotvec(R_rel_last)
    total = np.degrees(np.linalg.norm(rv))
    axis = rv / (np.linalg.norm(rv) + 1e-12)
    ex, ey, ez = np.degrees(rv)                   # rotvec components in fork0 frame

    # Same rotation expressed in the WORLD frame (note the reversed order) so you
    # can see fork-frame vs world-frame axes are genuinely different.
    rv_w = rotvec(Rs[-1] @ R0.T)                  # axis in WORLD frame
    axis_w = rv_w / (np.linalg.norm(rv_w) + 1e-12)

    idxs = np.linspace(0, H - 1, min(PROFILE_PTS, H)).astype(int)
    prof = []
    for k in idxs:
        a = np.degrees(np.linalg.norm(rotvec(R0.T @ Rs[k])))
        prof.append(f'{k:>2}:{a:5.1f}')
    print(f"  [{label:>4}] total={total:5.1f}deg  axis(FORK)=[{axis[0]:+.2f} "
          f"{axis[1]:+.2f} {axis[2]:+.2f}]  axis(WORLD)=[{axis_w[0]:+.2f} "
          f"{axis_w[1]:+.2f} {axis_w[2]:+.2f}]  | profile: {'  '.join(prof)}")
    return np.array([abs(ex), abs(ey), abs(ez)]), total


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = os.path.join(_PKG, 'datas', 'Trajectories_preprocess')
    ckpt_path = os.path.join(_PKG, 'models', CKPT_NAME)
    ds = Robot3DDataset(data_path, mode='val', val_ratio=0.2, seed=42,
                        num_points=1024, obs_horizon=2, pred_horizon=16,
                        data_source='all')
    print(f"Validation windows: {len(ds)}  | device={device}")

    ckpt = torch.load(ckpt_path, map_location=device)
    stats = ckpt.get('stats', None)
    model = FlowMatchingAgent(
        obs_dim=9, action_dim=9, obs_horizon=2, pred_horizon=16,
        encoder_output_dim=64, diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024], kernel_size=5, n_groups=8,
        stats=stats).to(device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)),
                          strict=False)
    model.eval()

    idxs = np.linspace(0, len(ds) - 1, N_WINDOWS).astype(int)
    gt_axis_acc, pred_axis_acc = [], []
    for idx in idxs:
        s = ds[int(idx)]
        gt = s['action'].numpy()
        obs = {'point_cloud': s['obs']['point_cloud'].unsqueeze(0).to(device),
               'agent_pos': s['obs']['agent_pos'].unsqueeze(0).to(device)}
        with torch.no_grad():
            pred = model.predict_action(obs, num_steps=NUM_STEPS,
                                        method='euler')['action_pred'].cpu().numpy()[0]
        print(f"window {idx}:")
        g_axis, g_tot = analyze(gt, 'GT')
        p_axis, p_tot = analyze(pred, 'PRED')
        if g_tot > 5.0:
            gt_axis_acc.append(g_axis)
        if p_tot > 5.0:
            pred_axis_acc.append(p_axis)

    print("\n=== aggregate mean |angle| per fork-0 axis (windows with >5deg net) ===")
    if gt_axis_acc:
        m = np.mean(gt_axis_acc, axis=0)
        print(f"  GT  : X={m[0]:5.1f}  Y={m[1]:5.1f}  Z={m[2]:5.1f}  (n={len(gt_axis_acc)})")
    if pred_axis_acc:
        m = np.mean(pred_axis_acc, axis=0)
        print(f"  PRED: X={m[0]:5.1f}  Y={m[1]:5.1f}  Z={m[2]:5.1f}  (n={len(pred_axis_acc)})")
    print("\nIf GT is X-dominant too, the model is faithful and the feeding "
          "rotation is fork-X in this frame (not Y). If GT is Y-dominant but "
          "PRED is X-dominant, it's a generation/convention bug.")


if __name__ == '__main__':
    main()
