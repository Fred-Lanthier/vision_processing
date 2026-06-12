#!/usr/bin/env python3
"""Diagnose suspected artifacts in the thesis SDF figures.

Checks, on the exact same sampling as make_thesis_results.collect():
  1. Domain clamp: fraction of points whose NEAREST link evaluation hits the
     [-1,1] Bernstein cube boundary, as a function of offset; correlation with
     distance error and gradient angle error.
  2. Ground-truth mesh tessellation (face count per link) -> near-surface
     gradient "exact" reference is faceted.
  3. Error stats split clamped vs unclamped.
"""
import os, sys
import numpy as np
import torch

SDF_BERNSTEIN = "/home/flanthier/Github/src/vision_processing/third_party/SDF_Bernstein_Basis"
sys.path.insert(0, os.path.join(SDF_BERNSTEIN, "..", "..")); sys.path.insert(0, SDF_BERNSTEIN)
os.chdir(SDF_BERNSTEIN)
import visualize_robot_sdf_layers as V

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_thesis_results import collect


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    links = V.CBF_PROTECTED_LINKS
    rl, w, core = V.build_sdf_stack(dev, links)
    pose = torch.eye(4, device=dev).unsqueeze(0)
    q9 = V.q7_to_q9(V.DEFAULT_Q, 0.001, dev)
    meshes = V.build_ground_truth_meshes(w, rl, pose, q9, links)

    print("=== mesh tessellation (ground truth) ===")
    for name, m in zip(links, meshes):
        print(f"  {name:12s} faces={len(m.faces):6d}  scale={float(core.scales[links.index(name)]):.4f} m")

    R = collect(core, meshes, links, pose, q9, dev)
    P = R["P"]  # (M,3) ordered link-major, ray-major, offset-minor
    M = len(P)
    n_off = len(R["off_mm"])

    # Recompute per-link scaled coords exactly as bernstein_core does
    x = torch.as_tensor(P, dtype=torch.float32, device=dev)
    trans_list = rl.get_transformations_each_link(pose, q9)
    matched = []
    for target_link in core.used_links:
        t_name = target_link.replace('panda_', '').replace('_w', '').replace('.pt', '')
        idx = None
        for i, info in enumerate(rl.meshes_info):
            i_name = info['link_name'].replace('panda_', '')
            if t_name in i_name or i_name in t_name:
                idx = i
                break
        matched.append(trans_list[idx])
    trans = torch.stack(matched, dim=1)[0]  # (K,4,4)
    Rm = trans[:, :3, :3]
    tv = trans[:, :3, 3]
    diff = x.unsqueeze(0) - tv.unsqueeze(1)            # (K,M,3)
    x_local = torch.einsum('kmi,kij->kmj', diff, Rm)   # = bmm(diff, R)
    x_scaled = (x_local - core.offsets.unsqueeze(1)) / core.scales.view(-1, 1, 1)

    # nearest link per point = argmin of per-link sdf
    with torch.no_grad():
        _, sdf_pl = core.get_whole_body_sdf_batch(x, pose, q9, return_per_link=True)
    nearest = sdf_pl[0].argmin(dim=0).cpu().numpy()    # (M,)
    clamped_pl = (x_scaled.abs() > (1.0 - 1e-2)).any(dim=-1).cpu().numpy()  # (K,M)
    clamped = clamped_pl[nearest, np.arange(M)]

    err_mm = R["err"] * 1000.0
    ang = R["ang"]
    off_idx = np.tile(np.arange(n_off), M // n_off)
    off_mm = R["off_mm"][off_idx]

    print("\n=== clamp fraction & errors by offset ===")
    print(f"{'off[mm]':>8} {'clamp%':>7} {'MAE_in':>7} {'MAE_cl':>7} {'ang_in':>7} {'ang_cl':>7}")
    for k in range(0, n_off, 3):
        sel = off_idx == k
        c = clamped[sel]
        mae_in = np.mean(np.abs(err_mm[sel][~c])) if (~c).any() else np.nan
        mae_cl = np.mean(np.abs(err_mm[sel][c])) if c.any() else np.nan
        a_in = np.median(ang[sel][~c]) if (~c).any() else np.nan
        a_cl = np.median(ang[sel][c]) if c.any() else np.nan
        print(f"{R['off_mm'][k]:8.1f} {100*c.mean():6.1f}% {mae_in:7.2f} {mae_cl:7.2f} {a_in:7.2f} {a_cl:7.2f}")

    print("\n=== overall split ===")
    for name, mask in (("unclamped", ~clamped), ("clamped", clamped)):
        if mask.any():
            print(f"  {name:10s} n={mask.sum():5d}  MAE={np.mean(np.abs(err_mm[mask])):.2f} mm  "
                  f"bias={np.mean(err_mm[mask]):+.2f} mm  ang_med={np.median(ang[mask]):.1f} deg  "
                  f"|grad|={R['gn'][mask].mean():.3f}")

    print("\n=== clamp% per link (at max offset 50mm) ===")
    sel50 = off_idx == n_off - 1
    for li, name in enumerate(links):
        m = sel50 & (nearest == li)
        if m.any():
            print(f"  {name:12s} n={m.sum():4d}  clamp={100*clamped[m].mean():5.1f}%  "
                  f"ang_med={np.median(ang[m]):.1f} deg  err_mean={err_mm[m].mean():+.2f} mm")

    # near-surface: is error dominated by faceting? compare angle error at first
    # offset vs the angle between exact gradient and the launch normal (should be 0
    # for a perfectly clean ray on a smooth surface).
    print("\n=== near-surface (1mm) stats ===")
    sel0 = off_idx == 0
    print(f"  angle median={np.median(ang[sel0]):.1f} deg  p90={np.percentile(ang[sel0],90):.1f} deg")
    print(f"  |grad| mean={R['gn'][sel0].mean():.3f}")


if __name__ == "__main__":
    main()
