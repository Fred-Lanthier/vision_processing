#!/usr/bin/env python3
"""Is the spiky gradient error from the SDF or from the faceted mesh ground truth?
Along each clean ray, measure the step-to-step direction change (jitter) of:
  - the Bernstein SDF gradient   (should be smooth: analytic polynomial derivative)
  - the exact mesh gradient      (P - closest)/||.||  (faceted -> may jump)
"""
import os, sys
import numpy as np, torch, trimesh
SDF = "/home/flanthier/Github/src/vision_processing/third_party/SDF_Bernstein_Basis"
sys.path.insert(0, os.path.join(SDF, "..", "..")); sys.path.insert(0, SDF); os.chdir(SDF)
import visualize_robot_sdf_layers as V

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
links = V.CBF_PROTECTED_LINKS
rl, w, core = V.build_sdf_stack(dev, links)
pose = torch.eye(4, device=dev).unsqueeze(0)
q9 = V.q7_to_q9(V.DEFAULT_Q, 0.001, dev)
meshes = V.build_ground_truth_meshes(w, rl, pose, q9, links)
offsets = np.linspace(0.001, 0.05, 25, dtype=np.float32)


def unit(v): return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9, None)
def step_jitter(dirs):  # dirs: (R,S,3) -> mean abs angle change between adjacent samples, per ray
    d = unit(dirs)
    cos = np.clip(np.sum(d[:, 1:] * d[:, :-1], axis=2), -1, 1)
    return np.degrees(np.arccos(cos)).mean(axis=1)  # (R,)


bern_j, exact_j = [], []
for name, m in zip(links, meshes):
    rp, gd, gg, _ = V.make_external_benchmark_rays([m], meshes, 60, offsets, 123)
    clean = np.all(np.abs(gd - offsets[None, :]) < 0.003, axis=1)
    rp, gg = rp[clean][:20], gg[clean][:20]
    R, S, _ = rp.shape
    _, gb = V.evaluate_sdf_and_grad(core, rp.reshape(-1, 3).astype(np.float32), pose, q9, 32768, dev)
    bern_j.append(step_jitter(gb.reshape(R, S, 3)))
    exact_j.append(step_jitter(gg))
bj, ej = np.concatenate(bern_j), np.concatenate(exact_j)

print("Step-to-step gradient DIRECTION change along clean rays (samples ~2 mm apart):")
print(f"  Bernstein SDF gradient : median={np.median(bj):.2f} deg  mean={bj.mean():.2f}  p95={np.percentile(bj,95):.2f}  max={bj.max():.2f}")
print(f"  Exact MESH gradient    : median={np.median(ej):.2f} deg  mean={ej.mean():.2f}  p95={np.percentile(ej,95):.2f}  max={ej.max():.2f}")
print(f"\n=> mesh ground-truth jitters {ej.mean()/max(bj.mean(),1e-6):.1f}x more than the Bernstein gradient")
