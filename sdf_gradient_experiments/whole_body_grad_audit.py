#!/usr/bin/env python3
"""ISOLATED audit (reads only; production models untouched).

Tests whether the 'weird' large gradient errors are MEASUREMENT artifacts, not the
production model. Evaluates the production whole-body min-SDF the way the CBF uses it:
  - production min over all protected links (not per-link in isolation),
  - only on GENUINE EXTERIOR points (true robot distance ~= offset, which excludes the
    buried mounting-stump patches the min never exposes in production),
  - reports error with and without degenerate medial-axis points (||grad||->0, where
    gradient direction is mathematically undefined).

Exact ground truth uses ONE concatenated robot mesh + a single AABB proximity query
(fast), instead of looping closest_point over every link.
"""

import os, sys
import numpy as np
import torch
import trimesh

SDF_BERNSTEIN = "/home/flanthier/Github/src/vision_processing/third_party/SDF_Bernstein_Basis"
sys.path.insert(0, os.path.join(SDF_BERNSTEIN, "..", ".."))
sys.path.insert(0, SDF_BERNSTEIN)
os.chdir(SDF_BERNSTEIN)
import visualize_robot_sdf_layers as V

OFFSET = 0.02     # 2 cm clearance band
TOL = 0.004       # |true distance - offset| < 4 mm  -> genuine exterior
DEGEN = 0.5       # ||grad|| below this = degenerate / medial-axis


def pct(a, p):
    return float(np.percentile(a, p))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    links = V.CBF_PROTECTED_LINKS
    rl, w, core = V.build_sdf_stack(dev, links)              # PRODUCTION whole-body min
    pose = torch.eye(4, device=dev).unsqueeze(0)
    q9 = V.q7_to_q9(V.DEFAULT_Q, 0.001, dev)

    protected_meshes = V.build_ground_truth_meshes(w, rl, pose, q9, links)
    all_meshes = V.build_ground_truth_meshes(w, rl, pose, q9, V.ALL_LINKS)
    combined = trimesh.util.concatenate(all_meshes)          # whole robot, one mesh
    pq = trimesh.proximity.ProximityQuery(combined)

    # candidate exterior points: push protected-link surfaces out along their normals
    P = []
    for m in protected_meshes:
        pts, fi = trimesh.sample.sample_surface(m, 1500, seed=1)
        P.append(pts + m.face_normals[fi] * OFFSET)
    P = np.vstack(P).astype(np.float32)

    closest, dist, tri = pq.on_surface(P)                    # single fast query
    disp = P - closest
    fn = combined.face_normals[tri]
    external = np.einsum("ij,ij->i", disp, fn) > 0
    keep = external & (np.abs(dist - OFFSET) < TOL)
    P, disp = P[keep], disp[keep]
    gex = disp / np.clip(np.linalg.norm(disp, axis=1, keepdims=True), 1e-9, None)

    _, gb = V.evaluate_sdf_and_grad(core, P, pose, q9, 32768, dev)
    gn = np.linalg.norm(gb, axis=1)
    gu = gb / np.clip(gn[:, None], 1e-9, None)
    ang = np.degrees(np.arccos(np.clip(np.sum(gu * gex, axis=1), -1, 1)))

    good = gn >= DEGEN
    print(f"Genuine exterior points at {OFFSET*100:.0f} cm: {len(P)}")
    print(f"||grad||: mean={gn.mean():.3f}  frac<{DEGEN}={np.mean(~good)*100:.1f}%")
    print()
    print("Gradient angular error (deg):")
    print(f"  ALL points            : median={np.median(ang):5.1f}  mean={ang.mean():5.1f}  "
          f"p90={pct(ang,90):5.1f}  p95={pct(ang,95):5.1f}")
    if good.sum():
        print(f"  EXCL degenerate grads : median={np.median(ang[good]):5.1f}  mean={ang[good].mean():5.1f}  "
              f"p90={pct(ang[good],90):5.1f}  p95={pct(ang[good],95):5.1f}   (n={good.sum()})")
    print()
    bad = ang > 30
    if bad.sum():
        print(f"Points with error >30 deg: {bad.mean()*100:.1f}% of all; "
              f"of those, {np.mean(~good[bad])*100:.1f}% are degenerate (||grad||<{DEGEN}).")
    else:
        print("No points exceed 30 deg error.")


if __name__ == "__main__":
    main()
