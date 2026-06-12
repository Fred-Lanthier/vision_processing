#!/usr/bin/env python3
"""Clean, apples-to-apples measurement of the production SDF.

Ground truth = union mesh of EXACTLY the links the SDF contains (no extra links).
Reports distance error and gradient error on genuine exterior points.
Degenerate medial-axis points (||grad||->0, direction mathematically undefined)
are reported separately, not mixed into the gradient stats.
"""

import os, sys
import numpy as np
import torch
import trimesh

SDF_BERNSTEIN = "/home/flanthier/Github/src/vision_processing/third_party/SDF_Bernstein_Basis"
sys.path.insert(0, os.path.join(SDF_BERNSTEIN, "..", "..")); sys.path.insert(0, SDF_BERNSTEIN)
os.chdir(SDF_BERNSTEIN)
import visualize_robot_sdf_layers as V

DEGEN = 0.5


def q(a, p): return float(np.percentile(a, p))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    links = V.CBF_PROTECTED_LINKS                      # the SDF's own link set
    rl, w, core = V.build_sdf_stack(dev, links)
    pose = torch.eye(4, device=dev).unsqueeze(0)
    q9 = V.q7_to_q9(V.DEFAULT_Q, 0.001, dev)

    meshes = V.build_ground_truth_meshes(w, rl, pose, q9, links)   # SAME links as SDF
    combined = trimesh.util.concatenate(meshes)        # union = what the SDF approximates
    pq = trimesh.proximity.ProximityQuery(combined)

    # sample exterior points across a 0.5-4 cm shell around the union surface
    rng = np.random.default_rng(0)
    pts, fi = trimesh.sample.sample_surface(combined, 20000, seed=1)
    off = rng.uniform(0.005, 0.04, size=(len(pts), 1))
    P = (pts + combined.face_normals[fi] * off).astype(np.float32)

    closest, dist, tri = pq.on_surface(P)
    disp = P - closest
    fn = combined.face_normals[tri]
    # genuine exterior: outside the union, and the marched offset really is the distance
    keep = (np.einsum("ij,ij->i", disp, fn) > 0) & (np.abs(dist - off[:, 0]) < 0.004)
    P, dist, disp = P[keep], dist[keep], disp[keep]
    gex = disp / np.clip(np.linalg.norm(disp, axis=1, keepdims=True), 1e-9, None)

    sdf, gb = V.evaluate_sdf_and_grad(core, P, pose, q9, 32768, dev)
    derr = sdf - dist                                  # signed distance error [m]
    gn = np.linalg.norm(gb, axis=1)
    gu = gb / np.clip(gn[:, None], 1e-9, None)
    ang = np.degrees(np.arccos(np.clip(np.sum(gu * gex, axis=1), -1, 1)))
    good = gn >= DEGEN

    print(f"Production SDF vs its OWN {len(links)}-link union | {len(P)} exterior points (0.5-4 cm)\n")
    print("DISTANCE error  (Bernstein - exact) [mm]:")
    print(f"  MAE={np.mean(np.abs(derr))*1000:.2f}  bias={np.mean(derr)*1000:+.2f}  "
          f"p95|err|={q(np.abs(derr),95)*1000:.2f}  max|err|={np.max(np.abs(derr))*1000:.2f}")
    print("\nGRADIENT direction error [deg]:")
    print(f"  all points           : median={np.median(ang):.1f}  mean={ang.mean():.1f}  p95={q(ang,95):.1f}")
    print(f"  excl. medial-axis    : median={np.median(ang[good]):.1f}  mean={ang[good].mean():.1f}  "
          f"p95={q(ang[good],95):.1f}   ({np.mean(~good)*100:.1f}% excluded, ||grad||<{DEGEN})")
    print(f"\n||grad|| (ideal 1.0): mean={gn.mean():.3f}  std={gn.std():.3f}")


if __name__ == "__main__":
    main()
