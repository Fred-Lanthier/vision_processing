#!/usr/bin/env python3
"""ISOLATED (reads only). Test the hypothesis: the big gradient errors are points
where the Bernstein min picks a DIFFERENT link than the true nearest surface
(junction 'wrong link wins the min'), or where the true nearest link is an
UNPROTECTED link the whole-body SDF cannot represent at all.

For each genuine-exterior point we record:
  - true_link      : link owning the closest face on the complete robot mesh
  - bernstein_link : argmin over per-link Bernstein SDFs (the link the min selects)
and break the >30 deg gradient-error points down by those.
"""

import os, sys
import numpy as np
import torch
import trimesh

SDF_BERNSTEIN = "/home/flanthier/Github/src/vision_processing/third_party/SDF_Bernstein_Basis"
sys.path.insert(0, os.path.join(SDF_BERNSTEIN, "..", "..")); sys.path.insert(0, SDF_BERNSTEIN)
os.chdir(SDF_BERNSTEIN)
import visualize_robot_sdf_layers as V

OFFSET, TOL = 0.02, 0.004


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    links = V.CBF_PROTECTED_LINKS
    rl, w, core = V.build_sdf_stack(dev, links)
    pose = torch.eye(4, device=dev).unsqueeze(0)
    q9 = V.q7_to_q9(V.DEFAULT_Q, 0.001, dev)

    protected_meshes = V.build_ground_truth_meshes(w, rl, pose, q9, links)
    all_meshes = V.build_ground_truth_meshes(w, rl, pose, q9, V.ALL_LINKS)
    combined = trimesh.util.concatenate(all_meshes)
    face_link = np.concatenate([[name] * len(m.faces) for name, m in zip(V.ALL_LINKS, all_meshes)])
    pq = trimesh.proximity.ProximityQuery(combined)

    P = []
    for m in protected_meshes:
        pts, fi = trimesh.sample.sample_surface(m, 1500, seed=1)
        P.append(pts + m.face_normals[fi] * OFFSET)
    P = np.vstack(P).astype(np.float32)

    closest, dist, tri = pq.on_surface(P)
    disp = P - closest
    fn = combined.face_normals[tri]
    keep = (np.einsum("ij,ij->i", disp, fn) > 0) & (np.abs(dist - OFFSET) < TOL)
    P, disp, tri = P[keep], disp[keep], tri[keep]
    gex = disp / np.clip(np.linalg.norm(disp, axis=1, keepdims=True), 1e-9, None)
    true_link = face_link[tri]

    xt = torch.tensor(P, device=dev)
    _, gb = V.evaluate_sdf_and_grad(core, P, pose, q9, 32768, dev)
    gu = gb / np.clip(np.linalg.norm(gb, axis=1)[:, None], 1e-9, None)
    ang = np.degrees(np.arccos(np.clip(np.sum(gu * gex, axis=1), -1, 1)))

    with torch.no_grad():
        _, per_link = core.get_whole_body_sdf_batch(xt, pose, q9, return_per_link=True)
    bern_link = np.array(links)[per_link[0].argmin(0).cpu().numpy()]

    protected = set(links)
    true_unprot = ~np.isin(true_link, list(protected))
    mism = (bern_link != true_link) & ~true_unprot     # wrong protected link wins min
    same = (bern_link == true_link)

    def stats(mask, label):
        if mask.sum() == 0:
            print(f"  {label:34s}: n=0"); return
        a = ang[mask]
        print(f"  {label:34s}: n={mask.sum():5d} ({mask.mean()*100:4.1f}%)  "
              f"err median={np.median(a):5.1f}  mean={a.mean():5.1f}  p95={np.percentile(a,95):5.1f}")

    print(f"\nGenuine exterior points: {len(P)} | overall err median={np.median(ang):.1f} mean={ang.mean():.1f}\n")
    print("Breakdown by link attribution (ALL points):")
    stats(same, "Bernstein link == true link")
    stats(mism, "wrong PROTECTED link wins min")
    stats(true_unprot, "true nearest is UNPROTECTED link")
    print("\nAmong the >30 deg error points:")
    bad = ang > 30
    print(f"  total >30deg: {bad.sum()} ({bad.mean()*100:.1f}% of all)")
    print(f"    of those, true nearest is UNPROTECTED link : {np.mean(true_unprot[bad])*100:.1f}%")
    print(f"    of those, wrong PROTECTED link wins the min: {np.mean(mism[bad])*100:.1f}%")
    print(f"    of those, Bernstein picked the RIGHT link  : {np.mean(same[bad])*100:.1f}%  "
          f"(=> genuine single-link gradient error)")
    # which links own the unprotected mismatches
    if true_unprot[bad].sum():
        u, c = np.unique(true_link[bad & true_unprot], return_counts=True)
        print("    unprotected culprits:", dict(zip(u, c.tolist())))


if __name__ == "__main__":
    main()
