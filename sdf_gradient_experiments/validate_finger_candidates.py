#!/usr/bin/env python3
"""Accuracy A/B: production (n_func=8) vs candidate (n_func=12) finger SDFs.

Everything happens in the link/mesh frame (no FK): sample genuine exterior
points in a 2-40 mm shell around the finger mesh, evaluate both models along
the deployed in-box path (u=(p-c)/s, t=(u+1)/2, f=s*P(t), grad=dP/dt/2), and
compare against the exact mesh distance and outward direction.

Safety-relevant sign convention: f - d_true > 0 means the model OVER-reports
clearance (under-braking risk); negative errors are the safe side.

    venv_sam3/bin/python3 sdf_gradient_experiments/validate_finger_candidates.py
"""
import importlib.util
import os
import sys

import numpy as np
import torch
import trimesh

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANDA_TEST = os.path.join(PKG, 'third_party/SDF_Bernstein_Basis/panda_test')

_spec = importlib.util.spec_from_file_location(
    'analytical_bernstein',
    os.path.join(PKG, 'src', 'vision_processing', 'analytical_bernstein.py'))
_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ab)

LINKS = ['panda_leftfinger', 'panda_rightfinger']
IN_BOX = 0.98        # |u| cutoff: strictly inside the deployed clamp box


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def eval_model(model, points):
    """Deployed in-box path: values [B] and link-frame gradients [B,3]."""
    n = int(model['n_func'])
    c = torch.as_tensor(np.asarray(model['centroid_offset']),
                        dtype=torch.float64)
    s = float(np.asarray(model['scale_factor']))
    w = torch.as_tensor(np.asarray(model['weights']),
                        dtype=torch.float64).reshape(1, n, n, n)
    u = (points - c) / s
    t = (0.5 * (u + 1.0)).reshape(1, 1, -1, 3)
    value, grad_t = _ab.tensor_product_bernstein_value_gradient(t, w)
    return s * value.reshape(-1), 0.5 * grad_t.reshape(-1, 3), u


def stats(name, f, grad, dist, gex):
    err = (f - dist).numpy() * 1000.0                       # mm, + = over-report
    gn = torch.linalg.vector_norm(grad, dim=1)
    gu = (grad / gn[:, None].clamp_min(1e-9)).numpy()
    ang = np.degrees(np.arccos(np.clip((gu * gex).sum(axis=1), -1, 1)))
    print(f"  {name:<22} MAE={np.abs(err).mean():5.2f}  bias={err.mean():+5.2f}  "
          f"p99|e|={np.percentile(np.abs(err), 99):5.2f}  "
          f"p99 over={np.percentile(err, 99):+5.2f}  max over={err.max():+5.2f}  | "
          f"ang mean={ang.mean():4.1f} p95={np.percentile(ang, 95):4.1f} deg  "
          f"|g| med={np.median(gn.numpy()):.2f}")


def main():
    rng = np.random.default_rng(0)
    for link in LINKS:
        mesh = trimesh.load(os.path.join(PANDA_TEST, 'Meshes', f'{link}.stl'),
                            force='mesh')
        prod = load(os.path.join(PANDA_TEST, 'Models', f'{link}_w.pt'))
        cand = load(os.path.join(PANDA_TEST, 'Models', 'nfunc12_candidate',
                                 f'{link}_w.pt'))

        pts, fi = trimesh.sample.sample_surface(mesh, 40000, seed=1)
        off = rng.uniform(0.002, 0.04, size=(len(pts), 1))
        P = pts + mesh.face_normals[fi] * off
        pq = trimesh.proximity.ProximityQuery(mesh)
        closest, dist, tri = pq.on_surface(P)
        disp = P - closest
        fn = mesh.face_normals[tri]
        keep = (np.einsum('ij,ij->i', disp, fn) > 0) \
            & (np.abs(dist - off[:, 0]) < 0.002)
        P, dist, disp = P[keep], dist[keep], disp[keep]
        gex = disp / np.clip(np.linalg.norm(disp, axis=1, keepdims=True),
                             1e-9, None)

        points = torch.from_numpy(P).to(torch.float64)
        d_true = torch.from_numpy(dist).to(torch.float64)
        f_p, g_p, u = eval_model(prod, points)
        f_c, g_c, _ = eval_model(cand, points)
        inside = (u.abs().max(dim=1).values <= IN_BOX)
        near = inside & (d_true <= 0.02)

        for label, sel in (('all in-box (2-40 mm)', inside),
                           ('near (<= 20 mm)', near)):
            print(f"\n{link} — {label}: {int(sel.sum())} pts")
            stats('production n=8', f_p[sel], g_p[sel], d_true[sel], gex[sel])
            stats('candidate  n=12', f_c[sel], g_c[sel], d_true[sel], gex[sel])


if __name__ == '__main__':
    main()
