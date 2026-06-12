#!/usr/bin/env python3
"""Find the worst distance-error points and show WHY (per-link SDF values, offset,
exact vs bernstein, box-coordinate), to tell measurement artifact from real error."""
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

P, lab, exact = [], [], []
for name, m in zip(links, meshes):
    rp, gd, gg, _ = V.make_external_benchmark_rays([m], meshes, 20, offsets, 123)
    P.append(rp.reshape(-1, 3).astype(np.float32)); exact.append(gd.reshape(-1)); lab += [name] * rp.reshape(-1,3).shape[0]
P = np.vstack(P); exact = np.concatenate(exact); lab = np.array(lab)

xt = torch.tensor(P, device=dev)
with torch.no_grad():
    val, per = core.get_whole_body_sdf_batch(xt, pose, q9, return_per_link=True)
val = val[0].cpu().numpy(); per = per[0].cpu().numpy()   # per: [K, N]
err = val - exact

order = np.argsort(np.abs(err))[::-1][:20]
print(f"{'link':12s} {'exact':>7s} {'bern':>7s} {'err':>7s}  per-link SDF [mm] (min wins)")
for i in order:
    pl = "  ".join(f"{links[k].replace('panda_','')[:6]}:{per[k,i]*1000:6.1f}" for k in range(len(links)))
    print(f"{lab[i]:12s} {exact[i]*1000:7.1f} {val[i]*1000:7.1f} {err[i]*1000:7.1f}  {pl}")

print(f"\nOverall: MAE={np.mean(np.abs(err))*1000:.2f}mm  "
      f">10mm err: {np.mean(np.abs(err)>0.010)*100:.2f}%  >20mm: {np.mean(np.abs(err)>0.020)*100:.2f}%")
# are big errors over- or under-reads, and on which links?
big = np.abs(err) > 0.010
if big.sum():
    u, c = np.unique(lab[big], return_counts=True)
    print("links with >10mm err:", dict(zip([x.replace('panda_','') for x in u], c.tolist())))
    print(f"of >10mm errors: {np.mean(err[big]>0)*100:.0f}% are OVER-reads (bern>exact)")
