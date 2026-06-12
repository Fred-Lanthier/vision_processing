#!/usr/bin/env python3
"""Verify the flat Bernstein timing: cost breakdown + scaling to large batches."""
import os, sys, time
import numpy as np
import torch

SDF_BERNSTEIN = "/home/flanthier/Github/src/vision_processing/third_party/SDF_Bernstein_Basis"
sys.path.insert(0, os.path.join(SDF_BERNSTEIN, "..", "..")); sys.path.insert(0, SDF_BERNSTEIN)
os.chdir(SDF_BERNSTEIN)
import visualize_robot_sdf_layers as V


def time_ms(fn, reps=40, warm=5):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter(); fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return np.median(ts)


def main():
    dev = torch.device("cuda")
    links = V.CBF_PROTECTED_LINKS
    rl, w, core = V.build_sdf_stack(dev, links)
    pose = torch.eye(4, device=dev).unsqueeze(0)
    q9 = V.q7_to_q9(V.DEFAULT_Q, 0.001, dev)

    rng = np.random.default_rng(0)
    big = rng.uniform(-0.5, 0.8, size=(2 ** 20, 3)).astype(np.float32)

    print(f"{'N':>8} {'full[ms]':>9} {'fwd_only[ms]':>12} {'fk_only[ms]':>11}")
    for n in [128, 256, 512, 1024, 4096, 16384, 65536, 262144, 1048576]:
        pts = big[:n]
        full = time_ms(lambda: V.evaluate_sdf_and_grad(core, pts, pose, q9, 2 ** 21, dev))

        x_gpu = torch.as_tensor(pts, device=dev)
        def fwd():
            with torch.no_grad():
                core.get_whole_body_sdf_batch(x_gpu, pose, q9)
        fwd_t = time_ms(fwd)
        fk_t = time_ms(lambda: rl.get_transformations_each_link(pose, q9))
        print(f"{n:>8} {full:9.2f} {fwd_t:12.2f} {fk_t:11.2f}")


if __name__ == "__main__":
    main()
