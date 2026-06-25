#!/usr/bin/env python3
"""Benchmark analytical and autograd Bernstein-RDF joint derivatives.

For L=1,...,8 protected geometries this script compares:

1. analytical_all_rows: explicit derivatives for all L independent link
   Softmin constraints, returning an [L,7] QP Jacobian;
2. autograd_all_rows: the identical [L,7] Jacobian using a batched VJP.

Run this benchmark without the ROS launch running if isolated kernel/algorithm
timings are desired.  Other CUDA processes create legitimate system-contention
latency but make derivative implementations difficult to compare.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import warnings

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PROTECTED_LINKS = [
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_hand",
    "panda_rightfinger",
    "panda_leftfinger",
    "fork_tip",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.005)
    parser.add_argument("--d-safe", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "sdf_gradient_experiments" / "analytical_vs_autograd.csv",
    )
    return parser.parse_args()


def load_analytical_module():
    path = ROOT / "src" / "vision_processing" / "analytical_bernstein.py"
    spec = importlib.util.spec_from_file_location("analytical_bernstein", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(wall_times):
    return {
        "wall_mean_ms": statistics.fmean(wall_times),
        "wall_std_ms": statistics.stdev(wall_times) if len(wall_times) > 1 else 0.0,
        "wall_median_ms": statistics.median(wall_times),
        "wall_p95_ms": percentile(wall_times, 95),
    }


def measure(callable_, warmup, repeats):
    """Measure synchronized end-to-end call latency."""
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()

    wall_times = []
    for _ in range(repeats):
        # Drain earlier work so each sample measures this call, not an unrelated
        # queue backlog.  The synchronization before the timer is not included.
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        callable_()
        torch.cuda.synchronize()
        wall_times.append((time.perf_counter() - wall_start) * 1000.0)
    return summarize(wall_times)


def per_link_softmin(sdf_per_link, temperature, d_safe):
    minimum = sdf_per_link.min(dim=-1, keepdim=True).values
    exponentials = torch.exp(-(sdf_per_link - minimum) / temperature)
    return (
        minimum.squeeze(-1)
        - temperature * torch.log(exponentials.sum(dim=-1))
        - d_safe
    )


def make_query_points(full_core, q, pose, point_count, seed):
    """Create one deterministic cloud distributed across all eight visual frames."""
    generator = torch.Generator(device=q.device).manual_seed(seed)
    transforms = full_core._stack_used_link_transforms(pose, q)[0]
    points = []
    for point_index in range(point_count):
        link_index = point_index % len(PROTECTED_LINKS)
        # Stay inside the polynomial domain so the benchmark measures Bernstein
        # evaluation rather than mostly the inexpensive outside-box residual.
        normalized = 1.4 * torch.rand(
            (3,), generator=generator, device=q.device, dtype=q.dtype) - 0.7
        local = (
            full_core.offsets[link_index]
            + full_core.scales[link_index] * normalized
        )
        rotation = transforms[link_index, :3, :3]
        origin = transforms[link_index, :3, 3]
        points.append(rotation @ local + origin)
    return torch.stack(points, dim=0).contiguous()


def format_metric(metric):
    return (
        f"{metric['wall_mean_ms']:8.3f} +/- "
        f"{metric['wall_std_ms']:6.3f}"
    )


def main():
    args = parse_args()
    if args.points < 1 or args.warmup < 0 or args.repeats < 1:
        raise ValueError("points/repeats must be positive and warmup non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    # Import the ROS/RDF dependencies only after argument validation, allowing
    # --help to work in lightweight environments.
    sys.path.insert(0, str(ROOT))
    import xacro
    from third_party.RDF.urdf_layer import URDFLayer
    from third_party.SDF_Bernstein_Basis.src.rdf_weights import RDF_Weights
    from third_party.SDF_Bernstein_Basis.bernstein_core import BernsteinCore

    analytical_module = load_analytical_module()
    AnalyticalBernsteinSoftmin = analytical_module.AnalyticalBernsteinSoftmin

    device = torch.device("cuda")
    dtype = torch.float32
    print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(
        f"points={args.points} warmup={args.warmup} repeats={args.repeats} "
        f"temperature={args.temperature:g}"
    )

    xacro_path = ROOT / "urdf" / "panda_camera.xacro"
    document = xacro.process_file(str(xacro_path))
    temporary_urdf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", delete=False)
    try:
        temporary_urdf.write(document.toxml())
        temporary_urdf.close()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            robot = URDFLayer(
                urdf_path=temporary_urdf.name,
                device=device,
                package_dir=str(ROOT),
                voxel_dir=str(
                    ROOT / "third_party" / "RDF" / "panda_layer" / "meshes" / "voxel_128"),
            )
    finally:
        try:
            os.unlink(temporary_urdf.name)
        except OSError:
            pass

    weight_handler = RDF_Weights(device=device, dtype=dtype)
    weight_handler.init_robot_folder(
        str(ROOT / "third_party" / "SDF_Bernstein_Basis" / "panda_test"),
        robot_name="panda",
    )
    weight_handler.add_models(PROTECTED_LINKS, robot_name="panda")

    q_value = torch.tensor(
        [[0.0, -0.5, 0.1, -2.0, 0.2, 1.5, 0.8]],
        device=device,
        dtype=dtype,
    )
    q_autograd = q_value.clone().requires_grad_(True)
    pose = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)

    # One full core is used only to create a common point cloud.  Every link-count
    # benchmark then creates the exact subset core used by both derivative paths.
    full_core = BernsteinCore(
        weight_handler, robot, device, PROTECTED_LINKS)
    points = make_query_points(full_core, q_value, pose, args.points, args.seed)
    del full_core
    torch.cuda.empty_cache()

    rows = []
    print()
    print("Times are mean +/- std wall milliseconds")
    print(
        "links | analytical all rows | autograd all rows | grad max error"
    )
    print("-" * 98)

    for link_count in range(1, len(PROTECTED_LINKS) + 1):
        links = PROTECTED_LINKS[:link_count]
        core = BernsteinCore(weight_handler, robot, device, links)
        analytical = AnalyticalBernsteinSoftmin(
            core, temperature=args.temperature, d_safe=args.d_safe)
        vjp_eye = torch.eye(link_count, device=device, dtype=dtype)

        def analytical_all_rows():
            return analytical.evaluate(q_value, points)

        def autograd_quantities():
            _, sdf_per_link = core.get_whole_body_sdf_batch(
                points, pose, q_autograd, return_per_link=True)
            return per_link_softmin(
                sdf_per_link, args.temperature, args.d_safe)[0]

        def autograd_all_rows():
            h_per_link = autograd_quantities()
            gradient_batch = torch.autograd.grad(
                outputs=h_per_link,
                inputs=q_autograd,
                grad_outputs=vjp_eye,
                is_grads_batched=True,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]
            return h_per_link.detach(), gradient_batch[:, 0, :].detach()

        # Correctness is checked before timing.  The analytical path and batched
        # autograd must return the same independent constraints and Jacobian rows.
        analytical_result = analytical_all_rows()
        autograd_h, autograd_gradient = autograd_all_rows()
        value_error = float(
            (analytical_result.h[0] - autograd_h).abs().max().detach().cpu())
        gradient_error = float(
            (analytical_result.grad_q[0] - autograd_gradient)
            .abs().max().detach().cpu())

        analytical_timing = measure(
            analytical_all_rows, args.warmup, args.repeats)
        autograd_rows_timing = measure(
            autograd_all_rows, args.warmup, args.repeats)

        row = {
            "links": link_count,
            "link_names": " ".join(links),
            "points": args.points,
            "value_max_abs_error": value_error,
            "gradient_max_abs_error": gradient_error,
        }
        for prefix, timing in (
            ("analytical", analytical_timing),
            ("autograd_all_rows", autograd_rows_timing),
        ):
            for key, value in timing.items():
                row[f"{prefix}_{key}"] = value
        row["all_rows_wall_speedup_autograd_over_analytical"] = (
            autograd_rows_timing["wall_median_ms"]
            / analytical_timing["wall_median_ms"]
        )
        rows.append(row)

        print(
            f"{link_count:5d} | {format_metric(analytical_timing)} | "
            f"{format_metric(autograd_rows_timing)} | "
            f"{gradient_error:.3e}"
        )

        del analytical, core
        torch.cuda.empty_cache()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written to {args.csv}")
    print(
        "analytical and autograd_all_rows return the equivalent Lx7 Jacobian."
    )


if __name__ == "__main__":
    main()
