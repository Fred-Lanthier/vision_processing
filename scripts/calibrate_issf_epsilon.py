#!/usr/bin/env python3
"""Run independent pick-place trials and identify the ISSf epsilon bound.

The bag for each valid trial intentionally contains only:

* /cbf_reflex/epsilon_sample -- controller-rate window maxima; and
* /pp_grasp/state            -- the RELEASED completion marker.

No figures or images are produced.  Epsilon is estimated only after all N
valid trajectories have been collected.  A 45 s task timeout remains a valid
safety-calibration trajectory; task completion is reported separately.  The
estimator applies an order statistic to independent *trajectory maxima*,
rather than treating correlated controller ticks as independent observations.
"""

import argparse
import csv
import datetime as _datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


EPSILON_TOPIC = "/cbf_reflex/epsilon_sample"
COMPLETION_TOPIC = "/pp_grasp/state"
START_TOPIC = EPSILON_TOPIC
MINIMAL_BAG_TOPICS = (EPSILON_TOPIC, COMPLETION_TOPIC)

REPRODUCIBILITY_PARAMETERS = (
    "/certified_cbf_velocity_controller/epsilon_sample_divisor",
    "/certified_cbf_velocity_controller/epsilon_normal_alpha_threshold",
    "/certified_cbf_velocity_controller/epsilon_motion_threshold",
    "/certified_cbf_velocity_controller/h_activate",
    "/certified_cbf_velocity_controller/issf_epsilon",
    "/certified_cbf_velocity_controller/issf_rho",
    "/cbf_safety_Bernstein/cbf_link_names",
    "/cbf_safety_Bernstein/cbf_h_activate",
    "/cbf_safety_Bernstein/cbf_issf_epsilon",
    "/cbf_safety_Bernstein/cbf_issf_rho",
    "/safe_generative_planner/fm_noise_seed",
)


def _json_dump(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _wall_duration_at_most_45(value):
    parsed = float(value)
    if not 0.0 < parsed <= 45.0:
        raise argparse.ArgumentTypeError("must be in the interval (0, 45]")
    return parsed


def _coverage(value):
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be strictly between 0 and 1")
    return parsed


def conformal_upper_order_statistic(values, target_coverage):
    """Return a trajectory-level, exchangeability-based upper estimate.

    With N calibration trajectories, the k-th order statistic has marginal
    rank coverage k/(N+1) for the maximum of a new exchangeable trajectory.
    If ceil((N+1)*target) is N+1, no finite distribution-free bound exists;
    the observed maximum is still returned, with its honest N/(N+1) rank.
    """
    cleaned = sorted(float(value) for value in values)
    if not cleaned:
        raise ValueError("at least one trajectory maximum is required")
    if not all(math.isfinite(value) and value >= 0.0 for value in cleaned):
        raise ValueError("trajectory maxima must be finite and non-negative")

    count = len(cleaned)
    requested_order = int(math.ceil((count + 1) * target_coverage))
    selected_order = min(requested_order, count)
    finite_target_bound_available = requested_order <= count
    return {
        "epsilon": cleaned[selected_order - 1],
        "sample_count": count,
        "requested_order": requested_order,
        "selected_order": selected_order,
        "target_coverage": float(target_coverage),
        "rank_coverage": selected_order / float(count + 1),
        "finite_target_bound_available": finite_target_bound_available,
        "minimum_runs_for_target": int(math.ceil(
            target_coverage / (1.0 - target_coverage))),
    }


def summarize_bag(bag_path):
    """Read only the two minimal calibration topics from one completed bag."""
    try:
        import rosbag
    except ImportError as exc:
        raise RuntimeError(
            "rosbag Python bindings are unavailable; source the ROS workspace"
        ) from exc

    normal_max = 0.0
    braking_max = 0.0
    normal_windows = 0
    braking_windows = 0
    epsilon_windows = 0
    states = []

    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, message, _ in bag.read_messages(
                topics=list(MINIMAL_BAG_TOPICS)):
            if topic == COMPLETION_TOPIC:
                states.append(str(message.data))
                continue
            epsilon_windows += 1
            if bool(message.has_normal):
                normal_windows += 1
                value = float(message.normal_adverse_projection)
                if value >= normal_max:
                    normal_max = value
            if bool(message.has_braking):
                braking_windows += 1
                value = float(message.braking_adverse_projection)
                if value >= braking_max:
                    braking_max = value

    return {
        "bag": str(Path(bag_path).resolve()),
        "complete": "RELEASED" in states,
        "grasp_states": states,
        "epsilon_windows": epsilon_windows,
        "normal_windows": normal_windows,
        "braking_windows": braking_windows,
        "normal_max": normal_max,
        "braking_max": braking_max,
        "global_max": max(normal_max, braking_max),
    }


def build_estimate(run_summaries, target_coverage):
    if not run_summaries:
        raise ValueError("no run summaries were supplied")
    normal = conformal_upper_order_statistic(
        [run["normal_max"] for run in run_summaries], target_coverage)
    braking = conformal_upper_order_statistic(
        [run["braking_max"] for run in run_summaries], target_coverage)
    global_bound = conformal_upper_order_statistic(
        [run["global_max"] for run in run_summaries], target_coverage)

    warnings = []
    if not global_bound["finite_target_bound_available"]:
        warnings.append(
            "The requested distribution-free coverage is impossible with "
            f"N={len(run_summaries)}. The reported epsilon is the observed "
            f"maximum and has rank coverage {global_bound['rank_coverage']:.6f}; "
            f"use at least {global_bound['minimum_runs_for_target']} runs for "
            f"target coverage {target_coverage:.6f}."
        )
    if all(run["braking_windows"] == 0 for run in run_summaries):
        warnings.append(
            "No braking windows were observed; the global estimate therefore "
            "characterizes normal execution only."
        )

    return {
        "method": "trajectory-maximum conformal order statistic",
        "assumption": (
            "The N seeded trajectories and the future trajectory are "
            "exchangeable; controller ticks within a trajectory are not "
            "treated as independent."
        ),
        "run_count": len(run_summaries),
        "task_completed_runs": sum(
            bool(run["complete"]) for run in run_summaries),
        "task_completion_rate": sum(
            bool(run["complete"]) for run in run_summaries)
            / float(len(run_summaries)),
        "velocity_measurement": (
            "ros_control JointHandle.getVelocity at tick k minus the applied "
            "command saved at tick k-1"
        ),
        "normal_execution": normal,
        "braking_execution": braking,
        "global_execution": global_bound,
        "recommended_cbf_issf_epsilon": (
            global_bound["epsilon"]
            if global_bound["finite_target_bound_available"] else None
        ),
        "observed_global_max": max(run["global_max"] for run in run_summaries),
        "warnings": warnings,
    }


def write_run_csv(path, run_summaries):
    fields = (
        "run", "seed", "complete", "capture_wall_seconds", "epsilon_windows",
        "normal_windows", "braking_windows",
        "normal_max", "braking_max", "global_max", "bag",
    )
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in run_summaries:
            writer.writerow({field: summary.get(field) for field in fields})


def _signal_process_group(process, process_signal):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, process_signal)
    except ProcessLookupError:
        pass


def _stop_process_group(process, interrupt_timeout=12.0):
    if process is None or process.poll() is not None:
        return
    _signal_process_group(process, signal.SIGINT)
    try:
        process.wait(timeout=interrupt_timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_process_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_process_group(process, signal.SIGKILL)
    process.wait(timeout=5.0)


def _topic_is_advertised(topic):
    result = subprocess.run(
        ["rostopic", "type", topic],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5.0,
        check=False,
    )
    return result.returncode == 0


def _wait_for_topics(topics, launch_process, timeout):
    deadline = time.monotonic() + timeout
    remaining = set(topics)
    while remaining and time.monotonic() < deadline:
        if launch_process.poll() is not None:
            return False, "roslaunch exited before calibration topics appeared"
        remaining = {topic for topic in remaining
                     if not _topic_is_advertised(topic)}
        if remaining:
            time.sleep(0.25)
    if remaining:
        return False, "timed out waiting for topics: " + ", ".join(
            sorted(remaining))
    return True, ""


def _wait_for_first_message(topic, launch_process, timeout):
    probe = subprocess.Popen(
        ["rostopic", "echo", "-n", "1", "--noarr", topic],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if probe.poll() is not None:
                return probe.returncode == 0
            if launch_process.poll() is not None:
                return False
            time.sleep(0.05)
        return False
    finally:
        _stop_process_group(probe, interrupt_timeout=1.0)


def _capture_until_release_or_deadline(watcher, launch_process, seconds):
    started = time.monotonic()
    deadline = started + seconds
    released = False
    reason = "wall_time_limit"
    while time.monotonic() < deadline:
        if launch_process.poll() is not None:
            reason = "roslaunch_exited"
            break
        if watcher.poll() is not None and watcher.returncode == 0:
            released = True
            reason = "released"
            break
        time.sleep(0.05)
    return {
        "released_live": released,
        "stop_reason": reason,
        "capture_wall_seconds": min(time.monotonic() - started, seconds),
    }


def _get_ros_parameters():
    parameters = {}
    try:
        import yaml
    except ImportError:
        yaml = None
    for name in REPRODUCIBILITY_PARAMETERS:
        result = subprocess.run(
            ["rosparam", "get", name],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5.0,
            check=False,
        )
        if result.returncode != 0:
            parameters[name] = None
        elif yaml is None:
            parameters[name] = result.stdout.strip()
        else:
            parameters[name] = yaml.safe_load(result.stdout)
    return parameters


def _run_one_attempt(args, attempt, seed, attempt_directory):
    attempt_directory.mkdir(parents=False, exist_ok=False)
    bag_path = attempt_directory / "epsilon.bag"
    launch_log_path = attempt_directory / "roslaunch.log"
    launch_output = (
        launch_log_path.open("w", encoding="utf-8")
        if args.save_launch_log else subprocess.DEVNULL
    )
    launch_arguments = [
        "gui:=false",
        "launch_rviz:=false",
        "use_cpp_certified_controller:=true",
        f"planner_fm_noise_seed:={seed}",
    ]
    launch_arguments.extend(args.launch_arg)
    launch_command = [
        "roslaunch", args.package, args.launch_file, *launch_arguments,
    ]

    launch_process = None
    bag_process = None
    state_watcher = None
    metadata = {
        "attempt": attempt,
        "seed": seed,
        "launch_command": launch_command,
        "minimal_bag_topics": list(MINIMAL_BAG_TOPICS),
        "max_active_wall_seconds": args.max_active_wall_seconds,
        "accepted": False,
    }
    try:
        launch_process = subprocess.Popen(
            launch_command,
            stdout=launch_output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ready, failure = _wait_for_topics(
            MINIMAL_BAG_TOPICS, launch_process,
            args.startup_timeout)
        if not ready:
            metadata["failure"] = failure
            return metadata, None

        bag_process = subprocess.Popen(
            [
                "rosbag", "record", "--lz4", "-O", str(bag_path),
                *MINIMAL_BAG_TOPICS,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        state_watcher = subprocess.Popen(
            [
                "rostopic", "echo", "-n", "1", "--filter",
                "m.data == 'RELEASED'", COMPLETION_TOPIC,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # The controller suppresses this topic until the first nonzero
        # certified motion command, so this is actual trajectory t=0 rather
        # than simulator/controller startup.
        time.sleep(0.25)
        if not _wait_for_first_message(
                START_TOPIC, launch_process, args.startup_timeout):
            metadata["failure"] = (
                "timed out waiting for the first calibration motion sample")
            return metadata, None

        metadata["trajectory_started_utc"] = (
            _datetime.datetime.now(_datetime.timezone.utc).isoformat())
        capture = _capture_until_release_or_deadline(
            state_watcher, launch_process, args.max_active_wall_seconds)
        metadata.update(capture)
        # Query while the trial's private parameters still exist.
        metadata["ros_parameters"] = _get_ros_parameters()
    except (OSError, subprocess.SubprocessError) as exc:
        metadata["failure"] = f"process error: {exc}"
    finally:
        _stop_process_group(state_watcher, interrupt_timeout=1.0)
        _stop_process_group(bag_process)
        _stop_process_group(launch_process, interrupt_timeout=20.0)
        if args.save_launch_log:
            launch_output.close()

    if not bag_path.exists():
        metadata["failure"] = "rosbag did not produce epsilon.bag"
        return metadata, None
    try:
        summary = summarize_bag(bag_path)
    except Exception as exc:  # Keep a failed bag for inspection and retry.
        metadata["failure"] = f"could not read calibration bag: {exc}"
        return metadata, None

    metadata["bag_summary"] = summary
    if summary["normal_windows"] + summary["braking_windows"] == 0:
        metadata["failure"] = "bag contains no active epsilon samples"
        return metadata, summary

    metadata["task_completed"] = summary["complete"]
    metadata["accepted"] = True
    return metadata, summary


def _parse_args(argv=None):
    timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Collect N independent, <=45 s pick-place trajectories in minimal "
            "bags, then estimate the projected ISSf disturbance bound once."
        )
    )
    parser.add_argument("-n", "--runs", required=True, type=_positive_int)
    parser.add_argument(
        "-o", "--output-dir",
        default=f"issf_epsilon_calibration_{timestamp}", type=Path)
    parser.add_argument(
        "--max-active-wall-seconds", type=_wall_duration_at_most_45,
        default=45.0,
        help="wall-clock cap measured from the first nonzero motion (max 45)")
    parser.add_argument(
        "--coverage", type=_coverage, default=0.95,
        help="target trajectory-level conformal rank coverage (default: 0.95)")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument(
        "--max-attempts", type=_positive_int, default=None,
        help="stop after this many attempts; default is 2*N")
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--package", default="vision_processing")
    parser.add_argument(
        "--launch-file", default="green_cube_feeding_casf_pp.launch")
    parser.add_argument(
        "--launch-arg", action="append", default=[], metavar="NAME:=VALUE",
        help="additional roslaunch argument; may be repeated")
    parser.add_argument(
        "--save-launch-log", action="store_true",
        help="save roslaunch stdout for debugging (off keeps output minimal)")
    args = parser.parse_args(argv)
    if args.startup_timeout <= 0.0:
        parser.error("--startup-timeout must be positive")
    if args.max_attempts is None:
        args.max_attempts = 2 * args.runs
    if args.max_attempts < args.runs:
        parser.error("--max-attempts cannot be smaller than --runs")
    for launch_arg in args.launch_arg:
        if ":=" not in launch_arg:
            parser.error(f"invalid --launch-arg {launch_arg!r}; use NAME:=VALUE")
    return args


def main(argv=None):
    args = _parse_args(argv)
    output_directory = args.output_dir.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        print(
            f"Refusing to overwrite non-empty directory: {output_directory}",
            file=sys.stderr,
        )
        return 2
    output_directory.mkdir(parents=True, exist_ok=True)

    configuration = {
        "created_utc": _datetime.datetime.now(
            _datetime.timezone.utc).isoformat(),
        "requested_runs": args.runs,
        "max_attempts": args.max_attempts,
        "max_active_wall_seconds": args.max_active_wall_seconds,
        "target_coverage": args.coverage,
        "seed_start": args.seed_start,
        "package": args.package,
        "launch_file": args.launch_file,
        "launch_arguments": list(args.launch_arg),
        "minimal_bag_topics": list(MINIMAL_BAG_TOPICS),
        "epsilon_estimator_version": 2,
        "velocity_measurement": (
            "ros_control JointHandle.getVelocity at tick k minus the applied "
            "command saved at tick k-1"
        ),
        "task_timeout_is_valid_calibration_data": True,
        "estimation_deferred_until_all_runs_exist": True,
    }
    _json_dump(output_directory / "batch_config.json", configuration)

    accepted = []
    for attempt in range(1, args.max_attempts + 1):
        if len(accepted) >= args.runs:
            break
        seed = args.seed_start + attempt - 1
        temporary_directory = output_directory / f"attempt_{attempt:03d}"
        print(
            f"[{len(accepted) + 1}/{args.runs}] attempt {attempt}/"
            f"{args.max_attempts}, seed={seed}",
            flush=True,
        )
        metadata, summary = _run_one_attempt(
            args, attempt, seed, temporary_directory)
        metadata["finished_utc"] = _datetime.datetime.now(
            _datetime.timezone.utc).isoformat()

        if metadata["accepted"]:
            run_number = len(accepted) + 1
            final_directory = output_directory / (
                f"run_{run_number:03d}_seed_{seed}")
            temporary_directory.replace(final_directory)
            summary["bag"] = str((final_directory / "epsilon.bag").resolve())
            summary["run"] = run_number
            summary["seed"] = seed
            summary["capture_wall_seconds"] = metadata[
                "capture_wall_seconds"]
            metadata["bag_summary"] = summary
            _json_dump(final_directory / "metadata.json", metadata)
            accepted.append(summary)
            task_status = "RELEASED" if summary["complete"] else "TIMEOUT"
            print(
                f"  valid ({task_status}): {summary['epsilon_windows']} windows, "
                f"global max={summary['global_max']:.6f} rad/s, "
                f"wall={summary['capture_wall_seconds']:.2f} s",
                flush=True,
            )
        else:
            failed_directory = output_directory / (
                f"failed_attempt_{attempt:03d}_seed_{seed}")
            temporary_directory.replace(failed_directory)
            _json_dump(failed_directory / "metadata.json", metadata)
            print(f"  rejected: {metadata.get('failure', 'unknown failure')}",
                  flush=True)

    if len(accepted) != args.runs:
        incomplete = {
            "status": "incomplete",
            "valid_runs": len(accepted),
            "requested_runs": args.runs,
            "epsilon_computed": False,
            "reason": "maximum attempt count reached",
        }
        _json_dump(output_directory / "INCOMPLETE.json", incomplete)
        print(
            f"Collected {len(accepted)}/{args.runs} valid trajectories. "
            "Epsilon was not computed.",
            file=sys.stderr,
        )
        return 1

    # Deliberately the first and only batch-estimation point: all N independent
    # trajectories are now present and immutable on disk.
    estimate = build_estimate(accepted, args.coverage)
    estimate["completed_utc"] = _datetime.datetime.now(
        _datetime.timezone.utc).isoformat()
    estimate["runs"] = accepted
    _json_dump(output_directory / "epsilon_calibration.json", estimate)
    write_run_csv(output_directory / "epsilon_run_maxima.csv", accepted)

    global_result = estimate["global_execution"]
    print(
        f"Completed {args.runs} valid trajectories; "
        f"task success {estimate['task_completed_runs']}/{args.runs}.")
    print(
        f"Global projected epsilon estimate: "
        f"{global_result['epsilon']:.6f} rad/s "
        f"(rank coverage {global_result['rank_coverage']:.6f})")
    if estimate["recommended_cbf_issf_epsilon"] is None:
        print(estimate["warnings"][0], file=sys.stderr)
    print(f"Report: {output_directory / 'epsilon_calibration.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
