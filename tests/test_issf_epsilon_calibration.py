import importlib.util
import math
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "calibrate_issf_epsilon",
    ROOT / "scripts" / "calibrate_issf_epsilon.py",
)
CALIBRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALIBRATION)


def test_nineteen_runs_support_a_finite_95_percent_rank_bound():
    result = CALIBRATION.conformal_upper_order_statistic(
        range(1, 20), 0.95)

    assert result["epsilon"] == 19.0
    assert result["selected_order"] == 19
    assert result["rank_coverage"] == 0.95
    assert result["finite_target_bound_available"]
    assert result["minimum_runs_for_target"] == 19


def test_too_few_runs_reports_the_achievable_coverage_honestly():
    result = CALIBRATION.conformal_upper_order_statistic(
        range(1, 19), 0.95)

    assert result["epsilon"] == 18.0
    assert result["requested_order"] == 19
    assert result["selected_order"] == 18
    assert result["rank_coverage"] == 18.0 / 19.0
    assert not result["finite_target_bound_available"]


def test_batch_estimate_uses_one_maximum_per_trajectory():
    runs = []
    for index in range(19):
        runs.append({
            "normal_max": 0.01 + index * 0.001,
            "braking_max": 0.02 if index == 3 else 0.0,
            "global_max": max(0.01 + index * 0.001,
                              0.02 if index == 3 else 0.0),
            "braking_windows": 1 if index == 3 else 0,
            "complete": index < 15,
        })

    estimate = CALIBRATION.build_estimate(runs, 0.95)

    assert estimate["run_count"] == 19
    assert math.isclose(estimate["global_execution"]["epsilon"], 0.028)
    assert math.isclose(estimate["recommended_cbf_issf_epsilon"], 0.028)
    assert estimate["task_completed_runs"] == 15
    assert math.isclose(estimate["task_completion_rate"], 15.0 / 19.0)
    assert estimate["warnings"] == []


def test_braking_absence_is_explicit_in_report():
    runs = [{
        "normal_max": 0.01,
        "braking_max": 0.0,
        "global_max": 0.01,
        "braking_windows": 0,
        "complete": True,
    } for _ in range(19)]

    estimate = CALIBRATION.build_estimate(runs, 0.95)

    assert any("No braking windows" in item for item in estimate["warnings"])
