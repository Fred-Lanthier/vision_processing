import importlib.util
import pathlib

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot_diagnostics = _load(
    "plot_cbf_diagnostics", "scripts/plot_cbf_diagnostics.py")
real_distance = _load("real_distance", "scripts/real_distance.py")


def test_asynchronous_finger_samples_are_interpolated_without_closed_padding():
    target = np.array([0.0, 0.5, 1.0])
    source = np.array([0.0, 0.5, 0.5, 1.0])
    # The second sample at t=0.5 is the newest message and must win.
    fingers = np.array([
        [0.040, 0.040],
        [0.035, 0.035],
        [0.030, 0.031],
        [0.020, 0.021],
    ])

    actual = plot_diagnostics._interpolate_joint_samples(
        target, source, fingers)

    np.testing.assert_allclose(
        actual,
        [[0.040, 0.040], [0.030, 0.031], [0.020, 0.021]],
        atol=1e-12,
    )


class _FingerAwareLayer:
    """Tiny FK stand-in whose two meshes translate with the finger columns."""

    def __init__(self):
        self.last_q = None

    def get_transformations_each_link(self, pose, q):
        self.last_q = q.detach().clone()
        batch = q.shape[0]
        left = torch.eye(4).expand(batch, 4, 4).clone()
        right = torch.eye(4).expand(batch, 4, 4).clone()
        left[:, 1, 3] = q[:, 7]
        right[:, 1, 3] = -q[:, 8]
        return [left, right]


def _fake_clearance():
    rc = real_distance.RealClearance.__new__(real_distance.RealClearance)
    rc.torch = torch
    rc.device = torch.device("cpu")
    rc.joint_names = list(real_distance.PANDA_JOINTS)
    rc.layer = _FingerAwareLayer()
    rc.mesh_idx = [0, 1]
    rc.local_pts = [
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
    ]
    return rc


def test_surface_points_follow_both_measured_finger_joints():
    rc = _fake_clearance()
    q = np.zeros((2, 9), dtype=np.float32)
    q[0, 7:] = [0.040, 0.040]
    q[1, 7:] = [0.020, 0.025]

    points = rc.surface_points(q)

    np.testing.assert_allclose(points[:, 0, 1], [0.040, 0.020])
    np.testing.assert_allclose(points[:, 1, 1], [-0.040, -0.025])
    np.testing.assert_array_equal(rc.layer.last_q.numpy(), q)


def test_surface_points_reject_arm_only_input_instead_of_closing_fingers():
    rc = _fake_clearance()

    with np.testing.assert_raises_regex(
            ValueError, "Both measured finger joints"):
        rc.surface_points(np.zeros((3, 7), dtype=np.float32))
