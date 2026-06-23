import importlib.util
import pathlib
from types import SimpleNamespace

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "vision_processing" / "analytical_bernstein.py"
SPEC = importlib.util.spec_from_file_location("analytical_bernstein", MODULE_PATH)
analytical_bernstein = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analytical_bernstein)

AnalyticalBernsteinSoftmin = analytical_bernstein.AnalyticalBernsteinSoftmin
bernstein_basis_and_derivative = analytical_bernstein.bernstein_basis_and_derivative
softmin_value_gradient = analytical_bernstein.softmin_value_gradient
tensor_product_bernstein_value_gradient = (
    analytical_bernstein.tensor_product_bernstein_value_gradient)
project_halfspaces_sequential = analytical_bernstein.project_halfspaces_sequential


def test_bernstein_derivative_matches_centered_difference():
    dtype = torch.float64
    t = torch.tensor([0.0, 0.13, 0.51, 0.91, 1.0], dtype=dtype)
    _, derivative = bernstein_basis_and_derivative(t, n_functions=6)

    eps = 1.0e-7
    plus, _ = bernstein_basis_and_derivative(t + eps, n_functions=6)
    minus, _ = bernstein_basis_and_derivative(t - eps, n_functions=6)
    finite_difference = (plus - minus) / (2.0 * eps)
    torch.testing.assert_close(derivative, finite_difference, rtol=2e-8, atol=2e-8)


def test_tensor_product_and_softmin_gradients_match_finite_difference():
    generator = torch.Generator().manual_seed(4)
    dtype = torch.float64
    t = 0.1 + 0.8 * torch.rand((1, 2, 5, 3), generator=generator, dtype=dtype)
    weights = torch.randn((2, 4, 4, 4), generator=generator, dtype=dtype)
    value, gradient_t = tensor_product_bernstein_value_gradient(t, weights)

    eps = 1.0e-7
    for axis in range(3):
        delta = torch.zeros_like(t)
        delta[..., axis] = eps
        plus = tensor_product_bernstein_value_gradient(t + delta, weights)[0]
        minus = tensor_product_bernstein_value_gradient(t - delta, weights)[0]
        finite_difference = (plus - minus) / (2.0 * eps)
        torch.testing.assert_close(
            gradient_t[..., axis], finite_difference, rtol=3e-8, atol=3e-8)

    point_gradient = torch.randn((1, 2, 5, 7), generator=generator, dtype=dtype)
    h, grad_h, probabilities = softmin_value_gradient(
        value, point_gradient, temperature=0.07)
    expected = (probabilities[..., None] * point_gradient).sum(dim=-2)
    torch.testing.assert_close(grad_h, expected)
    assert h.shape == (1, 2)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones((1, 2), dtype=dtype))


def test_softmin_point_mask_partitions_constraint_rows():
    dtype = torch.float64
    values = torch.tensor(
        [[[0.1, 0.2, -9.0, -8.0], [-7.0, -6.0, 0.3, 0.5]]], dtype=dtype)
    gradients = torch.arange(16, dtype=dtype).reshape(1, 2, 4, 2)
    mask = torch.tensor(
        [[True, True, False, False], [False, False, True, True]])

    h, grad_h, weights = softmin_value_gradient(
        values, gradients, temperature=0.05, point_mask=mask)
    expected_h_0, expected_grad_0, _ = softmin_value_gradient(
        values[:, :1, :2], gradients[:, :1, :2], temperature=0.05)
    expected_h_1, expected_grad_1, _ = softmin_value_gradient(
        values[:, 1:, 2:], gradients[:, 1:, 2:], temperature=0.05)

    torch.testing.assert_close(h[:, :1], expected_h_0)
    torch.testing.assert_close(h[:, 1:], expected_h_1)
    torch.testing.assert_close(grad_h[:, :1], expected_grad_0)
    torch.testing.assert_close(grad_h[:, 1:], expected_grad_1)
    assert torch.count_nonzero(weights[..., 2:][0, 0]) == 0
    assert torch.count_nonzero(weights[..., :2][0, 1]) == 0


def _fake_core(dtype=torch.float64):
    device = torch.device("cpu")
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
    skew = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, -0.0], [-0.0, 0.0, 0.0]],
        dtype=dtype,
    )
    offset = torch.eye(4, dtype=dtype)
    offset[0, 3] = 0.25
    robot = SimpleNamespace(
        dof=1,
        kinematic_tree=[{
            "child_link": "link1",
            "parent_link": "base",
            "type": "revolute",
            "offset": offset,
            "axis": axis,
            "K": skew,
            "K_sq": skew @ skew,
            "idx": 0,
        }],
    )

    visual_offset = torch.eye(4, dtype=dtype)
    visual_offset[0, 3] = 0.35
    mesh_info = {"link_name": "link1", "visual_offset": visual_offset}
    generator = torch.Generator().manual_seed(11)
    n_functions = 4
    weights = torch.randn((1, n_functions ** 3), generator=generator, dtype=dtype) * 0.05
    group = {
        "n_func": n_functions,
        "indices": [0],
        "indices_tensor": torch.tensor([0]),
        "offsets": torch.zeros((1, 3), dtype=dtype),
        "scales": torch.tensor([0.8], dtype=dtype),
        "weights": weights,
    }
    return SimpleNamespace(
        robot=robot,
        device=device,
        K=1,
        _link_mesh_infos=[mesh_info],
        groups={n_functions: group},
    )


def test_full_kinematic_softmin_gradient_matches_centered_difference_without_autograd():
    evaluator = AnalyticalBernsteinSoftmin(
        _fake_core(), temperature=0.08, d_safe=0.015)
    q = torch.tensor([[0.37]], dtype=torch.float64)
    points = torch.tensor(
        [[0.71, 0.18, 0.05], [0.62, -0.11, -0.09], [0.80, 0.04, 0.12]],
        dtype=torch.float64,
    )

    result = evaluator.evaluate(q, points)
    assert not result.h.requires_grad
    assert not result.grad_q.requires_grad

    eps = 1.0e-6
    h_plus = evaluator.evaluate(q + eps, points).h
    h_minus = evaluator.evaluate(q - eps, points).h
    finite_difference = (h_plus - h_minus) / (2.0 * eps)
    torch.testing.assert_close(
        result.grad_q[..., 0], finite_difference, rtol=2e-6, atol=2e-7)


def test_optional_grasp_cloud_gradient_is_analytical_too():
    core = _fake_core()
    core.grasp_attach_link = "link1"
    core.grasp_points = torch.tensor(
        [[0.18, 0.02, 0.0], [0.24, -0.03, 0.04]], dtype=torch.float64)
    core.grasp_active = torch.tensor(1.0, dtype=torch.float64)
    core.grasp_radius = torch.tensor(0.01, dtype=torch.float64)
    core.grasp_softmin_beta = 0.025
    core.grasp_dsafe_offset = -0.005
    evaluator = AnalyticalBernsteinSoftmin(core, temperature=0.06, d_safe=0.015)
    q = torch.tensor([[-0.24]], dtype=torch.float64)
    points = torch.tensor(
        [[0.72, 0.16, 0.01], [0.63, -0.08, 0.05], [0.77, 0.02, -0.02]],
        dtype=torch.float64,
    )

    result = evaluator.evaluate(q, points)
    assert result.h.shape == (1, 2)  # one RDF link plus the rigid grasp cloud
    eps = 1.0e-6
    finite_difference = (
        evaluator.evaluate(q + eps, points).h
        - evaluator.evaluate(q - eps, points).h
    ) / (2.0 * eps)
    torch.testing.assert_close(
        result.grad_q[..., 0], finite_difference, rtol=2e-6, atol=2e-7)


def test_on_device_halfspace_projection_matches_previous_numpy_algorithm():
    import numpy as np

    dtype = torch.float64
    gradients = torch.tensor(
        [[1.0, -0.2, 0.4], [-0.3, 0.8, 0.1], [0.2, 0.1, -0.9]],
        dtype=dtype,
    )
    bounds = torch.tensor([0.25, -0.04, 0.12], dtype=dtype)
    dq_nominal = torch.tensor([-0.4, 0.15, 0.3], dtype=dtype)
    iterations = 4
    relaxation = 0.85
    max_velocity = 0.35

    actual = project_halfspaces_sequential(
        dq_nominal,
        gradients,
        bounds,
        iterations=iterations,
        relaxation=relaxation,
        max_velocity=max_velocity,
    )

    # Exact reference for the NumPy implementation removed from the ROS node.
    g_np = gradients.numpy()
    b_np = bounds.numpy()
    expected = dq_nominal.numpy().copy()
    row_norm_squared = np.einsum("ij,ij->i", g_np, g_np) + 1.0e-6
    for _ in range(iterations):
        updated = False
        for row_index in range(g_np.shape[0]):
            violation = b_np[row_index] - g_np[row_index].dot(expected)
            if violation > 0.0:
                expected += (
                    relaxation
                    * violation
                    * g_np[row_index]
                    / row_norm_squared[row_index]
                )
                np.clip(expected, -max_velocity, max_velocity, out=expected)
                updated = True
        if not updated:
            break

    torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=0.0, atol=1e-12)
    assert not actual.requires_grad
