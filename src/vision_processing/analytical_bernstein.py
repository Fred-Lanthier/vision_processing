"""Analytical Bernstein-RDF collision constraints (no autograd).

The hot path in this module consists only of ordinary tensor operations under
``torch.inference_mode``.  It evaluates one smooth obstacle constraint per
protected robot link and its exact, first-order joint gradient.

Notation (column-vector convention):

    p_l = R_wl.T @ (p_w - o_wl)
    f_i = RDF_l(p_l[i])
    h_l = -tau * log(sum_i exp(-f_i / tau)) - d_safe

``tau`` is the Softmin temperature.  For the alternative convention
``-log(sum(exp(-alpha*f))) / alpha`` used in some papers, ``tau = 1/alpha``.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

import torch


Tensor = torch.Tensor


class LinkSoftminResult(NamedTuple):
    """Batched per-link constraint result.

    Shapes use ``B`` configurations, ``K`` links, ``M`` query points and ``Q``
    actuated joints (Q=7 for the Panda arm).
    """

    h: Tensor                 # [B, K]
    grad_q: Tensor            # [B, K, Q]
    sdf: Tensor               # [B, K, M]
    point_weights: Tensor     # [B, K, M]


def _binomial_row(degree: int, reference: Tensor) -> Tensor:
    return reference.new_tensor([math.comb(degree, i) for i in range(degree + 1)])


def bernstein_basis_and_derivative(
    t: Tensor,
    n_functions: int,
    binomial: Optional[Tensor] = None,
    lower_binomial: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Return ``B_i^n(t)`` and its exact derivative for every basis index.

    ``n_functions = n + 1`` and ``t`` may have any leading shape.  The
    derivative is evaluated with the stable Bernstein identity

        d B_i^n / dt = n (B_{i-1}^{n-1} - B_i^{n-1}),

    with out-of-range lower-degree basis functions defined as zero.  This
    avoids divisions by ``t`` or ``1-t`` and is well-defined at both endpoints.
    """
    if n_functions < 1:
        raise ValueError("n_functions must be positive")

    degree = n_functions - 1
    if binomial is None:
        binomial = _binomial_row(degree, t)
    else:
        binomial = binomial.to(device=t.device, dtype=t.dtype)

    i = torch.arange(n_functions, device=t.device, dtype=torch.int64)
    basis = (
        binomial
        * torch.pow(t.unsqueeze(-1), i)
        * torch.pow((1.0 - t).unsqueeze(-1), degree - i)
    )

    if degree == 0:
        return basis, torch.zeros_like(basis)

    if lower_binomial is None:
        lower_binomial = _binomial_row(degree - 1, t)
    else:
        lower_binomial = lower_binomial.to(device=t.device, dtype=t.dtype)
    j = torch.arange(degree, device=t.device, dtype=torch.int64)
    lower = (
        lower_binomial
        * torch.pow(t.unsqueeze(-1), j)
        * torch.pow((1.0 - t).unsqueeze(-1), degree - 1 - j)
    )

    # left[..., i] = B_{i-1}^{n-1}; right[..., i] = B_i^{n-1}.
    zero = torch.zeros_like(lower[..., :1])
    left = torch.cat((zero, lower), dim=-1)
    right = torch.cat((lower, zero), dim=-1)
    derivative = degree * (left - right)
    return basis, derivative


def tensor_product_bernstein_value_gradient(
    t: Tensor,
    weights: Tensor,
    binomial: Optional[Tensor] = None,
    lower_binomial: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Evaluate a 3-D tensor-product Bernstein polynomial and ``d/dt``.

    Args:
        t: Normalized points ``[B,K,M,3]`` in ``[0,1]^3``.
        weights: Per-link coefficients ``[K,n+1,n+1,n+1]``.

    Returns:
        Polynomial value ``[B,K,M]`` and normalized-space gradient
        ``[B,K,M,3]``.
    """
    if t.ndim != 4 or t.shape[-1] != 3:
        raise ValueError(f"t must have shape [B,K,M,3], got {tuple(t.shape)}")
    if weights.ndim != 4 or len(set(weights.shape[1:])) != 1:
        raise ValueError(
            f"weights must have shape [K,N,N,N], got {tuple(weights.shape)}")
    if weights.shape[0] != t.shape[1]:
        raise ValueError("the link dimensions of t and weights must match")

    n_functions = weights.shape[1]
    bx, dbx = bernstein_basis_and_derivative(
        t[..., 0], n_functions, binomial, lower_binomial)
    by, dby = bernstein_basis_and_derivative(
        t[..., 1], n_functions, binomial, lower_binomial)
    bz, dbz = bernstein_basis_and_derivative(
        t[..., 2], n_functions, binomial, lower_binomial)

    # P(t) = sum_abc W_abc B_a(t_x) B_b(t_y) B_c(t_z).
    value = torch.einsum("bkmx,bkmy,bkmz,kxyz->bkm", bx, by, bz, weights)
    grad_x = torch.einsum("bkmx,bkmy,bkmz,kxyz->bkm", dbx, by, bz, weights)
    grad_y = torch.einsum("bkmx,bkmy,bkmz,kxyz->bkm", bx, dby, bz, weights)
    grad_z = torch.einsum("bkmx,bkmy,bkmz,kxyz->bkm", bx, by, dbz, weights)
    return value, torch.stack((grad_x, grad_y, grad_z), dim=-1)


def softmin_value_gradient(
    values: Tensor,
    gradients: Tensor,
    temperature: float,
    point_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Soft-min over the last point axis and its exact analytical gradient.

    For ``h = -tau log(sum_i exp(-f_i/tau))``:

        pi_i = exp(-f_i/tau) / sum_j exp(-f_j/tau)
        grad h = sum_i pi_i grad f_i.

    The minimum shift makes the exponentials numerically stable and has no
    effect on the result or derivative.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be strictly positive")
    if gradients.shape[:-1] != values.shape:
        raise ValueError("gradients must have shape values.shape + [Q]")
    if values.shape[-1] == 0:
        raise ValueError("Softmin requires at least one point")

    if point_mask is not None:
        if point_mask.ndim == 2:
            point_mask = point_mask.unsqueeze(0).expand(values.shape[0], -1, -1)
        if point_mask.shape != values.shape:
            raise ValueError("point_mask must have shape [K,M] or [B,K,M]")
        masked_values = torch.where(
            point_mask, values, torch.full_like(values, float("inf")))
    else:
        masked_values = values

    minimum = masked_values.min(dim=-1, keepdim=True).values
    exponentials = torch.exp(-(masked_values - minimum) / temperature)
    if point_mask is not None:
        exponentials = torch.where(
            point_mask, exponentials, torch.zeros_like(exponentials))
    normalizer = exponentials.sum(dim=-1, keepdim=True)
    point_weights = exponentials / normalizer
    h = minimum.squeeze(-1) - temperature * torch.log(normalizer.squeeze(-1))
    grad_h = torch.einsum("bkm,bkmq->bkq", point_weights, gradients)
    return h, grad_h, point_weights


def project_local_gradient_to_joints(
    points_world: Tensor,
    transforms_world_from_link: Tensor,
    jacobian_linear_world: Tensor,
    jacobian_angular_world: Tensor,
    gradient_local: Tensor,
) -> Tensor:
    """Project fixed-world-point RDF gradients into joint space exactly.

    The Jacobians are geometric Jacobians at each link/visual-frame origin,
    expressed in the world frame, with shape ``[B,K,3,Q]``.  For a fixed world
    query point and ``r = p_w - o_l``:

        dp_l/dq = -R_wl.T (J_v + J_omega x r)

    Therefore, with ``g_w = R_wl g_l``, the scalar SDF derivative is

        df/dq = -J_v.T g_w - J_omega.T (r x g_w).

    This form avoids materializing a ``[B,K,M,3,Q]`` point Jacobian.
    """
    if points_world.ndim != 3 or points_world.shape[-1] != 3:
        raise ValueError("points_world must have shape [B,M,3]")
    if transforms_world_from_link.ndim != 4:
        raise ValueError("transforms must have shape [B,K,4,4]")
    if gradient_local.shape[:3] != (
        points_world.shape[0], transforms_world_from_link.shape[1], points_world.shape[1]
    ) or gradient_local.shape[-1] != 3:
        raise ValueError("gradient_local must have shape [B,K,M,3]")
    if jacobian_linear_world.shape != jacobian_angular_world.shape:
        raise ValueError("linear and angular Jacobians must have the same shape")

    rotation = transforms_world_from_link[..., :3, :3]       # [B,K,3,3]
    origin = transforms_world_from_link[..., :3, 3]          # [B,K,3]
    relative = points_world[:, None, :, :] - origin[:, :, None, :]
    gradient_world = torch.einsum("bkij,bkmj->bkmi", rotation, gradient_local)
    moment_gradient = torch.cross(relative, gradient_world, dim=-1)

    linear_term = torch.einsum(
        "bkmi,bkiq->bkmq", gradient_world, jacobian_linear_world)
    angular_term = torch.einsum(
        "bkmi,bkiq->bkmq", moment_gradient, jacobian_angular_world)
    return -(linear_term + angular_term)


@torch.inference_mode()
def project_halfspaces_sequential(
    dq_nominal: Tensor,
    gradients: Tensor,
    bounds: Tensor,
    iterations: int,
    relaxation: float = 1.0,
    max_velocity: Optional[float] = None,
    denominator_epsilon: float = 1.0e-6,
) -> Tensor:
    """Project a joint velocity onto ``gradients @ dq >= bounds`` on-device.

    This is the fixed-iteration Gauss--Seidel/POCS solve used by the CBF.  Each
    violated half-space row is projected analytically:

        dq <- dq + relaxation * relu(b_i - g_i.T dq)
                   * g_i / (g_i.T g_i + epsilon).

    All conditions remain tensors, so CUDA execution never reads a device scalar
    in Python and never requires an intermediate GPU-to-CPU transfer.  The loops
    are over the small, fixed QP dimensions (at most nine rows for the configured
    Panda system) and are suitable for subsequent CUDA-graph capture.
    """
    if dq_nominal.ndim != 1:
        raise ValueError("dq_nominal must have shape [Q]")
    if gradients.ndim != 2 or gradients.shape[1] != dq_nominal.shape[0]:
        raise ValueError("gradients must have shape [K,Q]")
    if bounds.ndim != 1 or bounds.shape[0] != gradients.shape[0]:
        raise ValueError("bounds must have shape [K]")
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    if relaxation <= 0.0:
        raise ValueError("relaxation must be positive")

    dq = dq_nominal.clone()
    row_norm_squared = (gradients * gradients).sum(dim=1) + denominator_epsilon
    for _ in range(iterations):
        for row_index in range(gradients.shape[0]):
            gradient = gradients[row_index]
            violation = bounds[row_index] - torch.dot(gradient, dq)
            correction = (
                relaxation
                * torch.clamp(violation, min=0.0)
                * gradient
                / row_norm_squared[row_index]
            )
            candidate = dq + correction
            if max_velocity is not None and max_velocity > 0.0:
                candidate = torch.clamp(
                    candidate, min=-max_velocity, max=max_velocity)
            # Preserve the old solver exactly: velocity clipping was applied only
            # when this row was violated, not on an already-feasible row.
            dq = torch.where(violation > 0.0, candidate, dq)
    return dq


class AnalyticalBernsteinSoftmin:
    """No-autograd adapter for the repository's ``BernsteinCore``/``URDFLayer``.

    The RDF implementation in this repository uses the local normalization

        u = (p_l - centroid) / scale,  t = (clamp(u) + 1) / 2,
        f = scale * sqrt(P(t)^2 + ||u - clamp(u)||^2)  outside (P >= 0),
        f = scale * P(t)                               inside.

    The out-of-domain extension is a guaranteed lower bound on the true
    distance whose gradient stays outward-oriented (monotone in the
    excursion); values and gradients match the existing ``BernsteinCore``
    outside its polynomial box as well as inside it.
    Polynomial orders may differ between links; equal-order links are evaluated
    together using the groups already constructed by ``BernsteinCore``.
    """

    def __init__(
        self,
        bernstein_core,
        temperature: float,
        d_safe: float = 0.0,
        clamp_margin: float = 1.0e-2,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be strictly positive")
        if not 0.0 <= clamp_margin < 1.0:
            raise ValueError("clamp_margin must be in [0,1)")

        self.core = bernstein_core
        self.robot = bernstein_core.robot
        self.temperature = float(temperature)
        self.d_safe = float(d_safe)
        self.clamp_margin = float(clamp_margin)
        self.link_count = int(bernstein_core.K)

        self._target_link_names = [
            info["link_name"] for info in bernstein_core._link_mesh_infos
        ]
        self._root_link = self._find_root_link()
        self._active_joint_mask = self._build_active_joint_mask()
        self._grasp_attach_link = getattr(bernstein_core, "grasp_attach_link", None)
        self._grasp_active_joint_mask = (
            None if self._grasp_attach_link is None
            else self._active_mask_for_link(self._grasp_attach_link)
        )
        self._revolute_joint_mask, self._prismatic_joint_mask = self._joint_type_masks()

        # Cache both binomial rows.  Runtime evaluation performs no coefficient
        # construction and no CPU/GPU synchronization.
        self._basis_cache = {}
        for group in self.core.groups.values():
            n_functions = int(group["n_func"])
            reference = group["weights"]
            degree = n_functions - 1
            self._basis_cache[n_functions] = (
                _binomial_row(degree, reference),
                None if degree == 0 else _binomial_row(degree - 1, reference),
            )

    def _find_root_link(self) -> str:
        children = {joint["child_link"] for joint in self.robot.kinematic_tree}
        for joint in self.robot.kinematic_tree:
            if joint["parent_link"] not in children:
                return joint["parent_link"]
        # A URDF with no joints is not useful here, but keep the failure explicit.
        raise ValueError("unable to determine the URDF root link")

    def _build_active_joint_mask(self) -> Tensor:
        mask = torch.zeros(
            (self.link_count, int(self.robot.dof)),
            dtype=torch.bool,
            device=self.core.device,
        )
        for link_index, target_link in enumerate(self._target_link_names):
            mask[link_index] = self._active_mask_for_link(target_link)
        return mask

    def _active_mask_for_link(self, target_link: str) -> Tensor:
        joint_by_child = {
            joint["child_link"]: joint for joint in self.robot.kinematic_tree
        }
        mask = torch.zeros(
            int(self.robot.dof), dtype=torch.bool, device=self.core.device)
        link = target_link
        while link in joint_by_child:
            joint = joint_by_child[link]
            if int(joint["idx"]) >= 0:
                mask[int(joint["idx"])] = True
            link = joint["parent_link"]
        return mask

    def _joint_type_masks(self) -> Tuple[Tensor, Tensor]:
        revolute = torch.zeros(
            int(self.robot.dof), dtype=torch.bool, device=self.core.device)
        prismatic = torch.zeros_like(revolute)
        for joint in self.robot.kinematic_tree:
            index = int(joint["idx"])
            if index < 0:
                continue
            if joint["type"] in ("revolute", "continuous"):
                revolute[index] = True
            elif joint["type"] == "prismatic":
                prismatic[index] = True
        return revolute, prismatic

    @staticmethod
    def _revolute_transform(joint, q: Tensor) -> Tensor:
        skew = joint["K"].to(device=q.device, dtype=q.dtype)
        skew_squared = joint["K_sq"].to(device=q.device, dtype=q.dtype)
        eye = torch.eye(3, device=q.device, dtype=q.dtype)
        rotation = (
            eye
            + torch.sin(q)[:, None, None] * skew
            + (1.0 - torch.cos(q))[:, None, None] * skew_squared
        )
        batch = q.shape[0]
        transform = torch.zeros((batch, 4, 4), device=q.device, dtype=q.dtype)
        transform[:, :3, :3] = rotation
        transform[:, 3, 3] = 1.0
        return transform

    @staticmethod
    def _prismatic_transform(joint, q: Tensor) -> Tensor:
        batch = q.shape[0]
        transform = torch.eye(4, device=q.device, dtype=q.dtype).expand(
            batch, 4, 4).clone()
        axis = joint["axis"].to(device=q.device, dtype=q.dtype)
        transform[:, :3, 3] = q[:, None] * axis
        return transform

    def _visual_kinematics(
        self,
        q: Tensor,
        pose: Tensor,
    ):
        """Return visual (and optional grasp-frame) kinematics in world."""
        batch, q_dimension = q.shape
        robot_dof = int(self.robot.dof)
        if q_dimension > robot_dof:
            raise ValueError(f"q has {q_dimension} joints but URDF has {robot_dof}")
        if q_dimension < robot_dof:
            # Uncontrolled tail (gripper fingers): use the live measured buffer
            # shared with BernsteinCore when attached, zeros otherwise. Read
            # through a view so in-place updates reach captured CUDA graphs.
            missing = robot_dof - q_dimension
            extra = getattr(self, 'q_extra', None)
            if (extra is not None and extra.numel() == missing
                    and extra.device == q.device and extra.dtype == q.dtype):
                tail = extra.reshape(1, missing).expand(batch, missing)
            else:
                tail = q.new_zeros((batch, missing))
            q_full = torch.cat((q, tail), dim=-1)
        else:
            q_full = q

        poses = {
            self._root_link: torch.eye(4, device=q.device, dtype=q.dtype).expand(
                batch, 4, 4)
        }
        joint_origins_base = q.new_zeros((batch, robot_dof, 3))
        joint_axes_base = q.new_zeros((batch, robot_dof, 3))

        # FK is necessarily sequential along the URDF tree.  All point/link/RDF
        # work below is batched; this loop is only over the small static joint tree.
        for joint in self.robot.kinematic_tree:
            parent = poses[joint["parent_link"]]
            offset = joint["offset"].to(device=q.device, dtype=q.dtype)
            pre_joint = parent @ offset.expand(batch, 4, 4)
            index = int(joint["idx"])
            if index >= 0:
                axis_local = joint["axis"].to(device=q.device, dtype=q.dtype)
                joint_origins_base[:, index] = pre_joint[:, :3, 3]
                joint_axes_base[:, index] = torch.einsum(
                    "bij,j->bi", pre_joint[:, :3, :3], axis_local)

            if joint["type"] in ("revolute", "continuous"):
                motion = self._revolute_transform(joint, q_full[:, index])
                child_pose = pre_joint @ motion
            elif joint["type"] == "prismatic":
                motion = self._prismatic_transform(joint, q_full[:, index])
                child_pose = pre_joint @ motion
            else:
                child_pose = pre_joint
            poses[joint["child_link"]] = child_pose

        visual_base = []
        for target_link, mesh_info in zip(
            self._target_link_names, self.core._link_mesh_infos
        ):
            visual_offset = mesh_info["visual_offset"].to(
                device=q.device, dtype=q.dtype)
            visual_base.append(poses[target_link] @ visual_offset.expand(batch, 4, 4))
        visual_base = torch.stack(visual_base, dim=1)              # [B,K,4,4]
        transforms = pose[:, None, :, :] @ visual_base

        base_rotation = pose[:, :3, :3]
        base_translation = pose[:, :3, 3]
        joint_axes = torch.einsum("bij,bqj->bqi", base_rotation, joint_axes_base)
        joint_origins = (
            torch.einsum("bij,bqj->bqi", base_rotation, joint_origins_base)
            + base_translation[:, None, :]
        )
        axes = joint_axes[:, None, :, :].expand(batch, self.link_count, robot_dof, 3)
        revolute = self._revolute_joint_mask[None, None, :, None]
        prismatic = self._prismatic_joint_mask[None, None, :, None]
        active = self._active_joint_mask[None, :, :, None]
        visual_origins = transforms[:, :, :3, 3]
        radius = visual_origins[:, :, None, :] - joint_origins[:, None, :, :]
        jacobian_linear = torch.where(
            revolute,
            torch.cross(axes, radius, dim=-1),
            torch.where(prismatic, axes, torch.zeros_like(axes)),
        ) * active
        jacobian_angular = torch.where(
            revolute, axes, torch.zeros_like(axes)) * active

        # Return only derivatives with respect to the supplied q, not padded URDF
        # joints (e.g. the two fixed-position Panda finger joints).
        jacobian_linear = jacobian_linear[:, :, :q_dimension, :].transpose(-1, -2)
        jacobian_angular = jacobian_angular[:, :, :q_dimension, :].transpose(-1, -2)

        grasp_kinematics = None
        if self._grasp_attach_link is not None:
            if self._grasp_attach_link not in poses:
                raise ValueError(
                    f"grasp attach link {self._grasp_attach_link!r} is not in the URDF")
            grasp_transform = pose @ poses[self._grasp_attach_link]
            grasp_origin = grasp_transform[:, :3, 3]
            grasp_radius = grasp_origin[:, None, :] - joint_origins
            grasp_axes = joint_axes
            grasp_active = self._grasp_active_joint_mask[None, :, None]
            grasp_jv = torch.where(
                self._revolute_joint_mask[None, :, None],
                torch.cross(grasp_axes, grasp_radius, dim=-1),
                torch.where(
                    self._prismatic_joint_mask[None, :, None],
                    grasp_axes,
                    torch.zeros_like(grasp_axes),
                ),
            ) * grasp_active
            grasp_jw = torch.where(
                self._revolute_joint_mask[None, :, None],
                grasp_axes,
                torch.zeros_like(grasp_axes),
            ) * grasp_active
            grasp_kinematics = (
                grasp_transform,
                grasp_jv[:, :q_dimension, :].transpose(-1, -2),
                grasp_jw[:, :q_dimension, :].transpose(-1, -2),
            )
        return transforms, jacobian_linear, jacobian_angular, grasp_kinematics

    def _evaluate_rdf_group(
        self,
        local_points: Tensor,
        group,
    ) -> Tuple[Tensor, Tensor]:
        """Repository-compatible RDF value/local gradient for one order group."""
        n_functions = int(group["n_func"])
        weights = group["weights"].reshape(
            len(group["indices"]), n_functions, n_functions, n_functions)
        centroids = group["offsets"].reshape(1, len(group["indices"]), 1, 3)
        scales = group["scales"].reshape(1, len(group["indices"]), 1, 1)

        normalized = (local_points - centroids) / scales
        lower = -1.0 + self.clamp_margin
        upper = 1.0 - self.clamp_margin
        bounded = torch.clamp(normalized, min=lower, max=upper)
        residual = normalized - bounded
        t = 0.5 * (bounded + 1.0)

        binomial, lower_binomial = self._basis_cache[n_functions]
        polynomial, gradient_t = tensor_product_bernstein_value_gradient(
            t, weights, binomial, lower_binomial)

        # f = scale * sqrt(P^2 + ||residual||^2) outside the box (P >= 0),
        # f = scale * P inside (residual == 0), normalized=(p_l-c)/scale.
        # Guaranteed LOWER bound on the true distance: every link surface
        # point s lies in the clamp cube, and on each clamped axis
        # |p_j - s_j| = |c_j - s_j| + |res_j|, hence
        # d_true^2 >= P^2 + ||res||^2 (matches BernsteinCore). This bound is
        # monotone increasing in the excursion, so the gradient keeps the
        # physical outward direction everywhere (required by the CBF and the
        # null-space clearance drive); it also joins nabla P smoothly at the
        # domain boundary. P < 0 at the clamped point (clamp-margin shell)
        # falls back to P + ||res|| to preserve the penetration sign. The
        # outer/inner scale factors cancel in df/dp_l. Clamp contributes only
        # on interior coordinates; the residual contributes outside.
        interior = ((normalized >= lower) & (normalized <= upper)).to(
            dtype=local_points.dtype)
        polynomial_gradient_local = 0.5 * gradient_t * interior
        residual_norm = torch.linalg.vector_norm(residual, dim=-1)
        tiny = torch.finfo(local_points.dtype).tiny
        residual_gradient_local = torch.where(
            residual_norm[..., None] > 0.0,
            residual / residual_norm[..., None].clamp_min(tiny),
            torch.zeros_like(residual),
        )
        outside = residual_norm > 0.0
        positive = polynomial >= 0.0
        pythagorean = torch.sqrt(
            polynomial.clamp_min(0.0).square() + residual_norm.square())
        value = torch.where(
            outside,
            torch.where(positive, pythagorean, polynomial + residual_norm),
            polynomial,
        )
        # Gradient of the active branch:
        #   sqrt branch:  (P * dP + res) / value  (all components outward;
        #                 -> dP smoothly as res -> 0)
        #   shell branch: dP + res_unit           (P < 0 continuation)
        #   inside:       dP
        safe_value = pythagorean.clamp_min(tiny)
        sqrt_gradient = (
            polynomial.clamp_min(0.0)[..., None] * polynomial_gradient_local
            + residual
        ) / safe_value[..., None]
        gradient_local = torch.where(
            (outside & positive)[..., None],
            sqrt_gradient,
            torch.where(
                outside[..., None],
                polynomial_gradient_local + residual_gradient_local,
                polynomial_gradient_local,
            ),
        )
        sdf = value * scales[..., 0]
        return sdf, gradient_local

    def _evaluate_grasp_sdf(
        self,
        points_world: Tensor,
        transform: Tensor,
        jacobian_linear: Tensor,
        jacobian_angular: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Exact SDF/joint gradient for the optional rigid grasp point cloud."""
        grasp_points = self.core.grasp_points.to(
            device=points_world.device, dtype=points_world.dtype)
        rotation = transform[:, :3, :3]
        origin = transform[:, :3, 3]
        centers = (
            torch.einsum("bij,gj->bgi", rotation, grasp_points)
            + origin[:, None, :]
        )
        difference = points_world[:, None, :, :] - centers[:, :, None, :]
        # Match BernsteinCore._grasp_cloud_sdf, including its small sqrt guard.
        distance = torch.sqrt((difference * difference).sum(dim=-1) + 1.0e-12)
        unit_center_to_obstacle = difference / distance[..., None]

        center_radius = centers - origin[:, None, :]
        moment_gradient = torch.cross(
            center_radius[:, :, None, :], unit_center_to_obstacle, dim=-1)
        gradient_each_center = -(
            torch.einsum(
                "bgmi,biq->bgmq", unit_center_to_obstacle, jacobian_linear)
            + torch.einsum(
                "bgmi,biq->bgmq", moment_gradient, jacobian_angular)
        )

        beta = float(getattr(self.core, "grasp_softmin_beta", 0.0))
        if beta > 0.0:
            minimum = distance.min(dim=1, keepdim=True).values
            exponentials = torch.exp(-(distance - minimum) / beta)
            inner_weights = exponentials / exponentials.sum(dim=1, keepdim=True)
            distance_sdf = (
                minimum.squeeze(1)
                - beta * torch.log(exponentials.sum(dim=1))
            )
            gradient_q = (inner_weights[..., None] * gradient_each_center).sum(dim=1)
        else:
            distance_sdf, nearest = distance.min(dim=1)
            gather_index = nearest[:, None, :, None].expand(
                -1, 1, -1, gradient_each_center.shape[-1])
            gradient_q = gradient_each_center.gather(1, gather_index).squeeze(1)

        radius = self.core.grasp_radius.to(
            device=points_world.device, dtype=points_world.dtype)
        active = self.core.grasp_active.to(
            device=points_world.device, dtype=points_world.dtype)
        sdf = (
            distance_sdf
            - radius
            - float(getattr(self.core, "grasp_dsafe_offset", 0.0))
            + (1.0 - active) * 1.0e3
        )
        return sdf, gradient_q

    @torch.inference_mode()
    def evaluate(
        self,
        q: Tensor,
        points_world: Tensor,
        pose: Optional[Tensor] = None,
        point_mask: Optional[Tensor] = None,
    ) -> LinkSoftminResult:
        """Evaluate all independent link constraints and their joint gradients.

        Args:
            q: ``[B,Q]`` configurations; ``Q=7`` for Panda arm control.
            points_world: shared ``[M,3]`` points or per-configuration ``[B,M,3]``.
            pose: optional world-from-URDF-root transform ``[4,4]`` or ``[B,4,4]``.

        Returns:
            ``h[B,K]`` and ``grad_q[B,K,Q]`` plus point-level diagnostics.  Rows
            ``grad_q[:,k]`` can be inserted directly into an SQP/QP constraint
            Jacobian.  No returned tensor has a computation graph.
        """
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if q.ndim != 2:
            raise ValueError(f"q must have shape [B,Q], got {tuple(q.shape)}")
        batch = q.shape[0]

        if points_world.ndim == 2:
            points_world = points_world.unsqueeze(0).expand(batch, -1, -1)
        elif points_world.ndim != 3 or points_world.shape[0] != batch:
            raise ValueError("points_world must have shape [M,3] or [B,M,3]")
        if points_world.shape[-1] != 3 or points_world.shape[1] == 0:
            raise ValueError("points_world must contain at least one 3-D point")
        points_world = points_world.to(device=q.device, dtype=q.dtype)

        if pose is None:
            pose = torch.eye(4, device=q.device, dtype=q.dtype).expand(batch, 4, 4)
        elif pose.ndim == 2:
            pose = pose.to(device=q.device, dtype=q.dtype).unsqueeze(0).expand(
                batch, 4, 4)
        elif pose.ndim == 3 and pose.shape[0] == batch:
            pose = pose.to(device=q.device, dtype=q.dtype)
        else:
            raise ValueError("pose must have shape [4,4] or [B,4,4]")

        transforms, jacobian_linear, jacobian_angular, grasp_kinematics = (
            self._visual_kinematics(q, pose))
        rotation = transforms[..., :3, :3]
        origin = transforms[..., :3, 3]
        difference = points_world[:, None, :, :] - origin[:, :, None, :]
        local_points = torch.matmul(difference, rotation)        # R.T @ difference

        sdf = q.new_empty((batch, self.link_count, points_world.shape[1]))
        gradient_local = q.new_empty(
            (batch, self.link_count, points_world.shape[1], 3))
        for group in self.core.groups.values():
            indices = group["indices_tensor"].to(device=q.device)
            sdf_group, gradient_group = self._evaluate_rdf_group(
                torch.index_select(local_points, 1, indices), group)
            sdf.index_copy_(1, indices, sdf_group)
            gradient_local.index_copy_(1, indices, gradient_group)

        gradient_q_per_point = project_local_gradient_to_joints(
            points_world,
            transforms,
            jacobian_linear,
            jacobian_angular,
            gradient_local,
        )
        if grasp_kinematics is not None and self.core.grasp_points is not None:
            grasp_sdf, grasp_gradient_q = self._evaluate_grasp_sdf(
                points_world, *grasp_kinematics)
            sdf = torch.cat((sdf, grasp_sdf[:, None, :]), dim=1)
            gradient_q_per_point = torch.cat(
                (gradient_q_per_point, grasp_gradient_q[:, None, :, :]), dim=1)
        if point_mask is not None:
            point_mask = point_mask.to(device=q.device, dtype=torch.bool)
        h, grad_q, point_weights = softmin_value_gradient(
            sdf, gradient_q_per_point, self.temperature, point_mask=point_mask)
        return LinkSoftminResult(h - self.d_safe, grad_q, sdf, point_weights)
