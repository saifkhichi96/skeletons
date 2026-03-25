from __future__ import annotations

import torch

from differential_skeletons.rotations import (
    axis_angle_to_matrix,
    matrix_to_rot6d,
    quaternion_to_matrix,
    rot6d_to_matrix,
    rotation_geodesic_distance,
)


def test_axis_angle_identity() -> None:
    matrix = axis_angle_to_matrix(torch.zeros(3))
    assert torch.allclose(matrix, torch.eye(3), atol=1e-6)


def test_quaternion_identity() -> None:
    matrix = quaternion_to_matrix(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(matrix, torch.eye(3), atol=1e-6)


def test_rot6d_identity() -> None:
    matrix = rot6d_to_matrix(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    assert torch.allclose(matrix, torch.eye(3), atol=1e-6)


def test_matrix_to_rot6d_roundtrip() -> None:
    base = axis_angle_to_matrix(torch.tensor([0.1, -0.2, 0.05]))
    rec = rot6d_to_matrix(matrix_to_rot6d(base))
    assert torch.allclose(rec, base, atol=1e-5)


def test_rotation_geodesic_distance_has_finite_gradient_near_pi() -> None:
    rotation_a = torch.eye(3).reshape(1, 3, 3).requires_grad_(True)
    rotation_b = torch.diag(torch.tensor([1.0, -1.0, -1.0])).reshape(1, 3, 3)

    loss = rotation_geodesic_distance(rotation_a, rotation_b, reduction="mean")
    loss.backward()

    assert torch.isfinite(loss)
    assert rotation_a.grad is not None
    assert torch.isfinite(rotation_a.grad).all()
