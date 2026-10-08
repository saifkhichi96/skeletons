from __future__ import annotations

import pytest
import torch

from skeletons import (
    joint_position_error,
    mean_per_joint_position_error,
    percentage_of_correct_keypoints,
    procrustes_align,
    procrustes_aligned_mpjpe,
)


def test_joint_position_error_returns_per_joint_distances() -> None:
    predicted = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
    target = torch.zeros_like(predicted)

    error = joint_position_error(predicted, target)

    assert torch.allclose(error, torch.tensor([0.0, 3.0]))


def test_mpjpe_supports_weights_and_none_reduction() -> None:
    predicted = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]],
        ]
    )
    target = torch.zeros_like(predicted)
    weights = torch.tensor([1.0, 0.0])

    error = mean_per_joint_position_error(
        predicted,
        target,
        weights=weights,
        reduction="none",
    )

    assert torch.allclose(error, torch.tensor([1.0, 3.0]))


def test_procrustes_align_recovers_similarity_transform() -> None:
    target = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=torch.float64,
    )
    angle = torch.tensor(0.6, dtype=torch.float64)
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    rotation = torch.tensor(
        [
            [cos_angle, -sin_angle, 0.0],
            [sin_angle, cos_angle, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    predicted = target @ rotation * 2.5 + torch.tensor(
        [10.0, -3.0, 6.0],
        dtype=torch.float64,
    )

    aligned = procrustes_align(predicted, target)
    error = procrustes_aligned_mpjpe(predicted, target)

    assert torch.allclose(aligned, target, atol=1e-10)
    assert torch.allclose(error, torch.tensor(0.0, dtype=torch.float64), atol=1e-10)


def test_percentage_of_correct_keypoints_thresholds_distances() -> None:
    predicted = torch.tensor([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
    target = torch.zeros_like(predicted)

    pck = percentage_of_correct_keypoints(predicted, target, threshold=1.0)

    assert torch.allclose(pck, torch.tensor(0.5))


def test_metrics_validate_shapes_weights_and_reductions() -> None:
    predicted = torch.zeros(2, 3)
    target = torch.zeros(3, 3)

    with pytest.raises(ValueError, match="same shape"):
        joint_position_error(predicted, target)

    with pytest.raises(ValueError, match="non-negative"):
        mean_per_joint_position_error(
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            weights=torch.tensor([-1.0, 1.0]),
        )

    with pytest.raises(ValueError, match="reduction"):
        percentage_of_correct_keypoints(
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            threshold=1.0,
            reduction="median",
        )

    with pytest.raises(ValueError, match="threshold"):
        percentage_of_correct_keypoints(
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            threshold=-1.0,
        )
