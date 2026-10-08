from __future__ import annotations

import pytest
import torch

from skeletons import (
    SequenceFittingResult,
    TemporalPriorTrainingResult,
    build_layer,
    denoise_joint_sequences,
    fit_2d_joint_sequences,
    generate_random_walk_sequences,
    make_weak_perspective_camera,
    train_temporal_prior,
)
from skeletons.fitting import SequenceDataset


def test_train_temporal_prior_returns_prior_and_epoch_history() -> None:
    model = build_layer("spinetrack")
    batch = generate_random_walk_sequences(
        model,
        batch_size=4,
        seq_len=4,
        pose_step_std=0.01,
        transl_step_std=0.1,
        depth=100.0,
        device="cpu",
        generator=torch.Generator().manual_seed(21),
    )

    result = train_temporal_prior(
        batch.joints,
        model=model,
        batch_size=2,
        epochs=1,
        hidden_dim=12,
        num_layers=1,
        generator=torch.Generator().manual_seed(22),
    )

    assert isinstance(result, TemporalPriorTrainingResult)
    assert result.temporal_prior.pose_dim == (model.NUM_JOINTS - 1) * 6
    assert len(result.history) == 1
    assert result.history[0].epoch == 1
    assert result.history[0].loss > 0.0


def test_denoise_joint_sequences_returns_dataset_shaped_result() -> None:
    model = build_layer("spinetrack")
    batch = generate_random_walk_sequences(
        model,
        batch_size=1,
        seq_len=3,
        pose_step_std=0.01,
        transl_step_std=0.1,
        depth=100.0,
        device="cpu",
        generator=torch.Generator().manual_seed(23),
    )
    noisy = batch.joints + 0.01 * torch.randn(
        batch.joints.shape,
        generator=torch.Generator().manual_seed(24),
    )

    result = denoise_joint_sequences(
        noisy,
        model=model,
        num_iters=1,
        optimize_scales=False,
    )
    dataset = result.to_sequence_dataset()

    assert isinstance(result, SequenceFittingResult)
    assert result.joints_3d.shape == noisy.shape
    assert result.target_joints_3d is not None
    assert result.iterations == 1
    assert "joints_3d" in result.losses
    assert isinstance(dataset, SequenceDataset)
    assert len(dataset) == 1
    with pytest.raises(ValueError, match="reprojection"):
        _ = result.mean_reprojection_error


def test_fit_2d_joint_sequences_returns_reprojection_diagnostics() -> None:
    model = build_layer("spinetrack")
    camera = make_weak_perspective_camera("cpu")
    batch = generate_random_walk_sequences(
        model,
        batch_size=1,
        seq_len=3,
        pose_step_std=0.01,
        transl_step_std=0.1,
        depth=100.0,
        device="cpu",
        generator=torch.Generator().manual_seed(25),
    )
    joints_2d = camera.project(batch.joints)
    confidences = torch.ones(1, 3, model.NUM_JOINTS)
    confidences[..., -1] = 0.0

    result = fit_2d_joint_sequences(
        joints_2d,
        camera,
        model=model,
        confidences=confidences,
        num_iters=1,
        optimize_scales=False,
    )
    dataset = result.to_sequence_dataset()

    assert result.joints_3d.shape == batch.joints.shape
    assert result.projected_joints_2d is not None
    assert result.projected_joints_2d.shape == joints_2d.shape
    assert result.confidences is not None
    assert result.reprojection_errors is not None
    assert result.reprojection_errors.shape == (1, 3)
    assert result.mean_reprojection_error >= 0.0
    assert result.iterations == 1
    assert "reprojection" in result.losses
    assert dataset.joints_2d is not None


def test_sequence_helpers_validate_shapes() -> None:
    model = build_layer("spinetrack")
    camera = make_weak_perspective_camera("cpu")

    with pytest.raises(ValueError, match="at least two frames"):
        train_temporal_prior(
            torch.zeros(1, 1, model.NUM_JOINTS, 3),
            model=model,
        )

    with pytest.raises(ValueError, match="confidences"):
        fit_2d_joint_sequences(
            torch.zeros(1, 2, model.NUM_JOINTS, 2),
            camera,
            model=model,
            confidences=torch.ones(1, 2, model.NUM_JOINTS, 1),
        )
