from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from skeletons import (
    ForwardKinematicsLoss,
    LifterTrainingState,
    PoseLifter,
    TemporalPoseLifter,
    build_layer,
    evaluate_lifter,
    generate_orthographic_lifting_dataset,
    generate_random_walk_sequences,
    make_perspective_camera,
    train_lifter_epoch,
)


def test_pose_lifter_outputs_axis_angle_pose_shape() -> None:
    lifter = PoseLifter(num_joints=17, hidden_dim=32)
    keypoints_2d = torch.zeros(4, 17, 2)

    output = lifter(keypoints_2d)

    assert output.shape == (4, 17, 3)


def test_temporal_pose_lifter_preserves_sequence_length() -> None:
    lifter = TemporalPoseLifter(num_joints=17, hidden_dim=32, kernel_size=3)
    keypoints_2d = torch.zeros(2, 9, 17, 2)

    output = lifter(keypoints_2d)

    assert output.shape == (2, 9, 17, 3)


def test_evaluate_lifter_handles_frame_batches() -> None:
    model = build_layer("human36m")
    dataset = generate_orthographic_lifting_dataset(
        model,
        num_samples=8,
        pose_std=0.05,
        generator=torch.Generator().manual_seed(17),
    )
    loader = DataLoader(
        TensorDataset(dataset.keypoints_2d, dataset.joints_3d),
        batch_size=4,
        shuffle=False,
    )
    lifter = PoseLifter(model.NUM_JOINTS, hidden_dim=32)

    state = evaluate_lifter(
        lifter,
        model,
        loader,
        fk_loss=ForwardKinematicsLoss(model),
    )

    assert state.num_frames == 8
    assert state.fk_loss >= 0.0
    assert state.mpjpe >= 0.0


def test_evaluate_lifter_handles_sequence_batches() -> None:
    model = build_layer("human36m")
    camera = make_perspective_camera()
    sequence = generate_random_walk_sequences(
        model,
        batch_size=2,
        seq_len=5,
        pose_step_std=0.01,
        transl_step_std=2.0,
        depth=2500.0,
        generator=torch.Generator().manual_seed(23),
    )
    keypoints_2d = camera.project(sequence.joints)
    loader = DataLoader(
        TensorDataset(keypoints_2d, sequence.joints),
        batch_size=1,
        shuffle=False,
    )
    lifter = TemporalPoseLifter(model.NUM_JOINTS, hidden_dim=32)

    state = evaluate_lifter(lifter, model, loader)

    assert state.num_frames == 10
    assert state.fk_loss >= 0.0
    assert state.mpjpe >= 0.0


def test_train_lifter_epoch_updates_model_and_reports_losses() -> None:
    model = build_layer("human36m")
    dataset = generate_orthographic_lifting_dataset(
        model,
        num_samples=4,
        pose_std=0.05,
        generator=torch.Generator().manual_seed(31),
    )
    loader = DataLoader(
        TensorDataset(dataset.keypoints_2d, dataset.joints_3d),
        batch_size=2,
    )
    lifter = PoseLifter(model.NUM_JOINTS, hidden_dim=16)
    optimizer = torch.optim.Adam(lifter.parameters(), lr=1e-3)
    before = next(lifter.parameters()).detach().clone()

    state = train_lifter_epoch(
        lifter,
        model,
        loader,
        optimizer=optimizer,
        fk_loss=ForwardKinematicsLoss(model),
        pose_reg=1e-4,
    )

    assert isinstance(state, LifterTrainingState)
    assert state.num_frames == 4
    assert state.loss > 0.0
    assert state.fk_loss > 0.0
    assert state.pose_regularization >= 0.0
    assert not torch.allclose(before, next(lifter.parameters()).detach())


def test_evaluate_lifter_validates_empty_and_invalid_batches() -> None:
    model = build_layer("human36m")
    lifter = PoseLifter(model.NUM_JOINTS, hidden_dim=32)

    with pytest.raises(ValueError, match="at least one frame"):
        evaluate_lifter(lifter, model, [])

    optimizer = torch.optim.Adam(lifter.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="pose_reg"):
        train_lifter_epoch(lifter, model, [], optimizer=optimizer, pose_reg=-1.0)

    bad_loader = [(torch.zeros(1, model.NUM_JOINTS, 2), torch.zeros(1, 3, 3))]
    with pytest.raises(ValueError, match="target joints"):
        evaluate_lifter(lifter, model, bad_loader)
