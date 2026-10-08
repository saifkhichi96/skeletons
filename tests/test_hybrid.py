from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from skeletons import (
    HybridRegressorTrainingState,
    JointRegressor,
    build_layer,
    evaluate_hybrid_ik_regressor,
    generate_random_pose_batch,
    make_perspective_camera,
    solve_hybrid_ik,
    train_joint_regressor_epoch,
)


def test_joint_regressor_outputs_3d_joint_shape() -> None:
    regressor = JointRegressor(num_joints=17, hidden_dim=32)
    keypoints_2d = torch.zeros(4, 17, 2)

    output = regressor(keypoints_2d)

    assert output.shape == (4, 17, 3)


def test_solve_hybrid_ik_reconstructs_rest_pose_joints() -> None:
    model = build_layer("human36m")
    joints = model().joints.detach()

    result = solve_hybrid_ik(joints, model)

    assert result.ik.local_rotations.shape == (1, model.NUM_JOINTS, 3, 3)
    assert result.reconstructed_joints.shape == joints.shape
    assert torch.allclose(result.reconstructed_joints, joints, atol=1e-5)


def test_evaluate_hybrid_ik_regressor_reports_joint_and_fk_metrics() -> None:
    model = build_layer("human36m")
    camera = make_perspective_camera()
    batch = generate_random_pose_batch(
        model,
        batch_size=6,
        pose_std=0.05,
        generator=torch.Generator().manual_seed(31),
    )
    keypoints_2d = camera.project(batch.joints)
    loader = DataLoader(
        TensorDataset(keypoints_2d, batch.joints),
        batch_size=3,
        shuffle=False,
    )
    regressor = JointRegressor(model.NUM_JOINTS, hidden_dim=32)

    state = evaluate_hybrid_ik_regressor(regressor, model, loader)

    assert state.num_samples == 6
    assert state.joint_mpjpe >= 0.0
    assert state.fk_consistency_mpjpe >= 0.0


def test_train_joint_regressor_epoch_updates_model_and_reports_loss() -> None:
    model = build_layer("human36m")
    camera = make_perspective_camera()
    batch = generate_random_pose_batch(
        model,
        batch_size=4,
        pose_std=0.05,
        generator=torch.Generator().manual_seed(32),
    )
    loader = DataLoader(
        TensorDataset(camera.project(batch.joints), batch.joints),
        batch_size=2,
        shuffle=False,
    )
    regressor = JointRegressor(model.NUM_JOINTS, hidden_dim=16)
    optimizer = torch.optim.Adam(regressor.parameters(), lr=1e-3)
    before = next(regressor.parameters()).detach().clone()

    state = train_joint_regressor_epoch(
        regressor,
        loader,
        optimizer=optimizer,
    )

    assert isinstance(state, HybridRegressorTrainingState)
    assert state.num_samples == 4
    assert state.loss > 0.0
    assert state.joint_mse == state.loss
    assert not torch.allclose(before, next(regressor.parameters()).detach())


def test_hybrid_ik_validates_shapes_and_empty_batches() -> None:
    model = build_layer("human36m")
    regressor = JointRegressor(model.NUM_JOINTS, hidden_dim=32)

    with pytest.raises(ValueError, match="predicted_joints"):
        solve_hybrid_ik(torch.zeros(1, model.NUM_JOINTS, 2), model)

    with pytest.raises(ValueError, match="at least one sample"):
        evaluate_hybrid_ik_regressor(regressor, model, [])

    optimizer = torch.optim.Adam(regressor.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="at least one sample"):
        train_joint_regressor_epoch(regressor, [], optimizer=optimizer)
