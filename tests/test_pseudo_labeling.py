from __future__ import annotations

import numpy as np
import pytest
import torch

from skeletons import (
    PseudoLabelPriors,
    build_layer,
    fit_2d_pseudo_labels,
    generate_random_pose_batch,
    make_weak_perspective_camera,
    save_pseudo_label_npz,
    train_pseudo_label_priors,
)
from skeletons.fitting import FrameDataset


def test_train_pseudo_label_priors_returns_bundle_with_epoch_history() -> None:
    model = build_layer("spinetrack")
    batch = generate_random_pose_batch(
        model,
        batch_size=6,
        pose_std=0.05,
        device="cpu",
        generator=torch.Generator().manual_seed(11),
    )

    priors = train_pseudo_label_priors(
        batch.joints,
        model=model,
        batch_size=3,
        epochs=1,
        latent_dim=4,
        hidden_dim=16,
        num_hidden_layers=1,
        generator=torch.Generator().manual_seed(12),
    )

    assert isinstance(priors, PseudoLabelPriors)
    assert priors.pose_prior.latent_dim == 4
    assert priors.joint_limit_prior.mean.shape == (model.NUM_JOINTS - 1, 3)
    assert len(priors.history) == 1
    assert priors.history[0].epoch == 1
    assert priors.history[0].loss > 0.0


def test_fit_2d_pseudo_labels_saves_dataset_payload(tmp_path) -> None:
    model = build_layer("spinetrack")
    camera = make_weak_perspective_camera("cpu")
    source = generate_random_pose_batch(
        model,
        batch_size=2,
        pose_std=0.02,
        transl_std=0.0,
        depth=0.0,
        device="cpu",
        generator=torch.Generator().manual_seed(13),
    )
    joints_2d = camera.project(source.joints)
    confidences = torch.ones(2, model.NUM_JOINTS)
    confidences[0, -1] = 0.0

    result = fit_2d_pseudo_labels(
        joints_2d,
        camera,
        model=model,
        confidences=confidences,
        batch_size=2,
        num_iters=1,
        optimize_scales=False,
    )
    output_path = tmp_path / "pseudo_labels.npz"
    save_pseudo_label_npz(output_path, result, extra={"split": np.array(["train"])})
    dataset = result.to_frame_dataset()

    with np.load(output_path) as payload:
        assert set(payload.files) == {
            "confidences",
            "iterations",
            "joints_2d",
            "joints_3d",
            "projected_joints_2d",
            "reprojection_errors",
            "split",
        }
        assert payload["joints_3d"].shape == (2, model.NUM_JOINTS, 3)
        assert payload["iterations"].tolist() == [1, 1]

    assert result.joints_3d.shape == (2, model.NUM_JOINTS, 3)
    assert result.projected_joints_2d.shape == joints_2d.shape
    assert result.reprojection_errors.shape == (2,)
    assert result.mean_reprojection_error >= 0.0
    assert len(result.fit_losses) == 1
    assert isinstance(dataset, FrameDataset)
    assert len(dataset) == 2


def test_pseudo_labeling_validates_shapes_and_reserved_save_keys(tmp_path) -> None:
    model = build_layer("spinetrack")
    camera = make_weak_perspective_camera("cpu")
    joints_2d = torch.zeros(1, model.NUM_JOINTS, 2)

    with pytest.raises(ValueError, match="joints_3d"):
        train_pseudo_label_priors(torch.zeros(1, model.NUM_JOINTS, 2), model=model)

    with pytest.raises(ValueError, match="confidences"):
        fit_2d_pseudo_labels(
            joints_2d,
            camera,
            model=model,
            confidences=torch.ones(1, model.NUM_JOINTS, 1),
        )

    result = fit_2d_pseudo_labels(
        joints_2d,
        camera,
        model=model,
        num_iters=1,
        optimize_scales=False,
    )
    with pytest.raises(ValueError, match="reserved"):
        save_pseudo_label_npz(
            tmp_path / "pseudo_labels.npz",
            result,
            extra={"joints_3d": np.zeros(1)},
        )
