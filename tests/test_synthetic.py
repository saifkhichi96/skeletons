from __future__ import annotations

import pytest
import torch

from skeletons import (
    SyntheticFittingDataset,
    build_layer,
    generate_orthographic_lifting_dataset,
    generate_random_pose_batch,
    generate_random_walk_sequences,
    generate_synthetic_fitting_dataset,
    make_perspective_camera,
    make_weak_perspective_camera,
)
from skeletons.fitting import FrameDataset


def test_generate_random_pose_batch_is_deterministic_with_generator() -> None:
    model = build_layer("human36m")
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)

    batch_a = generate_random_pose_batch(
        model,
        batch_size=3,
        pose_std=0.1,
        device="cpu",
        generator=generator_a,
    )
    batch_b = generate_random_pose_batch(
        model,
        batch_size=3,
        pose_std=0.1,
        device="cpu",
        generator=generator_b,
    )

    assert batch_a.full_pose.shape == (3, model.NUM_JOINTS, 3)
    assert batch_a.transl.shape == (3, 3)
    assert batch_a.scales.shape == (3, model.NUM_JOINTS, 3)
    assert batch_a.joints.shape == (3, model.NUM_JOINTS, 3)
    assert torch.allclose(batch_a.full_pose, batch_b.full_pose)
    assert torch.allclose(batch_a.transl, batch_b.transl)
    assert torch.allclose(batch_a.joints, batch_b.joints)


def test_generate_random_walk_sequences_shapes_and_depth() -> None:
    model = build_layer("spinetrack")
    batch = generate_random_walk_sequences(
        model,
        batch_size=2,
        seq_len=5,
        pose_step_std=0.01,
        transl_step_std=0.1,
        depth=1200.0,
        device="cpu",
        generator=torch.Generator().manual_seed(5),
    )

    assert batch.full_pose.shape == (2, 5, model.NUM_JOINTS, 3)
    assert batch.transl.shape == (2, 5, 3)
    assert batch.scales.shape == (2, model.NUM_JOINTS, 3)
    assert batch.joints.shape == (2, 5, model.NUM_JOINTS, 3)
    assert torch.all(batch.transl[..., 2] > 1199.0)


def test_generate_orthographic_lifting_dataset_root_centers_and_normalizes_2d() -> None:
    model = build_layer("human36m")
    dataset = generate_orthographic_lifting_dataset(
        model,
        num_samples=4,
        pose_std=0.1,
        noise_std=0.0,
        device="cpu",
        generator=torch.Generator().manual_seed(9),
    )

    root_keypoints = dataset.keypoints_2d[:, model.root_index]
    max_radius = torch.linalg.vector_norm(dataset.keypoints_2d, dim=-1).amax(dim=-1)

    assert dataset.keypoints_2d.shape == (4, model.NUM_JOINTS, 2)
    assert dataset.joints_3d.shape == (4, model.NUM_JOINTS, 3)
    assert dataset.full_pose.shape == (4, model.NUM_JOINTS, 3)
    assert torch.allclose(root_keypoints, torch.zeros_like(root_keypoints))
    assert torch.allclose(max_radius, torch.ones_like(max_radius), atol=1e-6)


def test_default_synthetic_cameras_project_expected_shapes() -> None:
    joints = torch.tensor([[[0.0, 0.0, 1000.0], [10.0, -5.0, 1000.0]]])
    perspective = make_perspective_camera("cpu")
    weak_perspective = make_weak_perspective_camera("cpu")

    assert perspective.project(joints).shape == (1, 2, 2)
    assert weak_perspective.project(joints).shape == (1, 2, 2)


def test_generate_synthetic_fitting_dataset_includes_2d_and_camera_payload() -> None:
    model = build_layer("spinetrack")
    dataset = generate_synthetic_fitting_dataset(
        model,
        num_frames=3,
        pose_std=0.05,
        noise_std=0.0,
        confidence_dropout=0.5,
        device="cpu",
        generator=torch.Generator().manual_seed(29),
    )
    first_sample = dataset.frame_dataset[0]

    assert isinstance(dataset, SyntheticFittingDataset)
    assert isinstance(dataset.frame_dataset, FrameDataset)
    assert dataset.pose_batch.joints.shape == (3, model.NUM_JOINTS, 3)
    assert dataset.clean_joints_2d.shape == (3, model.NUM_JOINTS, 2)
    assert dataset.noisy_joints_2d.shape == (3, model.NUM_JOINTS, 2)
    assert dataset.confidences.shape == (3, model.NUM_JOINTS)
    assert torch.allclose(dataset.clean_joints_2d, dataset.noisy_joints_2d)
    assert "joints_2d" in first_sample
    assert "confidences" in first_sample
    assert {"fx", "fy", "cx", "cy"}.issubset(first_sample)
    assert dataset.frame_dataset.metadata["synthetic"] is True


def test_synthetic_generators_validate_positive_sizes() -> None:
    model = build_layer("human36m")

    with pytest.raises(ValueError, match="batch_size"):
        generate_random_pose_batch(model, batch_size=0, pose_std=0.1)

    with pytest.raises(ValueError, match="seq_len"):
        generate_random_walk_sequences(
            model,
            batch_size=1,
            seq_len=0,
            pose_step_std=0.1,
            transl_step_std=0.1,
            depth=1.0,
        )

    with pytest.raises(ValueError, match="num_samples"):
        generate_orthographic_lifting_dataset(model, num_samples=0, pose_std=0.1)

    with pytest.raises(ValueError, match="num_frames"):
        generate_synthetic_fitting_dataset(model, num_frames=0, pose_std=0.1)

    with pytest.raises(ValueError, match="confidence_dropout"):
        generate_synthetic_fitting_dataset(
            model,
            num_frames=1,
            pose_std=0.1,
            confidence_dropout=1.1,
        )
