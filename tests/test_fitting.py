from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from skeletons import (
    SkeletalModelLayer,
    build_layer,
    matrix_to_rot6d,
    rot6d_to_matrix,
)
from skeletons.fitting import (
    FrameDataset,
    JointLimitPrior,
    PerspectiveCamera,
    PoseVAE,
    PoseVAETrainer,
    SequenceDataset,
    SkeletalFitter,
    WeakPerspectiveCamera,
    prepare_frame_batch,
)
from skeletons.ik import estimate_rotations_from_joints


def _make_synthetic_batch(
    batch_size: int = 8,
) -> tuple[SkeletalModelLayer, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model = build_layer("human36m")
    generator = torch.Generator().manual_seed(42)
    global_orient = torch.randn(batch_size, 6, generator=generator) * 0.05
    global_orient[..., 0] += 1.0
    global_orient[..., 4] += 1.0
    body_pose = (
        torch.randn(batch_size, model.NUM_JOINTS - 1, 6, generator=generator) * 0.05
    )
    body_pose[..., 0] += 1.0
    body_pose[..., 4] += 1.0
    scales = torch.ones(batch_size, model.NUM_JOINTS, 3)
    transl = torch.randn(batch_size, 3, generator=generator) * 25.0
    transl[..., 2] += 2500.0
    joints = model(
        global_orient=global_orient,
        body_pose=body_pose,
        scales=scales,
        transl=transl,
        pose_repr="rot6d",
    ).joints
    return model, joints, global_orient, body_pose, transl


def test_ik_reconstructs_joints() -> None:
    model, joints, _, _, transl = _make_synthetic_batch(batch_size=2)
    centered = joints - transl[:, None, :]
    ik = estimate_rotations_from_joints(centered, model)
    full_pose = matrix_to_rot6d(ik.local_rotations)
    recon = model(full_pose=full_pose, scales=ik.scales, pose_repr="rot6d").joints
    assert torch.allclose(recon, centered, atol=1e-4)


def test_joint_limit_prior_fit() -> None:
    model, joints, _, _, transl = _make_synthetic_batch(batch_size=16)
    centered = joints - transl[:, None, :]
    prepared = prepare_frame_batch(centered, model=model)
    prior = JointLimitPrior.fit(rot6d_to_matrix(prepared.body_pose_rot6d))
    loss = prior(rot6d_to_matrix(prepared.body_pose_rot6d))
    assert torch.isfinite(loss)


def test_joint_limit_prior_gradients_with_pose_vae_latent() -> None:
    """Ensure joint-limit prior produces finite gradients when used with VAE latent optimization.
    
    This is a regression test for the issue where axis-angle conversion produced
    unstable derivatives causing NaN gradients during latent optimization.
    """
    model, joints, _, body_pose, transl = _make_synthetic_batch(batch_size=2)
    centered = joints - transl[:, None, :]
    prepared = prepare_frame_batch(centered, model=model)
    
    # Fit joint-limit prior on the synthetic data
    local_rotations = rot6d_to_matrix(prepared.body_pose_rot6d)
    prior = JointLimitPrior.fit(local_rotations)
    
    # Create a simple pose VAE
    pose_vae = PoseVAE(
        num_joints=model.NUM_JOINTS - 1,
        latent_dim=8,
        hidden_dim=64,
        num_hidden_layers=1,
    )
    pose_vae.eval()
    
    # Encode the body pose to latent space and optimize latent with joint-limit loss
    with torch.no_grad():
        mu, _ = pose_vae.encode(prepared.body_pose_rot6d)
    
    latent = torch.nn.Parameter(mu.clone())
    optimizer = torch.optim.Adam([latent], lr=1e-3)
    
    # One optimization step to check for NaN/inf in gradients
    for _ in range(1):
        optimizer.zero_grad()
        
        # Decode latent to body pose
        decoded = pose_vae.decode(latent)
        
        # Run model forward
        output = model(
            global_orient=prepared.global_orient_rot6d,
            body_pose=decoded,
            scales=torch.ones_like(prepared.scales),
            transl=torch.zeros(prepared.scales.shape[0], 3),
            pose_repr="rot6d",
            return_local_rotations=True,
        )
        
        # Compute joint-limit loss
        body_local_rot = output.local_rotations[..., list(model.non_root_joint_indices), :, :]
        loss = prior(body_local_rot)
        
        # Check that loss is finite
        assert torch.isfinite(loss), f"Joint-limit loss is not finite: {loss}"
        
        # Backward pass should not produce NaN gradients
        loss.backward()
        
        assert latent.grad is not None
        assert torch.isfinite(latent.grad).all(), (
            f"Latent gradients contain NaN/inf: nan={torch.isnan(latent.grad).any()}, "
            f"inf={torch.isinf(latent.grad).any()}"
        )
        
        # Optimizer step should work
        optimizer.step()
        assert torch.isfinite(latent).all()


def test_pose_vae_training_step() -> None:
    model, joints, _, _, transl = _make_synthetic_batch(batch_size=16)
    centered = joints - transl[:, None, :]
    dataset = FrameDataset(centered, expected_num_joints=model.NUM_JOINTS)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    prior = PoseVAE(
        num_joints=model.NUM_JOINTS - 1,
        latent_dim=8,
        hidden_dim=64,
        num_hidden_layers=1,
    )
    trainer = PoseVAETrainer(prior, model=model, lr=1e-3)
    state = trainer.train_epoch(loader)
    assert state.loss > 0.0


def test_fitter_recovers_synthetic_joints() -> None:
    model, joints, _, _, _ = _make_synthetic_batch(batch_size=2)
    fitter = SkeletalFitter(model=model)
    result = fitter.fit_3d(
        joints,
        num_iters=80,
        lr=5e-2,
        optimize_scales=False,
        use_pose_prior_latent=False,
    )
    pred = result.model_output.joints
    error = torch.linalg.vector_norm(pred - joints, dim=-1).mean().item()
    assert error < 5.0


def test_2d_fitter_runs() -> None:
    model, joints, _, _, _ = _make_synthetic_batch(batch_size=1)
    camera = PerspectiveCamera(
        fx=torch.tensor(1000.0),
        fy=torch.tensor(1000.0),
        cx=torch.tensor(512.0),
        cy=torch.tensor(512.0),
    )
    target_2d = camera.project(joints)
    fitter = SkeletalFitter(model=model)
    result = fitter.fit_2d(
        target_2d,
        camera,
        num_iters=20,
        lr=1e-2,
        optimize_scales=False,
        use_pose_prior_latent=False,
    )
    assert result.model_output.joints.shape == joints.shape


def test_camera_projection_broadcasts_per_frame_intrinsics() -> None:
    points = torch.tensor(
        [
            [[1.0, 2.0, 10.0], [3.0, 4.0, 10.0]],
            [[2.0, 1.0, 20.0], [4.0, 2.0, 20.0]],
        ]
    )
    camera = PerspectiveCamera(
        fx=torch.tensor([1000.0, 2000.0]),
        fy=torch.tensor([500.0, 1000.0]),
        cx=torch.tensor([10.0, 20.0]),
        cy=torch.tensor([30.0, 40.0]),
    )
    weak_camera = WeakPerspectiveCamera(
        scale=torch.tensor([2.0, 3.0]),
        tx=torch.tensor([10.0, 20.0]),
        ty=torch.tensor([30.0, 40.0]),
    )

    projected = camera.project(points)
    weak_projected = weak_camera.project(points)

    assert projected.shape == (2, 2, 2)
    assert torch.allclose(projected[0, 0], torch.tensor([110.0, 130.0]))
    assert torch.allclose(projected[1, 0], torch.tensor([220.0, 90.0]))
    assert torch.allclose(weak_projected[0, 0], torch.tensor([12.0, 34.0]))
    assert torch.allclose(weak_projected[1, 0], torch.tensor([26.0, 43.0]))


def test_default_fitter_uses_parameter_free_model() -> None:
    fitter = SkeletalFitter(model=build_layer("human36m"))

    assert not hasattr(fitter.model, "global_orient")
    assert not hasattr(fitter.model, "body_pose")
    assert not hasattr(fitter.model, "scales")
    assert not hasattr(fitter.model, "transl")

    model, joints, _, _, _ = _make_synthetic_batch(batch_size=1)
    result = fitter.fit_3d(
        joints,
        num_iters=2,
        lr=1e-2,
        optimize_scales=False,
        use_pose_prior_latent=False,
    )
    assert result.model_output.joints.shape == joints.shape


def test_frame_dataset_from_npz_accepts_alias_keys(tmp_path) -> None:
    model = build_layer("human36m")
    num_frames = 3
    joints_3d = np.random.randn(num_frames, model.NUM_JOINTS, 4).astype("float32")
    joints_2d_with_conf = np.random.randn(num_frames, model.NUM_JOINTS, 3).astype(
        "float32"
    )
    camera_translation = np.random.randn(num_frames, 3).astype("float32")
    camera_rotation = np.random.randn(num_frames, 3, 3).astype("float32")

    path = tmp_path / "frame_aliases.npz"
    np.savez(
        path,
        S=joints_3d,
        Part=joints_2d_with_conf,
        Fx=np.full((num_frames,), 1000.0, dtype="float32"),
        FY=np.full((num_frames,), 900.0, dtype="float32"),
        PrincipalX=np.full((num_frames,), 512.0, dtype="float32"),
        principal_point_y=np.full((num_frames,), 384.0, dtype="float32"),
        Cam_T=camera_translation,
        Camera_Rot=camera_rotation,
        frame_index=np.arange(num_frames, dtype="int64"),
    )

    dataset = FrameDataset.from_npz(path, expected_num_joints=model.NUM_JOINTS)
    sample = dataset[0]

    assert sample["joints_3d"].shape == (model.NUM_JOINTS, 3)
    assert sample["joints_2d"].shape == (model.NUM_JOINTS, 2)
    assert sample["confidences"].shape == (model.NUM_JOINTS,)
    assert torch.allclose(
        sample["confidences"], torch.from_numpy(joints_2d_with_conf[0, :, 2])
    )
    assert set(dataset.cameras) == {
        "fx",
        "fy",
        "cx",
        "cy",
        "camera_translation",
        "camera_rotation",
    }
    assert torch.allclose(
        dataset.cameras["camera_translation"][0],
        torch.from_numpy(camera_translation[0]),
    )
    assert torch.allclose(
        dataset.cameras["camera_rotation"][0],
        torch.from_numpy(camera_rotation[0]),
    )
    assert "frame_index" in dataset.metadata


def test_frame_dataset_accepts_scalar_camera_constants() -> None:
    model = build_layer("human36m")
    joints_3d = torch.zeros(2, model.NUM_JOINTS, 3)
    dataset = FrameDataset(
        joints_3d,
        cameras={
            "fx": torch.tensor(1000.0),
            "fy": torch.tensor(900.0),
            "cx": torch.tensor(512.0),
            "cy": torch.tensor(384.0),
        },
        expected_num_joints=model.NUM_JOINTS,
    )
    sample = dataset[1]

    assert sample["fx"].shape == ()
    assert sample["fx"].item() == 1000.0
    assert sample["cy"].item() == 384.0


def test_sequence_dataset_from_npz_accepts_alias_keys(tmp_path) -> None:
    model = build_layer("human36m")
    num_sequences = 2
    num_frames = 4
    joints_3d = np.random.randn(num_sequences, num_frames, model.NUM_JOINTS, 3).astype(
        "float32"
    )
    joints_2d = np.random.randn(num_sequences, num_frames, model.NUM_JOINTS, 2).astype(
        "float32"
    )
    confidences = np.random.rand(num_sequences, num_frames, model.NUM_JOINTS).astype(
        "float32"
    )

    path = tmp_path / "sequence_aliases.npz"
    np.savez(
        path,
        Keypoints3D=joints_3d,
        Pose2D=joints_2d,
        Scores=confidences,
        Focal_Length_X=np.full((num_sequences,), 1200.0, dtype="float32"),
        Focal_Length_Y=np.full((num_sequences,), 1180.0, dtype="float32"),
        C_X=np.full((num_sequences,), 640.0, dtype="float32"),
        C_Y=np.full((num_sequences,), 360.0, dtype="float32"),
        cam_translation=np.random.randn(num_sequences, 3).astype("float32"),
        Cam_R=np.random.randn(num_sequences, 3, 3).astype("float32"),
        sequence_index=np.arange(num_sequences, dtype="int64"),
    )

    dataset = SequenceDataset.from_npz(path, expected_num_joints=model.NUM_JOINTS)
    sample = dataset[0]

    assert sample["joints_3d"].shape == (num_frames, model.NUM_JOINTS, 3)
    assert sample["joints_2d"].shape == (num_frames, model.NUM_JOINTS, 2)
    assert sample["confidences"].shape == (num_frames, model.NUM_JOINTS)
    assert set(dataset.cameras) == {
        "fx",
        "fy",
        "cx",
        "cy",
        "camera_translation",
        "camera_rotation",
    }
    assert "sequence_index" in dataset.metadata


def test_sequence_dataset_accepts_scalar_camera_constants() -> None:
    model = build_layer("human36m")
    joints_3d = torch.zeros(2, 3, model.NUM_JOINTS, 3)
    dataset = SequenceDataset(
        joints_3d,
        cameras={
            "fx": torch.tensor(1200.0),
            "fy": torch.tensor(1180.0),
        },
        expected_num_joints=model.NUM_JOINTS,
    )
    sample = dataset[0]

    assert sample["fx"].shape == ()
    assert sample["fx"].item() == 1200.0
    assert sample["fy"].item() == 1180.0
