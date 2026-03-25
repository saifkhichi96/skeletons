from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from skelix import Human36MModel, matrix_to_rot6d, rot6d_to_matrix
from skelix.fitting import (
    H36MFrameDataset,
    H36MFitter,
    JointLimitPrior,
    PoseVAE,
    H36MPoseVAETrainer,
    PerspectiveCamera,
    prepare_h36m_frame_batch,
)
from skelix.ik import estimate_rotations_from_joints


def _make_synthetic_batch(batch_size: int = 8) -> tuple[Human36MModel, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model = Human36MModel(create_global_orient=False, create_body_pose=False)
    generator = torch.Generator().manual_seed(42)
    global_orient = torch.randn(batch_size, 6, generator=generator) * 0.05
    global_orient[..., 0] += 1.0
    global_orient[..., 4] += 1.0
    body_pose = torch.randn(batch_size, model.num_joints - 1, 6, generator=generator) * 0.05
    body_pose[..., 0] += 1.0
    body_pose[..., 4] += 1.0
    bone_scales = torch.ones(batch_size, model.num_joints, 3)
    transl = torch.randn(batch_size, 3, generator=generator) * 25.0
    transl[..., 2] += 2500.0
    joints = model(
        global_orient=global_orient,
        body_pose=body_pose,
        bone_scales=bone_scales,
        transl=transl,
        pose_repr='rot6d',
    ).joints
    return model, joints, global_orient, body_pose, transl


def test_ik_reconstructs_h36m_joints() -> None:
    model, joints, _, _, transl = _make_synthetic_batch(batch_size=2)
    centered = joints - transl[:, None, :]
    ik = estimate_rotations_from_joints(centered, model)
    full_pose = matrix_to_rot6d(ik.local_rotations)
    recon = model(full_pose=full_pose, bone_scales=ik.bone_scales, pose_repr='rot6d').joints
    assert torch.allclose(recon, centered, atol=1e-4)


def test_h36m_joint_limit_prior_fit() -> None:
    model, joints, _, _, transl = _make_synthetic_batch(batch_size=16)
    centered = joints - transl[:, None, :]
    prepared = prepare_h36m_frame_batch(centered, model=model)
    prior = JointLimitPrior.fit(rot6d_to_matrix(prepared.body_pose_rot6d))
    loss = prior(rot6d_to_matrix(prepared.body_pose_rot6d))
    assert torch.isfinite(loss)


def test_h36m_pose_vae_training_step() -> None:
    model, joints, _, _, transl = _make_synthetic_batch(batch_size=16)
    centered = joints - transl[:, None, :]
    dataset = H36MFrameDataset(centered)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    prior = PoseVAE(num_joints=model.num_joints - 1, latent_dim=8, hidden_dim=64, num_hidden_layers=1)
    trainer = H36MPoseVAETrainer(prior, model=model, lr=1e-3)
    state = trainer.train_epoch(loader)
    assert state.loss > 0.0


def test_h36m_3d_fitter_recovers_synthetic_joints() -> None:
    model, joints, _, _, _ = _make_synthetic_batch(batch_size=2)
    fitter = H36MFitter(model=model)
    result = fitter.fit_3d(joints, num_iters=80, lr=5e-2, optimize_bone_scales=False, use_pose_prior_latent=False)
    pred = result.model_output.joints
    error = torch.linalg.vector_norm(pred - joints, dim=-1).mean().item()
    assert error < 5.0


def test_h36m_2d_fitter_runs() -> None:
    model, joints, _, _, _ = _make_synthetic_batch(batch_size=1)
    camera = PerspectiveCamera(
        fx=torch.tensor(1000.0),
        fy=torch.tensor(1000.0),
        cx=torch.tensor(512.0),
        cy=torch.tensor(512.0),
    )
    target_2d = camera.project(joints)
    fitter = H36MFitter(model=model)
    result = fitter.fit_2d(target_2d, camera, num_iters=20, lr=1e-2, optimize_bone_scales=False, use_pose_prior_latent=False)
    assert result.model_output.joints.shape == joints.shape
