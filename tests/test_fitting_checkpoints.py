from __future__ import annotations

import pytest
import torch

from skeletons import build_layer
from skeletons.artifacts import write_last_checkpoint
from skeletons.fitting import (
    FittingPriorBundle,
    JointLimitPrior,
    JointLimitStatistics,
    PoseVAE,
    SkeletalFitter,
    build_fitting_prior_checkpoint,
    infer_pose_vae_config,
    load_fitting_prior_checkpoint,
    save_fitting_prior_checkpoint,
)


def _make_joint_limit_prior(num_joints: int) -> JointLimitPrior:
    stats = JointLimitStatistics(
        mean=torch.zeros(num_joints, 3),
        std=torch.ones(num_joints, 3),
        lower=-torch.ones(num_joints, 3),
        upper=torch.ones(num_joints, 3),
    )
    return JointLimitPrior(stats, barrier_scale=7.0)


def test_fitting_prior_checkpoint_roundtrip_from_run_directory(tmp_path) -> None:
    model = build_layer("spinetrack")
    pose_prior = PoseVAE(
        num_joints=model.NUM_JOINTS - 1,
        latent_dim=4,
        hidden_dim=12,
        num_hidden_layers=1,
    )
    joint_limit_prior = _make_joint_limit_prior(model.NUM_JOINTS - 1)
    checkpoint_path = tmp_path / "epoch_1.pth"

    saved_path = save_fitting_prior_checkpoint(
        checkpoint_path,
        model=model,
        pose_prior=pose_prior,
        joint_limit_prior=joint_limit_prior,
        history=[{"epoch": 1, "loss": 0.5}],
        extra={"note": "unit-test"},
    )
    write_last_checkpoint(tmp_path, saved_path)
    loaded = load_fitting_prior_checkpoint(tmp_path, skeleton="spinetrack")
    fitter = loaded.make_fitter(model=model)

    assert saved_path == checkpoint_path
    assert isinstance(loaded, FittingPriorBundle)
    assert loaded.pose_prior is not None
    assert loaded.pose_prior.latent_dim == 4
    assert loaded.joint_limit_prior is not None
    assert loaded.joint_limit_prior.barrier_scale == 7.0
    assert loaded.skeleton == "spinetrack"
    assert loaded.checkpoint_path == checkpoint_path
    assert loaded.history == ({"epoch": 1, "loss": 0.5},)
    assert loaded.metadata == {"note": "unit-test"}
    assert isinstance(fitter, SkeletalFitter)


def test_fitting_prior_checkpoint_infers_legacy_pose_config(tmp_path) -> None:
    model = build_layer("spinetrack")
    pose_prior = PoseVAE(
        num_joints=model.NUM_JOINTS - 1,
        latent_dim=3,
        hidden_dim=10,
        num_hidden_layers=2,
    )
    checkpoint = build_fitting_prior_checkpoint(model=model, pose_prior=pose_prior)
    checkpoint.pop("pose_prior_config")
    checkpoint_path = tmp_path / "legacy.pth"
    torch.save(checkpoint, checkpoint_path)

    loaded = load_fitting_prior_checkpoint(checkpoint_path, skeleton="spinetrack")

    assert infer_pose_vae_config(pose_prior.state_dict()) == {
        "num_joints": model.NUM_JOINTS - 1,
        "latent_dim": 3,
        "hidden_dim": 10,
        "num_hidden_layers": 2,
    }
    assert loaded.pose_prior is not None
    assert loaded.pose_prior.num_joints == model.NUM_JOINTS - 1
    assert loaded.pose_prior.latent_dim == 3


def test_fitting_prior_checkpoint_validates_inputs(tmp_path) -> None:
    model = build_layer("spinetrack")
    pose_prior = PoseVAE(num_joints=model.NUM_JOINTS - 1)

    with pytest.raises(ValueError, match="At least one prior"):
        build_fitting_prior_checkpoint(model=model)

    with pytest.raises(ValueError, match="reserved"):
        build_fitting_prior_checkpoint(
            model=model,
            pose_prior=pose_prior,
            extra={"pose_prior": "collision"},
        )

    checkpoint_path = save_fitting_prior_checkpoint(
        tmp_path / "epoch_1.pth",
        model=model,
        pose_prior=pose_prior,
    )
    with pytest.raises(ValueError, match="does not match"):
        load_fitting_prior_checkpoint(checkpoint_path, skeleton="human36m")
