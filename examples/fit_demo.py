from __future__ import annotations

import argparse
from pathlib import Path

import torch
from _work_dir import resolve_checkpoint_reference

from differential_skeletons import SUPPORTED_SKELETONS, build_layer
from differential_skeletons.fitting import (
    FrameDataset,
    JointLimitPrior,
    JointLimitStatistics,
    PerspectiveCamera,
    PoseVAE,
    SkeletalFitter,
)


def _normalize_skeleton_name(name: str) -> str:
    return name.lower().replace("-", "_")


def _infer_pose_prior_config(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    linear_keys = sorted(
        key
        for key, value in state_dict.items()
        if key.startswith("encoder.") and key.endswith(".weight") and value.ndim == 2
    )
    if not linear_keys:
        raise ValueError("Could not infer PoseVAE architecture from the checkpoint.")
    input_dim = int(state_dict[linear_keys[0]].shape[1])
    hidden_dim = int(state_dict[linear_keys[0]].shape[0])
    latent_dim = int(state_dict["encoder_mu.weight"].shape[0])
    if input_dim % 6 != 0:
        raise ValueError(
            f"PoseVAE input dimension must be divisible by 6, got {input_dim}."
        )
    return {
        "num_joints": input_dim // 6,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "num_hidden_layers": len(linear_keys),
    }


def _load_priors(
    path: Path,
    *,
    skeleton: str,
) -> tuple[PoseVAE | None, JointLimitPrior | None]:
    checkpoint_path = resolve_checkpoint_reference(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Expected a dict checkpoint in {checkpoint_path}, got {type(checkpoint).__name__}."
        )

    checkpoint_skeleton = checkpoint.get("skeleton")
    if checkpoint_skeleton is not None and _normalize_skeleton_name(
        str(checkpoint_skeleton)
    ) != _normalize_skeleton_name(skeleton):
        raise ValueError(
            f"Prior checkpoint skeleton {checkpoint_skeleton!r} does not match requested skeleton {skeleton!r}.",
        )

    pose_prior = None
    pose_state = checkpoint.get("pose_prior")
    if pose_state is not None:
        pose_config = checkpoint.get("pose_prior_config")
        if pose_config is None:
            pose_config = _infer_pose_prior_config(pose_state)
        pose_prior = PoseVAE(**pose_config)
        pose_prior.load_state_dict(pose_state)
        pose_prior.eval()

    joint_limit_prior = None
    joint_limit_state = checkpoint.get("joint_limit_prior")
    if joint_limit_state is not None:
        joint_limit_config = checkpoint.get("joint_limit_prior_config", {})
        joint_limit_prior = JointLimitPrior(
            JointLimitStatistics(
                mean=joint_limit_state["mean"],
                std=joint_limit_state["std"],
                lower=joint_limit_state["lower"],
                upper=joint_limit_state["upper"],
            ),
            barrier_scale=float(joint_limit_config.get("barrier_scale", 10.0)),
        )
        joint_limit_prior.load_state_dict(joint_limit_state)
        joint_limit_prior.eval()

    if pose_prior is None and joint_limit_prior is None:
        raise ValueError(f"No supported priors were found in {path}.")
    return pose_prior, joint_limit_prior


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a skeletal model to a shared .npz dataset of 2D or 3D joints.",
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to an .npz file with joints_3d and optionally joints_2d.",
    )
    parser.add_argument(
        "--skeleton",
        choices=SUPPORTED_SKELETONS,
        default="human36m",
        help="Skeleton layout used by the dataset arrays.",
    )
    parser.add_argument("--mode", choices=("2d", "3d"), default="3d")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--priors",
        type=Path,
        default=None,
        help="Optional prior checkpoint path, last_checkpoint file, or work_dir.",
    )
    args = parser.parse_args()

    model = build_layer(args.skeleton)
    dataset = FrameDataset.from_npz(args.dataset, expected_num_joints=model.NUM_JOINTS)
    sample = dataset[args.sample_index]
    pose_prior = None
    joint_limit_prior = None
    if args.priors is not None:
        pose_prior, joint_limit_prior = _load_priors(
            args.priors, skeleton=model.spec.name
        )
    fitter = SkeletalFitter(
        model=model,
        pose_prior=pose_prior,
        joint_limit_prior=joint_limit_prior,
    )

    if args.mode == "3d":
        result = fitter.fit_3d(sample["joints_3d"].unsqueeze(0), num_iters=200)
        print({"skeleton": model.spec.name, **result.losses})
        print(result.model_output.joints.shape)
        return

    if "joints_2d" not in sample:
        raise ValueError("The dataset file does not contain joints_2d.")
    camera = PerspectiveCamera(
        fx=sample.get("fx", torch.tensor(1000.0)).reshape(()),
        fy=sample.get("fy", torch.tensor(1000.0)).reshape(()),
        cx=sample.get("cx", torch.tensor(512.0)).reshape(()),
        cy=sample.get("cy", torch.tensor(512.0)).reshape(()),
    )
    confidences = sample.get("confidences")
    result = fitter.fit_2d(
        sample["joints_2d"].unsqueeze(0),
        camera,
        confidences=None if confidences is None else confidences.unsqueeze(0),
        num_iters=300,
    )
    print({"skeleton": model.spec.name, **result.losses})
    print(result.model_output.joints.shape)


if __name__ == "__main__":
    main()
