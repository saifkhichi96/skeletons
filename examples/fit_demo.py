from __future__ import annotations

import argparse
from pathlib import Path

import torch

from skeletons import SUPPORTED_SKELETONS, build_layer
from skeletons.fitting import (
    FrameDataset,
    PerspectiveCamera,
    SkeletalFitter,
    load_fitting_prior_checkpoint,
)


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
    parser.add_argument(
        "--frame-index",
        "--sample-index",
        type=int,
        default=0,
        dest="frame_index",
        help="Starting frame index in the dataset.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=1,
        help="Number of consecutive frames to fit. If > 1, uses warm-start from previous frame.",
    )
    parser.add_argument(
        "--fit-iters",
        type=int,
        default=None,
        help="Override optimizer iterations for smoke tests or quick demos.",
    )
    parser.add_argument(
        "--priors",
        type=Path,
        default=None,
        help="Optional prior checkpoint path, last_checkpoint file, or work_dir.",
    )
    args = parser.parse_args()

    model = build_layer(args.skeleton)
    dataset = FrameDataset.from_npz(args.dataset, expected_num_joints=model.NUM_JOINTS)

    if args.priors is not None:
        priors = load_fitting_prior_checkpoint(args.priors, skeleton=model.spec.name)
        fitter = priors.make_fitter(model=model)
    else:
        fitter = SkeletalFitter(model=model)

    # Fit a sequence of frames with warm-start from previous frame.
    start_idx = args.frame_index
    seq_length = args.seq_length
    num_iters_per_frame = (
        (200 if args.mode == "3d" else 300)
        if args.fit_iters is None
        else args.fit_iters
    )

    # Load the sequence
    frames = [
        dataset[start_idx + i] for i in range(min(seq_length, len(dataset) - start_idx))
    ]
    print(
        f"Fitting {len(frames)} frame(s) for {model.spec.name} "
        f"starting from index {start_idx}."
    )

    init_global_orient = None
    init_body_pose = None
    init_scales = None
    init_transl = None

    for frame_idx, sample in enumerate(frames):
        abs_frame_idx = start_idx + frame_idx
        print(f"\n--- Frame {abs_frame_idx} ({frame_idx + 1}/{len(frames)}) ---")

        if args.mode == "3d":
            result = fitter.fit_3d(
                sample["joints_3d"].unsqueeze(0),
                num_iters=num_iters_per_frame,
                init_global_orient=init_global_orient,
                init_body_pose=init_body_pose,
                init_scales=init_scales,
                init_transl=init_transl,
            )
            print(f"Losses: {result.losses}")

            # Extract fitted parameters for warm-start of next frame.
            with torch.no_grad():
                from skeletons.rotations import matrix_to_rot6d

                init_global_orient = matrix_to_rot6d(
                    result.model_output.local_rotations[..., model.root_index, :, :]
                )
                init_body_pose = matrix_to_rot6d(
                    result.model_output.local_rotations[
                        ..., list(model.non_root_joint_indices), :, :
                    ]
                )
                # Use fitted scales if available, otherwise let fitter re-estimate.
                init_scales = (
                    result.model_output.scales.detach()
                    if result.model_output.scales is not None
                    else None
                )
                init_transl = result.model_output.joints[
                    ..., model.root_index, :
                ].detach()
        else:
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
                num_iters=num_iters_per_frame,
                init_global_orient=init_global_orient,
                init_body_pose=init_body_pose,
                init_scales=init_scales,
                init_transl=init_transl,
            )
            print(f"Losses: {result.losses}")

            # Extract fitted parameters for warm-start of next frame.
            with torch.no_grad():
                from skeletons.rotations import matrix_to_rot6d

                init_global_orient = matrix_to_rot6d(
                    result.model_output.local_rotations[..., model.root_index, :, :]
                )
                init_body_pose = matrix_to_rot6d(
                    result.model_output.local_rotations[
                        ..., list(model.non_root_joint_indices), :, :
                    ]
                )
                # Use fitted scales if available, otherwise let fitter re-estimate.
                init_scales = (
                    result.model_output.scales.detach()
                    if result.model_output.scales is not None
                    else None
                )
                init_transl = result.model_output.joints[
                    ..., model.root_index, :
                ].detach()

    print(f"\nFitting complete. Final result shape: {result.model_output.joints.shape}")


if __name__ == "__main__":
    main()
