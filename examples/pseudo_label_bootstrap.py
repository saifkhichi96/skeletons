from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
from _synthetic import make_perspective_camera, make_random_pose_batch
from _work_dir import log_status, make_run_name, resolve_work_dir, save_json
from torch.utils.data import DataLoader

from differential_skeletons import build_layer
from differential_skeletons.fitting import (
    FrameDataset,
    JointLimitTrainer,
    PoseVAE,
    PoseVAETrainer,
    SkeletalFitter,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate 3D pseudo-labels from 2D detections via skeletal fitting."
    )
    parser.add_argument("--skeleton", default="human36m")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--pose-std", type=float, default=0.18)
    parser.add_argument("--noise-std", type=float, default=8.0)
    parser.add_argument("--prior-epochs", type=int, default=4)
    parser.add_argument("--fit-iters", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dir = resolve_work_dir(
        Path(__file__),
        run_name=make_run_name(args.skeleton, "bootstrap", f"seed{args.seed}"),
        work_dir=args.work_dir,
    )

    model = build_layer(args.skeleton).to(device)
    camera = make_perspective_camera(device)
    _, _, _, joints_3d = make_random_pose_batch(
        model, batch_size=args.num_samples, pose_std=args.pose_std, device=device
    )
    joints_2d = camera.project(joints_3d)
    noisy_2d = joints_2d + args.noise_std * torch.randn_like(joints_2d)
    confidences = torch.ones(args.num_samples, model.NUM_JOINTS, device=device)
    confidences[torch.rand_like(confidences) < 0.15] = 0.0

    dataset = FrameDataset(
        joints_3d.cpu(),
        joints_2d=noisy_2d.cpu(),
        confidences=confidences.cpu(),
        expected_num_joints=model.NUM_JOINTS,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    joint_limit_prior = JointLimitTrainer(model=model).fit_from_loader(loader)
    pose_prior = PoseVAE(num_joints=model.NUM_JOINTS - 1, latent_dim=16, hidden_dim=256)
    pose_trainer = PoseVAETrainer(pose_prior, model=model, device=device)
    for epoch in range(1, args.prior_epochs + 1):
        state = pose_trainer.train_epoch(loader)
        log_status(Path(__file__).stem, f"prior epoch={epoch} loss={state.loss:.5f}")

    fitter = SkeletalFitter(
        model=model,
        pose_prior=pose_prior,
        joint_limit_prior=joint_limit_prior,
        device=device,
    )
    pseudo_labels = []
    reprojection_errors = []
    for index in range(args.num_samples):
        result = fitter.fit_2d(
            noisy_2d[index : index + 1],
            camera,
            confidences=confidences[index : index + 1],
            num_iters=args.fit_iters,
            optimize_scales=True,
            use_pose_prior_latent=True,
        )
        pseudo_labels.append(result.model_output.joints[0].detach().cpu())
        reprojection_errors.append(float(result.losses.get("reprojection", 0.0)))

    pseudo_joints_3d = torch.stack(pseudo_labels, dim=0)
    pseudo_path = work_dir / "pseudo_labels.npz"
    np.savez_compressed(
        pseudo_path,
        joints_2d=noisy_2d.cpu().numpy(),
        confidences=confidences.cpu().numpy(),
        joints_3d=pseudo_joints_3d.numpy(),
    )

    pseudo_mpjpe = (
        torch.linalg.vector_norm(pseudo_joints_3d.to(device) - joints_3d, dim=-1)
        .mean()
        .item()
    )
    metrics = {
        "pseudo_mpjpe": pseudo_mpjpe,
        "mean_reprojection": float(sum(reprojection_errors) / len(reprojection_errors)),
        "output": str(pseudo_path),
    }
    save_json(work_dir / "metrics.json", metrics)
    log_status(
        Path(__file__).stem, f"pseudo_mpjpe={pseudo_mpjpe:.4f} output={pseudo_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
