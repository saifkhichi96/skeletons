from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from skeletons import build_layer, mean_per_joint_position_error
from skeletons.artifacts import (
    log_status,
    make_run_name,
    resolve_work_dir,
    save_json,
)
from skeletons.pseudo_labeling import (
    fit_2d_pseudo_labels,
    save_pseudo_label_npz,
    train_pseudo_label_priors,
)
from skeletons.synthetic import (
    generate_random_pose_batch,
    make_perspective_camera,
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
    pose_batch = generate_random_pose_batch(
        model, batch_size=args.num_samples, pose_std=args.pose_std, device=device
    )
    joints_3d = pose_batch.joints
    joints_2d = camera.project(joints_3d)
    noisy_2d = joints_2d + args.noise_std * torch.randn_like(joints_2d)
    confidences = torch.ones(args.num_samples, model.NUM_JOINTS, device=device)
    confidences[torch.rand_like(confidences) < 0.15] = 0.0

    priors = train_pseudo_label_priors(
        joints_3d,
        model=model,
        batch_size=32,
        epochs=args.prior_epochs,
        latent_dim=16,
        hidden_dim=256,
        device=device,
    )
    for state in priors.history:
        log_status(
            Path(__file__).stem,
            f"prior epoch={state.epoch} loss={state.loss:.5f}",
        )

    result = fit_2d_pseudo_labels(
        noisy_2d,
        camera,
        model=model,
        confidences=confidences,
        pose_prior=priors.pose_prior,
        joint_limit_prior=priors.joint_limit_prior,
        batch_size=1,
        num_iters=args.fit_iters,
        optimize_scales=True,
        use_pose_prior_latent=True,
        device=device,
    )
    pseudo_path = work_dir / "pseudo_labels.npz"
    save_pseudo_label_npz(pseudo_path, result)

    pseudo_mpjpe = mean_per_joint_position_error(
        result.joints_3d.to(device),
        joints_3d,
    )
    metrics = {
        "pseudo_mpjpe": float(pseudo_mpjpe.item()),
        "mean_reprojection": result.mean_reprojection_error,
        "output": str(pseudo_path),
    }
    save_json(work_dir / "metrics.json", metrics)
    log_status(
        Path(__file__).stem, f"pseudo_mpjpe={pseudo_mpjpe:.4f} output={pseudo_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
