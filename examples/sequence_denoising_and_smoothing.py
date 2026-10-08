from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from skeletons import (
    build_layer,
    denoise_joint_sequences,
    mean_per_joint_position_error,
    train_temporal_prior,
)
from skeletons.artifacts import (
    log_status,
    make_run_name,
    resolve_work_dir,
    save_json,
)
from skeletons.fitting import MotionSmoothnessPrior
from skeletons.synthetic import generate_random_walk_sequences


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Denoise noisy 3D motion with sequence fitting and learned temporal priors."
    )
    parser.add_argument("--skeleton", default="human36m")
    parser.add_argument("--train-sequences", type=int, default=128)
    parser.add_argument("--eval-sequences", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--noise-std", type=float, default=18.0)
    parser.add_argument("--prior-epochs", type=int, default=6)
    parser.add_argument("--fit-iters", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dir = resolve_work_dir(
        Path(__file__),
        run_name=make_run_name(args.skeleton, "denoise", f"seed{args.seed}"),
        work_dir=args.work_dir,
    )
    save_json(
        work_dir / "config.json",
        vars(args) | {"device": str(device), "work_dir": str(work_dir)},
    )

    model = build_layer(args.skeleton).to(device)
    train_batch = generate_random_walk_sequences(
        model,
        batch_size=args.train_sequences,
        seq_len=args.seq_len,
        pose_step_std=0.03,
        transl_step_std=3.0,
        depth=2500.0,
        device=device,
    )
    eval_batch = generate_random_walk_sequences(
        model,
        batch_size=args.eval_sequences,
        seq_len=args.seq_len,
        pose_step_std=0.03,
        transl_step_std=3.0,
        depth=2500.0,
        device=device,
    )
    train_joints = train_batch.joints
    clean_eval = eval_batch.joints
    noisy_eval = clean_eval + args.noise_std * torch.randn_like(clean_eval)

    temporal_training = train_temporal_prior(
        train_joints,
        model=model,
        batch_size=16,
        epochs=args.prior_epochs,
        hidden_dim=256,
        device=device,
    )
    for state in temporal_training.history:
        log_status(
            Path(__file__).stem,
            f"temporal prior epoch={state.epoch} loss={state.loss:.5f}",
        )

    result = denoise_joint_sequences(
        noisy_eval,
        model=model,
        temporal_prior=temporal_training.temporal_prior,
        smoothness_prior=MotionSmoothnessPrior(
            velocity_weight=1.0, acceleration_weight=2.0
        ),
        num_iters=args.fit_iters,
        optimize_scales=False,
        use_pose_prior_latent=False,
        device=device,
    )
    denoised = result.joints_3d.to(device)

    noisy_mpjpe = mean_per_joint_position_error(
        noisy_eval,
        clean_eval,
    )
    denoised_mpjpe = mean_per_joint_position_error(
        denoised,
        clean_eval,
    )
    noisy_mpjpe_value = float(noisy_mpjpe.item())
    denoised_mpjpe_value = float(denoised_mpjpe.item())
    metrics = {
        "noisy_mpjpe": noisy_mpjpe_value,
        "denoised_mpjpe": denoised_mpjpe_value,
        **result.losses,
    }
    save_json(work_dir / "metrics.json", metrics)
    log_status(
        Path(__file__).stem,
        f"noisy_mpjpe={noisy_mpjpe_value:.4f} "
        f"denoised_mpjpe={denoised_mpjpe_value:.4f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
