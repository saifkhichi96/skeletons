from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from _synthetic import make_random_walk_sequences
from _work_dir import log_status, make_run_name, resolve_work_dir, save_json
from torch.utils.data import DataLoader

from differential_skeletons import build_layer
from differential_skeletons.fitting import (
    MotionSmoothnessPrior,
    SequenceDataset,
    SkeletalFitter,
    TemporalPrior,
    TemporalPriorTrainer,
)


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
    _, _, _, train_joints = make_random_walk_sequences(
        model,
        batch_size=args.train_sequences,
        seq_len=args.seq_len,
        pose_step_std=0.03,
        transl_step_std=3.0,
        depth=2500.0,
        device=device,
    )
    _, _, _, clean_eval = make_random_walk_sequences(
        model,
        batch_size=args.eval_sequences,
        seq_len=args.seq_len,
        pose_step_std=0.03,
        transl_step_std=3.0,
        depth=2500.0,
        device=device,
    )
    noisy_eval = clean_eval + args.noise_std * torch.randn_like(clean_eval)

    temporal_prior = TemporalPrior(pose_dim=(model.NUM_JOINTS - 1) * 6, hidden_dim=256)
    trainer = TemporalPriorTrainer(temporal_prior, model=model, device=device)
    loader = DataLoader(
        SequenceDataset(train_joints.cpu()), batch_size=16, shuffle=True
    )
    for epoch in range(1, args.prior_epochs + 1):
        state = trainer.train_epoch(loader)
        log_status(
            Path(__file__).stem, f"temporal prior epoch={epoch} loss={state.loss:.5f}"
        )

    fitter = SkeletalFitter(
        model=model,
        temporal_prior=temporal_prior,
        smoothness_prior=MotionSmoothnessPrior(
            velocity_weight=1.0, acceleration_weight=2.0
        ),
        device=device,
    )
    result = fitter.fit_sequence_3d(
        noisy_eval,
        num_iters=args.fit_iters,
        optimize_scales=False,
        use_pose_prior_latent=False,
    )
    denoised = result.model_output.joints.reshape_as(clean_eval)

    noisy_mpjpe = (
        torch.linalg.vector_norm(noisy_eval - clean_eval, dim=-1).mean().item()
    )
    denoised_mpjpe = (
        torch.linalg.vector_norm(denoised - clean_eval, dim=-1).mean().item()
    )
    metrics = {
        "noisy_mpjpe": noisy_mpjpe,
        "denoised_mpjpe": denoised_mpjpe,
        **result.losses,
    }
    save_json(work_dir / "metrics.json", metrics)
    log_status(
        Path(__file__).stem,
        f"noisy_mpjpe={noisy_mpjpe:.4f} denoised_mpjpe={denoised_mpjpe:.4f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
