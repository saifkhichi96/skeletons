from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from skeletons import SUPPORTED_SKELETONS, build_layer
from skeletons.artifacts import (
    log_status,
    make_run_name,
    resolve_work_dir,
    save_json,
    write_last_checkpoint,
)
from skeletons.fitting import (
    FrameDataset,
    JointLimitTrainer,
    PoseVAE,
    PoseVAETrainer,
    save_fitting_prior_checkpoint,
)


def main() -> None:
    script_name = Path(__file__).stem
    parser = argparse.ArgumentParser(
        description="Train generic fitting priors from a shared .npz joints dataset.",
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to an .npz file with joints_3d shaped as [N, J, 3].",
    )
    parser.add_argument(
        "--skeleton",
        choices=SUPPORTED_SKELETONS,
        default="human36m",
        help="Skeleton layout used by the dataset arrays.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to work_dirs/train_prior/<skeleton>_<dataset_stem>.",
    )
    parser.add_argument("--output", type=Path, dest="work_dir", help=argparse.SUPPRESS)
    args = parser.parse_args()

    run_name = make_run_name(args.skeleton, args.dataset.stem)
    work_dir = resolve_work_dir(
        Path(__file__),
        run_name=run_name,
        work_dir=args.work_dir,
    )
    save_json(
        work_dir / "config.json",
        {
            "dataset": str(args.dataset),
            "skeleton": args.skeleton,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "num_hidden_layers": args.num_hidden_layers,
            "work_dir": str(work_dir),
        },
    )
    log_status(script_name, f"work_dir={work_dir}")
    log_status(script_name, f"loading rig {args.skeleton!r}")
    model = build_layer(args.skeleton)
    log_status(script_name, f"loading dataset from {args.dataset}")
    dataset = FrameDataset.from_npz(args.dataset, expected_num_joints=model.NUM_JOINTS)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    log_status(
        script_name,
        f"loaded {len(dataset)} frames with {model.NUM_JOINTS} joints and batch_size={args.batch_size}",
    )

    log_status(script_name, "fitting joint-limit prior")
    joint_limit_trainer = JointLimitTrainer(model=model)
    joint_limit_prior = joint_limit_trainer.fit_from_loader(loader)
    log_status(script_name, "joint-limit prior ready")

    log_status(
        script_name,
        f"training pose VAE for {args.epochs} epochs (latent_dim={args.latent_dim}, hidden_dim={args.hidden_dim})",
    )
    vae = PoseVAE(
        num_joints=model.NUM_JOINTS - 1,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
    )
    trainer = PoseVAETrainer(vae, model=model)
    history: list[dict[str, float | int]] = []
    for epoch in range(args.epochs):
        state = trainer.train_epoch(loader)
        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": float(state.loss),
            "recon": float(state.metrics.get("recon", 0.0)),
            "kl": float(state.metrics.get("kl", 0.0)),
        }
        history.append(epoch_metrics)
        log_status(
            script_name,
            f"epoch [{epoch + 1}/{args.epochs}] "
            f"loss={epoch_metrics['loss']:.6f} "
            f"recon={epoch_metrics['recon']:.6f} "
            f"kl={epoch_metrics['kl']:.6f}",
        )

    checkpoint_path = work_dir / f"epoch_{args.epochs}.pth"
    save_fitting_prior_checkpoint(
        checkpoint_path,
        model=model,
        pose_prior=vae,
        joint_limit_prior=joint_limit_prior,
        history=history,
    )
    write_last_checkpoint(work_dir, checkpoint_path)
    save_json(
        work_dir / "metrics.json",
        {
            "final_epoch": history[-1]["epoch"] if history else 0,
            "history": history,
            "checkpoint": str(checkpoint_path),
        },
    )
    log_status(script_name, f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
