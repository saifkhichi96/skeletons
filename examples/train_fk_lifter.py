from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch.utils.data import DataLoader, TensorDataset

from skeletons import (
    SUPPORTED_SKELETONS,
    ForwardKinematicsLoss,
    build_layer,
)
from skeletons.artifacts import (
    log_status,
    make_run_name,
    resolve_work_dir,
    save_json,
    write_last_checkpoint,
)
from skeletons.lifting import (
    PoseLifter,
    evaluate_lifter,
    train_lifter_epoch,
)
from skeletons.synthetic import generate_orthographic_lifting_dataset


def main() -> int:
    script_name = Path(__file__).stem
    parser = argparse.ArgumentParser(
        description="Train a simple 2D-to-3D lifting model with ForwardKinematicsLoss.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pose-std", type=float, default=0.20)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--pose-reg", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skeleton",
        choices=SUPPORTED_SKELETONS,
        default="human36m",
        help="Skeleton layout used for synthetic data generation and FK supervision.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to work_dirs/train_fk_lifter/<skeleton>_synthetic_seed<seed>.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dir = resolve_work_dir(
        Path(__file__),
        run_name=make_run_name(args.skeleton, "synthetic", f"seed{args.seed}"),
        work_dir=args.work_dir,
    )
    save_json(
        work_dir / "config.json",
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_samples": args.train_samples,
            "val_samples": args.val_samples,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "pose_std": args.pose_std,
            "noise_std": args.noise_std,
            "pose_reg": args.pose_reg,
            "seed": args.seed,
            "skeleton": args.skeleton,
            "device": str(device),
            "work_dir": str(work_dir),
        },
    )
    log_status(script_name, f"work_dir={work_dir}")
    log_status(script_name, f"device={device}")

    log_status(script_name, f"loading rig {args.skeleton!r}")
    model = build_layer(args.skeleton).to(device)
    fk_loss = ForwardKinematicsLoss(model)
    lifter = PoseLifter(model.NUM_JOINTS, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(lifter.parameters(), lr=args.lr)

    log_status(
        script_name,
        f"building synthetic train/val datasets (train={args.train_samples}, val={args.val_samples})",
    )
    train_dataset = generate_orthographic_lifting_dataset(
        model,
        num_samples=args.train_samples,
        pose_std=args.pose_std,
        noise_std=args.noise_std,
        device=device,
    )
    val_dataset = generate_orthographic_lifting_dataset(
        model,
        num_samples=args.val_samples,
        pose_std=args.pose_std,
        noise_std=args.noise_std,
        device=device,
    )
    train_2d = train_dataset.keypoints_2d.cpu()
    train_3d = train_dataset.joints_3d.cpu()
    val_2d = val_dataset.keypoints_2d.cpu()
    val_3d = val_dataset.joints_3d.cpu()
    log_status(script_name, "dataset generation complete")

    train_loader = DataLoader(
        TensorDataset(train_2d, train_3d),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_2d, val_3d),
        batch_size=args.batch_size,
        shuffle=False,
    )
    log_status(
        script_name,
        f"starting training for {args.epochs} epochs with batch_size={args.batch_size} and hidden_dim={args.hidden_dim}",
    )

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_state = train_lifter_epoch(
            lifter,
            model,
            train_loader,
            optimizer=optimizer,
            fk_loss=fk_loss,
            device=device,
            pose_reg=args.pose_reg,
        )
        train_loss = train_state.loss
        val_state = evaluate_lifter(
            lifter,
            model,
            val_loader,
            fk_loss=fk_loss,
            device=device,
        )
        val_fk = val_state.fk_loss
        val_mpjpe = val_state.mpjpe
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_fk": float(val_fk),
                "val_mpjpe": float(val_mpjpe),
            }
        )
        log_status(
            script_name,
            f"epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_fk={val_fk:.4f} "
            f"val_mpjpe={val_mpjpe:.4f}",
        )

    sample_2d = val_2d[:1].to(device)
    with torch.no_grad():
        sample_pose = lifter(sample_2d)
        sample_3d = model(full_pose=sample_pose).joints[0].cpu()
    log_status(
        script_name, f"example predicted 3D joints shape={tuple(sample_3d.shape)}"
    )
    log_status(script_name, f"first five joints=\n{sample_3d[:5]}")

    checkpoint = {
        "format_version": 1,
        "model_type": "pose_lifter",
        "skeleton": model.spec.name,
        "state_dict": lifter.state_dict(),
        "hidden_dim": args.hidden_dim,
        "history": history,
        "sample_prediction": sample_3d,
    }
    checkpoint_path = work_dir / f"epoch_{args.epochs}.pth"
    torch.save(checkpoint, checkpoint_path)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
