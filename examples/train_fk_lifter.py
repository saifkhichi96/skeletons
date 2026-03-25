from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn
from _work_dir import (
    log_status,
    make_run_name,
    resolve_work_dir,
    save_json,
    write_last_checkpoint,
)
from torch.utils.data import DataLoader, TensorDataset

from differential_skeletons import (
    SUPPORTED_SKELETONS,
    ForwardKinematicsLoss,
    SkeletalModel,
    build_layer,
)


class PoseLifter(nn.Module):
    def __init__(self, num_joints: int, hidden_dim: int = 256) -> None:
        super().__init__()
        input_dim = num_joints * 2
        output_dim = num_joints * 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.num_joints = num_joints

    def forward(self, keypoints_2d: torch.Tensor) -> torch.Tensor:
        batch_size = keypoints_2d.shape[0]
        flat = keypoints_2d.reshape(batch_size, -1)
        return self.net(flat).reshape(batch_size, self.num_joints, 3)


def make_synthetic_dataset(
    model: SkeletalModel,
    *,
    num_samples: int,
    pose_std: float,
    noise_std: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        full_pose = (
            torch.randn(num_samples, model.NUM_JOINTS, 3, device=device) * pose_std
        )
        joints_3d = model(full_pose=full_pose).joints.detach()

        keypoints_2d = joints_3d[..., [0, 1]]
        keypoints_2d = keypoints_2d + noise_std * torch.randn_like(keypoints_2d)

        root_2d = keypoints_2d[:, model.root_index : model.root_index + 1]
        keypoints_2d = keypoints_2d - root_2d

        scale = (
            torch.linalg.vector_norm(keypoints_2d, dim=-1)
            .amax(dim=-1, keepdim=True)
            .clamp_min(1e-6)
        )
        keypoints_2d = keypoints_2d / scale.unsqueeze(-1)
        return keypoints_2d.cpu(), joints_3d.cpu()


def evaluate(
    lifter: PoseLifter,
    model: SkeletalModel,
    loader: DataLoader,
    fk_loss: ForwardKinematicsLoss,
    device: torch.device,
) -> tuple[float, float]:
    lifter.eval()
    total_fk = 0.0
    total_mpjpe = 0.0
    total_samples = 0

    with torch.no_grad():
        for keypoints_2d, target_3d in loader:
            keypoints_2d = keypoints_2d.to(device)
            target_3d = target_3d.to(device)

            pred_pose = lifter(keypoints_2d)
            loss = fk_loss(target_3d, full_pose=pred_pose)
            pred_3d = model(full_pose=pred_pose).joints
            mpjpe = torch.linalg.vector_norm(pred_3d - target_3d, dim=-1).mean()

            batch_size = keypoints_2d.shape[0]
            total_fk += loss.item() * batch_size
            total_mpjpe += mpjpe.item() * batch_size
            total_samples += batch_size

    return total_fk / total_samples, total_mpjpe / total_samples


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
    train_2d, train_3d = make_synthetic_dataset(
        model,
        num_samples=args.train_samples,
        pose_std=args.pose_std,
        noise_std=args.noise_std,
        device=device,
    )
    val_2d, val_3d = make_synthetic_dataset(
        model,
        num_samples=args.val_samples,
        pose_std=args.pose_std,
        noise_std=args.noise_std,
        device=device,
    )
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
        lifter.train()
        running_loss = 0.0
        num_seen = 0

        for keypoints_2d, target_3d in train_loader:
            keypoints_2d = keypoints_2d.to(device)
            target_3d = target_3d.to(device)

            optimizer.zero_grad()
            pred_pose = lifter(keypoints_2d)
            loss_fk = fk_loss(target_3d, full_pose=pred_pose)
            loss_reg = args.pose_reg * pred_pose.square().mean()
            loss = loss_fk + loss_reg
            loss.backward()
            optimizer.step()

            batch_size = keypoints_2d.shape[0]
            running_loss += loss.item() * batch_size
            num_seen += batch_size

        train_loss = running_loss / num_seen
        val_fk, val_mpjpe = evaluate(lifter, model, val_loader, fk_loss, device)
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
