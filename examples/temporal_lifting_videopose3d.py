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
from _synthetic import make_perspective_camera, make_random_walk_sequences
from _work_dir import log_status, make_run_name, resolve_work_dir, save_json
from torch.utils.data import DataLoader, TensorDataset

from differential_skeletons import ForwardKinematicsLoss, build_layer


class TemporalPoseLifter(nn.Module):
    """A small temporal lifting baseline inspired by VideoPose3D/PoseFormer-style sequence models."""

    def __init__(
        self, num_joints: int, hidden_dim: int = 256, kernel_size: int = 3
    ) -> None:
        super().__init__()
        input_dim = num_joints * 2
        output_dim = num_joints * 3
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=2 * padding,
                dilation=2,
            ),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1),
        )
        self.num_joints = num_joints

    def forward(self, keypoints_2d: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, _ = keypoints_2d.shape
        x = keypoints_2d.reshape(batch_size, seq_len, -1).transpose(1, 2)
        y = self.encoder(x).transpose(1, 2)
        return y.reshape(batch_size, seq_len, self.num_joints, 3)


def evaluate(
    lifter: TemporalPoseLifter,
    model,
    loader: DataLoader,
    fk_loss: ForwardKinematicsLoss,
    device: torch.device,
) -> tuple[float, float]:
    lifter.eval()
    total_fk = 0.0
    total_mpjpe = 0.0
    total_frames = 0
    with torch.no_grad():
        for keypoints_2d, joints_3d in loader:
            keypoints_2d = keypoints_2d.to(device)
            joints_3d = joints_3d.to(device)
            pred_pose = lifter(keypoints_2d)
            flat_pose = pred_pose.reshape(-1, model.NUM_JOINTS, 3)
            flat_joints = joints_3d.reshape(-1, model.NUM_JOINTS, 3)
            loss = fk_loss(flat_joints, full_pose=flat_pose)
            pred_joints = model(full_pose=flat_pose).joints.reshape_as(joints_3d)
            mpjpe = torch.linalg.vector_norm(pred_joints - joints_3d, dim=-1).mean()
            frames = keypoints_2d.shape[0] * keypoints_2d.shape[1]
            total_fk += float(loss) * frames
            total_mpjpe += float(mpjpe) * frames
            total_frames += frames
    return total_fk / total_frames, total_mpjpe / total_frames


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a temporal 2D-to-3D lifter with FK supervision."
    )
    parser.add_argument("--skeleton", default="human36m")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-sequences", type=int, default=512)
    parser.add_argument("--val-sequences", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=27)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--pose-step-std", type=float, default=0.03)
    parser.add_argument("--noise-std", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dir = resolve_work_dir(
        Path(__file__),
        run_name=make_run_name(args.skeleton, "temporal_lifter", f"seed{args.seed}"),
        work_dir=args.work_dir,
    )
    save_json(
        work_dir / "config.json",
        vars(args) | {"device": str(device), "work_dir": str(work_dir)},
    )
    log_status(Path(__file__).stem, f"device={device}")

    model = build_layer(args.skeleton).to(device)
    fk_loss = ForwardKinematicsLoss(model)
    camera = make_perspective_camera(device)

    _, _, _, train_3d = make_random_walk_sequences(
        model,
        batch_size=args.train_sequences,
        seq_len=args.seq_len,
        pose_step_std=args.pose_step_std,
        transl_step_std=4.0,
        depth=2500.0,
        device=device,
    )
    _, _, _, val_3d = make_random_walk_sequences(
        model,
        batch_size=args.val_sequences,
        seq_len=args.seq_len,
        pose_step_std=args.pose_step_std,
        transl_step_std=4.0,
        depth=2500.0,
        device=device,
    )
    train_2d = camera.project(train_3d) + args.noise_std * torch.randn_like(
        train_3d[..., :2]
    )
    val_2d = camera.project(val_3d) + args.noise_std * torch.randn_like(val_3d[..., :2])

    train_loader = DataLoader(
        TensorDataset(train_2d.cpu(), train_3d.cpu()),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_2d.cpu(), val_3d.cpu()),
        batch_size=args.batch_size,
        shuffle=False,
    )

    lifter = TemporalPoseLifter(model.NUM_JOINTS, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(lifter.parameters(), lr=args.lr)

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        lifter.train()
        running = 0.0
        seen = 0
        for keypoints_2d, joints_3d in train_loader:
            keypoints_2d = keypoints_2d.to(device)
            joints_3d = joints_3d.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_pose = lifter(keypoints_2d)
            loss = fk_loss(
                joints_3d.reshape(-1, model.NUM_JOINTS, 3),
                full_pose=pred_pose.reshape(-1, model.NUM_JOINTS, 3),
            )
            loss.backward()
            optimizer.step()
            running += float(loss) * keypoints_2d.shape[0]
            seen += keypoints_2d.shape[0]

        train_loss = running / max(1, seen)
        val_fk, val_mpjpe = evaluate(lifter, model, val_loader, fk_loss, device)
        log_status(
            Path(__file__).stem,
            f"epoch={epoch} train_fk={train_loss:.4f} val_fk={val_fk:.4f} val_mpjpe={val_mpjpe:.4f}",
        )
        history.append(
            {
                "epoch": epoch,
                "train_fk": train_loss,
                "val_fk": val_fk,
                "val_mpjpe": val_mpjpe,
            }
        )

    save_json(work_dir / "metrics.json", {"history": history})
    torch.save(
        {"model_state": lifter.state_dict(), "history": history, "config": vars(args)},
        work_dir / "temporal_lifter.pth",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
