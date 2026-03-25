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
from _synthetic import make_perspective_camera, make_random_pose_batch
from _work_dir import log_status, make_run_name, resolve_work_dir, save_json
from torch.utils.data import DataLoader, TensorDataset

from differential_skeletons import build_layer, estimate_rotations_from_joints


class JointRegressor(nn.Module):
    """Predict 3D joints from 2D keypoints, then recover rotations analytically via IK.

    This mirrors the central idea behind HybrIK-like pipelines: let the network predict
    geometry that is easy to supervise directly, then recover articulated rotations from it.
    """

    def __init__(self, num_joints: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.net = nn.Sequential(
            nn.Linear(num_joints * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_joints * 3),
        )

    def forward(self, keypoints_2d: torch.Tensor) -> torch.Tensor:
        batch_size = keypoints_2d.shape[0]
        return self.net(keypoints_2d.reshape(batch_size, -1)).reshape(
            batch_size, self.num_joints, 3
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a HybrIK-style 2D-to-3D-to-IK pipeline on synthetic data."
    )
    parser.add_argument("--skeleton", default="human36m")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--pose-std", type=float, default=0.20)
    parser.add_argument("--noise-std", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dir = resolve_work_dir(
        Path(__file__),
        run_name=make_run_name(args.skeleton, "hybrik_style", f"seed{args.seed}"),
        work_dir=args.work_dir,
    )
    save_json(
        work_dir / "config.json",
        vars(args) | {"device": str(device), "work_dir": str(work_dir)},
    )

    model = build_layer(args.skeleton).to(device)
    camera = make_perspective_camera(device)

    _, _, _, train_3d = make_random_pose_batch(
        model, batch_size=args.train_samples, pose_std=args.pose_std, device=device
    )
    _, _, _, val_3d = make_random_pose_batch(
        model, batch_size=args.val_samples, pose_std=args.pose_std, device=device
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

    regressor = JointRegressor(model.NUM_JOINTS, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(regressor.parameters(), lr=args.lr)

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        regressor.train()
        total_loss = 0.0
        seen = 0
        for keypoints_2d, joints_3d in train_loader:
            keypoints_2d = keypoints_2d.to(device)
            joints_3d = joints_3d.to(device)
            pred_joints = regressor(keypoints_2d)
            loss = (pred_joints - joints_3d).pow(2).sum(dim=-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * keypoints_2d.shape[0]
            seen += keypoints_2d.shape[0]

        regressor.eval()
        joint_errors = []
        fk_errors = []
        with torch.no_grad():
            for keypoints_2d, joints_3d in val_loader:
                keypoints_2d = keypoints_2d.to(device)
                joints_3d = joints_3d.to(device)
                pred_joints = regressor(keypoints_2d)
                ik = estimate_rotations_from_joints(pred_joints, model)
                recon_joints = model(
                    full_pose=ik.local_rotations, pose_repr="rotmat", scales=ik.scales
                ).joints
                joint_errors.append(
                    torch.linalg.vector_norm(pred_joints - joints_3d, dim=-1).mean()
                )
                fk_errors.append(
                    torch.linalg.vector_norm(recon_joints - pred_joints, dim=-1).mean()
                )
        val_joint = torch.stack(joint_errors).mean().item()
        val_fk = torch.stack(fk_errors).mean().item()
        train_loss = total_loss / max(1, seen)
        log_status(
            Path(__file__).stem,
            f"epoch={epoch} train_joint={train_loss:.4f} val_joint={val_joint:.4f} ik_fk_consistency={val_fk:.4f}",
        )
        history.append(
            {
                "epoch": epoch,
                "train_joint": train_loss,
                "val_joint": val_joint,
                "ik_fk_consistency": val_fk,
            }
        )

    save_json(work_dir / "metrics.json", {"history": history})
    torch.save(
        {
            "model_state": regressor.state_dict(),
            "history": history,
            "config": vars(args),
        },
        work_dir / "hybrid_ik_regressor.pth",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
