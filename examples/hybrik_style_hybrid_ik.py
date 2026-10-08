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

from skeletons import build_layer
from skeletons.artifacts import (
    log_status,
    make_run_name,
    resolve_work_dir,
    save_json,
)
from skeletons.hybrid import (
    JointRegressor,
    evaluate_hybrid_ik_regressor,
    train_joint_regressor_epoch,
)
from skeletons.synthetic import (
    generate_random_pose_batch,
    make_perspective_camera,
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

    train_batch = generate_random_pose_batch(
        model, batch_size=args.train_samples, pose_std=args.pose_std, device=device
    )
    val_batch = generate_random_pose_batch(
        model, batch_size=args.val_samples, pose_std=args.pose_std, device=device
    )
    train_3d = train_batch.joints
    val_3d = val_batch.joints
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
        train_state = train_joint_regressor_epoch(
            regressor,
            train_loader,
            optimizer=optimizer,
            device=device,
        )
        val_state = evaluate_hybrid_ik_regressor(
            regressor,
            model,
            val_loader,
            device=device,
        )
        val_joint = val_state.joint_mpjpe
        val_fk = val_state.fk_consistency_mpjpe
        train_loss = train_state.loss
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
