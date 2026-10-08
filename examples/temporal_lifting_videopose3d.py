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

from skeletons import ForwardKinematicsLoss, build_layer
from skeletons.artifacts import (
    log_status,
    make_run_name,
    resolve_work_dir,
    save_json,
)
from skeletons.lifting import (
    TemporalPoseLifter,
    evaluate_lifter,
    train_lifter_epoch,
)
from skeletons.synthetic import (
    generate_random_walk_sequences,
    make_perspective_camera,
)


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

    train_batch = generate_random_walk_sequences(
        model,
        batch_size=args.train_sequences,
        seq_len=args.seq_len,
        pose_step_std=args.pose_step_std,
        transl_step_std=4.0,
        depth=2500.0,
        device=device,
    )
    val_batch = generate_random_walk_sequences(
        model,
        batch_size=args.val_sequences,
        seq_len=args.seq_len,
        pose_step_std=args.pose_step_std,
        transl_step_std=4.0,
        depth=2500.0,
        device=device,
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

    lifter = TemporalPoseLifter(model.NUM_JOINTS, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(lifter.parameters(), lr=args.lr)

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_state = train_lifter_epoch(
            lifter,
            model,
            train_loader,
            optimizer=optimizer,
            fk_loss=fk_loss,
            device=device,
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
