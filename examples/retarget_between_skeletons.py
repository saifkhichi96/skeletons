from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from skeletons import build_layer
from skeletons.artifacts import (
    log_status,
    make_run_name,
    resolve_work_dir,
    save_json,
)
from skeletons.retargeting import build_joint_mapping, retarget_skeleton
from skeletons.synthetic import generate_random_pose_batch


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retarget motion between skeleton conventions by fitting the target rig to shared joints."
    )
    parser.add_argument("--source", default="coco")
    parser.add_argument("--target", default="halpe26")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pose-std", type=float, default=0.20)
    parser.add_argument("--fit-iters", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dir = resolve_work_dir(
        Path(__file__),
        run_name=make_run_name(args.source, "to", args.target, f"seed{args.seed}"),
        work_dir=args.work_dir,
    )

    source_model = build_layer(args.source).to(device)
    target_model = build_layer(args.target).to(device)
    mapping = build_joint_mapping(source_model, target_model, min_joints=4)

    pose_batch = generate_random_pose_batch(
        source_model, batch_size=args.batch_size, pose_std=args.pose_std, device=device
    )
    source_joints = pose_batch.joints
    result = retarget_skeleton(
        source_joints,
        source_model=source_model,
        target_model=target_model,
        mapping=mapping,
        num_iters=args.fit_iters,
        optimize_scales=True,
        use_pose_prior_latent=False,
    )

    metrics = {
        "shared_joint_mpjpe": result.shared_joint_mpjpe,
        "shared_joint_count": len(mapping),
        **result.fitting_result.losses,
    }
    save_json(work_dir / "metrics.json", metrics)
    log_status(
        Path(__file__).stem,
        f"shared_joint_count={len(mapping)} shared_joint_mpjpe={result.shared_joint_mpjpe:.4f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
