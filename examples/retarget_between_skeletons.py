from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from _synthetic import make_random_pose_batch
from _work_dir import log_status, make_run_name, resolve_work_dir, save_json

from differential_skeletons import build_layer
from differential_skeletons.fitting import SkeletalFitter


def shared_joint_mapping(source_model, target_model) -> list[tuple[int, int, str]]:
    target_lookup = {
        name: idx for idx, name in enumerate(target_model.spec.joint_names)
    }
    mapping: list[tuple[int, int, str]] = []
    for src_idx, name in enumerate(source_model.spec.joint_names):
        dst_idx = target_lookup.get(name)
        if dst_idx is not None:
            mapping.append((src_idx, dst_idx, name))
    return mapping


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
    mapping = shared_joint_mapping(source_model, target_model)
    if len(mapping) < 4:
        raise ValueError(
            f"Not enough shared joints between {args.source} and {args.target}: {mapping}"
        )

    _, _, _, source_joints = make_random_pose_batch(
        source_model, batch_size=args.batch_size, pose_std=args.pose_std, device=device
    )
    target_observations = torch.zeros(
        args.batch_size, target_model.NUM_JOINTS, 3, device=device
    )
    weights = torch.zeros(args.batch_size, target_model.NUM_JOINTS, device=device)
    for src_idx, dst_idx, _name in mapping:
        target_observations[:, dst_idx] = source_joints[:, src_idx]
        weights[:, dst_idx] = 1.0

    fitter = SkeletalFitter(model=target_model, device=device)
    result = fitter.fit_3d(
        target_observations,
        weights=weights,
        num_iters=args.fit_iters,
        optimize_scales=True,
        use_pose_prior_latent=False,
    )

    shared_errors = []
    for src_idx, dst_idx, _name in mapping:
        shared_errors.append(
            torch.linalg.vector_norm(
                result.model_output.joints[:, dst_idx] - source_joints[:, src_idx],
                dim=-1,
            )
        )
    shared_mpjpe = torch.stack(shared_errors, dim=-1).mean().item()
    metrics = {
        "shared_joint_mpjpe": shared_mpjpe,
        "shared_joint_count": len(mapping),
        **result.losses,
    }
    save_json(work_dir / "metrics.json", metrics)
    log_status(
        Path(__file__).stem,
        f"shared_joint_count={len(mapping)} shared_joint_mpjpe={shared_mpjpe:.4f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
