from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from _synthetic import make_random_walk_sequences, make_weak_perspective_camera
from _work_dir import log_status, make_run_name, resolve_work_dir, save_json

from differential_skeletons import build_layer
from differential_skeletons.fitting import MotionSmoothnessPrior, SkeletalFitter


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit a SpineTrack sequence from weak-perspective 2D observations."
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--noise-std", type=float, default=2.5)
    parser.add_argument("--fit-iters", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dir = resolve_work_dir(
        Path(__file__),
        run_name=make_run_name("spinetrack", "sequence_fit", f"seed{args.seed}"),
        work_dir=args.work_dir,
    )

    model = build_layer("spinetrack").to(device)
    camera = make_weak_perspective_camera(device)
    _, _, _, clean_joints = make_random_walk_sequences(
        model,
        batch_size=1,
        seq_len=args.seq_len,
        pose_step_std=0.01,
        transl_step_std=0.3,
        depth=1200.0,
        device=device,
    )
    target_2d = camera.project(clean_joints) + args.noise_std * torch.randn_like(
        clean_joints[..., :2]
    )
    confidences = torch.ones(1, args.seq_len, model.NUM_JOINTS, device=device)

    fitter = SkeletalFitter(
        model=model,
        smoothness_prior=MotionSmoothnessPrior(
            velocity_weight=1.0, acceleration_weight=4.0
        ),
        device=device,
    )
    result = fitter.fit_sequence_2d(
        target_2d,
        camera,
        confidences=confidences,
        num_iters=args.fit_iters,
        optimize_scales=False,
        use_pose_prior_latent=False,
    )
    fitted = result.model_output.joints.reshape_as(clean_joints)
    mpjpe = torch.linalg.vector_norm(fitted - clean_joints, dim=-1).mean().item()
    metrics = {"mpjpe": mpjpe, **result.losses}
    save_json(work_dir / "metrics.json", metrics)
    log_status(Path(__file__).stem, f"mpjpe={mpjpe:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
