from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from skeletons import (
    build_layer,
    fit_2d_joint_sequences,
    mean_per_joint_position_error,
)
from skeletons.artifacts import (
    log_status,
    make_run_name,
    resolve_work_dir,
    save_json,
)
from skeletons.fitting import MotionSmoothnessPrior
from skeletons.synthetic import (
    generate_random_walk_sequences,
    make_weak_perspective_camera,
)


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
    sequence = generate_random_walk_sequences(
        model,
        batch_size=1,
        seq_len=args.seq_len,
        pose_step_std=0.01,
        transl_step_std=0.3,
        depth=1200.0,
        device=device,
    )
    clean_joints = sequence.joints
    target_2d = camera.project(clean_joints) + args.noise_std * torch.randn_like(
        clean_joints[..., :2]
    )
    confidences = torch.ones(1, args.seq_len, model.NUM_JOINTS, device=device)

    result = fit_2d_joint_sequences(
        target_2d,
        camera,
        model=model,
        confidences=confidences,
        smoothness_prior=MotionSmoothnessPrior(
            velocity_weight=1.0, acceleration_weight=4.0
        ),
        num_iters=args.fit_iters,
        optimize_scales=False,
        use_pose_prior_latent=False,
        device=device,
    )
    fitted = result.joints_3d.to(device)
    mpjpe = mean_per_joint_position_error(fitted, clean_joints)
    mpjpe_value = float(mpjpe.item())
    metrics = {
        "mpjpe": mpjpe_value,
        "mean_reprojection": result.mean_reprojection_error,
        **result.losses,
    }
    save_json(work_dir / "metrics.json", metrics)
    log_status(Path(__file__).stem, f"mpjpe={mpjpe_value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
