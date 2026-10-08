from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from skeletons import build_layer, generate_random_pose_batch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _example_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_example(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=_example_env(),
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
    )


@pytest.mark.parametrize(
    ("script", "args"),
    [
        (
            "retarget_between_skeletons.py",
            [
                "--source",
                "coco",
                "--target",
                "halpe26",
                "--batch-size",
                "1",
                "--fit-iters",
                "1",
            ],
        ),
        (
            "pseudo_label_bootstrap.py",
            [
                "--skeleton",
                "spinetrack",
                "--num-samples",
                "3",
                "--prior-epochs",
                "0",
                "--fit-iters",
                "1",
            ],
        ),
        (
            "spine_sequence_fit.py",
            ["--seq-len", "2", "--fit-iters", "1"],
        ),
        (
            "sequence_denoising_and_smoothing.py",
            [
                "--skeleton",
                "spinetrack",
                "--train-sequences",
                "2",
                "--eval-sequences",
                "1",
                "--seq-len",
                "2",
                "--prior-epochs",
                "0",
                "--fit-iters",
                "1",
            ],
        ),
        (
            "train_fk_lifter.py",
            [
                "--skeleton",
                "spinetrack",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--train-samples",
                "2",
                "--val-samples",
                "2",
                "--hidden-dim",
                "8",
            ],
        ),
        (
            "temporal_lifting_videopose3d.py",
            [
                "--skeleton",
                "spinetrack",
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--train-sequences",
                "2",
                "--val-sequences",
                "1",
                "--seq-len",
                "3",
                "--hidden-dim",
                "8",
            ],
        ),
        (
            "hybrik_style_hybrid_ik.py",
            [
                "--skeleton",
                "spinetrack",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--train-samples",
                "2",
                "--val-samples",
                "2",
                "--hidden-dim",
                "8",
            ],
        ),
    ],
)
def test_runnable_examples_smoke(
    script: str,
    args: list[str],
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / script.removesuffix(".py")

    _run_example(
        [
            str(EXAMPLES / script),
            *args,
            "--work-dir",
            str(work_dir),
        ]
    )

    assert work_dir.exists()


def test_prior_training_and_fit_demo_smoke(tmp_path: Path) -> None:
    model = build_layer("spinetrack")
    batch = generate_random_pose_batch(
        model,
        batch_size=4,
        pose_std=0.02,
        device="cpu",
        generator=torch.Generator().manual_seed(41),
    )
    dataset_path = tmp_path / "spinetrack_frames.npz"
    np.savez_compressed(
        dataset_path,
        joints_3d=batch.joints.numpy(),
        joints_2d=batch.joints[..., :2].numpy(),
        confidences=np.ones(batch.joints.shape[:-1], dtype=np.float32),
    )
    priors_dir = tmp_path / "priors"

    _run_example(
        [
            str(EXAMPLES / "train_prior.py"),
            str(dataset_path),
            "--skeleton",
            "spinetrack",
            "--epochs",
            "0",
            "--batch-size",
            "2",
            "--latent-dim",
            "4",
            "--hidden-dim",
            "8",
            "--num-hidden-layers",
            "1",
            "--work-dir",
            str(priors_dir),
        ]
    )
    fit_result = _run_example(
        [
            str(EXAMPLES / "fit_demo.py"),
            str(dataset_path),
            "--skeleton",
            "spinetrack",
            "--mode",
            "3d",
            "--sample-index",
            "0",
            "--fit-iters",
            "1",
            "--priors",
            str(priors_dir),
        ]
    )

    assert (priors_dir / "last_checkpoint").exists()
    assert "spinetrack" in fit_result.stdout
