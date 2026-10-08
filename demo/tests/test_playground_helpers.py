from __future__ import annotations

import json

import numpy as np
import torch
from playground.workbenches.fitting_lab import (
    SyntheticFittingControls,
    build_fit_export_payload,
    create_synthetic_fitting_dataset,
)

from skeletons import build_layer


def test_create_synthetic_fitting_dataset_uses_controls() -> None:
    model = build_layer("spinetrack")
    generated = create_synthetic_fitting_dataset(
        model,
        SyntheticFittingControls(
            num_frames=4,
            pose_std=0.05,
            noise_std=0.0,
            confidence_dropout=0.25,
            seed=7,
        ),
    )
    sample = generated.frame_dataset[0]

    assert len(generated.frame_dataset) == 4
    assert generated.pose_batch.joints.shape == (4, model.NUM_JOINTS, 3)
    assert generated.noisy_joints_2d.shape == (4, model.NUM_JOINTS, 2)
    assert sample["fx"].shape == ()
    assert "joints_2d" in sample
    assert "confidences" in sample


def test_build_fit_export_payload_includes_targets_and_metadata() -> None:
    fit_payload = {
        "full_pose": torch.zeros(2, 3, 3),
        "transl": torch.zeros(2, 3),
        "scales": torch.ones(2, 3, 3),
        "joints": np.ones((2, 3, 3), dtype=np.float64),
        "losses": {"joint": 0.25},
        "iterations": 12,
        "mode": "3d",
        "frame_count": 2,
    }
    range_sequence = {
        "joints_3d": torch.full((2, 3, 3), 2.0),
        "joints_2d": torch.full((2, 3, 2), 3.0),
        "confidences": torch.ones(2, 3),
        "fx": torch.full((2,), 1000.0),
    }
    export = build_fit_export_payload(
        fit_payload,
        frame_indices=[5, 6],
        range_sequence=range_sequence,
        metadata={"skeleton": "toy"},
    )

    assert export["joints_3d"].dtype == np.float32
    assert export["target_joints_3d"].shape == (2, 3, 3)
    assert export["target_joints_2d"].shape == (2, 3, 2)
    assert export["target_confidences"].shape == (2, 3)
    assert export["target_fx"].tolist() == [1000.0, 1000.0]
    assert export["frame_indices"].tolist() == [5, 6]
    assert json.loads(export["losses_json"].item()) == {"joint": 0.25}
    assert json.loads(export["metadata_json"].item()) == {"skeleton": "toy"}
    assert export["mode"].item() == "3d"
