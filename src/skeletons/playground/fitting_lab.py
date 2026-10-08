from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypeAlias

import numpy as np
import torch

from ..model import SkeletalModel
from ..synthetic import (
    SyntheticFittingDataset,
    generate_synthetic_fitting_dataset,
)

FittingExportValue: TypeAlias = np.ndarray | np.generic
FittingExportPayload: TypeAlias = dict[str, FittingExportValue]


@dataclass(frozen=True)
class SyntheticFittingControls:
    """Configuration for playground synthetic fitting-data generation.

    Parameters
    ----------
    num_frames : int
        Number of independent frames to generate.
    pose_std : float
        Standard deviation for random axis-angle pose samples.
    noise_std : float
        Pixel-space standard deviation for synthetic 2D detector noise.
    confidence_dropout : float
        Probability that a generated per-joint confidence is set to zero.
    transl_std : float
        Standard deviation for random translations.
    depth : float
        Positive z offset applied to generated translations.
    seed : int or None
        Optional random seed for reproducible playground datasets.
    """

    num_frames: int = 48
    pose_std: float = 0.18
    noise_std: float = 4.0
    confidence_dropout: float = 0.05
    transl_std: float = 25.0
    depth: float = 2500.0
    seed: int | None = 12345


def _json_scalar(payload: Mapping[str, Any] | None) -> np.ndarray:
    return np.asarray(json.dumps(payload or {}, sort_keys=True))


def _to_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _make_generator(
    seed: int | None,
    *,
    device: torch.device,
) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def create_synthetic_fitting_dataset(
    model: SkeletalModel,
    controls: SyntheticFittingControls,
    *,
    device: torch.device | str | None = None,
) -> SyntheticFittingDataset:
    """Create a synthetic fitting dataset from playground controls.

    Parameters
    ----------
    model : SkeletalModel
        Skeleton model used to generate synthetic joints.
    controls : SyntheticFittingControls
        Generation controls collected from the playground UI.
    device : torch.device or str, optional
        Device used for synthetic generation. Defaults to the model device.

    Returns
    -------
    SyntheticFittingDataset
        Generated fitting dataset and its source tensors.

    Raises
    ------
    ValueError
        Propagated from ``generate_synthetic_fitting_dataset`` for invalid
        controls.
    """

    resolved_device = (
        torch.device(device)
        if device is not None
        else torch.device(model.rest_offsets.device)
    )
    generator = _make_generator(controls.seed, device=resolved_device)
    return generate_synthetic_fitting_dataset(
        model,
        num_frames=controls.num_frames,
        pose_std=controls.pose_std,
        noise_std=controls.noise_std,
        confidence_dropout=controls.confidence_dropout,
        transl_std=controls.transl_std,
        depth=controls.depth,
        device=resolved_device,
        generator=generator,
    )


def build_fit_export_payload(
    fit_payload: Mapping[str, object],
    *,
    frame_indices: Sequence[int] | np.ndarray | torch.Tensor,
    range_sequence: Mapping[str, torch.Tensor] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FittingExportPayload:
    """Convert a playground sequence-fit result into a stable ``.npz`` payload.

    Parameters
    ----------
    fit_payload : Mapping[str, object]
        Result payload produced by the playground fitting worker.
    frame_indices : sequence of int or tensor
        Absolute frame indices covered by the fitted range.
    range_sequence : mapping, optional
        Original target sequence. Known target, confidence, and camera keys are
        copied into the export when present.
    metadata : mapping, optional
        JSON-serializable export metadata.

    Returns
    -------
    dict[str, numpy.ndarray]
        Arrays and JSON scalar fields suitable for ``numpy.savez_compressed``.

    Raises
    ------
    ValueError
        If a required fitted result key is missing.
    """

    required_keys = ("full_pose", "transl", "scales", "joints")
    missing = [key for key in required_keys if key not in fit_payload]
    if missing:
        raise ValueError(f"Fit payload is missing required keys: {missing}.")

    export: FittingExportPayload = {
        "full_pose": _to_numpy(fit_payload["full_pose"]).astype("float32"),
        "transl": _to_numpy(fit_payload["transl"]).astype("float32"),
        "scales": _to_numpy(fit_payload["scales"]).astype("float32"),
        "joints_3d": _to_numpy(fit_payload["joints"]).astype("float32"),
        "frame_indices": _to_numpy(frame_indices).astype("int64"),
        "losses_json": _json_scalar(fit_payload.get("losses")),
        "metadata_json": _json_scalar(metadata),
        "mode": np.asarray(str(fit_payload.get("mode", "unknown"))),
        "iterations": np.asarray(int(fit_payload.get("iterations", 0))),
        "frame_count": np.asarray(int(fit_payload.get("frame_count", 0))),
    }

    if range_sequence is None:
        return export

    optional_keys = (
        "joints_3d",
        "joints_2d",
        "confidences",
        "fx",
        "fy",
        "cx",
        "cy",
        "camera_translation",
        "camera_rotation",
    )
    for key in optional_keys:
        if key in range_sequence:
            export[f"target_{key}"] = _to_numpy(range_sequence[key]).astype("float32")
    return export
