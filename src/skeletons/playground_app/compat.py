from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable

import torch

PACKAGE_CANDIDATES = (
    "skeletons",
)


class CompatibilityImportError(ImportError):
    pass


@dataclass(frozen=True)
class SkeletonApi:
    package_name: str
    supported_skeletons: tuple[str, ...]
    create: Callable[..., Any]
    frame_dataset_cls: type
    perspective_camera_cls: type
    load_fitting_prior_checkpoint: Callable[..., Any]
    axis_names: tuple[str, ...]
    animation_preset_cls: type
    axis_rom_limit_cls: type
    synthetic_fitting_controls_cls: type
    apply_rom_payload: Callable[..., Any]
    available_animation_presets: Callable[..., Any]
    build_fit_export_payload: Callable[..., Any]
    create_synthetic_fitting_dataset: Callable[..., Any]
    default_rom_limits: Callable[..., Any]
    serialize_rom_limits: Callable[..., Any]
    fitting_worker_cls: type


@dataclass(frozen=True)
class IKApi:
    package_name: str
    ccd_ik_cls: type
    dls_ik_cls: type
    gradient_ik_cls: type
    axis_angle_to_matrix: Callable[..., Any]


def _import_root() -> tuple[str, Any]:
    errors: list[str] = []
    for package_name in PACKAGE_CANDIDATES:
        try:
            return package_name, import_module(package_name)
        except Exception as exc:  # pragma: no cover - runtime dependency
            errors.append(f"{package_name}: {exc}")
    raise CompatibilityImportError(
        "Could not import skeletons under any known package name. "
        + " | ".join(errors)
    )


def _import_submodule(package_name: str, suffix: str) -> Any:
    return import_module(f"{package_name}{suffix}")


def load_skeleton_api() -> SkeletonApi:
    package_name, root = _import_root()
    fitting_mod = _import_submodule(package_name, ".fitting")
    playground_mod = _import_submodule(package_name, ".playground")
    workers_mod = _import_submodule(package_name, ".playground.qt_workers")

    checkpoint_loader = getattr(fitting_mod, "load_fitting_prior_checkpoint", None)
    if checkpoint_loader is None:
        checkpoint_loader = lambda path, skeleton=None: torch.load(
            path, map_location="cpu"
        )

    return SkeletonApi(
        package_name=package_name,
        supported_skeletons=tuple(getattr(root, "SUPPORTED_SKELETONS")),
        create=getattr(root, "create"),
        frame_dataset_cls=getattr(fitting_mod, "FrameDataset"),
        perspective_camera_cls=getattr(fitting_mod, "PerspectiveCamera"),
        load_fitting_prior_checkpoint=checkpoint_loader,
        axis_names=tuple(getattr(playground_mod, "AXIS_NAMES")),
        animation_preset_cls=getattr(playground_mod, "AnimationPreset"),
        axis_rom_limit_cls=getattr(playground_mod, "AxisRomLimit"),
        synthetic_fitting_controls_cls=getattr(
            playground_mod, "SyntheticFittingControls"
        ),
        apply_rom_payload=getattr(playground_mod, "apply_rom_payload"),
        available_animation_presets=getattr(
            playground_mod, "available_animation_presets"
        ),
        build_fit_export_payload=getattr(playground_mod, "build_fit_export_payload"),
        create_synthetic_fitting_dataset=getattr(
            playground_mod, "create_synthetic_fitting_dataset"
        ),
        default_rom_limits=getattr(playground_mod, "default_rom_limits"),
        serialize_rom_limits=getattr(playground_mod, "serialize_rom_limits"),
        fitting_worker_cls=getattr(workers_mod, "FittingWorker"),
    )


def load_ik_api() -> IKApi:
    package_name, root = _import_root()

    return IKApi(
        package_name=package_name,
        ccd_ik_cls=getattr(root, "CyclicCoordinateDescentIK"),
        dls_ik_cls=getattr(root, "DampedLeastSquaresIK"),
        gradient_ik_cls=getattr(root, "GradientIK"),
        axis_angle_to_matrix=getattr(root, "axis_angle_to_matrix"),
    )
