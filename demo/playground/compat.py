from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable


class CompatibilityImportError(ImportError):
    pass


@dataclass(frozen=True)
class SkeletonApi:
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
    ccd_ik_cls: type
    dls_ik_cls: type
    gradient_ik_cls: type
    axis_angle_to_matrix: Callable[..., Any]


def _import_root() -> Any:
    try:
        return import_module("skeletons")
    except ImportError as exc:
        raise CompatibilityImportError("Could not import skeletons.") from exc


def load_skeleton_api() -> SkeletonApi:
    root = _import_root()
    from skeletons import fitting as fitting_mod

    from . import animations, rom
    from .qt_workers import FittingWorker
    from .workbenches import fitting_lab

    return SkeletonApi(
        supported_skeletons=tuple(root.SUPPORTED_SKELETONS),
        create=root.create,
        frame_dataset_cls=fitting_mod.FrameDataset,
        perspective_camera_cls=fitting_mod.PerspectiveCamera,
        load_fitting_prior_checkpoint=fitting_mod.load_fitting_prior_checkpoint,
        axis_names=tuple(rom.AXIS_NAMES),
        animation_preset_cls=animations.AnimationPreset,
        axis_rom_limit_cls=rom.AxisRomLimit,
        synthetic_fitting_controls_cls=fitting_lab.SyntheticFittingControls,
        apply_rom_payload=rom.apply_rom_payload,
        available_animation_presets=animations.available_animation_presets,
        build_fit_export_payload=fitting_lab.build_fit_export_payload,
        create_synthetic_fitting_dataset=fitting_lab.create_synthetic_fitting_dataset,
        default_rom_limits=rom.default_rom_limits,
        serialize_rom_limits=rom.serialize_rom_limits,
        fitting_worker_cls=FittingWorker,
    )


def load_ik_api() -> IKApi:
    root = _import_root()

    return IKApi(
        ccd_ik_cls=root.CyclicCoordinateDescentIK,
        dls_ik_cls=root.DampedLeastSquaresIK,
        gradient_ik_cls=root.GradientIK,
        axis_angle_to_matrix=root.axis_angle_to_matrix,
    )
