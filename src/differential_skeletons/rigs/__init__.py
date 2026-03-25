from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

from ._spec import SkeletonSpec

_SPEC_FACTORY_NAMES = {
    "coco": (".coco", "coco_spec"),
    "mpii": (".mpii", "mpii_spec"),
    "human36m": (".human36m", "human36m_spec"),
    "halpe26": (".halpe26", "halpe26_spec"),
    "hand21": (".hand21", "hand21_spec"),
    "face68": (".face68", "face68_spec"),
    "halpefullbody": (".halpe_fullbody", "halpe_fullbody_spec"),
    "halpe_fullbody": (".halpe_fullbody", "halpe_fullbody_spec"),
    "cocowholebody": (".coco_wholebody", "coco_wholebody_spec"),
    "coco_wholebody": (".coco_wholebody", "coco_wholebody_spec"),
    "spinetrack": (".spinetrack", "spinetrack_spec"),
}

_MODEL_CLASS_NAMES = {
    "coco": (".coco", "CocoModel"),
    "mpii": (".mpii", "MPIIModel"),
    "human36m": (".human36m", "Human36MModel"),
    "halpe26": (".halpe26", "Halpe26Model"),
    "hand21": (".hand21", "Hand21Model"),
    "face68": (".face68", "Face68Model"),
    "halpefullbody": (".halpe_fullbody", "HalpeFullBodyModel"),
    "halpe_fullbody": (".halpe_fullbody", "HalpeFullBodyModel"),
    "cocowholebody": (".coco_wholebody", "CocoWholeBodyModel"),
    "coco_wholebody": (".coco_wholebody", "CocoWholeBodyModel"),
    "spinetrack": (".spinetrack", "SpineTrackModel"),
}

_LAYER_CLASS_NAMES = {
    "coco": (".coco", "CocoModelLayer"),
    "mpii": (".mpii", "MPIIModelLayer"),
    "human36m": (".human36m", "Human36MModelLayer"),
    "halpe26": (".halpe26", "Halpe26ModelLayer"),
    "hand21": (".hand21", "Hand21ModelLayer"),
    "face68": (".face68", "Face68ModelLayer"),
    "halpefullbody": (".halpe_fullbody", "HalpeFullBodyModelLayer"),
    "halpe_fullbody": (".halpe_fullbody", "HalpeFullBodyModelLayer"),
    "cocowholebody": (".coco_wholebody", "CocoWholeBodyModelLayer"),
    "coco_wholebody": (".coco_wholebody", "CocoWholeBodyModelLayer"),
    "spinetrack": (".spinetrack", "SpineTrackModelLayer"),
}

SUPPORTED_SKELETONS = tuple(_SPEC_FACTORY_NAMES.keys())


def _normalize_name(name: str) -> str:
    return name.lower().replace("-", "_")


def _load_attr(target: tuple[str, str]) -> Any:
    module_name, attr_name = target
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)


def get_spec(name: str, *, dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    key = _normalize_name(name)
    if key not in _SPEC_FACTORY_NAMES:
        raise KeyError(f"Unknown skeleton spec: {name!r}")
    factory = _load_attr(_SPEC_FACTORY_NAMES[key])
    return factory(dtype=dtype)


def create(name: str, *args, dtype: torch.dtype = torch.float32, **kwargs) -> Any:
    key = _normalize_name(name)
    if key not in _MODEL_CLASS_NAMES:
        raise KeyError(f"Unknown skeleton model: {name!r}")
    model_cls = _load_attr(_MODEL_CLASS_NAMES[key])
    return model_cls(*args, dtype=dtype, **kwargs)


def build_layer(name: str, *args, dtype: torch.dtype = torch.float32, **kwargs) -> Any:
    key = _normalize_name(name)
    if key not in _LAYER_CLASS_NAMES:
        raise KeyError(f"Unknown skeleton layer: {name!r}")
    layer_cls = _load_attr(_LAYER_CLASS_NAMES[key])
    return layer_cls(*args, dtype=dtype, **kwargs)


__all__ = [
    "SUPPORTED_SKELETONS",
    "SkeletonSpec",
    "create",
    "build_layer",
    "get_spec",
]
