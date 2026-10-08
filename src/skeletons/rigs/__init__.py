from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from types import MappingProxyType
from typing import Any

import torch

from ._schemas import KeypointSchema, available_schemas, get_schema
from ._spec import ContactSpec, LinkSpec, MarkerSpec, SkeletonSpec

_SPEC_FACTORY_NAMES = {
    "coco": (".coco", "coco_spec"),
    "mpii": (".mpii", "mpii_spec"),
    "human36m": (".human36m", "human36m_spec"),
    "halpe26": (".halpe26", "halpe26_spec"),
    "hand21": (".hand21", "hand21_spec"),
    "face68": (".face68", "face68_spec"),
    "halpe_fullbody": (".halpe_fullbody", "halpe_fullbody_spec"),
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
    "halpe_fullbody": (".halpe_fullbody", "HalpeFullBodyModel"),
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
    "halpe_fullbody": (".halpe_fullbody", "HalpeFullBodyModelLayer"),
    "coco_wholebody": (".coco_wholebody", "CocoWholeBodyModelLayer"),
    "spinetrack": (".spinetrack", "SpineTrackModelLayer"),
}

SUPPORTED_SKELETONS = tuple(_SPEC_FACTORY_NAMES.keys())
SKELETON_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "halpefullbody": "halpe_fullbody",
        "cocowholebody": "coco_wholebody",
    }
)
_SUPPORTED_SKELETON_SET = frozenset(SUPPORTED_SKELETONS)


def _normalize_name(name: str) -> str:
    return name.lower().replace("-", "_")


def _load_attr(target: tuple[str, str]) -> Any:
    module_name, attr_name = target
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)


def canonicalize_skeleton_name(name: str) -> str:
    """Resolve a public skeleton name to its canonical registry key.

    Parameters
    ----------
    name : str
        Skeleton name or alias accepted by the public factory API.

    Returns
    -------
    str
        Canonical skeleton registry key used by the package.

    Raises
    ------
    KeyError
        If ``name`` does not resolve to a supported skeleton.
    """

    normalized_name = _normalize_name(name)
    if normalized_name in _SUPPORTED_SKELETON_SET:
        return normalized_name
    if normalized_name not in SKELETON_ALIASES:
        raise KeyError(f"Unknown skeleton name: {name!r}")
    return SKELETON_ALIASES[normalized_name]


def list_supported_skeletons(*, include_aliases: bool = False) -> tuple[str, ...]:
    """List supported skeleton names.

    Parameters
    ----------
    include_aliases : bool, optional
        When ``True``, return every accepted public name, including aliases.
        When ``False``, return only canonical skeleton names.

    Returns
    -------
    tuple[str, ...]
        Ordered tuple of supported skeleton names.
    """

    if include_aliases:
        return SUPPORTED_SKELETONS + tuple(SKELETON_ALIASES)
    return SUPPORTED_SKELETONS


def get_spec(
    name: str, *, dtype: torch.dtype = torch.float32, **options: Any
) -> SkeletonSpec:
    """Build a skeleton specification by name.

    Parameters
    ----------
    name : str
        Canonical skeleton name or accepted alias.
    dtype : torch.dtype, optional
        Floating-point dtype used for the returned specification tensors.
    **options
        Optional articulation, link, marker, contact, or metadata overrides.
        Joint types, axes, and limits accept mappings keyed by joint name.

    Returns
    -------
    SkeletonSpec
        Skeleton specification for the requested rig.

    Raises
    ------
    KeyError
        If ``name`` does not resolve to a supported skeleton.
    """

    canonical_name = canonicalize_skeleton_name(name)
    factory = _load_attr(_SPEC_FACTORY_NAMES[canonical_name])
    spec = factory(dtype=dtype)
    return spec.configured(**options) if options else spec


def create(name: str, *args, dtype: torch.dtype = torch.float32, **kwargs) -> Any:
    """Construct a parameter-owning skeleton model by name.

    Parameters
    ----------
    name : str
        Canonical skeleton name or accepted alias.
    dtype : torch.dtype, optional
        Floating-point dtype passed to the model constructor.
    *args
        Positional arguments forwarded to the model constructor.
    **kwargs
        Keyword arguments forwarded to the model constructor.

    Returns
    -------
    Any
        Instantiated skeleton model for the requested rig.

    Raises
    ------
    KeyError
        If ``name`` does not resolve to a supported skeleton.
    """

    canonical_name = canonicalize_skeleton_name(name)
    model_cls = _load_attr(_MODEL_CLASS_NAMES[canonical_name])
    return model_cls(*args, dtype=dtype, **kwargs)


def build_layer(name: str, *args, dtype: torch.dtype = torch.float32, **kwargs) -> Any:
    """Construct a parameter-free skeleton layer by name.

    Parameters
    ----------
    name : str
        Canonical skeleton name or accepted alias.
    dtype : torch.dtype, optional
        Floating-point dtype passed to the layer constructor.
    *args
        Positional arguments forwarded to the layer constructor.
    **kwargs
        Keyword arguments forwarded to the layer constructor.

    Returns
    -------
    Any
        Instantiated skeleton layer for the requested rig.

    Raises
    ------
    KeyError
        If ``name`` does not resolve to a supported skeleton.
    """

    canonical_name = canonicalize_skeleton_name(name)
    layer_cls = _load_attr(_LAYER_CLASS_NAMES[canonical_name])
    return layer_cls(*args, dtype=dtype, **kwargs)


__all__ = [
    "SKELETON_ALIASES",
    "SUPPORTED_SKELETONS",
    "ContactSpec",
    "LinkSpec",
    "MarkerSpec",
    "SkeletonSpec",
    "KeypointSchema",
    "available_schemas",
    "canonicalize_skeleton_name",
    "create",
    "build_layer",
    "get_schema",
    "get_spec",
    "list_supported_skeletons",
]
