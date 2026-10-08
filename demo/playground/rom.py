from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

AXIS_NAMES = ("X", "Y", "Z")
AXIS_FILE_KEYS = ("x", "y", "z")


@dataclass
class AxisRomLimit:
    """Per-axis range-of-motion limit used by the playground editor.

    Parameters
    ----------
    enabled : bool, optional
        Whether this axis limit should be enforced.
    minimum_deg : float, optional
        Minimum allowed axis-angle value in degrees.
    maximum_deg : float, optional
        Maximum allowed axis-angle value in degrees.
    """

    enabled: bool = False
    minimum_deg: float = -180.0
    maximum_deg: float = 180.0


def normalize_skeleton_name(name: str) -> str:
    """Normalize a skeleton identifier for ROM payload comparisons.

    Parameters
    ----------
    name : str
        Skeleton identifier to normalize.

    Returns
    -------
    str
        Lowercase identifier with hyphen separators converted to underscores.
    """

    return name.lower().replace("-", "_")


def default_rom_limits(joint_names: Iterable[str]) -> dict[str, list[AxisRomLimit]]:
    """Create disabled ROM limits for every joint.

    Parameters
    ----------
    joint_names : iterable of str
        Joint names that should receive three axis limits.

    Returns
    -------
    dict[str, list[AxisRomLimit]]
        Mapping from joint name to X/Y/Z axis ROM limits.
    """

    return {
        joint_name: [AxisRomLimit() for _ in AXIS_NAMES] for joint_name in joint_names
    }


def serialize_rom_limits(
    rom_limits: Mapping[str, Iterable[AxisRomLimit]],
    *,
    skeleton: str,
) -> dict[str, object]:
    """Serialize enabled ROM limits to the playground JSON schema.

    Parameters
    ----------
    rom_limits : mapping
        Joint-to-axis limit mapping.
    skeleton : str
        Skeleton name recorded in the payload.

    Returns
    -------
    dict[str, object]
        JSON-compatible ROM payload.
    """

    payload: dict[str, object] = {}
    for joint_name, limits in rom_limits.items():
        joint_payload: dict[str, object] = {}
        for axis_key, limit in zip(AXIS_FILE_KEYS, limits):
            if not limit.enabled:
                continue
            joint_payload[axis_key] = {
                "enabled": True,
                "min_deg": limit.minimum_deg,
                "max_deg": limit.maximum_deg,
            }
        if joint_payload:
            payload[joint_name] = joint_payload
    return {
        "format_version": 1,
        "skeleton": skeleton,
        "joint_limits": payload,
    }


def apply_rom_payload(
    payload: Mapping[str, Any],
    *,
    skeleton: str,
    joint_names: Iterable[str],
) -> dict[str, list[AxisRomLimit]]:
    """Parse a ROM payload for a concrete skeleton.

    Parameters
    ----------
    payload : mapping
        JSON-decoded ROM payload.
    skeleton : str
        Active skeleton name.
    joint_names : iterable of str
        Joints accepted by the active skeleton.

    Returns
    -------
    dict[str, list[AxisRomLimit]]
        Parsed ROM limit mapping with unknown joints ignored.

    Raises
    ------
    ValueError
        If the payload targets another skeleton or has an invalid schema.
    """

    skeleton_name = payload.get("skeleton")
    if skeleton_name is not None and normalize_skeleton_name(
        str(skeleton_name)
    ) != normalize_skeleton_name(skeleton):
        raise ValueError(
            f"ROM file skeleton {skeleton_name!r} does not match current skeleton {skeleton!r}.",
        )

    rom_limits = default_rom_limits(joint_names)
    joint_payload = payload.get("joint_limits", {})
    if not isinstance(joint_payload, Mapping):
        raise ValueError("ROM file must contain a 'joint_limits' object.")

    for joint_name, axis_payload in joint_payload.items():
        if joint_name not in rom_limits or not isinstance(axis_payload, Mapping):
            continue
        for axis, axis_key in enumerate(AXIS_FILE_KEYS):
            axis_limit = axis_payload.get(axis_key)
            if axis_limit is None:
                continue
            if not isinstance(axis_limit, Mapping):
                raise ValueError(
                    f"ROM limit for {joint_name}.{axis_key} must be an object."
                )
            minimum = float(axis_limit.get("min_deg", -180.0))
            maximum = float(axis_limit.get("max_deg", 180.0))
            if minimum > maximum:
                raise ValueError(
                    f"ROM limit for {joint_name}.{axis_key} has min greater than max."
                )
            rom_limits[joint_name][axis] = AxisRomLimit(
                enabled=bool(axis_limit.get("enabled", True)),
                minimum_deg=minimum,
                maximum_deg=maximum,
            )
    return rom_limits
