from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeypointSchema:
    """Named observation layout with explicit ordering and flip pairs.

    Parameters
    ----------
    name : str
        Layout identifier.
    joint_names : tuple of str
        Observation names in target tensor order.
    flip_pairs : tuple of pairs of str
        Symmetric observation names exchanged when flipping horizontally.
    """

    name: str
    joint_names: tuple[str, ...]
    flip_pairs: tuple[tuple[str, str], ...]

    def index(self, keypoint: str) -> int:
        """Return the index of a named observation.

        Parameters
        ----------
        keypoint : str
            Observation name.

        Returns
        -------
        int
            Index in joint_names.

        Raises
        ------
        ValueError
            If the observation is absent.
        """
        return self.joint_names.index(keypoint)

    def has(self, keypoint: str) -> bool:
        """Check whether an observation exists.

        Parameters
        ----------
        keypoint : str
            Observation name.

        Returns
        -------
        bool
            Whether the name belongs to this layout.
        """
        return keypoint in self.joint_names

    def indices(self, names: list[str] | tuple[str, ...]) -> list[int]:
        """Return indices in the requested observation order.

        Parameters
        ----------
        names : sequence of str
            Observation names to select.

        Returns
        -------
        list[int]
            Corresponding layout indices.

        Raises
        ------
        ValueError
            If any observation is absent.
        """
        return [self.index(name) for name in names]


MEDIAPIPE33 = KeypointSchema(
    name="mediapipe33",
    joint_names=(
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ),
    flip_pairs=(),
)

_SCHEMA_ALIASES = {"coco17": "coco", "mpii16": "mpii", "h36m17": "human36m"}


def _flip_pairs(names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    pairs = []
    for name in names:
        if name.startswith("left_"):
            partner = "right_" + name[5:]
        elif name.endswith("_left"):
            partner = name[:-5] + "_right"
        else:
            continue
        if partner in names:
            pairs.append((name, partner))
    return tuple(pairs)


def available_schemas() -> list[str]:
    """List rig layouts, accepted aliases, and standalone observation schemas.

    Returns
    -------
    list[str]
        Sorted names accepted by get_schema.
    """
    from . import SKELETON_ALIASES, SUPPORTED_SKELETONS

    return sorted(
        set(SUPPORTED_SKELETONS)
        | set(SKELETON_ALIASES)
        | set(_SCHEMA_ALIASES)
        | {MEDIAPIPE33.name}
    )


def get_schema(name: str) -> KeypointSchema:
    """Get observation ordering from a rig or a standalone detector layout.

    Parameters
    ----------
    name : str
        Rig name, rig alias, coco17/mpii16/h36m17 layout alias, or mediapipe33.
        Rig-backed layouts use the rig's exact names and ordering.

    Returns
    -------
    KeypointSchema
        Keypoint names and left/right flip pairs.

    Raises
    ------
    KeyError
        If the layout is unavailable.
    """
    from . import get_spec

    normalized = name.lower().replace("-", "_")
    if normalized == MEDIAPIPE33.name:
        return KeypointSchema(
            MEDIAPIPE33.name,
            MEDIAPIPE33.joint_names,
            _flip_pairs(MEDIAPIPE33.joint_names),
        )
    try:
        spec = get_spec(_SCHEMA_ALIASES.get(normalized, normalized))
    except KeyError as exc:
        raise KeyError(
            f"Unknown keypoint schema {name!r}; available: {available_schemas()}"
        ) from exc
    return KeypointSchema(normalized, spec.joint_names, _flip_pairs(spec.joint_names))
