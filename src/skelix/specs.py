from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from .templates import (
    absolute_positions_to_offsets,
    body_hand21_points,
    common_body_points,
    face68_points,
    generic_hand21_points,
    merge_points,
)


@dataclass(frozen=True)
class SkeletonSpec:
    name: str
    joint_names: tuple[str, ...]
    parents: tuple[int, ...]
    rest_offsets: torch.Tensor
    root_index: int
    metadata: dict[str, Any] | None = None

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)


COCO_NAMES = (
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
)

MPII_NAMES = (
    'right_ankle', 'right_knee', 'right_hip', 'left_hip', 'left_knee',
    'left_ankle', 'pelvis', 'thorax', 'upper_neck', 'head_top',
    'right_wrist', 'right_elbow', 'right_shoulder', 'left_shoulder',
    'left_elbow', 'left_wrist',
)

HUMAN36M_NAMES = (
    'root', 'right_hip', 'right_knee', 'right_foot', 'left_hip',
    'left_knee', 'left_foot', 'spine', 'thorax', 'neck_base', 'head',
    'left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder',
    'right_elbow', 'right_wrist',
)

HALPE26_NAMES = (
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'head', 'neck', 'hip', 'left_big_toe', 'right_big_toe',
    'left_small_toe', 'right_small_toe', 'left_heel', 'right_heel',
)

HAND21_NAMES = (
    'wrist', 'thumb1', 'thumb2', 'thumb3', 'thumb4',
    'forefinger1', 'forefinger2', 'forefinger3', 'forefinger4',
    'middle_finger1', 'middle_finger2', 'middle_finger3', 'middle_finger4',
    'ring_finger1', 'ring_finger2', 'ring_finger3', 'ring_finger4',
    'pinky_finger1', 'pinky_finger2', 'pinky_finger3', 'pinky_finger4',
)

FACE68_NAMES = tuple(f'face_{idx}' for idx in range(68))

LEFT_HAND_WHOLEBODY_NAMES = (
    'left_hand_root', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'left_thumb4',
    'left_forefinger1', 'left_forefinger2', 'left_forefinger3', 'left_forefinger4',
    'left_middle_finger1', 'left_middle_finger2', 'left_middle_finger3', 'left_middle_finger4',
    'left_ring_finger1', 'left_ring_finger2', 'left_ring_finger3', 'left_ring_finger4',
    'left_pinky_finger1', 'left_pinky_finger2', 'left_pinky_finger3', 'left_pinky_finger4',
)

RIGHT_HAND_WHOLEBODY_NAMES = (
    'right_hand_root', 'right_thumb1', 'right_thumb2', 'right_thumb3', 'right_thumb4',
    'right_forefinger1', 'right_forefinger2', 'right_forefinger3', 'right_forefinger4',
    'right_middle_finger1', 'right_middle_finger2', 'right_middle_finger3', 'right_middle_finger4',
    'right_ring_finger1', 'right_ring_finger2', 'right_ring_finger3', 'right_ring_finger4',
    'right_pinky_finger1', 'right_pinky_finger2', 'right_pinky_finger3', 'right_pinky_finger4',
)

HALPE_LEFT_HAND_NAMES = LEFT_HAND_WHOLEBODY_NAMES
HALPE_RIGHT_HAND_NAMES = RIGHT_HAND_WHOLEBODY_NAMES

COCO_WHOLEBODY_NAMES = (
    COCO_NAMES
    + ('left_big_toe', 'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel')
    + tuple(f'face-{idx}' for idx in range(68))
    + LEFT_HAND_WHOLEBODY_NAMES
    + RIGHT_HAND_WHOLEBODY_NAMES
)

HALPE_FULLBODY_NAMES = (
    HALPE26_NAMES
    + tuple(f'face-{idx}' for idx in range(68))
    + HALPE_LEFT_HAND_NAMES
    + HALPE_RIGHT_HAND_NAMES
)

SPINETRACK_NAMES = (
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
    'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck', 'hip',
    'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe',
    'left_heel', 'right_heel', 'spine_01', 'spine_02', 'spine_03',
    'spine_04', 'spine_05', 'left_latissimus', 'right_latissimus',
    'left_clavicle', 'right_clavicle', 'neck_02', 'neck_03',
)


def _face68_parent_mapping(root_parent: int) -> list[int]:
    parents = [0] * 68
    parents[27] = root_parent

    parents[0] = 27
    for idx in range(1, 17):
        parents[idx] = idx - 1

    parents[17] = 27
    for idx in range(18, 22):
        parents[idx] = idx - 1

    parents[22] = 27
    for idx in range(23, 27):
        parents[idx] = idx - 1

    for idx in range(28, 36):
        parents[idx] = idx - 1

    parents[36] = 27
    for idx in range(37, 42):
        parents[idx] = idx - 1

    parents[42] = 27
    for idx in range(43, 48):
        parents[idx] = idx - 1

    parents[48] = 27
    for idx in range(49, 60):
        parents[idx] = idx - 1

    parents[60] = 27
    for idx in range(61, 68):
        parents[idx] = idx - 1
    return parents


def _coco_parents() -> tuple[int, ...]:
    return (-1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14)


def _mpii_parents() -> tuple[int, ...]:
    return (1, 2, 6, 6, 3, 4, -1, 6, 7, 8, 11, 12, 8, 8, 13, 14)


def _human36m_parents() -> tuple[int, ...]:
    return (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)


def _halpe26_parents() -> tuple[int, ...]:
    return (17, 0, 0, 1, 2, 18, 18, 5, 6, 7, 8, 19, 19, 11, 12, 13, 14, 18, 19, -1, 15, 16, 20, 21, 15, 16)


def _hand21_parents(root_parent: int = -1) -> tuple[int, ...]:
    parents = [root_parent, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    return tuple(parents)


def _coco_wholebody_parents() -> tuple[int, ...]:
    parents = [-1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14, 15, 17, 15, 16, 20, 16]

    face_root_parent = 0
    face_parents = _face68_parent_mapping(root_parent=-999)
    for parent in face_parents:
        if parent == -999:
            parents.append(face_root_parent)
        else:
            parents.append(23 + parent)

    parents.extend([9, 91, 92, 93, 94, 91, 96, 97, 98, 91, 100, 101, 102, 91, 104, 105, 106, 91, 108, 109, 110])
    parents.extend([10, 112, 113, 114, 115, 112, 117, 118, 119, 112, 121, 122, 123, 112, 125, 126, 127, 112, 129, 130, 131])
    return tuple(parents)


def _halpe_fullbody_parents() -> tuple[int, ...]:
    parents = list(_halpe26_parents())

    face_root_parent = 0
    face_parents = _face68_parent_mapping(root_parent=-999)
    for parent in face_parents:
        if parent == -999:
            parents.append(face_root_parent)
        else:
            parents.append(26 + parent)

    parents.extend([9, 94, 95, 96, 97, 94, 99, 100, 101, 94, 103, 104, 105, 94, 107, 108, 109, 94, 111, 112, 113])
    parents.extend([10, 115, 116, 117, 118, 115, 120, 121, 122, 115, 124, 125, 126, 115, 128, 129, 130, 115, 132, 133, 134])
    return tuple(parents)


def _spinetrack_parents() -> tuple[int, ...]:
    return (17, 0, 0, 1, 2, 33, 34, 5, 6, 7, 8, 19, 19, 11, 12, 13, 14, 36, 30, -1, 15, 16, 20, 21, 15, 16, 19, 26, 27, 28, 29, 29, 29, 18, 18, 18, 35)


@lru_cache(maxsize=None)
def coco_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = _coco_parents()
    offsets = absolute_positions_to_offsets(COCO_NAMES, parents, common_body_points(), dtype=dtype)
    return SkeletonSpec('coco', COCO_NAMES, parents, offsets, root_index=0)


@lru_cache(maxsize=None)
def mpii_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = _mpii_parents()
    offsets = absolute_positions_to_offsets(MPII_NAMES, parents, common_body_points(), dtype=dtype)
    return SkeletonSpec('mpii', MPII_NAMES, parents, offsets, root_index=6)


@lru_cache(maxsize=None)
def human36m_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = _human36m_parents()
    offsets = absolute_positions_to_offsets(HUMAN36M_NAMES, parents, common_body_points(), dtype=dtype)
    return SkeletonSpec('human36m', HUMAN36M_NAMES, parents, offsets, root_index=0)


@lru_cache(maxsize=None)
def halpe26_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = _halpe26_parents()
    offsets = absolute_positions_to_offsets(HALPE26_NAMES, parents, common_body_points(), dtype=dtype)
    return SkeletonSpec('halpe26', HALPE26_NAMES, parents, offsets, root_index=19)


@lru_cache(maxsize=None)
def hand21_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = _hand21_parents()
    offsets = absolute_positions_to_offsets(HAND21_NAMES, parents, generic_hand21_points(), dtype=dtype)
    return SkeletonSpec('hand21', HAND21_NAMES, parents, offsets, root_index=0)


@lru_cache(maxsize=None)
def face68_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = tuple(_face68_parent_mapping(root_parent=-1))
    offsets = absolute_positions_to_offsets(FACE68_NAMES, parents, face68_points(prefix='face_', standalone=True), dtype=dtype)
    return SkeletonSpec('face68', FACE68_NAMES, parents, offsets, root_index=27)


@lru_cache(maxsize=None)
def coco_wholebody_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = _coco_wholebody_parents()
    positions = merge_points(
        common_body_points(),
        face68_points(prefix='face-'),
        body_hand21_points('left'),
        body_hand21_points('right'),
    )
    offsets = absolute_positions_to_offsets(COCO_WHOLEBODY_NAMES, parents, positions, dtype=dtype)
    return SkeletonSpec('coco_wholebody', COCO_WHOLEBODY_NAMES, parents, offsets, root_index=0)


@lru_cache(maxsize=None)
def halpe_fullbody_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = _halpe_fullbody_parents()
    positions = merge_points(
        common_body_points(),
        face68_points(prefix='face-'),
        body_hand21_points('left'),
        body_hand21_points('right'),
    )
    offsets = absolute_positions_to_offsets(HALPE_FULLBODY_NAMES, parents, positions, dtype=dtype)
    return SkeletonSpec('halpe_fullbody', HALPE_FULLBODY_NAMES, parents, offsets, root_index=19)


@lru_cache(maxsize=None)
def spinetrack_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = _spinetrack_parents()
    offsets = absolute_positions_to_offsets(SPINETRACK_NAMES, parents, common_body_points(), dtype=dtype)
    return SkeletonSpec('spinetrack', SPINETRACK_NAMES, parents, offsets, root_index=19)


SPEC_REGISTRY = {
    'coco': coco_spec,
    'mpii': mpii_spec,
    'human36m': human36m_spec,
    'halpe26': halpe26_spec,
    'hand21': hand21_spec,
    'face68': face68_spec,
    'halpefullbody': halpe_fullbody_spec,
    'halpe_fullbody': halpe_fullbody_spec,
    'cocowholebody': coco_wholebody_spec,
    'coco_wholebody': coco_wholebody_spec,
    'spinetrack': spinetrack_spec,
}


def get_spec(name: str, *, dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    key = name.lower().replace('-', '_')
    if key not in SPEC_REGISTRY:
        raise KeyError(f'Unknown skeleton spec: {name!r}')
    return SPEC_REGISTRY[key](dtype=dtype)
