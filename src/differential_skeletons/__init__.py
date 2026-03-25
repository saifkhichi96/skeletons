from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .ik import (
    InverseKinematicsResult,
    estimate_rotations_from_joints,
    estimate_scales_from_joints,
)
from .losses import ForwardKinematicsLoss
from .model import SkeletalModel, SkeletalModelLayer
from .rigs import SUPPORTED_SKELETONS, SkeletonSpec, build_layer, create, get_spec
from .rotations import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rot6d,
    normalize_pose_repr,
    pose_repr_size,
    quaternion_to_matrix,
    rot6d_to_matrix,
    rotation_geodesic_distance,
    to_rotation_matrix,
)
from .utils import ModelOutput

if TYPE_CHECKING:
    from .rigs.coco import CocoModel, CocoModelLayer
    from .rigs.coco_wholebody import CocoWholeBodyModel, CocoWholeBodyModelLayer
    from .rigs.face68 import Face68Model, Face68ModelLayer
    from .rigs.halpe26 import Halpe26Model, Halpe26ModelLayer
    from .rigs.halpe_fullbody import HalpeFullBodyModel, HalpeFullBodyModelLayer
    from .rigs.hand21 import Hand21Model, Hand21ModelLayer
    from .rigs.human36m import Human36MModel, Human36MModelLayer
    from .rigs.mpii import MPIIModel, MPIIModelLayer
    from .rigs.spinetrack import SpineTrackModel, SpineTrackModelLayer


_RIG_EXPORTS = {
    "CocoModel": (".rigs.coco", "CocoModel"),
    "CocoModelLayer": (".rigs.coco", "CocoModelLayer"),
    "MPIIModel": (".rigs.mpii", "MPIIModel"),
    "MPIIModelLayer": (".rigs.mpii", "MPIIModelLayer"),
    "Human36MModel": (".rigs.human36m", "Human36MModel"),
    "Human36MModelLayer": (".rigs.human36m", "Human36MModelLayer"),
    "Halpe26Model": (".rigs.halpe26", "Halpe26Model"),
    "Halpe26ModelLayer": (".rigs.halpe26", "Halpe26ModelLayer"),
    "Hand21Model": (".rigs.hand21", "Hand21Model"),
    "Hand21ModelLayer": (".rigs.hand21", "Hand21ModelLayer"),
    "Face68Model": (".rigs.face68", "Face68Model"),
    "Face68ModelLayer": (".rigs.face68", "Face68ModelLayer"),
    "HalpeFullBodyModel": (".rigs.halpe_fullbody", "HalpeFullBodyModel"),
    "HalpeFullBodyModelLayer": (".rigs.halpe_fullbody", "HalpeFullBodyModelLayer"),
    "CocoWholeBodyModel": (".rigs.coco_wholebody", "CocoWholeBodyModel"),
    "CocoWholeBodyModelLayer": (".rigs.coco_wholebody", "CocoWholeBodyModelLayer"),
    "SpineTrackModel": (".rigs.spinetrack", "SpineTrackModel"),
    "SpineTrackModelLayer": (".rigs.spinetrack", "SpineTrackModelLayer"),
}


def __getattr__(name: str) -> object:
    target = _RIG_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "SUPPORTED_SKELETONS",
    "SkeletalModel",
    "SkeletalModelLayer",
    "ModelOutput",
    "SkeletonSpec",
    "ForwardKinematicsLoss",
    "CocoModel",
    "CocoModelLayer",
    "MPIIModel",
    "MPIIModelLayer",
    "Human36MModel",
    "Human36MModelLayer",
    "Halpe26Model",
    "Halpe26ModelLayer",
    "Hand21Model",
    "Hand21ModelLayer",
    "Face68Model",
    "Face68ModelLayer",
    "HalpeFullBodyModel",
    "HalpeFullBodyModelLayer",
    "CocoWholeBodyModel",
    "CocoWholeBodyModelLayer",
    "SpineTrackModel",
    "SpineTrackModelLayer",
    "create",
    "build_layer",
    "get_spec",
    "normalize_pose_repr",
    "pose_repr_size",
    "to_rotation_matrix",
    "axis_angle_to_matrix",
    "matrix_to_axis_angle",
    "matrix_to_rot6d",
    "quaternion_to_matrix",
    "rot6d_to_matrix",
    "rotation_geodesic_distance",
    "InverseKinematicsResult",
    "estimate_scales_from_joints",
    "estimate_rotations_from_joints",
]

__version__ = "0.2.0"
