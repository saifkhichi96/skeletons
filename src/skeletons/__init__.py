from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .artifacts import (
    RunArtifacts,
    log_status,
    make_run_name,
    resolve_checkpoint_reference,
    resolve_work_dir,
    sanitize_run_token,
    save_json,
    write_last_checkpoint,
)
from .fitting.checkpoints import (
    FittingPriorBundle,
    build_fitting_prior_checkpoint,
    infer_pose_vae_config,
    load_fitting_prior_checkpoint,
    save_fitting_prior_checkpoint,
)
from .hybrid import (
    HybridIKEvaluation,
    HybridIKResult,
    HybridRegressorTrainingState,
    JointRegressor,
    evaluate_hybrid_ik_regressor,
    solve_hybrid_ik,
    train_joint_regressor_epoch,
)
from .ik import (
    CyclicCoordinateDescentIK,
    DampedLeastSquaresIK,
    GradientIK,
    IKSolution,
    InverseKinematicsResult,
    estimate_rotations_from_joints,
    estimate_scales_from_joints,
)
from .lifting import (
    LifterTrainingState,
    LiftingEvaluation,
    PoseLifter,
    TemporalPoseLifter,
    evaluate_lifter,
    train_lifter_epoch,
)
from .losses import (
    ForwardKinematicsLoss,
    center_of_mass_support_loss,
    foot_sliding_loss,
    joint_limit_loss,
    keypoint_3d_loss,
    reprojection_loss,
    robust_penalty,
    temporal_smoothness_loss,
)
from .metrics import (
    joint_position_error,
    mean_per_joint_position_error,
    mpjpe,
    pa_mpjpe,
    percentage_of_correct_keypoints,
    procrustes_align,
    procrustes_aligned_mpjpe,
)
from .model import SkeletalModel, SkeletalModelLayer
from .playground import (
    ANIMATION_PRESETS,
    AXIS_FILE_KEYS,
    AXIS_NAMES,
    AnimationPreset,
    AxisRomLimit,
    FittingExportPayload,
    SyntheticFittingControls,
    apply_rom_payload,
    available_animation_presets,
    build_fit_export_payload,
    build_walk_tensors,
    create_synthetic_fitting_dataset,
    default_rom_limits,
    normalize_skeleton_name,
    serialize_rom_limits,
)
from .pseudo_labeling import (
    PseudoLabelPriors,
    PseudoLabelResult,
    fit_2d_pseudo_labels,
    save_pseudo_label_npz,
    train_pseudo_label_priors,
)
from .retargeting import (
    JointMapping,
    RetargetedJoints,
    RetargetingResult,
    build_joint_mapping,
    retarget_joint_positions,
    retarget_skeleton,
)
from .rigs import (
    SKELETON_ALIASES,
    SUPPORTED_SKELETONS,
    ContactSpec,
    KeypointSchema,
    LinkSpec,
    MarkerSpec,
    SkeletonSpec,
    available_schemas,
    build_layer,
    canonicalize_skeleton_name,
    create,
    get_schema,
    get_spec,
    list_supported_skeletons,
)
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
from .sequences import (
    SequenceFittingResult,
    TemporalPriorTrainingResult,
    denoise_joint_sequences,
    fit_2d_joint_sequences,
    train_temporal_prior,
)
from .synthetic import (
    SyntheticFittingDataset,
    SyntheticLiftingDataset,
    SyntheticPoseBatch,
    SyntheticSequenceBatch,
    generate_orthographic_lifting_dataset,
    generate_random_pose_batch,
    generate_random_walk_sequences,
    generate_synthetic_fitting_dataset,
    make_perspective_camera,
    make_weak_perspective_camera,
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
    "SKELETON_ALIASES",
    "SUPPORTED_SKELETONS",
    "RunArtifacts",
    "FittingPriorBundle",
    "JointRegressor",
    "HybridIKEvaluation",
    "HybridIKResult",
    "HybridRegressorTrainingState",
    "LinkSpec",
    "MarkerSpec",
    "ContactSpec",
    "KeypointSchema",
    "SkeletalModel",
    "SkeletalModelLayer",
    "PseudoLabelPriors",
    "PseudoLabelResult",
    "PoseLifter",
    "TemporalPoseLifter",
    "LifterTrainingState",
    "LiftingEvaluation",
    "ANIMATION_PRESETS",
    "AXIS_FILE_KEYS",
    "AXIS_NAMES",
    "AnimationPreset",
    "AxisRomLimit",
    "FittingExportPayload",
    "SyntheticFittingControls",
    "JointMapping",
    "RetargetedJoints",
    "RetargetingResult",
    "SequenceFittingResult",
    "TemporalPriorTrainingResult",
    "ModelOutput",
    "SkeletonSpec",
    "ForwardKinematicsLoss",
    "IKSolution",
    "DampedLeastSquaresIK",
    "GradientIK",
    "CyclicCoordinateDescentIK",
    "evaluate_lifter",
    "log_status",
    "make_run_name",
    "resolve_checkpoint_reference",
    "resolve_work_dir",
    "sanitize_run_token",
    "save_json",
    "write_last_checkpoint",
    "evaluate_hybrid_ik_regressor",
    "solve_hybrid_ik",
    "train_joint_regressor_epoch",
    "build_fitting_prior_checkpoint",
    "available_schemas",
    "center_of_mass_support_loss",
    "fit_2d_pseudo_labels",
    "foot_sliding_loss",
    "get_schema",
    "infer_pose_vae_config",
    "joint_limit_loss",
    "keypoint_3d_loss",
    "load_fitting_prior_checkpoint",
    "reprojection_loss",
    "robust_penalty",
    "save_pseudo_label_npz",
    "save_fitting_prior_checkpoint",
    "train_pseudo_label_priors",
    "apply_rom_payload",
    "available_animation_presets",
    "build_fit_export_payload",
    "build_walk_tensors",
    "create_synthetic_fitting_dataset",
    "default_rom_limits",
    "normalize_skeleton_name",
    "serialize_rom_limits",
    "train_lifter_epoch",
    "temporal_smoothness_loss",
    "joint_position_error",
    "mean_per_joint_position_error",
    "mpjpe",
    "pa_mpjpe",
    "percentage_of_correct_keypoints",
    "procrustes_align",
    "procrustes_aligned_mpjpe",
    "build_joint_mapping",
    "retarget_joint_positions",
    "retarget_skeleton",
    "denoise_joint_sequences",
    "fit_2d_joint_sequences",
    "train_temporal_prior",
    "SyntheticFittingDataset",
    "SyntheticLiftingDataset",
    "SyntheticPoseBatch",
    "SyntheticSequenceBatch",
    "generate_orthographic_lifting_dataset",
    "generate_random_pose_batch",
    "generate_random_walk_sequences",
    "generate_synthetic_fitting_dataset",
    "make_perspective_camera",
    "make_weak_perspective_camera",
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
    "canonicalize_skeleton_name",
    "list_supported_skeletons",
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

__version__ = "1.0.0"
