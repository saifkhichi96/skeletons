from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import torch

from .fitting import FittingResult, SkeletalFitter
from .metrics import joint_position_error
from .model import SkeletalModel
from .rigs._spec import SkeletonSpec

Tensor = torch.Tensor
SkeletonLike: TypeAlias = SkeletalModel | SkeletonSpec

_JOINT_SYNONYMS = {
    "left_ankle": "left_foot",
    "right_ankle": "right_foot",
}


@dataclass(frozen=True)
class JointMapping:
    """Index mapping between a source and target skeleton.

    Parameters
    ----------
    source_indices : tuple[int, ...]
        Source skeleton joint indices.
    target_indices : tuple[int, ...]
        Target skeleton joint indices matched to ``source_indices``.
    joint_names : tuple[str, ...]
        Target-side semantic names for each mapped joint.

    Raises
    ------
    ValueError
        If the mapping fields have inconsistent lengths.
    """

    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    joint_names: tuple[str, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.source_indices),
            len(self.target_indices),
            len(self.joint_names),
        }
        if len(lengths) != 1:
            raise ValueError("Mapping fields must have the same length.")

    def __len__(self) -> int:
        return len(self.source_indices)

    def pairs(self) -> tuple[tuple[int, int, str], ...]:
        """Return ``(source_index, target_index, joint_name)`` tuples.

        Returns
        -------
        tuple[tuple[int, int, str], ...]
            Mapping entries in source-joint order.
        """

        return tuple(
            zip(self.source_indices, self.target_indices, self.joint_names)
        )

    def index_tensors(self, *, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
        """Return source and target index tensors.

        Parameters
        ----------
        device : torch.device, optional
            Device for the returned tensors.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Long tensors containing source and target indices.
        """

        return (
            torch.tensor(self.source_indices, dtype=torch.long, device=device),
            torch.tensor(self.target_indices, dtype=torch.long, device=device),
        )


@dataclass(frozen=True)
class RetargetedJoints:
    """Sparse target-skeleton observations produced from source joints.

    Parameters
    ----------
    joints : torch.Tensor
        Target-layout joint tensor with shape ``[..., target_joints, 3]``.
    weights : torch.Tensor
        Joint-confidence weights with shape ``[..., target_joints]``. Unmapped
        joints have zero weight.
    mapping : JointMapping
        Source-to-target mapping used to fill ``joints`` and ``weights``.
    """

    joints: Tensor
    weights: Tensor
    mapping: JointMapping


@dataclass(frozen=True)
class RetargetingResult:
    """Output returned by optimization-based skeleton retargeting.

    Parameters
    ----------
    fitting_result : FittingResult
        Result returned by ``SkeletalFitter.fit_3d`` for the target skeleton.
    observations : RetargetedJoints
        Sparse target-layout observations used for fitting.
    shared_joint_mpjpe : float
        Mean target error over mapped joints only.
    """

    fitting_result: FittingResult
    observations: RetargetedJoints
    shared_joint_mpjpe: float


def _as_spec(value: SkeletonLike) -> SkeletonSpec:
    if isinstance(value, SkeletalModel):
        return value.spec
    return value


def _normalize_joint_name(name: str) -> str:
    return name.lower().replace("-", "_")


def _semantic_joint_key(name: str) -> str:
    normalized = _normalize_joint_name(name)
    return _JOINT_SYNONYMS.get(normalized, normalized)


def build_joint_mapping(
    source: SkeletonLike,
    target: SkeletonLike,
    *,
    joint_map: Mapping[str, str] | None = None,
    use_synonyms: bool = True,
    min_joints: int = 1,
) -> JointMapping:
    """Build a source-to-target joint-name mapping.

    Parameters
    ----------
    source : SkeletonLike
        Source skeleton model or specification.
    target : SkeletonLike
        Target skeleton model or specification.
    joint_map : Mapping[str, str], optional
        Explicit mapping from source joint names to target joint names.
    use_synonyms : bool, optional
        When ``True``, match known semantic synonyms such as ankle/foot after
        exact name matching.
    min_joints : int, optional
        Minimum number of mapped joints required.

    Returns
    -------
    JointMapping
        Matched source and target joint indices.

    Raises
    ------
    ValueError
        If fewer than ``min_joints`` joints are mapped or if a target joint is
        mapped more than once.
    """

    source_spec = _as_spec(source)
    target_spec = _as_spec(target)
    explicit_map = {
        _normalize_joint_name(source_name): _normalize_joint_name(target_name)
        for source_name, target_name in (joint_map or {}).items()
    }
    target_by_name = {
        _normalize_joint_name(name): idx
        for idx, name in enumerate(target_spec.joint_names)
    }
    target_by_semantic_name = {
        _semantic_joint_key(name): idx
        for idx, name in enumerate(target_spec.joint_names)
    }

    source_indices: list[int] = []
    target_indices: list[int] = []
    joint_names: list[str] = []
    seen_targets: set[int] = set()

    for source_idx, source_name in enumerate(source_spec.joint_names):
        normalized_source_name = _normalize_joint_name(source_name)
        target_name = explicit_map.get(normalized_source_name, normalized_source_name)
        target_idx = target_by_name.get(target_name)
        if target_idx is None and use_synonyms:
            target_idx = target_by_semantic_name.get(_semantic_joint_key(target_name))
        if target_idx is None:
            continue
        if target_idx in seen_targets:
            raise ValueError(
                f"Target joint {target_spec.joint_names[target_idx]!r} is mapped more than once."
            )
        seen_targets.add(target_idx)
        source_indices.append(source_idx)
        target_indices.append(target_idx)
        joint_names.append(target_spec.joint_names[target_idx])

    if len(source_indices) < min_joints:
        raise ValueError(
            f"Expected at least {min_joints} mapped joints between "
            f"{source_spec.name!r} and {target_spec.name!r}, got {len(source_indices)}."
        )

    return JointMapping(
        source_indices=tuple(source_indices),
        target_indices=tuple(target_indices),
        joint_names=tuple(joint_names),
    )


def retarget_joint_positions(
    source_joints: Tensor,
    source: SkeletonLike,
    target: SkeletonLike,
    *,
    mapping: JointMapping | None = None,
    source_weights: Tensor | None = None,
    fill_value: float = 0.0,
) -> RetargetedJoints:
    """Project source joints into a sparse target skeleton layout.

    Parameters
    ----------
    source_joints : torch.Tensor
        Source-layout joints with shape ``[..., source_joints, 3]``.
    source : SkeletonLike
        Source skeleton model or specification.
    target : SkeletonLike
        Target skeleton model or specification.
    mapping : JointMapping, optional
        Precomputed source-to-target mapping. If omitted, one is built by name.
    source_weights : torch.Tensor, optional
        Source-layout confidence weights broadcastable to
        ``source_joints.shape[:-1]``.
    fill_value : float, optional
        Coordinate value used for unmapped target joints.

    Returns
    -------
    RetargetedJoints
        Sparse target-layout observations and fitting weights.

    Raises
    ------
    ValueError
        If input shapes or weights are invalid.
    """

    source_spec = _as_spec(source)
    target_spec = _as_spec(target)
    if source_joints.shape[-2:] != (source_spec.num_joints, 3):
        raise ValueError(
            f"source_joints must have shape [..., {source_spec.num_joints}, 3]."
        )

    mapping = mapping or build_joint_mapping(source_spec, target_spec)
    source_index, target_index = mapping.index_tensors(device=source_joints.device)
    batch_shape = source_joints.shape[:-2]
    target_joints = source_joints.new_full(
        batch_shape + (target_spec.num_joints, 3),
        fill_value,
    )
    target_weights = source_joints.new_zeros(batch_shape + (target_spec.num_joints,))

    target_joints[..., target_index, :] = source_joints[..., source_index, :]
    if source_weights is None:
        target_weights[..., target_index] = 1.0
    else:
        source_weights = source_weights.to(
            dtype=source_joints.dtype,
            device=source_joints.device,
        )
        try:
            source_weights = torch.broadcast_to(
                source_weights,
                batch_shape + (source_spec.num_joints,),
            )
        except RuntimeError as exc:
            raise ValueError(
                "source_weights must broadcast to source_joints.shape[:-1]."
            ) from exc
        if (source_weights < 0).any().item():
            raise ValueError("source_weights must be non-negative.")
        target_weights[..., target_index] = source_weights[..., source_index]

    return RetargetedJoints(
        joints=target_joints,
        weights=target_weights,
        mapping=mapping,
    )


def retarget_skeleton(
    source_joints: Tensor,
    *,
    source_model: SkeletalModel,
    target_model: SkeletalModel,
    mapping: JointMapping | None = None,
    source_weights: Tensor | None = None,
    fitter: SkeletalFitter | None = None,
    num_iters: int = 300,
    lr: float = 1e-2,
    optimize_scales: bool = True,
    use_pose_prior_latent: bool = False,
    init_from_ik: bool | None = None,
) -> RetargetingResult:
    """Fit a target skeleton to source skeleton joints.

    Parameters
    ----------
    source_joints : torch.Tensor
        Source-layout joints with shape ``[..., source_joints, 3]``.
    source_model : SkeletalModel
        Source skeleton model.
    target_model : SkeletalModel
        Target skeleton model to optimize.
    mapping : JointMapping, optional
        Precomputed source-to-target mapping. If omitted, one is built by name.
    source_weights : torch.Tensor, optional
        Source-layout confidence weights broadcastable to
        ``source_joints.shape[:-1]``.
    fitter : SkeletalFitter, optional
        Existing fitter configured for ``target_model``.
    num_iters : int, optional
        Number of optimization iterations.
    lr : float, optional
        Adam learning rate used by the fitter.
    optimize_scales : bool, optional
        Whether to optimize target skeleton scales.
    use_pose_prior_latent : bool, optional
        Whether to optimize through a pose-prior latent when the fitter has one.
    init_from_ik : bool, optional
        Whether to initialize from IK. Defaults to ``True`` only when every
        target joint is observed.

    Returns
    -------
    RetargetingResult
        Fitted target model output, sparse observations, and shared-joint error.

    Raises
    ------
    ValueError
        If the fitter is configured for a different target model.
    """

    mapping = mapping or build_joint_mapping(source_model, target_model)
    observations = retarget_joint_positions(
        source_joints,
        source_model,
        target_model,
        mapping=mapping,
        source_weights=source_weights,
    )
    if fitter is None:
        fitter = SkeletalFitter(model=target_model, device=source_joints.device)
    elif fitter.model is not target_model:
        raise ValueError("fitter must be configured with target_model.")

    if init_from_ik is None:
        init_from_ik = len(mapping) == target_model.NUM_JOINTS

    fitting_result = fitter.fit_3d(
        observations.joints,
        weights=observations.weights,
        num_iters=num_iters,
        lr=lr,
        optimize_scales=optimize_scales,
        use_pose_prior_latent=use_pose_prior_latent,
        init_from_ik=init_from_ik,
    )
    _, target_index = mapping.index_tensors(device=observations.joints.device)
    shared_errors = joint_position_error(
        fitting_result.model_output.joints[..., target_index, :],
        observations.joints[..., target_index, :],
    )
    shared_weights = observations.weights[..., target_index]
    shared_joint_mpjpe = (
        (shared_errors * shared_weights).sum()
        / shared_weights.sum().clamp_min(torch.finfo(shared_errors.dtype).eps)
    ).item()

    return RetargetingResult(
        fitting_result=fitting_result,
        observations=observations,
        shared_joint_mpjpe=shared_joint_mpjpe,
    )
