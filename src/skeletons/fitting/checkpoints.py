from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..artifacts import resolve_checkpoint_reference
from ..model import SkeletalModel
from ..rigs import canonicalize_skeleton_name
from .fitter import SkeletalFitter
from .priors import JointLimitPrior, JointLimitStatistics, PoseVAE

RESERVED_CHECKPOINT_KEYS = {
    "format_version",
    "skeleton",
    "pose_prior_config",
    "pose_prior",
    "joint_limit_prior_config",
    "joint_limit_prior",
    "history",
}


@dataclass(frozen=True)
class FittingPriorBundle:
    """Loaded fitting priors and checkpoint metadata.

    Parameters
    ----------
    pose_prior : PoseVAE or None
        Loaded pose prior, if present in the checkpoint.
    joint_limit_prior : JointLimitPrior or None
        Loaded joint-limit prior, if present in the checkpoint.
    skeleton : str or None
        Skeleton name recorded by the checkpoint.
    checkpoint_path : pathlib.Path or None
        Resolved checkpoint path when the bundle was loaded from disk.
    history : tuple[dict[str, Any], ...]
        Training history records stored in the checkpoint.
    metadata : dict[str, Any]
        Additional checkpoint entries not consumed by the loader.
    """

    pose_prior: PoseVAE | None
    joint_limit_prior: JointLimitPrior | None
    skeleton: str | None = None
    checkpoint_path: Path | None = None
    history: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] | None = None

    def make_fitter(
        self,
        *,
        model: SkeletalModel,
        device: torch.device | str = "cpu",
    ) -> SkeletalFitter:
        """Create a skeletal fitter configured with the loaded priors.

        Parameters
        ----------
        model : SkeletalModel
            Skeleton model to fit.
        device : torch.device or str, optional
            Device used by the fitter.

        Returns
        -------
        SkeletalFitter
            Fitter configured with ``pose_prior`` and ``joint_limit_prior``.
        """

        return SkeletalFitter(
            model=model,
            pose_prior=self.pose_prior,
            joint_limit_prior=self.joint_limit_prior,
            device=device,
        )


def _normalize_skeleton_name(name: str) -> str:
    try:
        return canonicalize_skeleton_name(name)
    except KeyError:
        return name.lower().replace("-", "_")


def _linear_encoder_index(key: str) -> int:
    try:
        return int(key.split(".")[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected PoseVAE encoder key {key!r}.") from exc


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, Tensor]:
    return {
        key: value.detach().cpu() if isinstance(value, Tensor) else value
        for key, value in module.state_dict().items()
    }


def _coerce_pose_prior_config(config: Mapping[str, Any]) -> dict[str, int]:
    required = ("num_joints", "latent_dim", "hidden_dim", "num_hidden_layers")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(
            "pose_prior_config missing required key(s): " + ", ".join(missing)
        )
    return {key: int(config[key]) for key in required}


def _coerce_history(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        raise ValueError("checkpoint history must be an iterable of mappings.")
    history: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("checkpoint history entries must be mappings.")
        history.append(dict(item))
    return tuple(history)


def infer_pose_vae_config(state_dict: Mapping[str, Tensor]) -> dict[str, int]:
    """Infer ``PoseVAE`` constructor arguments from a state dict.

    Parameters
    ----------
    state_dict : Mapping[str, torch.Tensor]
        Pose VAE state dictionary.

    Returns
    -------
    dict[str, int]
        Constructor configuration with ``num_joints``, ``latent_dim``,
        ``hidden_dim``, and ``num_hidden_layers``.

    Raises
    ------
    ValueError
        If the state dictionary does not look like a supported ``PoseVAE``.
    """

    linear_keys = sorted(
        (
            key
            for key, value in state_dict.items()
            if key.startswith("encoder.")
            and key.endswith(".weight")
            and isinstance(value, Tensor)
            and value.ndim == 2
        ),
        key=_linear_encoder_index,
    )
    if not linear_keys:
        raise ValueError("Could not infer PoseVAE architecture from the checkpoint.")
    first_weight = state_dict[linear_keys[0]]
    input_dim = int(first_weight.shape[1])
    hidden_dim = int(first_weight.shape[0])
    try:
        latent_dim = int(state_dict["encoder_mu.weight"].shape[0])
    except KeyError as exc:
        raise ValueError("PoseVAE state dict is missing 'encoder_mu.weight'.") from exc
    if input_dim % 6 != 0:
        raise ValueError(
            f"PoseVAE input dimension must be divisible by 6, got {input_dim}."
        )
    return {
        "num_joints": input_dim // 6,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "num_hidden_layers": len(linear_keys),
    }


def build_fitting_prior_checkpoint(
    *,
    model: SkeletalModel,
    pose_prior: PoseVAE | None = None,
    joint_limit_prior: JointLimitPrior | None = None,
    history: Iterable[Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serializable checkpoint payload for fitting priors.

    Parameters
    ----------
    model : SkeletalModel
        Skeleton model associated with the priors.
    pose_prior : PoseVAE, optional
        Pose prior to include.
    joint_limit_prior : JointLimitPrior, optional
        Joint-limit prior to include.
    history : Iterable[Mapping[str, Any]], optional
        Training history records to store.
    extra : Mapping[str, Any], optional
        Additional checkpoint entries.

    Returns
    -------
    dict[str, Any]
        Checkpoint payload compatible with ``load_fitting_prior_checkpoint``.

    Raises
    ------
    ValueError
        If no priors are provided or extra keys collide with reserved keys.
    """

    if pose_prior is None and joint_limit_prior is None:
        raise ValueError("At least one prior must be provided.")
    payload: dict[str, Any] = {
        "format_version": 1,
        "skeleton": model.spec.name,
        "history": [dict(item) for item in history or ()],
    }
    if pose_prior is not None:
        pose_state = _cpu_state_dict(pose_prior)
        payload["pose_prior_config"] = infer_pose_vae_config(pose_state)
        payload["pose_prior"] = pose_state
    if joint_limit_prior is not None:
        payload["joint_limit_prior_config"] = {
            "barrier_scale": float(joint_limit_prior.barrier_scale),
        }
        payload["joint_limit_prior"] = _cpu_state_dict(joint_limit_prior)
    if extra:
        collisions = sorted(RESERVED_CHECKPOINT_KEYS.intersection(extra))
        if collisions:
            raise ValueError(
                "extra contains reserved checkpoint key(s): " + ", ".join(collisions)
            )
        payload.update(dict(extra))
    return payload


def save_fitting_prior_checkpoint(
    path: Path | str,
    *,
    model: SkeletalModel,
    pose_prior: PoseVAE | None = None,
    joint_limit_prior: JointLimitPrior | None = None,
    history: Iterable[Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Save fitting priors to a checkpoint file.

    Parameters
    ----------
    path : pathlib.Path or str
        Destination checkpoint path.
    model : SkeletalModel
        Skeleton model associated with the priors.
    pose_prior : PoseVAE, optional
        Pose prior to include.
    joint_limit_prior : JointLimitPrior, optional
        Joint-limit prior to include.
    history : Iterable[Mapping[str, Any]], optional
        Training history records to store.
    extra : Mapping[str, Any], optional
        Additional checkpoint entries.

    Returns
    -------
    pathlib.Path
        Written checkpoint path.

    Raises
    ------
    ValueError
        If no priors are provided or extra keys collide with reserved keys.
    """

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = build_fitting_prior_checkpoint(
        model=model,
        pose_prior=pose_prior,
        joint_limit_prior=joint_limit_prior,
        history=history,
        extra=extra,
    )
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_fitting_prior_checkpoint(
    path: Path | str,
    *,
    skeleton: str | None = None,
    map_location: torch.device | str = "cpu",
) -> FittingPriorBundle:
    """Load fitting priors from a checkpoint file or ``last_checkpoint`` reference.

    Parameters
    ----------
    path : pathlib.Path or str
        Checkpoint file, run directory, or ``last_checkpoint`` reference.
    skeleton : str, optional
        Expected skeleton name. If provided, it must match the checkpoint
        skeleton after canonicalization.
    map_location : torch.device or str, optional
        Device mapping passed to ``torch.load``.

    Returns
    -------
    FittingPriorBundle
        Loaded priors and checkpoint metadata.

    Raises
    ------
    ValueError
        If the checkpoint payload is malformed, has the wrong skeleton, or does
        not contain supported priors.
    FileNotFoundError
        If the checkpoint reference cannot be resolved.
    """

    checkpoint_path = resolve_checkpoint_reference(path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            f"Expected a mapping checkpoint in {checkpoint_path}, "
            f"got {type(checkpoint).__name__}."
        )

    checkpoint_skeleton = checkpoint.get("skeleton")
    if (
        skeleton is not None
        and checkpoint_skeleton is not None
        and _normalize_skeleton_name(str(checkpoint_skeleton))
        != _normalize_skeleton_name(skeleton)
    ):
        raise ValueError(
            f"Prior checkpoint skeleton {checkpoint_skeleton!r} does not "
            f"match requested skeleton {skeleton!r}."
        )

    pose_prior = None
    pose_state = checkpoint.get("pose_prior")
    if pose_state is not None:
        if not isinstance(pose_state, Mapping):
            raise ValueError("pose_prior must be a state-dict mapping.")
        pose_config = checkpoint.get("pose_prior_config")
        if pose_config is None:
            pose_config = infer_pose_vae_config(pose_state)
        elif not isinstance(pose_config, Mapping):
            raise ValueError("pose_prior_config must be a mapping.")
        pose_prior = PoseVAE(**_coerce_pose_prior_config(pose_config))
        pose_prior.load_state_dict(pose_state)
        pose_prior.eval()

    joint_limit_prior = None
    joint_limit_state = checkpoint.get("joint_limit_prior")
    if joint_limit_state is not None:
        if not isinstance(joint_limit_state, Mapping):
            raise ValueError("joint_limit_prior must be a state-dict mapping.")
        missing = [
            key
            for key in ("mean", "std", "lower", "upper")
            if key not in joint_limit_state
        ]
        if missing:
            raise ValueError(
                "joint_limit_prior missing required key(s): " + ", ".join(missing)
            )
        joint_limit_config = checkpoint.get("joint_limit_prior_config", {})
        if not isinstance(joint_limit_config, Mapping):
            raise ValueError("joint_limit_prior_config must be a mapping.")
        joint_limit_prior = JointLimitPrior(
            JointLimitStatistics(
                mean=joint_limit_state["mean"],
                std=joint_limit_state["std"],
                lower=joint_limit_state["lower"],
                upper=joint_limit_state["upper"],
            ),
            barrier_scale=float(joint_limit_config.get("barrier_scale", 10.0)),
        )
        joint_limit_prior.load_state_dict(joint_limit_state)
        joint_limit_prior.eval()

    if pose_prior is None and joint_limit_prior is None:
        raise ValueError(f"No supported priors were found in {checkpoint_path}.")

    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key not in RESERVED_CHECKPOINT_KEYS
    }
    return FittingPriorBundle(
        pose_prior=pose_prior,
        joint_limit_prior=joint_limit_prior,
        skeleton=str(checkpoint_skeleton) if checkpoint_skeleton is not None else None,
        checkpoint_path=checkpoint_path,
        history=_coerce_history(checkpoint.get("history")),
        metadata=metadata,
    )
