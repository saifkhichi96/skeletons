from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PySide6 import QtCore

from ..fitting import (
    PerspectiveCamera,
    SkeletalFitter,
    load_fitting_prior_checkpoint,
)
from ..rigs import build_layer
from ..rotations import matrix_to_axis_angle, matrix_to_rot6d


class FittingWorker(QtCore.QObject):
    """Qt worker that fits each frame in a selected playground range.

    Parameters
    ----------
    skeleton_name : str
        Skeleton model name to instantiate in the worker thread.
    sequence : dict[str, torch.Tensor]
        Stacked frame targets and optional 2D/camera tensors.
    priors_path : pathlib.Path or None
        Optional fitting-prior checkpoint.
    mode : str
        Fitting mode, either ``"3d"`` or ``"2d"``.
    num_iters : int
        Optimizer iterations per frame.
    lr : float
        Optimizer learning rate.
    optimize_scales : bool
        Whether to optimize body scales.
    use_pose_prior_latent : bool
        Whether to use latent pose-prior fitting when available.
    init_from_ik : bool
        Whether the first 3D frame should initialize from inverse kinematics.
    progress_interval : int
        Progress callback cadence in optimizer steps.
    """

    progress = QtCore.Signal(object)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    canceled = QtCore.Signal()

    def __init__(
        self,
        *,
        skeleton_name: str,
        sequence: dict[str, torch.Tensor],
        priors_path: Path | None,
        mode: str,
        num_iters: int,
        lr: float,
        optimize_scales: bool,
        use_pose_prior_latent: bool,
        init_from_ik: bool,
        progress_interval: int,
    ) -> None:
        super().__init__()
        self.skeleton_name = skeleton_name
        self.sequence = sequence
        self.priors_path = priors_path
        self.mode = mode
        self.num_iters = int(num_iters)
        self.lr = float(lr)
        self.optimize_scales = optimize_scales
        self.use_pose_prior_latent = use_pose_prior_latent
        self.init_from_ik = init_from_ik
        self.progress_interval = max(int(progress_interval), 1)
        self._cancel_requested = False

    @QtCore.Slot()
    def run(self) -> None:
        """Execute the fitting loop and emit progress/result signals."""

        try:
            model = build_layer(self.skeleton_name)
            if self.priors_path is not None:
                priors = load_fitting_prior_checkpoint(
                    self.priors_path,
                    skeleton=model.spec.name,
                )
                fitter = priors.make_fitter(model=model)
            else:
                fitter = SkeletalFitter(model=model)
            total_frames = int(self.sequence["joints_3d"].shape[0])
            frame_joints: list[np.ndarray | None] = [None] * total_frames
            frame_full_pose: list[torch.Tensor] = []
            frame_transl: list[torch.Tensor] = []
            frame_scales: list[torch.Tensor] = []
            prev_state: dict[str, torch.Tensor] | None = None
            last_losses: dict[str, float] = {}
            total_completed_iters = 0

            for frame_index in range(total_frames):
                if self._cancel_requested:
                    self.canceled.emit()
                    return

                sample = self._slice_frame(frame_index)

                def on_progress(
                    step: int,
                    total: int,
                    joints: torch.Tensor,
                    losses: dict[str, float],
                ) -> bool:
                    joints_np = joints.squeeze(0).detach().cpu().numpy()
                    frame_joints[frame_index] = joints_np
                    self.progress.emit(
                        {
                            "frame_index": frame_index,
                            "frame_count": total_frames,
                            "iter": step,
                            "iters_per_frame": total,
                            "total_iter": frame_index * self.num_iters + step,
                            "total_iters": total_frames * self.num_iters,
                            "joints": joints_np,
                            "losses": dict(losses),
                        }
                    )
                    return not self._cancel_requested

                fit_kwargs = {
                    "num_iters": self.num_iters,
                    "lr": self.lr,
                    "optimize_scales": self.optimize_scales,
                    "use_pose_prior_latent": self.use_pose_prior_latent,
                    "progress_callback": on_progress,
                    "progress_interval": self.progress_interval,
                }
                if prev_state is not None:
                    fit_kwargs.update(prev_state)

                if self.mode == "3d":
                    result = fitter.fit_3d(
                        sample["joints_3d"].unsqueeze(0),
                        init_from_ik=self.init_from_ik and prev_state is None,
                        **fit_kwargs,
                    )
                elif self.mode == "2d":
                    if "joints_2d" not in sample:
                        raise ValueError(
                            "The selected range does not contain joints_2d."
                        )
                    camera = PerspectiveCamera(
                        fx=sample.get("fx", torch.tensor(1000.0)).reshape(()),
                        fy=sample.get("fy", torch.tensor(1000.0)).reshape(()),
                        cx=sample.get("cx", torch.tensor(512.0)).reshape(()),
                        cy=sample.get("cy", torch.tensor(512.0)).reshape(()),
                    )
                    confidences = sample.get("confidences")
                    result = fitter.fit_2d(
                        sample["joints_2d"].unsqueeze(0),
                        camera,
                        confidences=None
                        if confidences is None
                        else confidences.unsqueeze(0),
                        **fit_kwargs,
                    )
                else:
                    raise ValueError(f"Unsupported fitting mode: {self.mode!r}")

                if self._cancel_requested:
                    self.canceled.emit()
                    return

                output = result.model_output
                frame_full_pose.append(
                    matrix_to_axis_angle(
                        output.local_rotations.squeeze(0).detach().cpu()
                    )
                )
                frame_transl.append(output.transl.squeeze(0).detach().cpu())
                frame_scales.append(output.scales.squeeze(0).detach().cpu())
                frame_joints[frame_index] = (
                    output.joints.squeeze(0).detach().cpu().numpy()
                )
                # Extract warm-start state for next frame: convert rotations to rot6d.
                prev_state = {
                    "init_global_orient": matrix_to_rot6d(
                        output.local_rotations[..., model.root_index, :, :]
                    )
                    .squeeze(0)
                    .detach()
                    .cpu(),
                    "init_body_pose": matrix_to_rot6d(
                        output.local_rotations[
                            ..., list(model.non_root_joint_indices), :, :
                        ]
                    )
                    .squeeze(0)
                    .detach()
                    .cpu(),
                    "init_scales": output.scales.squeeze(0).detach().cpu(),
                    "init_transl": output.transl.squeeze(0).detach().cpu(),
                }
                last_losses = dict(result.losses)
                total_completed_iters += int(result.iterations)

            payload = {
                "full_pose": torch.stack(frame_full_pose, dim=0),
                "transl": torch.stack(frame_transl, dim=0),
                "scales": torch.stack(frame_scales, dim=0),
                "joints": np.stack(
                    [value for value in frame_joints if value is not None], axis=0
                ),
                "losses": last_losses,
                "iterations": total_completed_iters,
                "iters_per_frame": self.num_iters,
                "mode": self.mode,
                "frame_count": total_frames,
            }
            self.finished.emit(payload)
        except Exception as exc:
            self.failed.emit(str(exc))

    @QtCore.Slot()
    def cancel(self) -> None:
        """Request cancellation at the next safe fitting boundary."""

        self._cancel_requested = True

    def _slice_frame(self, frame_index: int) -> dict[str, torch.Tensor]:
        sample: dict[str, torch.Tensor] = {}
        total_frames = int(self.sequence["joints_3d"].shape[0])
        for key, value in self.sequence.items():
            if not isinstance(value, torch.Tensor):
                continue
            sample[key] = (
                value[frame_index] if value.shape[0] == total_frames else value
            )
        return sample
