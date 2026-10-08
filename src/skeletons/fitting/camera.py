from __future__ import annotations

from dataclasses import dataclass

import torch


def _match_point_batch(value: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    while value.ndim < points.ndim - 1:
        value = value.unsqueeze(-1)
    return value


@dataclass
class PerspectiveCamera:
    """Minimal pinhole camera with broadcast-friendly intrinsics."""

    fx: torch.Tensor
    fy: torch.Tensor
    cx: torch.Tensor
    cy: torch.Tensor

    def to(self, device: torch.device | str) -> "PerspectiveCamera":
        """Move camera tensors to `device`."""

        device = torch.device(device)
        return PerspectiveCamera(
            fx=self.fx.to(device),
            fy=self.fy.to(device),
            cx=self.cx.to(device),
            cy=self.cy.to(device),
        )

    def project(self, points_3d: torch.Tensor) -> torch.Tensor:
        """Project 3D points to image coordinates."""

        z = points_3d[..., 2].clamp_min(1e-6)
        fx = _match_point_batch(self.fx, points_3d)
        fy = _match_point_batch(self.fy, points_3d)
        cx = _match_point_batch(self.cx, points_3d)
        cy = _match_point_batch(self.cy, points_3d)
        x = fx * (points_3d[..., 0] / z) + cx
        y = fy * (points_3d[..., 1] / z) + cy
        return torch.stack([x, y], dim=-1)


@dataclass
class WeakPerspectiveCamera:
    """Weak-perspective camera with isotropic scale and image translation."""

    scale: torch.Tensor
    tx: torch.Tensor
    ty: torch.Tensor

    def to(self, device: torch.device | str) -> "WeakPerspectiveCamera":
        """Move camera tensors to `device`."""

        device = torch.device(device)
        return WeakPerspectiveCamera(
            scale=self.scale.to(device),
            tx=self.tx.to(device),
            ty=self.ty.to(device),
        )

    def project(self, points_3d: torch.Tensor) -> torch.Tensor:
        """Project 3D points with a weak-perspective transform."""

        scale = _match_point_batch(self.scale, points_3d)
        tx = _match_point_batch(self.tx, points_3d)
        ty = _match_point_batch(self.ty, points_3d)
        x = scale * points_3d[..., 0] + tx
        y = scale * points_3d[..., 1] + ty
        return torch.stack([x, y], dim=-1)
