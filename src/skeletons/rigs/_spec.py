from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import torch

JointType = Literal["free", "fixed", "hinge", "ball"]


@dataclass(frozen=True)
class LinkSpec:
    name: str
    mass: float = 1.0
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia_diag: tuple[float, float, float] = (1e-3, 1e-3, 1e-3)
    visual: dict[str, Any] | None = None
    collision: dict[str, Any] | None = None


@dataclass(frozen=True)
class MarkerSpec:
    name: str
    parent: str
    local_xyz: tuple[float, float, float]
    schema_name: str | None = None
    schema_index: int | None = None


@dataclass(frozen=True)
class ContactSpec:
    name: str
    parent: str
    local_xyz: tuple[float, float, float]
    radius: float = 0.02


@dataclass(frozen=True)
class SkeletonSpec:
    name: str
    joint_names: tuple[str, ...]
    parents: tuple[int, ...]
    rest_offsets: torch.Tensor
    root_index: int
    joint_types: tuple[JointType, ...] | None = None
    joint_axes: tuple[tuple[float, float, float] | None, ...] | None = None
    joint_limits: tuple[tuple[tuple[float, float], ...] | None, ...] | None = None
    links: tuple[LinkSpec, ...] = field(default_factory=tuple)
    markers: tuple[MarkerSpec, ...] = field(default_factory=tuple)
    contacts: tuple[ContactSpec, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        count = len(self.joint_names)
        if not count or len(set(self.joint_names)) != count:
            raise ValueError("Joint names must be nonempty and unique.")
        if len(self.parents) != count or self.rest_offsets.shape != (count, 3):
            raise ValueError("Parents and rest_offsets must match the joint count.")
        if not 0 <= self.root_index < count:
            raise ValueError("root_index must refer to an existing joint.")
        if self.parents[self.root_index] != -1:
            raise ValueError("The root parent must be -1.")
        for index in range(count):
            current = index
            visited: set[int] = set()
            while current != self.root_index:
                if current in visited or not 0 <= current < count:
                    raise ValueError("Parents must form a tree rooted at root_index.")
                visited.add(current)
                current = self.parents[current]
        if not torch.isfinite(self.rest_offsets).all():
            raise ValueError("Rest offsets must be finite.")
        for name in ("joint_types", "joint_axes", "joint_limits"):
            values = getattr(self, name)
            if values is not None and len(values) != count:
                raise ValueError(f"{name} must have one entry per joint.")
        for index in range(count):
            self.joint_dof(index)
            if index != self.root_index and self.joint_type(index) == "free":
                raise ValueError("Only the root may have a free joint.")
            if self.joint_axes is not None and self.joint_axes[index] is not None:
                axis = torch.as_tensor(self.joint_axes[index])
                if (
                    axis.shape != (3,)
                    or not torch.isfinite(axis).all()
                    or axis.norm() == 0
                ):
                    raise ValueError("Joint axes must be finite, nonzero 3-vectors.")
            limits = None if self.joint_limits is None else self.joint_limits[index]
            if limits is not None:
                if len(limits) != self.joint_dof(index):
                    raise ValueError("Limits must match the joint's rotational DOFs.")
                if any(not lo <= hi for lo, hi in limits):
                    raise ValueError("Joint limit minimum must not exceed maximum.")
        if self.links:
            if tuple(link.name for link in self.links) != self.joint_names:
                raise ValueError("Links must match joint names and ordering.")
            for link in self.links:
                if len(link.com) != 3 or len(link.inertia_diag) != 3:
                    raise ValueError(
                        "Link COMs and diagonal inertias must be 3-vectors."
                    )
                values = torch.as_tensor((link.mass, *link.com, *link.inertia_diag))
                if values.shape != (7,) or not torch.isfinite(values).all():
                    raise ValueError("Link masses, COMs, and inertias must be finite.")
                if link.mass <= 0 or any(value <= 0 for value in link.inertia_diag):
                    raise ValueError(
                        "Link masses and principal inertias must be positive."
                    )
        for frames in (self.markers, self.contacts):
            if len({frame.name for frame in frames}) != len(frames):
                raise ValueError(
                    "Attached frame names must be unique within their kind."
                )
            if any(frame.parent not in self.joint_names for frame in frames):
                raise ValueError("Attached frames must name an existing parent joint.")
            for frame in frames:
                point = torch.as_tensor(frame.local_xyz)
                if point.shape != (3,) or not torch.isfinite(point).all():
                    raise ValueError("Attached positions must be finite 3-vectors.")
        if any(not contact.radius > 0 for contact in self.contacts):
            raise ValueError("Contact radii must be positive.")

    def configured(self, **options: Any) -> SkeletonSpec:
        """Copy this rig with optional articulation and physical metadata.

        Parameters
        ----------
        **options
            Joint types, axes, or limits as mappings keyed by joint name or
            full sequences; links, markers, contacts, and metadata as values.

        Returns
        -------
        SkeletonSpec
            A configured copy; cached defaults remain unchanged.

        Raises
        ------
        ValueError
            If options, joint names, or metadata are invalid.
        """
        allowed = {
            "joint_types",
            "joint_axes",
            "joint_limits",
            "links",
            "markers",
            "contacts",
            "metadata",
        }
        if options.keys() - allowed:
            raise ValueError(
                f"Unknown spec options: {sorted(options.keys() - allowed)}"
            )
        updates = {}
        for name, value in options.items():
            if name in {"joint_types", "joint_axes", "joint_limits"} and isinstance(
                value, Mapping
            ):
                existing = getattr(self, name)
                values = (
                    list(existing)
                    if existing is not None
                    else [
                        self.joint_type(i) if name == "joint_types" else None
                        for i in range(self.num_joints)
                    ]
                )
                for joint_name, entry in value.items():
                    if joint_name not in self.joint_names:
                        raise ValueError(f"Unknown joint {joint_name!r}.")
                    values[self.joint_names.index(joint_name)] = entry
                updates[name] = tuple(values)
            else:
                updates[name] = (
                    tuple(value) if name != "metadata" and value is not None else value
                )
        return replace(self, **updates)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def joint_type(self, index: int) -> JointType:
        if self.joint_types is None:
            return "free" if index == self.root_index else "ball"
        return self.joint_types[index]

    def joint_dof(self, index: int) -> int:
        joint_type = self.joint_type(index)
        if joint_type == "fixed":
            return 0
        if joint_type == "hinge":
            return 1
        if joint_type in {"free", "ball"}:
            return 3
        raise ValueError(f"Unsupported joint type: {joint_type}")

    def marker_names(self) -> tuple[str, ...]:
        return tuple(marker.name for marker in self.markers)

    def contact_names(self) -> tuple[str, ...]:
        return tuple(contact.name for contact in self.contacts)
