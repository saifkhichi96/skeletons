from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .rigs._spec import SkeletonSpec
from .rotations import (
    identity_pose,
    normalize_pose_repr,
    pose_repr_size,
    to_rotation_matrix,
)
from .utils import ModelOutput


def _broadcast_batch_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    filtered = [shape for shape in shapes if len(shape) > 0]
    if not filtered:
        return ()
    return torch.broadcast_shapes(*filtered)


def _expand_to_batch(
    tensor: torch.Tensor, batch_shape: tuple[int, ...], tail_dims: int
) -> torch.Tensor:
    tensor_batch = tensor.shape[:-tail_dims]
    if tensor_batch == batch_shape:
        return tensor
    if len(tensor_batch) > len(batch_shape):
        raise ValueError(
            f"Cannot broadcast batch shape {tensor_batch} to {batch_shape}."
        )
    reshaped = tensor.reshape(
        (1,) * (len(batch_shape) - len(tensor_batch)) + tensor.shape
    )
    return reshaped.expand(batch_shape + tensor.shape[-tail_dims:])


def _make_transform(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    batch_shape = rotation.shape[:-2]
    transform = torch.zeros(
        batch_shape + (4, 4), dtype=rotation.dtype, device=rotation.device
    )
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    transform[..., 3, 3] = 1.0
    return transform


@dataclass(frozen=True)
class ArticulatedJoint:
    index: int
    name: str
    joint_type: str
    parent_body_index: int | None
    child_body_index: int
    pose_index: int
    body_scale_index: int


@dataclass(frozen=True)
class RigidBodyNode:
    index: int
    name: str
    parent_index: int | None
    incoming_joint_index: int
    child_body_indices: tuple[int, ...]


@dataclass(frozen=True)
class ArticulatedSkeleton:
    name: str
    root_body_index: int
    body_nodes: tuple[RigidBodyNode, ...]
    joints: tuple[ArticulatedJoint, ...]
    topological_order: tuple[int, ...]

    @classmethod
    def from_spec(cls, spec: SkeletonSpec) -> ArticulatedSkeleton:
        parents = tuple(int(parent) for parent in spec.parents)
        roots = [idx for idx, parent in enumerate(parents) if parent == -1]
        if len(roots) != 1:
            raise ValueError(f"Exactly one root is required, got {roots}.")
        root_body_index = roots[0]
        if root_body_index != int(spec.root_index):
            raise ValueError(
                f"Spec root_index {spec.root_index} does not match tree root {root_body_index}."
            )

        children = [[] for _ in parents]
        for child_idx, parent_idx in enumerate(parents):
            if parent_idx == -1:
                continue
            if parent_idx < 0 or parent_idx >= len(parents):
                raise ValueError(
                    f"Invalid parent index {parent_idx} for joint {child_idx}."
                )
            children[parent_idx].append(child_idx)

        topological_order = cls._compute_topological_order(root_body_index, children)

        body_nodes = tuple(
            RigidBodyNode(
                index=body_idx,
                name=spec.joint_names[body_idx],
                parent_index=None if parents[body_idx] == -1 else parents[body_idx],
                incoming_joint_index=body_idx,
                child_body_indices=tuple(children[body_idx]),
            )
            for body_idx in range(len(parents))
        )
        joints = tuple(
            ArticulatedJoint(
                index=body_idx,
                name=spec.joint_names[body_idx],
                joint_type="free" if parents[body_idx] == -1 else "ball",
                parent_body_index=None
                if parents[body_idx] == -1
                else parents[body_idx],
                child_body_index=body_idx,
                pose_index=body_idx,
                body_scale_index=body_idx,
            )
            for body_idx in range(len(parents))
        )

        return cls(
            name=spec.name,
            root_body_index=root_body_index,
            body_nodes=body_nodes,
            joints=joints,
            topological_order=topological_order,
        )

    @staticmethod
    def _compute_topological_order(
        root_body_index: int, children: list[list[int]]
    ) -> tuple[int, ...]:
        order: list[int] = []
        stack = [root_body_index]
        visited: set[int] = set()
        while stack:
            body_idx = stack.pop()
            if body_idx in visited:
                continue
            visited.add(body_idx)
            order.append(body_idx)
            stack.extend(reversed(children[body_idx]))

        if len(order) != len(children):
            raise ValueError("The articulated skeleton must be a connected tree.")
        return tuple(order)


class SkeletalModel(nn.Module):
    def __init__(
        self,
        spec: SkeletonSpec,
        *,
        pose_repr: str = "axis_angle",
        create_global_orient: bool = True,
        global_orient: torch.Tensor | None = None,
        create_body_pose: bool = True,
        body_pose: torch.Tensor | None = None,
        create_transl: bool = True,
        transl: torch.Tensor | None = None,
        create_scales: bool = True,
        scales: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        batch_size: int = 1,
    ) -> None:
        """SkeletalModel constructor

        Parameters
        ----------
        create_global_orient: bool, optional
            Flag for creating a member variable for the global orientation
            of the body. (default = True)
        global_orient: torch.tensor, optional, Bx3
            The default value for the global orientation variable.
            (default = None)
        create_body_pose: bool, optional
            Flag for creating a member variable for the pose of the body.
            (default = True)
        body_pose: torch.tensor, optional, Bx(Body Joints * 3)
            The default value for the body pose variable.
            (default = None)
        create_transl: bool, optional
            Flag for creating a member variable for the translation
            of the body. (default = True)
        transl: torch.tensor, optional, Bx3
            The default value for the transl variable.
            (default = None)
        create_scales: bool, optional
            Flag for creating a member variable for the scales.
            (default = True)
        scales: torch.tensor, optional, Bx(Joints) or Bx(Joints*3)
            The default value for the scales variable.
            (default = None)
        dtype: torch.dtype, optional
            The data type for the created variables
        batch_size: int, optional
            The batch size used for creating the member variables
        """
        super().__init__()
        self._rig = ArticulatedSkeleton.from_spec(spec)
        self.skeleton = self._rig
        self.spec = spec
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        self.dtype = dtype
        self._parameter_batch_shape = (self.batch_size,)

        self.pose_repr = normalize_pose_repr(pose_repr)
        self.body_nodes = self._rig.body_nodes
        self.joints = self._rig.joints
        self.joint_names = tuple(spec.joint_names)
        self.parents = tuple(
            -1 if body.parent_index is None else body.parent_index
            for body in self.body_nodes
        )
        self.child_body_indices = tuple(
            body.child_body_indices for body in self.body_nodes
        )
        self.root_index = int(spec.root_index)

        self.NUM_BODIES = len(self.body_nodes)
        self.NUM_JOINTS = len(self.joint_names)
        self.NUM_BODY_JOINTS = self.NUM_JOINTS - 1

        self.non_root_joint_indices = tuple(
            idx for idx in range(self.NUM_JOINTS) if idx != self.root_index
        )
        self.joint_name_to_index = {
            name: idx for idx, name in enumerate(self.joint_names)
        }
        self.topological_order = self._rig.topological_order
        offset_scale_body_indices = [self.root_index] * self.NUM_JOINTS
        for joint_idx, parent_idx in enumerate(self.parents):
            if parent_idx != -1:
                offset_scale_body_indices[joint_idx] = parent_idx
        self.register_buffer(
            "offset_scale_body_indices",
            torch.tensor(offset_scale_body_indices, dtype=torch.long),
            persistent=False,
        )

        rest_offsets = spec.rest_offsets.detach().clone().to(dtype=dtype)
        self.register_buffer("rest_offsets", rest_offsets)

        self._register_optional_state(
            name="global_orient",
            value=global_orient,
            default=self._default_global_orient(
                self._parameter_batch_shape, pose_repr=self.pose_repr, dtype=dtype
            ),
            create=create_global_orient,
        )
        self._register_optional_state(
            name="body_pose",
            value=body_pose,
            default=self._default_body_pose(
                self._parameter_batch_shape, pose_repr=self.pose_repr, dtype=dtype
            ),
            create=create_body_pose,
        )
        self._register_optional_state(
            name="transl",
            value=transl,
            default=self._default_translation(self._parameter_batch_shape, dtype=dtype),
            create=create_transl,
        )
        self._register_optional_state(
            name="scales",
            value=scales,
            default=self._default_scales(self._parameter_batch_shape, dtype=dtype),
            create=create_scales,
        )

    def name(self) -> str:
        return self._rig.name

    def extra_repr(self) -> str:
        msg = [
            f"Name: {self.name()}",
            f"Pose representation: {self.pose_repr}",
            f"Number of joints: {self.NUM_JOINTS}",
            f"Root joint: {self.root_index} ({self.joint_names[self.root_index]})",
            f"Batch size: {self.batch_size}",
        ]
        return "\n".join(msg)

    def joint_index(self, name: str) -> int:
        return self.joint_name_to_index[name]

    def get_body_node(self, body: int | str) -> RigidBodyNode:
        body_index = self.joint_index(body) if isinstance(body, str) else body
        return self.body_nodes[body_index]

    def get_joint(self, joint: int | str) -> ArticulatedJoint:
        joint_index = self.joint_index(joint) if isinstance(joint, str) else joint
        return self.joints[joint_index]

    def freeze(self, *names: str) -> None:
        for name in names:
            if not hasattr(self, name):
                continue
            value = getattr(self, name)
            if isinstance(value, nn.Parameter):
                value.requires_grad_(False)

    def unfreeze(self, *names: str) -> None:
        for name in names:
            if not hasattr(self, name):
                continue
            value = getattr(self, name)
            if isinstance(value, nn.Parameter):
                value.requires_grad_(True)

    @torch.no_grad()
    def reset_params(self, **params_dict: torch.Tensor) -> None:
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                value = torch.as_tensor(
                    params_dict[param_name], dtype=param.dtype, device=param.device
                )
            else:
                value = self._default_parameter_like(param_name, param)
            param.copy_(self._coerce_parameter_value(param_name, value, param))

    def rest_joints(self, scales: torch.Tensor | None = None) -> torch.Tensor:
        scales = self._canonicalize_scales(
            scales, dtype=self.rest_offsets.dtype, device=self.rest_offsets.device
        )
        scaled_offsets = self._scaled_offsets(
            scales,
            batch_shape=scales.shape[:-2],
            dtype=self.rest_offsets.dtype,
            device=self.rest_offsets.device,
        )
        joints = torch.zeros(
            scaled_offsets.shape,
            dtype=scaled_offsets.dtype,
            device=scaled_offsets.device,
        )
        for body_idx in self.topological_order:
            body = self.body_nodes[body_idx]
            incoming_joint = self.joints[body.incoming_joint_index]
            if incoming_joint.parent_body_index is None:
                continue
            joints[..., body_idx, :] = (
                joints[..., incoming_joint.parent_body_index, :]
                + scaled_offsets[..., body_idx, :]
            )
        return joints

    def forward_shape(
        self,
        scales: torch.Tensor | None = None,
    ) -> ModelOutput | dict[str, Any]:
        resolved_scales = self._resolve_state("scales", scales)
        batch_shape = self._infer_batch_shape(
            pose_repr=self.pose_repr, scales=resolved_scales
        )
        if resolved_scales is None:
            resolved_scales = self._default_scales(
                batch_shape,
                dtype=self.rest_offsets.dtype,
                device=self.rest_offsets.device,
            )
        joints = self.rest_joints(scales=resolved_scales)
        output = ModelOutput(
            joints=joints,
            scales=self._canonicalize_scales(
                resolved_scales,
                dtype=self.rest_offsets.dtype,
                device=self.rest_offsets.device,
            ),
            joint_names=self.joint_names,
        )
        return output

    def _register_optional_state(
        self,
        *,
        name: str,
        value: torch.Tensor | None,
        default: torch.Tensor,
        create: bool,
    ) -> None:
        if not create:
            return
        tensor = default if value is None else value.detach().clone()
        tensor = tensor.to(dtype=self.rest_offsets.dtype)
        self.register_parameter(name, nn.Parameter(tensor))

    def _reshape_global_orient(
        self, value: torch.Tensor, pose_repr: str
    ) -> torch.Tensor:
        if pose_repr == "rotmat":
            if value.shape[-2:] == (3, 3):
                if value.ndim == 2:
                    return value.unsqueeze(0)
                if value.shape[-3] == 1:
                    return value
                return value.unsqueeze(-3)
            raise ValueError(
                f"global_orient must have shape [3,3] or [...,1,3,3], got {tuple(value.shape)}."
            )

        feat = pose_repr_size(pose_repr)
        if value.ndim == 1 and value.shape[0] == feat:
            return value.view(1, feat)
        if value.shape[-1] == feat and value.ndim >= 2 and value.shape[-2] == 1:
            return value
        if value.shape[-1] == feat:
            return value.unsqueeze(-2)
        raise ValueError(
            f"global_orient must end with {feat}, got {tuple(value.shape)}."
        )

    def _reshape_joint_pose(
        self, value: torch.Tensor, num_joints: int, pose_repr: str
    ) -> torch.Tensor:
        if pose_repr == "rotmat":
            if value.shape[-2:] != (3, 3):
                raise ValueError(
                    f"Rotation-matrix pose must end in (3,3), got {tuple(value.shape)}."
                )
            if value.shape[-3] != num_joints:
                raise ValueError(
                    f"Expected {num_joints} joints, got {value.shape[-3]}."
                )
            return value

        feat = pose_repr_size(pose_repr)
        if value.ndim == 2 and value.shape == (num_joints, feat):
            return value
        if (
            value.shape[-1] == feat
            and value.ndim >= 2
            and value.shape[-2] == num_joints
        ):
            return value
        if value.shape[-1] == num_joints * feat:
            return value.reshape(value.shape[:-1] + (num_joints, feat))
        if value.ndim == 1 and value.shape[0] == num_joints * feat:
            return value.reshape(num_joints, feat)
        raise ValueError(
            f"Pose must have shape [...,{num_joints},{feat}] or [...,{num_joints * feat}], got {tuple(value.shape)}."
        )

    def _reshape_translation(self, transl: torch.Tensor) -> torch.Tensor:
        if transl.ndim == 1 and transl.shape[0] == 3:
            return transl
        if transl.shape[-1] != 3:
            raise ValueError(f"transl must end in 3, got {tuple(transl.shape)}.")
        return transl

    def _default_global_orient(
        self,
        batch_shape: tuple[int, ...],
        *,
        pose_repr: str,
        dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        ident = identity_pose(1, pose_repr, dtype=dtype, device=device).squeeze(-2)
        if not batch_shape:
            return ident
        return ident.expand(batch_shape + ident.shape).clone()

    def _default_body_pose(
        self,
        batch_shape: tuple[int, ...],
        *,
        pose_repr: str,
        dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        body_pose = identity_pose(
            self.NUM_BODY_JOINTS, pose_repr, dtype=dtype, device=device
        )
        tail_dims = 3 if pose_repr == "rotmat" else 2
        if not batch_shape:
            return body_pose
        return _expand_to_batch(body_pose, batch_shape, tail_dims=tail_dims).clone()

    def _default_translation(
        self,
        batch_shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        return torch.zeros(batch_shape + (3,), dtype=dtype, device=device)

    def _default_scales(
        self,
        batch_shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        return torch.ones(
            batch_shape + (self.NUM_JOINTS, 3), dtype=dtype, device=device
        )

    def _default_parameter_like(self, name: str, param: torch.Tensor) -> torch.Tensor:
        if name == "global_orient":
            return self._default_global_orient(
                param.shape[:-1] if self.pose_repr != "rotmat" else param.shape[:-2],
                pose_repr=self.pose_repr,
                dtype=param.dtype,
                device=param.device,
            )
        if name == "body_pose":
            batch_shape = (
                param.shape[:-2] if self.pose_repr != "rotmat" else param.shape[:-3]
            )
            return self._default_body_pose(
                batch_shape,
                pose_repr=self.pose_repr,
                dtype=param.dtype,
                device=param.device,
            )
        if name == "transl":
            return self._default_translation(
                param.shape[:-1], dtype=param.dtype, device=param.device
            )
        if name == "scales":
            return self._default_scales(
                param.shape[:-2], dtype=param.dtype, device=param.device
            )
        raise KeyError(f"Unsupported parameter name: {name}")

    def _coerce_parameter_value(
        self, name: str, value: torch.Tensor, param: torch.Tensor
    ) -> torch.Tensor:
        value = value.to(dtype=param.dtype, device=param.device)
        if value.shape == param.shape:
            return value
        if value.shape == param.shape[len(self._parameter_batch_shape) :]:
            return value.expand(param.shape).clone()
        raise ValueError(
            f"Parameter {name} must have shape {tuple(param.shape)} or "
            f"{tuple(param.shape[len(self._parameter_batch_shape) :])}, got {tuple(value.shape)}."
        )

    def _infer_batch_shape(
        self,
        *,
        pose_repr: str,
        global_orient: torch.Tensor | None = None,
        body_pose: torch.Tensor | None = None,
        full_pose: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
        transl: torch.Tensor | None = None,
    ) -> tuple[int, ...]:
        shapes: list[tuple[int, ...]] = []
        if global_orient is not None:
            reshaped_global = self._reshape_global_orient(global_orient, pose_repr)
            shapes.append(
                reshaped_global.shape[:-3]
                if pose_repr == "rotmat"
                else reshaped_global.shape[:-2]
            )
        if body_pose is not None:
            reshaped_body = self._reshape_joint_pose(
                body_pose, self.NUM_BODY_JOINTS, pose_repr
            )
            shapes.append(
                reshaped_body.shape[:-3]
                if pose_repr == "rotmat"
                else reshaped_body.shape[:-2]
            )
        if full_pose is not None:
            reshaped_full = self._reshape_joint_pose(
                full_pose, self.NUM_JOINTS, pose_repr
            )
            shapes.append(
                reshaped_full.shape[:-3]
                if pose_repr == "rotmat"
                else reshaped_full.shape[:-2]
            )
        if scales is not None:
            canonical_scales = self._canonicalize_scales(
                scales,
                dtype=self.rest_offsets.dtype,
                device=self.rest_offsets.device,
            )
            shapes.append(canonical_scales.shape[:-2])
        if transl is not None:
            shapes.append(self._reshape_translation(transl).shape[:-1])
        if shapes:
            return _broadcast_batch_shape(*shapes)
        return (self.batch_size,)

    def _canonicalize_scales(
        self,
        scales: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if scales is None:
            return torch.ones(self.NUM_JOINTS, 3, dtype=dtype, device=device)

        value = scales.to(dtype=dtype, device=device)
        if value.ndim >= 2 and value.shape[-2:] == (self.NUM_JOINTS, 3):
            return value.clone()
        if value.ndim >= 2 and value.shape[-2:] == (self.NUM_JOINTS - 1, 3):
            full = torch.ones(
                value.shape[:-2] + (self.NUM_JOINTS, 3), dtype=dtype, device=device
            )
            full[..., list(self.non_root_joint_indices), :] = value
            return full
        if value.ndim == 1 and value.shape[0] == self.NUM_JOINTS - 1:
            full = torch.ones(self.NUM_JOINTS, 3, dtype=dtype, device=device)
            full[list(self.non_root_joint_indices), :] = value.unsqueeze(-1).expand(
                self.NUM_JOINTS - 1, 3
            )
            return full
        if value.shape[-1] == self.NUM_JOINTS - 1:
            full = torch.ones(
                value.shape[:-1] + (self.NUM_JOINTS, 3), dtype=dtype, device=device
            )
            full[..., list(self.non_root_joint_indices), :] = value.unsqueeze(
                -1
            ).expand(value.shape + (3,))
            return full
        if value.ndim == 1 and value.shape[0] == self.NUM_JOINTS:
            return value.unsqueeze(-1).expand(self.NUM_JOINTS, 3).clone()
        if value.shape[-1] == self.NUM_JOINTS:
            return value.unsqueeze(-1).expand(value.shape + (3,)).clone()
        raise ValueError(
            "scales must have shape [..., J], [..., J - 1], [..., J, 3], or [..., J - 1, 3] "
            f"with J={self.NUM_JOINTS}, got {tuple(value.shape)}."
        )

    def _scaled_offsets(
        self,
        scales: torch.Tensor,
        *,
        batch_shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        offsets = self.rest_offsets.to(dtype=dtype, device=device)
        scales = scales.to(dtype=dtype, device=device)
        offsets = _expand_to_batch(offsets, batch_shape, tail_dims=2)
        scales = _expand_to_batch(scales, batch_shape, tail_dims=2)
        scale_owner_indices = self.offset_scale_body_indices.to(device=device)
        # OpenSim-style scaling applies joint locations in the parent body's local frame.
        owner_scales = torch.index_select(scales, dim=-2, index=scale_owner_indices)
        scaled = offsets * owner_scales
        scaled[..., self.root_index, :] = 0.0
        return scaled

    def _resolve_state(
        self, name: str, value: torch.Tensor | None
    ) -> torch.Tensor | None:
        if value is not None:
            return value
        if hasattr(self, name):
            return getattr(self, name)
        return None

    def _compose_full_pose(
        self,
        *,
        global_orient: torch.Tensor,
        body_pose: torch.Tensor,
        pose_repr: str,
    ) -> torch.Tensor:
        global_orient = self._reshape_global_orient(global_orient, pose_repr)
        body_pose = self._reshape_joint_pose(body_pose, self.NUM_JOINTS - 1, pose_repr)

        if pose_repr == "rotmat":
            global_batch = global_orient.shape[:-3]
            body_batch = body_pose.shape[:-3]
            batch_shape = _broadcast_batch_shape(global_batch, body_batch)
            global_orient = _expand_to_batch(global_orient, batch_shape, tail_dims=3)
            body_pose = _expand_to_batch(body_pose, batch_shape, tail_dims=3)
            full = identity_pose(
                self.NUM_JOINTS,
                "rotmat",
                dtype=global_orient.dtype,
                device=global_orient.device,
            )
            full = _expand_to_batch(full, batch_shape, tail_dims=3).clone()
            full[..., self.root_index, :, :] = global_orient[..., 0, :, :]
            full[..., list(self.non_root_joint_indices), :, :] = body_pose
            return full

        global_batch = global_orient.shape[:-2]
        body_batch = body_pose.shape[:-2]
        batch_shape = _broadcast_batch_shape(global_batch, body_batch)
        global_orient = _expand_to_batch(global_orient, batch_shape, tail_dims=2)
        body_pose = _expand_to_batch(body_pose, batch_shape, tail_dims=2)
        full = identity_pose(
            self.NUM_JOINTS,
            pose_repr,
            dtype=global_orient.dtype,
            device=global_orient.device,
        )
        full = _expand_to_batch(full, batch_shape, tail_dims=2).clone()
        full[..., self.root_index, :] = global_orient[..., 0, :]
        full[..., list(self.non_root_joint_indices), :] = body_pose
        return full

    def _split_full_pose(
        self, full_pose: torch.Tensor, pose_repr: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        canonical = self._reshape_joint_pose(full_pose, self.NUM_JOINTS, pose_repr)
        if pose_repr == "rotmat":
            return (
                canonical[..., self.root_index, :, :],
                canonical[..., list(self.non_root_joint_indices), :, :],
            )
        return (
            canonical[..., self.root_index, :],
            canonical[..., list(self.non_root_joint_indices), :],
        )

    def _full_pose_to_local_rotmats(
        self, full_pose: torch.Tensor, pose_repr: str
    ) -> torch.Tensor:
        canonical = self._reshape_joint_pose(full_pose, self.NUM_JOINTS, pose_repr)
        return to_rotation_matrix(canonical, pose_repr)

    def fk(
        self,
        local_rotations: torch.Tensor,
        *,
        scales: torch.Tensor | None = None,
        transl: torch.Tensor | None = None,
        return_local_transforms: bool = False,
        return_global_transforms: bool = False,
    ) -> dict[str, torch.Tensor]:
        if local_rotations.shape[-3:] != (self.NUM_JOINTS, 3, 3):
            raise ValueError(
                f"local_rotations must have shape [...,{self.NUM_JOINTS},3,3], got {tuple(local_rotations.shape)}."
            )

        batch_shape = local_rotations.shape[:-3]
        dtype = local_rotations.dtype
        device = local_rotations.device

        scales_full = self._canonicalize_scales(scales, dtype=dtype, device=device)
        if scales_full.shape[:-2] != batch_shape:
            scales_full = scales_full.expand(batch_shape + (self.NUM_JOINTS, 3))
        scaled_offsets = self._scaled_offsets(
            scales_full, batch_shape=batch_shape, dtype=dtype, device=device
        )

        if transl is None:
            transl_value = torch.zeros(batch_shape + (3,), dtype=dtype, device=device)
        else:
            transl_value = self._reshape_translation(transl).to(
                dtype=dtype, device=device
            )
            transl_value = _expand_to_batch(transl_value, batch_shape, tail_dims=1)

        joint_positions: list[torch.Tensor | None] = [None] * self.NUM_JOINTS
        global_rotation_list: list[torch.Tensor | None] = [None] * self.NUM_JOINTS
        local_transform_list: list[torch.Tensor | None] | None = (
            [None] * self.NUM_JOINTS if return_local_transforms else None
        )
        global_transform_list: list[torch.Tensor | None] | None = (
            [None] * self.NUM_JOINTS if return_global_transforms else None
        )

        for body_idx in self.topological_order:
            body = self.body_nodes[body_idx]
            joint = self.joints[body.incoming_joint_index]
            local_rotation = local_rotations[..., joint.pose_index, :, :]

            if joint.parent_body_index is None:
                local_translation = transl_value
                position = transl_value
                global_rotation = local_rotation
            else:
                parent_rot = global_rotation_list[joint.parent_body_index]
                parent_pos = joint_positions[joint.parent_body_index]
                assert parent_rot is not None and parent_pos is not None
                local_translation = scaled_offsets[..., joint.child_body_index, :]
                offset_world = (parent_rot @ local_translation.unsqueeze(-1)).squeeze(
                    -1
                )
                position = parent_pos + offset_world
                global_rotation = parent_rot @ local_rotation

            joint_positions[body_idx] = position
            global_rotation_list[body_idx] = global_rotation

            if local_transform_list is not None:
                local_transform_list[body_idx] = _make_transform(
                    local_rotation, local_translation
                )
            if global_transform_list is not None:
                global_transform_list[body_idx] = _make_transform(
                    global_rotation, position
                )

        joints = torch.stack(
            [value for value in joint_positions if value is not None], dim=-2
        )
        global_rotations = torch.stack(
            [value for value in global_rotation_list if value is not None], dim=-3
        )

        output = {
            "joints": joints,
            "global_rotations": global_rotations,
            "scaled_offsets": scaled_offsets,
            "scales": scales_full,
            "transl": transl_value,
        }
        if local_transform_list is not None:
            output["local_transforms"] = torch.stack(
                [value for value in local_transform_list if value is not None], dim=-3
            )
        if global_transform_list is not None:
            output["global_transforms"] = torch.stack(
                [value for value in global_transform_list if value is not None], dim=-3
            )
        return output

    def forward(
        self,
        *,
        global_orient: torch.Tensor | None = None,
        body_pose: torch.Tensor | None = None,
        full_pose: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
        transl: torch.Tensor | None = None,
        pose_repr: str | None = None,
        return_full_pose: bool = False,
        return_local_rotations: bool = False,
        return_global_rotations: bool = False,
        return_local_transforms: bool = False,
        return_global_transforms: bool = False,
        return_scaled_offsets: bool = False,
        pose2rot: bool | None = None,
    ) -> ModelOutput | dict[str, Any]:
        if pose2rot is False:
            if pose_repr is not None and normalize_pose_repr(pose_repr) != "rotmat":
                raise ValueError("pose2rot=False requires pose_repr='rotmat'.")
            pose_repr = "rotmat"
        pose_repr = normalize_pose_repr(pose_repr or self.pose_repr)

        resolved_scales = self._resolve_state("scales", scales)
        resolved_transl = self._resolve_state("transl", transl)

        resolved_global_orient = self._resolve_state("global_orient", global_orient)
        resolved_body_pose = self._resolve_state("body_pose", body_pose)
        batch_shape = self._infer_batch_shape(
            pose_repr=pose_repr,
            global_orient=resolved_global_orient,
            body_pose=resolved_body_pose,
            full_pose=full_pose,
            scales=resolved_scales,
            transl=resolved_transl,
        )

        pose_input: torch.Tensor
        if full_pose is not None:
            pose_input = self._reshape_joint_pose(full_pose, self.NUM_JOINTS, pose_repr)
        else:
            if resolved_global_orient is None:
                resolved_global_orient = self._default_global_orient(
                    batch_shape,
                    pose_repr=pose_repr,
                    dtype=self.rest_offsets.dtype,
                    device=self.rest_offsets.device,
                )
            if resolved_body_pose is None:
                resolved_body_pose = self._default_body_pose(
                    batch_shape,
                    pose_repr=pose_repr,
                    dtype=self.rest_offsets.dtype,
                    device=self.rest_offsets.device,
                )
            pose_input = self._compose_full_pose(
                global_orient=resolved_global_orient,
                body_pose=resolved_body_pose,
                pose_repr=pose_repr,
            )
        pose_input = _expand_to_batch(
            pose_input, batch_shape, tail_dims=3 if pose_repr == "rotmat" else 2
        )
        output_global_orient, output_body_pose = self._split_full_pose(
            pose_input, pose_repr
        )

        local_rotations = self._full_pose_to_local_rotmats(pose_input, pose_repr)

        fk_output = self.fk(
            local_rotations,
            scales=resolved_scales,
            transl=resolved_transl,
            return_local_transforms=return_local_transforms,
            return_global_transforms=return_global_transforms,
        )
        joints = fk_output["joints"]
        scales = fk_output["scales"]

        global_orient = None

        # if apply_trans:
        #     joints += transl.unsqueeze(dim=1)
        #     vertices += transl.unsqueeze(dim=1)

        output = ModelOutput(
            global_orient=output_global_orient,
            body_pose=output_body_pose,
            joints=joints,
            scales=fk_output["scales"],
            full_pose=pose_input if return_full_pose else None,
            transl=fk_output["transl"],
            scaled_offsets=fk_output["scaled_offsets"]
            if return_scaled_offsets
            else None,
            local_rotations=local_rotations if return_local_rotations else None,
            global_rotations=fk_output["global_rotations"]
            if return_global_rotations
            else None,
            local_transforms=fk_output.get("local_transforms"),
            global_transforms=fk_output.get("global_transforms"),
            joint_names=self.joint_names,
        )
        return output


class SkeletalModelLayer(SkeletalModel):
    def __init__(self, *args, **kwargs) -> None:
        kwargs = dict(kwargs)
        kwargs.setdefault("create_global_orient", False)
        kwargs.setdefault("create_body_pose", False)
        kwargs.setdefault("create_transl", False)
        kwargs.setdefault("create_scales", False)
        super().__init__(*args, **kwargs)
