from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .output import SkeletalOutput
from .rotations import (
    identity_pose,
    normalize_pose_repr,
    pose_repr_size,
    to_rotation_matrix,
)
from .specs import SkeletonSpec


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
        create_body_pose: bool = True,
        create_bone_scales: bool = False,
        create_transl: bool = False,
        global_orient: torch.Tensor | None = None,
        body_pose: torch.Tensor | None = None,
        bone_scales: torch.Tensor | None = None,
        transl: torch.Tensor | None = None,
        learn_global_orient: bool = True,
        learn_body_pose: bool = True,
        learn_bone_scales: bool = True,
        learn_transl: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.pose_repr = normalize_pose_repr(pose_repr)
        self.spec = spec
        self.skeleton = ArticulatedSkeleton.from_spec(spec)
        self.body_nodes = self.skeleton.body_nodes
        self.joints = self.skeleton.joints
        self.joint_names = tuple(spec.joint_names)
        self.parents = tuple(
            -1 if body.parent_index is None else body.parent_index
            for body in self.body_nodes
        )
        self.child_body_indices = tuple(
            body.child_body_indices for body in self.body_nodes
        )
        self.root_index = int(spec.root_index)
        self.num_joints = len(self.joint_names)
        self.num_bodies = len(self.body_nodes)
        self.non_root_joint_indices = tuple(
            idx for idx in range(self.num_joints) if idx != self.root_index
        )
        self.joint_name_to_index = {
            name: idx for idx, name in enumerate(self.joint_names)
        }
        self.topological_order = self.skeleton.topological_order
        offset_scale_body_indices = [self.root_index] * self.num_joints
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
            default=identity_pose(1, self.pose_repr, dtype=dtype).squeeze(0),
            create=create_global_orient,
            learn=learn_global_orient,
        )
        self._register_optional_state(
            name="body_pose",
            value=body_pose,
            default=identity_pose(self.num_joints - 1, self.pose_repr, dtype=dtype),
            create=create_body_pose,
            learn=learn_body_pose,
        )
        self._register_optional_state(
            name="bone_scales",
            value=bone_scales,
            default=torch.ones(self.num_joints, 3, dtype=dtype),
            create=create_bone_scales,
            learn=learn_bone_scales,
        )
        self._register_optional_state(
            name="transl",
            value=transl,
            default=torch.zeros(3, dtype=dtype),
            create=create_transl,
            learn=learn_transl,
        )

    def _register_optional_state(
        self,
        *,
        name: str,
        value: torch.Tensor | None,
        default: torch.Tensor,
        create: bool,
        learn: bool,
    ) -> None:
        if not create:
            return
        tensor = default if value is None else value.detach().clone()
        tensor = tensor.to(dtype=self.rest_offsets.dtype)
        if learn:
            setattr(self, name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

    def joint_index(self, name: str) -> int:
        return self.joint_name_to_index[name]

    def get_body_node(self, body: int | str) -> RigidBodyNode:
        body_index = self.joint_index(body) if isinstance(body, str) else body
        return self.body_nodes[body_index]

    def get_joint(self, joint: int | str) -> ArticulatedJoint:
        joint_index = self.joint_index(joint) if isinstance(joint, str) else joint
        return self.joints[joint_index]

    def has_state(self, name: str) -> bool:
        return hasattr(self, name)

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

    def rest_joints(self, bone_scales: torch.Tensor | None = None) -> torch.Tensor:
        scales = self._canonicalize_bone_scales(
            bone_scales, dtype=self.rest_offsets.dtype, device=self.rest_offsets.device
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

    def _canonicalize_bone_scales(
        self,
        bone_scales: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if bone_scales is None:
            return torch.ones(self.num_joints, 3, dtype=dtype, device=device)

        value = bone_scales.to(dtype=dtype, device=device)
        if value.ndim >= 2 and value.shape[-2:] == (self.num_joints, 3):
            return value.clone()
        if value.ndim >= 2 and value.shape[-2:] == (self.num_joints - 1, 3):
            full = torch.ones(
                value.shape[:-2] + (self.num_joints, 3), dtype=dtype, device=device
            )
            full[..., list(self.non_root_joint_indices), :] = value
            return full
        if value.ndim == 1 and value.shape[0] == self.num_joints - 1:
            full = torch.ones(self.num_joints, 3, dtype=dtype, device=device)
            full[list(self.non_root_joint_indices), :] = value.unsqueeze(-1).expand(
                self.num_joints - 1, 3
            )
            return full
        if value.shape[-1] == self.num_joints - 1:
            full = torch.ones(
                value.shape[:-1] + (self.num_joints, 3), dtype=dtype, device=device
            )
            full[..., list(self.non_root_joint_indices), :] = value.unsqueeze(
                -1
            ).expand(value.shape + (3,))
            return full
        if value.ndim == 1 and value.shape[0] == self.num_joints:
            return value.unsqueeze(-1).expand(self.num_joints, 3).clone()
        if value.shape[-1] == self.num_joints:
            return value.unsqueeze(-1).expand(value.shape + (3,)).clone()
        raise ValueError(
            "bone_scales must have shape [..., J], [..., J - 1], [..., J, 3], or [..., J - 1, 3] "
            f"with J={self.num_joints}, got {tuple(value.shape)}."
        )

    def _scaled_offsets(
        self,
        bone_scales: torch.Tensor,
        *,
        batch_shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        offsets = self.rest_offsets.to(dtype=dtype, device=device)
        scales = bone_scales.to(dtype=dtype, device=device)
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
        body_pose = self._reshape_joint_pose(body_pose, self.num_joints - 1, pose_repr)

        if pose_repr == "rotmat":
            global_batch = global_orient.shape[:-3]
            body_batch = body_pose.shape[:-3]
            batch_shape = _broadcast_batch_shape(global_batch, body_batch)
            global_orient = _expand_to_batch(global_orient, batch_shape, tail_dims=3)
            body_pose = _expand_to_batch(body_pose, batch_shape, tail_dims=3)
            full = identity_pose(
                self.num_joints,
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
        feat = global_orient.shape[-1]
        full = identity_pose(
            self.num_joints,
            pose_repr,
            dtype=global_orient.dtype,
            device=global_orient.device,
        )
        full = _expand_to_batch(full, batch_shape, tail_dims=2).clone()
        full[..., self.root_index, :] = global_orient[..., 0, :]
        full[..., list(self.non_root_joint_indices), :] = body_pose
        return full

    def _full_pose_to_local_rotmats(
        self, full_pose: torch.Tensor, pose_repr: str
    ) -> torch.Tensor:
        canonical = self._reshape_joint_pose(full_pose, self.num_joints, pose_repr)
        return to_rotation_matrix(canonical, pose_repr)

    def forward_kinematics(
        self,
        local_rotations: torch.Tensor,
        *,
        bone_scales: torch.Tensor | None = None,
        transl: torch.Tensor | None = None,
        return_local_transforms: bool = False,
        return_global_transforms: bool = False,
    ) -> dict[str, torch.Tensor]:
        if local_rotations.shape[-3:] != (self.num_joints, 3, 3):
            raise ValueError(
                f"local_rotations must have shape [...,{self.num_joints},3,3], got {tuple(local_rotations.shape)}."
            )

        batch_shape = local_rotations.shape[:-3]
        dtype = local_rotations.dtype
        device = local_rotations.device

        bone_scales_full = self._canonicalize_bone_scales(
            bone_scales, dtype=dtype, device=device
        )
        if bone_scales_full.shape[:-2] != batch_shape:
            bone_scales_full = bone_scales_full.expand(
                batch_shape + (self.num_joints, 3)
            )
        scaled_offsets = self._scaled_offsets(
            bone_scales_full, batch_shape=batch_shape, dtype=dtype, device=device
        )

        if transl is None:
            transl_value = torch.zeros(batch_shape + (3,), dtype=dtype, device=device)
        else:
            transl_value = self._reshape_translation(transl).to(
                dtype=dtype, device=device
            )
            transl_value = _expand_to_batch(transl_value, batch_shape, tail_dims=1)

        joint_positions: list[torch.Tensor | None] = [None] * self.num_joints
        global_rotation_list: list[torch.Tensor | None] = [None] * self.num_joints
        local_transform_list: list[torch.Tensor | None] | None = (
            [None] * self.num_joints if return_local_transforms else None
        )
        global_transform_list: list[torch.Tensor | None] | None = (
            [None] * self.num_joints if return_global_transforms else None
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
            "bone_scales": bone_scales_full,
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
        bone_scales: torch.Tensor | None = None,
        transl: torch.Tensor | None = None,
        pose_repr: str | None = None,
        return_full_pose: bool = False,
        return_local_rotations: bool = False,
        return_global_rotations: bool = False,
        return_local_transforms: bool = False,
        return_global_transforms: bool = False,
        return_scaled_offsets: bool = False,
        return_dict: bool = False,
    ) -> SkeletalOutput | dict[str, Any]:
        pose_repr = normalize_pose_repr(pose_repr or self.pose_repr)

        resolved_bone_scales = self._resolve_state("bone_scales", bone_scales)
        resolved_transl = self._resolve_state("transl", transl)

        resolved_global_orient = self._resolve_state("global_orient", global_orient)
        resolved_body_pose = self._resolve_state("body_pose", body_pose)

        pose_input: torch.Tensor
        if full_pose is not None:
            pose_input = self._reshape_joint_pose(full_pose, self.num_joints, pose_repr)
        else:
            if resolved_global_orient is None or resolved_body_pose is None:
                raise ValueError(
                    "Either full_pose must be provided, or both global_orient and body_pose must be available "
                    "from the call arguments or registered model state."
                )
            pose_input = self._compose_full_pose(
                global_orient=resolved_global_orient,
                body_pose=resolved_body_pose,
                pose_repr=pose_repr,
            )

        local_rotations = self._full_pose_to_local_rotmats(pose_input, pose_repr)

        fk_output = self.forward_kinematics(
            local_rotations,
            bone_scales=resolved_bone_scales,
            transl=resolved_transl,
            return_local_transforms=return_local_transforms,
            return_global_transforms=return_global_transforms,
        )

        model_output = SkeletalOutput(
            joints=fk_output["joints"],
            full_pose=pose_input if return_full_pose else None,
            global_orient=self._resolve_state("global_orient", global_orient),
            body_pose=self._resolve_state("body_pose", body_pose),
            bone_scales=fk_output["bone_scales"],
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
        return model_output.asdict() if return_dict else model_output
