from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn

from .rigs._spec import SkeletonSpec
from .rotations import (
    identity_pose,
    matrix_to_axis_angle,
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
                joint_type=spec.joint_type(body_idx),
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
        self.links = list(spec.links)
        self.markers = list(spec.markers)
        self.contacts = list(spec.contacts)
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
        self.marker_names = tuple(marker.name for marker in self.spec.markers)
        self.contact_names = tuple(contact.name for contact in self.spec.contacts)
        self.marker_name_to_index = {
            name: idx for idx, name in enumerate(self.marker_names)
        }
        self.contact_name_to_index = {
            name: idx for idx, name in enumerate(self.contact_names)
        }
        self._q_joint_indices = tuple(
            idx for idx in self.non_root_joint_indices if self.spec.joint_dof(idx) > 0
        )
        self._q_slices: dict[str, slice] = {}
        q_cursor = 0
        for joint_idx in self._q_joint_indices:
            dof = self.spec.joint_dof(joint_idx)
            self._q_slices[self.joint_names[joint_idx]] = slice(
                q_cursor, q_cursor + dof
            )
            q_cursor += dof
        self.num_dofs = q_cursor
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
        self.register_buffer(
            "_link_mass",
            self._link_mass_tensor(dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "_link_com",
            self._link_com_tensor(dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "_marker_local",
            self._marker_local_tensor(dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "_contact_local",
            self._contact_local_tensor(dtype=dtype),
            persistent=False,
        )

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

    def marker_index(self, name: str) -> int:
        return self.marker_name_to_index[name]

    def contact_index(self, name: str) -> int:
        return self.contact_name_to_index[name]

    def dof_slice(self, joint_name: str) -> slice:
        return self._q_slices[joint_name]

    def zero_pose(
        self,
        batch_size: int | tuple[int, ...] = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Create zero non-root rotational coordinates.

        Parameters
        ----------
        batch_size : int or tuple of int
            Leading batch dimensions.
        device : torch.device or str, optional
            Defaults to the model device.
        dtype : torch.dtype, optional
            Defaults to the model dtype.

        Returns
        -------
        torch.Tensor
            Coordinates with shape [*batch_size, num_dofs].
        """
        batch_shape = (
            (batch_size,) if isinstance(batch_size, int) else tuple(batch_size)
        )
        dtype = dtype if dtype is not None else self.rest_offsets.dtype
        device = (
            torch.device(device) if device is not None else self.rest_offsets.device
        )
        return torch.zeros(batch_shape + (self.num_dofs,), dtype=dtype, device=device)

    def identity_root(
        self,
        batch_size: int | tuple[int, ...] = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Create identity world-placement transforms.

        Parameters
        ----------
        batch_size : int or tuple of int
            Leading batch dimensions.
        device : torch.device or str, optional
            Defaults to the model device.
        dtype : torch.dtype, optional
            Defaults to the model dtype.

        Returns
        -------
        torch.Tensor
            Homogeneous transforms with shape [*batch_size, 4, 4].
        """
        batch_shape = (
            (batch_size,) if isinstance(batch_size, int) else tuple(batch_size)
        )
        dtype = dtype if dtype is not None else self.rest_offsets.dtype
        device = (
            torch.device(device) if device is not None else self.rest_offsets.device
        )
        eye = torch.eye(4, dtype=dtype, device=device)
        return eye.expand(batch_shape + (4, 4)).clone()

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

    def _link_mass_tensor(self, *, dtype: torch.dtype) -> torch.Tensor:
        if self.spec.links:
            masses = [link.mass for link in self.spec.links]
        else:
            masses = [1.0 for _ in self.joint_names]
        return torch.tensor(masses, dtype=dtype)

    def _link_com_tensor(self, *, dtype: torch.dtype) -> torch.Tensor:
        if self.spec.links:
            com = [link.com for link in self.spec.links]
        else:
            com = [(0.0, 0.0, 0.0) for _ in self.joint_names]
        return torch.tensor(com, dtype=dtype)

    def _marker_local_tensor(self, *, dtype: torch.dtype) -> torch.Tensor:
        if not self.spec.markers:
            return torch.empty(0, 3, dtype=dtype)
        return torch.tensor(
            [marker.local_xyz for marker in self.spec.markers], dtype=dtype
        )

    def _contact_local_tensor(self, *, dtype: torch.dtype) -> torch.Tensor:
        if not self.spec.contacts:
            return torch.empty(0, 3, dtype=dtype)
        return torch.tensor(
            [contact.local_xyz for contact in self.spec.contacts], dtype=dtype
        )

    def _q_to_body_pose(self, q: torch.Tensor) -> torch.Tensor:
        if q.shape[-1] != self.num_dofs:
            raise ValueError(
                f"Expected q.shape[-1] == {self.num_dofs}, got {q.shape[-1]}."
            )
        body_pose = self._default_body_pose(
            q.shape[:-1], pose_repr="axis_angle", dtype=q.dtype, device=q.device
        )
        for joint_idx in self._q_joint_indices:
            body_idx = self.non_root_joint_indices.index(joint_idx)
            q_slice = self.dof_slice(self.joint_names[joint_idx])
            if self.spec.joint_dof(joint_idx) == 1:
                axis = (self.spec.joint_axes or (None,) * self.NUM_JOINTS)[
                    joint_idx
                ] or (1.0, 0.0, 0.0)
                axis_tensor = torch.tensor(axis, dtype=q.dtype, device=q.device)
                axis_tensor = axis_tensor / axis_tensor.norm()
                body_pose[..., body_idx, :] = q[..., q_slice] * axis_tensor
            else:
                body_pose[..., body_idx, :] = q[..., q_slice]
        return body_pose

    def _root_to_orient_transl(
        self, root: torch.Tensor | None, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if root is None:
            global_orient = torch.zeros(
                q.shape[:-1] + (3,), dtype=q.dtype, device=q.device
            )
            transl = torch.zeros(q.shape[:-1] + (3,), dtype=q.dtype, device=q.device)
            return global_orient, transl
        if root.shape[-2:] != (4, 4):
            raise ValueError(
                f"root must have shape [*, 4, 4], got {tuple(root.shape)}."
            )
        return matrix_to_axis_angle(root[..., :3, :3].to(q)), root[..., :3, 3].to(q)

    def forward_kinematics(
        self,
        q: torch.Tensor,
        root: torch.Tensor | None = None,
        *,
        scales: torch.Tensor | None = None,
    ) -> ModelOutput:
        """Evaluate generalized coordinates using this rig.

        Parameters
        ----------
        q : torch.Tensor
            Non-root rotational DOFs in radians, shape [..., D].
        root : torch.Tensor, optional
            World placement transforms, shape [..., 4, 4].
        scales : torch.Tensor, optional
            Per-body XYZ scales; defaults to model state.

        Returns
        -------
        ModelOutput
            Joints, rotations, transforms, markers, and contacts.

        Raises
        ------
        ValueError
            If pose or transform shapes are invalid.
        """
        global_orient, transl = self._root_to_orient_transl(root, q)
        output = self(
            global_orient=global_orient,
            body_pose=self._q_to_body_pose(q),
            transl=transl,
            pose_repr="axis_angle",
            scales=scales,
            return_global_rotations=True,
            return_global_transforms=True,
        )
        output.marker_positions = self._positions_from_attached_frames(
            output.global_transforms,
            self.spec.markers,
            self._marker_local,
            output.scales,
        )
        output.contact_positions = self._positions_from_attached_frames(
            output.global_transforms,
            self.spec.contacts,
            self._contact_local,
            output.scales,
        )
        return output

    def marker_positions(
        self,
        q: torch.Tensor,
        root: torch.Tensor | None = None,
        marker_names: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Evaluate attached marker positions.

        Parameters
        ----------
        q : torch.Tensor
            Generalized coordinates in radians.
        root : torch.Tensor, optional
            World placement transforms.
        marker_names : sequence of str, optional
            Defaults to all configured frames of this kind.

        Returns
        -------
        torch.Tensor
            Positions in requested order, shape [..., F, 3].

        Raises
        ------
        KeyError
            If a requested attached frame does not exist.
        """
        output = self.forward_kinematics(q, root=root)
        positions = output.marker_positions
        assert positions is not None
        if marker_names is None:
            return positions
        idx = torch.tensor(
            [self.marker_index(name) for name in marker_names],
            device=q.device,
            dtype=torch.long,
        )
        return positions.index_select(-2, idx)

    def contact_positions(
        self,
        q: torch.Tensor,
        root: torch.Tensor | None = None,
        contact_names: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Evaluate attached contact positions.

        Parameters
        ----------
        q : torch.Tensor
            Generalized coordinates in radians.
        root : torch.Tensor, optional
            World placement transforms.
        contact_names : sequence of str, optional
            Defaults to all configured frames of this kind.

        Returns
        -------
        torch.Tensor
            Positions in requested order, shape [..., F, 3].

        Raises
        ------
        KeyError
            If a requested attached frame does not exist.
        """
        output = self.forward_kinematics(q, root=root)
        positions = output.contact_positions
        assert positions is not None
        if contact_names is None:
            return positions
        idx = torch.tensor(
            [self.contact_index(name) for name in contact_names],
            device=q.device,
            dtype=torch.long,
        )
        return positions.index_select(-2, idx)

    def frame_positions(
        self,
        q: torch.Tensor,
        root: torch.Tensor | None = None,
        frame_names: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Evaluate named joints or attached frames.

        Parameters
        ----------
        q : torch.Tensor
            Generalized coordinates, shape [..., D].
        root : torch.Tensor, optional
            World placement transforms.
        frame_names : sequence of str, optional
            Defaults to joints in rig order. Prefix with joint:, marker:, or contact:
            to disambiguate attached names.

        Returns
        -------
        torch.Tensor
            Positions in requested order, shape [..., F, 3].

        Raises
        ------
        KeyError
            If a frame is missing or ambiguous.
        """
        output = self.forward_kinematics(q, root=root)
        markers = output.marker_positions
        contacts = output.contact_positions
        assert markers is not None and contacts is not None
        if frame_names is None:
            return output.joints
        if not frame_names:
            return output.joints[..., :0, :]
        positions = []
        for name in frame_names:
            kind, index = self._resolve_frame(name)
            source = {"joint": output.joints, "marker": markers, "contact": contacts}[
                kind
            ]
            positions.append(source[..., index, :])
        return torch.stack(positions, dim=-2)

    def _resolve_frame(self, name: str) -> tuple[str, int]:
        registries = {
            "joint": self.joint_name_to_index,
            "marker": self.marker_name_to_index,
            "contact": self.contact_name_to_index,
        }
        if ":" in name:
            kind, label = name.split(":", 1)
            if kind not in registries or label not in registries[kind]:
                raise KeyError(f"Unknown frame {name!r}.")
            return kind, registries[kind][label]
        if name in self.joint_name_to_index:
            return "joint", self.joint_index(name)
        matches = [
            (kind, lookup[name])
            for kind, lookup in registries.items()
            if name in lookup
        ]
        if len(matches) != 1:
            raise KeyError(f"Unknown or ambiguous frame {name!r}; use kind:name.")
        return matches[0]

    def frame_parent_index(self, name: str) -> int:
        """Resolve a named frame to its owning joint.

        Parameters
        ----------
        name : str
            Joint name or qualified ``marker:name`` / ``contact:name``.

        Returns
        -------
        int
            Owning joint index.

        Raises
        ------
        KeyError
            If the frame is unknown or ambiguous.
        """
        kind, index = self._resolve_frame(name)
        if kind == "joint":
            return index
        frames = self.spec.markers if kind == "marker" else self.spec.contacts
        return self.joint_index(frames[index].parent)

    def frame_position_jacobian(
        self,
        q: torch.Tensor,
        root: torch.Tensor | None = None,
        frame_names: Sequence[str] | None = None,
        create_graph: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate frame positions and their derivatives with respect to q.

        Parameters
        ----------
        q : torch.Tensor
            Generalized coordinates, shape [B, D].
        root : torch.Tensor, optional
            World placement transforms.
        frame_names : sequence of str, optional
            Defaults to all joints.
        create_graph : bool
            Preserve gradients through the returned Jacobian.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Positions [B, F, 3] and Jacobians [B, 3 * F, D].

        Raises
        ------
        ValueError
            If q does not have a single batch dimension.
        KeyError
            If a frame is unavailable.
        """
        if q.ndim != 2:
            raise ValueError(
                "frame_position_jacobian currently expects q with shape [B, D]."
            )
        with torch.enable_grad():
            q_req = q if q.requires_grad else q.detach().clone().requires_grad_(True)
            positions = self.frame_positions(q_req, root, frame_names)
            flat = positions.reshape(q.shape[0], -1)
            jac_rows = []
            for col in range(flat.shape[-1]):
                grad = None
                if flat[:, col].requires_grad:
                    grad = torch.autograd.grad(
                        flat[:, col].sum(),
                        q_req,
                        retain_graph=True,
                        create_graph=create_graph,
                        allow_unused=True,
                    )[0]
                jac_rows.append(torch.zeros_like(q_req) if grad is None else grad)
            jacobian = (
                torch.stack(jac_rows, dim=1)
                if jac_rows
                else q.new_zeros(q.shape[0], 0, q.shape[1])
            )
        return (positions if create_graph else positions.detach()), jacobian

    def clamp_q(self, q: torch.Tensor) -> torch.Tensor:
        """Clamp rotational coordinates to configured component limits.

        Parameters
        ----------
        q : torch.Tensor
            Generalized coordinates in radians.

        Returns
        -------
        torch.Tensor
            Coordinates clamped to each configured lower and upper limit.
        """
        if self.spec.joint_limits is None:
            return q
        out = q.clone()
        for joint_idx in self._q_joint_indices:
            limits = self.spec.joint_limits[joint_idx]
            if limits is None:
                continue
            q_slice = self.dof_slice(self.joint_names[joint_idx])
            limit_tensor = torch.tensor(limits, dtype=q.dtype, device=q.device)
            out[..., q_slice] = out[..., q_slice].clamp(
                limit_tensor[:, 0], limit_tensor[:, 1]
            )
        return out

    def joint_limit_loss(self, q: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
        """Penalize violation of configured component limits.

        Parameters
        ----------
        q : torch.Tensor
            Generalized coordinates in radians.
        margin : float
            Inward limit margin, in radians.

        Returns
        -------
        torch.Tensor
            Mean squared violation; zero when limits are absent.
        """
        if self.spec.joint_limits is None:
            return q.new_tensor(0.0)
        penalties = []
        for joint_idx in self._q_joint_indices:
            limits = self.spec.joint_limits[joint_idx]
            if limits is None:
                continue
            q_slice = self.dof_slice(self.joint_names[joint_idx])
            values = q[..., q_slice]
            limit_tensor = torch.tensor(limits, dtype=q.dtype, device=q.device)
            lower = limit_tensor[:, 0] + margin
            upper = limit_tensor[:, 1] - margin
            penalties.append(
                torch.relu(lower - values).square()
                + torch.relu(values - upper).square()
            )
        if not penalties:
            return q.new_tensor(0.0)
        return torch.cat(penalties, dim=-1).mean()

    def center_of_mass(
        self, q: torch.Tensor, root: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute COM from explicit link masses and scaled local COMs.

        Parameters
        ----------
        q : torch.Tensor
            Generalized coordinates, shape [..., D].
        root : torch.Tensor, optional
            World placement transforms.

        Returns
        -------
        torch.Tensor
            World COM, shape [..., 3].

        Raises
        ------
        ValueError
            If physical link metadata is absent.
        """
        if not self.spec.links:
            raise ValueError(
                "Center of mass requires explicit link masses and local COMs."
            )
        output = self.forward_kinematics(q, root=root)
        com = self._positions_from_local_points(
            output.global_transforms, self._link_com.to(q) * output.scales
        )
        masses = self._link_mass.to(q)
        weighted = com * masses.view(*((1,) * (com.ndim - 2)), -1, 1)
        return weighted.sum(dim=-2) / masses.sum().clamp_min(1e-8)

    def export_urdf(self, path: str) -> None:
        """Export this rig specification to URDF.

        Parameters
        ----------
        path : str
            Output filename; uses unscaled specification offsets.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If metadata cannot be represented faithfully.
        OSError
            If the output cannot be written.
        """
        from .rigs._export import export_urdf

        export_urdf(self.spec, path)

    def export_mjcf(self, path: str) -> None:
        """Export this rig specification to MJCF.

        Parameters
        ----------
        path : str
            Output filename; uses unscaled specification offsets.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If metadata cannot be represented faithfully.
        OSError
            If the output cannot be written.
        """
        from .rigs._export import export_mjcf

        export_mjcf(self.spec, path)

    def _positions_from_local_points(
        self, transforms: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        points = points.to(device=transforms.device, dtype=transforms.dtype)
        points = points.expand(transforms.shape[:-3] + points.shape[-2:])
        return (transforms[..., :3, :3] @ points.unsqueeze(-1)).squeeze(
            -1
        ) + transforms[..., :3, 3]

    def _positions_from_attached_frames(
        self,
        transforms: torch.Tensor | None,
        frames: Sequence[Any],
        local_points: torch.Tensor,
        scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if transforms is None:
            raise ValueError(
                "global transforms are required for attached frame positions."
            )
        if not frames:
            return transforms.new_zeros(transforms.shape[:-3] + (0, 3))
        positions = []
        local_points = local_points.to(device=transforms.device, dtype=transforms.dtype)
        for frame_i, frame in enumerate(frames):
            parent_i = self.joint_index(frame.parent)
            parent_t = transforms[..., parent_i, :, :]
            local_xyz = local_points[frame_i].expand(transforms.shape[:-3] + (3,))
            if scales is not None:
                local_xyz = local_xyz * scales[..., parent_i, :]
            positions.append(
                (parent_t[..., :3, :3] @ local_xyz.unsqueeze(-1)).squeeze(-1)
                + parent_t[..., :3, 3]
            )
        return torch.stack(positions, dim=-2)

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
