from __future__ import annotations

import torch

from skelix import (
    CocoModel,
    CocoWholeBodyModel,
    Face68Model,
    ForwardKinematicsLoss,
    Halpe26Model,
    HalpeFullBodyModel,
    Hand21Model,
    Human36MModel,
    MPIIModel,
    SpineTrackModel,
)


def test_model_sizes_and_roots() -> None:
    models = [
        (CocoModel(create_global_orient=False, create_body_pose=False), 17, 0),
        (MPIIModel(create_global_orient=False, create_body_pose=False), 16, 6),
        (Human36MModel(create_global_orient=False, create_body_pose=False), 17, 0),
        (Halpe26Model(create_global_orient=False, create_body_pose=False), 26, 19),
        (Hand21Model(create_global_orient=False, create_body_pose=False), 21, 0),
        (Face68Model(create_global_orient=False, create_body_pose=False), 68, 27),
        (HalpeFullBodyModel(create_global_orient=False, create_body_pose=False), 136, 19),
        (CocoWholeBodyModel(create_global_orient=False, create_body_pose=False), 133, 0),
        (SpineTrackModel(create_global_orient=False, create_body_pose=False), 37, 19),
    ]
    for model, num_joints, root_index in models:
        assert model.num_joints == num_joints
        assert model.root_index == root_index
        assert len(model.parents) == num_joints
        assert len(model.topological_order) == num_joints


def test_forward_axis_angle_shapes() -> None:
    model = Human36MModel(
        create_global_orient=False,
        create_body_pose=False,
        create_bone_scales=False,
        create_transl=False,
    )
    batch_size = 4
    body_pose = torch.zeros(batch_size, model.num_joints - 1, 3)
    global_orient = torch.zeros(batch_size, 3)
    transl = torch.zeros(batch_size, 3)
    output = model(
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        return_global_rotations=True,
        return_scaled_offsets=True,
    )
    assert output.joints.shape == (batch_size, model.num_joints, 3)
    assert output.global_rotations.shape == (batch_size, model.num_joints, 3, 3)
    assert output.scaled_offsets.shape == (batch_size, model.num_joints, 3)


def test_face68_accepts_full_pose_when_root_is_not_first() -> None:
    model = Face68Model(create_global_orient=False, create_body_pose=False)
    full_pose = torch.zeros(model.num_joints, 3)
    output = model(full_pose=full_pose, return_full_pose=True)
    assert output.joints.shape == (model.num_joints, 3)
    assert output.full_pose.shape == (model.num_joints, 3)


def test_bone_scale_changes_segment_length() -> None:
    model = Hand21Model(create_global_orient=False, create_body_pose=False, create_bone_scales=False)
    base = model.rest_joints()
    scales = torch.ones(model.num_joints - 1)
    thumb2_index = model.joint_index('thumb2')
    thumb2_non_root = model.non_root_joint_indices.index(thumb2_index)
    scales[thumb2_non_root] = 2.0
    scaled = model.rest_joints(bone_scales=scales)

    parent = model.parents[thumb2_index]
    base_length = torch.linalg.vector_norm(base[thumb2_index] - base[parent]).item()
    scaled_length = torch.linalg.vector_norm(scaled[thumb2_index] - scaled[parent]).item()
    assert scaled_length > base_length * 1.9


def test_rot6d_full_pose() -> None:
    model = CocoModel(create_global_orient=False, create_body_pose=False, pose_repr='rot6d')
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).repeat(model.num_joints, 1)
    output = model(full_pose=ident, pose_repr='rot6d')
    assert output.joints.shape == (model.num_joints, 3)


def test_forward_kinematics_loss_zero_for_identical_targets() -> None:
    model = Halpe26Model(create_global_orient=False, create_body_pose=False)
    full_pose = torch.zeros(2, model.num_joints, 3)
    target = model(full_pose=full_pose).joints.detach()
    loss_fn = ForwardKinematicsLoss(model)
    loss = loss_fn(target, full_pose=full_pose)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
