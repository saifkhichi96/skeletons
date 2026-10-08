from __future__ import annotations

import torch

from skeletons import (
    CocoModel,
    CocoWholeBodyModel,
    Face68Model,
    ForwardKinematicsLoss,
    Halpe26Model,
    HalpeFullBodyModel,
    Hand21Model,
    Human36MModel,
    Human36MModelLayer,
    MPIIModel,
    SpineTrackModel,
    rigs,
)


def test_model_sizes_and_roots() -> None:
    models = [
        (CocoModel(create_global_orient=False, create_body_pose=False), 17, 0),
        (MPIIModel(create_global_orient=False, create_body_pose=False), 16, 6),
        (Human36MModel(create_global_orient=False, create_body_pose=False), 17, 0),
        (Halpe26Model(create_global_orient=False, create_body_pose=False), 26, 19),
        (Hand21Model(create_global_orient=False, create_body_pose=False), 21, 0),
        (Face68Model(create_global_orient=False, create_body_pose=False), 68, 27),
        (
            HalpeFullBodyModel(create_global_orient=False, create_body_pose=False),
            136,
            19,
        ),
        (
            CocoWholeBodyModel(create_global_orient=False, create_body_pose=False),
            133,
            0,
        ),
        (SpineTrackModel(create_global_orient=False, create_body_pose=False), 37, 19),
    ]
    for model, num_joints, root_index in models:
        assert num_joints == model.NUM_JOINTS
        assert model.root_index == root_index
        assert len(model.parents) == num_joints
        assert len(model.topological_order) == num_joints


def test_model_exposes_articulated_body_tree() -> None:
    model = Human36MModel(create_global_orient=False, create_body_pose=False)

    root_joint = model.get_joint(model.root_index)
    left_knee = model.get_joint("left_knee")
    left_knee_body = model.get_body_node("left_knee")
    left_hip_index = model.joint_index("left_hip")

    assert model._rig.root_body_index == model.root_index
    assert len(model.body_nodes) == model.NUM_JOINTS
    assert len(model.joints) == model.NUM_JOINTS
    assert root_joint.joint_type == "free"
    assert root_joint.parent_body_index is None
    assert left_knee.joint_type == "ball"
    assert left_knee.parent_body_index == left_hip_index
    assert left_knee.child_body_index == model.joint_index("left_knee")
    assert left_knee_body.parent_index == left_hip_index
    assert left_knee_body.incoming_joint_index == left_knee.index
    assert (
        model.joint_index("left_knee")
        in model.get_body_node("left_hip").child_body_indices
    )


def test_model_exposes_smpl_style_api_aliases() -> None:
    model = Human36MModel(create_global_orient=False, create_body_pose=False)

    assert model.skeleton is model._rig
    assert model.NUM_JOINTS == model.NUM_JOINTS
    assert model.NUM_JOINTS - 1 == model.NUM_BODY_JOINTS
    assert model.NUM_BODY_JOINTS == model.NUM_JOINTS - 1
    assert "Number of joints" in model.extra_repr()


def test_rigs_module_exposes_direct_constructors() -> None:
    model = rigs.build_layer("human36m")
    spec = rigs.get_spec("human36m")

    assert model.spec.name == "human36m"
    assert spec.name == "human36m"
    assert spec.num_joints == model.NUM_JOINTS


def test_forward_axis_angle_shapes() -> None:
    model = Human36MModel(
        create_global_orient=False,
        create_body_pose=False,
        create_scales=False,
        create_transl=False,
    )
    batch_size = 4
    body_pose = torch.zeros(batch_size, model.NUM_JOINTS - 1, 3)
    global_orient = torch.zeros(batch_size, 3)
    transl = torch.zeros(batch_size, 3)
    output = model(
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        return_global_rotations=True,
        return_scaled_offsets=True,
    )
    assert output.joints.shape == (batch_size, model.NUM_JOINTS, 3)
    assert output.global_rotations.shape == (batch_size, model.NUM_JOINTS, 3, 3)
    assert output.scaled_offsets.shape == (batch_size, model.NUM_JOINTS, 3)
    assert output.scales.shape == (batch_size, model.NUM_JOINTS, 3)


def test_face68_accepts_full_pose_when_root_is_not_first() -> None:
    model = Face68Model(
        create_global_orient=False,
        create_body_pose=False,
        create_scales=False,
        create_transl=False,
    )
    full_pose = torch.zeros(model.NUM_JOINTS, 3)
    output = model(full_pose=full_pose, return_full_pose=True)
    assert output.joints.shape == (model.NUM_JOINTS, 3)
    assert output.full_pose.shape == (model.NUM_JOINTS, 3)


def test_full_pose_output_is_split_from_pose_used() -> None:
    model = Face68Model()
    full_pose = torch.zeros(model.NUM_JOINTS, 3)
    full_pose[model.root_index] = torch.tensor([0.1, 0.2, 0.3])
    full_pose[model.joint_index("face_0")] = torch.tensor([0.4, 0.5, 0.6])

    output = model(full_pose=full_pose)

    assert torch.allclose(
        output.global_orient, full_pose[model.root_index].unsqueeze(0)
    )
    face_index = model.non_root_joint_indices.index(model.joint_index("face_0"))
    assert torch.allclose(
        output.body_pose[0, face_index], full_pose[model.joint_index("face_0")]
    )


def test_model_layer_defaults_without_registered_state() -> None:
    model = Human36MModelLayer()

    assert not hasattr(model, "global_orient")
    assert not hasattr(model, "body_pose")
    assert not hasattr(model, "transl")
    assert not hasattr(model, "scales")

    output = model()

    assert output.joints.shape == (1, model.NUM_JOINTS, 3)
    assert output.global_orient.shape == (1, 3)
    assert output.body_pose.shape == (1, model.NUM_JOINTS - 1, 3)
    assert output.scales.shape == (1, model.NUM_JOINTS, 3)
    assert output.transl.shape == (1, 3)


def test_created_parameters_respect_batch_size_even_for_one() -> None:
    model = Human36MModel()

    assert model.global_orient.shape == (1, 3)
    assert model.body_pose.shape == (1, model.NUM_JOINTS - 1, 3)
    assert model.scales.shape == (1, model.NUM_JOINTS, 3)
    assert model.transl.shape == (1, 3)
    assert model().joints.shape == (1, model.NUM_JOINTS, 3)


def test_reset_params_uses_model_defaults() -> None:
    model = Human36MModel(batch_size=2)
    model.global_orient.data.fill_(1.0)
    model.body_pose.data.fill_(1.0)
    model.scales.data.fill_(3.0)
    model.transl.data.fill_(2.0)

    model.reset_params(transl=torch.ones(2, 3))

    assert torch.allclose(model.global_orient, torch.zeros(2, 3))
    assert torch.allclose(model.body_pose, torch.zeros(2, model.NUM_JOINTS - 1, 3))
    assert torch.allclose(model.scales, torch.ones(2, model.NUM_JOINTS, 3))
    assert torch.allclose(model.transl, torch.ones(2, 3))


def test_bone_scale_shorthand_scales_outgoing_body_offsets() -> None:
    model = Hand21Model(
        create_global_orient=False, create_body_pose=False, create_scales=False
    )
    base = model.rest_joints()
    scales = torch.ones(model.NUM_JOINTS - 1)
    thumb1_index = model.joint_index("thumb1")
    thumb1_non_root = model.non_root_joint_indices.index(thumb1_index)
    thumb2_index = model.joint_index("thumb2")
    scales[thumb1_non_root] = 2.0
    scaled = model.rest_joints(scales=scales)

    parent = model.parents[thumb2_index]
    base_length = torch.linalg.vector_norm(base[thumb2_index] - base[parent]).item()
    scaled_length = torch.linalg.vector_norm(
        scaled[thumb2_index] - scaled[parent]
    ).item()
    assert scaled_length > base_length * 1.9


def test_bone_scale_vec3_matches_parent_body_scaling() -> None:
    model = Hand21Model(
        create_global_orient=False, create_body_pose=False, create_scales=False
    )
    wrist_index = model.root_index
    thumb1_index = model.joint_index("thumb1")
    base = model.rest_joints()
    scales = torch.ones(model.NUM_JOINTS, 3)
    scales[wrist_index] = torch.tensor([2.0, 0.5, 1.5])
    scaled = model.rest_joints(scales=scales)

    base_offset = base[thumb1_index] - base[wrist_index]
    scaled_offset = scaled[thumb1_index] - scaled[wrist_index]
    expected_offset = model.rest_offsets[thumb1_index] * scales[wrist_index]

    assert torch.allclose(scaled_offset, expected_offset)
    assert not torch.allclose(scaled_offset, base_offset)


def test_rot6d_full_pose() -> None:
    model = CocoModel(
        create_global_orient=False, create_body_pose=False, pose_repr="rot6d"
    )
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).repeat(model.NUM_JOINTS, 1)
    output = model(full_pose=ident, pose_repr="rot6d")
    assert output.joints.shape == (1, model.NUM_JOINTS, 3)


def test_forward_kinematics_loss_zero_for_identical_targets() -> None:
    model = Halpe26Model(create_global_orient=False, create_body_pose=False)
    full_pose = torch.zeros(2, model.NUM_JOINTS, 3)
    target = model(full_pose=full_pose).joints.detach()
    loss_fn = ForwardKinematicsLoss(model)
    loss = loss_fn(target, full_pose=full_pose)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
