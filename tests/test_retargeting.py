from __future__ import annotations

import pytest
import torch

from skeletons import (
    build_joint_mapping,
    build_layer,
    retarget_joint_positions,
    retarget_skeleton,
)


def test_build_joint_mapping_matches_exact_names_and_synonyms() -> None:
    coco = build_layer("coco")
    human36m = build_layer("human36m")

    mapping = build_joint_mapping(coco, human36m, min_joints=12)
    pairs = set(mapping.pairs())

    assert (
        coco.joint_index("left_ankle"),
        human36m.joint_index("left_foot"),
        "left_foot",
    ) in pairs
    assert (
        coco.joint_index("right_ankle"),
        human36m.joint_index("right_foot"),
        "right_foot",
    ) in pairs
    assert (
        coco.joint_index("left_shoulder"),
        human36m.joint_index("left_shoulder"),
        "left_shoulder",
    ) in pairs


def test_build_joint_mapping_accepts_explicit_name_map() -> None:
    coco = build_layer("coco")
    human36m = build_layer("human36m")

    mapping = build_joint_mapping(
        coco,
        human36m,
        joint_map={"nose": "head"},
    )

    assert (
        coco.joint_index("nose"),
        human36m.joint_index("head"),
        "head",
    ) in mapping.pairs()


def test_retarget_joint_positions_transfers_sparse_observations_and_weights() -> None:
    source = build_layer("coco")
    target = build_layer("human36m")
    source_joints = torch.arange(source.NUM_JOINTS * 3, dtype=torch.float32).reshape(
        source.NUM_JOINTS,
        3,
    )
    source_weights = torch.linspace(0.1, 1.0, source.NUM_JOINTS)
    mapping = build_joint_mapping(source, target)

    retargeted = retarget_joint_positions(
        source_joints,
        source,
        target,
        mapping=mapping,
        source_weights=source_weights,
        fill_value=-1.0,
    )

    src_index = source.joint_index("left_ankle")
    dst_index = target.joint_index("left_foot")
    assert torch.allclose(retargeted.joints[dst_index], source_joints[src_index])
    assert torch.allclose(retargeted.weights[dst_index], source_weights[src_index])
    assert torch.all(retargeted.weights >= 0.0)
    assert retargeted.weights[target.root_index] == 0.0
    assert torch.all(retargeted.joints[target.root_index] == -1.0)


def test_retarget_skeleton_recovers_same_rig_pose_from_dense_mapping() -> None:
    source = build_layer("human36m")
    target = build_layer("human36m")
    generator = torch.Generator().manual_seed(7)
    full_pose = torch.randn(2, source.NUM_JOINTS, 3, generator=generator) * 0.04
    transl = torch.tensor([[0.2, -0.3, 1.0], [-0.1, 0.4, 0.2]])
    source_joints = source(full_pose=full_pose, transl=transl).joints.detach()

    result = retarget_skeleton(
        source_joints,
        source_model=source,
        target_model=target,
        num_iters=2,
        optimize_scales=False,
    )

    assert result.observations.joints.shape == source_joints.shape
    assert result.observations.weights.shape == source_joints.shape[:-1]
    assert result.shared_joint_mpjpe < 1e-3
    max_error = torch.linalg.vector_norm(
        result.fitting_result.model_output.joints - source_joints,
        dim=-1,
    ).max()
    assert max_error < 2e-3


def test_retargeting_validates_inputs() -> None:
    source = build_layer("coco")
    target = build_layer("human36m")

    with pytest.raises(ValueError, match="source_joints"):
        retarget_joint_positions(torch.zeros(source.NUM_JOINTS, 2), source, target)

    with pytest.raises(ValueError, match="at least"):
        build_joint_mapping(source, target, min_joints=source.NUM_JOINTS + 1)

    with pytest.raises(ValueError, match="non-negative"):
        retarget_joint_positions(
            torch.zeros(source.NUM_JOINTS, 3),
            source,
            target,
            source_weights=torch.full((source.NUM_JOINTS,), -1.0),
        )
