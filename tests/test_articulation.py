from __future__ import annotations

import torch

from skeletons import MarkerSpec, available_schemas, build_layer, get_spec
from skeletons.ik import CyclicCoordinateDescentIK, DampedLeastSquaresIK
from skeletons.losses import keypoint_3d_loss


def test_skeleton_registries() -> None:
    assert "coco17" in available_schemas()
    assert get_spec("coco").num_joints == 17


def test_skeleton_forward_kinematics_shapes_and_gradients() -> None:
    model = build_layer("human36m")
    q = model.zero_pose(batch_size=3).requires_grad_(True)
    root = model.identity_root(batch_size=3)

    fk = model.forward_kinematics(q, root)
    loss = fk.joints.square().mean()
    loss.backward()

    assert fk.global_transforms.shape == (3, model.NUM_JOINTS, 4, 4)
    assert fk.marker_positions.shape[-1] == 3
    assert q.grad is not None
    assert q.grad.shape == q.shape


def test_skeleton_keypoint_loss_is_zero_for_matching_targets() -> None:
    model = build_layer("human36m")
    q = model.zero_pose(batch_size=2)
    root = model.identity_root(batch_size=2)
    target = model.frame_positions(q, root=root)

    loss = keypoint_3d_loss(model, q, target, root=root)

    assert float(loss) < 1e-8


def test_skeleton_schema_keypoint_loss_selects_detector_layout() -> None:
    model = build_layer(
        "coco", markers=(MarkerSpec("tip", "left_wrist", (0.0, 0.05, 0.0)),)
    )
    q = model.zero_pose(batch_size=2)
    target = model.frame_positions(q)
    loss = keypoint_3d_loss(model, q, target, schema="coco17")
    assert float(loss) < 1e-8


def test_skeleton_dls_ik_reduces_error() -> None:
    torch.manual_seed(1)
    model = build_layer("human36m")
    q0 = model.zero_pose(batch_size=1)
    root = model.identity_root(batch_size=1)
    frames = ["left_wrist", "right_wrist"]
    target_q = q0.clone()
    target_q[:, model.dof_slice("left_shoulder")] = torch.tensor([0.2, 0.0, 0.4])
    target_q[:, model.dof_slice("right_shoulder")] = torch.tensor([-0.2, 0.0, -0.4])
    target = model.frame_positions(target_q, root, frames)
    before = (model.frame_positions(q0, root, frames) - target).norm(dim=-1).mean()

    solver = DampedLeastSquaresIK(
        model, frames, damping=1e-2, step_size=0.8, max_iter=15
    )
    solution = solver.solve(q0, target, root=root)

    assert solution.final_error.mean() < before


def test_skeleton_ccd_ik_reduces_error() -> None:
    model = build_layer("human36m")
    q0 = model.zero_pose(batch_size=1)
    root = model.identity_root(batch_size=1)
    frames = ["left_wrist"]
    target_q = q0.clone()
    target_q[:, model.dof_slice("left_shoulder")] = torch.tensor([0.1, 0.0, 0.3])
    target = model.frame_positions(target_q, root, frames)
    before = (model.frame_positions(q0, root, frames) - target).norm(dim=-1).mean()

    solver = CyclicCoordinateDescentIK(model, frames, max_iter=10, step_size=0.7)
    solution = solver.solve(q0, target, root=root)

    assert solution.final_error.mean() < before


def test_skeleton_export_files(tmp_path) -> None:
    model = build_layer("human36m")
    urdf = tmp_path / "model.urdf"
    mjcf = tmp_path / "model.xml"

    model.export_urdf(str(urdf))
    model.export_mjcf(str(mjcf))

    assert "<robot" in urdf.read_text()
    assert "<mujoco" in mjcf.read_text()
