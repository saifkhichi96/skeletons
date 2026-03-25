from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

from differential_skeletons import Human36MModel

WALK_POSE_PARAMS = [
    {
        "label": "left_contact",
        "global_orient": [0.02, 0.0, -0.04],
        "transl": [0.0, 0.00, 0.00],
        "legs": {
            "left_hip": [-0.45, 0.0, 0.0],
            "left_knee": [-0.55, 0.0, 0.0],
            "right_hip": [0.25, 0.0, 0.0],
            "right_knee": [-0.10, 0.0, 0.0],
        },
    },
    {
        "label": "left_down",
        "global_orient": [0.01, 0.0, -0.02],
        "transl": [0.0, 0.02, 0.08],
        "legs": {
            "left_hip": [-0.25, 0.0, 0.0],
            "left_knee": [-0.30, 0.0, 0.0],
            "right_hip": [0.10, 0.0, 0.0],
            "right_knee": [-0.05, 0.0, 0.0],
        },
    },
    {
        "label": "passing",
        "global_orient": [0.0, 0.0, 0.0],
        "transl": [0.0, 0.03, 0.16],
        "legs": {
            "left_hip": [0.00, 0.0, 0.0],
            "left_knee": [-0.05, 0.0, 0.0],
            "right_hip": [0.00, 0.0, 0.0],
            "right_knee": [-0.05, 0.0, 0.0],
        },
    },
    {
        "label": "right_down",
        "global_orient": [0.01, 0.0, 0.02],
        "transl": [0.0, 0.02, 0.24],
        "legs": {
            "left_hip": [0.10, 0.0, 0.0],
            "left_knee": [-0.05, 0.0, 0.0],
            "right_hip": [-0.25, 0.0, 0.0],
            "right_knee": [-0.30, 0.0, 0.0],
        },
    },
    {
        "label": "right_contact",
        "global_orient": [0.02, 0.0, 0.04],
        "transl": [0.0, 0.00, 0.32],
        "legs": {
            "left_hip": [0.25, 0.0, 0.0],
            "left_knee": [-0.10, 0.0, 0.0],
            "right_hip": [-0.45, 0.0, 0.0],
            "right_knee": [-0.55, 0.0, 0.0],
        },
    },
]


def build_walk_tensors(
    model: Human36MModel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = model.rest_offsets.dtype
    device = model.rest_offsets.device
    num_frames = len(WALK_POSE_PARAMS)

    global_orient = torch.tensor(
        [frame["global_orient"] for frame in WALK_POSE_PARAMS],
        dtype=dtype,
        device=device,
    )
    transl = torch.tensor(
        [frame["transl"] for frame in WALK_POSE_PARAMS],
        dtype=dtype,
        device=device,
    )
    body_pose = torch.zeros(
        num_frames, model.num_joints - 1, 3, dtype=dtype, device=device
    )

    body_pose_indices = {
        model.joint_names[joint_index]: body_pose_index
        for body_pose_index, joint_index in enumerate(model.non_root_joint_indices)
    }
    for frame_index, frame in enumerate(WALK_POSE_PARAMS):
        for joint_name, axis_angle in frame["legs"].items():
            body_pose[frame_index, body_pose_indices[joint_name]] = torch.tensor(
                axis_angle,
                dtype=dtype,
                device=device,
            )

    return body_pose, global_orient, transl


def set_equal_axes(ax: plt.Axes, joints: torch.Tensor) -> None:
    mins = joints.amin(dim=(0, 1))
    maxs = joints.amax(dim=(0, 1))
    center = (mins + maxs) * 0.5
    radius = float((maxs - mins).amax().item() * 0.5 + 0.1)

    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - radius), float(center[2] + radius))
    ax.set_box_aspect((1.0, 1.0, 1.0))


def main() -> None:
    model = Human36MModel(
        create_global_orient=False,
        create_body_pose=False,
        create_bone_scales=False,
        create_transl=False,
    )

    body_pose, global_orient, transl = build_walk_tensors(model)
    output = model(
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
    )
    joints = output.joints.detach().cpu()

    print(joints.shape)  # [num_frames, J, 3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=14, azim=-62)
    set_equal_axes(ax, joints)

    points = ax.scatter([], [], [], color="tab:blue", s=35)
    bones = [
        ax.plot([], [], [], color="tab:red", linewidth=2)[0]
        for _ in range(model.num_joints - 1)
    ]
    title = ax.set_title("")

    def update(frame_index: int):
        frame_joints = joints[frame_index].numpy()
        points._offsets3d = (
            frame_joints[:, 0],
            frame_joints[:, 1],
            frame_joints[:, 2],
        )

        for line_index, joint_index in enumerate(range(1, model.num_joints)):
            parent_index = model.parents[joint_index]
            line = bones[line_index]
            line.set_data(
                [frame_joints[parent_index, 0], frame_joints[joint_index, 0]],
                [frame_joints[parent_index, 1], frame_joints[joint_index, 1]],
            )
            line.set_3d_properties(
                [frame_joints[parent_index, 2], frame_joints[joint_index, 2]]
            )

        title.set_text(
            f"Human3.6M walk keyframe {frame_index + 1}/{len(WALK_POSE_PARAMS)}: "
            f"{WALK_POSE_PARAMS[frame_index]['label']}"
        )
        return [points, *bones, title]

    update(0)
    backend = plt.get_backend().lower()
    if "agg" in backend:
        for frame_index in range(1, len(WALK_POSE_PARAMS)):
            update(frame_index)
            fig.canvas.draw()
        plt.close(fig)
        return

    animation = FuncAnimation(
        fig,
        update,
        frames=len(WALK_POSE_PARAMS),
        interval=220,
        blit=False,
        repeat=False,
    )
    fig.walk_animation = animation
    plt.show()


if __name__ == "__main__":
    main()
