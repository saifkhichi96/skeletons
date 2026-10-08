"""Use optional frames and physical metadata on an existing rig."""

from __future__ import annotations

from skeletons import (
    LinkSpec,
    MarkerSpec,
    build_layer,
    get_spec,
    keypoint_3d_loss,
)


def main() -> None:
    spec = get_spec("human36m")
    model = build_layer(
        "human36m",
        markers=(MarkerSpec("wrist_tip", "left_wrist", (0.0, 0.05, 0.0)),),
        links=tuple(LinkSpec(name) for name in spec.joint_names),
    )
    q = model.zero_pose(batch_size=4)
    fk = model.forward_kinematics(q)
    print("joints:", fk.joints.shape)
    print("markers:", fk.marker_positions.shape)
    # Unit masses are illustrative, not calibrated body properties.
    print("illustrative center of mass:", model.center_of_mass(q).shape)
    print("joint loss:", float(keypoint_3d_loss(model, q, fk.joints)))
    print(
        "marker loss:",
        float(
            keypoint_3d_loss(
                model, q, fk.marker_positions, frame_names=["marker:wrist_tip"]
            )
        ),
    )


if __name__ == "__main__":
    main()
