"""Solve named-joint targets on any registered rig."""

from __future__ import annotations

import argparse

from skeletons import CyclicCoordinateDescentIK, build_layer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skeleton", default="human36m")
    args = parser.parse_args()
    model = build_layer(args.skeleton)
    q = model.zero_pose(batch_size=2)
    frames = [
        model.joint_names[i] for i in range(model.NUM_JOINTS) if i not in model.parents
    ]
    target = model.frame_positions(q + 0.15, frame_names=frames).detach()
    solution = CyclicCoordinateDescentIK(model, frames, max_iter=30).solve(q, target)
    print("iterations:", solution.iterations)
    print("mean final frame error:", float(solution.final_error.mean()))


if __name__ == "__main__":
    main()
