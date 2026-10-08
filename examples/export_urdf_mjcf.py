from __future__ import annotations

import argparse
from pathlib import Path

from skeletons import build_layer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a registered rig to URDF and MJCF."
    )
    parser.add_argument("--skeleton", default="human36m")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("work_dirs/rig_exports"),
        help="Directory where exported files are written.",
    )
    args = parser.parse_args()

    model = build_layer(args.skeleton)
    out_dir = args.work_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model.export_urdf(str(out_dir / f"{args.skeleton}.urdf"))
    model.export_mjcf(str(out_dir / f"{args.skeleton}.xml"))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
