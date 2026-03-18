# skelix Playground

Small PySide6 + pyqtgraph playground for the local `skelix` checkout.

## Install

```bash
pip install -e .[app]
```

## Run

From the repository root:

```bash
python -m app.main
```

The launcher also works from a source checkout because it inserts `src/` into `sys.path` before importing `skelix`.

## What it does

- Select any bundled skeleton spec.
- See the current pose in a 3D viewer with bones rendered as pyramid meshes and joints rendered as sphere meshes.
- Play deterministic preset animations such as root sway, walk cycles, arm waves, and finger waves when the current skeleton supports them.
- Edit any joint's axis-angle pose with three sliders.
- Edit global translation with three sliders.
- Scale individual bones with a dedicated selector and slider.
- Reset the selected joint, reset the selected bone, or reset the full scene.
