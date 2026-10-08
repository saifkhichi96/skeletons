# Skeletons Playground

The playground is an interactive Qt application for posing articulated
skeletons, fitting motion data, and solving inverse kinematics on registered rigs.
One workspace shares a light theme, a 3D viewport with a checkerboard floor,
a 2D projection preview, an inspector, metric plots, and an event log.

The left sidebar contains five scrollable tabs:

- **Scene:** select a skeleton and inspect joints, markers, links, contacts,
  datasets, cameras, and IK targets in the scene explorer and property inspector.
  The explorer and inspector fill the tab's available height.
- **Data:** load a `.npz` dataset, choose a motion preset, or generate synthetic
  fitting data. Select the dataset frame range here.
- **Move:** play or scrub the selected motion preset or dataset/fitted sequence.
  Edit joint poses, ROM limits, translation, and per-body XYZ scales with compact
  controls. Pose and scale share one joint selector. Save and load pose states
  and ROM limits as JSON.
- **Fit:** load an optional prior checkpoint and run 3D or 2D sequence fitting
  in a background worker. Inspect live loss/progress, cancel a fit, and export
  its `.npz` results. Frames after the first warm-start from the preceding result.
- **IK:** configure CCD, damped least squares, or gradient IK, root following,
  tick delay, and iteration count. Add, edit, or remove up to eight targets,
  each bound to a unique joint. Targets start static; enable moving behavior
  or randomize a target's motion. Place a target at its joint, enter XYZ
  coordinates, or choose **Place in scene** and click to position X/Z at the
  entered Y height. Press **Esc** to cancel placement.
  Run continuously, stop, step once, or reset the shared pose.
  With a dataset loaded, choose **Dataset end effectors** to use its valid 3D
  leaf-joint positions as targets. **Run IK** solves the entire selected frame
  range, using the configured iteration count per frame and carrying each pose
  forward as the next frame's starting point. **Stop** retains the current pose;
  **Step** performs one iteration on the current frame. Completed sequences can
  be played in Move and saved with **Export Sequence**. Missing or zero-confidence
  targets are ignored; frames with no valid end effectors retain the preceding
  pose. Manual targets remain available when switching back to manual mode.

The right sidebar holds display options and the **Metrics / Logs** tabs.
Choose visible layers, fit the camera, and toggle the floor here.
The floor sits below the rig when it is loaded and stays fixed during motion.
Its pale checkerboard renders behind scene elements without hiding them.
IK display options are disabled and their layers hidden outside the IK tab;
their checked preferences are restored when returning to IK.

Manual editing, playback, fitting, and IK use the same rig and pose state.
Starting fitting pauses playback and IK; manual pose edits pause IK.

## Launch

Install the optional GUI dependencies from the project root:

```bash
pip install -e .
pip install -r demo/requirements.txt
```

Launch from the project root:

```bash
python demo/run.py
```

The View menu also provides camera framing and floor visibility controls
(shortcuts **F** and **G**).

The GUI and its animation, ROM, synthetic-data controls, export helpers, and Qt
fitting worker live in this demo; they are separate from the core library.

Run the headless helper tests from the project root:

```bash
PYTHONPATH=src:demo python -m pytest demo/tests
```
