# CHANGELOG

## Streams
- Core Implementation
- Rig Extensions
- Training & Evaluation
- Documentation
- Examples
- CI & Release

## Completed
- [Examples] Positioned the playground floor below loaded rigs and
  rendered its pale checkerboard behind scene elements without occluding them.
- [Examples] Added dataset end-effector IK over selected frame ranges,
  warm-started poses, sequence playback, and export through the playground.
- [Examples] Unified playground rig, data, movement, fitting, and IK
  controls in one workspace with compact controls and shared display/analysis.
- [Examples] Expanded the Scene inspector vertically, shared the Move
  joint selector, and scoped IK display layers to the IK tab. Added up to eight
  uniquely bound static or moving targets with coordinate and scene placement.
- [Examples] Applied a consistent light Qt theme, slim scrollbars,
  scrollable playground sidebars, and initial sizing within the available screen.
- [Documentation] Compressed repository agent instructions.
- [Examples] Corrected playground CCD confidence dimensions,
  locked dataset and fitting configuration during active fits, and preserved
  scene explorer selection during refreshes.
- [Core Implementation] Extended existing rigs with optional joint
  types, axes, limits, link metadata, markers, contacts, and named-frame FK/Jacobians.
- [Core Implementation] Added generic CCD, DLS, gradient IK, robust
  keypoint losses, physical/temporal losses, and URDF/MJCF export.
- [Examples] Added rig-selectable IK, FK/loss, and export examples
  and a generic IK Lab in the playground.
- [Examples] Split playground animation presets, ROM-limit schema
  handling, and the Qt fitting worker into package-backed modules while keeping
  ``playground.py`` as the launcher/application shell.
- [Core Implementation] Added tested playground ROM helpers for
  default limit creation, payload serialization, skeleton validation, and schema
  error reporting.
- [Examples] Added tested procedural animation helpers for preset
  discovery, model-shaped animation buffers, ROM wander limits, and authored
  walk-cycle tensor generation.
- [Examples] Turned the playground fitting panel into a self-contained
  fitting lab with in-app synthetic dataset generation, shared prior-checkpoint
  loading, and fitted-sequence ``.npz`` export.
- [Training & Evaluation] Added
  ``generate_synthetic_fitting_dataset()`` and ``SyntheticFittingDataset`` so
  examples, tests, and the playground can create paired 3D/2D fitting data
  without duplicating generator code.
- [Core Implementation] Added headless
  ``skeletons.playground`` fitting-lab helpers for synthetic
  controls and stable fit-export payloads.
- [Training & Evaluation] Made frame and sequence datasets accept
  scalar camera constants and made camera projection broadcast per-frame
  intrinsics over joint dimensions.
- [Core Implementation] Clarified rig-name discovery so
  ``SUPPORTED_SKELETONS`` contains canonical names, ``SKELETON_ALIASES`` records
  accepted aliases, and ``canonicalize_skeleton_name()`` normalizes external
  inputs for ``build_layer()``, ``create()``, and ``get_spec()``.
- [Training & Evaluation] Added tensor-native evaluation metrics for
  joint-position error, MPJPE, Procrustes-aligned MPJPE, and PCK.
- [Training & Evaluation] Added public cross-skeleton retargeting
  helpers for joint mapping, sparse target observations, and optimization-based
  target-rig fitting.
- [Training & Evaluation] Updated iterative fitting to restore the
  best parameter state seen during optimization instead of blindly returning the
  final Adam step.
- [Examples] Promoted reusable synthetic pose, sequence, camera, and
  lifting-dataset helpers from examples into ``skeletons.synthetic``.
- [Examples] Promoted run directory, JSON artifact, status logging,
  and checkpoint-reference helpers into ``skeletons.artifacts``.
- [Training & Evaluation] Promoted frame and temporal 2D-to-3D
  lifter baselines plus shared evaluation into ``skeletons.lifting``.
- [Training & Evaluation] Promoted the HybrIK-style joint regressor,
  analytical IK recovery, and hybrid evaluation metrics into
  ``skeletons.hybrid``.
- [Training & Evaluation] Promoted pseudo-label bootstrap prior
  training, 2D fitting, dataset conversion, and ``.npz`` export helpers into
  ``skeletons.pseudo_labeling``.
- [Training & Evaluation] Promoted temporal-prior training, 3D
  sequence denoising, 2D sequence reconstruction, and sequence fitting result
  diagnostics into ``skeletons.sequences``.
- [Training & Evaluation] Updated sequence fitting results to report
  the optimizer iteration count completed by ``SkeletalFitter``.
- [Training & Evaluation] Promoted fitting-prior checkpoint schema,
  PoseVAE architecture inference, skeleton validation, save/load helpers, and
  fitter construction into ``skeletons.fitting.checkpoints``.
- [Training & Evaluation] Promoted supervised epoch helpers for
  frame/temporal lifters and HybrIK-style joint regressors into the package.
- [CI & Release] Added subprocess smoke coverage for representative
  examples with tiny workloads, including prior checkpoint training and loading.
- [Examples] Added ``--fit-iters`` to ``examples/fit_demo.py`` so
  quick demos and smoke tests can run the fitting path with bounded work.
- [CI & Release] Added GitHub Actions jobs for Ruff linting,
  multi-version pytest, Sphinx documentation builds, and source/wheel package
  builds.
- [Documentation] Updated the quickstart to
  describe canonical rig names, alias resolution, evaluation metrics, and
  retargeting.
- Project scaffolding (README, basic package layout)
- Added initial unit test structure
- Created quickstart and documentation skeleton

## Planned
- Continue splitting ``playground.py`` into tested package-backed seams:
  viewport/rendering widgets, reusable Qt controls, projection preview, and
  application state synchronization.
- Promote remaining reusable example code into package modules, especially
  fitter configuration presets, dataset adapters, and learned model checkpoint
  helpers.
- Extend retargeting and pseudo-labeling with richer semantic maps, temporal
  warm starts, and detector confidence propagation across multi-stage flows.
- Add SMPL-compatible and dataset-specific adapter layers where they can be
  expressed without introducing heavyweight mesh dependencies.
- Add stronger release-quality checks: type checking, wheel install tests, and
  docs warning-as-error builds.
- Add release publishing automation after PyPI credentials are configured
- Publish first release to PyPI


## Detailed Plan

### Phase 1: Differentiable kinematics
-  Skeleton graph
-  Batched FK
-  Marker FK
-  Rotation conversions
-  Joint limits
-  COCO/MPII/H36M/OpenPose/MediaPipe schemas
-  Basic visualization

### Phase 2: IK and fitting
-  CCD for chains and trees
-  Damped least squares IK
-  Gradient-based fitting
-  Robust losses
-  Confidence masks
-  Temporal smoothing
-  Bone-length scaling

### Phase 3: Retargeting
-  Schema maps
-  Model-to-model maps
-  Scale estimation
-  Contact-aware retargeting
-  BVH import/export
-  SMPL/SMPL-X adapters

### Phase 4: Dynamics
-  Link inertias
-  COM
-  RNEA inverse dynamics
-  Mass matrix / CRBA
-  Simple forward dynamics
-  Soft contacts
-  Physics losses

### Phase 5: Simulation I/O
-  URDF exporter
-  MJCF exporter
-  MuJoCo examples
-  Optional OpenSim/OSIM export or import
