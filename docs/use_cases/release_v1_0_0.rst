Skeletons 1.0.0
===============

Skeletons is a differentiable PyTorch library for articulated rigs, forward
kinematics, inverse kinematics, fitting, and motion analysis. This release
brings the library, documentation, examples, and interactive playground
together under a rig-centered API.

Articulated rigs and kinematics
-------------------------------

- Nine built-in rig layouts cover body, hand, face, whole-body, and spine
  landmarks, with canonical names and alias resolution.
- Rig metadata supports joint types, axes and limits, links, markers, contacts,
  named frames, and configurable rest geometry.
- Batched forward kinematics exposes joint and named-frame positions,
  rotations, Jacobians, link centers of mass, and global rig properties.
- URDF and MJCF exporters describe supported rig joints and geometry.

Inverse kinematics, fitting, and motion
---------------------------------------

- Generic CCD, damped least-squares, and gradient-based IK work with rig joints
  and named frames.
- Robust keypoint, joint-limit, center-of-mass, foot-sliding, and temporal
  smoothness losses support structured pose and motion objectives.
- Frame and sequence fitting handle 2D or 3D observations, camera parameters,
  confidence weights, learned priors, and sequential warm starts.
- Synthetic datasets, evaluation metrics, cross-rig retargeting, temporal
  denoising, lifting baselines, hybrid IK, and pseudo-label bootstrapping support
  end-to-end research workflows.

Interactive playground and documentation
-----------------------------------------

- The optional Qt playground combines scene inspection, data loading, motion
  control, fitting, and manual or dataset-driven IK in one workspace.
- Dataset-driven IK follows leaf-joint targets through a selected frame range;
  solved poses can be played back and exported.
- Runnable examples demonstrate rig articulation, IK solvers, fitting, motion
  analysis, retargeting, training, and simulation export.
- Sphinx guides cover the public API, core concepts, fitting, training, and
  research workflows.

Install the library with ``pip install skeletons``. Install the interactive
playground with ``pip install skeletons[playground]``.
