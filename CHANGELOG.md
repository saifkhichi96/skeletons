# Changelog

All notable changes to Skeletons are documented here, grouped by version and
change type. Unreleased entries describe completed changes awaiting release.

## [Unreleased]

### Changed

- No unreleased changes.

## [0.1.0]

### Added

- Differentiable, batched forward kinematics with stateful models and stateless
  layers; axis-angle, quaternion, 6D, and rotation-matrix pose representations.
- Nine built-in rigs for body, hand, face, whole-body, and spine landmarks.
- Configurable joint types, axes, limits, link metadata, markers, contacts,
  named frames, centers of mass, and position Jacobians.
- Generic CCD, damped least-squares, and gradient-based inverse kinematics.
- Robust keypoint, joint-limit, center-of-mass, foot-sliding, and temporal losses.
- Frame and sequence fitting for 2D/3D observations, confidence weights, camera
  intrinsics, skeletal scales, and learned priors.
- CLI sequence fitting with frame-range selection and sequential warm starts.
- Joint-limit, VAE, and temporal priors; reusable checkpoint loading, saving,
  architecture inference, and rig validation.
- Synthetic pose, motion, camera, lifting, and paired 2D/3D fitting datasets.
- Joint-position error, MPJPE, Procrustes-aligned MPJPE, and PCK metrics.
- Cross-rig joint mapping and optimization-based retargeting.
- Frame and temporal lifting baselines, hybrid IK, supervised training helpers,
  pseudo-label bootstrapping, and sequence denoising and reconstruction.
- Run directories, JSON artifacts, status logging, and checkpoint references.
- Experimental URDF and MJCF export, with documented joint-representation limits.
- Runnable IK, fitting, lifting, training, retargeting, and export examples.
- Sphinx API references, concepts, tutorials, and research workflow guides.
- Unit tests and subprocess smoke coverage for representative examples.

### Changed

- Standardized rig discovery through canonical names, aliases, and shared
  `build_layer()`, `create()`, and `get_spec()` entry points.
- Exposed reusable research helpers through dedicated library modules.

### Fixed

- Improved small-angle rotations, 6D rotation normalization, and prior stability.
- Restored the best parameter state observed during iterative fitting.
- Supported scalar camera constants and broadcast intrinsics over joints.
- Reported completed optimizer iterations in sequence fitting results.
