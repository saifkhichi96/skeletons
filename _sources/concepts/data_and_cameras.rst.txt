Data Loading and Cameras
========================

Frame and sequence datasets
---------------------------

The fitting package provides two dataset containers:

- ``FrameDataset`` for arrays shaped like ``[N, J, ...]``
- ``SequenceDataset`` for arrays shaped like ``[N, T, J, ...]``

Both loaders use the same ``.npz`` schema and preserve unknown fields as
metadata.


Canonical ``.npz`` inputs
-------------------------

Recognized canonical keys are:

- ``joints_3d`` for required 3D targets
- ``joints_2d`` for optional image-space targets
- ``confidences`` for optional joint weights
- ``fx``, ``fy``, ``cx``, ``cy`` for intrinsics
- ``camera_translation`` and ``camera_rotation`` for extrinsics

The loaders also accept common aliases. Examples include:

- ``S`` or ``keypoints_3d`` for ``joints_3d``
- ``part`` or ``pose2d`` for ``joints_2d``
- ``scores`` or ``weights`` for ``confidences``
- ``cam_t`` and ``cam_r`` for camera extrinsics

Keys are normalized before matching, so differences in case, underscores, and
similar punctuation usually do not matter.


Automatic normalization
-----------------------

The loaders apply a few common normalizations:

- 3D arrays with a trailing size of ``4`` are truncated to XYZ
- 2D arrays with a trailing size of ``3`` are interpreted as XY plus confidence
  when no separate confidence array is present


Preparing fitting batches
-------------------------

The helper functions:

- ``prepare_frame_batch(...)``
- ``prepare_sequence_batch(...)``

run inverse-kinematics-style initialization to convert 3D joints into:

- root rotation in 6D form
- non-root joint rotations in 6D form
- per-body scales

This is the bridge between raw joint datasets and prior training.


Camera models
-------------

The fitting package includes two simple camera models:

``PerspectiveCamera``
   Uses ``fx``, ``fy``, ``cx``, and ``cy`` to project 3D points.

``WeakPerspectiveCamera``
   Uses a scalar scale plus 2D translation.

Both are broadcast-friendly and can be passed directly into 2D fitting code.
