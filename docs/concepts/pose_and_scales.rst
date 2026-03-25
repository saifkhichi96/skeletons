Pose Representations and Scales
===============================

Pose inputs
-----------

Models accept either:

- ``global_orient`` and ``body_pose`` separately, or
- a single ``full_pose`` tensor

Supported pose representations are:

- ``axis_angle``
- ``quat``
- ``rot6d``
- ``rotmat``

The helper functions in :mod:`differential_skeletons.rotations` convert between
these formats.


Typical shapes
--------------

For a model with ``J`` joints:

``global_orient``
   ``[..., 3]`` for axis-angle, ``[..., 4]`` for quaternion,
   ``[..., 6]`` for rotation-6D, or ``[..., 3, 3]`` for rotation matrices.

``body_pose``
   ``[..., J - 1, feat]`` or ``[..., J - 1, 3, 3]``.

``full_pose``
   ``[..., J, feat]`` or ``[..., J, 3, 3]``.


OpenSim-style body scales
-------------------------

The package uses an OpenSim-like body-scaling interpretation rather than a
shape-space such as ``betas`` from SMPL-family models.

Each body has a scale factor, and child joint offsets are scaled in the parent
body frame.

Canonical scale shape:

- ``[..., J, 3]``

Supported shorthand forms:

- ``[..., J]`` for isotropic per-body scale
- ``[..., J - 1]`` for isotropic non-root scale
- ``[..., J - 1, 3]`` for explicit non-root per-axis scale

The model canonicalizes all of these into ``[..., J, 3]`` internally.


Rest shape vs posed output
--------------------------

Two useful entry points are:

``rest_joints(scales=...)``
   Returns the scaled rest skeleton in joint coordinates.

``forward(...)``
   Applies articulation and translation to produce posed joints, rotations, and
   optional transforms.

If you only want the scaled template skeleton, ``forward_shape()`` is the most
direct interface.
