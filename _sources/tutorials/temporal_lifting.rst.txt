Tutorial: Temporal Lifting with FK Supervision
==============================================

Goal
----

Show how to use a rig as the differentiable decoding layer inside a temporal
2D-to-3D lifting model.

Why this tutorial exists
------------------------

Modern sequence models such as VideoPose3D, PoseFormer, PoseFormerV2, and
MotionBERT all exploit temporal context to improve 2D-to-3D reconstruction.
The architecture can change, but the articulated layer remains useful: predict a
pose-like representation, decode it through FK, and supervise in joint space.

Runnable example
----------------

.. literalinclude:: ../../examples/temporal_lifting_videopose3d.py
   :language: python
   :caption: examples/temporal_lifting_videopose3d.py

What to notice
--------------

- the network predicts a per-frame full pose tensor
- ``ForwardKinematicsLoss`` converts that into joint supervision
- the tutorial is skeleton-agnostic and works with any supported rig
- the same pattern scales to more advanced temporal encoders
