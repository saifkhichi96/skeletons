Tutorial: Fit a Model to 2D Joints
==================================

Goal
----

This tutorial shows how to recover a 3D articulated skeleton from 2D keypoints
and a camera model.


Perspective-camera fitting
--------------------------

.. code-block:: python

   import torch
   from differential_skeletons import build_layer
   from differential_skeletons.fitting import PerspectiveCamera, SkeletalFitter

   model = build_layer("human36m")
   target_2d = torch.randn(1, model.NUM_JOINTS, 2)

   camera = PerspectiveCamera(
       fx=torch.tensor(1000.0),
       fy=torch.tensor(1000.0),
       cx=torch.tensor(512.0),
       cy=torch.tensor(512.0),
   )

   fitter = SkeletalFitter(model=model)
   result = fitter.fit_2d(target_2d, camera, num_iters=300)


Confidence weights
------------------

If your detector provides confidence scores, pass them into ``fit_2d(...)``:

.. code-block:: python

   result = fitter.fit_2d(
       target_2d,
       camera,
       confidences=confidences,
       num_iters=300,
   )

Confidence weighting is especially helpful when a few joints are unreliable or
missing.


Why priors matter more in 2D
----------------------------

2D fitting is more ambiguous than 3D fitting because:

- depth is not directly observed
- global pose can trade off with camera depth
- local twists are often underconstrained

In practice, 2D fitting benefits strongly from:

- ``PoseVAE``
- ``JointLimitPrior``
- ``TemporalPrior`` for sequences
- ``MotionSmoothnessPrior``


Sequence fitting from 2D
------------------------

For motion sequences, use:

- ``fit_sequence_2d(...)``

This is the correct path for videos or frame ranges rather than isolated images.
