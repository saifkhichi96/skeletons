Tutorial: Fit Motion Sequences
==============================

Goal
----

This tutorial covers the sequence-fitting entry points and when to use temporal
priors instead of fitting each frame independently.


When to use sequence fitting
----------------------------

Use the sequence API when your data has a time dimension and you care about:

- temporal continuity
- smoother pose trajectories
- warm-starting later frames from earlier ones
- temporal priors trained on motion clips

The relevant methods are:

- ``fit_sequence_3d(...)``
- ``fit_sequence_2d(...)``


Minimal 3D sequence example
---------------------------

.. code-block:: python

   import torch
   from skeletons import build_layer
   from skeletons.fitting import MotionSmoothnessPrior, SkeletalFitter

   model = build_layer("spinetrack")
   target = torch.randn(2, 60, model.NUM_JOINTS, 3)

   fitter = SkeletalFitter(
       model=model,
       smoothness_prior=MotionSmoothnessPrior(),
   )
   result = fitter.fit_sequence_3d(target, num_iters=200)

   print(result.model_output.joints.shape)  # [2, 60, J, 3]


Using a temporal prior
----------------------

If you have trained a ``TemporalPrior``, pass it into the fitter:

.. code-block:: python

   fitter = SkeletalFitter(
       model=model,
       temporal_prior=temporal_prior,
       smoothness_prior=MotionSmoothnessPrior(),
   )

This adds a learned motion-model term on top of the per-frame fitting losses.


Loading sequence data from ``.npz``
-----------------------------------

Use ``SequenceDataset.from_npz(...)`` when your file stores arrays shaped like
``[N, T, J, ...]``:

.. code-block:: python

   from skeletons.fitting import SequenceDataset

   dataset = SequenceDataset.from_npz(
       "clips.npz",
       expected_num_joints=model.NUM_JOINTS,
   )
   sample = dataset[0]
   target = sample["joints_3d"].unsqueeze(0)


3D versus 2D sequences
----------------------

``fit_sequence_3d(...)``
   Use when you have 3D joint targets and want the strongest geometric signal.

``fit_sequence_2d(...)``
   Use when you have image-space keypoints plus a camera model. Priors are more
   important here because reprojection alone leaves depth and twist ambiguities.


Practical advice
----------------

- Keep the skeleton name consistent across the dataset, fitted model, and priors.
- Start with ``MotionSmoothnessPrior`` even if you do not yet have a trained
  ``TemporalPrior``.
- Use sequence fitting instead of manual frame-by-frame loops when you want the
  optimizer to reason about the whole clip jointly.
