Tutorial: Fit a Model to 3D Joints
==================================

Goal
----

This tutorial covers the standard 3D fitting path with ``SkeletalFitter``.


Minimal fitting example
-----------------------

.. code-block:: python

   import torch
   from differential_skeletons import build_layer
   from differential_skeletons.fitting import SkeletalFitter

   model = build_layer("human36m")
   target = torch.randn(1, model.NUM_JOINTS, 3)

   fitter = SkeletalFitter(model=model)
   result = fitter.fit_3d(target, num_iters=200)

   print(result.losses)
   print(result.model_output.joints.shape)


What happens internally
-----------------------

By default, ``fit_3d(...)``:

1. estimates body scales from the target joints
2. estimates an initial pose using inverse kinematics
3. optimizes pose and translation to match the target joints
4. optionally optimizes body scales

The most important options are:

- ``num_iters``
- ``lr``
- ``optimize_scales``
- ``use_pose_prior_latent``
- ``init_from_ik``


Adding a pose prior
-------------------

If you have a trained ``PoseVAE``, pass it into the fitter:

.. code-block:: python

   fitter = SkeletalFitter(
       model=model,
       pose_prior=pose_prior,
       joint_limit_prior=joint_limit_prior,
   )

Then the fitter can optimize in the latent space of the VAE rather than
directly over body-pose parameters.


Fitting a dataset sample
------------------------

.. code-block:: python

   from differential_skeletons import build_layer
   from differential_skeletons.fitting import FrameDataset, SkeletalFitter

   model = build_layer("spinetrack")
   dataset = FrameDataset.from_npz("dataset.npz", expected_num_joints=model.NUM_JOINTS)
   sample = dataset[0]

   fitter = SkeletalFitter(model=model)
   result = fitter.fit_3d(sample["joints_3d"].unsqueeze(0))


Sequence fitting
----------------

For temporal data, use:

- ``fit_sequence_3d(...)``

This activates temporal and smoothness losses when those priors are present and
is the correct entry point for motion sequences rather than independent frames.
