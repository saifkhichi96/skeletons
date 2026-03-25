Tutorial: Use Trained Priors
============================

Goal
----

This tutorial shows how to load trained priors and apply them during fitting.


Use the command-line fitter
---------------------------

The highest-level path is:

.. code-block:: bash

   python examples/fit_demo.py dataset.npz --skeleton spinetrack --priors work_dirs/train_prior/spinetrack_my_dataset

The ``--priors`` argument accepts:

- a checkpoint file such as ``epoch_10.pth``
- a ``last_checkpoint`` file
- a full run directory that contains ``last_checkpoint``


Apply priors in Python
----------------------

The fitting API itself does not load checkpoints for you. Instead, you create
the prior objects and pass them into ``SkeletalFitter``:

.. code-block:: python

   import torch
   from differential_skeletons import build_layer
   from differential_skeletons.fitting import SkeletalFitter

   model = build_layer("spinetrack")
   fitter = SkeletalFitter(
       model=model,
       pose_prior=pose_prior,
       joint_limit_prior=joint_limit_prior,
   )

   result = fitter.fit_3d(target_joints, num_iters=200)


When to use each prior
----------------------

``JointLimitPrior``
   Good for discouraging obviously implausible local rotations.

``PoseVAE``
   Good when you want a learned, low-dimensional articulation prior. This is
   especially useful for sparse 2D fitting or noisy 3D detections.

``TemporalPrior``
   Good for full motion sequences where frame-to-frame continuity matters.

``MotionSmoothnessPrior``
   Good when you want a simple deterministic smoothness term even without a
   learned temporal model.


3D fitting with priors
----------------------

Common pattern:

.. code-block:: python

   result = fitter.fit_3d(
       target_joints,
       num_iters=300,
       optimize_scales=False,
       use_pose_prior_latent=True,
   )

With ``use_pose_prior_latent=True``, the optimizer works in the VAE latent space
instead of directly optimizing the entire non-root pose tensor.


2D fitting with priors
----------------------

Priors are even more valuable for ``fit_2d(...)`` because 2D observations do not
fully specify the underlying 3D articulation.

Typical pattern:

.. code-block:: python

   result = fitter.fit_2d(
       target_joints_2d,
       camera,
       confidences=confidences,
       num_iters=400,
   )


Sequence fitting with priors
----------------------------

For sequences, use:

- ``fit_sequence_3d(...)``
- ``fit_sequence_2d(...)``

These are the places where ``TemporalPrior`` and ``MotionSmoothnessPrior`` are
most useful.


Applications
------------

Common applications of trained priors include:

- regularized optimization for pose reconstruction
- stabilizing noisy detector outputs
- generating plausible initializations for downstream optimization
- building dataset-specific motion models
- measuring how far a pose lies from the training distribution
