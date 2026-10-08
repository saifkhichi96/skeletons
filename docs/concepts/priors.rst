Priors
======

What a prior means here
-----------------------

In Skeletons, a prior is a regularizer or learned model over
skeleton pose rather than over mesh shape or texture.

Priors are especially useful because 2D and sparse 3D keypoints usually do not
fully constrain articulated motion. A prior helps steer optimization toward
plausible solutions.


Available priors
----------------

``JointLimitPrior``
   An empirical axis-angle-space prior over non-root joints. It learns a mean,
   standard deviation, and lower and upper quantiles from a dataset of poses.
   During fitting, it penalizes both z-score deviation and excursions beyond the
   learned quantile range.

``PoseVAE``
   A variational autoencoder over non-root joint rotations in 6D form. It learns
   a compact latent representation of articulation and can be used either as a
   training target, as a latent regularizer, or as a generator of plausible pose
   codes.

``TemporalPrior``
   A GRU-based autoregressive prior over sequences of body poses in 6D form.

``MotionSmoothnessPrior``
   A finite-difference regularizer over pose and optional translation sequences.


What priors are not
-------------------

These priors do not learn:

- mesh shape
- skinning
- surface appearance
- a full anatomical body model with hidden twist parameters

They are priors over observable articulated skeleton motion.


Typical applications
--------------------

Priors are useful for:

- regularized 3D fitting from noisy joint detections
- 2D-to-3D fitting when reprojection alone is underconstrained
- sequence fitting with temporal coherence
- dataset-specific articulation modeling
- latent-space interpolation and sampling
- pose plausibility scoring and outlier detection


How priors are consumed during fitting
--------------------------------------

The main entry point is ``SkeletalFitter``. You pass any combination of priors
into its constructor:

.. code-block:: python

   from skeletons import build_layer
   from skeletons.fitting import SkeletalFitter

   model = build_layer("spinetrack")
   fitter = SkeletalFitter(
       model=model,
       pose_prior=pose_prior,
       joint_limit_prior=joint_limit_prior,
       temporal_prior=temporal_prior,
   )

Then choose one of:

- ``fit_3d(...)``
- ``fit_2d(...)``
- ``fit_sequence_3d(...)``
- ``fit_sequence_2d(...)``


Artifacts produced by training
------------------------------

The shipped training scripts write outputs under ``work_dirs`` in an
MMEngine-style layout:

- ``config.json``
- ``metrics.json``
- ``epoch_<n>.pth``
- ``last_checkpoint``

The ``fit_demo.py`` script accepts either the ``.pth`` file itself or the run
directory containing ``last_checkpoint``.
