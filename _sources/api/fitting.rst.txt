Fitting Package
===============

The public entry points most users import from
``skeletons.fitting`` are:

- camera models: ``PerspectiveCamera``, ``WeakPerspectiveCamera``
- dataset utilities: ``FrameDataset``, ``SequenceDataset``,
  ``prepare_frame_batch``, ``prepare_sequence_batch``
- fitting API: ``SkeletalFitter``, ``FittingResult``, ``FittingWeights``
- checkpoint helpers: ``FittingPriorBundle``,
  ``load_fitting_prior_checkpoint``, ``save_fitting_prior_checkpoint``
- priors: ``JointLimitPrior``, ``JointLimitStatistics``, ``PoseVAE``,
  ``TemporalPrior``, ``MotionSmoothnessPrior``
- training helpers: ``JointLimitTrainer``, ``PoseVAETrainer``,
  ``TemporalPriorTrainer``, ``TrainerState``

Camera Models
-------------

.. automodule:: skeletons.fitting.camera
   :members:
   :show-inheritance:

Dataset and Preparation Helpers
-------------------------------

.. automodule:: skeletons.fitting.data
   :members:
   :show-inheritance:

Fitter
------

.. automodule:: skeletons.fitting.fitter
   :members:
   :show-inheritance:

Checkpoints
-----------

.. automodule:: skeletons.fitting.checkpoints
   :members:
   :show-inheritance:

Priors
------

.. automodule:: skeletons.fitting.priors
   :members:
   :show-inheritance:

Training
--------

.. automodule:: skeletons.fitting.training
   :members:
   :show-inheritance:
