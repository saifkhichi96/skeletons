Fitting Package
===============

The public entry points most users import from
``differential_skeletons.fitting`` are:

- camera models: ``PerspectiveCamera``, ``WeakPerspectiveCamera``
- dataset utilities: ``FrameDataset``, ``SequenceDataset``,
  ``prepare_frame_batch``, ``prepare_sequence_batch``
- fitting API: ``SkeletalFitter``, ``FittingResult``, ``FittingWeights``
- priors: ``JointLimitPrior``, ``JointLimitStatistics``, ``PoseVAE``,
  ``TemporalPrior``, ``MotionSmoothnessPrior``
- training helpers: ``JointLimitTrainer``, ``PoseVAETrainer``,
  ``TemporalPriorTrainer``, ``TrainerState``

Camera Models
-------------

.. automodule:: differential_skeletons.fitting.camera
   :members:
   :show-inheritance:

Dataset and Preparation Helpers
-------------------------------

.. automodule:: differential_skeletons.fitting.data
   :members:
   :show-inheritance:

Fitter
------

.. automodule:: differential_skeletons.fitting.fitter
   :members:
   :show-inheritance:

Priors
------

.. automodule:: differential_skeletons.fitting.priors
   :members:
   :show-inheritance:

Training
--------

.. automodule:: differential_skeletons.fitting.training
   :members:
   :show-inheritance:
