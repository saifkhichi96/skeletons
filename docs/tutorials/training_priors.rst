Tutorial: Train Priors
======================

Goal
----

This tutorial explains how to train the two priors covered by the shipped
``train_prior.py`` example:

- ``JointLimitPrior``
- ``PoseVAE``


What the training script expects
--------------------------------

The script consumes a frame-wise ``.npz`` dataset that contains 3D joints in the
joint order of the selected skeleton.

Canonical minimum input:

- ``joints_3d`` with shape ``[N, J, 3]``

The loader also accepts common aliases such as ``S`` and ``keypoints_3d``. See
:doc:`../concepts/data_and_cameras` for details.


Train from the command line
---------------------------

.. code-block:: bash

   python examples/train_prior.py path/to/data.npz --skeleton spinetrack

Important options:

- ``--skeleton`` selects the rig and joint topology
- ``--batch-size`` controls loader throughput
- ``--epochs`` controls VAE training duration
- ``--latent-dim``, ``--hidden-dim``, and ``--num-hidden-layers`` control the VAE
- ``--work-dir`` overrides the default MMEngine-style output directory


What gets written to ``work_dirs``
----------------------------------

By default, outputs land in:

.. code-block:: text

   work_dirs/train_prior/<skeleton>_<dataset_stem>/

The directory contains:

``config.json``
   The training configuration used for the run.

``metrics.json``
   Epoch-level loss history and the last checkpoint path.

``epoch_<n>.pth``
   The saved checkpoint with the trained priors.

``last_checkpoint``
   A pointer file that contains the path to the latest checkpoint.


What is inside the checkpoint
-----------------------------

The saved checkpoint includes:

- ``skeleton``
- ``pose_prior_config``
- ``pose_prior``
- ``joint_limit_prior_config``
- ``joint_limit_prior``
- ``history``

That is exactly what ``examples/fit_demo.py`` expects when you pass ``--priors``.


Training with custom code
-------------------------

The training script is thin. You can reproduce its core logic directly:

.. code-block:: python

   from torch.utils.data import DataLoader
   from differential_skeletons import build_layer
   from differential_skeletons.fitting import (
       FrameDataset,
       JointLimitTrainer,
       PoseVAE,
       PoseVAETrainer,
   )

   model = build_layer("spinetrack")
   dataset = FrameDataset.from_npz("dataset.npz", expected_num_joints=model.NUM_JOINTS)
   loader = DataLoader(dataset, batch_size=512, shuffle=True)

   joint_limit_prior = JointLimitTrainer(model=model).fit_from_loader(loader)

   pose_prior = PoseVAE(num_joints=model.NUM_BODY_JOINTS, latent_dim=32)
   trainer = PoseVAETrainer(pose_prior, model=model)
   state = trainer.train_epoch(loader)


Training a temporal prior
-------------------------

The package also exposes:

- ``TemporalPrior``
- ``TemporalPriorTrainer``

These are intended for sequence data and use ``prepare_sequence_batch(...)``.
There is no standalone example script yet, but the public training classes are
available for custom pipelines.


Practical advice
----------------

- Match ``--skeleton`` to the actual joint ordering of the dataset.
- Start with ``build_layer(...)`` rather than a parameter-owning model.
- If you see numerical issues, fail-fast checks in the trainer will now raise an
  error instead of silently continuing with ``nan`` losses.
