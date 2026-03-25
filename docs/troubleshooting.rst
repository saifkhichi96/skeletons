Troubleshooting
===============

Dataset and skeleton mismatch
-----------------------------

Symptom:

- fitting behaves badly
- priors train poorly
- reconstructed poses look implausible

Check:

- the dataset joint order really matches ``--skeleton``
- the dataset has the same number of joints as the selected rig


Non-finite losses during prior training
---------------------------------------

The trainer now raises immediately if it sees a non-finite loss or gradient.
Typical causes are:

- invalid input data
- numerical instability introduced by a custom change
- a mismatch between dataset semantics and skeleton choice

When debugging:

1. Inspect ``FrameDataset.from_npz(...)`` output for finiteness.
2. Run ``prepare_frame_batch(...)`` on a subset and check the resulting
   ``body_pose_rot6d``.
3. Reduce batch size and verify the problem is reproducible.


Using checkpoints from ``work_dirs``
------------------------------------

``examples/fit_demo.py --priors`` accepts:

- a checkpoint file
- a ``last_checkpoint`` file
- a whole run directory

If loading fails, verify that the run directory actually contains
``last_checkpoint`` and that the referenced ``.pth`` file still exists.


Choosing between ``create`` and ``build_layer``
-----------------------------------------------

Use ``build_layer(...)`` when:

- you are fitting
- you are writing inference code
- you want to pass explicit tensors on every forward call

Use ``create(...)`` when:

- you want registered parameters inside the module
- you want to optimize the module's internal state directly


Body scales vs body-shape coefficients
--------------------------------------

The package does not implement SMPL-style shape coefficients. If you are looking
for ``betas``, the corresponding concept here is articulated per-body ``scales``.
