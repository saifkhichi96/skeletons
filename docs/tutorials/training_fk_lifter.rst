Tutorial: Train a Small 2D-to-3D Lifter
=======================================

Goal
----

This tutorial explains the purpose of ``train_fk_lifter.py`` and how it relates
to the rest of the package.


What the script does
--------------------

The script trains a small multilayer perceptron that maps 2D keypoints to a
predicted 3D full pose tensor. Supervision is applied through
``ForwardKinematicsLoss``, which compares the joints produced by the articulated
model against target 3D joints.

This makes it a useful demonstration of the package as a differentiable
articulation module inside another learning system.


Run the script
--------------

.. code-block:: bash

   python examples/train_fk_lifter.py --skeleton human36m

Key options:

- ``--skeleton`` selects the rig used to generate synthetic data and to compute FK
- ``--train-samples`` and ``--val-samples`` control dataset size
- ``--hidden-dim`` controls model width
- ``--work-dir`` overrides the default output directory


Outputs
-------

Like the prior training script, this example writes to:

.. code-block:: text

   work_dirs/train_fk_lifter/<run_name>/

The directory contains:

- ``config.json``
- ``metrics.json``
- ``epoch_<n>.pth``
- ``last_checkpoint``


Why this matters
----------------

Even if you do not plan to use this exact script, it shows a core design goal of
Skeletons:

- articulated skeletons are first-class differentiable modules
- kinematics can be embedded inside learning systems
- training can happen against joints rather than meshes

For most users, the more immediately relevant workflows are still the fitting and
prior-training tutorials, but this script is a compact example of FK-based
learning with the package.
