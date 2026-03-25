Overview: Where a Skeleton-First Library Fits
=============================================

DifferentialSkeletons sits between raw keypoints and full mesh models.
That makes it useful in a surprisingly wide range of pipelines.

Core categories
---------------

Learning with structural supervision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the rig as a differentiable layer inside a network and supervise through
joint-space losses, FK losses, reprojection losses, or consistency losses.

Examples:

- 2D-to-3D lifting with ``ForwardKinematicsLoss``
- direct rotation prediction with joint supervision
- self-supervised 2D back-projection loops
- cross-dataset distillation between incompatible skeleton layouts

Optimization-based fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``SkeletalFitter`` to solve inverse problems from partial or noisy
observations.

Examples:

- monocular 2D fitting with a camera model
- 3D fitting to noisy pose detector outputs
- sequence fitting with temporal priors and smoothness
- partial-joint fitting under confidence weights or missing joints

Prior learning
~~~~~~~~~~~~~~

Learn priors directly in skeletal pose space rather than on a skinned mesh.

Examples:

- VAE body-pose priors
- joint-range priors from motion corpora
- temporal motion priors for sequence repair
- rig-specific priors for hand, spine, or whole-body layouts

Motion cleanup and retargeting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because the package operates directly on articulated trees, it is also useful
for manipulating motion data itself.

Examples:

- denoising jittery motion capture
- converting between Human3.6M, COCO, HALPE, and other layouts
- canonicalizing sequences before downstream learning
- fitting a target rig to the output of another detector or estimator

Data generation and pseudo-labeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A fitted articulated model gives you a route from 2D detections to structured
3D pseudo-labels.

Examples:

- create 3D pseudo-labels from 2D datasets
- bootstrap temporal training data with fitter outputs
- generate consistent body-scale estimates across samples
- convert detector outputs into rotation-space labels for prior training

Animation, simulation, and robotics-adjacent workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same FK layer that supports vision pipelines also supports motion-centric
applications.

Examples:

- retarget motion into game-engine rigs
- export fitted joint trajectories to robotics or simulation pipelines
- bridge human motion data into articulated control or IK systems
- create simplified articulated surrogates when meshes are unnecessary

What the package does *not* try to be
-------------------------------------

DifferentialSkeletons is not a replacement for every mesh model. It does not
model skinning, self-contact geometry, silhouettes, or surface-based losses.
Its strength is the opposite: it gives you a light, transparent, rig-native
layer for applications where joint-space structure matters more than surface
geometry.
