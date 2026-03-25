Modern Research Patterns
========================

Many recent methods can be read as recurring design patterns rather than as
single fixed architectures. DifferentialSkeletons can act as the articulated
layer inside several of those patterns.

Pattern 1: Optimization-based fitting with learned priors
---------------------------------------------------------

This is the family associated with SMPLify, SMPLify-X, and VPoser-style pose
priors.

The package analogue is:

- ``SkeletalFitter`` for the inverse problem
- ``PoseVAE`` for a low-dimensional pose manifold
- ``JointLimitPrior`` for range regularization
- ``TemporalPrior`` and ``MotionSmoothnessPrior`` for sequence refinement

Representative example scripts:

- ``examples/fit_demo.py``
- ``examples/pseudo_label_bootstrap.py``
- ``examples/spine_sequence_fit.py``

Pattern 2: Hybrid analytic-neural IK
------------------------------------

Methods such as HybrIK separate the easier geometry from the harder latent
ambiguities. The network predicts information that is easy to supervise, and an
analytic or semi-analytic stage recovers articulated rotations.

The package analogue is:

- predict 3D joints from 2D or image features
- recover rotations with ``estimate_rotations_from_joints``
- optionally refine in a fitter or prior space

Representative example script:

- ``examples/hybrik_style_hybrid_ik.py``

Pattern 3: Temporal lifting from keypoint sequences
---------------------------------------------------

VideoPose3D, PoseFormer, PoseFormerV2, and MotionBERT all show that temporal
context matters for 2D-to-3D lifting. The architecture can vary, but the core
idea is stable: use a long sequence of 2D keypoints to recover a cleaner 3D
motion sequence.

The package analogue is:

- predict skeletal pose parameters or joints per frame
- decode them through a rig layer
- supervise in 3D joint space, reprojection space, or both
- add temporal priors or smoothness when fitting or refining

Representative example scripts:

- ``examples/temporal_lifting_videopose3d.py``
- ``examples/sequence_denoising_and_smoothing.py``

Pattern 4: Dataset bootstrapping and pseudo-label generation
------------------------------------------------------------

A common modern recipe is: fit a structured model to weak supervision, keep the
high-confidence results, and train a direct regressor from those pseudo-labels.

DifferentialSkeletons supports this pattern directly:

- fit 2D detections with ``SkeletalFitter``
- save fitted 3D joints or rotations
- train a student network on the fitted outputs

Representative example script:

- ``examples/pseudo_label_bootstrap.py``

Pattern 5: Cross-skeleton distillation and retargeting
------------------------------------------------------

Many practical systems must move between skeleton conventions. A detector may
produce COCO joints, while a downstream model expects HALPE or Human3.6M.

The package analogue is:

- map shared joints between source and target layouts
- fit the target articulated rig to the source observations
- use the fitted target motion as supervision or exported motion data

Representative example script:

- ``examples/retarget_between_skeletons.py``
