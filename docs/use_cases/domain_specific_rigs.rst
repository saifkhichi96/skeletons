Domain-Specific Rigs
====================

Different rigs expose different kinds of priors and fitting problems.

Body-only rigs
--------------

Examples: Human3.6M, COCO, MPII, HALPE26.

These are the most natural place to start with:

- 2D-to-3D lifting
- motion denoising
- VAE pose priors
- monocular or multi-view fitting
- cross-skeleton conversion between dataset conventions

Whole-body rigs
---------------

Examples: COCO WholeBody, HALPE FullBody.

Whole-body rigs are useful when you need a unified representation across body,
feet, hands, and face keypoints. They are particularly relevant for dataset
conversion, structured pseudo-label generation, and optimization-based
whole-body fitting.

Hand rigs
---------

Example: Hand21.

Hand rigs are a natural fit for:

- articulated fitting from 2D landmarks
- hand-pose prior learning
- retargeting into animation or teleoperation pipelines
- analytical warm starts from recovered 3D joints

Face rigs
---------

Example: Face68.

Face68 is less anatomically articulated than the body or hand rigs, so the most
useful applications are:

- structured landmark fitting
- landmark cleanup and sequence smoothing
- dataset conversion and normalization
- weak articulation experiments for jaw/neck/head coupled pipelines

Spine rigs
----------

Example: SpineTrack.

SpineTrack is one of the strongest arguments for a skeleton-first library. A
mesh model is often unnecessary when the task is vertebral or centerline motion
analysis.

Typical applications include:

- sequence fitting from weak-perspective 2D observations
- temporal denoising of spinal motion estimates
- prior learning from domain-specific 3D spine data
- biomechanics-aware preprocessing before downstream analysis

Representative example script:

- ``examples/spine_sequence_fit.py``
