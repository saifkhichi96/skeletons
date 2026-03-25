Release Notes: v1.0.0
=====================

This release turns the project from an exploratory package into a public-facing
library with a documented application surface.

Highlights
----------

- generic articulated body and fitting abstractions
- multiple public rig conventions under a common API
- reusable fitting priors and training helpers
- stateful ``*Model`` and stateless ``*ModelLayer`` variants
- shared dataset loaders for frame and sequence fitting data
- richer public examples showing modern research patterns
- Sphinx documentation expanded beyond API coverage into application guidance

New showcase examples
---------------------

- ``examples/temporal_lifting_videopose3d.py``
- ``examples/hybrik_style_hybrid_ik.py``
- ``examples/sequence_denoising_and_smoothing.py``
- ``examples/retarget_between_skeletons.py``
- ``examples/pseudo_label_bootstrap.py``
- ``examples/spine_sequence_fit.py``

What this release emphasizes
----------------------------

The library is intentionally positioned as a skeleton-first alternative for
problems where articulated structure matters but a mesh is unnecessary or
undesirable. That includes research code, fitting pipelines, structural losses,
retargeting, pseudo-label generation, and domain-specific articulated systems
such as spine motion.
