Tutorial: Pseudo-Label Bootstrapping from 2D Detections
=======================================================

Goal
----

Fit 2D detections with a skeletal model, save the recovered 3D motion, and use
it as structured pseudo-labels for downstream training.

Runnable example
----------------

.. literalinclude:: ../../examples/pseudo_label_bootstrap.py
   :language: python
   :caption: examples/pseudo_label_bootstrap.py

Why this matters
----------------

This is the skeleton-native analogue of fitting-first pipelines that bootstrap
pseudo-labels from weak supervision.
