Tutorial: Hybrid Analytic-Neural IK
===================================

Goal
----

Demonstrate a HybrIK-style design pattern in which a network predicts 3D joints
and an analytic stage recovers articulated rotations.

Runnable example
----------------

.. literalinclude:: ../../examples/hybrik_style_hybrid_ik.py
   :language: python
   :caption: examples/hybrik_style_hybrid_ik.py

Why this matters
----------------

This pattern is useful when joint supervision is easier to obtain than full
rotation supervision. It lets you train the network on joint geometry while
still recovering a structured articulated state.
