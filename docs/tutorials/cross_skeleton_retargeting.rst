Tutorial: Cross-Skeleton Retargeting
====================================

Goal
----

Retarget motion from one joint convention to another by fitting the target rig
to the source rig's shared joints.

Runnable example
----------------

.. literalinclude:: ../../examples/retarget_between_skeletons.py
   :language: python
   :caption: examples/retarget_between_skeletons.py

Why this matters
----------------

This is one of the most practical uses of the library:

- unify incompatible detector outputs
- create training targets for another skeleton convention
- export motion into the rig expected by a downstream application
