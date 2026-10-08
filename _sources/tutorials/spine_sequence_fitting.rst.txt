Tutorial: Spine Sequence Fitting
================================

Goal
----

Show that the same fitting machinery also applies to a domain-specific rig such
as SpineTrack.

Runnable example
----------------

.. literalinclude:: ../../examples/spine_sequence_fit.py
   :language: python
   :caption: examples/spine_sequence_fit.py

Why this matters
----------------

It demonstrates that the library is not only a body-pose package. The same
articulated abstractions can support specialized rigs such as the spine.
The example uses ``fit_2d_joint_sequences(...)`` so domain-specific rigs can
share the same 2D sequence reconstruction workflow as human-pose rigs.
