Tutorial: Sequence Denoising with Temporal Priors
=================================================

Goal
----

Clean a noisy 3D motion sequence by fitting the rig under temporal and
smoothness constraints.

Runnable example
----------------

.. literalinclude:: ../../examples/sequence_denoising_and_smoothing.py
   :language: python
   :caption: examples/sequence_denoising_and_smoothing.py

Key idea
--------

A learned temporal prior and an explicit smoothness prior are complementary:

- the temporal prior captures dataset-like motion transitions
- the smoothness prior penalizes frame-to-frame jitter and acceleration spikes
