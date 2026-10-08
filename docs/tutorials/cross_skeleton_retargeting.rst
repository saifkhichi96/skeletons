Tutorial: Cross-Skeleton Retargeting
====================================

Goal
----

Retarget motion from one joint convention to another by fitting the target rig
to the source rig's shared joints.

Core API
--------

The retargeting API has two layers:

- ``build_joint_mapping(...)`` finds source and target joints with compatible
  names, including common aliases such as ankle/foot.
- ``retarget_skeleton(...)`` builds sparse target-layout observations and fits
  the target skeleton to those observations.

.. code-block:: python

   from skeletons import build_layer, retarget_skeleton

   source_model = build_layer("coco")
   target_model = build_layer("human36m")
   source_joints = source_model().joints

   result = retarget_skeleton(
       source_joints,
       source_model=source_model,
       target_model=target_model,
       num_iters=120,
       optimize_scales=True,
   )

   print(result.fitting_result.model_output.joints.shape)
   print(result.shared_joint_mpjpe)

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
