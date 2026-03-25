Quickstart
==========

Construct a rig
---------------

The simplest entry point is ``build_layer()``, which returns a parameter-free
model layer suitable for fitting and inference:

.. code-block:: python

   from differential_skeletons import build_layer

   model = build_layer("human36m")
   print(model.spec.name)   # human36m
   print(model.NUM_JOINTS)  # 17

Supported rig names are available as ``SUPPORTED_SKELETONS``.


Run forward kinematics
----------------------

.. code-block:: python

   import torch
   from differential_skeletons import build_layer

   model = build_layer("human36m")

   global_orient = torch.zeros(4, 3)
   body_pose = torch.zeros(4, model.NUM_BODY_JOINTS, 3)
   transl = torch.zeros(4, 3)

   output = model(
       global_orient=global_orient,
       body_pose=body_pose,
       transl=transl,
       return_global_rotations=True,
   )

   print(output.joints.shape)
   print(output.global_rotations.shape)

The model accepts:

- split pose inputs through ``global_orient`` and ``body_pose``
- full pose input through ``full_pose``
- multiple pose representations such as axis-angle, quaternion, 6D, and matrices
- body scales through ``scales``


Fit to 3D joints
----------------

.. code-block:: python

   import torch
   from differential_skeletons import build_layer
   from differential_skeletons.fitting import SkeletalFitter

   model = build_layer("human36m")
   target_joints = torch.randn(1, model.NUM_JOINTS, 3)

   fitter = SkeletalFitter(model=model)
   result = fitter.fit_3d(target_joints, num_iters=100)

   print(result.losses)
   print(result.model_output.joints.shape)


Train priors
------------

The core training path for priors is:

1. Load a frame dataset with ``FrameDataset.from_npz(...)``.
2. Fit a ``JointLimitPrior`` from inverse-kinematics-prepared poses.
3. Train a ``PoseVAE`` over non-root joint rotations in 6D form.
4. Save the resulting checkpoint and reuse it during fitting.

The higher-level walkthrough for this appears in :doc:`tutorials/training_priors`.
