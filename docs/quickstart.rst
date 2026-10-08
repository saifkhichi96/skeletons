Quickstart
==========

Construct a rig
---------------

The simplest entry point is ``build_layer()``, which returns a parameter-free
model layer suitable for fitting and inference:

.. code-block:: python

   from skeletons import build_layer

   model = build_layer("human36m")
   print(model.spec.name)   # human36m
   print(model.NUM_JOINTS)  # 17

If you accept user-provided names, use ``canonicalize_skeleton_name()`` to
normalize aliases before storing or displaying them:

.. code-block:: python

   from skeletons import (
       canonicalize_skeleton_name,
       list_supported_skeletons,
   )

   print(list_supported_skeletons())
   print(canonicalize_skeleton_name("halpe-fullbody"))  # halpe_fullbody

Use ``list_supported_skeletons(include_aliases=True)`` if you need every public
input accepted by ``build_layer()``, ``create()``, and ``get_spec()``.


Run forward kinematics
----------------------

.. code-block:: python

   import torch
   from skeletons import build_layer

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
   from skeletons import build_layer
   from skeletons.fitting import SkeletalFitter

   model = build_layer("human36m")
   target_joints = torch.randn(1, model.NUM_JOINTS, 3)

   fitter = SkeletalFitter(model=model)
   result = fitter.fit_3d(target_joints, num_iters=100)

   print(result.losses)
   print(result.model_output.joints.shape)


Solve inverse kinematics
-----------------------------

Use IK to find a pose that matches a set of target frame positions.

.. code-block:: python

   from skeletons import build_layer, DampedLeastSquaresIK, keypoint_3d_loss

   model = build_layer("human36m")
   q = model.zero_pose(batch_size=1)
   frames = ["left_wrist", "right_wrist"]
   target = model.frame_positions(q + 0.1, frame_names=frames).detach()
   solution = DampedLeastSquaresIK(model, frames).solve(q, target)
   loss = keypoint_3d_loss(model, solution.q, target, frame_names=frames)

``CyclicCoordinateDescentIK`` and ``GradientIK`` share the same target convention.
Targets have shape ``[B, F, 3]``; optional confidence has shape ``[B, F]``.
``q`` contains non-root rotational DOFs in radians. Supply root placement
separately as a homogeneous transform. Joint targets need no marker setup.

Optional articulation and attached frames use the existing factory:

.. code-block:: python

   from skeletons import MarkerSpec

   model = build_layer(
       "human36m",
       joint_types={"left_elbow": "hinge"},
       joint_axes={"left_elbow": (0.0, 0.0, 1.0)},
       joint_limits={"left_elbow": ((-1.5, 1.5),)},
       markers=(MarkerSpec("tip", "left_wrist", (0.0, 0.05, 0.0)),),
   )
   tip = model.frame_positions(model.zero_pose(), frame_names=["marker:tip"])

Named frames accept joint names or ``joint:``, ``marker:``, and ``contact:``
prefixes. Losses default to joint ordering; cross-rig detector layouts require
an explicit ``frame_mapping``. Configure links with measured masses and COMs
before using physical losses. Contact losses use configured contacts and Y-up
by default, with an explicit ``up_axis`` override.

``get_schema("coco")`` uses the COCO rig's names and ordering. The aliases
``coco17``, ``mpii16``, and ``h36m17`` select ``coco``, ``mpii``, and
``human36m`` respectively, including those rigs' exact joint names.
``mediapipe33`` describes observations without introducing another rig.

``model.export_urdf(path)`` and ``model.export_mjcf(path)`` export the same rig.
Default geometry is illustrative. URDF ball joints expand into three Euler
rotation axes; exported coordinates differ from the model's axis-angle values.
Axis-angle component limits on ball joints cannot be exported faithfully and
raise ``ValueError``. Hinge axes and limits are supported.

Evaluate predictions
--------------------

.. code-block:: python

   from skeletons import (
       mean_per_joint_position_error,
       percentage_of_correct_keypoints,
       procrustes_aligned_mpjpe,
   )

   predicted_joints = result.model_output.joints
   mpjpe = mean_per_joint_position_error(predicted_joints, target_joints)
   pa_mpjpe = procrustes_aligned_mpjpe(predicted_joints, target_joints)
   pck = percentage_of_correct_keypoints(
       predicted_joints,
       target_joints,
       threshold=50.0,
   )

   print(mpjpe, pa_mpjpe, pck)


Retarget between rigs
---------------------

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


Generate synthetic batches
--------------------------

.. code-block:: python

   import torch
   from skeletons import build_layer, generate_random_pose_batch

   model = build_layer("human36m")
   batch = generate_random_pose_batch(
       model,
       batch_size=16,
       pose_std=0.2,
       generator=torch.Generator().manual_seed(0),
   )

   print(batch.full_pose.shape)
   print(batch.joints.shape)

For fitting pipelines, generate a frame dataset with paired 3D targets, 2D
projections, confidences, and camera intrinsics:

.. code-block:: python

   import torch
   from skeletons import (
       build_layer,
       generate_synthetic_fitting_dataset,
   )

   model = build_layer("human36m")
   fitting_data = generate_synthetic_fitting_dataset(
       model,
       num_frames=8,
       pose_std=0.15,
       noise_std=3.0,
       confidence_dropout=0.05,
       generator=torch.Generator().manual_seed(7),
   )

   print(len(fitting_data.frame_dataset))
   print(fitting_data.noisy_joints_2d.shape)


Generate 3D pseudo-labels from 2D detections
--------------------------------------------

``train_pseudo_label_priors()`` and ``fit_2d_pseudo_labels()`` provide the
reusable package path behind the pseudo-label bootstrap example:

.. code-block:: python

   import torch
   from skeletons import (
       build_layer,
       fit_2d_pseudo_labels,
       generate_random_pose_batch,
       make_perspective_camera,
       train_pseudo_label_priors,
   )

   model = build_layer("human36m")
   camera = make_perspective_camera()
   bootstrap = generate_random_pose_batch(
       model,
       batch_size=32,
       pose_std=0.15,
       generator=torch.Generator().manual_seed(0),
   )
   priors = train_pseudo_label_priors(
       bootstrap.joints,
       model=model,
       epochs=2,
   )
   detections_2d = camera.project(bootstrap.joints)
   pseudo = fit_2d_pseudo_labels(
       detections_2d,
       camera,
       model=model,
       pose_prior=priors.pose_prior,
       joint_limit_prior=priors.joint_limit_prior,
       num_iters=80,
   )

   print(pseudo.joints_3d.shape)
   print(pseudo.mean_reprojection_error)


Denoise motion sequences
------------------------

Use ``train_temporal_prior()`` and ``denoise_joint_sequences()`` when the input
contains a time axis:

.. code-block:: python

   from skeletons import (
       build_layer,
       denoise_joint_sequences,
       generate_random_walk_sequences,
       train_temporal_prior,
   )

   model = build_layer("human36m")
   train = generate_random_walk_sequences(
       model,
       batch_size=16,
       seq_len=12,
       pose_step_std=0.03,
       transl_step_std=2.0,
       depth=2500.0,
   )
   temporal = train_temporal_prior(train.joints, model=model, epochs=2)
   result = denoise_joint_sequences(
       train.joints,
       model=model,
       temporal_prior=temporal.temporal_prior,
       num_iters=80,
   )

   print(result.joints_3d.shape)
   print(result.losses)


Train priors
------------

The core training path for priors is:

1. Load a frame dataset with ``FrameDataset.from_npz(...)``.
2. Fit a ``JointLimitPrior`` from inverse-kinematics-prepared poses.
3. Train a ``PoseVAE`` over non-root joint rotations in 6D form.
4. Save the resulting checkpoint with ``save_fitting_prior_checkpoint(...)``.
5. Reload it with ``load_fitting_prior_checkpoint(...)`` and call
   ``bundle.make_fitter(model=...)`` when fitting new data.

The higher-level walkthrough for this appears in :doc:`tutorials/training_priors`.
