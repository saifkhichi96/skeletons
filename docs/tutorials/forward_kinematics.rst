Tutorial: Forward Kinematics
============================

Goal
----

This tutorial shows the minimal workflow for constructing a skeleton, posing it,
and inspecting the outputs returned by the forward pass.


Step 1: build a layer
---------------------

.. code-block:: python

   import torch
   from differential_skeletons import build_layer

   model = build_layer("spinetrack")

Use ``build_layer(...)`` unless you specifically want registered parameters.


Step 2: prepare pose inputs
---------------------------

Axis-angle is the most direct representation to start with:

.. code-block:: python

   batch_size = 2
   global_orient = torch.zeros(batch_size, 3)
   body_pose = torch.zeros(batch_size, model.NUM_BODY_JOINTS, 3)
   transl = torch.zeros(batch_size, 3)

You can also use:

- ``full_pose`` instead of split pose tensors
- ``rot6d`` if you are fitting or learning rotations
- ``rotmat`` if you already have rotation matrices


Step 3: run the model
---------------------

.. code-block:: python

   output = model(
       global_orient=global_orient,
       body_pose=body_pose,
       transl=transl,
       return_full_pose=True,
       return_local_rotations=True,
       return_global_rotations=True,
       return_scaled_offsets=True,
   )

The returned object is a ``ModelOutput`` dataclass. Common fields are:

- ``output.joints``
- ``output.global_orient``
- ``output.body_pose``
- ``output.full_pose``
- ``output.local_rotations``
- ``output.global_rotations``
- ``output.scales``
- ``output.transl``


Step 4: inspect rest shape and body scaling
-------------------------------------------

.. code-block:: python

   rest = model.rest_joints()
   bigger = model.rest_joints(scales=torch.ones(model.NUM_JOINTS, 3) * 1.1)

This is useful when you want to reason about the unposed skeleton separately
from articulation.


Step 5: use a parameter-owning model if needed
----------------------------------------------

.. code-block:: python

   from differential_skeletons import create

   stateful = create("spinetrack", batch_size=4)
   output = stateful()

This is the right pattern when you want to optimize the model's internal pose,
translation, or scales directly as registered parameters.
