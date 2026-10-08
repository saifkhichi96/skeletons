Models and Rigs
===============

Skeletal rigs
--------------------------------------

Each supported skeleton is defined as a rig: a named articulated tree with:

- joint names
- parent indices
- canonical rest offsets
- a single root joint

A skeletal rig is internally represented as a ``SkeletonSpec`` object with the following attributes:

- ``joint_names``
- ``parents``
- ``child_body_indices``
- ``body_nodes``
- ``joints``
- ``topological_order``
- ``root_index``

The root is treated as a free body, and non-root links are articulated relative
to their parent. Use ``get_spec(name)`` to retrieve the underlying ``SkeletonSpec`` for a rig.

Supported skeletons
~~~~~~~~~~~~~~~~~~~

Use ``list_supported_skeletons()`` to see all built-in skeletons. This includes
``human36m``, ``coco``, ``mpii``, ``halpe26``, ``hand21``, ``face68``,
``halpe_fullbody``, ``coco_wholebody``, and ``spinetrack``.

Differentiable models
-------------------------------------------

Two main classes mirroring `smplx`_ semantics represent the rigs as differentiable models:

.. _smplx: https://github.com/vchoutas/smplx

``SkeletalModel``
   Owns registered parameters such as ``global_orient``, ``body_pose``,
   ``scales``, and ``transl`` when those are created during construction.

``SkeletalModelLayer``
   Disables those registered parameters by default and expects values to be
   provided explicitly during each forward call.

Use the following factory functions to construct models and layers:

.. code-block:: python

   from skeletons import build_layer, create

   layer = build_layer("spinetrack")            # for a stateless layer
   model = create("spinetrack", batch_size=8)   # for a parameter-owning model

In practice, use:

- ``build_layer(...)`` for fitting and inference
- ``create(...)`` when you want the module itself to own learnable state


Batch semantics
---------------

Created parameters always keep a leading batch dimension, even when
``batch_size=1``. For example:

.. code-block:: python

   from skeletons import create

   model = create("human36m", batch_size=2)
   print(model.global_orient.shape)  # [2, 3]
   print(model.body_pose.shape)      # [2, J - 1, 3]
   print(model.scales.shape)         # [2, J, 3]
   print(model.transl.shape)         # [2, 3]

This makes the package easier to use in batched optimization and keeps behavior
close to mesh-model libraries that always allocate batched default parameters.
