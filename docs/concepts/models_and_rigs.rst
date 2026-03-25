Models and Rigs
===============

What a rig is in DifferentialSkeletons
--------------------------------------

Each supported skeleton is defined as a rig: a named articulated tree with:

- joint names
- parent indices
- canonical rest offsets
- a single root joint

The package ships several ready-made rigs, including ``human36m``, ``coco``,
``mpii``, ``hand21``, ``face68``, ``halpe_fullbody``, ``coco_wholebody``, and
``spinetrack``.

You normally access them through:

- ``build_layer(name)`` for a stateless layer
- ``create(name, ...)`` for a parameter-owning model
- ``get_spec(name)`` if you want the underlying ``SkeletonSpec``


``SkeletalModel`` vs ``SkeletalModelLayer``
-------------------------------------------

The package mirrors the rough interaction style of ``smplx``:

``SkeletalModel``
   Owns registered parameters such as ``global_orient``, ``body_pose``,
   ``scales``, and ``transl`` when those are created during construction.

``SkeletalModelLayer``
   Disables those registered parameters by default and expects values to be
   provided explicitly during each forward call.

In practice:

- use ``build_layer(...)`` or ``*ModelLayer`` classes for fitting and inference
- use ``create(...)`` or ``*Model`` classes when you want the module itself to
  own learnable state


Batch semantics
---------------

Created parameters always keep a leading batch dimension, even when
``batch_size=1``. For example:

.. code-block:: python

   from differential_skeletons import create

   model = create("human36m", batch_size=2)
   print(model.global_orient.shape)  # [2, 3]
   print(model.body_pose.shape)      # [2, J - 1, 3]
   print(model.scales.shape)         # [2, J, 3]
   print(model.transl.shape)         # [2, 3]

This makes the package easier to use in batched optimization and keeps behavior
close to mesh-model libraries that always allocate batched default parameters.


Articulated structure
---------------------

Internally, a model is represented as an articulated tree rather than a flat
parent index list alone. The public model object exposes:

- ``joint_names``
- ``parents``
- ``child_body_indices``
- ``body_nodes``
- ``joints``
- ``topological_order``
- ``root_index``

The root is treated as a free body, and non-root links are articulated relative
to their parent.


When to use the factory API
---------------------------

The recommended public surface is the factory API rather than importing a rig
module directly:

.. code-block:: python

   from differential_skeletons import build_layer, create

   layer = build_layer("spinetrack")
   model = create("spinetrack", batch_size=8)

This keeps code independent of internal rig module organization and is the best
choice for user code, scripts, and notebooks.
