# skelix

`skelix` is a lightweight PyTorch package for differentiable skeletal kinematics. It is intended for use cases that need a reusable articulated-body module without a skinned surface model: FK-based losses, pose regularization, skeleton retargeting, canonical rest-pose animation, and optimization over bone-length multipliers.

The public API is deliberately close to the interaction style of `smplx`: models can own default parameters (`global_orient`, `body_pose`, `bone_scales`, `transl`) but explicit per-call tensors always override the stored state.

## What is included

- A generic `SkeletalModel` base class with differentiable forward kinematics.
- Split-pose and full-pose inputs.
- Axis-angle, quaternion, 6D, and rotation-matrix pose representations.
- Optional learnable or frozen per-bone length multipliers.
- Canonical template rigs for:
  - COCO
  - MPII
  - Human3.6M
  - HALPE26
  - Hand21
  - Face68
  - HALPE FullBody
  - COCO WholeBody
  - SpineTrack
- A simple `ForwardKinematicsLoss` module.

## Installation

```bash
pip install skelix
```

For editable development:

```bash
pip install -e .[dev]
```

## Quick start

```python
import torch
from skelix import Human36MModel

model = Human36MModel(
    create_global_orient=False,
    create_body_pose=False,
    create_bone_scales=False,
    create_transl=False,
)

B = 8
body_pose = torch.zeros(B, model.num_joints - 1, 3)  # axis-angle
global_orient = torch.zeros(B, 3)
transl = torch.zeros(B, 3)

out = model(
    global_orient=global_orient,
    body_pose=body_pose,
    transl=transl,
    return_global_rotations=True,
    return_scaled_offsets=True,
)

print(out.joints.shape)           # [B, J, 3]
print(out.global_rotations.shape) # [B, J, 3, 3]
```

## Using a fixed morphology

```python
import torch
from skelix import Halpe26Model

model = Halpe26Model(
    create_bone_scales=True,
    bone_scales=torch.ones(25),
    learn_bone_scales=False,
)

model.freeze("bone_scales")
```

## Using a full pose tensor

This is often more convenient when the root joint is not the first joint in the dataset ordering.

```python
import torch
from skelix import Face68Model

model = Face68Model(create_global_orient=False, create_body_pose=False)
full_pose = torch.zeros(model.num_joints, 3)
out = model(full_pose=full_pose, return_full_pose=True)
```

## Notes on the canonical rest pose

`skelix` does **not** ship learned dataset-specific body shapes. The provided `rest_offsets` are canonical template offsets designed to be useful and stable in practice:

- Full-body rigs use a neutral adult T-pose.
- Hand rigs use an open-hand neutral pose.
- Face rigs use a neutral frontal landmark template.
- `bone_scales` are multiplicative per-bone length factors, not SMPL-style latent shape coefficients.

This makes the package suitable for FK losses and optimization, but the templates should be treated as canonical priors rather than dataset ground truth.

## Notes on hierarchy choices

Some datasets do not define a clean articulated tree on their own. In those cases `skelix` uses a documented kinematic proxy:

- COCO and COCO WholeBody are rooted at the **nose**, because those keypoint sets do not provide an explicit pelvis or neck root.
- Face68 is implemented as a pseudo-kinematic tree following the standard 68-point ordering.
- SpineTrack includes a few torso landmarks that are not fully specified as a tree in the metainfo; those are attached through a canonical spine-and-clavicle hierarchy.

## API summary

### Base model

```python
from skelix import SkeletalModel
```

### Concrete models

```python
from skelix import (
    CocoModel,
    MPIIModel,
    Human36MModel,
    Halpe26Model,
    Hand21Model,
    Face68Model,
    HalpeFullBodyModel,
    CocoWholeBodyModel,
    SpineTrackModel,
)
```

### Forward signature

```python
output = model(
    global_orient=None,
    body_pose=None,
    full_pose=None,
    bone_scales=None,
    transl=None,
    pose_repr=None,
    return_full_pose=False,
    return_local_rotations=False,
    return_global_rotations=False,
    return_local_transforms=False,
    return_global_transforms=False,
    return_scaled_offsets=False,
    return_dict=False,
)
```

### Output fields

`SkeletalOutput` provides:

- `joints`
- `full_pose`
- `global_orient`
- `body_pose`
- `bone_scales`
- `transl`
- `scaled_offsets`
- `local_rotations`
- `global_rotations`
- `local_transforms`
- `global_transforms`
- `joint_names`

## Attribution

The dataset keypoint orderings and skeleton connectivity were cross-checked against public metadata from OpenMMLab MMPose, Pose2Sim, and the SpinePose / SpineTrack metainfo. This package re-implements the kinematic logic and does not include any mesh model or licensed SMPL-X assets.
