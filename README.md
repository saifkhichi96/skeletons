# DifferentialSkeletons

A lightweight PyTorch package for differentiable skeletal kinematics. It is intended for use cases that need a reusable articulated-body module without a skinned surface model: FK-based losses, pose regularization, skeleton retargeting, canonical rest-pose animation, optimization over OpenSim-style body scale factors, and skeleton-native prior learning.

The public API is deliberately close to the interaction style of `smplx`: models can own default parameters (`global_orient`, `body_pose`, `bone_scales`, `transl`) but explicit per-call tensors always override the stored state.

## What is included

- A generic `SkeletalModel` base class with differentiable forward kinematics.
- Split-pose and full-pose inputs.
- Axis-angle, quaternion, 6D, and rotation-matrix pose representations.
- Optional learnable or frozen OpenSim-style per-body scale factors.
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
- Generic inverse-kinematics helpers for recovering local rotations from 3D joint trajectories.
- A concrete Human3.6M prior-learning and fitting stack:
  - empirical joint-limit prior
  - VAE body-pose prior in rotation-6D space
  - GRU temporal prior
  - single-frame and sequence fitting from 3D or 2D joints

## Installation

```bash
pip install differential_skeletons
```

For editable development:

```bash
pip install -e .[dev]
```

## Quick start

```python
import torch
import differential_skeletons as ds

model = ds.Human36MModel(
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

## H36M prior learning and fitting

```python
import torch
from differential_skeletons.h36m import (
    H36MPoseVAE,
    H36MJointLimitTrainer,
    H36MFitter,
    PerspectiveCamera,
)

joints_3d = torch.randn(32, 17, 3)
prior = H36MPoseVAE(num_joints=16, latent_dim=32)
limits = H36MJointLimitTrainer().fit_from_joints(joints_3d)
fitter = H36MFitter(pose_prior=prior, joint_limit_prior=limits)

camera = PerspectiveCamera(
    fx=torch.tensor(1000.0),
    fy=torch.tensor(1000.0),
    cx=torch.tensor(512.0),
    cy=torch.tensor(512.0),
)
```

The Human3.6M stack expects dataset-style arrays with the official 17-joint ordering and can be driven either from raw tensors or from `.npz` files loaded by `H36MFrameDataset` / `H36MSequenceDataset`.

## Notes on the canonical rest pose

`differential_skeletons` does **not** ship learned dataset-specific body shapes. The provided `rest_offsets` are canonical template offsets designed to be useful and stable in practice:

- Full-body rigs use a neutral adult T-pose.
- Hand rigs use an open-hand neutral pose.
- Face rigs use a neutral frontal landmark template.
- `bone_scales` follow an OpenSim-like body-scaling convention: each body can carry an isotropic scalar or an anisotropic `x/y/z` scale, and child joint offsets are scaled in the parent body's local frame.
- Passing `[..., J - 1]` or `[..., J]` scale tensors remains supported as isotropic shorthand; passing `[..., J - 1, 3]` or `[..., J, 3]` enables explicit per-axis body scaling.

This makes the package suitable for FK losses and optimization, but the templates should be treated as canonical priors rather than dataset ground truth.

## Important note on H36M pose priors

Human3.6M keypoints constrain articulated pose well enough for skeleton-native priors and fitting, but they do not uniquely identify arbitrary twist about each bone axis. The provided inverse-kinematics preparation therefore estimates observable joint orientation from outgoing child vectors and resolves underdetermined twist with a minimal-twist convention. The H36M prior stack is therefore best understood as a prior over **observable skeletal articulation** rather than a full anatomical DOF model.

## Attribution

The dataset keypoint orderings and skeleton connectivity were cross-checked against public metadata from OpenMMLab MMPose, Pose2Sim, and the SpinePose / SpineTrack metainfo. This package re-implements the kinematic logic and does not include any mesh model or licensed SMPL-X assets.
