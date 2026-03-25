# DifferentialSkeletons

**v1.0.0**

DifferentialSkeletons is a PyTorch package for articulated skeletons. It gives you differentiable forward kinematics, inverse-kinematics-style initialization, fitting utilities for 2D or 3D keypoints, and trainable skeletal priors without requiring a skinned mesh model.

If you already know `smplx`, the intended feel is similar:

- `*Model` classes own default parameters such as pose, translation, and scales.
- `*ModelLayer` classes are stateless and expect inputs on each forward pass.

## Install

From a local checkout:

```bash
pip install -e .
```

If you want the interactive playground too:

```bash
pip install -e .[playground]
```

For development and tests:

```bash
pip install -e .[dev]
```

For the Sphinx documentation toolchain:

```bash
pip install -e .[docs]
```

For development, tests, and the playground together:

```bash
pip install -e .[dev,playground]
```

If you want development tools, docs, and the playground together:

```bash
pip install -e .[dev,docs,playground]
```

## Typical use cases

- FK-supervised 2D-to-3D lifting
- optimization-based 2D and 3D fitting
- sequence denoising and temporal refinement
- joint-limit, VAE, and temporal prior training
- cross-skeleton retargeting between public layouts
- pseudo-label generation from weak 2D supervision
- rig-specific pipelines for body, hand, face, whole-body, and spine

## Documentation

User documentation lives under [`docs/`](/workspace/projects/skelix/docs) and is
structured as a Sphinx site for Read the Docs.

Build it locally with:

```bash
sphinx-build -b html docs docs/_build/html
```

or:

```bash
make -C docs html
```

## 1. Pick a rig

The simplest entry point is `build_layer()`:

```python
from differential_skeletons import build_layer

model = build_layer("human36m")
print(model.spec.name)   # human36m
print(model.NUM_JOINTS)  # 17
```

Supported rig names:

- `human36m`
- `coco`
- `mpii`
- `halpe26`
- `hand21`
- `face68`
- `halpe_fullbody`
- `coco_wholebody`
- `spinetrack`

You can also use concrete classes from the package root, for example `Human36MModel`, `Human36MModelLayer`, `CocoModel`, and `Hand21Model`.

## 2. Run forward kinematics

```python
import torch
from differential_skeletons import build_layer

model = build_layer("human36m")

B = 4
global_orient = torch.zeros(B, 3)
body_pose = torch.zeros(B, model.NUM_BODY_JOINTS, 3)  # axis-angle
transl = torch.zeros(B, 3)

out = model(
    global_orient=global_orient,
    body_pose=body_pose,
    transl=transl,
    return_global_rotations=True,
)

print(out.joints.shape)           # [4, 17, 3]
print(out.global_rotations.shape) # [4, 17, 3, 3]
```

Useful options:

- Pass `full_pose` with shape `[B, J, pose_dim]` if you prefer a single pose tensor.
- Use `pose_repr="axis_angle"`, `pose_repr="quaternion"`, `pose_repr="rot6d"`, or `pose_repr="rotmat"`.
- Pass `scales` to adjust body sizes. Full control uses shape `[B, J, 3]`; isotropic shorthand such as `[B, J]` also works.

## 3. Use a stateful model when you want learnable parameters

`build_layer()` is usually the right choice for fitting and inference. If you want a module with registered parameters, use a concrete `*Model` class or `create()`:

```python
from differential_skeletons import create

model = create("human36m", batch_size=2)
print(model.global_orient.shape)  # [2, 3]
print(model.body_pose.shape)      # [2, J - 1, 3]

out = model()
print(out.joints.shape)           # [2, J, 3]
```

Created parameters always keep the leading batch dimension, including `batch_size=1`.

## 4. Fit a model to 3D joints

```python
import torch
from differential_skeletons import build_layer
from differential_skeletons.fitting import SkeletalFitter

model = build_layer("human36m")
target_joints_3d = torch.randn(1, model.NUM_JOINTS, 3)

fitter = SkeletalFitter(model=model)
result = fitter.fit_3d(target_joints_3d, num_iters=100)

print(result.losses)
print(result.model_output.joints.shape)  # [1, J, 3]
```

If you want the fitter to adapt scales as well, pass `optimize_scales=True`.

## 5. Fit from 2D joints and a camera

```python
import torch
from differential_skeletons import build_layer
from differential_skeletons.fitting import PerspectiveCamera, SkeletalFitter

model = build_layer("human36m")
target_joints_2d = torch.randn(1, model.NUM_JOINTS, 2)

camera = PerspectiveCamera(
    fx=torch.tensor(1000.0),
    fy=torch.tensor(1000.0),
    cx=torch.tensor(512.0),
    cy=torch.tensor(512.0),
)

fitter = SkeletalFitter(model=model)
result = fitter.fit_2d(target_joints_2d, camera, num_iters=150)
print(result.losses)
```

## 6. Load frame or sequence data from `.npz`

Frame-wise data:

```python
from differential_skeletons import build_layer
from differential_skeletons.fitting import FrameDataset, SkeletalFitter

model = build_layer("human36m")
dataset = FrameDataset.from_npz("sample.npz", expected_num_joints=model.NUM_JOINTS)
sample = dataset[0]

fitter = SkeletalFitter(model=model)
result = fitter.fit_3d(sample["joints_3d"].unsqueeze(0))
```

Sequence data uses `SequenceDataset.from_npz(...)` and the same fitter also provides `fit_sequence_3d()` and `fit_sequence_2d()`.

Expected `.npz` keys:

- `joints_3d`: required
- `joints_2d`: optional
- `confidences`: optional
- `fx`, `fy`, `cx`, `cy`: optional camera intrinsics
- `camera_translation`, `camera_rotation`: optional camera extrinsics

Typical shapes:

- Frames: `joints_3d` as `[N, J, 3]`
- Sequences: `joints_3d` as `[N, T, J, 3]`

## 7. Train priors

The fitting package includes a pose VAE, joint-limit prior tools, and training helpers.

Small example:

```python
from differential_skeletons import build_layer
from differential_skeletons.fitting import JointLimitTrainer, PoseVAE

model = build_layer("human36m")
vae = PoseVAE(num_joints=model.NUM_BODY_JOINTS, latent_dim=32)
trainer = JointLimitTrainer(model=model)
```

For end-to-end scripts, use the examples below.

## 8. Use the examples and playground

Install the playground extra first if you have not already:

```bash
pip install -e .[playground]
```

Launch the interactive GUI:

```bash
python playground.py
```

Fit a dataset sample from the command line:

```bash
python examples/fit_demo.py path/to/data.npz --skeleton human36m --mode 3d
```

To use trained priors, pass either the checkpoint file itself or the training
run directory that contains `last_checkpoint`:

```bash
python examples/fit_demo.py path/to/data.npz --skeleton human36m --priors work_dirs/train_prior/human36m_my_dataset
```

Train fitting priors from a dataset:

```bash
python examples/train_prior.py path/to/data.npz --skeleton human36m
```

Both training scripts write configs, metrics, checkpoints, and a `last_checkpoint`
pointer under `work_dirs/<script_name>/<run_name>/` by default.

Train a small 2D-to-3D lifting network with FK supervision:

```bash
python examples/train_fk_lifter.py
```

## Practical Notes

- This package is skeleton-only. It does not ship a mesh, skinning weights, or SMPL-family assets.
- Body scaling follows an OpenSim-style convention: scales apply per body, and child joint offsets are scaled in the parent body frame.
- For most fitting code, start with `build_layer(...)`. Use `*Model` classes when you want persistent learnable state inside the module.


## Additional example scripts

The `examples/` directory now includes extended, runnable examples for:

- temporal lifting inspired by VideoPose3D / PoseFormer-style pipelines
- hybrid analytic-neural IK inspired by HybrIK-style designs
- sequence denoising with temporal priors and smoothness regularization
- cross-skeleton retargeting between public layouts
- pseudo-label bootstrapping from 2D detections
- spine-specific sequence fitting
