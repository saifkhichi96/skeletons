from __future__ import annotations

import argparse
from pathlib import Path

import torch

from skelix.fitting import FrameDataset, PerspectiveCamera, SkeletalFitter
from skelix.models import create_model

SUPPORTED_SKELETONS = (
    'human36m',
    'coco',
    'mpii',
    'halpe26',
    'hand21',
    'face68',
    'halpe_fullbody',
    'coco_wholebody',
    'spinetrack',
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Fit a skeletal model to a shared .npz dataset of 2D or 3D joints.',
    )
    parser.add_argument(
        'dataset',
        type=Path,
        help='Path to an .npz file with joints_3d and optionally joints_2d.',
    )
    parser.add_argument(
        '--skeleton',
        choices=SUPPORTED_SKELETONS,
        default='human36m',
        help='Skeleton layout used by the dataset arrays.',
    )
    parser.add_argument('--mode', choices=('2d', '3d'), default='3d')
    parser.add_argument('--sample-index', type=int, default=0)
    args = parser.parse_args()

    model = create_model(args.skeleton, create_global_orient=False, create_body_pose=False)
    dataset = FrameDataset.from_npz(args.dataset, expected_num_joints=model.num_joints)
    sample = dataset[args.sample_index]
    fitter = SkeletalFitter(model=model)

    if args.mode == '3d':
        result = fitter.fit_3d(sample['joints_3d'].unsqueeze(0), num_iters=200)
        print({'skeleton': model.spec.name, **result.losses})
        print(result.model_output.joints.shape)
        return

    if 'joints_2d' not in sample:
        raise ValueError('The dataset file does not contain joints_2d.')
    camera = PerspectiveCamera(
        fx=sample.get('fx', torch.tensor(1000.0)).reshape(()),
        fy=sample.get('fy', torch.tensor(1000.0)).reshape(()),
        cx=sample.get('cx', torch.tensor(512.0)).reshape(()),
        cy=sample.get('cy', torch.tensor(512.0)).reshape(()),
    )
    confidences = sample.get('confidences')
    result = fitter.fit_2d(
        sample['joints_2d'].unsqueeze(0),
        camera,
        confidences=None if confidences is None else confidences.unsqueeze(0),
        num_iters=300,
    )
    print({'skeleton': model.spec.name, **result.losses})
    print(result.model_output.joints.shape)


if __name__ == '__main__':
    main()
