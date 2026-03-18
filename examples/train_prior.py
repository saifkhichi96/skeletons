from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from skelix.fitting import FrameDataset, JointLimitTrainer, PoseVAE, PoseVAETrainer
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
        description='Train generic fitting priors from a shared .npz joints dataset.',
    )
    parser.add_argument(
        'dataset',
        type=Path,
        help='Path to an .npz file with joints_3d shaped as [N, J, 3].',
    )
    parser.add_argument(
        '--skeleton',
        choices=SUPPORTED_SKELETONS,
        default='human36m',
        help='Skeleton layout used by the dataset arrays.',
    )
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--num-hidden-layers', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    model = create_model(args.skeleton, create_global_orient=False, create_body_pose=False)
    dataset = FrameDataset.from_npz(args.dataset, expected_num_joints=model.num_joints)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    joint_limit_trainer = JointLimitTrainer(model=model)
    joint_limit_prior = joint_limit_trainer.fit_from_loader(loader)

    vae = PoseVAE(
        num_joints=model.num_joints - 1,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
    )
    trainer = PoseVAETrainer(vae, model=model)
    for epoch in range(args.epochs):
        state = trainer.train_epoch(loader)
        print(
            f'epoch={epoch + 1} loss={state.loss:.6f} '
            f'recon={state.metrics.get("recon", 0.0):.6f} '
            f'kl={state.metrics.get("kl", 0.0):.6f}',
        )

    torch.save(
        {
            'format_version': 1,
            'skeleton': model.spec.name,
            'pose_prior_config': {
                'num_joints': model.num_joints - 1,
                'latent_dim': args.latent_dim,
                'hidden_dim': args.hidden_dim,
                'num_hidden_layers': args.num_hidden_layers,
            },
            'pose_prior': vae.state_dict(),
            'joint_limit_prior_config': {
                'barrier_scale': joint_limit_prior.barrier_scale,
            },
            'joint_limit_prior': joint_limit_prior.state_dict(),
        },
        args.output / f'{model.spec.name}_priors.pt',
    )


if __name__ == '__main__':
    main()
