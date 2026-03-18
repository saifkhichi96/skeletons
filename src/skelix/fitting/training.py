"""Compatibility exports for fitting trainers."""

from .core.training import JointLimitTrainer, PoseVAETrainer, TemporalPriorTrainer, TrainerState
from .h36m import H36MJointLimitTrainer, H36MPoseVAETrainer, H36MTemporalPriorTrainer

__all__ = [
    'JointLimitTrainer',
    'PoseVAETrainer',
    'TemporalPriorTrainer',
    'TrainerState',
    'H36MJointLimitTrainer',
    'H36MPoseVAETrainer',
    'H36MTemporalPriorTrainer',
]
