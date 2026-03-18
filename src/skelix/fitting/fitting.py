"""Compatibility exports for fitting utilities."""

from .core.fitter import FittingResult, FittingWeights, SkeletalFitter
from .h36m import H36MFitter

__all__ = ['FittingResult', 'FittingWeights', 'SkeletalFitter', 'H36MFitter']
