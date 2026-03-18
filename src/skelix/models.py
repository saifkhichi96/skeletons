from __future__ import annotations

import torch

from .model import SkeletalModel
from .specs import (
    SkeletonSpec,
    coco_spec,
    coco_wholebody_spec,
    face68_spec,
    get_spec,
    halpe26_spec,
    halpe_fullbody_spec,
    hand21_spec,
    human36m_spec,
    mpii_spec,
    spinetrack_spec,
)


class _SpecBackedModel(SkeletalModel):
    SPEC_FACTORY = None

    def __init__(self, *args, dtype: torch.dtype = torch.float32, **kwargs) -> None:
        if self.SPEC_FACTORY is None:
            raise RuntimeError('SPEC_FACTORY must be defined on subclasses.')
        spec: SkeletonSpec = self.SPEC_FACTORY(dtype=dtype)
        super().__init__(spec, *args, dtype=dtype, **kwargs)


class CocoModel(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(coco_spec)


class MPIIModel(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(mpii_spec)


class Human36MModel(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(human36m_spec)


class Halpe26Model(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(halpe26_spec)


class Hand21Model(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(hand21_spec)


class Face68Model(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(face68_spec)


class HalpeFullBodyModel(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(halpe_fullbody_spec)


class CocoWholeBodyModel(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(coco_wholebody_spec)


class SpineTrackModel(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(spinetrack_spec)


def create_model(name: str, *args, dtype: torch.dtype = torch.float32, **kwargs) -> SkeletalModel:
    spec = get_spec(name, dtype=dtype)
    return SkeletalModel(spec, *args, dtype=dtype, **kwargs)
