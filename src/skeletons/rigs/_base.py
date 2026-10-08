from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import torch

from ..model import SkeletalModel, SkeletalModelLayer
from ._spec import SkeletonSpec


class _SpecBackedModel(SkeletalModel):
    SPEC_FACTORY: ClassVar[Callable[[torch.dtype], SkeletonSpec] | None] = None

    def __init__(self, *args, dtype: torch.dtype = torch.float32, **kwargs) -> None:
        if self.SPEC_FACTORY is None:
            raise RuntimeError("SPEC_FACTORY must be defined on subclasses.")
        spec = self.SPEC_FACTORY(dtype=dtype)
        spec_options = {
            key: kwargs.pop(key)
            for key in (
                "joint_types",
                "joint_axes",
                "joint_limits",
                "links",
                "markers",
                "contacts",
                "metadata",
            )
            if key in kwargs
        }
        if spec_options:
            spec = spec.configured(**spec_options)
        super().__init__(spec, *args, dtype=dtype, **kwargs)


class _SpecBackedModelLayer(SkeletalModelLayer):
    SPEC_FACTORY: ClassVar[Callable[[torch.dtype], SkeletonSpec] | None] = None

    def __init__(self, *args, dtype: torch.dtype = torch.float32, **kwargs) -> None:
        if self.SPEC_FACTORY is None:
            raise RuntimeError("SPEC_FACTORY must be defined on subclasses.")
        spec = self.SPEC_FACTORY(dtype=dtype)
        spec_options = {
            key: kwargs.pop(key)
            for key in (
                "joint_types",
                "joint_axes",
                "joint_limits",
                "links",
                "markers",
                "contacts",
                "metadata",
            )
            if key in kwargs
        }
        if spec_options:
            spec = spec.configured(**spec_options)
        super().__init__(spec, *args, dtype=dtype, **kwargs)
