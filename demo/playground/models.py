from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SceneNode:
    group: str
    label: str
    kind: str
    payload: Any
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TargetMotionSpec:
    name: str
    frame: str
    base: np.ndarray
    color: tuple[float, float, float, float]
    phase: float
    speed: float
    height: float
    x_amp: float
    y_amp: float
    drift: np.ndarray
    frozen: bool = False

    def position(self, time_s: float) -> np.ndarray:
        if self.frozen:
            return self.base.copy()
        u = (self.speed * time_s + self.phase) % 1.0
        parabola = self.height * (1.0 - (2.0 * u - 1.0) ** 2)
        wave = 2.0 * np.pi * u
        pos = self.base + self.drift * time_s
        pos = pos + np.array(
            [
                self.x_amp * np.sin(wave + self.phase),
                parabola,
                self.y_amp * np.cos(1.3 * wave + 0.5 * self.phase),
            ],
            dtype=float,
        )
        return pos


@dataclass(slots=True)
class FitHistoryPoint:
    step: int
    value: float
    label: str = "loss"
