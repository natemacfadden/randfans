"""
6-DOF player in R³ with flight controls.
State: position + orthonormal frame (forward, up, right=forward×up).
"""
from __future__ import annotations
import numpy as np


class Player3D:
    def __init__(
        self,
        position: tuple = (0.0, 0.0, -5.0),
        forward:  tuple = (0.0, 0.0,  1.0),
        up:       tuple = (0.0, 1.0,  0.0),
        speed:    float = 0.15,
    ):
        self._pos = np.array(position, dtype=float)
        fwd = np.array(forward, dtype=float)
        self._fwd = fwd / np.linalg.norm(fwd)
        up_ = np.array(up, dtype=float)
        up_ -= np.dot(up_, self._fwd) * self._fwd
        self._up    = up_ / np.linalg.norm(up_)
        self._right = np.cross(self._up, self._fwd)
        self.speed  = float(speed)

    # ------------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self._pos.copy()

    @property
    def forward(self) -> np.ndarray:
        return self._fwd.copy()

    @property
    def up(self) -> np.ndarray:
        return self._up.copy()

    @property
    def right(self) -> np.ndarray:
        return self._right.copy()

    # ------------------------------------------------------------------
    def _reorthogonalize(self) -> None:
        # Symmetric Gram-Schmidt: split error evenly so neither axis accumulates drift
        dot = float(np.dot(self._fwd, self._up))
        fwd = self._fwd - 0.5 * dot * self._up
        up  = self._up  - 0.5 * dot * self._fwd
        self._fwd   = fwd / np.linalg.norm(fwd)
        self._up    = up  / np.linalg.norm(up)
        self._right = np.cross(self._up, self._fwd)

    def pitch(self, angle: float) -> None:
        """Nose up (+) / down (−)."""
        c, s = np.cos(angle), np.sin(angle)
        fwd = c * self._fwd + s * self._up
        up  = -s * self._fwd + c * self._up
        self._fwd, self._up = fwd, up
        self._reorthogonalize()

    def yaw(self, angle: float) -> None:
        """Nose left (+) / right (−)."""
        c, s = np.cos(angle), np.sin(angle)
        self._fwd = c * self._fwd - s * self.right
        self._reorthogonalize()

    def roll(self, angle: float) -> None:
        """Roll right (+) / left (−)."""
        c, s = np.cos(angle), np.sin(angle)
        self._up = c * self._up + s * self.right
        self._reorthogonalize()

    def thrust(self, scale: float = 1.0) -> None:
        self._pos += scale * self.speed * self._fwd

    def strafe(self, scale: float = 1.0) -> None:
        self._pos += scale * self.speed * self.right

    def lift(self, scale: float = 1.0) -> None:
        self._pos += scale * self.speed * self._up

    def __repr__(self) -> str:
        p = self._pos
        f = self._fwd
        return (f"Player3D(pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
                f"fwd=({f[0]:+.2f},{f[1]:+.2f},{f[2]:+.2f}))")
