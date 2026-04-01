"""
Player classes for vcgame3d.

Player3D  — flat R³ flight (used for the reference cube scene).
Player4D  — geodesic flight on S³ (used for fan scenes).
            Position is a unit 4-vector; movement follows great-circle arcs so
            going forward long enough returns you to your starting point.
"""
from __future__ import annotations
import numpy as np


class Player3D:
    SPEED_MIN  = 0.01
    SPEED_MAX  = 2.0
    SPEED_STEP = 0.05

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


class Player4D:
    """Player living on S³ with geodesic (great-circle) movement.

    State
    -----
    _pos4d   : unit 4-vector — position on S³
    _fwd4d   : unit 4-vector in T_{pos4d}S³ — forward tangent
    _right4d : unit 4-vector in T_{pos4d}S³ — right tangent
    _up4d    : unit 4-vector in T_{pos4d}S³ — up tangent

    The four vectors form an orthonormal basis of R⁴.

    Movement
    --------
    Thrust/strafe/lift rotate pos4d along the corresponding great-circle arc
    and parallel-transport the frame.  Pitch/yaw/roll rotate the frame in
    place without moving pos4d.

    Rendering
    ---------
    _pos / _fwd / _right / _up  are 3D vectors computed from the numerical
    Jacobian of the supplied S³→R³ projection callable, Gram-Schmidt
    orthogonalised so the renderer gets a proper camera frame.
    """

    SPEED_MIN  = 0.002
    SPEED_MAX  = 0.30
    SPEED_STEP = 0.003
    _JAC_EPS   = 1e-5   # step for numerical Jacobian

    def __init__(
        self,
        pos4d:   np.ndarray,
        fwd4d:   np.ndarray,
        right4d: np.ndarray,
        up4d:    np.ndarray,
        project,            # callable S³ → R³
        speed: float = 0.05,
    ):
        self._pos4d   = np.array(pos4d,   dtype=float)
        self._fwd4d   = np.array(fwd4d,   dtype=float)
        self._right4d = np.array(right4d, dtype=float)
        self._up4d    = np.array(up4d,    dtype=float)
        self._project = project
        self.speed    = float(speed)
        self._dirty   = True
        self._cache   = None   # (pos3d, fwd3d, right3d, up3d)
        self._ortho4d_inplace()   # normalize initial frame

    # ── internal helpers ──────────────────────────────────────────────

    def _ortho4d_inplace(self) -> None:
        """Gram-Schmidt orthonormalize {pos4d, fwd4d, right4d, up4d} in-place.
        Does NOT touch the dirty flag — callers manage it."""
        p = self._pos4d / np.linalg.norm(self._pos4d)

        f = self._fwd4d - np.dot(self._fwd4d, p) * p
        f /= np.linalg.norm(f)

        r = self._right4d
        r = r - np.dot(r, p) * p - np.dot(r, f) * f
        r /= np.linalg.norm(r)

        u = self._up4d
        u = u - np.dot(u, p) * p - np.dot(u, f) * f - np.dot(u, r) * r
        u /= np.linalg.norm(u)

        self._pos4d, self._fwd4d, self._right4d, self._up4d = p, f, r, u

    def _build_cache(self) -> None:
        """Compute 3D position/frame via numerical Jacobian of the projection.
        Also stabilises the 4D frame to prevent drift."""
        self._ortho4d_inplace()          # stabilise before computing Jacobian

        p     = self._pos4d
        pr    = self._project
        eps   = self._JAC_EPS
        pos3d = pr(p)

        def jac(v4d: np.ndarray) -> np.ndarray:
            q = p + eps * v4d
            q = q / np.linalg.norm(q)
            d = pr(q) - pos3d
            n = np.linalg.norm(d)
            return d / n if n > 1e-12 else d

        # Numerical partial derivatives, then Gram-Schmidt in R³
        f = jac(self._fwd4d)
        n = np.linalg.norm(f); f = f / n if n > 1e-12 else f

        r = jac(self._right4d)
        r = r - np.dot(r, f) * f
        n = np.linalg.norm(r); r = r / n if n > 1e-12 else r

        u = jac(self._up4d)
        u = u - np.dot(u, f) * f - np.dot(u, r) * r
        n = np.linalg.norm(u); u = u / n if n > 1e-12 else u

        self._cache = (pos3d, f, r, u)
        self._dirty = False

    def _get(self) -> tuple:
        if self._dirty:
            self._build_cache()
        return self._cache

    # ── renderer-compatible interface ─────────────────────────────────

    @property
    def _pos(self)   -> np.ndarray: return self._get()[0]
    @property
    def _fwd(self)   -> np.ndarray: return self._get()[1]
    @property
    def _right(self) -> np.ndarray: return self._get()[2]
    @property
    def _up(self)    -> np.ndarray: return self._get()[3]

    @property
    def position(self) -> np.ndarray: return self._get()[0].copy()
    @property
    def forward(self)  -> np.ndarray: return self._get()[1].copy()
    @property
    def right(self)    -> np.ndarray: return self._get()[2].copy()
    @property
    def up(self)       -> np.ndarray: return self._get()[3].copy()

    # ── geodesic movement ─────────────────────────────────────────────
    # Thrust along axis v:  new_pos = cos(dt)·pos + sin(dt)·v
    #                       new_v   = −sin(dt)·pos + cos(dt)·v
    # Other tangent axes are unchanged (parallel transport).

    def thrust(self, scale: float = 1.0) -> None:
        dt = scale * self.speed; c, s = np.cos(dt), np.sin(dt)
        p, f = self._pos4d, self._fwd4d
        self._pos4d, self._fwd4d = c*p + s*f, -s*p + c*f
        self._dirty = True

    def strafe(self, scale: float = 1.0) -> None:
        dt = scale * self.speed; c, s = np.cos(dt), np.sin(dt)
        p, r = self._pos4d, self._right4d
        self._pos4d, self._right4d = c*p + s*r, -s*p + c*r
        self._dirty = True

    def lift(self, scale: float = 1.0) -> None:
        dt = scale * self.speed; c, s = np.cos(dt), np.sin(dt)
        p, u = self._pos4d, self._up4d
        self._pos4d, self._up4d = c*p + s*u, -s*p + c*u
        self._dirty = True

    # ── rotations (tangent-space only, pos4d unchanged) ──────────────

    def pitch(self, angle: float) -> None:
        """Nose up (+) / down (−)."""
        c, s = np.cos(angle), np.sin(angle)
        f, u = self._fwd4d, self._up4d
        self._fwd4d, self._up4d = c*f + s*u, -s*f + c*u
        self._dirty = True

    def yaw(self, angle: float) -> None:
        """Nose left (+) / right (−)."""
        c, s = np.cos(angle), np.sin(angle)
        f, r = self._fwd4d, self._right4d
        self._fwd4d, self._right4d = c*f - s*r, s*f + c*r
        self._dirty = True

    def roll(self, angle: float) -> None:
        """Roll right (+) / left (−)."""
        c, s = np.cos(angle), np.sin(angle)
        r, u = self._right4d, self._up4d
        self._right4d, self._up4d = c*r - s*u, s*r + c*u
        self._dirty = True

    def __repr__(self) -> str:
        p = self._pos4d
        return (f"Player4D(pos4d=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f},{p[3]:+.2f}) "
                f"spd={self.speed:.3f})")
