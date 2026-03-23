"""
Player position and heading on S².
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warnings

import numpy as np

if TYPE_CHECKING:
    from regfans import Fan, VectorConfiguration


class Player:
    """
    **Description:**
    A player on S² (the unit 2-sphere embedded in R³). The player has a
    position (a unit vector) and a heading (a unit tangent vector at
    that position, perpendicular to position).

    **Attributes:**
    - `position`: Unit vector in R³ representing the player's location on S².
    - `heading`: Unit tangent vector at `position` (perpendicular to
      `position`).
    """

    def __init__(self, position: np.ndarray, heading: np.ndarray) -> None:
        """
        **Description:**
        Initialise a player from a position and heading, both of which are
        normalised on construction.

        **Arguments:**
        - `position`: A non-zero vector in R³. Will be normalised to
          lie on S².
        - `heading`: A non-zero vector in R³. Its component along `position` is
          removed (Gram-Schmidt) and the result is normalised, so that
          it lies in the tangent plane at `position`.

        **Returns:**
        Nothing.

        **Raises:**
        - `ValueError`: If `position` is the zero vector.
        - `ValueError`: If `heading` is the zero vector or is parallel to
          `position` (so no tangent component survives).
        """
        position = np.asarray(position, dtype=float)
        heading  = np.asarray(heading,  dtype=float)

        if position.shape != (3,):
            raise ValueError(
                f"position must be a 3-vector, got shape {position.shape}"
            )
        if heading.shape != (3,):
            raise ValueError(
                f"heading must be a 3-vector, got shape {heading.shape}"
            )

        p_norm = np.linalg.norm(position)
        if p_norm == 0.0:
            raise ValueError("position must be non-zero")
        self._position = position / p_norm

        h = heading - np.dot(heading, self._position) * self._position
        h_norm = np.linalg.norm(h)
        if h_norm == 0.0:
            raise ValueError(
                "heading has no component tangent to position "
                "(parallel or zero)"
            )
        self._heading  = h / h_norm

    @property
    def position(self) -> np.ndarray:
        """**Description:** Current position on S² (read-only copy)."""
        return self._position.copy()

    @property
    def heading(self) -> np.ndarray:
        """**Description:** Current heading tangent vector (read-only copy)."""
        return self._heading.copy()

    def turn(self, angle: float) -> None:
        """
        **Description:**
        Rotate the heading in the tangent plane at the current position.
        Positive angle turns left; negative turns right.

        **Arguments:**
        - `angle`: Rotation angle in radians.

        **Returns:**
        Nothing.
        """
        self._heading = (
            np.cos(angle) * self._heading
            + np.sin(angle) * np.cross(self._position, self._heading)
        )

    def pointed_facet(self, fan: Fan) -> tuple[int, int] | None:
        """
        **Description:**
        Return the facet of the player's current cone that the heading points
        most directly toward, as a sorted label pair.

        **Arguments:**
        - `fan`: A `regfans.Fan` whose cones partition the support.

        **Returns:**
        `tuple[int, int]` `(min_label, max_label)` of the most-aimed-at facet,
        or `None` if the heading is parallel to all facets.
        """
        cone = self.current_cone(fan)
        i, j, k = cone
        facets = [(i, j, k), (j, k, i), (i, k, j)]  # (a, b, c) where c is third

        best_facet: tuple[int, int] | None = None
        best_dot = 0.0

        for a, b, c in facets:
            v_a = fan.vectors(which=(a,))[0]
            v_b = fan.vectors(which=(b,))[0]
            v_c = fan.vectors(which=(c,))[0]
            n = np.cross(v_a, v_b)
            if np.dot(n, v_c) > 0:
                n = -n
            d = np.dot(self._heading, n)
            if d > best_dot:
                best_dot = d
                best_facet = (min(a, b), max(a, b))

        return best_facet

    def move(
        self, step: float, fan: Fan | None = None,
    ) -> tuple[int, int] | None:
        """
        **Description:**
        Advance the player's position along its heading by `step` radians of
        arc on the great circle defined by `position` and `heading`.

        **Arguments:**
        - `step`: Arc-length step in radians (may be negative to move backward).
        - `fan`: Optional `regfans.Fan`. If provided, returns the crossed facet
          label pair when the move crosses a cone boundary, else `None`.

        **Returns:**
        `tuple[int, int]` of the shared facet labels if a cone boundary was
        crossed and `fan` is provided; `None` otherwise.
        """
        old_cone = self.current_cone(fan) if fan is not None else None

        p, h = self._position, self._heading
        c, s = np.cos(step), np.sin(step)

        new_p = c * p + s * h
        new_h = -s * p + c * h

        new_p /= np.linalg.norm(new_p)
        new_h -= np.dot(new_h, new_p) * new_p
        new_h /= np.linalg.norm(new_h)

        self._position = new_p
        self._heading  = new_h

        if fan is None:
            return None
        new_cone = self.current_cone(fan)
        if new_cone == old_cone:
            return None
        shared = set(old_cone) & set(new_cone)
        if len(shared) == 2:
            return tuple(sorted(shared))
        return None

    def current_cone(self, fan: Fan) -> tuple[int, ...]:
        """
        **Description:**
        Return the label tuple of the cone in `fan` that contains the player's
        current position. A position `p` lies in cone `(v1, v2, v3)` iff there
        exist coefficients α1, α2, α3 > 0 such that
        `p = α1*v1 + α2*v2 + α3*v3`.

        **Arguments:**
        - `fan`: A `regfans.Fan` whose cones partition the support.

        **Returns:**
        The label tuple `(i, j, k)` of the containing cone.

        **Raises:**
        - `ValueError`: If no cone contains the current position.
        """
        for cone in fan.cones():
            alpha, _, _, _ = np.linalg.lstsq(fan.vectors(which=cone).T,
                                              self._position, rcond=None)
            if np.all(alpha > -1e-10):
                return cone
        raise ValueError("position is not contained in any cone of the fan")

    def find_circuit_for_crossing(
        self,
        old_cone: tuple[int, ...],
        new_cone: tuple[int, ...],
        fan: Fan,
    ) -> object | None:
        """
        **Description:**
        Find the circuit from `fan.circuits()` whose support matches the union
        of `old_cone` and `new_cone`. These circuits have `.Tpos`/`.Tneg`
        populated and are suitable for `fan.flip()`.

        **Arguments:**
        - `old_cone`: Label tuple of the cone before the crossing.
        - `new_cone`: Label tuple of the cone after the crossing.
        - `fan`: A `regfans.Fan` whose circuits will be searched.

        **Returns:**
        The matching `Circuit`, or `None` if not found.
        """
        target = set(old_cone) | set(new_cone)
        for circ in fan.circuits():
            if set(circ.Z) == target:
                return circ
        return None

    def crossed_circuit(
        self,
        old_cone: tuple[int, ...],
        new_cone: tuple[int, ...],
        vc: VectorConfiguration,
    ) -> object | None:
        """
        **Description:**
        Return the Circuit for the wall crossing between `old_cone` and
        `new_cone`, or `None` if the crossing is degenerate.

        **Arguments:**
        - `old_cone`: Label tuple of the cone before the crossing.
        - `new_cone`: Label tuple of the cone after the crossing.
        - `vc`: A `regfans.VectorConfiguration` for the ambient point set.

        **Returns:**
        A `Circuit` whose support is the union of both cone label sets, or
        `None` if the four rays are coplanar (degenerate flip).
        """
        shared = set(old_cone) & set(new_cone)
        c = (set(old_cone) - shared).pop()
        d = (set(new_cone) - shared).pop()
        circuit_labels = tuple(sorted(shared)) + (c, d)
        circ = vc.circuit(circuit_labels)
        if circ is None:
            warnings.warn(
                f"circuit({circuit_labels}) is None (degenerate/coplanar)"
            )
            return None
        return circ

    def __repr__(self) -> str:
        p, h = self._position, self._heading
        return (
            f"Player(position=[{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}], "
            f"heading=[{h[0]:.4f}, {h[1]:.4f}, {h[2]:.4f}])"
        )
