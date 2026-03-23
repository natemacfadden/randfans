"""
Generate a vector configuration and triangulation from the
vertices of the truncated octahedron (permutohedron).

The 24 vertices are all distinct points of the form (0, ±1, ±2)
and its permutations. The shape tiles R³ by translation, has 36
edges and 14 faces (8 regular hexagons + 6 squares), and gives a
richer fan structure than the cube.
"""

from __future__ import annotations

from itertools import permutations, product
from typing import TYPE_CHECKING

from regfans import VectorConfiguration

if TYPE_CHECKING:
    from regfans import Fan


def trunc_oct_vectors() -> list[list[int]]:
    """
    **Description:**
    Return the 24 vertices of the truncated octahedron: all
    distinct permutations of (0, ±1, ±2).

    **Returns:**
    A list of 24 integer 3-vectors.
    """
    pts: set[tuple[int, ...]] = set()
    for p in permutations([0, 1, 2]):
        for signs in product((-1, 1), repeat=3):
            pts.add(tuple(s * c for s, c in zip(signs, p)))
    pts.discard((0, 0, 0))
    return [list(p) for p in sorted(pts)]


def trunc_oct_vc() -> VectorConfiguration:
    """
    **Description:**
    Return the VectorConfiguration of the truncated octahedron
    vertices.

    **Returns:**
    A VectorConfiguration.
    """
    return VectorConfiguration(trunc_oct_vectors())


def trunc_oct_fan() -> Fan:
    """
    **Description:**
    Return a triangulation of the truncated octahedron vector
    configuration.

    **Returns:**
    A Fan.
    """
    return VectorConfiguration(trunc_oct_vectors()).triangulate()
