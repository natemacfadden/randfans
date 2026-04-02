# =============================================================================
#    Copyright (C) 2026  Nate MacFadden
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================

"""Shape/vector-configuration generation."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from regfans import VectorConfiguration

from .cube import cube_vectors
from .random import random_vectors
from .reflexive import reflexive_vectors
from .trunc_oct import trunc_oct_vectors

if TYPE_CHECKING:
    from regfans import Fan


_REGISTRY: dict[str, object] = {
    "cube":      lambda **kw: cube_vectors(kw["n"] or 3),
    "random":    lambda **kw: random_vectors(
                     seed=kw["seed"],
                     n_vectors=kw["n_vectors"] or 12,
                     max_coord=kw["max_coord"],
                 ),
    "reflexive": lambda **kw: reflexive_vectors(polytope_id=kw["polytope_id"]),
    "trunc_oct": lambda **kw: trunc_oct_vectors(),
}
_SHAPES = tuple(_REGISTRY)


def get_vectors(
    name: str,
    *,
    seed: int = 1102,
    polytope_id: int = 0,
    n: int | None = None,
    n_vectors: int | None = None,
    max_coord: int = 3,
) -> list[list[int]]:
    """Return integer vectors for the named shape.

    Parameters
    ----------
    name : str
        One of ``"cube"``, ``"random"``, ``"reflexive"``, ``"trunc_oct"``.
    seed : int, optional
        RNG seed for ``"random"`` shapes.
    polytope_id : int, optional
        Polytope index for ``"reflexive"`` shapes (0–4318).
    n : int or None, optional
        Grid size for ``"cube"`` (default 3). Ignored for other shapes.
    n_vectors : int or None, optional
        Seed vector count for ``"random"`` (default 12). Ignored for other shapes.
    max_coord : int, optional
        Coordinate range for ``"random"`` shapes (default 3).

    Returns
    -------
    list[list[int]]
        Integer 3-vectors on the polytope boundary.

    Raises
    ------
    ValueError
        If ``name`` is not a recognised shape.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown shape {name!r}. Choose from: {', '.join(_SHAPES)}"
        )
    return _REGISTRY[name](
        n=n, n_vectors=n_vectors, seed=seed,
        polytope_id=polytope_id, max_coord=max_coord,
    )


def vectors_to_fan(vectors: list[list[int]]) -> Fan:
    """Triangulate a list of integer vectors into a fan.

    Parameters
    ----------
    vectors : list[list[int]]
        Integer 3-vectors.

    Returns
    -------
    Fan
        A triangulated fan of the vectors.
    """
    return VectorConfiguration(vectors).triangulate()


def load_shape(
    name: str,
    *,
    seed: int = 1102,
    polytope_id: int = 0,
    n: int | None = None,
    n_vectors: int | None = None,
    max_coord: int = 3,
) -> Fan:
    """Generate vectors and triangulate into a fan in one step.

    Parameters
    ----------
    name : str
        One of ``"cube"``, ``"random"``, ``"reflexive"``, ``"trunc_oct"``.
    seed : int, optional
        RNG seed for ``"random"`` shapes.
    polytope_id : int, optional
        Polytope index for ``"reflexive"`` shapes (0–4318).
    n : int or None, optional
        Grid size for ``"cube"`` (default 3).
    n_vectors : int or None, optional
        Seed vector count for ``"random"`` (default 12).
    max_coord : int, optional
        Coordinate range for ``"random"`` shapes (default 3).

    Returns
    -------
    Fan
        A triangulated fan.
    """
    return vectors_to_fan(
        get_vectors(
            name, seed=seed, polytope_id=polytope_id,
            n=n, n_vectors=n_vectors, max_coord=max_coord,
        )
    )
