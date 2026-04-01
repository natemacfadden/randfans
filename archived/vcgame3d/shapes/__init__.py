"""Shape/vector-configuration generation for vcgame3d."""
from __future__ import annotations

from typing import TYPE_CHECKING

from regfans import VectorConfiguration

from .reflexive import reflexive_vectors

if TYPE_CHECKING:
    from regfans import Fan


_REGISTRY: dict[str, object] = {
    "reflexive": lambda **kw: reflexive_vectors(
        h11=kw["h11"],
        polytope_id=kw["polytope_id"],
    ),
}
_SHAPES = tuple(_REGISTRY)


def get_vectors(
    name: str,
    *,
    h11: int = 1,
    polytope_id: int = 0,
) -> list[list[int]]:
    """Return integer vectors for the named shape.

    Parameters
    ----------
    name : str
        One of ``"reflexive"``.
    h11 : int, optional
        Hodge number h^{1,1} for ``"reflexive"`` shapes.
    polytope_id : int, optional
        Index into polytopes with the given h11. Defaults to 0.

    Returns
    -------
    list[list[int]]
        Integer 4-vectors (row format).

    Raises
    ------
    ValueError
        If ``name`` is not a recognised shape.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown shape {name!r}. Choose from: {', '.join(_SHAPES)}"
        )
    return _REGISTRY[name](h11=h11, polytope_id=polytope_id)


def vectors_to_fan(vectors: list[list[int]]) -> "Fan":
    """Triangulate a list of integer vectors into a fan.

    Parameters
    ----------
    vectors : list[list[int]]
        Integer 4-vectors (row format).

    Returns
    -------
    Fan
        A triangulated fan of the vectors.
    """
    return VectorConfiguration(vectors).triangulate()


def load_shape(
    name: str,
    *,
    h11: int = 1,
    polytope_id: int = 0,
) -> "Fan":
    """Fetch vectors and triangulate into a fan in one step.

    Parameters
    ----------
    name : str
        One of ``"reflexive"``.
    h11 : int, optional
        Hodge number h^{1,1} for ``"reflexive"`` shapes.
    polytope_id : int, optional
        Index into polytopes with the given h11. Defaults to 0.

    Returns
    -------
    Fan
        A triangulated fan.
    """
    return vectors_to_fan(get_vectors(name, h11=h11, polytope_id=polytope_id))
