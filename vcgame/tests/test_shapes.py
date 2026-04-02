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

"""Tests for shapes/ — vector generation."""
from __future__ import annotations

import pytest

from shapes import get_vectors, _SHAPES
from shapes.cube import cube_vectors
from shapes.random import random_vectors
from shapes.reflexive import reflexive_vectors
from shapes.trunc_oct import trunc_oct_vectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_shapes_vectors():
    """Return vectors for one instance of each shape (no network shapes)."""
    return [
        get_vectors("cube", n=3),
        get_vectors("random", seed=1102),
        get_vectors("trunc_oct"),
    ]


# ---------------------------------------------------------------------------
# get_vectors — shared structural checks (all non-network shapes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vectors", _all_shapes_vectors())
def test_output_type(vectors):
    assert isinstance(vectors, list)
    assert all(isinstance(v, list) for v in vectors)


@pytest.mark.parametrize("vectors", _all_shapes_vectors())
def test_vector_dimension(vectors):
    assert all(len(v) == 3 for v in vectors)


@pytest.mark.parametrize("vectors", _all_shapes_vectors())
def test_integer_values(vectors):
    for v in vectors:
        assert all(isinstance(x, int) for x in v)


@pytest.mark.parametrize("vectors", _all_shapes_vectors())
def test_no_duplicates(vectors):
    tuples = [tuple(v) for v in vectors]
    assert len(tuples) == len(set(tuples))


@pytest.mark.parametrize("vectors", _all_shapes_vectors())
def test_no_origin(vectors):
    assert [0, 0, 0] not in vectors


# Determinism tested per-shape for clarity:

def test_determinism_cube():
    assert cube_vectors(3) == cube_vectors(3)
    assert cube_vectors(5) == cube_vectors(5)


def test_determinism_random():
    assert random_vectors(seed=42) == random_vectors(seed=42)


def test_determinism_trunc_oct():
    assert trunc_oct_vectors() == trunc_oct_vectors()


def test_unknown_shape_raises():
    with pytest.raises(ValueError, match="Unknown shape"):
        get_vectors("dodecahedron")


# ---------------------------------------------------------------------------
# cube
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, expected_count", [
    (3,  26),
    (5,  98),
    (7, 218),
])
def test_cube_count(n, expected_count):
    assert len(cube_vectors(n)) == expected_count


@pytest.mark.parametrize("n", [3, 5, 7])
def test_cube_count_formula(n):
    """Count matches n^3 - (n-2)^3."""
    assert len(cube_vectors(n)) == n**3 - (n - 2)**3


@pytest.mark.parametrize("n", [3, 5, 7])
def test_cube_boundary(n):
    """Every vector has at least one component equal to ±(n-1)/2."""
    half = (n - 1) // 2
    for v in cube_vectors(n):
        assert any(abs(x) == half for x in v)


@pytest.mark.parametrize("n", [3, 5, 7])
def test_cube_range(n):
    """All components lie in [-(n-1)/2, (n-1)/2]."""
    half = (n - 1) // 2
    for v in cube_vectors(n):
        assert all(-half <= x <= half for x in v)


def test_cube_even_n_raises():
    with pytest.raises(ValueError):
        cube_vectors(4)


def test_cube_small_n_raises():
    with pytest.raises(ValueError):
        cube_vectors(1)


# ---------------------------------------------------------------------------
# random
# ---------------------------------------------------------------------------

def test_random_seed_reproducibility():
    assert random_vectors(seed=0)    == random_vectors(seed=0)
    assert random_vectors(seed=1102) == random_vectors(seed=1102)
    assert random_vectors(seed=99)   == random_vectors(seed=99)


def test_random_different_seeds_differ():
    assert random_vectors(seed=0) != random_vectors(seed=1)


# ---------------------------------------------------------------------------
# reflexive (requires network)
# ---------------------------------------------------------------------------

_REFLEXIVE_EXPECTED = {
    0: [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]],
    100: [
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, -1],
        [-3, -2, 0], [-1, -1, 0], [-2, -1, 0], [-1, 0, 0],
    ],
    1000: [
        [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1],
        [-2, -1, -1], [0, -1, 1], [-3, -1, -1], [-1, -1, 1],
        [-2, -1, 0], [-1, 0, 0], [-1, -1, 0],
    ],
    4000: [
        [1, 0, 0], [0, 1, 0], [1, 0, 2], [-3, 0, -2],
        [-11, -4, -6], [-8, -3, -4], [-5, -2, -2], [-2, -1, 0],
        [-9, -3, -5], [-6, -2, -3], [-3, -1, -1], [0, 0, 1],
        [-7, -2, -4], [-4, -1, -2], [-1, 0, 0], [-5, -1, -3],
        [-2, 0, -1], [-5, -2, -3], [-2, -1, -1], [1, 0, 1],
        [-3, -1, -2], [-1, 0, -1],
    ],
}


@pytest.mark.network
@pytest.mark.parametrize("polytope_id", [0, 100, 1000, 4000])
def test_reflexive_known_vectors(polytope_id):
    assert reflexive_vectors(polytope_id) == _REFLEXIVE_EXPECTED[polytope_id]


@pytest.mark.network
def test_reflexive_bad_id_raises():
    with pytest.raises(ValueError):
        reflexive_vectors(polytope_id=-1)
    with pytest.raises(ValueError):
        reflexive_vectors(polytope_id=9999)


# ---------------------------------------------------------------------------
# trunc_oct
# ---------------------------------------------------------------------------

def test_trunc_oct_count():
    assert len(trunc_oct_vectors()) == 24


def test_trunc_oct_values():
    """Every vector is a permutation of (0, ±1, ±2)."""
    target = {0, 1, -1, 2, -2}
    for v in trunc_oct_vectors():
        assert set(map(abs, v)) == {0, 1, 2}, f"unexpected vector {v}"
        assert sorted(map(abs, v)) == [0, 1, 2], f"unexpected vector {v}"
