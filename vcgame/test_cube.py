"""Tests for generate_cube.py"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.generate_cube import cube_fan, cube_vectors

if TYPE_CHECKING:
    from regfans import Fan


@pytest.fixture(scope="module")
def fan3() -> Fan:
    return cube_fan(3)


@pytest.mark.parametrize("n, expected", [(3, 26), (5, 98), (7, 218)])
def test_cube_vectors_count(n: int, expected: int) -> None:
    assert len(cube_vectors(n)) == expected

def test_cube_vectors_are_3d() -> None:
    for v in cube_vectors(3):
        assert len(v) == 3

def test_cube_vectors_centered() -> None:
    for n in (3, 5, 7):
        half = (n - 1) // 2
        for v in cube_vectors(n):
            assert all(-half <= c <= half for c in v)

def test_cube_vectors_no_origin() -> None:
    for n in (3, 5, 7):
        assert [0, 0, 0] not in cube_vectors(n)

def test_cube_vectors_all_boundary() -> None:
    for n in (3, 5, 7):
        half = (n - 1) // 2
        for v in cube_vectors(n):
            assert any(abs(c) == half for c in v)

def test_cube_vectors_no_duplicates() -> None:
    for n in (3, 5):
        vecs = cube_vectors(n)
        assert len(vecs) == len(set(map(tuple, vecs)))

def test_cube_vectors_rejects_even() -> None:
    with pytest.raises(ValueError):
        cube_vectors(4)

def test_cube_vectors_rejects_small() -> None:
    with pytest.raises(ValueError):
        cube_vectors(1)

def test_cube_fan_n3_cone_count(fan3: Fan) -> None:
    assert len(fan3.cones()) == 48

def test_cube_fan_is_valid(fan3: Fan) -> None:
    assert fan3.is_valid()

def test_cube_fan_is_fine(fan3: Fan) -> None:
    assert fan3.is_fine()

def test_cube_fan_is_regular(fan3: Fan) -> None:
    assert fan3.is_regular()

def test_cube_fan_cones_are_triples(fan3: Fan) -> None:
    for cone in fan3.cones():
        assert len(cone) == 3
