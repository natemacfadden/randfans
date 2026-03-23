"""Tests for src/display.py (pure functions only — no curses)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from src.display import _cone_edge_map, _project, _viridis_rgb

if TYPE_CHECKING:
    from regfans import Fan


# ---------------------------------------------------------------
# _viridis_rgb
# ---------------------------------------------------------------

def test_viridis_returns_triple() -> None:
    result = _viridis_rgb(0.5)
    assert len(result) == 3


def test_viridis_values_are_ints() -> None:
    r, g, b = _viridis_rgb(0.5)
    assert isinstance(r, int)
    assert isinstance(g, int)
    assert isinstance(b, int)


@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_viridis_values_in_range(t: float) -> None:
    r, g, b = _viridis_rgb(t)
    assert 0 <= r <= 1000
    assert 0 <= g <= 1000
    assert 0 <= b <= 1000


def test_viridis_clamps_below_zero() -> None:
    assert _viridis_rgb(-1.0) == _viridis_rgb(0.0)


def test_viridis_clamps_above_one() -> None:
    assert _viridis_rgb(2.0) == _viridis_rgb(1.0)


def test_viridis_endpoints_differ() -> None:
    assert _viridis_rgb(0.0) != _viridis_rgb(1.0)


def test_viridis_green_increases_end_to_end() -> None:
    _, g0, _ = _viridis_rgb(0.0)
    _, g1, _ = _viridis_rgb(1.0)
    assert g1 > g0


def test_viridis_midpoint_between_endpoints() -> None:
    r0, g0, b0 = _viridis_rgb(0.0)
    r1, g1, b1 = _viridis_rgb(1.0)
    rm, gm, bm = _viridis_rgb(0.5)
    assert min(r0, r1) <= rm <= max(r0, r1)
    assert min(g0, g1) <= gm <= max(g0, g1)
    assert min(b0, b1) <= bm <= max(b0, b1)


# ---------------------------------------------------------------
# _project
# ---------------------------------------------------------------

@pytest.fixture
def std_basis():
    p  = np.array([0.0, 0.0, 1.0])
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.array([1.0, 0.0, 0.0])
    return p, e1, e2


def test_project_forward_not_none(std_basis) -> None:
    p, e1, e2 = std_basis
    assert _project(np.array([0.0, 0.0, 2.0]), p, e1, e2) is not None


def test_project_antipodal_is_none(std_basis) -> None:
    p, e1, e2 = std_basis
    assert _project(np.array([0.0, 0.0, -2.0]), p, e1, e2) is None


def test_project_zero_vector_is_none(std_basis) -> None:
    p, e1, e2 = std_basis
    assert _project(np.zeros(3), p, e1, e2) is None


def test_project_returns_two_floats(std_basis) -> None:
    p, e1, e2 = std_basis
    result = _project(np.array([0.0, 0.0, 1.0]), p, e1, e2)
    assert result is not None
    x, y = result
    assert isinstance(x, float)
    assert isinstance(y, float)


def test_project_e1_direction(std_basis) -> None:
    p, e1, e2 = std_basis
    # vector in e1 direction → positive y, zero x
    x, y = _project(np.array([0.0, 1.0, 1.0]), p, e1, e2)
    assert y > 0.0
    assert abs(x) < 1e-10


def test_project_e2_direction(std_basis) -> None:
    p, e1, e2 = std_basis
    # vector in e2 direction → positive x, zero y
    x, y = _project(np.array([1.0, 0.0, 1.0]), p, e1, e2)
    assert x > 0.0
    assert abs(y) < 1e-10


def test_project_proportional_to_scale(std_basis) -> None:
    p, e1, e2 = std_basis
    v      = np.array([1.0, 1.0, 1.0])
    r1     = _project(v,       p, e1, e2)
    r2     = _project(2.0 * v, p, e1, e2)
    assert r1 is not None and r2 is not None
    np.testing.assert_allclose(
        np.array(r2), 2.0 * np.array(r1), atol=1e-10,
    )


def test_project_near_horizon_not_none(std_basis) -> None:
    p, e1, e2 = std_basis
    # vector nearly perpendicular to p — still in front hemisphere
    v = np.array([1.0, 0.0, 0.1])
    assert _project(v, p, e1, e2) is not None


# ---------------------------------------------------------------
# _cone_edge_map
# ---------------------------------------------------------------

def test_cone_edge_map_keys_are_sorted_pairs(fan3: Fan) -> None:
    for a, b in _cone_edge_map(fan3):
        assert a < b


def test_cone_edge_map_no_self_loops(fan3: Fan) -> None:
    for a, b in _cone_edge_map(fan3):
        assert a != b


def test_cone_edge_map_all_cone_edges_present(fan3: Fan) -> None:
    edge_map = _cone_edge_map(fan3)
    for cone in fan3.cones():
        labels = list(cone)
        for i in range(len(labels)):
            a, b = labels[i], labels[(i + 1) % len(labels)]
            assert (min(a, b), max(a, b)) in edge_map


def test_cone_edge_map_each_edge_in_at_least_one_cone(
    fan3: Fan,
) -> None:
    for cones in _cone_edge_map(fan3).values():
        assert len(cones) >= 1


def test_cone_edge_map_cone_values_are_sorted_tuples(
    fan3: Fan,
) -> None:
    for cones in _cone_edge_map(fan3).values():
        for ct in cones:
            assert ct == tuple(sorted(ct))


def test_cone_edge_map_has_interior_edges(fan3: Fan) -> None:
    # a valid simplicial fan should have edges shared by >=2 cones
    shared = [
        e for e, c in _cone_edge_map(fan3).items() if len(c) >= 2
    ]
    assert len(shared) > 0


def test_cone_edge_map_cone_references_are_valid(
    fan3: Fan,
) -> None:
    valid = set(tuple(sorted(c)) for c in fan3.cones())
    for cones in _cone_edge_map(fan3).values():
        for ct in cones:
            assert ct in valid
