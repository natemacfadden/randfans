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

"""Tests for renderer/ (pure functions only — no curses)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from renderer.renderer import (
    _cone_edge_map,
    _project,
    _compute_p_surface,
    _pixel_row_positions,
    _sphere_row_hits,
    _shadow_blocked,
    _compute_brightness,
    _SUN_AMBIENT,
    _DIM_LEVEL,
    _FOV_DIST,
)
from renderer.colors import _viridis_rgb

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


def test_viridis_midpoint_in_range() -> None:
    # Viridis is not monotone per channel (blue peaks near t=0.5),
    # so only check that all midpoint values are valid curses color values.
    rm, gm, bm = _viridis_rgb(0.5)
    for v in (rm, gm, bm):
        assert 0 <= v <= 1000


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


# ---------------------------------------------------------------
# Shared geometry for backward-rendering tests
# ---------------------------------------------------------------
# A right-angle triangle on the x=1 plane.
#   v0 = [1, 0, 0],  v1 = [1, 1, 0],  v2 = [1, 0, 1]
#   face normal = [1, 0, 0] (pointing in +x)
# Player looks along +x: p = [1, 0, 0]

@pytest.fixture
def flat_tri():
    v0 = np.array([1.0, 0.0, 0.0])
    v1 = np.array([1.0, 1.0, 0.0])
    v2 = np.array([1.0, 0.0, 1.0])
    n  = np.array([1.0, 0.0, 0.0])
    p  = np.array([1.0, 0.0, 0.0])   # unit viewing direction
    e1 = np.array([0.0, 0.0, 1.0])   # up
    e2 = np.array([0.0, 1.0, 0.0])   # right
    return dict(v0=v0, v1=v1, v2=v2, n=n, p=p, e1=e1, e2=e2)


# ---------------------------------------------------------------
# _compute_p_surface  (Task 1)
# ---------------------------------------------------------------

def test_p_surface_collinear_with_p(flat_tri) -> None:
    ps = _compute_p_surface(flat_tri["p"], flat_tri["v0"], flat_tri["n"])
    # p_surface must be a scalar multiple of p → cross product is zero
    cross = np.cross(ps, flat_tri["p"])
    np.testing.assert_allclose(cross, np.zeros(3), atol=1e-10)


def test_p_surface_lies_on_face_plane(flat_tri) -> None:
    ps = _compute_p_surface(flat_tri["p"], flat_tri["v0"], flat_tri["n"])
    # dot(p_surface - v0, face_normal) == 0
    residual = float(np.dot(ps - flat_tri["v0"], flat_tri["n"]))
    assert abs(residual) < 1e-10


def test_p_surface_degenerate_returns_p(flat_tri) -> None:
    # ray perpendicular to face normal → fallback to p
    p_perp = np.array([0.0, 1.0, 0.0])
    ps = _compute_p_surface(p_perp, flat_tri["v0"], flat_tri["n"])
    np.testing.assert_allclose(ps, p_perp, atol=1e-10)


def test_p_surface_known_value(flat_tri) -> None:
    # face at x=1, p=[1,0,0] → intersection at x=1, so p_surface = [1,0,0]
    ps = _compute_p_surface(flat_tri["p"], flat_tri["v0"], flat_tri["n"])
    np.testing.assert_allclose(ps, np.array([1.0, 0.0, 0.0]), atol=1e-10)


# ---------------------------------------------------------------
# _pixel_row_positions  (Task 1)
# ---------------------------------------------------------------

def test_pixel_row_positions_shape(flat_tri) -> None:
    sc    = np.array([2.0, 0.0, 0.0])   # screen center
    c_arr = np.arange(80, dtype=float)
    out   = _pixel_row_positions(20, c_arr, sc, flat_tri["e1"], flat_tri["e2"],
                                  10.0, 40, 20)
    assert out.shape == (80, 3)


def test_pixel_row_center_equals_screen_center(flat_tri) -> None:
    cx, cy  = 40, 20
    scale   = 10.0
    sc      = np.array([2.0, 0.0, 0.0])
    c_arr   = np.arange(80, dtype=float)
    # row == cy, col == cx → s=0, u=0 → position should equal screen_center
    row_pos = _pixel_row_positions(cy, c_arr, sc, flat_tri["e1"], flat_tri["e2"],
                                    scale, cx, cy)
    center_col = cx  # c_arr[cx] = cx
    np.testing.assert_allclose(row_pos[cx], sc, atol=1e-10)


def test_pixel_row_positions_vary_along_e2(flat_tri) -> None:
    sc    = np.zeros(3)
    c_arr = np.array([0.0, 1.0, 2.0])
    # scale=1, cx=0 → u = c/(2) increments of 0.5; positions differ along e2
    out   = _pixel_row_positions(0, c_arr, sc, flat_tri["e1"], flat_tri["e2"],
                                  1.0, 0, 0)
    # each step in c should shift by e2/2
    diff = out[1] - out[0]
    np.testing.assert_allclose(diff, 0.5 * flat_tri["e2"], atol=1e-10)



# ---------------------------------------------------------------
# _sphere_row_hits  (Task 3)
# ---------------------------------------------------------------

def test_sphere_row_hits_center_pixel() -> None:
    # Player at [0,0,1], screen center at [0,0,2] (d=1), ray direction -p=[0,0,-1]
    p  = np.array([0.0, 0.0, 1.0])
    sc = np.array([0.0, 0.0, 2.0])   # screen center = p_surface + FOV*p, with p_surface=p
    pixel_row = sc[np.newaxis, :]     # single center pixel
    t = _sphere_row_hits(pixel_row, p)
    assert np.isfinite(t[0])
    # hit point should be on unit sphere
    hit = pixel_row[0] - t[0] * p
    np.testing.assert_allclose(np.linalg.norm(hit), 1.0, atol=1e-6)


def test_sphere_row_hits_miss_returns_inf() -> None:
    p         = np.array([0.0, 0.0, 1.0])
    # Far corner: offset of 2 units in each tangent direction → misses unit sphere
    pixel_row = np.array([[2.0, 2.0, 2.0]])
    t         = _sphere_row_hits(pixel_row, p)
    assert not np.isfinite(t[0])


def test_sphere_row_hit_point_unit_norm() -> None:
    p  = np.array([0.0, 0.0, 1.0])
    e2 = np.array([1.0, 0.0, 0.0])
    sc = p + 1.0 * p   # screen center at [0,0,2]
    # Small offset from center
    pixel_row = (sc + 0.3 * e2)[np.newaxis, :]
    t         = _sphere_row_hits(pixel_row, p)
    if np.isfinite(t[0]):
        hit = pixel_row[0] - t[0] * p
        np.testing.assert_allclose(np.linalg.norm(hit), 1.0, atol=1e-6)


def test_sphere_row_hits_smallest_positive_root() -> None:
    # Ray through sphere hits near side first
    p         = np.array([0.0, 0.0, 1.0])
    pixel_row = np.array([[0.0, 0.0, 3.0]])   # well outside sphere along p
    t         = _sphere_row_hits(pixel_row, p)
    assert np.isfinite(t[0])
    assert t[0] > 0
    # Near intersection should be smaller than far one: t1 = 3-1=2, t2=3+1=4
    np.testing.assert_allclose(t[0], 2.0, atol=1e-6)


# ---------------------------------------------------------------
# _shadow_blocked  (Task 4)
# ---------------------------------------------------------------

def _make_verts(v0, v1, v2):
    """Stack a single triangle into (1,3) arrays for _shadow_blocked."""
    return (np.array([v0]), np.array([v1]), np.array([v2]))


def test_shadow_not_blocked_empty_scene(flat_tri) -> None:
    hit_pos = np.array([1.0, 0.1, 0.1])
    target  = np.array([20.0, 0.1, 0.1])
    empty   = np.zeros((0, 3))
    assert not _shadow_blocked(hit_pos, target, empty, empty, empty)


def test_shadow_blocked_by_triangle(flat_tri) -> None:
    # Blocker: the flat_tri triangle sits at x=1
    # Ray from x=0 to x=5 should be blocked
    v0s, v1s, v2s = _make_verts(flat_tri["v0"], flat_tri["v1"], flat_tri["v2"])
    hit_pos = np.array([0.0, 0.1, 0.1])
    target  = np.array([5.0, 0.1, 0.1])
    assert _shadow_blocked(hit_pos, target, v0s, v1s, v2s)


def test_shadow_not_blocked_when_skip_ct(flat_tri) -> None:
    v0s, v1s, v2s = _make_verts(flat_tri["v0"], flat_tri["v1"], flat_tri["v2"])
    hit_pos = np.array([0.0, 0.1, 0.1])
    target  = np.array([5.0, 0.1, 0.1])
    # Skip the only blocker (index 0) → not blocked
    assert not _shadow_blocked(hit_pos, target, v0s, v1s, v2s, skip_idx=0)


def test_shadow_not_blocked_past_target(flat_tri) -> None:
    # Triangle at x=3, target at x=2 → blocker is behind target
    v0 = np.array([3.0, 0.0, 0.0])
    v1 = np.array([3.0, 1.0, 0.0])
    v2 = np.array([3.0, 0.0, 1.0])
    v0s, v1s, v2s = _make_verts(v0, v1, v2)
    hit_pos = np.array([0.0, 0.1, 0.1])
    target  = np.array([2.0, 0.1, 0.1])
    assert not _shadow_blocked(hit_pos, target, v0s, v1s, v2s)


# ---------------------------------------------------------------
# _compute_brightness  (Task 4)
# ---------------------------------------------------------------

def _empty_verts():
    """Empty stacked triangle arrays for scenes with no blockers."""
    e = np.zeros((0, 3))
    return e, e, e


def _single_verts(flat_tri):
    """Single triangle stacked into (1,3) arrays."""
    return (
        np.array([flat_tri["v0"]]),
        np.array([flat_tri["v1"]]),
        np.array([flat_tri["v2"]]),
    )


def test_brightness_dim_level_when_color_mode_0(flat_tri) -> None:
    e = _empty_verts()
    brt = _compute_brightness(
        flat_tri["v0"], flat_tri["n"], 0, 0,
        color_mode=0, r_max=3.0, sun_pos=None,
        sphere_mode=False, flashlight=False,
        p_src=None, h_proj=None, cos_tmax=0.0,
        v0s=e[0], v1s=e[1], v2s=e[2],
        fl_v0s=e[0], fl_v1s=e[1], fl_v2s=e[2],
    )
    assert abs(brt - _DIM_LEVEL) < 1e-10


def test_brightness_radius_scales_with_distance(flat_tri) -> None:
    hit_near = np.array([1.0, 0.0, 0.0])
    hit_far  = np.array([2.0, 0.0, 0.0])
    r_max    = 3.0
    e = _empty_verts()
    brt_near = _compute_brightness(
        hit_near, flat_tri["n"], 0, 0,
        1, r_max, None, False, False, None, None, 0.0, *e, *e,
    )
    brt_far = _compute_brightness(
        hit_far, flat_tri["n"], 0, 0,
        1, r_max, None, False, False, None, None, 0.0, *e, *e,
    )
    assert brt_far > brt_near


def test_brightness_sun_ambient_when_shadowed(flat_tri) -> None:
    # Place a blocking triangle between hit_pos and sun.
    # hit_idx=1 (some other cone), blocker is at index 0.
    sv = _single_verts(flat_tri)
    e  = _empty_verts()
    hit_pos = np.array([0.0, 0.1, 0.1])
    sun_pos = np.array([20.0, 1.0, 0.5])
    face_n  = np.array([-1.0, 0.0, 0.0])
    brt = _compute_brightness(
        hit_pos, face_n, 1, 0,   # hit_idx=1 (not the blocker), curr_idx=0
        2, 3.0, sun_pos, False, False, None, None, 0.0, *sv, *e,
    )
    assert abs(brt - _SUN_AMBIENT) < 1e-6


def test_brightness_sun_no_occlusion_in_sphere_mode(flat_tri) -> None:
    # In sphere mode occlusion is skipped; face lit by sun should be > ambient
    hit_pos = np.array([0.0, 0.0, 1.0])
    sun_pos = np.array([0.0, 0.0, 20.0])
    face_n  = np.array([0.0, 0.0, 1.0])   # facing sun directly
    e = _empty_verts()
    brt = _compute_brightness(
        hit_pos, face_n, 0, 0,
        2, 3.0, sun_pos, True,   # sphere_mode=True
        False, None, None, 0.0, *e, *e,
    )
    assert brt > _SUN_AMBIENT
