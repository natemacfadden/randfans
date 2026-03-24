"""
Curses-based ASCII renderer for the fan and player position on S².
"""

from __future__ import annotations

import curses
import math
from typing import TYPE_CHECKING

import numpy as np

from .colors import (
    _init_colors,
    _RADIUS_PAIR_START,
    _EDGE_PAIR_BASE,
    _IREG_BG_PAIR,
    _FILL_PAIR,
)

if TYPE_CHECKING:
    from regfans import Fan
    from _curses import _CursesWindow

_SLERP_STEP   = 0.04   # arc-length step for spherical triangle sampling
_SUN_DISTANCE = 20.0
_SUN_REF      = np.array([1.0, 1.0, 1.0])

_COLOR_LABELS  = ("off", "radius", "sun")
# Symbol styles: (label, ramp_string).  Brightness t∈[0,1] indexes the ramp.
_SYMBOL_STYLES: tuple = (
    ("block",   "\u2591\u2592\u2593\u2588"),   # ░▒▓█  — block shading
    ("digits",  "123456789"),                   # numeric brightness 1–9
    ("ascii",   " .:-=+*#%@"),                  # classic ASCII density ramp
)
_M3_HEIGHT     = 0.003  # player elevation above current face (flashlight mode)
_M3_THETA_MAX  = 55.0   # flashlight cone half-angle from heading, degrees

# Point light for "sun" mode (mode 3).
# Placed diagonally so all cube axes shade differently.
# _SUN_BRIGHTNESS normalises intensity so the closest expected surface
# (at ~1 unit from origin, ~19 units from the sun) maps to roughly 1.
_SUN_POS: np.ndarray = np.array([1.0, 2.0, 3.0])
_SUN_POS = _SUN_POS / float(np.linalg.norm(_SUN_POS)) * _SUN_DISTANCE
_SUN_BRIGHTNESS   = float(np.dot(_SUN_POS - _SUN_REF, _SUN_POS - _SUN_REF))
_SUN_AMBIENT      = 0.12   # base illumination on all surfaces, including shadowed ones
_SUN_MAX          = 0.72   # cap on sun brightness (prevents over-saturation at peak)
_DIM_LEVEL        = 0.45   # default brightness when flashlight is off

_HUD_ROWS = 2  # number of rows reserved at screen bottom for HUD


def _addstr(scr, r: int, c: int, text: str, attr: int = 0) -> None:
    """Write text to the curses screen, silently ignoring out-of-bounds errors."""
    try:
        scr.addstr(r, c, text, attr)
    except curses.error:
        pass


def _orient_normal(n: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Return n flipped so it points toward ref (dot(n, ref) > 0)."""
    return n if np.dot(n, ref) >= 0.0 else -n


def _edge_attrs(
    edge: tuple,
    is_active: bool,
    flip_status: dict | None,
) -> tuple[str, int]:
    """Return (character, curses attr) for rendering an edge."""
    if is_active and flip_status is not None:
        flippable = flip_status.get(edge, False)
        return "+", curses.color_pair(
            _EDGE_PAIR_BASE + (2 if flippable else 3)
        )
    if is_active:
        return "+", curses.color_pair(2) | curses.A_BOLD
    return ".", curses.color_pair(1)


def _project(
    v: np.ndarray,
    p: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> tuple[float, float] | None:
    """Project a 3D vector onto the tangent plane at ``p``.

    Returns 2D screen coordinates, or ``None`` if the vector is nearly
    antipodal to ``p``.

    Parameters
    ----------
    v : np.ndarray
        Vector to project (need not be unit).
    p : np.ndarray
        Player position (unit vector), normal to tangent plane.
    e1 : np.ndarray
        Tangent basis vector pointing "up" on screen.
    e2 : np.ndarray
        Tangent basis vector pointing "right" on screen.

    Returns
    -------
    tuple[float, float] or None
        ``(x_screen, y_screen)``, or ``None`` if clipped.
    """
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    vn = v / n
    if np.dot(vn, p) < -0.95:
        return None
    v_proj = v - np.dot(v, p) * p
    return float(np.dot(v_proj, e2)), float(np.dot(v_proj, e1))


def _cone_edge_map(fan: Fan) -> dict[tuple[int, int], set[tuple[int, ...]]]:
    """Return a mapping from each edge to the set of cones containing it.

    Parameters
    ----------
    fan : regfans.Fan
        The fan to process.

    Returns
    -------
    dict[tuple[int, int], set[tuple[int, ...]]]
        Maps each sorted ray-label pair ``(min_label, max_label)`` to the
        set of cone label tuples that contain that edge.
    """
    edge_map: dict[tuple[int, int], set[tuple[int, ...]]] = {}
    for cone in fan.cones():
        labels = list(cone)
        ct = tuple(sorted(labels))
        for i in range(len(labels)):
            a, b = labels[i], labels[(i + 1) % len(labels)]
            key = (min(a, b), max(a, b))
            edge_map.setdefault(key, set()).add(ct)
    return edge_map


def _draw_line(
    scr: _CursesWindow,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    ch: str,
    attr: int,
) -> None:
    """Draw a line between two screen positions using Bresenham's algorithm.

    Parameters
    ----------
    scr : _CursesWindow
        Curses window.
    r0 : int
        Start row.
    c0 : int
        Start column.
    r1 : int
        End row.
    c1 : int
        End column.
    ch : str
        Character to draw.
    attr : int
        Curses attribute.
    """
    rows, cols = scr.getmaxyx()

    def put(r: int, c: int) -> None:
        if 0 <= r < rows - _HUD_ROWS and 0 <= c < cols - 1:
            _addstr(scr, r, c, ch, attr)

    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    r, c = r0, c0

    if dc >= dr:
        err = dc // 2
        while c != c1:
            put(r, c)
            err -= dr
            if err < 0:
                r += sr
                err += dc
            c += sc
    else:
        err = dr // 2
        while r != r1:
            put(r, c)
            err -= dc
            if err < 0:
                c += sc
                err += dr
            r += sr
    put(r1, c1)


def _fill_triangle(
    scr: _CursesWindow,
    pts: list[tuple[int, int]],
    ch: str,
    attr: int,
    v3d: list[np.ndarray] | None = None,
    shade_fn=None,
    view_dir: np.ndarray | None = None,
    depth_buf: np.ndarray | None = None,
) -> None:
    """Fill a flat-projected triangle.

    If ``v3d`` (three 3-D vertex positions matching ``pts``) and
    ``shade_fn(pos, normal, depth, r, c) -> (ch, attr) | None`` are both
    supplied, the position is linearly interpolated across the triangle and
    the face normal ``normalize(cross(v1−v0, v2−v0))`` is the constant
    normal. ``depth = dot(pos, view_dir)`` if ``view_dir`` is provided,
    else 0. Returning ``None`` from ``shade_fn`` skips the pixel.

    If ``depth_buf`` (rows×cols float array, init to ``-inf``) is supplied,
    each pixel is only drawn when its depth exceeds the stored value.

    Parameters
    ----------
    scr : _CursesWindow
        Curses window.
    pts : list[tuple[int, int]]
        Screen coordinates of the three triangle vertices as (row, col).
    ch : str
        Fill character used when no shade function is supplied.
    attr : int
        Curses attribute used when no shade function is supplied.
    v3d : list[np.ndarray] or None, optional
        Three 3-D vertex positions corresponding to ``pts``.
    shade_fn : callable or None, optional
        Per-pixel shading function with signature
        ``(pos, normal, depth, r, c) -> (ch, attr) | None``.
    view_dir : np.ndarray or None, optional
        View direction used to compute depth.
    depth_buf : np.ndarray or None, optional
        Rows×cols float array (initialised to ``-inf``) for depth testing.
    """
    rows, cols = scr.getmaxyx()

    # Sort vertices by screen row.
    order = sorted(range(3), key=lambda i: pts[i][0])
    (r0, c0), (r1, c1), (r2, c2) = [pts[o] for o in order]
    if r0 == r2:
        return

    # Precompute face normal and sorted 3-D vertices when shading.
    face_normal: np.ndarray | None = None
    vv: list[np.ndarray] = []
    if v3d is not None and shade_fn is not None:
        vv = [np.asarray(v3d[o], dtype=float) for o in order]
        _fn = np.cross(vv[1] - vv[0], vv[2] - vv[0])
        _fn_n = float(np.linalg.norm(_fn))
        face_normal = (_fn / _fn_n) if _fn_n > 1e-12 else np.zeros(3)
        # Ensure normal points away from origin (consistent across view changes).
        _centroid = (vv[0] + vv[1] + vv[2]) / 3.0
        face_normal = _orient_normal(face_normal, _centroid)

    def _il(a: float, b: float, ra: int, rb: int, r: int) -> float:
        return a if ra == rb else a + (b - a) * (r - ra) / (rb - ra)

    for r in range(max(0, r0), min(rows - _HUD_ROWS, r2 + 1)):
        if r <= r1:
            cl = _il(c0, c1, r0, r1, r);  cr = _il(c0, c2, r0, r2, r)
            vl = _il(vv[0], vv[1], r0, r1, r) if vv else None
            vr = _il(vv[0], vv[2], r0, r2, r) if vv else None
        else:
            cl = _il(c1, c2, r1, r2, r);  cr = _il(c0, c2, r0, r2, r)
            vl = _il(vv[1], vv[2], r1, r2, r) if vv else None
            vr = _il(vv[0], vv[2], r0, r2, r) if vv else None
        if cl > cr:
            cl, cr = cr, cl
            if vl is not None:
                vl, vr = vr, vl
        left  = int(round(cl))
        right = int(round(cr))
        for c in range(max(0, left), min(cols - 1, right + 1)):
            if shade_fn is not None and face_normal is not None and vl is not None:
                tc    = (c - left) / (right - left) if right > left else 0.0
                pos   = vl + tc * (vr - vl)
                depth = float(np.dot(pos, view_dir)) if view_dir is not None else 0.0
                if depth_buf is not None and depth <= depth_buf[r, c]:
                    continue
                result = shade_fn(pos, face_normal, depth, r, c)
                if result is None:
                    continue
                if depth_buf is not None:
                    depth_buf[r, c] = depth
                c_ch, c_attr = result
            else:
                c_ch, c_attr = ch, attr
            _addstr(scr, r, c, c_ch, c_attr)


def _ray_intersects_triangle(
    orig: np.ndarray,
    d: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> float | None:
    """Möller–Trumbore ray–triangle intersection.

    Parameters
    ----------
    orig : np.ndarray
        Ray origin.
    d : np.ndarray
        Ray direction (need not be normalised).
    v0 : np.ndarray
        First triangle vertex.
    v1 : np.ndarray
        Second triangle vertex.
    v2 : np.ndarray
        Third triangle vertex.

    Returns
    -------
    float or None
        ``t > 0`` such that ``orig + t*d`` lies on the triangle, or
        ``None`` if there is no intersection.
    """
    e1 = v1 - v0
    e2 = v2 - v0
    h  = np.cross(d, e2)
    a  = float(np.dot(e1, h))
    if abs(a) < 1e-8:
        return None
    f  = 1.0 / a
    s  = orig - v0
    u  = float(f * np.dot(s, h))
    if u < 0.0 or u > 1.0:
        return None
    q  = np.cross(s, e1)
    vv = float(f * np.dot(d, q))
    if vv < 0.0 or u + vv > 1.0:
        return None
    t  = float(f * np.dot(e2, q))
    return t if t > 1e-6 else None


def _fill_sph_triangle(
    scr: _CursesWindow,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    scale: float,
    ch: str,
    attr: int,
    shade_fn=None,
    depth_buf: np.ndarray | None = None,
) -> None:
    """Fill the spherical triangle whose sides are great-circle arcs u–v–w.

    Each screen row is processed as a numpy array so the per-pixel
    back-projection and triangle test are vectorised. A single ``addstr``
    call per contiguous filled run eliminates per-pixel Python overhead.

    If ``shade_fn(pos, normal, depth, r, c) -> (ch, attr) | None`` is
    provided it is called for every filled character. ``pos`` and ``normal``
    are the unit-sphere point (= radial normal) at that pixel; ``depth`` is
    the Z-component ``dot(pos, p)``; ``r, c`` are screen coordinates.
    Returning ``None`` skips the pixel. Without ``shade_fn`` the fast
    run-based path is used.

    If ``depth_buf`` (rows×cols float array, init to ``-inf``) is supplied,
    each pixel is only drawn when its depth exceeds the stored value, and
    the buffer is updated on draw.

    Parameters
    ----------
    scr : _CursesWindow
        Curses window.
    u : np.ndarray
        First vertex of the spherical triangle (unit vector).
    v : np.ndarray
        Second vertex of the spherical triangle (unit vector).
    w : np.ndarray
        Third vertex of the spherical triangle (unit vector).
    p : np.ndarray
        View direction (unit vector); defines the projection centre.
    e1 : np.ndarray
        Tangent basis vector pointing "up" on screen.
    e2 : np.ndarray
        Tangent basis vector pointing "right" on screen.
    scale : float
        Projection scale factor (pixels per unit radius).
    ch : str
        Fill character used on the fast (no shade function) path.
    attr : int
        Curses attribute used on the fast path.
    shade_fn : callable or None, optional
        Per-pixel shading function with signature
        ``(pos, normal, depth, r, c) -> (ch, attr) | None``.
    depth_buf : np.ndarray or None, optional
        Rows×cols float array (initialised to ``-inf``) for depth testing.
    """
    rows, cols = scr.getmaxyx()
    cx, cy = cols // 2, rows // 2

    # Edge normals for spherical point-in-triangle test.
    nAB = np.cross(u, v)
    nBC = np.cross(v, w)
    nCA = np.cross(w, u)
    if float(np.dot(nAB, u + v + w)) < 0.0:
        nAB = -nAB; nBC = -nBC; nCA = -nCA


    # Use the full visible disk as the bounding box.  Arc-sample bounding
    # boxes fail for large triangles (especially those with back-hemisphere
    # vertices) because the arc projections don't bound the front-hemisphere
    # fill region.  The on_sphere + inside tests below handle all clipping.
    rmin, rmax = 0, rows - _HUD_ROWS - 1
    cmin, cmax = 0, cols - 2

    # Project each edge normal onto the tangent-plane basis + p once.
    # dot(n, direction) = n_e2*tx + n_e1*ty + n_p*sqrt(1-mag2)
    nAB_e2 = float(np.dot(nAB, e2))
    nAB_e1 = float(np.dot(nAB, e1))
    nAB_p  = float(np.dot(nAB, p))
    nBC_e2 = float(np.dot(nBC, e2))
    nBC_e1 = float(np.dot(nBC, e1))
    nBC_p  = float(np.dot(nBC, p))
    nCA_e2 = float(np.dot(nCA, e2))
    nCA_e1 = float(np.dot(nCA, e1))
    nCA_p  = float(np.dot(nCA, p))

    sc2 = scale * 2.0
    c_arr = np.arange(cmin, cmax + 1)
    TX    = (c_arr - cx) / sc2        # shape (ncols,)

    for r in range(rmin, rmax + 1):
        ty   = (cy - r) / scale
        mag2 = TX * TX + ty * ty
        on_sphere = mag2 < 1.0
        if not np.any(on_sphere):
            continue
        Z      = np.sqrt(np.maximum(0.0, 1.0 - mag2))  # vectorised sqrt
        inside = (on_sphere
                  & (nAB_e2 * TX + nAB_e1 * ty + nAB_p * Z > 0.0)
                  & (nBC_e2 * TX + nBC_e1 * ty + nBC_p * Z > 0.0)
                  & (nCA_e2 * TX + nCA_e1 * ty + nCA_p * Z > 0.0))
        idx = np.where(inside)[0]
        if idx.size == 0:
            continue
        if shade_fn is not None:
            # Per-character path: compute position/normal for each pixel.
            for j in idx:
                tx    = float(TX[j])
                z     = float(Z[j])
                c_abs = cmin + int(j)
                if depth_buf is not None and z <= depth_buf[r, c_abs]:
                    continue
                pos    = tx * e2 + ty * e1 + z * p   # unit-sphere point; normal = radial
                result = shade_fn(pos, pos, z, r, c_abs)
                if result is None:
                    continue
                if depth_buf is not None:
                    depth_buf[r, c_abs] = z
                c_ch, c_attr = result
                _addstr(scr, r, c_abs, c_ch, c_attr)
        else:
            # Fast path: one addstr per contiguous run.
            gaps   = np.where(np.diff(idx) > 1)[0] + 1
            starts = np.concatenate(([0], gaps))
            ends   = np.concatenate((gaps - 1, [len(idx) - 1]))
            for s, e in zip(starts, ends):
                c0 = cmin + int(idx[s])
                n  = int(idx[e]) - int(idx[s]) + 1
                _addstr(scr, r, c0, ch * n, attr)


class Renderer:
    """Curses-based renderer for a fan and player position on S².

    Projects 3D cone edges as flat line segments onto the tangent plane at
    the player's position.

    Parameters
    ----------
    fan : regfans.Fan
        The fan whose cones will be drawn.
    stdscr : _CursesWindow
        A curses window (full screen).
    """

    def __init__(self, fan: Fan, stdscr: _CursesWindow) -> None:
        self._fan        = fan
        self._stdscr     = stdscr
        self._edge_map   = _cone_edge_map(fan)
        self._depth_buf  = None
        self._depth_shape = (0, 0)
        _init_colors(self)

    def draw(
        self,
        player_pos: np.ndarray,
        player_heading: np.ndarray,
        current_cone: tuple[int, ...],
        pointed_facet: tuple[int, int] | None = None,
        locked: bool = False,
        allow_deletion: bool = True,
        color_mode:   int   = 0,
        view_scale:   float = 1.0,
        flip_status:  dict  | None = None,
        is_irregular: bool  = False,
        sphere_mode:  bool  = False,
        agent_active: bool  = False,
        sun_angle:    float = 0.0,
        flashlight:   bool  = False,
        symbol_mode:  int   = 0,
    ) -> None:
        """Render one frame.

        Draws all cone edges as flat projected line segments, highlights the
        active cone and the pointed-at facet, and renders the player marker.

        Parameters
        ----------
        player_pos : np.ndarray
            Unit vector in R³ giving the player's position on S².
        player_heading : np.ndarray
            Unit tangent vector at ``player_pos`` pointing "up".
        current_cone : tuple[int, ...]
            Label tuple of the cone containing the player.
        pointed_facet : tuple[int, int] or None, optional
            Sorted label pair of the facet the player is aiming at, or
            ``None``.
        locked : bool, optional
            Whether movement is locked.
        allow_deletion : bool, optional
            Whether deletion mode is active.
        color_mode : int, optional
            Fill mode — 0 wireframe, 1 radius, 2 sun.
        view_scale : float, optional
            Projection scale multiplier.
        flip_status : dict or None, optional
            Maps each active edge to a bool indicating flippability.
        is_irregular : bool, optional
            Whether the current fan is irregular.
        sphere_mode : bool, optional
            Whether to render edges as great-circle arcs.
        agent_active : bool, optional
            Whether the agent is currently driving movement.
        sun_angle : float, optional
            Current rotation angle of the sun (radians).
        flashlight : bool, optional
            Whether to overlay the flashlight cone (independent of color
            mode).
        symbol_mode : int, optional
            Index into the symbol ramp styles.
        """
        scr = self._stdscr
        scr.bkgd(' ', curses.color_pair(_IREG_BG_PAIR) if is_irregular else 0)
        scr.erase()
        rows, cols = scr.getmaxyx()
        if self._depth_shape != (rows, cols):
            self._depth_buf   = np.full((rows, cols), -np.inf)
            self._depth_shape = (rows, cols)
        else:
            self._depth_buf.fill(-np.inf)
        depth_buf = self._depth_buf
        cy, cx = rows // 2, cols // 2
        scale  = float(min(rows, cols // 2) // 2 - 2) * 0.75 * view_scale
        if sphere_mode:
            # Fit the equator (max projected distance = 1.0) to within
            # 2 rows of the screen edge, leaving a small margin.
            scale = float(max(1, rows // 2 - _HUD_ROWS - 1))

        p        = player_pos
        e1       = player_heading
        view_dir = p
        e1_new   = e1
        e2_new   = np.cross(p, e1)

        fan    = self._fan
        labels = list(current_cone)
        active_edge_set: set[tuple[int, int]] = set()
        for i in range(len(labels)):
            a, b = labels[i], labels[(i + 1) % len(labels)]
            active_edge_set.add((min(a, b), max(a, b)))

        ray_cache: dict[int, np.ndarray] = {}

        def ray(label: int) -> np.ndarray:
            if label not in ray_cache:
                v = fan.vectors(which=(label,))[0]
                if sphere_mode:
                    n = float(np.linalg.norm(v))
                    if n > 1e-12:
                        v = v / n
                ray_cache[label] = v
            return ray_cache[label]

        front_cones: set[tuple[int, ...]] = set()
        all_cones_list: list[tuple[int, ...]] = []
        cone_normals: dict[tuple[int, ...], np.ndarray] = {}
        for cone in fan.cones():
            clabels = list(cone)
            ct = tuple(sorted(clabels))
            all_cones_list.append(ct)
            vs = [ray(l) for l in clabels]
            n  = np.cross(vs[1] - vs[0], vs[2] - vs[0])
            n  = _orient_normal(n, vs[0])
            nn = np.linalg.norm(n)
            if nn > 1e-12:
                n = n / nn
            cone_normals[ct] = n
            if np.dot(n, view_dir) > 0:
                front_cones.add(ct)

        def screen_pt(label: int) -> tuple[int, int] | None:
            coord = _project(ray(label), view_dir, e1_new, e2_new)
            if coord is None:
                return None
            col = cx + int(round(coord[0] * scale * 2))
            row = cy - int(round(coord[1] * scale))
            return (row, col)

        def _draw_edge(a: int, b: int, ch: str, attr: int) -> None:
            if not sphere_mode:
                ca = _project(ray(a), view_dir, e1_new, e2_new)
                cb = _project(ray(b), view_dir, e1_new, e2_new)
                if ca is None or cb is None:
                    return
                c0 = cx + int(round(ca[0] * scale * 2))
                r0 = cy - int(round(ca[1] * scale))
                c1 = cx + int(round(cb[0] * scale * 2))
                r1 = cy - int(round(cb[1] * scale))
                _draw_line(scr, r0,     c0, r1,     c1, ch, attr)
                _draw_line(scr, r0 + 1, c0, r1 + 1, c1, ch, attr)
                return
            # Sphere mode: trace the great circle arc via SLERP.
            # ray() already returns unit vectors in sphere mode.
            u = ray(a)
            v = ray(b)
            cos_a   = float(np.clip(np.dot(u, v), -1.0, 1.0))
            theta   = float(np.arccos(cos_a))
            n_steps = max(2, int(theta / _SLERP_STEP))
            sin_th  = float(np.sin(theta))
            prev: tuple[int, int] | None = None
            for i in range(n_steps + 1):
                t = i / n_steps
                w = (
                    (np.sin((1.0 - t) * theta) / sin_th) * u
                    + (np.sin(t * theta) / sin_th) * v
                ) if sin_th > 1e-9 else u
                # Clip at the equator: don't draw back-hemisphere arc segments.
                if float(np.dot(w, view_dir)) < 0.0:
                    prev = None
                    continue
                coord = _project(w, view_dir, e1_new, e2_new)
                if coord is None:
                    prev = None
                    continue
                col_w = cx + int(round(coord[0] * scale * 2))
                row_w = cy - int(round(coord[1] * scale))
                if prev is not None:
                    _draw_line(scr, prev[0],     prev[1], row_w,     col_w, ch, attr)
                    _draw_line(scr, prev[0] + 1, prev[1], row_w + 1, col_w, ch, attr)
                prev = (row_w, col_w)

        if color_mode == 1:  # radius
            _r_max = float(
                np.linalg.norm(fan.vectors(), axis=1).max()
            ) or 1.0

        sorted_front = sorted(
            front_cones,
            key=lambda ct: float(
                np.dot(np.mean([ray(l) for l in ct], axis=0), view_dir)
            ),
        )

        if flashlight:
            # Flashlight source: above the current face centroid along its outward
            # normal.  This is correct for any polytope, not just unit-sphere fans.
            _cos_tmax = float(np.cos(np.radians(_M3_THETA_MAX)))
            _curr_vv = [np.asarray(ray(l), float) for l in current_cone]
            _curr_nf = np.cross(
                _curr_vv[1] - _curr_vv[0], _curr_vv[2] - _curr_vv[0],
            )
            _curr_nn = float(np.linalg.norm(_curr_nf))
            if _curr_nn > 1e-12:
                _curr_nf = _curr_nf / _curr_nn
            _curr_nf = _orient_normal(_curr_nf, _curr_vv[0])
            # Source: project p onto the current face plane, then step outward
            # along the face normal.  p (unit sphere) is inside the cube at
            # diagonal positions, so p + eps*nf would place the source inside
            # the face — causing incorrect occlusion near edges.
            _curr_plane_d = float(np.dot(
                (_curr_vv[0] + _curr_vv[1] + _curr_vv[2]) / 3.0, _curr_nf
            ))
            _curr_denom   = float(np.dot(p, _curr_nf))
            if abs(_curr_denom) > 1e-12:
                _t_face = _curr_plane_d / _curr_denom
            else:
                _t_face = float(np.linalg.norm(
                    (_curr_vv[0] + _curr_vv[1] + _curr_vv[2]) / 3.0
                ))
            _p_src = p * _t_face + _M3_HEIGHT * _curr_nf
            _curr_ct = tuple(sorted(current_cone))
            # h_proj: face-plane projection of heading.  Used ONLY for the 3D
            # hemisphere gate (prevents the screen-space cone from leaking
            # around polytope corners).  It is NOT used as the 2D beam
            # direction — cube face normals deviate from p, so projecting
            # e1_new onto the face plane would tilt the on-screen beam away
            # from the heading direction.  The 2D cone always points straight
            # up on screen (= forward, matching the heading direction).
            _cam_denom = float(np.dot(p, _curr_nf))
            if abs(_cam_denom) > 1e-12:
                _r_proj   = float(np.dot(e1_new, _curr_nf)) / _cam_denom
                _h_face_raw = e1_new - _r_proj * p
            else:
                _h_face_raw = e1_new - float(np.dot(e1_new, _curr_nf)) * _curr_nf
            _h_face_norm = float(np.linalg.norm(_h_face_raw))
            _h_proj = _h_face_raw / _h_face_norm if _h_face_norm > 1e-12 else e1_new

            # Build per-face data (vertices, centroid, outward normal).
            _m3_faces: dict = {}
            for _ct0 in sorted_front:
                _vv0 = [np.asarray(ray(l), float) for l in _ct0]
                _c0  = (_vv0[0] + _vv0[1] + _vv0[2]) / 3.0
                _nf0 = np.cross(_vv0[1] - _vv0[0], _vv0[2] - _vv0[0])
                _nn0 = float(np.linalg.norm(_nf0))
                if _nn0 > 1e-12:
                    _nf0 = _nf0 / _nn0
                _nf0 = _orient_normal(_nf0, _vv0[0])
                _m3_faces[_ct0] = (_vv0, _c0, _nf0)

            # Per-face flashlight brightness: occlusion + smooth cone falloff.
            # _curr_ct is excluded from occlusion checks — it is the face the
            # player stands on and its plane always intersects forward rays.
            # Per-face occlusion gate: 1 if face is reachable from _p_src,
            # 0 if blocked.  Cone clipping is done per-pixel in the shade_fn.
            _EPS = 1e-3
            _fl_occluded: dict = {}
            for _ct0, (_vv0, _c0, _nf0) in _m3_faces.items():
                _dv    = _c0 - _p_src
                _dist0 = float(np.linalg.norm(_dv))
                if _dist0 < 1e-12:
                    _fl_occluded[_ct0] = True
                    continue
                _dir0 = _dv / _dist0
                _ok = True
                for _ct1, (_vv1, _c1, _nf1) in _m3_faces.items():
                    if _ct1 == _ct0 or _ct1 == _curr_ct:
                        continue
                    # Skip triangles co-planar with the current face.
                    # _p_src is placed just outside the current face plane, so
                    # co-planar triangles intersect every outgoing ray at t≈0
                    # and would spuriously occlude all adjacent-side faces.
                    if float(np.dot(_nf1, _curr_nf)) > 0.99:
                        continue
                    _t1 = _ray_intersects_triangle(
                        _p_src, _dir0, _vv1[0], _vv1[1], _vv1[2],
                    )
                    if _t1 is not None and _EPS < _t1 < _dist0 - 1e-3:
                        _ok = False
                        break
                _fl_occluded[_ct0] = not _ok

        if color_mode == 2:  # sun
            # Rotate the sun position around the z-axis by sun_angle.
            _sc, _ss = float(np.cos(sun_angle)), float(np.sin(sun_angle))
            _sun_pos_cur = np.array([
                _sc * _SUN_POS[0] - _ss * _SUN_POS[1],
                _ss * _SUN_POS[0] + _sc * _SUN_POS[1],
                _SUN_POS[2],
            ])

            # Build per-triangle sun visibility: face must face the sun and
            # the ray from centroid to sun must not be blocked by another face.
            _sun_all: dict = {}
            for _ct0 in all_cones_list:
                _vv0 = [np.asarray(ray(l), float) for l in _ct0]
                _c0  = (_vv0[0] + _vv0[1] + _vv0[2]) / 3.0
                _nf0 = cone_normals[_ct0]
                _sun_all[_ct0] = (_vv0, _c0, _nf0)

            _EPS_SUN = 1e-3
            # Per-triangle sun factor in [0, 1].
            # Non-occluded faces: max(0, dot(face_normal, sun_dir)) — smoothly
            # fades to 0 at the terminator rather than cutting off abruptly.
            # Occluded faces: 0 (hard shadow, geometrically correct).
            _sun_factor: dict = {}
            for _ct0, (_vv0, _c0, _nf0) in _sun_all.items():
                _to_sun = _sun_pos_cur - _c0
                _dist0  = float(np.linalg.norm(_to_sun))
                if _dist0 < 1e-12:
                    _sun_factor[_ct0] = 0.0
                    continue
                _dir0     = _to_sun / _dist0
                _face_dot = max(0.0, float(np.dot(_nf0, _dir0)))
                # Occlusion: ray from centroid toward sun must clear all faces.
                _ok = True
                for _ct1, (_vv1, _c1, _nf1) in _sun_all.items():
                    if _ct1 == _ct0:
                        continue
                    _t1 = _ray_intersects_triangle(
                        _c0 + _EPS_SUN * _dir0, _dir0,
                        _vv1[0], _vv1[1], _vv1[2],
                    )
                    if _t1 is not None and _t1 < _dist0 - _EPS_SUN:
                        _ok = False
                        break
                _sun_factor[_ct0] = _face_dot if _ok else 0.0

        # Build a per-pixel shade function.
        _n_r    = self._n_radius
        _brt_ref: list = [_DIM_LEVEL]        # still used for per-face occlusion gate
        _sun_factor_ref: list | None = None

        _FL_BOOST = 1.55   # max brightness increase above _DIM_LEVEL
        _sym_ramp = _SYMBOL_STYLES[symbol_mode % len(_SYMBOL_STYLES)][1]

        def _sym_char(t: float) -> str:
            """Map brightness t∈[0,1] to a character from the current ramp."""
            idx = round(t * (len(_sym_ramp) - 1))
            return _sym_ramp[max(0, min(len(_sym_ramp) - 1, idx))]

        def _fl_brightness(pos: np.ndarray, r: int, c: int) -> float:
            """Flashlight boost using a true 3D cone from p_src along h_proj,
            with distance falloff.  Screen coords are unused — the 3D direction
            from the source to the pixel position is what determines membership,
            so the cone cannot bleed around polytope corners."""
            dv   = pos - _p_src
            dist = float(np.linalg.norm(dv))
            if dist < 1e-12:
                fl = 1.0
            else:
                cos_a = float(np.dot(dv / dist, _h_proj))
                if cos_a <= _cos_tmax:
                    return 0.0
                fl = (cos_a - _cos_tmax) / (1.0 - _cos_tmax)
            dist_fall = 1.0 / (1.0 + 20.0 * dist * dist)
            return fl * fl * fl * dist_fall

        if color_mode == 1:
            _r_max_val = _r_max
            def _shade_fn(
                pos: np.ndarray, normal: np.ndarray,
                depth: float, r: int, c: int,
            ):
                fl_b = _fl_brightness(pos, r, c) if (flashlight and _brt_ref[0] > 0) else 0.0
                brt  = _DIM_LEVEL + fl_b * _FL_BOOST
                t = max(0.0, min(1.0, float(np.linalg.norm(pos)) / _r_max_val * brt))
                pair = _RADIUS_PAIR_START + round(t * (_n_r - 1))
                return _sym_char(t), curses.color_pair(pair) | curses.A_BOLD

        elif color_mode == 2:  # sun
            _sun_factor_ref = [1.0]
            def _shade_fn(  # type: ignore[misc]
                pos: np.ndarray, normal: np.ndarray,
                depth: float, r: int, c: int,
            ):
                fl_b = _fl_brightness(pos, r, c) if (flashlight and _brt_ref[0] > 0) else 0.0
                pos_n = float(np.linalg.norm(pos))
                n = pos / pos_n if pos_n > 1e-12 else normal
                to_sun = _sun_pos_cur - pos
                dist   = float(np.linalg.norm(to_sun))
                factor = _sun_factor_ref[0]  # type: ignore[index]
                if dist < 1e-12:
                    t = _SUN_AMBIENT
                else:
                    lam = max(0.0, float(np.dot(n, to_sun / dist)))
                    t = min(_SUN_MAX, _SUN_AMBIENT
                            + factor * (1.0 - _SUN_AMBIENT) * lam * _SUN_BRIGHTNESS / (dist * dist))
                # Additive flashlight: lifts dark areas noticeably, barely
                # visible in already-bright sun-lit areas.
                t = max(0.0, min(1.0, t + fl_b * 0.55))
                pair = _RADIUS_PAIR_START + round(t * (_n_r - 1))
                return _sym_char(t), curses.color_pair(pair) | curses.A_BOLD

        else:
            _shade_fn = None  # type: ignore[assignment]

        if sphere_mode:
            # Sphere visibility: include a cone if any vertex is in the front
            # hemisphere (dot > 0).  Both sets use the same condition; the per-edge
            # check below (`both endpoints behind`) prevents drawing fully invisible
            # arcs.  A centroid check (sum of dots > 0) was previously used for
            # sphere_front_edge, but it excluded equator-straddling cones whose
            # edges are still partially visible — causing many edges to disappear.
            sphere_front_edge: set[tuple[int, ...]] = set()
            sphere_front_fill: set[tuple[int, ...]] = set()
            for ct in all_cones_list:
                vs   = [ray(l) for l in ct]
                dots = [float(np.dot(v, view_dir)) for v in vs]
                if any(d > 0 for d in dots):
                    sphere_front_edge.add(ct)
                    sphere_front_fill.add(ct)

            # Fill pass (color_mode != 0): back-to-front, correct arc-bounded
            # geometry via back-projection.
            if color_mode != 0:
                _sorted_sph = sorted(
                    sphere_front_fill,
                    key=lambda ct: float(
                        np.dot(np.mean([ray(l) for l in ct], axis=0), view_dir)
                    ),
                )
                for ct in _sorted_sph:
                    u, v, w = [ray(l) for l in ct]
                    _brt_ref[0] = _DIM_LEVEL
                    _fill_sph_triangle(
                        scr, u, v, w,
                        view_dir, e1_new, e2_new, scale,
                        "\u2591", curses.color_pair(_FILL_PAIR),
                        shade_fn=_shade_fn,
                        depth_buf=depth_buf if _shade_fn is not None else None,
                    )

            # Arc pass — drawn after fills so arcs always appear on top.
            _drawn_edges: set[tuple[int, int]] = set()
            for ct in sphere_front_edge:
                clabels = list(ct)
                for i in range(len(clabels)):
                    a, b = clabels[i], clabels[(i + 1) % len(clabels)]
                    edge = (min(a, b), max(a, b))
                    if edge in _drawn_edges:
                        continue
                    _drawn_edges.add(edge)
                    # Skip edges where both endpoints are behind the equator.
                    if (float(np.dot(ray(a), view_dir)) < 0 and
                            float(np.dot(ray(b), view_dir)) < 0):
                        continue
                    if pointed_facet and edge == pointed_facet:
                        continue
                    is_active = edge in active_edge_set
                    ch_e, attr_e = _edge_attrs(edge, is_active, flip_status)
                    _draw_edge(a, b, ch_e, attr_e)
        else:
            for ct in sorted_front:
                clabels = list(ct)
                pts = [screen_pt(l) for l in clabels]
                if any(pt is None for pt in pts):
                    continue
                # Painter's-algorithm occlusion: flood the triangle with background
                # before drawing content so near faces erase far edges beneath them.
                v3d_ct = [np.asarray(ray(l), float) for l in clabels]
                _fill_triangle(scr, pts, " ", 0)  # type: ignore[arg-type]
                # Gate: 0 if occluded by flashlight or more than 90° from the
                # face-projected heading in 3D (prevents screen-space cone from
                # leaking around corners). Per-pixel cone clipping in _shade_fn.
                if flashlight:
                    if ct == _curr_ct:
                        _brt_ref[0] = 1
                    else:
                        # Use per-vertex hemisphere check: allow the per-pixel shader
                        # to run if ANY vertex is in the forward hemisphere.  A
                        # centroid check would kill an entire face whose centroid
                        # just barely falls behind h_proj, producing hard cutoff edges.
                        _in_3d_hemi = any(
                            float(np.dot(v - _p_src, _h_proj)) > 0.0
                            for v in v3d_ct
                        )
                        _brt_ref[0] = 0 if (_fl_occluded.get(ct, False) or not _in_3d_hemi) else 1
                else:
                    _brt_ref[0] = _DIM_LEVEL
                if _shade_fn is not None:
                    if color_mode == 2 and _sun_factor_ref is not None:
                        _sun_factor_ref[0] = _sun_factor.get(ct, 0.0)
                    _fill_triangle(
                        scr, pts, "\u2592", 0,  # type: ignore[arg-type]
                        v3d=v3d_ct, shade_fn=_shade_fn,
                        view_dir=view_dir, depth_buf=depth_buf,
                    )
                for i in range(len(clabels)):
                    a, b = clabels[i], clabels[(i + 1) % len(clabels)]
                    edge = (min(a, b), max(a, b))
                    if pointed_facet and edge == pointed_facet:
                        continue
                    is_active = edge in active_edge_set
                    ch_e, attr_e = _edge_attrs(edge, is_active, flip_status)
                    _draw_edge(a, b, ch_e, attr_e)

        if pointed_facet:
            a, b = pointed_facet
            if self._edge_map.get(pointed_facet, set()) & front_cones:
                if flip_status is not None and pointed_facet in active_edge_set:
                    flippable = flip_status.get(pointed_facet, False)
                    attr_e = (curses.color_pair(_EDGE_PAIR_BASE + (0 if flippable else 1))
                              | curses.A_BOLD)
                    _draw_edge(a, b, "*", attr_e)
                else:
                    _draw_edge(a, b, "*", curses.color_pair(5) | curses.A_BOLD)

        coord = _project(p, view_dir, e1_new, e2_new)
        if coord is not None:
            col = cx + int(round(coord[0] * scale * 2))
            row = cy - int(round(coord[1] * scale))
            attr = curses.color_pair(3) | curses.A_BOLD
            for dr, s in ((-1, "^^"), (0, "||")):
                r, c = row + dr, col
                if 0 <= r < rows - _HUD_ROWS and 0 <= c + 1 < cols - 1:
                    _addstr(scr, r, c, s, attr)


        if is_irregular:
            _ireg_lines = [
                "                                    ",
                "   I  R  R  E  G  U  L  A  R        ",
                "                                    ",
            ]
            _ireg_attr = curses.color_pair(_IREG_BG_PAIR) | curses.A_BOLD
            for _ii, _il in enumerate(_ireg_lines):
                _ir = _ii
                _ic = 0
                if 0 <= _ir < rows - _HUD_ROWS:
                    _addstr(scr, _ir, _ic, _il[: cols - 1 - _ic], _ireg_attr)

        facet_str  = str(pointed_facet) if pointed_facet else "none"
        tail       = "[q]uit"
        cone_str   = f"  cone={current_cone}"
        facet_row1 = f"  facet={facet_str}"
        agt_str    = "  [W]agent:ON" if agent_active else "  [W]agent:off"
        agt_attr   = (curses.color_pair(2) | curses.A_BOLD
                      if agent_active else curses.color_pair(4))
        sph_str    = "  [A]sphere:ON" if sphere_mode else "  [A]sphere:off"
        sph_attr   = (curses.color_pair(2) | curses.A_BOLD
                      if sphere_mode else curses.color_pair(4))
        del_str    = "  [D]del:ON" if allow_deletion else "  [D]del:off"
        del_attr   = (curses.color_pair(2) | curses.A_BOLD
                      if allow_deletion else curses.color_pair(4))
        lock_str   = "  [C]fix:ON" if locked else "  [C]fix:off"
        lock_attr  = (curses.color_pair(2) | curses.A_BOLD
                      if locked else curses.color_pair(4))
        col_str    = f"  [S]fill:{_COLOR_LABELS[color_mode]}"
        sym_str    = f"  [Z]sym:{_SYMBOL_STYLES[symbol_mode % len(_SYMBOL_STYLES)][0]}"
        lit_str    = "  [X]light:ON" if flashlight else "  [X]light:off"
        lit_attr   = (curses.color_pair(2) | curses.A_BOLD
                      if flashlight else curses.color_pair(4))
        # ── HUD row 0 (rows-2): [q]uit  cone=…  [A]sphere  [S]:color  [D]el  [W]agent  [F]dbg
        # ── HUD row 1 (rows-1):          facet=…  [Z]:symbol  [X]ight   [C]ix
        #
        # Stacked key columns match physical keyboard columns (Q-A-Z, W-S-X, E-D-C).
        # cone/facet are stacked; [Z]/[X]/[C] align under [A]/[S]/[D] respectively.
        try:
            # Clear HUD rows so the irregular-fan red background doesn't bleed in.
            _blank = " " * (cols - 1)
            for _hr in range(_HUD_ROWS):
                scr.addstr(rows - _HUD_ROWS + _hr, 0, _blank, curses.color_pair(4))

            # --- row 0 ---
            col = 0
            r0  = rows - _HUD_ROWS

            scr.addstr(r0, col, tail[: cols - 1], curses.color_pair(4))
            col += len(tail)

            # Column widths are max(row-0 item, row-1 item) so neither row clips.
            _cone_w = max(len(cone_str), len(facet_row1))
            _sph_w  = max(len(sph_str),  len(sym_str))
            _col_w  = max(len(col_str),  len(lit_str))
            _del_w  = max(len(del_str),  len(lock_str))

            cone_col = col
            if col < cols - 1:
                scr.addstr(r0, col, cone_str[: cols - 1 - col], curses.color_pair(4))
                col += _cone_w

            sph_col = col
            if col < cols - 1:
                scr.addstr(r0, col, sph_str[: cols - 1 - col], sph_attr)
                col += _sph_w

            col_col = col
            if col < cols - 1:
                scr.addstr(r0, col, col_str[: cols - 1 - col], curses.color_pair(4))
                col += _col_w

            del_col = col
            if col < cols - 1:
                scr.addstr(r0, col, del_str[: cols - 1 - col], del_attr)
                col += _del_w

            if col < cols - 1:
                scr.addstr(r0, col, agt_str[: cols - 1 - col], agt_attr)
                col += len(agt_str)

            if col < cols - 1:
                scr.addstr(r0, col, "  [F]dbg"[: cols - 1 - col],
                           curses.color_pair(4))

            # --- row 1 ---
            r1 = rows - 1

            # facet aligns under cone
            # Each row-1 item is truncated to its column span (next col − current col)
            # to prevent overlap when the label is wider than the row-0 counterpart.
            if cone_col < cols - 1:
                scr.addstr(r1, cone_col,
                           facet_row1[: sph_col - cone_col], curses.color_pair(4))

            if sph_col < cols - 1:
                scr.addstr(r1, sph_col,
                           sym_str[: col_col - sph_col], curses.color_pair(4))

            if col_col < cols - 1:
                scr.addstr(r1, col_col,
                           lit_str[: del_col - col_col], lit_attr)

            if del_col < cols - 1:
                scr.addstr(r1, del_col,
                           lock_str[: cols - 1 - del_col], lock_attr)

        except curses.error:
            pass
