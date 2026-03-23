"""
Curses-based ASCII renderer for the fan and player position on S².
"""

from __future__ import annotations

import curses
from typing import TYPE_CHECKING

import numpy as np

from .player import Player

if TYPE_CHECKING:
    from regfans import Fan
    from _curses import _CursesWindow

_RADIUS_PAIR_START = 6
_EDGE_PAIR_BASE    = 40   # pairs 40-43: front-flip, front-noflip, other-flip, other-noflip
_IREG_BG_PAIR      = 50   # pair for irregular-fan background tint
_FILL_PAIR         = 51   # dim fill for visible surface patches

# (r, g, b) in 0–1000 range for curses
_VIRIDIS_KEYS: list[tuple[int, int, int]] = [
    (267,   4, 329),
    (231, 322, 545),
    (129, 569, 549),
    (173, 694, 478),
    (369, 788, 384),
    (675, 863, 204),
    (867, 902, 114),
    (992, 906, 145),
]


def _viridis_rgb(t: float) -> tuple[int, int, int]:
    t  = max(0.0, min(1.0, t))
    s  = t * (len(_VIRIDIS_KEYS) - 1)
    lo = int(s)
    hi = min(lo + 1, len(_VIRIDIS_KEYS) - 1)
    f  = s - lo
    r0, g0, b0 = _VIRIDIS_KEYS[lo]
    r1, g1, b1 = _VIRIDIS_KEYS[hi]
    return int(r0 + f*(r1-r0)), int(g0 + f*(g1-g0)), int(b0 + f*(b1-b0))


def _project(
    v: np.ndarray,
    p: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> tuple[float, float] | None:
    """
    **Description:**
    Project a 3D vector onto the tangent plane at `p`, returning 2D
    screen coords.
    Returns `None` if the vector is nearly antipodal to `p`.

    **Arguments:**
    - `v`: Vector to project (need not be unit).
    - `p`: Player position (unit vector), normal to tangent plane.
    - `e1`: Tangent basis vector pointing "up" on screen.
    - `e2`: Tangent basis vector pointing "right" on screen.

    **Returns:**
    `(x_screen, y_screen)` or `None` if clipped.
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
    """
    **Description:**
    Return a mapping from each edge (sorted ray-label pair) to the set of cones
    (label tuples) that contain it.

    **Arguments:**
    - `fan`: A `regfans.Fan`.

    **Returns:**
    Dict of `(min_label, max_label)` → set of cone label tuples.
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
    """
    **Description:**
    Draw a line between two screen positions using Bresenham's algorithm.

    **Arguments:**
    - `scr`: Curses window.
    - `r0`, `c0`: Start row and column.
    - `r1`, `c1`: End row and column.
    - `ch`: Character to draw.
    - `attr`: Curses attribute.

    **Returns:**
    Nothing.
    """
    rows, cols = scr.getmaxyx()

    def put(r: int, c: int) -> None:
        if 0 <= r < rows - 1 and 0 <= c < cols - 1:
            try:
                scr.addstr(r, c, ch, attr)
            except curses.error:
                pass

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
) -> None:
    rows, cols = scr.getmaxyx()
    pts_sorted = sorted(pts, key=lambda rc: rc[0])
    (r0, c0), (r1, c1), (r2, c2) = pts_sorted
    if r0 == r2:
        return

    def interp(ra: int, ca: int, rb: int, cb: int, r: int) -> float:
        if ra == rb:
            return float(ca)
        return ca + (cb - ca) * (r - ra) / (rb - ra)

    for r in range(max(0, r0), min(rows - 1, r2 + 1)):
        if r <= r1:
            cl = interp(r0, c0, r1, c1, r)
            cr = interp(r0, c0, r2, c2, r)
        else:
            cl = interp(r1, c1, r2, c2, r)
            cr = interp(r0, c0, r2, c2, r)
        left  = int(round(min(cl, cr)))
        right = int(round(max(cl, cr)))
        for c in range(max(0, left), min(cols - 1, right + 1)):
            try:
                scr.addstr(r, c, ch, attr)
            except curses.error:
                pass


_COLOR_LABELS  = ("none", "radius", "fwd", "sun")
_M3_HEIGHT     = 0.15   # player elevation above sphere (flashlight mode)
_M3_THETA_MAX  = 70.0   # flashlight cone half-angle from heading, degrees

# Fixed world-space lights for "sun" mode (mode 3).
# Primary: diagonal so all cube axes shade differently.
# Fill: nearly opposite primary (offset ~3°) to reveal dark-side variation.
_SUN_DIR: np.ndarray = np.array([1.0, 2.0, 3.0])
_SUN_DIR = _SUN_DIR / float(np.linalg.norm(_SUN_DIR))
_SUN_PERP: np.ndarray = np.cross(_SUN_DIR, np.array([1.0, 0.0, 0.0]))
_SUN_PERP = _SUN_PERP / float(np.linalg.norm(_SUN_PERP))
_SUN_FILL: np.ndarray = -_SUN_DIR + 0.05 * _SUN_PERP
_SUN_FILL = _SUN_FILL / float(np.linalg.norm(_SUN_FILL))
_SUN_AMBIENT      = 0.25   # pale floor — dark side maps to mid viridis, not deep purple
_SUN_PRIMARY_W    = 0.65   # primary contribution
_SUN_FILL_W       = 0.08   # dim fill — just enough to show dark-side topology


def _ray_intersects_triangle(
    orig: np.ndarray,
    d: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> float | None:
    """
    **Description:**
    Möller–Trumbore ray–triangle intersection.

    **Arguments:**
    - `orig`: Ray origin.
    - `d`: Ray direction (need not be normalised).
    - `v0`, `v1`, `v2`: Triangle vertices.

    **Returns:**
    `t > 0` with `orig + t*d` on the triangle, or `None`.
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
) -> None:
    """
    Fill the spherical triangle whose sides are great-circle arcs u–v–w.

    Each screen row is processed as a numpy array so the per-pixel back-
    projection and triangle test are vectorised.  A single addstr call per
    contiguous filled run eliminates per-pixel Python overhead.
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
    rmin, rmax = 0, rows - 2
    cmin, cmax = 0, cols - 2

    # Project each edge normal onto the tangent-plane basis + p once.
    # dot(n, direction) = n_e2*tx + n_e1*ty + n_p*sqrt(1-mag2)
    nAB_e2, nAB_e1, nAB_p = float(np.dot(nAB, e2)), float(np.dot(nAB, e1)), float(np.dot(nAB, p))
    nBC_e2, nBC_e1, nBC_p = float(np.dot(nBC, e2)), float(np.dot(nBC, e1)), float(np.dot(nBC, p))
    nCA_e2, nCA_e1, nCA_p = float(np.dot(nCA, e2)), float(np.dot(nCA, e1)), float(np.dot(nCA, p))

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
        # Draw each contiguous run with a single addstr call.
        gaps   = np.where(np.diff(idx) > 1)[0] + 1
        starts = np.concatenate(([0], gaps))
        ends   = np.concatenate((gaps - 1, [len(idx) - 1]))
        for s, e in zip(starts, ends):
            c0 = cmin + int(idx[s])
            n  = int(idx[e]) - int(idx[s]) + 1
            try:
                scr.addstr(r, c0, ch * n, attr)
            except curses.error:
                pass


def _fill_triangle_colored(
    scr: _CursesWindow,
    pts: list[tuple[int, int]],
    v3d: list[np.ndarray],
    color_fn: object,
    n_pairs: int,
    pair_start: int,
) -> None:
    """
    **Description:**
    Fill a triangle, mapping each pixel's true interpolated 3D position
    through `color_fn` to a Viridis palette index.  The 3D vector at each
    pixel is computed by bilinear interpolation of the corner vectors so
    that, e.g., the radius coloring reflects the actual norm of the
    interpolated point—not a linear blend of corner norms.

    **Arguments:**
    - `scr`: Curses window.
    - `pts`: Three `(row, col)` screen points.
    - `v3d`: Three 3D float vectors corresponding to `pts`.
    - `color_fn`: `(np.ndarray) -> float` mapping an interpolated 3D
      vector to a value in [0, 1].
    - `n_pairs`: Number of colour pairs available.
    - `pair_start`: First colour-pair index.

    **Returns:**
    Nothing.
    """
    rows, cols = scr.getmaxyx()
    order = sorted(range(3), key=lambda i: pts[i][0])
    (r0, c0), (r1, c1), (r2, c2) = [pts[o] for o in order]
    vv0 = np.asarray(v3d[order[0]], dtype=float)
    vv1 = np.asarray(v3d[order[1]], dtype=float)
    vv2 = np.asarray(v3d[order[2]], dtype=float)

    if r0 == r2:
        return

    def _l(a, b, ra: int, rb: int, r: int):  # type: ignore[no-untyped-def]
        return a if ra == rb else a + (b - a) * (r - ra) / (rb - ra)

    for r in range(max(0, r0), min(rows - 1, r2 + 1)):
        if r <= r1:
            cl = _l(c0,  c1,  r0, r1, r)
            cr = _l(c0,  c2,  r0, r2, r)
            vl = _l(vv0, vv1, r0, r1, r)
            vr = _l(vv0, vv2, r0, r2, r)
        else:
            cl = _l(c1,  c2,  r1, r2, r)
            cr = _l(c0,  c2,  r0, r2, r)
            vl = _l(vv1, vv2, r1, r2, r)
            vr = _l(vv0, vv2, r0, r2, r)
        if cl > cr:
            cl, cr, vl, vr = cr, cl, vr, vl
        left, right = int(round(cl)), int(round(cr))
        for c in range(max(0, left), min(cols - 1, right + 1)):
            tc       = (c - left) / (right - left) if right > left else 0.0
            v_interp = vl + tc * (vr - vl)
            t        = max(0.0, min(1.0, color_fn(v_interp)))  # type: ignore
            pair     = pair_start + round(t * (n_pairs - 1))
            try:
                attr = curses.color_pair(pair) | curses.A_BOLD
                scr.addstr(r, c, "\u2592", attr)
            except curses.error:
                pass


class Renderer:
    """
    **Description:**
    Curses-based renderer for a fan and player position on S². Projects 3D cone
    edges as flat line segments onto the tangent plane at the player's position.

    **Arguments:**
    - `fan`: A `regfans.Fan` whose cones will be drawn.
    - `stdscr`: A curses window (full screen).
    """

    def __init__(self, fan: Fan, stdscr: _CursesWindow) -> None:
        """
        **Description:**
        Initialise the renderer with a fan and curses screen.

        **Arguments:**
        - `fan`: A `regfans.Fan`.
        - `stdscr`: A curses window.

        **Returns:**
        Nothing.
        """
        self._fan      = fan
        self._stdscr   = stdscr
        self._edge_map = _cone_edge_map(fan)
        self._init_colors()

    def _init_colors(self) -> None:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN,    -1)
        curses.init_pair(2, curses.COLOR_YELLOW,  -1)
        curses.init_pair(3, curses.COLOR_GREEN,   -1)
        curses.init_pair(4, curses.COLOR_WHITE,   -1)
        curses.init_pair(5, curses.COLOR_WHITE,   -1)
        if curses.can_change_color() and curses.COLORS >= 32:
            n_avail = curses.COLOR_PAIRS - _RADIUS_PAIR_START
            n = max(2, min(32, curses.COLORS - 16, n_avail))
            for i in range(n):
                r, g, b = _viridis_rgb(i / (n - 1))
                curses.init_color(16 + i, r, g, b)
                curses.init_pair(_RADIUS_PAIR_START + i, 16 + i, -1)
            self._n_radius = n
            # Edge-flip indicator colours (dark green / dark red variants).
            _cb = 16 + n   # first free custom colour slot
            if curses.COLORS > _cb + 3 and curses.COLOR_PAIRS > _EDGE_PAIR_BASE + 3:
                curses.init_color(_cb + 0,   0, 800,   0)  # bright green  (front, flip)
                curses.init_color(_cb + 1, 800,   0,   0)  # bright red    (front, no-flip)
                curses.init_color(_cb + 2,   0, 500,   0)  # medium green  (other, flip)
                curses.init_color(_cb + 3, 500,   0,   0)  # medium red    (other, no-flip)
                for j in range(4):
                    curses.init_pair(_EDGE_PAIR_BASE + j, _cb + j, -1)
            else:
                curses.init_pair(_EDGE_PAIR_BASE + 0, curses.COLOR_GREEN, -1)
                curses.init_pair(_EDGE_PAIR_BASE + 1, curses.COLOR_RED,   -1)
                curses.init_pair(_EDGE_PAIR_BASE + 2, curses.COLOR_GREEN, -1)
                curses.init_pair(_EDGE_PAIR_BASE + 3, curses.COLOR_RED,   -1)
            # Irregular-fan background: dark red bg, default fg.
            if curses.COLORS > _cb + 4 and curses.COLOR_PAIRS > _IREG_BG_PAIR:
                curses.init_color(_cb + 4, 280, 0, 0)
                curses.init_pair(_IREG_BG_PAIR, -1, _cb + 4)
            else:
                curses.init_pair(_IREG_BG_PAIR, -1, curses.COLOR_RED)
            # Dim fill for visible surface patches (foreground colour so block
            # chars are visible without needing a background colour).
            if curses.COLORS > _cb + 5 and curses.COLOR_PAIRS > _FILL_PAIR:
                curses.init_color(_cb + 5, 250, 280, 480)  # dim blue-gray
                curses.init_pair(_FILL_PAIR, _cb + 5, -1)
            else:
                curses.init_pair(_FILL_PAIR, curses.COLOR_BLUE, -1)
        else:
            for i, fg in enumerate([curses.COLOR_BLUE, curses.COLOR_CYAN,
                                     curses.COLOR_GREEN, curses.COLOR_YELLOW,
                                     curses.COLOR_RED]):
                curses.init_pair(_RADIUS_PAIR_START + i, fg, -1)
            self._n_radius = 5
            curses.init_pair(_EDGE_PAIR_BASE + 0, curses.COLOR_GREEN, -1)
            curses.init_pair(_EDGE_PAIR_BASE + 1, curses.COLOR_RED,   -1)
            curses.init_pair(_EDGE_PAIR_BASE + 2, curses.COLOR_GREEN, -1)
            curses.init_pair(_EDGE_PAIR_BASE + 3, curses.COLOR_RED,   -1)
            curses.init_pair(_IREG_BG_PAIR, -1, curses.COLOR_RED)
            curses.init_pair(_FILL_PAIR, curses.COLOR_BLUE, -1)

    def draw(
        self,
        player_pos: np.ndarray,
        player_heading: np.ndarray,
        current_cone: tuple[int, ...],
        pointed_facet: tuple[int, int] | None = None,
        locked: bool = False,
        allow_deletion: bool = True,
        color_mode:   int   = 2,
        view_scale:   float = 1.0,
        flip_status:  dict  | None = None,
        is_irregular: bool  = False,
        sphere_mode:  bool  = False,
        agent_active: bool  = False,
    ) -> None:
        """
        **Description:**
        Render one frame: all cone edges as flat projected line segments,
        the active cone highlighted, the pointed-at facet highlighted,
        and the player marker.

        **Arguments:**
        - `player_pos`: Unit vector in R³ giving the player's position on S².
        - `player_heading`: Unit tangent vector at `player_pos` pointing "up".
        - `current_cone`: Label tuple of the cone containing the player.
        - `pointed_facet`: Sorted label pair of the facet the player is
          aiming at, or `None`.
        - `locked`: Whether movement is locked.
        - `allow_deletion`: Whether deletion mode is active.
        - `color_mode`: Fill mode — 0 none, 1 radius, 2 pos, 3 fwd.

        **Returns:**
        Nothing.
        """
        scr = self._stdscr
        scr.bkgd(' ', curses.color_pair(_IREG_BG_PAIR) if is_irregular else 0)
        scr.erase()
        rows, cols = scr.getmaxyx()
        cy, cx = rows // 2, cols // 2
        scale  = float(min(rows, cols // 2) // 2 - 2) * 0.75 * view_scale
        if sphere_mode:
            # Fit the equator (max projected distance = 1.0) to within
            # 2 rows of the screen edge, leaving a small margin.
            scale = float(max(1, rows // 2 - 2))

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
            if np.dot(n, vs[0]) < 0:
                n = -n
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
            n_steps = max(2, int(theta / 0.04))
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

        if color_mode == 1:
            _r_max = float(
                np.linalg.norm(fan.vectors(), axis=1).max()
            ) or 1.0

            def _color_fn(v: np.ndarray) -> float:
                return float(np.linalg.norm(v)) / _r_max

        elif color_mode == 2:
            # Flashlight: cone + occlusion.  Handled per-triangle below
            # once sorted_front is known.
            _p_src    = p * (1.0 + _M3_HEIGHT)
            _cos_tmax = float(np.cos(np.radians(_M3_THETA_MAX)))
            _color_fn = None  # type: ignore[assignment]

        elif color_mode == 3:
            # Sun: two fixed world-space lights; primary + dim fill from
            # nearly-opposite direction to reveal dark-side variation.
            def _color_fn(v: np.ndarray) -> float:  # type: ignore[misc]
                n = float(np.linalg.norm(v))
                if n < 1e-12:
                    return 0.0
                vn      = v / n
                primary = max(0.0, float(np.dot(vn, _SUN_DIR)))
                fill    = max(0.0, float(np.dot(vn, _SUN_FILL)))
                return min(1.0, _SUN_AMBIENT
                           + _SUN_PRIMARY_W * primary
                           + _SUN_FILL_W    * fill)

        else:
            _color_fn = None  # type: ignore[assignment]

        sorted_front = sorted(
            front_cones,
            key=lambda ct: float(
                np.dot(np.mean([ray(l) for l in ct], axis=0), view_dir)
            ),
        )

        if color_mode == 2:
            # Project heading onto the current simplex to get the
            # effective flashlight axis.  This tilts the cone down
            # toward the surface the player is standing on.
            _curr_vv = [np.asarray(ray(l), float) for l in current_cone]
            _curr_nf = np.cross(
                _curr_vv[1] - _curr_vv[0], _curr_vv[2] - _curr_vv[0],
            )
            _curr_nn = float(np.linalg.norm(_curr_nf))
            if _curr_nn > 1e-12:
                _curr_nf = _curr_nf / _curr_nn
            if float(np.dot(_curr_nf, _curr_vv[0])) < 0:
                _curr_nf = -_curr_nf
            _h_raw  = e1_new - float(np.dot(e1_new, _curr_nf)) * _curr_nf
            _h_norm = float(np.linalg.norm(_h_raw))
            _h_proj = _h_raw / _h_norm if _h_norm > 1e-12 else e1_new

            # Build per-face data (vertices, centroid, outward normal).
            _m3_faces: dict = {}
            for _ct0 in sorted_front:
                _vv0 = [np.asarray(ray(l), float) for l in _ct0]
                _c0  = (_vv0[0] + _vv0[1] + _vv0[2]) / 3.0
                _nf0 = np.cross(_vv0[1] - _vv0[0], _vv0[2] - _vv0[0])
                _nn0 = float(np.linalg.norm(_nf0))
                if _nn0 > 1e-12:
                    _nf0 = _nf0 / _nn0
                if float(np.dot(_nf0, _vv0[0])) < 0:
                    _nf0 = -_nf0
                _m3_faces[_ct0] = (_vv0, _c0, _nf0)

            # Determine visibility: cone-angle check + occlusion test.
            _EPS = _M3_HEIGHT * 3.0
            _m3_vis: dict = {}
            for _ct0, (_vv0, _c0, _nf0) in _m3_faces.items():
                _dv    = _c0 - _p_src
                _dist0 = float(np.linalg.norm(_dv))
                if _dist0 < 1e-12:
                    _m3_vis[_ct0] = False
                    continue
                _dir0 = _dv / _dist0
                # Occlusion: no other face may block the ray.
                _ok = True
                for _ct1, (_vv1, _c1, _nf1) in _m3_faces.items():
                    if _ct1 == _ct0:
                        continue
                    _t1 = _ray_intersects_triangle(
                        _p_src, _dir0, _vv1[0], _vv1[1], _vv1[2],
                    )
                    if _t1 is not None and _EPS < _t1 < _dist0 - 1e-3:
                        _ok = False
                        break
                _m3_vis[_ct0] = _ok

        if sphere_mode:
            # Sphere visibility.  Two separate sets:
            #
            #   sphere_front_edge: centroid dot view_dir > 0.  Used for edges.
            #     An edge is drawn if it belongs to any cone in this set AND
            #     at least one of its two vertices is in the front hemisphere
            #     (avoids drawing edges where both endpoints are behind the equator
            #     even though the triangle centroid is just barely front-facing).
            #
            #   sphere_front_fill: any vertex dot view_dir > 0.  Used for fills.
            #     More inclusive so triangles near the equator whose centroid is
            #     slightly behind still have their visible area filled.
            sphere_front_edge: set[tuple[int, ...]] = set()
            sphere_front_fill: set[tuple[int, ...]] = set()
            for ct in all_cones_list:
                vs   = [ray(l) for l in ct]
                dots = [float(np.dot(v, view_dir)) for v in vs]
                if dots[0] + dots[1] + dots[2] > 0:
                    sphere_front_edge.add(ct)
                if any(d > 0 for d in dots):
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
                    _fill_sph_triangle(
                        scr, u, v, w,
                        view_dir, e1_new, e2_new, scale,
                        "\u2591", curses.color_pair(_FILL_PAIR),
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
                    if is_active and flip_status is not None:
                        flippable = flip_status.get(edge, False)
                        ch_e   = "+"
                        attr_e = curses.color_pair(
                            _EDGE_PAIR_BASE + (2 if flippable else 3)
                        )
                    elif is_active:
                        ch_e   = "+"
                        attr_e = curses.color_pair(2) | curses.A_BOLD
                    else:
                        ch_e   = "."
                        attr_e = curses.color_pair(1)
                    _draw_edge(a, b, ch_e, attr_e)
        else:
            for ct in sorted_front:
                clabels = list(ct)
                pts = [screen_pt(l) for l in clabels]
                if any(pt is None for pt in pts):
                    continue
                # Painter's-algorithm occlusion: flood the triangle with background
                # before drawing content so near faces erase far edges beneath them.
                _fill_triangle(scr, pts, " ", 0)  # type: ignore[arg-type]
                if color_mode == 2:
                    if _m3_vis.get(ct, False):
                        _vv_f, _c_f, _nf_f = _m3_faces[ct]
                        # Per-pixel brightness: cone falloff × Lambertian.
                        # cone_t = 1 at heading axis, 0 at cone edge.
                        # lam    = face orientation relative to incoming ray.
                        def _cfn(
                            v:    np.ndarray,
                            _src: np.ndarray = _p_src,
                            _e1:  np.ndarray = _h_proj,
                            _nf:  np.ndarray = _nf_f,
                            _ct:  float      = _cos_tmax,
                        ) -> float:
                            _dv  = v - _src
                            _dn  = float(np.linalg.norm(_dv))
                            if _dn < 1e-12:
                                return 0.0
                            _dir  = _dv / _dn
                            cos_a = float(np.dot(_dir, _e1))
                            if cos_a <= _ct:
                                return 0.0
                            cone_t    = (cos_a - _ct) / (1.0 - _ct)
                            lam       = max(0.0, float(np.dot(_nf, -_dir)))
                            dist_fall = 1.0 / (1.0 + _dn * _dn)
                            return cone_t * cone_t * lam * dist_fall
                        _fill_triangle_colored(
                            scr, pts, _vv_f, _cfn,
                            self._n_radius, _RADIUS_PAIR_START,
                        )
                elif _color_fn is not None:
                    _fill_triangle_colored(  # type: ignore[arg-type]
                        scr, pts,
                        [ray(l) for l in clabels],
                        _color_fn,
                        self._n_radius, _RADIUS_PAIR_START,
                    )
                for i in range(len(clabels)):
                    a, b = clabels[i], clabels[(i + 1) % len(clabels)]
                    edge = (min(a, b), max(a, b))
                    if pointed_facet and edge == pointed_facet:
                        continue
                    is_active = edge in active_edge_set
                    if is_active and flip_status is not None:
                        flippable = flip_status.get(edge, False)
                        ch_e   = "+"
                        attr_e = curses.color_pair(
                            _EDGE_PAIR_BASE + (2 if flippable else 3)
                        )
                    elif is_active:
                        ch_e   = "+"
                        attr_e = curses.color_pair(2) | curses.A_BOLD
                    else:
                        ch_e   = "."
                        attr_e = curses.color_pair(1)
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
                if 0 <= r < rows - 1 and 0 <= c + 1 < cols - 1:
                    try:
                        scr.addstr(r, c, s, attr)
                    except curses.error:
                        pass

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
                if 0 <= _ir < rows - 1:
                    try:
                        scr.addstr(_ir, _ic, _il[: cols - 1 - _ic], _ireg_attr)
                    except curses.error:
                        pass

        facet_str = str(pointed_facet) if pointed_facet else "none"
        hud_base = (
            f" pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})"
            f"  cone={current_cone}"
            f"  facet={facet_str}"
            f"  "
        )
        tail      = "[q]uit"
        agt_str   = "  [A]gent:ON" if agent_active else "  [A]gent:off"
        agt_attr  = (curses.color_pair(2) | curses.A_BOLD
                     if agent_active else curses.color_pair(4))
        sph_str   = "  [S]ph:ON" if sphere_mode else "  [S]ph:off"
        sph_attr  = (curses.color_pair(2) | curses.A_BOLD
                     if sphere_mode else curses.color_pair(4))
        del_str   = "  [D]el:ON" if allow_deletion else "  [D]el:off"
        del_attr  = (curses.color_pair(2) | curses.A_BOLD
                     if allow_deletion else curses.color_pair(4))
        lock_str  = "  [F]ix:ON" if locked else "  [F]ix:off"
        lock_attr = (curses.color_pair(2) | curses.A_BOLD
                     if locked else curses.color_pair(4))
        col_str   = f"  [C]:{_COLOR_LABELS[color_mode]}"
        col = 0
        try:
            scr.addstr(rows - 1, col,
                       hud_base[: cols - 1], curses.color_pair(4))
            col += len(hud_base)
            if col < cols - 1:
                scr.addstr(rows - 1, col,
                           tail[: cols - 1 - col], curses.color_pair(4))
                col += len(tail)
            if col < cols - 1:
                scr.addstr(rows - 1, col, agt_str[: cols - 1 - col], agt_attr)
                col += len(agt_str)
            if col < cols - 1:
                scr.addstr(rows - 1, col, sph_str[: cols - 1 - col], sph_attr)
                col += len(sph_str)
            if col < cols - 1:
                scr.addstr(rows - 1, col, del_str[: cols - 1 - col], del_attr)
                col += len(del_str)
            if col < cols - 1:
                scr.addstr(rows - 1, col,
                           lock_str[: cols - 1 - col], lock_attr)
                col += len(lock_str)
            if col < cols - 1:
                scr.addstr(rows - 1, col,
                           col_str[: cols - 1 - col], curses.color_pair(4))
        except curses.error:
            pass

        # ------------------------------------------------------------------ WIP
        if color_mode == 2:
            wip_lines = [
                "##############################################",
                "##                                          ##",
                "##   W I P  --  FLASHLIGHT  MODE           ##",
                "##   lighting model is not yet correct      ##",
                "##                                          ##",
                "##############################################",
            ]
            wip_attr = curses.color_pair(2) | curses.A_BOLD
            start_row = max(0, rows // 2 - len(wip_lines) // 2 - 8)
            for i, line in enumerate(wip_lines):
                r = start_row + i
                c = max(0, cols // 2 - len(line) // 2)
                if 0 <= r < rows - 1:
                    try:
                        scr.addstr(r, c, line[: cols - 1 - c], wip_attr)
                    except curses.error:
                        pass
        # -------------------------------------------------------------------


def run_display_demo(
    fan: Fan, vc: object, agent: object = None, allow_deletion: bool = False,
) -> None:
    """
    **Description:**
    Launch a curses demo: renders the fan with a player and waits
    for 'q' to quit.

    **Arguments:**
    - `fan`: A `regfans.Fan` to display.
    - `vc`: A `regfans.VectorConfiguration` for circuit queries.
    - `agent`: Optional agent with `.player` and `.advance(fan)`.
      When provided the agent drives movement; arrow keys are
      disabled and the loop runs at ~20 fps.

    **Returns:**
    Nothing.
    """
    TURN       = 0.12
    MAX_SPEED  = 0.15   # top speed (arc-length per keypress)
    MIN_SPEED  = 0.01
    ACCEL      = 0.006  # speed gain per forward keypress
    DECEL      = 0.78   # speed multiplier applied when over-turning
    LAT_ACCEL  = 0.006  # max lateral "acceleration"; critical speed = LAT_ACCEL/TURN
    # Normalise display so vectors of max norm project to a fixed visual radius.
    _TARGET_NORM    = 1.9
    _max_norm       = float(np.linalg.norm(fan.vectors(), axis=1).max()) or 1.0
    _view_scale     = _TARGET_NORM / _max_norm
    _allow_deletion  = allow_deletion   # capture before _main shadows the name
    _pending_prints: list[str] = []    # snapshots queued by 'p', printed after curses exits

    def _snapshot(f: object) -> str:
        """Return a human-readable summary of vectors and simplices."""
        cones  = list(f.cones())      # list of label tuples
        # Collect all labels to enumerate vectors in label order.
        all_labels = sorted({l for cone in cones for l in cone})
        def _fmt_vec(v: np.ndarray) -> str:
            return "[" + ", ".join(str(int(x)) for x in v) + "]"

        vecs_str  = "[" + ", ".join(
            _fmt_vec(f.vectors(which=(l,))[0]) for l in all_labels
        ) + "]"
        labels_str = "[" + ", ".join(str(l) for l in all_labels) + "]"
        simplices_str = "[" + ", ".join(
            "[" + ", ".join(str(l) for l in cone) + "]"
            for cone in sorted(cones)
        ) + "]"
        return (
            f"vectors   = {vecs_str}\n"
            f"labels    = {labels_str}\n"
            f"simplices = {simplices_str}"
        )

    def _main(stdscr: _CursesWindow) -> None:
        curses.curs_set(0)
        stdscr.keypad(True)
        curses.mousemask(0)
        stdscr.timeout(50)   # non-blocking so agent and manual modes can share the loop
        if agent is None:
            player = Player([1.0, 0.2, 0.1], [0.0, 1.0, 0.0])
            from agents.random_agent import RandomAgent as _RA
            _agent_obj = _RA(player)
        else:
            player     = agent.player
            _agent_obj = agent
        renderer       = Renderer(fan, stdscr)
        allow_deletion = _allow_deletion
        locked         = False
        sphere_mode    = False
        color_mode     = 0
        agent_active   = agent is not None
        _speed         = LAT_ACCEL / TURN  # start at the critical speed

        nonlocal_fan  = [fan]
        _irregularity = [not fan.is_regular()]   # updated on every flip

        def _try_move(step: float) -> None:
            f = nonlocal_fan[0]
            old_cone = player.current_cone(f)
            crossed  = player.move(step, f)
            if crossed is None or locked:
                return
            new_cone = player.current_cone(f)
            circ     = player.find_circuit_for_crossing(old_cone, new_cone, f)
            if circ is None:
                return
            if min(circ.signature) == 1 and not allow_deletion:
                return
            new_fan = f.flip(circ)
            if new_fan is None:
                return
            nonlocal_fan[0]    = new_fan
            renderer._fan      = new_fan
            renderer._edge_map = _cone_edge_map(new_fan)
            _irregularity[0]   = not new_fan.is_regular()

        def _agent_step() -> None:
            f        = nonlocal_fan[0]
            old_cone = player.current_cone(f)
            _agent_obj.advance(f)
            if locked:
                return
            new_cone = player.current_cone(f)
            if new_cone == old_cone:
                return
            circ = player.find_circuit_for_crossing(
                old_cone, new_cone, f,
            )
            if circ is None:
                return
            if min(circ.signature) == 1 and not allow_deletion:
                return
            new_fan = f.flip(circ)
            if new_fan is None:
                return
            nonlocal_fan[0]    = new_fan
            renderer._fan      = new_fan
            renderer._edge_map = _cone_edge_map(new_fan)
            _irregularity[0]   = not new_fan.is_regular()

        _agent_rate = 1.0   # steps per frame (can be fractional)
        _agent_acc  = 0.0   # fractional accumulator

        while True:
            if agent_active:
                _agent_acc += _agent_rate
                while _agent_acc >= 1.0:
                    _agent_step()
                    _agent_acc -= 1.0
            f       = nonlocal_fan[0]
            cone    = player.current_cone(f)
            facet   = player.pointed_facet(f)
            # Compute flippability for each edge of the current cone.
            _flip_status: dict[tuple[int, int], bool] = {}
            _cl = list(cone)
            for _i in range(len(_cl)):
                _ea = _cl[_i];  _eb = _cl[(_i + 1) % len(_cl)]
                _ek = (min(_ea, _eb), max(_ea, _eb))
                _adjs = renderer._edge_map.get(_ek, set()) - {cone}
                _ok = False
                for _adj in _adjs:
                    _c = player.find_circuit_for_crossing(cone, _adj, f)
                    if _c is not None and f.flip(_c) is not None:
                        _ok = True
                        break
                _flip_status[_ek] = _ok
            renderer.draw(player.position, player.heading, cone,
                          facet, locked, allow_deletion, color_mode,
                          _view_scale, _flip_status, _irregularity[0],
                          sphere_mode, agent_active)
            stdscr.refresh()
            key = stdscr.getch()
            if   key == ord("q"):  break
            elif key == ord("p"):  _pending_prints.append(_snapshot(nonlocal_fan[0]))
            elif key == ord("a"):  agent_active = not agent_active
            elif key == ord("s"):  sphere_mode = not sphere_mode
            elif key == ord("d"):  allow_deletion = not allow_deletion
            elif key == ord("f"):  locked = not locked
            elif key == ord("c"):  color_mode = (color_mode + 1) % len(_COLOR_LABELS)
            elif not agent_active:
                if key == curses.KEY_UP:
                    _try_move(_speed)
                    _speed = min(MAX_SPEED, _speed + ACCEL)
                    curses.flushinp()   # drop burst of scroll events
                elif key == curses.KEY_DOWN:
                    _try_move(-_speed)
                    _speed = min(MAX_SPEED, _speed + ACCEL)
                    curses.flushinp()
                elif key == curses.KEY_LEFT:
                    player.turn(-TURN)
                    # Brake if requested curvature exceeds centripetal limit.
                    if TURN * _speed > LAT_ACCEL:
                        _speed = max(MIN_SPEED, _speed * DECEL)
                elif key == curses.KEY_RIGHT:
                    player.turn(TURN)
                    if TURN * _speed > LAT_ACCEL:
                        _speed = max(MIN_SPEED, _speed * DECEL)
            elif agent_active:
                _RATE_FACTOR = 1.5
                _RATE_MIN    = 0.05   # 1 step per 20 frames
                _RATE_MAX    = 8.0    # 8 steps per frame
                _NUDGE       = 0.20
                if   key == curses.KEY_UP:
                    _agent_rate = min(_RATE_MAX, _agent_rate * _RATE_FACTOR)
                    curses.flushinp()
                elif key == curses.KEY_DOWN:
                    _agent_rate = max(_RATE_MIN, _agent_rate / _RATE_FACTOR)
                    curses.flushinp()
                elif key == curses.KEY_LEFT:
                    _agent_obj.player.turn(-_NUDGE)
                elif key == curses.KEY_RIGHT:
                    _agent_obj.player.turn(_NUDGE)

    curses.wrapper(_main)
    for i, s in enumerate(_pending_prints):
        if len(_pending_prints) > 1:
            print(f"\n--- snapshot {i + 1} of {len(_pending_prints)} ---")
        print(s)
