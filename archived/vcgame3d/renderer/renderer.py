"""
Curses perspective renderer for 3D flight.
"""
from __future__ import annotations

import curses
import numpy as np
from ..game.player import Player3D

_HUD_ROWS  = 4
_FOV       = 1.2    # focal length; larger = narrower FOV
_NEAR_CLIP = 0.5

# curses color pair indices
_CP_DEFAULT   = 1
_CP_CUBE      = 2
_CP_AXIS      = 3
_CP_GRID      = 4
_CP_HUD       = 5
_CP_LABEL      = 6   # vertex labels
_CP_FACE       = 7   # simplex face fill (cyan)
_CP_HIGHLIGHT  = 8   # highlighted simplex edges
_CP_FACE_FRONT = 9   # front-facing simplex face (green)

# Fill character progressions per face type (sparse → dense).
# Face type = face_idx % 4.
_FACE_FILL_TYPES = [
    list("⠁⠃⠇⡇⣇⣷⣿"),   # 0: braille dots  (7 levels)
    list("▏▎▍▌▋▊▉█"),     # 1: vertical bars  (8 levels, increasing width)
    list("▁▂▃▄▅▆▇█"),     # 2: horizontal bars (8 levels, increasing height)
    list("░▒▓█"),          # 3: block fill     (4 levels)
]


def _fill_char(face_idx: int, proximity: float) -> str:
    """Map face type + proximity [0,1] to a fill character."""
    levels = _FACE_FILL_TYPES[face_idx % len(_FACE_FILL_TYPES)]
    idx = min(int(proximity * len(levels)), len(levels) - 1)
    return levels[idx]


def init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(_CP_DEFAULT,    curses.COLOR_WHITE,  -1)
    curses.init_pair(_CP_CUBE,       curses.COLOR_CYAN,   -1)
    curses.init_pair(_CP_AXIS,       curses.COLOR_YELLOW, -1)
    curses.init_pair(_CP_GRID,       curses.COLOR_WHITE,  -1)
    curses.init_pair(_CP_HUD,        curses.COLOR_GREEN,  -1)
    curses.init_pair(_CP_LABEL,      curses.COLOR_YELLOW, -1)
    curses.init_pair(_CP_FACE,       curses.COLOR_CYAN,   -1)
    curses.init_pair(_CP_HIGHLIGHT,  curses.COLOR_WHITE,  -1)
    curses.init_pair(_CP_FACE_FRONT, curses.COLOR_GREEN,  -1)


_STYLE_ATTR: dict[str, tuple[str, int]] = {
    # style -> (draw char, color pair)
    "cube":    ("*", _CP_CUBE),
    "axis":    ("+", _CP_AXIS),
    "grid":    (".", _CP_GRID),
    "simplex": ("*", _CP_CUBE),
}


def _project(p_world: np.ndarray, player: Player3D):
    """Project world point to (sx, sy) normalised screen coords.
    Returns None if behind near clip plane."""
    rel = p_world - player._pos
    x   = float(np.dot(rel, player.right))
    y   = float(np.dot(rel, player._up))
    z   = float(np.dot(rel, player._fwd))
    if z < _NEAR_CLIP:
        return None
    return x / z * _FOV, -y / z * _FOV   # negate y: rows increase downward


def _addstr(scr, r: int, c: int, text: str, attr: int = 0) -> None:
    try:
        scr.addstr(r, c, text, attr)
    except curses.error:
        pass


def _draw_line(scr, r0, c0, r1, c1, ch: str, attr: int, max_r: int, max_c: int) -> None:
    """Bresenham line between two screen points."""
    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    for _ in range(dr + dc + 1):
        if 0 <= r < max_r and 0 <= c < max_c - 1:
            _addstr(scr, r, c, ch, attr)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc; r += sr
        if e2 <  dr:
            err += dr; c += sc


def _bresenham_pts(r0, c0, r1, c1):
    """Yield (r, c) pairs along the Bresenham line from (r0,c0) to (r1,c1)."""
    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    for _ in range(dr + dc + 1):
        yield r, c
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc: err -= dc; r += sr
        if e2 <  dr: err += dr; c += sc


def _draw_face(scr, cam_arc_lists, cx, cy, scale, draw_rows, cols,
               fill: str = "░", attr: int = 0) -> None:
    """Fill the interior of a curved triangle face.

    cam_arc_lists : list of 3 arcs.  Each arc is a list of (x, y, z)
                    camera-space tuples ordered along the arc.
    fill : fill character to use.
    attr : curses attribute (color pair etc.).
    The fill is drawn BEFORE edges so that edges always render on top
    (naturally semi-transparent).

    When any arc vertex is behind the camera the face wraps around the
    viewport: fill the entire drawing area instead of scanline filling.
    """
    # Detect whether the face surrounds the camera (any vertex behind clip plane)
    any_behind = any(z < _NEAR_CLIP for arc in cam_arc_lists for (x, y, z) in arc)
    if any_behind:
        for r in range(draw_rows):
            for c in range(cols - 1):
                _addstr(scr, r, c, fill, attr)
        return

    row_to_cols: dict[int, list[int]] = {}

    for cam_arc in cam_arc_lists:
        for k in range(len(cam_arc) - 1):
            x0, y0, z0 = cam_arc[k]
            x1, y1, z1 = cam_arc[k + 1]

            sc0 = cx + int(round(x0/z0 * _FOV * scale * 2))
            sr0 = cy + int(round(-y0/z0 * _FOV * scale))
            sc1 = cx + int(round(x1/z1 * _FOV * scale * 2))
            sr1 = cy + int(round(-y1/z1 * _FOV * scale))

            for r, c in _bresenham_pts(sr0, sc0, sr1, sc1):
                row_to_cols.setdefault(r, []).append(c)

    # Scanline fill between leftmost and rightmost boundary pixel per row
    for r, col_list in row_to_cols.items():
        if not (0 <= r < draw_rows):
            continue
        lo, hi = min(col_list), max(col_list)
        for c in range(lo, hi + 1):
            if 0 <= c < cols - 1:
                _addstr(scr, r, c, fill, attr)


def draw(
    scr,
    player:  Player3D,
    pts:     list,
    edges:   list,
    styles:  list,
    hud:     bool = True,
    edge_labels:       list | None = None,      # per-edge (min_label, max_label)
    highlighted_pairs: frozenset | None = None, # label pairs of current cone edges
    face_arc_pts:      list | None = None,      # list of (face_idx, proximity, 3 arc lists)
    show_only_simplex: bool = False,            # hide edges outside current cone
) -> None:
    rows, cols = scr.getmaxyx()
    draw_rows  = rows - _HUD_ROWS
    cx = cols // 2
    cy = draw_rows // 2
    scale = min(draw_rows, cols // 2) * 0.45

    scr.erase()

    # Pre-compute camera-space coordinates for all points
    right = player._right
    up    = player._up
    fwd   = player._fwd
    pos   = player._pos
    cam = []
    for p in pts:
        rel = p - pos
        cam.append((float(np.dot(rel, right)),
                    float(np.dot(rel, up)),
                    float(np.dot(rel, fwd))))

    # ── Phase 1: face fills (drawn first so edges render on top) ─────
    # face_arc_pts: list of (face_idx, proximity, list-of-3-arcs)
    if face_arc_pts:
        # Project all faces to camera space
        face_data = []   # (mean_z, any_behind, face_idx, proximity, cam_arcs)
        for face_idx, proximity, face_world_arcs in face_arc_pts:
            cam_arcs = []
            z_vals = []
            for arc_world in face_world_arcs:
                cam_arc = []
                for p in arc_world:
                    rel = p - pos
                    xc = float(np.dot(rel, right))
                    yc = float(np.dot(rel, up))
                    zc = float(np.dot(rel, fwd))
                    cam_arc.append((xc, yc, zc))
                    z_vals.append(zc)
                cam_arcs.append(cam_arc)
            mean_z    = sum(z_vals) / len(z_vals) if z_vals else 0.0
            any_behind = any(z < _NEAR_CLIP for z in z_vals)
            face_data.append((mean_z, any_behind, face_idx, proximity, cam_arcs))

        # Determine front face (= face at the crosshair) from rendered geometry.
        # "any_behind" faces wrap around the viewport and always cover the crosshair;
        # among those, the one with highest mean_z (drawn last) wins.
        # Otherwise, use whichever face's screen-space arc centroid is closest to
        # (0, 0) — the normalised screen centre where the crosshair sits.
        behind_candidates = [(mz, fi) for mz, ab, fi, pr, ca in face_data if ab]
        if behind_candidates:
            front_face_idx = max(behind_candidates, key=lambda t: t[0])[1]
        else:
            front_face_idx = None
            min_d2 = float('inf')
            for mz, ab, fi, pr, cam_arcs in face_data:
                pts2d = [
                    (x / z * _FOV, -y / z * _FOV)
                    for arc in cam_arcs for (x, y, z) in arc if z > _NEAR_CLIP
                ]
                if pts2d:
                    sx = sum(p[0] for p in pts2d) / len(pts2d)
                    sy = sum(p[1] for p in pts2d) / len(pts2d)
                    d2 = sx * sx + sy * sy
                    if d2 < min_d2:
                        min_d2 = d2
                        front_face_idx = fi

        # Draw back-to-front (lowest mean_z first) so nearer faces overwrite
        face_data.sort(key=lambda t: t[0])
        for mean_z, any_behind, face_idx, proximity, cam_arcs in face_data:
            fill_ch   = _fill_char(face_idx, proximity)
            face_attr = curses.color_pair(
                _CP_FACE_FRONT if face_idx == front_face_idx else _CP_FACE
            )
            _draw_face(scr, cam_arcs, cx, cy, scale, draw_rows, cols,
                       fill=fill_ch, attr=face_attr)

    # ── Phase 2: edges back-to-front (painter's algorithm) ───────────
    order = sorted(range(len(edges)),
                   key=lambda k: cam[edges[k][0]][2] + cam[edges[k][1]][2],
                   reverse=True)
    for k in order:
        # Optionally hide edges outside the current cone
        if (show_only_simplex and highlighted_pairs is not None
                and edge_labels is not None
                and edge_labels[k] not in highlighted_pairs):
            continue
        (i, j), style = edges[k], styles[k]
        x0, y0, z0 = cam[i]
        x1, y1, z1 = cam[j]
        if z0 < _NEAR_CLIP and z1 < _NEAR_CLIP:
            continue
        if z0 < _NEAR_CLIP:
            t = (_NEAR_CLIP - z0) / (z1 - z0)
            x0, y0, z0 = x0 + t*(x1-x0), y0 + t*(y1-y0), _NEAR_CLIP
        elif z1 < _NEAR_CLIP:
            t = (_NEAR_CLIP - z1) / (z0 - z1)
            x1, y1, z1 = x1 + t*(x0-x1), y1 + t*(y0-y1), _NEAR_CLIP
        c0 = cx + int(round(x0/z0 * _FOV * scale * 2))
        r0 = cy + int(round(-y0/z0 * _FOV * scale))
        c1 = cx + int(round(x1/z1 * _FOV * scale * 2))
        r1 = cy + int(round(-y1/z1 * _FOV * scale))

        # Highlight edges belonging to the current simplex
        if (highlighted_pairs is not None and edge_labels is not None
                and edge_labels[k] in highlighted_pairs):
            ch   = "*"
            attr = curses.color_pair(_CP_HIGHLIGHT) | curses.A_BOLD
        else:
            ch, cp = _STYLE_ATTR.get(style, (".", _CP_DEFAULT))
            attr   = curses.color_pair(cp)

        _draw_line(scr, r0, c0, r1, c1, ch, attr, draw_rows, cols)

    # crosshair (┼ at true centre cx)
    attr = curses.color_pair(_CP_DEFAULT) | curses.A_BOLD
    _addstr(scr, cy,     cx - 1, "─", attr)
    _addstr(scr, cy,     cx,     "┼", attr)
    _addstr(scr, cy,     cx + 1, "─", attr)
    _addstr(scr, cy - 1, cx,     "│", attr)
    _addstr(scr, cy + 1, cx,     "│", attr)

    # HUD
    _draw_hud(scr, rows, cols, player, hud)


def _draw_hud(scr, rows: int, cols: int, player: Player3D, hud: bool = True) -> None:
    attr_hud = curses.color_pair(_CP_HUD)
    r0       = rows - _HUD_ROWS
    blank    = " " * (cols - 1)
    for hr in range(_HUD_ROWS):
        _addstr(scr, r0 + hr, 0, blank, attr_hud)

    if not hud:
        _addstr(scr, r0, 0, "[H]UD off", attr_hud)
        return

    pos = player.position
    fwd = player.forward
    up  = player.up

    line0 = (
        f"pos ({pos[0]:+6.2f}, {pos[1]:+6.2f}, {pos[2]:+6.2f})  "
        f"fwd ({fwd[0]:+5.2f}, {fwd[1]:+5.2f}, {fwd[2]:+5.2f})  "
        f"up ({up[0]:+5.2f}, {up[1]:+5.2f}, {up[2]:+5.2f})  "
        f"spd {player.speed:.2f}"
    )
    line1 = "[↑↓] pitch   [←→] yaw   [q/e] roll   [w/s] thrust   [a/d] strafe   [r/f] lift"
    line2 = "[+/-] speed  [space] brake  [H]UD on  [esc] quit"

    _addstr(scr, r0,     0, line0[:cols - 1], attr_hud)
    _addstr(scr, r0 + 1, 0, line1[:cols - 1], attr_hud)
    _addstr(scr, r0 + 2, 0, line2[:cols - 1], attr_hud)
