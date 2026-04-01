"""
Headless rendering: mock curses screen that dumps to plain text.
Use this for automated visual inspection without a live terminal.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from vcgame3d.game.player        import Player3D
from vcgame3d.renderer.renderer  import _STYLE_ATTR, _HUD_ROWS, _FOV, _NEAR_CLIP
from vcgame3d.game.scene         import build_scene

_ROWS = 40
_COLS = 120


class MockScreen:
    """Minimal curses screen substitute."""
    def __init__(self, rows=_ROWS, cols=_COLS):
        self.rows = rows
        self.cols = cols
        self._buf = [[" "] * cols for _ in range(rows)]

    def getmaxyx(self):
        return self.rows, self.cols

    def erase(self):
        self._buf = [[" "] * self.cols for _ in range(self.rows)]

    def addstr(self, r, c, text, attr=0):
        for i, ch in enumerate(text):
            if 0 <= r < self.rows and 0 <= c + i < self.cols - 1:
                self._buf[r][c + i] = ch

    def addch(self, r, c, ch, attr=0):
        if 0 <= r < self.rows and 0 <= c < self.cols - 1:
            self._buf[r][c] = chr(ch) if isinstance(ch, int) else ch

    def refresh(self): pass

    def dump(self) -> str:
        return "\n".join("".join(row) for row in self._buf)


# ── stub out curses so renderer imports cleanly ───────────────────────
import types, curses as _curses

_CURSES_STUB = types.SimpleNamespace(
    color_pair=lambda n: 0,
    A_BOLD=0,
    COLOR_WHITE=7, COLOR_CYAN=6, COLOR_YELLOW=3, COLOR_GREEN=2,
    error=Exception,
)

# Patch renderer to use MockScreen-compatible calls
import vcgame3d.renderer.renderer as _rend
_orig_addstr = _rend._addstr

def _mock_addstr(scr, r, c, text, attr=0):
    try:
        scr.addstr(r, c, text, attr)
    except Exception:
        pass

_rend._addstr = _mock_addstr

# Patch color_pair to return 0
import curses
_orig_color_pair = curses.color_pair if hasattr(curses, "color_pair") else None


def render_frame(
    player: Player3D,
    label: str = "",
    pts=None,
    edges=None,
    styles=None,
) -> str:
    """Render one frame and return as a string."""
    if pts is None or edges is None or styles is None:
        pts, edges, styles = build_scene()
    scr = MockScreen()

    rows, cols = scr.getmaxyx()
    draw_rows  = rows - _HUD_ROWS
    cx = cols // 2
    cy = draw_rows // 2
    scale = min(draw_rows, cols // 2) * 0.45

    scr.erase()

    # Pre-compute camera-space coordinates
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

    # Draw edges back-to-front (painter's algorithm) with near-plane clipping
    _CH = {"cube": "*", "axis": "+", "grid": "."}
    order = sorted(range(len(edges)),
                   key=lambda k: cam[edges[k][0]][2] + cam[edges[k][1]][2],
                   reverse=True)
    for k in order:
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
        ch = _CH.get(style, ".")
        _bresenham(scr, r0, c0, r1, c1, ch, draw_rows, cols)

    # crosshair
    scr.addstr(cy,     cx - 1, "─")
    scr.addstr(cy,     cx,     "┼")
    scr.addstr(cy,     cx + 1, "─")
    scr.addstr(cy - 1, cx,     "│")
    scr.addstr(cy + 1, cx,     "│")

    # HUD
    pos = player.position
    fwd = player.forward
    up  = player.up
    r0h = rows - _HUD_ROWS
    blank = " " * (cols - 1)
    for hr in range(_HUD_ROWS):
        scr.addstr(r0h + hr, 0, blank)
    scr.addstr(r0h,     0,
        f"pos ({pos[0]:+6.2f},{pos[1]:+6.2f},{pos[2]:+6.2f})  "
        f"fwd ({fwd[0]:+5.2f},{fwd[1]:+5.2f},{fwd[2]:+5.2f})  "
        f"spd {player.speed:.2f}"
    )
    scr.addstr(r0h + 1, 0,
        "[↑↓]pitch [←→]yaw [q/e]roll [w/s]thrust [a/d]strafe [r/f]lift [+/-]spd [esc]quit"
    )

    header = f"── {label} ".ljust(cols - 1, "─") if label else "─" * (cols - 1)
    return header + "\n" + scr.dump()


def _bresenham(scr, r0, c0, r1, c1, ch, max_r, max_c):
    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    for _ in range(dr + dc + 1):
        if 0 <= r < max_r and 0 <= c < max_c - 1:
            scr.addstr(r, c, ch)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc: err -= dc; r += sr
        if e2 <  dr: err += dr; c += sc


# ── control sequence helpers ──────────────────────────────────────────
def _apply(player: Player3D, ops: list) -> Player3D:
    """Apply a list of (method, *args) operations to a player copy."""
    import copy
    p = copy.deepcopy(player)
    for op in ops:
        getattr(p, op[0])(*op[1:])
    return p


# ── test scenarios ────────────────────────────────────────────────────
def run_scenarios() -> str:
    base = dict(position=(0, 0, -5), forward=(0, 0, 1), up=(0, 1, 0))

    # Each entry: (label, player, prediction_text)
    cases: list[tuple[str, Player3D, str]] = []

    # 1. Static views
    cases.append((
        "static: initial (z=-5, looking +z)",
        Player3D(**base),
        "PREDICT: Cube centred on crosshair, front face fills ~half screen width. "
        "Grid recedes toward top. Axes visible bottom-left of cube."
    ))
    cases.append((
        "static: close-up (z=-2)",
        Player3D(position=(0,0,-2), forward=(0,0,1), up=(0,1,0)),
        "PREDICT: Cube larger than initial — edges extend toward screen corners. "
        "Near face very wide, far face smaller (perspective). Grid nearly out of view."
    ))
    cases.append((
        "static: side view (x=-5, looking +x)",
        Player3D(position=(-5,0,0), forward=(1,0,0), up=(0,1,0)),
        "PREDICT: Cube appears as a square (symmetric left/right). "
        "Grid stretches away horizontally. Axes reoriented."
    ))
    cases.append((
        "static: top-down (y=5, looking -y)",
        Player3D(position=(0,5,0), forward=(0,-1,0), up=(0,0,1)),
        "PREDICT: Cube appears as a square from above. "
        "Grid fills the view as a flat plane. Axes point away from camera."
    ))
    cases.append((
        "static: diagonal corner view",
        Player3D(position=(-4,3,-4), forward=(1,-0.5,1), up=(0,1,0)),
        "PREDICT: Cube visible with 3 faces showing (corner perspective). "
        "Asymmetric — no face is parallel to screen. Grid tilted."
    ))
    cases.append((
        "static: inside cube (z=0, y=0, x=0)",
        Player3D(position=(0,0,0), forward=(0,0,1), up=(0,1,0)),
        "PREDICT: Surrounding edges visible from inside. "
        "Far face in centre, near face behind camera (clipped). Grid below."
    ))

    # 2. Control: thrust forward (w) — move from z=-5 to z=-2
    p_thrust = Player3D(**base)
    for _ in range(20): p_thrust.thrust(1.0)
    cases.append((
        "control: thrust forward x20 (should be close-up)",
        p_thrust,
        "PREDICT: Same as close-up view — cube larger, grid nearly out of frame. "
        "Position z should be near -2.0."
    ))

    # 3. Control: thrust backward (s)
    p_back = Player3D(**base)
    for _ in range(20): p_back.thrust(-1.0)
    cases.append((
        "control: thrust backward x20 (should be far away)",
        p_back,
        "PREDICT: Cube very small, centred. Grid lines tightly packed near horizon. "
        "Position z should be near -8.0."
    ))

    # 4. Control: yaw left (←) x20
    p_yaw = Player3D(**base)
    for _ in range(20): p_yaw.yaw(0.04)
    cases.append((
        "control: yaw left x20 (~0.8 rad)",
        p_yaw,
        "PREDICT: Cube shifted to the right of centre (camera turned left). "
        "Forward vector x-component should be negative (pointing left)."
    ))

    # 5. Control: yaw right (→) x20
    p_yaw_r = Player3D(**base)
    for _ in range(20): p_yaw_r.yaw(-0.04)
    cases.append((
        "control: yaw right x20 (~0.8 rad)",
        p_yaw_r,
        "PREDICT: Cube shifted to the left of centre. "
        "Forward x-component should be positive."
    ))

    # 6. Control: pitch up (↑) x20
    p_pitch_u = Player3D(**base)
    for _ in range(20): p_pitch_u.pitch(0.04)
    cases.append((
        "control: pitch up x20 (~0.8 rad)",
        p_pitch_u,
        "PREDICT: Cube shifted downward in frame (camera pitched up). "
        "Forward y-component should be positive. Grid hidden below horizon."
    ))

    # 7. Control: pitch down (↓) x20
    p_pitch_d = Player3D(**base)
    for _ in range(20): p_pitch_d.pitch(-0.04)
    cases.append((
        "control: pitch down x20 (~0.8 rad)",
        p_pitch_d,
        "PREDICT: Cube shifted upward in frame. "
        "Forward y-component should be negative. Grid more visible."
    ))

    # 8. Control: roll right (c) x20
    p_roll = Player3D(**base)
    for _ in range(20): p_roll.roll(0.04)
    cases.append((
        "control: roll right x20 (~0.8 rad)",
        p_roll,
        "PREDICT: Cube rotated counter-clockwise in frame (horizon tilts — left side up, right side down). "
        "Up vector x-component should be positive (tilted to the right, +x). Crosshair still centred."
    ))

    # 9. Control: strafe right (d) x20
    p_strafe = Player3D(**base)
    for _ in range(20): p_strafe.strafe(1.0)
    cases.append((
        "control: strafe right x20",
        p_strafe,
        "PREDICT: Cube shifted to the left of centre (camera moved right). "
        "Position x should be positive (~3.0). Forward vector unchanged."
    ))

    # 10. Control: lift up (r) x20
    p_lift = Player3D(**base)
    for _ in range(20): p_lift.lift(1.0)
    cases.append((
        "control: lift up x20",
        p_lift,
        "PREDICT: Cube shifted downward in frame (camera moved up). "
        "Position y should be positive (~3.0). Grid far below."
    ))

    # 11. Combined: yaw left then thrust
    p_combo = Player3D(**base)
    for _ in range(10): p_combo.yaw(0.04)
    for _ in range(15): p_combo.thrust(1.0)
    cases.append((
        "control: yaw left x10 then thrust x15",
        p_combo,
        "PREDICT: Camera facing diagonally left, has moved forward in that direction. "
        "Cube off to the right, closer than initial. Position x should be negative."
    ))

    # 12. Combined roll + thrust: roll right 45° then thrust — does camera move in rolled direction?
    import math
    p_roll_thrust = Player3D(**base)
    for _ in range(20): p_roll_thrust.roll(0.04)   # ~0.8 rad roll right
    pos_before = p_roll_thrust.position.copy()
    for _ in range(10): p_roll_thrust.thrust(1.0)
    pos_after  = p_roll_thrust.position
    cases.append((
        "control: roll right x20 then thrust x10",
        p_roll_thrust,
        "PREDICT: Camera has rolled right then moved forward along original fwd axis (roll doesn't change fwd). "
        "Position z should increase by ~1.5; x,y unchanged. Cube appears counter-clockwise rotated and slightly closer."
    ))

    # 13. Full 360° yaw — returns to start
    p_360 = Player3D(**base)
    steps = 157   # 157 * 0.04 = 6.28 ≈ 2π
    for _ in range(steps): p_360.yaw(0.04)
    cases.append((
        "control: full 360° yaw (157 steps * 0.04 rad)",
        p_360,
        "PREDICT: Camera back to near-initial orientation. fwd should be ~(0,0,1). "
        "up should be ~(0,1,0). Cube near-centred. State numerical error < 0.01."
    ))

    # 14. Pitch to vertical (looking straight up)
    p_vert = Player3D(**base)
    for _ in range(40): p_vert.pitch(0.04)   # ~1.57 rad ≈ 90°
    cases.append((
        "control: pitch up x40 (~90°, looking straight up)",
        p_vert,
        "PREDICT: Camera looking nearly straight up. fwd.y should be ~+1.0. "
        "Cube and grid both have negative z_cam (behind camera), clipped. "
        "Mostly blank frame expected. Crosshair visible at centre."
    ))

    # 15. Pitch to vertical downward
    p_vert_d = Player3D(**base)
    for _ in range(40): p_vert_d.pitch(-0.04)  # ~-1.57 rad ≈ -90°
    cases.append((
        "control: pitch down x40 (~90°, looking straight down)",
        p_vert_d,
        "PREDICT: Camera looking nearly straight down. fwd.y should be ~-1.0. "
        "Cube and grid both project very far off-screen (camera.up is now +z, grid extends radially). "
        "Mostly blank frame expected. Crosshair visible at centre."
    ))

    # 16. Multiple objects at different depths: camera moved back to z=-10
    p_far = Player3D(position=(0, 0, -10), forward=(0, 0, 1), up=(0, 1, 0))
    cases.append((
        "static: far back (z=-10) — depth ordering check",
        p_far,
        "PREDICT: Cube very small, centred. Grid lines spread very tightly near horizon. "
        "Axis markers tiny. Smaller cube = farther = correct depth ordering."
    ))

    # 17. Camera outside the grid (x=10, y=0, z=-5 looking at origin)
    import numpy as _np
    _dir = _np.array([-10.0, 0.0, 5.0])
    _dir = _dir / _np.linalg.norm(_dir)
    p_outside = Player3D(position=(10, 0, -5), forward=tuple(_dir), up=(0, 1, 0))
    cases.append((
        "static: camera outside grid (x=10, z=-5, looking toward origin)",
        p_outside,
        "PREDICT: Grid visible off to the camera's left. Cube visible roughly centred "
        "or slightly left. Grid lines still render even though camera is beyond grid extent."
    ))

    # 18. Fast movement: large thrust steps
    p_fast = Player3D(**base)
    p_fast.speed = 1.0
    for _ in range(5): p_fast.thrust(1.0)
    cases.append((
        "control: fast thrust x5 (speed=1.0)",
        p_fast,
        "PREDICT: Camera moved from z=-5 to ~z=0 (5 units forward). "
        "Cube surrounds camera (inside view). fwd, up unchanged. Numerically stable."
    ))

    # 19. Numerical stability: 1000 small yaw steps — does up drift?
    p_drift = Player3D(**base)
    for _ in range(1000): p_drift.yaw(0.00628)   # ~1000 * 0.00628 ≈ 2π
    cases.append((
        "control: numerical stability — 1000 small yaw steps (~2π total)",
        p_drift,
        "PREDICT: Camera back near initial orientation after ~360°. "
        "up should be very close to (0,1,0) — drift should be < 0.001. "
        "fwd should be ~(0,0,1). Crosshair at centre."
    ))

    frames = []
    for label, player, prediction in cases:
        frame = render_frame(player, label)
        pos = player.position
        fwd = player.forward
        up  = player.up
        actual_info = (
            f"ACTUAL state — pos: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})  "
            f"fwd: ({fwd[0]:+.3f}, {fwd[1]:+.3f}, {fwd[2]:+.3f})  "
            f"up: ({up[0]:+.3f}, {up[1]:+.3f}, {up[2]:+.3f})"
        )
        frames.append(prediction + "\n" + actual_info + "\n" + frame)

    return "\n\n".join(frames)


if __name__ == "__main__":
    print(run_scenarios())
