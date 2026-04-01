"""
Reference scene: a unit cube, coordinate axes, and an xz ground grid.
All geometry is (points, edges) where edges index into points.
"""
import numpy as np

# ── unit cube ────────────────────────────────────────────────────────
_CUBE_PTS = [
    (-1, -1, -1), ( 1, -1, -1), ( 1,  1, -1), (-1,  1, -1),
    (-1, -1,  1), ( 1, -1,  1), ( 1,  1,  1), (-1,  1,  1),
]
_CUBE_EDGES = [
    (0,1),(1,2),(2,3),(3,0),   # back face
    (4,5),(5,6),(6,7),(7,4),   # front face
    (0,4),(1,5),(2,6),(3,7),   # connecting
]

# ── coordinate axes (length 2) ───────────────────────────────────────
# origin + tips for x, y, z
_AXIS_PTS   = [(0,0,0), (2,0,0), (0,2,0), (0,0,2)]
_AXIS_EDGES = [(0,1), (0,2), (0,3)]   # x, y, z

# ── ground grid (xz plane, y=−1, from −3 to 3 step 1) ───────────────
def _make_grid(y=-1.0, lo=-3, hi=3):
    pts, edges = [], []
    # x lines (varying x, fixed z)
    for z in range(lo, hi + 1):
        i = len(pts)
        pts.append((lo, y, z))
        pts.append((hi, y, z))
        edges.append((i, i + 1))
    # z lines (fixed x, varying z)
    for x in range(lo, hi + 1):
        i = len(pts)
        pts.append((x, y, lo))
        pts.append((x, y, hi))
        edges.append((i, i + 1))
    return pts, edges

_GRID_PTS, _GRID_EDGES = _make_grid(y=0.0)


def build_scene():
    """Return (points, edges, edge_styles) for the full reference scene.

    edge_styles: list of str, one per edge — 'cube', 'axis', 'grid'
    """
    pts    = []
    edges  = []
    styles = []

    def _add(src_pts, src_edges, style):
        offset = len(pts)
        pts.extend(src_pts)
        for a, b in src_edges:
            edges.append((a + offset, b + offset))
            styles.append(style)

    _add(_CUBE_PTS,  _CUBE_EDGES,  "cube")
    _add(_AXIS_PTS,  _AXIS_EDGES,  "axis")
    _add(_GRID_PTS,  _GRID_EDGES,  "grid")

    return [np.array(p, dtype=float) for p in pts], edges, styles
