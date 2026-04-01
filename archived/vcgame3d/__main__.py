"""
vcgame3d entry point.

Usage
-----
python -m vcgame3d                        # 4-simplex fan (default, no dependencies)
python -m vcgame3d --fan crosspolytope    # 4D cross-polytope fan (needs regfans)
python -m vcgame3d --fan cytools [h11]    # reflexive fan from CYTools (h11 default: 3)
python -m vcgame3d --fan cube             # reference cube/grid scene
python -m vcgame3d --fan cytools [h11] --subdivisions N
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
from vcgame3d.game.loop import run


def _build_fan(fan_type: str, h11: int):
    if fan_type == "cube":
        return None   # signals loop to use the reference cube/grid scene

    if fan_type == "simplex4d":
        from vcgame3d.game.fan_scene import _Simplex4dFan
        return _Simplex4dFan()

    if fan_type == "crosspolytope":
        from vcgame3d.game.fan_scene import _crosspolytope_fan
        return _crosspolytope_fan()

    if fan_type == "cytools":
        try:
            from cytools import fetch_polytopes
        except ImportError:
            sys.exit("CYTools not available. "
                     "Try: conda run -n cytools-dev python -m vcgame3d --fan cytools")
        polys = fetch_polytopes(h11=h11, limit=1)
        if not polys:
            sys.exit(f"No reflexive polytopes found for h11={h11}")
        return polys[0].triangulate().fan()

    sys.exit(f"Unknown --fan value: {fan_type!r}. "
             f"Use 'simplex4d', 'crosspolytope', 'cytools', or 'cube'.")


def main():
    parser = argparse.ArgumentParser(prog="python -m vcgame3d")
    parser.add_argument("--fan", metavar="TYPE", default="simplex4d",
                        help="Scene: 'simplex4d' (default), 'crosspolytope', 'cytools', 'cube'")
    parser.add_argument("h11", nargs="?", type=int, default=3,
                        help="h11 for CYTools reflexive polytope (default: 3)")
    parser.add_argument("--subdivisions", type=int, default=4,
                        help="Arc subdivisions per fan edge (default: 4)")
    parser.add_argument("--proj", metavar="TYPE", default="spherical",
                        choices=["spherical", "stereo"],
                        help="Projection: 'spherical' (default) or 'stereo'")
    args = parser.parse_args()

    fan = _build_fan(args.fan, args.h11)
    run(fan=fan, n_subdivisions=args.subdivisions, proj=args.proj)


main()
