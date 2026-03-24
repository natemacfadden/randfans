"""
One-time script to fetch all 4319 reflexive polytope vector sets and save
them to reflexive_cache.json in this directory.

Data source: http://coates.ma.ic.ac.uk/3DReflexivePolytopes/
(Coates–Corti–Galkin–Golyshev–Kasprzyk 3D Reflexive Polytopes database)

Run once from the repo root:
    python -m shapes._fetch_reflexive_cache
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from shapes.reflexive import reflexive_vectors, ReflexiveFetchError, N_POLYTOPES

_OUT = Path(__file__).parent / "reflexive_cache.json"
_WORKERS = 20


def _fetch(pid: int) -> tuple[int, list | str]:
    try:
        return pid, reflexive_vectors(pid)
    except (ReflexiveFetchError, ValueError) as exc:
        return pid, f"ERROR: {exc}"


def main() -> None:
    cache: dict[str, list] = {}
    errors: list[str] = []

    print(f"Fetching {N_POLYTOPES} polytopes with {_WORKERS} workers…")

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {pool.submit(_fetch, pid): pid for pid in range(N_POLYTOPES)}
        for i, fut in enumerate(as_completed(futures), 1):
            pid, result = fut.result()
            if isinstance(result, str):
                errors.append(f"  polytope {pid}: {result}")
            else:
                cache[str(pid)] = result
            if i % 200 == 0 or i == N_POLYTOPES:
                print(f"  {i}/{N_POLYTOPES} done, {len(errors)} errors so far")

    _OUT.write_text(json.dumps(cache, separators=(",", ":")))
    print(f"\nSaved {len(cache)} entries to {_OUT}")
    if errors:
        print(f"{len(errors)} errors:")
        for e in errors:
            print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
