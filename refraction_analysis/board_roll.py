"""Split a session's checkerboard into sensor and board anisotropy, and judge the corner by it.

    uv run python refraction_analysis/board_roll.py \\
        --boards /home/chris/data/2026-09-25/Checkerboard \\
        --corner data/corner_2026-09-25_raw.npz

Each board frame's roll is measured from its grid (see `board_roll`). If the
session has only one roll, the 2024 garage set -- the same board and camera,
confirmed by matching distortion -- stands in for the other, and the report says
so: that pairing is cross-session, which is the caveat a same-session pair removes.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fishsense_wuwnet.board_roll import calibrate, detect_boards, roll_of, split  # noqa: E402
from fishsense_wuwnet.brick_calibration import MIN_FACES, reject_views  # noqa: E402
from fishsense_wuwnet.dataset import cache_by_session, load_corner_cache  # noqa: E402

DATA = REPO / "data"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boards", type=Path, nargs="+", required=True,
                        help="folder(s) of raw checkerboard frames, any rolls mixed")
    parser.add_argument("--cache", type=Path, help="board corner cache (default: data/<session>_boards_raw.npz)")
    parser.add_argument("--corner", type=Path, help="a corner-target correspondence cache to judge against it")
    args = parser.parse_args()

    first = args.boards[0]
    session = first.parent.name if first.name.lower().startswith("checker") else first.name
    cache = args.cache or DATA / f"checkerboard_{session}_raw.npz"
    boards = {}
    for folder in args.boards:
        boards.update(detect_boards(folder, cache))
    by_roll = {"horizontal": [], "vertical": []}
    for stem in sorted(boards):
        by_roll[roll_of(boards[stem])].append(boards[stem])
    print(f"{len(boards)} boards found: {len(by_roll['horizontal'])} horizontal, "
          f"{len(by_roll['vertical'])} vertical (long side across / along the image)")

    source = {roll: "this session" for roll in by_roll if len(by_roll[roll]) >= 4}
    missing = [roll for roll in by_roll if len(by_roll[roll]) < 4]
    if missing:
        garage = cache_by_session(load_corner_cache(DATA / "board_corners_raw.npz"), "garage_lens")
        for roll in missing:
            stand_in = [v.reshape(-1, 2).astype(np.float32) for v in garage.values()
                        if roll_of(v.reshape(-1, 2)) == roll]
            if len(stand_in) >= 4:
                by_roll[roll] = stand_in
                source[roll] = f"2024 garage set, {len(stand_in)} frames (cross-session stand-in)"

    result = split(by_roll)
    print()
    for roll in ("horizontal", "vertical"):
        if roll in result:
            r = result[roll]
            print(f"  {roll:10s}: fx {r['K'][0, 0]:.1f}  fy {r['K'][1, 1]:.1f}  fx/fy {r['ratio']:.5f}  "
                  f"rms {r['rms']:.2f}  ({source[roll]})")
    if "sensor" in result:
        print(f"\n  sensor fx/fy = {result['sensor']:.5f}   board print = {100 * (result['board'] - 1):+.2f} %")
        if len(source) < 2 or any("stand-in" in s for s in source.values()):
            print("  (one roll is a cross-session stand-in; copy in the session's other roll to remove that)")

    if args.corner:
        session_views = [v for vs in (by_roll[r] for r in by_roll if source.get(r) == "this session") for v in vs]
        K, dist, _ = calibrate(session_views)
        z = np.load(args.corner)
        views = {k[4:]: (z[f"obj/{k[4:]}"], z[f"img/{k[4:]}"]) for k in z.files if k.startswith("obj/")}
        names = sorted(n for n, v in views.items() if len(v[0]) >= 4 * MIN_FACES)
        c = reject_views(names, [views[n][0] for n in names], [views[n][1] for n in names], K, dist)
        ratio = c.K[0, 0] / c.K[1, 1]
        print(f"\ncorner ({args.corner.name}): {len(c.views)} views, rms {c.rms:.2f}, "
              f"fx {c.K[0, 0]:.1f} ({100 * (c.K[0, 0] / K[0, 0] - 1):+.2f} % vs this session's boards), "
              f"fx/fy {ratio:.5f}")
        if "sensor" in result:
            print(f"  corner fx/fy vs sensor: {100 * (ratio / result['sensor'] - 1):+.2f} %")


if __name__ == "__main__":
    main()
