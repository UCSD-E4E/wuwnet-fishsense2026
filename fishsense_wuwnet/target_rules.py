"""The rules a brick calibration target has to satisfy, checked against a model file.

Each rule is here because something measured says it matters, and the threshold
comes from that measurement rather than from taste. `check` runs all of them on a
Studio `.io` or a bare `.ldr` and says which pass, so a redesign is judged by
running it rather than by reading it.

1. *Sourcing*: every part, in every colour, comes out of one LEGO Classic 10698.
2. *Bricks carry the features*: plates only as a base. A plate course inside a
   wall is a 3.2 mm stripe, too thin to detect and a break in every seam.
3. *Palette*: every visible brick is a colour the classifier reads (see
   `COLOUR_CLASS`). Dark and washed-out colours are not detected at all.
4. *Joints*: no two touching visible faces read as the same colour group, or
   the detector sees one blob where there are two bricks.
5. *Locally distinguishable*: 3-course by 4-stud colour windows repeat no more
   than on the build that was photographed and calibrated correctly -- 19 %
   ambiguous, none repeated more than five times.
6. *Running bond*: every seam intersection is a T. Where four bricks meet at a
   cross, a course can shear, which is the slop the build is limited by.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .calibration_model import (
    BRICK_MM,
    MODEL_IO,
    STUD_MM,
    UNVERIFIED_COLOURS,
    colour_class,
    COLOUR_CLASS,
    exposed_faces,
    load_bricks,
)

INVENTORY_CSV = Path(__file__).resolve().parent.parent / "data" / "box_10698_inventory.csv"

#: The photographed build (2026-09-20) measured 19 % of 3x4 windows ambiguous,
#: worst repeated five times, and calibrated to within 0.04 % of the checkerboard.
#: That is the level known to be good enough; nothing looser has been tested.
MAX_AMBIGUOUS = 0.20
MAX_REPEAT = 5


@dataclass(frozen=True)
class RuleResult:
    rule: str
    passed: bool
    detail: str


def load_inventory(path: Path = INVENTORY_CSV) -> Counter:
    """Pieces per ``(part, colour)``, parts named as in LDraw (``3001.dat``)."""
    counts: Counter = Counter()
    with open(path) as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in rows:
            counts[(f"{row['part_num']}.dat", int(row["color_id"]))] += int(row["quantity"])
    return counts


def _visible(path):
    """Side faces a camera can see. Top faces are studded and carry no features."""
    return exposed_faces(path, sides_only=True)


def _shares_edge(a, b, tol=0.3, min_overlap=1.0) -> bool:
    """Whether two faces meet along a segment -- a seam a detector has to resolve."""
    for i in range(4):
        p, q = a.corners[i], a.corners[(i + 1) % 4]
        d = q - p
        length = np.linalg.norm(d)
        d = d / length
        for j in range(4):
            r, s = b.corners[j], b.corners[(j + 1) % 4]
            off = [np.linalg.norm((x - p) - ((x - p) @ d) * d) for x in (r, s)]
            if max(off) > tol:
                continue
            t = sorted(((r - p) @ d, (s - p) @ d))
            if min(t[1], length) - max(t[0], 0.0) > min_overlap:
                return True
    return False


def adjacent_pairs(path=MODEL_IO):
    """Pairs of visible faces from different bricks that share a seam."""
    faces = _visible(path)
    centres = np.array([f.centre for f in faces])
    pairs = []
    for i in range(len(faces)):
        # Neighbours are within a long brick's half-length; this keeps it O(n) in practice.
        near = np.flatnonzero(np.linalg.norm(centres - centres[i], axis=1) < 60.0)
        for j in near[near > i]:
            if faces[i].brick != faces[j].brick and _shares_edge(faces[i], faces[j]):
                pairs.append((faces[i], faces[j]))
    return faces, pairs


def _wall_grids(faces):
    """Each coplanar wall as a course-by-stud grid of colour groups, -1 where empty."""
    walls = defaultdict(list)
    for f in faces:
        walls[(tuple(np.round(f.normal, 6)), round(float(f.centre @ f.normal), 2))].append(f)
    up = np.array([0.0, 0.0, 1.0])
    grids = []
    for (normal, _), wall in walls.items():
        along = np.cross(up, np.array(normal))
        boxes = [(f.corners @ along, f.corners @ up, colour_class(f.colour)) for f in wall]
        a0 = min(a.min() for a, _, _ in boxes)
        z0 = min(z.min() for _, z, _ in boxes)
        na = int(round((max(a.max() for a, _, _ in boxes) - a0) / STUD_MM))
        nz = int(round((max(z.max() for _, z, _ in boxes) - z0) / BRICK_MM))
        grid = np.full((nz, na), -1, int)
        for i in range(nz):
            for j in range(na):
                pa, pz = a0 + (j + 0.5) * STUD_MM, z0 + (i + 0.5) * BRICK_MM
                for a, z, c in boxes:
                    if a.min() <= pa <= a.max() and z.min() <= pz <= z.max():
                        grid[i, j] = c
                        break
        grids.append(grid)
    return grids


def window_ambiguity(path=MODEL_IO, faces=None):
    """``(placements, ambiguous fraction, worst repeat)`` over 3x4 class windows."""
    faces = _visible(path) if faces is None else faces
    windows: Counter = Counter()
    for grid in _wall_grids(faces):
        for i in range(grid.shape[0] - 2):
            for j in range(grid.shape[1] - 3):
                w = grid[i : i + 3, j : j + 4]
                if (w >= 0).all():
                    windows[w.tobytes()] += 1
    placements = sum(windows.values())
    if not placements:
        return 0, 1.0, 0
    ambiguous = sum(c for c in windows.values() if c > 1)
    return placements, ambiguous / placements, max(windows.values())


def cross_junctions(faces) -> int:
    """Seam points where four different bricks meet on one wall."""
    count = Counter()
    for f in faces:
        plane = (tuple(np.round(f.normal, 6)), round(float(f.centre @ f.normal), 2))
        for c in f.corners:
            count[(plane, tuple(np.round(c, 1)))] += 1
    return sum(1 for n in count.values() if n >= 4)


def check(path=MODEL_IO) -> List[RuleResult]:
    """Run every rule on a model. Order follows the module docstring."""
    bricks = load_bricks(path)
    results = []

    need = Counter((b.part, b.colour) for b in bricks)
    have = load_inventory()
    short = {k: (n, have.get(k, 0)) for k, n in need.items() if n > have.get(k, 0)}
    detail = "all within the box" if not short else "; ".join(
        f"{p[:-4]} colour {c}: need {n}, box has {h}" for (p, c), (n, h) in sorted(short.items())
    )
    results.append(RuleResult("1 one 10698 box", not short,
                              f"{sum(need.values())} pieces; {detail}"))

    base = min(b.centre[2] - b.half[2] for b in bricks)
    raised = [b for b in bricks if b.is_plate and b.centre[2] - b.half[2] > base + 1e-3]
    results.append(RuleResult("2 plates only as a base", not raised,
                              f"{len(raised)} plate(s) above the base"))

    faces, pairs = adjacent_pairs(path)
    off = Counter(f.colour for f in faces if f.colour not in COLOUR_CLASS)
    unverified = sorted({f.colour for f in faces} & set(UNVERIFIED_COLOURS))
    detail = f"{len(faces)} visible faces"
    if off:
        detail += "; unreadable colours on " + ", ".join(f"{n}x{c}" for c, n in off.most_common())
    if unverified:
        detail += f"; predicted but not yet photographed: {unverified}"
    results.append(RuleResult("3 readable palette", not off, detail))

    clash = [(a, b) for a, b in pairs if colour_class(a.colour) == colour_class(b.colour)]
    results.append(RuleResult("4 no same-group joint", not clash,
                              f"{len(clash)} of {len(pairs)} visible joints join one colour group"))

    placements, ambiguous, worst = window_ambiguity(faces=faces)
    ok = placements > 0 and ambiguous <= MAX_AMBIGUOUS and worst <= MAX_REPEAT
    results.append(RuleResult("5 locally distinguishable", ok,
                              f"{placements} windows, {ambiguous:.0%} ambiguous, worst x{worst} "
                              f"(limit {MAX_AMBIGUOUS:.0%}, x{MAX_REPEAT})"))

    crosses = cross_junctions(faces)
    results.append(RuleResult("6 running bond", crosses == 0, f"{crosses} cross junction(s)"))
    return results


def report(path=MODEL_IO) -> str:
    lines = [f"{Path(path).name}"]
    for r in check(path):
        lines.append(f"  [{'pass' if r.passed else 'FAIL'}] {r.rule}: {r.detail}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:] or [MODEL_IO]:
        print(report(arg))
