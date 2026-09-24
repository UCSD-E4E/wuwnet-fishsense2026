"""Generate a corner (L) target that passes `target_rules`, from one LEGO Classic 10698.

Why a corner rather than a box: a camera sees at most two walls of a box at once,
so two walls at a right angle give every view the same information -- two
non-coplanar planes -- at half the bricks. And half the bricks is what the box
allows: after the palette rule, one 10698 holds 54 two-wide bricks (172 stud
lengths) in readable colours, which is not enough for four walls of useful size.

Only two-wide bricks are used, so both faces of a wall are real brick faces in a
readable colour. One-wide bricks would need building in back-to-back pairs, and
on a corner (unlike a box) the back of the pair is on show from the concave side.

The layout is searched, not constructed: bricks are laid course by course in
running bond with the corner interlocked, coloured under the joint and inventory
constraints, and each candidate is scored by `target_rules`; the best is kept.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .calibration_model import COLOUR_CLASS, UNVERIFIED_COLOURS, colour_class
from .target_rules import check, load_inventory

#: Two-wide bricks by length in studs.
TWO_WIDE = {2: "3003.dat", 3: "3002.dat", 4: "3001.dat", 6: "2456.dat", 8: "3007.dat"}

Placement = Tuple[str, int, str, float, int, int]  # part, colour, arm, start, length, course


def _partition(span: int, forbidden: set, avail: Counter, rng, first_min: int = 2) -> Optional[List[int]]:
    """Split `span` studs into brick lengths whose internal seams avoid `forbidden`.

    Lengths are drawn against `avail`, the bricks of each length still in stock:
    the readable palette holds only four 3-stud bricks, and drawing lengths freely
    exhausts them within a course.
    """
    for _ in range(200):
        out, pos, used = [], 0, Counter()
        while pos < span:
            options = [s for s in avail if avail[s] > used[s] and pos + s <= span
                       and (pos + s == span or pos + s not in forbidden)]
            if not out:
                options = [s for s in options if s >= first_min]
            if not options:
                break
            weights = np.array([avail[s] - used[s] for s in options], float)
            s = int(options[rng.choice(len(options), p=weights / weights.sum())])
            out.append(s)
            used[s] += 1
            pos += s
        if pos == span:
            return out
    return None


def design_corner(arm: int = 10, courses: int = 7, seed: int = 0, tries: int = 400):
    """A corner target: two walls of `arm` studs, `courses` high. Best of `tries` layouts."""
    rng = np.random.default_rng(seed)
    inventory = load_inventory()
    stock0 = {k: n for k, n in inventory.items() if k[1] in COLOUR_CLASS and k[0] in TWO_WIDE.values()}

    best, best_score = None, None
    for _ in range(tries):
        stock = Counter(stock0)
        placements: List[Placement] = []
        seams = {"x": set(), "y": set()}  # seam positions on each outer face, previous course
        ok = True
        prev_course: List[Tuple[str, float, float, int]] = []  # (arm, a0, a1, class)
        for k in range(courses):
            # A dead end in one course is retried from the same stock before the
            # whole layout is abandoned; most failures are local.
            for _attempt in range(60):
                trial = _lay_course(k, arm, seams, prev_course, Counter(stock), rng)
                if trial is not None:
                    break
            if trial is None:
                ok = False
                break
            course_placements, coloured, cur_seams, stock = trial
            placements.extend(course_placements)
            prev_course, seams = coloured, cur_seams
        if not ok:
            continue
        score = _score(placements, arm)
        if best_score is None or score < best_score:
            best, best_score = placements, score
            if score[0] == 0 and score[1] == 0.0:
                break
    return best


def _lay_course(k, arm, seams, prev_course, stock, rng):
    """One course: lengths in running bond, then colours. None at a dead end."""
    # The arm that owns the corner alternates, which is what ties the walls together.
    owner = "x" if k % 2 == 0 else "y"
    course = []
    cur_seams = {"x": set(), "y": set()}
    for arm_name in ("x", "y"):
        start = 0 if arm_name == owner else 2
        span = arm - start
        # On this arm's outer face, the other arm's corner brick shows its end
        # over [0, 2] on the courses it owns: a seam at 2 there.
        forbidden = {p - start for p in seams[arm_name] if p > start}
        avail = Counter()
        for s, part in TWO_WIDE.items():
            avail[s] = sum(m for (p, _), m in stock.items() if p == part)
            avail[s] -= sum(1 for c in course if c[4] == s)
        avail = Counter({s: m for s, m in avail.items() if m > 0})
        lengths = _partition(span, forbidden, avail, rng, first_min=3 if start == 0 else 2)
        if lengths is None:
            return None
        pos = start
        for n in lengths:
            course.append((arm_name, pos, pos + n, -1, n))
            pos += n
            if pos < arm:
                cur_seams[arm_name].add(pos)
        if start == 2:
            cur_seams[arm_name].add(2)
    # Colour each brick: a group unlike its neighbours along the course, the
    # bricks it touches in the course below, and the corner brick.
    coloured, placements = [], []
    for arm_name, a0, a1, _, n in course:
        banned = set()
        for other_arm, b0, b1, cls in prev_course:
            if other_arm == arm_name and b0 < a1 and a0 < b1:
                banned.add(cls)
            if a0 == 0 and b0 == 0 and other_arm != arm_name:
                banned.add(cls)  # corner bricks stack over one another
            if a0 == 2 and b0 == 0 and other_arm != arm_name:
                banned.add(cls)  # sits beside the corner brick below's end face
        for other_arm, b0, b1, cls in coloured:
            if other_arm == arm_name and (b1 == a0 or a1 == b0):
                banned.add(cls)
            if {a0, b0} == {0, 2} and other_arm != arm_name:
                banned.add(cls)  # beside the corner brick's end face, same course
        part = TWO_WIDE[n]
        choices = [(p, c) for (p, c), m in stock.items() if p == part and m > 0
                   and colour_class(c) not in banned]
        if not choices:
            return None
        # A colour that has been photographed beats one that is only predicted to
        # read: medium azure is predicted at saturation 127 against a floor of 125,
        # and the first design drew it three times purely by chance.
        measured = [(p, c) for p, c in choices if c not in UNVERIFIED_COLOURS]
        choices = measured or choices
        # Prefer the group with most stock left, so no colour runs out early.
        left = Counter()
        for (p, c), m in stock.items():
            left[colour_class(c)] += m
        weights = np.array([left[colour_class(c)] for _, c in choices], float)
        p, c = choices[int(rng.choice(len(choices), p=weights / weights.sum()))]
        stock[(p, c)] -= 1
        coloured.append((arm_name, a0, a1, colour_class(c)))
        placements.append((p, c, arm_name, a0, n, k))
    return placements, coloured, cur_seams, stock


def _score(placements, arm):
    """Failed rules first, then window ambiguity -- smaller is better."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".ldr", delete=False) as f:
        f.write(to_ldr(placements))
        path = f.name
    results = check(path)
    Path(path).unlink()
    failed = sum(not r.passed for r in results)
    detail = next(r.detail for r in results if r.rule.startswith("5"))
    ambiguous = float(detail.split(",")[1].strip().split("%")[0]) / 100
    return failed, ambiguous


def to_ldr(placements, name: str = "corner_target") -> str:
    """LDraw text. LDraw is y-down with 20 LDU to the stud and 24 to a brick."""
    lines = [f"0 FILE {name}", f"0 {name}", f"0 Name: {name}", "0 Author: target_design.py",
             f"0 NumOfBricks {len(placements)}"]
    for part, colour, arm, a0, n, k in placements:
        top = -(k + 1) * 24
        centre = (a0 + n / 2.0) * 20
        if arm == "x":  # long axis along model x, occupying y in [0, 2] studs
            x, z, rot = centre, 20, "1 0 0 0 1 0 0 0 1"
        else:  # long axis along model y, occupying x in [0, 2] studs
            x, z, rot = 20, centre, "0 0 -1 0 1 0 1 0 0"
        lines.append(f"1 {colour} {x:.1f} {top} {z:.1f} {rot} {part}")
    return "\n".join(lines) + "\n"


def compare(built, design) -> dict:
    """Differences between a model as built (a Studio export) and the design it follows.

    Studio places a model wherever it was assembled and at any heading, so the two
    are aligned first: each quarter turn about the vertical is tried, with the
    lowest corner of the bricks moved to the origin, and the turn that lines up
    most bricks is kept. Bricks are then keyed by part, position and the axis
    their long side runs along, which is invariant to the 180-degree flips Studio
    makes freely on a symmetric brick. What is left over is reported.
    """
    from .calibration_model import brick_corners, load_bricks

    def keyed(bricks, turns):
        c, s = np.round(np.cos(turns * np.pi / 2)), np.round(np.sin(turns * np.pi / 2))
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        low = np.min([brick_corners(b) @ rot.T for b in bricks], axis=(0, 1))
        out = {}
        for b in bricks:
            centre = rot @ b.centre - low
            axis = tuple(np.abs(np.round(rot @ b.rotation[:, 0], 3)))
            out[(b.part, tuple(np.round(centre, 1)), axis)] = b.colour
        return out

    want = keyed(load_bricks(design), 0)
    built_bricks = load_bricks(built)
    got = max((keyed(built_bricks, k) for k in range(4)), key=lambda g: len(set(g) & set(want)))
    shared = set(got) & set(want)
    return {
        "matched": len(shared),
        "recoloured": sorted((k[0], k[1], want[k], got[k]) for k in shared if want[k] != got[k]),
        "missing": sorted((k[0], k[1], want[k]) for k in set(want) - set(got)),
        "extra": sorted((k[0], k[1], got[k]) for k in set(got) - set(want)),
    }
