"""Mockup sheet for a brick target design: rendered views, unfolded walls, parts list.

    uv run python refraction_analysis/make_target_mockup.py [model]

Writes `figures/target-mockup.png` for the target, or `target-mockup-<name>.png`
for any other model given. The views are the same renderer the
detector tests use, so what is drawn is what the pipeline is scored against.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fishsense_wuwnet.calibration_model import MODEL_IO, PARTS, brick_corners, exposed_faces, load_bricks  # noqa: E402
from fishsense_wuwnet.target_render import LDRAW_BGR, look_at, render  # noqa: E402
from fishsense_wuwnet.target_rules import INVENTORY_CSV, check  # noqa: E402

DESIGN = MODEL_IO
PART_NAMES = {p: f"{s[1]}x{s[0]}" for p, s in PARTS.items()}


def _colour_names():
    names = {}
    with open(INVENTORY_CSV) as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            names[int(row["color_id"])] = row["colour_name"]
    return names


def _rgb(colour):
    b, g, r = LDRAW_BGR[colour]
    return (r / 255, g / 255, b / 255)


def _crop(image, pad=40):
    mask = np.abs(image.astype(int) - image[0, 0].astype(int)).sum(-1) > 30
    ys, xs = np.nonzero(mask)
    return image[max(ys.min() - pad, 0) : ys.max() + pad, max(xs.min() - pad, 0) : xs.max() + pad]


def _outer_wall(faces, normal):
    wall = [f for f in faces if np.allclose(f.normal, normal)]
    depth = min(f.centre @ normal for f in wall)
    return [f for f in wall if abs(f.centre @ normal - depth) < 1e-3]  # the outer plane only


def _top_course(wall, right, names):
    """A wall's name by what is on it -- its top course, read left to right."""
    top = max(f.corners[:, 2].max() for f in wall)
    row = sorted((f for f in wall if abs(f.corners[:, 2].max() - top) < 1e-3),
                 key=lambda f: float(f.centre @ right))
    return ", ".join(names.get(f.colour, str(f.colour)).lower() for f in row)


def _elevation(ax, faces, bricks, normal, right, title):
    """One outer wall, flat, as seen from outside: brick faces in true proportion.

    Walls are named by their top course, never "left" and "right": handedness
    depends on where the builder stands, and a build guide labelled by it has
    already had its walls swapped once.
    """
    wall = _outer_wall(faces, normal)
    for f in wall:
        pts = np.stack([f.corners @ right, f.corners[:, 2]], axis=1)
        ax.add_patch(Polygon(pts, closed=True, facecolor=_rgb(f.colour), edgecolor="#1b1b1b", lw=1.2))
        # The end of a brick belonging to the other wall shows at the corner; label
        # it as such, or a 2x4 seen end-on reads as a 2-stud brick.
        end_on = abs(float(bricks[f.brick].rotation[:, 0] @ normal)) > 0.5
        width = np.ptp(pts[:, 0]) / 8.0
        label = "end" if end_on else f"{width:.0f}"
        text = "#1b1b1b" if sum(_rgb(f.colour)) > 1.6 else "white"
        ax.text(*pts.mean(0), label, ha="center", va="center", fontsize=8, color=text)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def main(design: Path = DESIGN):
    faces = exposed_faces(design, sides_only=True)
    bricks = load_bricks(design)
    corners = np.vstack([brick_corners(b) for b in bricks])
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    aim = (lo + hi) / 2.0
    width, height = 1600, 1200
    K = np.array([[3600.0, 0, width / 2], [0, 3600.0, height / 2], [0, 0, 1]])

    fig = plt.figure(figsize=(13, 8.6))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], width_ratios=[1, 1, 1, 1.05])
    eyes = {"at the corner": (-420, -420, 250), "facing wall A": (-560, -140, 200), "from above": (-300, -300, 520)}
    for i, (name, eye) in enumerate(eyes.items()):
        R, t = look_at(np.array(eye) + aim * [1, 1, 0], aim)
        view = render(K, R, t, size=(width, height), path=design, background=(245, 245, 245),
                      noise=0.0, seed=None)
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(cv2.cvtColor(_crop(view.image), cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    names = _colour_names()
    walls = [(np.array([-1.0, 0, 0]), np.array([0, -1.0, 0]), "corner at its right-hand end"),
             (np.array([0, -1.0, 0]), np.array([1.0, 0, 0]), "corner at its left-hand end")]
    for slot, (letter, (normal, right, where)) in zip([grid[1, 0:2], grid[1, 2]], zip("AB", walls)):
        top = _top_course(_outer_wall(faces, normal), right, names)
        _elevation(fig.add_subplot(slot), faces, bricks, normal, right,
                   f"Wall {letter}, seen from outside, {where}\ntop course: {top}")
    fig.text(0.02, 0.02, "Numbers are brick length in studs; 'end' is the end of the other "
             "wall's corner brick.", fontsize=8.5)

    parts = Counter((PART_NAMES[b.part], names.get(b.colour, str(b.colour))) for b in bricks)
    ax = fig.add_subplot(grid[:, 3])
    ax.axis("off")
    rows = [[f"{n}", p, c] for (p, c), n in sorted(parts.items(), key=lambda kv: (kv[0][1], kv[0][0]))]
    table = ax.table(cellText=rows, colLabels=["qty", "brick", "colour"], loc="upper center",
                     colWidths=[0.14, 0.2, 0.62])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.25)
    results = check(design)
    lines = [f"{'pass' if r.passed else 'FAIL'}  {r.rule}" for r in results]
    size = hi - lo
    ax.text(0.0, 0.02,
            f"{len(bricks)} bricks, all from one LEGO Classic 10698\n"
            f"{size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm\n\n" + "\n".join(lines),
            transform=ax.transAxes, fontsize=8.5, va="bottom", family="monospace")
    fig.suptitle("Corner calibration target", fontsize=12)
    fig.tight_layout()
    name = "target-mockup" if Path(design).resolve() == DESIGN else f"target-mockup-{Path(design).stem}"
    out = REPO / "figures" / f"{name}.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DESIGN)
