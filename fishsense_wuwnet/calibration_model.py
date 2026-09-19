"""The LEGO calibration target: its exact geometry, and fitting a reconstruction to it.

The target is a hollow rectangular tower of 2x4 bricks, designed in BrickLink
Studio and stored as `calibration_model/calibration_model.io`. Studio's `.io` is
a zip carrying an LDraw `model.ldr`, which gives every brick's position and
orientation exactly -- so the target's geometry is *known by construction* rather
than measured, which is the whole reason for using it.

Why this rather than a printed board. A printed checkerboard's pitch has to be
measured before it can be trusted, and ours could not be pinned down well enough
to settle its own anisotropy (see the paper's limitations). LEGO is injection
moulded to a far tighter tolerance than any print, its geometry is published, and
every copy is identical. It is also genuinely three-dimensional, which a plane is
not: a planar target carries degeneracies that a tower does not, and it is those
degeneracies that make a single-orientation dataset unable to separate a target's
anisotropy from a sensor's.

Units. LDraw measures in LDU, 0.4 mm each: a stud pitch is 20 LDU, a plate 8, a
brick 24. LDraw's y axis points **down**, which this module flips on the way out
so that model coordinates are right-handed with z up, in millimetres.
"""

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

LDU_MM = 0.4
STUD_MM = 8.0
BRICK_MM = 9.6

#: The design, as committed. Studio writes several `.ldr` variants into the zip;
#: `model.ldr` is the one it treats as current.
MODEL_IO = Path(__file__).resolve().parent.parent / "calibration_model" / "calibration_model.io"
MODEL_ENTRY = "model.ldr"

#: LDraw colour codes actually used by the target. The four saturated colours are
#: face markers -- one hue per wall -- so a detected marker identifies which face
#: is being seen, which is what makes correspondence tractable when only two or
#: three walls are visible at once.
COLOURS = {0: "black", 15: "white", 1: "blue", 4: "red", 2: "green", 14: "yellow"}
MARKER_COLOURS = (1, 4, 2, 14)

#: Footprint of part 3001 (the 2x4 brick) in studs, and its body height.
BRICK_3001_STUDS = (4, 2)


@dataclass(frozen=True)
class Brick:
    """One placed brick, in millimetres, z up."""

    colour: int
    centre: np.ndarray  # (3,) centre of the brick body
    rotation: np.ndarray  # (3, 3) model <- brick
    part: str

    @property
    def is_marker(self) -> bool:
        return self.colour in MARKER_COLOURS


def _ldraw_lines(path: Path = MODEL_IO, entry: str = MODEL_ENTRY):
    with zipfile.ZipFile(path) as zf:
        return zf.read(entry).decode("utf8", "replace").splitlines()


def load_bricks(path: Path = MODEL_IO) -> list:
    """Parse the Studio file into placed bricks, in millimetres with z up.

    LDraw type-1 lines are ``1 <colour> x y z a b c d e f g h i <part>``, where
    the nine numbers are the rotation in row-major order. Only type-1 lines
    matter here; the target is built from one part, so no sub-file recursion is
    needed and none is attempted.
    """
    flip = np.diag([1.0, -1.0, 1.0])  # LDraw y is down; make z up, right-handed
    swap = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])  # y-up -> z-up
    to_model = swap @ flip
    bricks = []
    for line in _ldraw_lines(path):
        fields = line.split()
        if len(fields) < 15 or fields[0] != "1":
            continue
        colour = int(fields[1])
        pos = np.array(list(map(float, fields[2:5]))) * LDU_MM
        rot = np.array(list(map(float, fields[5:14]))).reshape(3, 3)
        bricks.append(
            Brick(colour, to_model @ pos, to_model @ rot @ to_model.T, fields[14])
        )
    if not bricks:
        raise ValueError(f"no LDraw part lines found in {path}:{MODEL_ENTRY}")
    return bricks


def brick_corners(brick: Brick, studs: Tuple[int, int] = BRICK_3001_STUDS) -> np.ndarray:
    """The eight corners of one brick's body, in model millimetres.

    Corners rather than studs because they are what a reconstruction actually
    recovers: a stud is a smooth cylinder and localises poorly, while the black
    and white brick edges are high-contrast and meet at a point.
    """
    half = np.array([studs[0] * STUD_MM, studs[1] * STUD_MM, BRICK_MM]) / 2.0
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return brick.centre + (signs * half) @ brick.rotation.T


def model_points(path: Path = MODEL_IO, markers_only: bool = False) -> np.ndarray:
    """Every brick corner in the target, as an ``(N, 3)`` cloud in millimetres."""
    bricks = [b for b in load_bricks(path) if not markers_only or b.is_marker]
    return np.vstack([brick_corners(b) for b in bricks])


def extent_mm(path: Path = MODEL_IO) -> np.ndarray:
    pts = model_points(path)
    return pts.max(axis=0) - pts.min(axis=0)


def fit_similarity(source, target) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Umeyama similarity fit of `source` onto `target`: scale, rotation, translation, rms.

    This is the measurement the target exists to support. A reconstruction is
    recovered only up to a similarity, so fitting one out is not a concession --
    what is left afterwards is *shape* error, independent of any scale the
    pipeline supplied. The fitted `scale` is separately meaningful: compare it
    against the scale the laser gives and the two together test the metric chain
    end to end.

    Both arrays are ``(N, 3)`` and must correspond row for row.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must be matching (N, 3) arrays")

    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    a, b = source - mu_s, target - mu_t
    u, s, vt = np.linalg.svd(b.T @ a / len(a))
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    variance = (a ** 2).sum() / len(a)
    scale = float((s * [1.0, 1.0, d]).sum() / variance) if variance > 0 else 1.0
    translation = mu_t - scale * rotation @ mu_s
    residual = target - (scale * source @ rotation.T + translation)
    return scale, rotation, translation, float(np.sqrt((residual ** 2).sum(axis=1).mean()))
