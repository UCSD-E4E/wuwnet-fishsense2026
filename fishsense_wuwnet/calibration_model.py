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

#: A brick's moulded body is narrower than the cell it occupies: LEGO holds an
#: 8.0 mm stud *pitch* but moulds an n-stud body to 8n - 0.2 mm, so two bricks
#: side by side leave a 0.2 mm gap. Vertically there is no such clearance --
#: bricks stack to exactly 9.6 mm -- but the moulded edge is chamfered, so a
#: horizontal joint also shows as a fine dark line.
#:
#: This is worth getting right rather than absorbing into a fudge. 0.2 mm on a
#: 32 mm brick is 0.6 %, the same order as the printed-board tolerance this
#: target exists to avoid, so a detector that localised *brick edges* and called
#: them cell boundaries would reintroduce exactly the error it was meant to
#: remove. The nominal lattice used throughout this module is the **centre** of
#: each joint, which is unbiased for both axes: horizontally it is the middle of
#: the 0.2 mm gap, vertically the middle of the chamfer line.
#:
#: The inset from the nominal cell to the *visible* face is therefore not the
#: same on the two axes -- clearance plus chamfer across, chamfer alone up --
#: and a detector working from visible edges has to apply both. It is a real
#: asymmetry, not a rendering detail, and it is why `visible_inset_mm` is
#: exported rather than left inside the renderer.
BRICK_CLEARANCE_MM = 0.2
CHAMFER_MM = 0.25

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


# --------------------------------------------------------------------------
# The target's exterior surface, and the feature lattice it carries.
#
# A checkerboard detector localises saddle points -- places where four cells
# meet. A running-bond wall has none: every vertical joint is spanned by the
# course above, so the joints form T-junctions instead. Running bond is not
# negotiable, because LEGO gets its strength from exactly that overlap; a
# stack-bond wall is a row of independent columns and falls apart in the hand.
#
# So the features here are seam intersections rather than saddle points. They
# are just as good: `calibrateCamera` and `solvePnP` want correspondences, and
# the pitch rides in the object points instead of being implied by the pattern.
# What they are not is self-identifying, which is what the marker colours are
# for -- see `LatticePoint.signature`.
# --------------------------------------------------------------------------

_EPS_MM = 1e-3


@dataclass(frozen=True)
class Face:
    """One exposed rectangular brick face, in model millimetres."""

    brick: int
    colour: int
    corners: np.ndarray  # (4, 3), ordered around the face
    normal: np.ndarray  # (3,) unit, pointing out of the target

    @property
    def centre(self) -> np.ndarray:
        return self.corners.mean(axis=0)

    @property
    def is_side(self) -> bool:
        """True for the four walls, false for the top and bottom courses."""
        return abs(float(self.normal[2])) < 0.5


def _brick_frames(bricks, studs: Tuple[int, int] = BRICK_3001_STUDS):
    half = np.array([studs[0] * STUD_MM, studs[1] * STUD_MM, BRICK_MM]) / 2.0
    centres = np.array([b.centre for b in bricks])
    rotations = np.array([b.rotation for b in bricks])
    return half, centres, rotations


def _occupied(points, half, centres, rotations) -> np.ndarray:
    """Which of `points` lie inside any brick."""
    local = np.einsum("nji,mj->nmi", rotations, np.atleast_2d(points))
    local = local - np.einsum("nji,nj->ni", rotations, centres)[:, None, :]
    return (np.abs(local) <= half + _EPS_MM).all(-1).any(0)


def _escapes(origin, direction, half, centres, rotations) -> bool:
    """Whether a ray leaves the target without entering a brick.

    This is what separates the outer surface from the cavity: the tower is
    hollow, so a brick's inward face has nothing immediately beyond it either,
    and only the ray test tells the two apart. It assumes the target is not
    concave enough to occlude itself, which holds for a tower and is checked by
    the face counts in the tests.
    """
    o = np.einsum("nji,j->ni", rotations, origin) - np.einsum("nji,nj->ni", rotations, centres)
    d = np.einsum("nji,j->ni", rotations, direction)
    with np.errstate(divide="ignore", invalid="ignore"):
        t0, t1 = (-half - o) / d, (half - o) / d
    low, high = np.minimum(t0, t1), np.maximum(t0, t1)
    parallel = np.abs(d) < 1e-12
    low = np.where(parallel, -np.inf, low)
    high = np.where(parallel, np.where(np.abs(o) <= half, np.inf, -np.inf), high)
    enter, leave = low.max(axis=1), high.min(axis=1)
    return not np.any((leave > enter) & (leave > _EPS_MM))


def exposed_faces(path: Path = MODEL_IO, sides_only: bool = False) -> list:
    """Every brick face on the target's outer surface, camera-visible in principle.

    A face survives two tests: nothing occupies the millimetre just outside any
    part of it, and a ray along its normal escapes the model. The first rejects
    buried and partly covered faces, the second rejects the cavity.
    """
    bricks = load_bricks(path)
    half, centres, rotations = _brick_frames(bricks)
    quadrant = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)

    faces = []
    for i, brick in enumerate(bricks):
        for axis in range(3):
            for sign in (-1.0, 1.0):
                normal = rotations[i][:, axis] * sign
                centre = brick.centre + normal * half[axis]
                # In-plane axes chosen so that (u, v, normal) is right-handed,
                # which makes the corner order consistent for a renderer.
                others = [k for k in range(3) if k != axis]
                u, v = rotations[i][:, others[0]], rotations[i][:, others[1]]
                if np.dot(np.cross(u, v), normal) < 0:
                    u, v = v, u
                    others = others[::-1]
                hu, hv = half[others[0]], half[others[1]]
                corners = centre + quadrant[:, 0:1] * u * hu + quadrant[:, 1:2] * v * hv
                samples = np.vstack([centre[None], centre + (corners - centre) * 0.9])
                if _occupied(samples + normal, half, centres, rotations).any():
                    continue
                if not _escapes(centre + normal * _EPS_MM, normal, half, centres, rotations):
                    continue
                face = Face(i, brick.colour, corners, normal)
                if sides_only and not face.is_side:
                    continue
                faces.append(face)
    return faces


@dataclass(frozen=True)
class LatticePoint:
    """A seam intersection on one wall: where brick faces meet at a point."""

    position: np.ndarray  # (3,) model millimetres
    normal: np.ndarray  # (3,) outward normal of the wall it lies on
    signature: Tuple[int, ...]  # brick colour in each quadrant, -1 off the wall
    faces: int  # distinct faces meeting here

    @property
    def is_junction(self) -> bool:
        """True for a point enclosed by brick on all four sides.

        These are the well-conditioned features: two seams cross, so both image
        coordinates are pinned by a brick-against-brick edge. A point on the
        wall's silhouette has an edge against the background on one side, which
        is far more sensitive to exposure and to whatever is behind the target.
        """
        return -1 not in self.signature

    @property
    def is_tee(self) -> bool:
        """True where the two quadrants on one side share a face -- a T-junction.

        In a running bond every interior junction is of this kind, which is
        exactly why a checkerboard detector finds nothing: a T has no saddle.
        """
        a, b, c, d = self.signature
        return self.is_junction and (a == c or b == d)


def lattice_points(path: Path = MODEL_IO, quadrant_mm: float = 3.0) -> list:
    """The detectable feature lattice: seam intersections, grouped by wall.

    A point at the target's vertical edge is shared by two walls and appears
    once for each, which is what a detector sees -- the two walls are imaged as
    separate surfaces with different normals.

    The `signature` is the correspondence key, and it is read the way a detector
    reads it: sample the wall a few millimetres into each of the four quadrants
    around the point, in the wall's own (along, up) frame, and record the brick
    colour found there. Sampling rather than enumerating incident faces matters
    because the brick spanning a T-junction covers two quadrants, and it is the
    repeated colour that identifies the junction as a T.
    """
    faces = [f for f in exposed_faces(path) if f.is_side]
    by_wall = {}
    for face in faces:
        by_wall.setdefault(tuple(np.round(face.normal, 6)), []).append(face)

    up = np.array([0.0, 0.0, 1.0])
    points = []
    for normal_key, wall in sorted(by_wall.items()):
        normal = np.array(normal_key)
        along = np.cross(up, normal)
        # Each wall is planar and axis-aligned, so a face is a rectangle in
        # (along, up) and "which face covers this sample" is an interval test.
        boxes = []
        for face in wall:
            a = face.corners @ along
            z = face.corners @ up
            boxes.append((a.min(), a.max(), z.min(), z.max(), face.colour))
        corners = np.vstack([f.corners for f in wall])
        _, first = np.unique(np.round(corners, 3), axis=0, return_index=True)

        for index in sorted(first):
            position = corners[index]
            pa, pz = float(position @ along), float(position @ up)
            signature = []
            for da in (-quadrant_mm, quadrant_mm):
                for dz in (-quadrant_mm, quadrant_mm):
                    found = -1
                    for a0, a1, z0, z1, colour in boxes:
                        if a0 <= pa + da <= a1 and z0 <= pz + dz <= z1:
                            found = colour
                            break
                    signature.append(found)
            touching = sum(
                np.min(np.abs(f.corners - position).sum(axis=1)) <= _EPS_MM for f in wall
            )
            points.append(LatticePoint(position, normal, tuple(signature), touching))
    return points


def visible_inset_mm() -> Tuple[float, float]:
    """Inset from a nominal cell boundary to the brick's visible edge, ``(across, up)``.

    A detector sees moulded faces, not cells. Across the wall a face stops half
    the moulding clearance short of the cell boundary and is then chamfered;
    up the wall the bricks touch, so only the chamfer applies. Subtracting these
    is what turns a detected brick outline back into the lattice the object
    points are expressed in.
    """
    return BRICK_CLEARANCE_MM / 2.0 + CHAMFER_MM, CHAMFER_MM
