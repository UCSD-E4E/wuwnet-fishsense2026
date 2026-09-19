"""Tests for the LEGO calibration target's geometry and its feature lattice.

The target's value is that its geometry is known by construction rather than
measured, so these tests are mostly assertions that the parsed model still
agrees with the design: if a brick moves in Studio and the numbers below stop
holding, the object points every calibration depends on have silently changed.

The one test that is not a regression guard is the last: it records *why*
correspondence has to be established globally rather than from a local patch.
"""

from collections import Counter, defaultdict

import numpy as np
import pytest

from fishsense_wuwnet.calibration_model import (
    BRICK_MM,
    MARKER_COLOURS,
    STUD_MM,
    exposed_faces,
    extent_mm,
    fit_similarity,
    lattice_points,
    load_bricks,
    model_points,
)


def test_the_target_is_one_part_repeated():
    bricks = load_bricks()
    assert len(bricks) == 135
    assert {b.part for b in bricks} == {"3001.dat"}


def test_extent_is_a_whole_number_of_studs_and_bricks():
    """18 x 16 studs by 9 bricks.

    The tolerance is a micron rather than exact because Studio rounds the
    rotation matrices it writes into the `.ldr`, which leaves about 40 nm on a
    144 mm span. That is four orders of magnitude below LEGO's own moulding
    tolerance, so it is a property of the file format and not of the target --
    but it is also why nothing downstream should compare model coordinates for
    exact equality.
    """
    extent = extent_mm()
    assert extent == pytest.approx([18 * STUD_MM, 16 * STUD_MM, 9 * BRICK_MM], abs=1e-3)


def test_every_wall_carries_two_marker_colours_and_the_pair_names_the_wall():
    """This is what lets a detector say which face it is looking at.

    Only two or three walls are visible at once, so correspondence cannot start
    from the whole target. It starts from the wall, and the wall is identified
    by which pair of saturated colours appears on it -- a pair, not a single
    colour, because each marker hue is shared between two adjoining walls.
    """
    walls = defaultdict(set)
    for face in exposed_faces(sides_only=True):
        if face.colour in MARKER_COLOURS:
            walls[tuple(np.round(face.normal, 6))].add(face.colour)

    assert len(walls) == 4
    assert all(len(pair) == 2 for pair in walls.values())
    assert len({frozenset(pair) for pair in walls.values()}) == 4


def test_exposed_faces_exclude_the_cavity():
    """The tower is hollow, so the inner wall surface passes a local probe.

    Only the outward ray test rejects it, and getting this wrong would roughly
    double the face count and put object points on surfaces no camera can see.
    """
    faces = exposed_faces()
    sides = [f for f in faces if f.is_side]
    assert len(faces) == 203
    assert Counter(f.is_side for f in faces) == {True: 173, False: 30}
    assert sorted(Counter(tuple(f.normal) for f in sides).values()) == [41, 42, 45, 45]


def test_every_lattice_point_sits_on_a_brick_corner():
    """The lattice is not a fitted grid; it is the model's own vertices."""
    corners = np.unique(np.round(model_points(), 3), axis=0)
    for point in lattice_points():
        assert np.abs(corners - point.position).sum(axis=1).min() < 1e-3


def test_the_bond_produces_tee_junctions_rather_than_saddle_points():
    """The fact that rules out every checkerboard detector.

    Running bond is not a choice that can be revisited -- a LEGO wall gets its
    strength from the overlap, and a stack-bond wall is a row of independent
    columns -- so the detector has to work with T-junctions.
    """
    points = lattice_points()
    junctions = [p for p in points if p.is_junction]
    assert len(points) == 358
    assert len(junctions) == 248
    tees = sum(p.is_tee for p in junctions)
    assert tees == 240
    assert tees / len(junctions) > 0.95


def test_a_local_colour_window_cannot_locate_itself_on_the_target():
    """Why correspondence is global, and what a re-colouring would buy.

    The interior of every wall is a plain two-colour running bond, so a patch
    of it looks like every other patch: even a window four bricks wide matches
    more than twenty places on the target. A detector therefore cannot identify
    points from their neighbourhood the way a ChArUco board does; it has to
    register a whole wall, anchored on the marker columns at the wall's ends.

    Recorded as a test because it is a property of the *colour layout*, not of
    the bond, and so it is the thing to change if partial views are ever needed.
    """
    walls = defaultdict(list)
    for face in exposed_faces(sides_only=True):
        walls[tuple(np.round(face.normal, 6))].append(face)

    up = np.array([0.0, 0.0, 1.0])
    windows = defaultdict(int)
    placements = 0
    for normal_key, wall in walls.items():
        along = np.cross(up, np.array(normal_key))
        boxes = [(f.corners @ along, f.corners @ up, f.colour) for f in wall]
        a0 = min(a.min() for a, _, _ in boxes)
        z0 = min(z.min() for _, z, _ in boxes)
        na = int(round((max(a.max() for a, _, _ in boxes) - a0) / STUD_MM))
        nz = int(round((max(z.max() for _, z, _ in boxes) - z0) / BRICK_MM))
        grid = np.full((nz, na), -1, int)
        for i in range(nz):
            for j in range(na):
                pa, pz = a0 + (j + 0.5) * STUD_MM, z0 + (i + 0.5) * BRICK_MM
                for a, z, colour in boxes:
                    if a.min() <= pa <= a.max() and z.min() <= pz <= z.max():
                        grid[i, j] = colour
                        break
        for i in range(nz - 2):
            for j in range(na - 3):
                window = grid[i : i + 3, j : j + 4]
                if (window < 0).any():
                    continue
                placements += 1
                windows[window.tobytes()] += 1

    assert placements > 300
    assert max(windows.values()) > 20, "if this drops to 1 the layout became unique"


def test_fitting_the_model_to_a_transformed_copy_recovers_the_transform():
    """The end of the pipeline: shape error and scale, separated.

    A reconstruction is recovered only up to a similarity, so the fit is not a
    concession -- the residual afterwards is shape error, and the fitted scale
    is the quantity the laser independently supplies.
    """
    points = model_points()
    angle = np.radians(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = 0.813 * points @ rotation.T + np.array([120.0, -35.0, 9.0])

    scale, fitted, translation, rms = fit_similarity(points, moved)
    assert scale == pytest.approx(0.813, rel=1e-9)
    assert fitted == pytest.approx(rotation, abs=1e-9)
    assert translation == pytest.approx([120.0, -35.0, 9.0], abs=1e-6)
    assert rms == pytest.approx(0.0, abs=1e-9)
