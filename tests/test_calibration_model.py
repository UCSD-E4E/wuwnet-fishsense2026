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


def test_no_joint_has_the_same_colour_on_both_sides():
    """The property the target was recoloured for, and it buys two things.

    A joint whose two faces share a colour is not a joint in the image: the
    blobs merge, and a merged pair looks like one brick of the wrong size. It
    also costs accuracy even when it does not merge, because a joint centre can
    only be read as the midpoint of two observed edges if both edges are there;
    with one side invisible the chamfer becomes a constant you have to know
    rather than one that cancels.

    The as-built target had 92 such joints of 375, sixty of them black-on-black.
    Black is the worst case twice over, since a black brick's own edge against
    the dark seam is not visible either -- so the palette is five colours and
    none of them is black.
    """
    up = np.array([0.0, 0.0, 1.0])
    walls = defaultdict(list)
    for face in exposed_faces(sides_only=True):
        walls[tuple(np.round(face.normal, 6))].append(face)

    assert 0 not in {f.colour for w in walls.values() for f in w}, "black is back"

    shared = 0
    for normal, wall in walls.items():
        along = np.cross(up, np.array(normal))
        boxes = [(f.corners @ along, f.corners @ up, f) for f in wall]
        for index, (a0, z0, first) in enumerate(boxes):
            for a1, z1, second in boxes[index + 1 :]:
                if first.brick == second.brick or first.colour != second.colour:
                    continue
                side = (abs(a0.max() - a1.min()) < 1e-3 or abs(a1.max() - a0.min()) < 1e-3) and (
                    min(z0.max(), z1.max()) - max(z0.min(), z1.min()) > 1e-3
                )
                stacked = (abs(z0.max() - z1.min()) < 1e-3 or abs(z1.max() - z0.min()) < 1e-3) and (
                    min(a0.max(), a1.max()) - max(a0.min(), a1.min()) > 1e-3
                )
                shared += side or stacked
    assert shared == 0


def test_exposed_faces_exclude_the_cavity():
    """The tower is hollow, so the inner wall surface passes a local probe.

    Only the outward ray test rejects it, and getting this wrong would roughly
    double the face count and put object points on surfaces no camera can see.

    The ray has to be cast from every corner of the face and not just its
    centre. Casting from the centre alone let two faces through -- the inside
    of one wall, whose centre ray left through a gap in the wall opposite -- and
    they were reported as part of that opposite wall, 128 mm from its plane.
    Nothing downstream noticed, because correspondence fits a homography per
    wall and simply absorbed them as outliers.
    """
    faces = exposed_faces()
    sides = [f for f in faces if f.is_side]
    assert len(faces) == 201
    assert Counter(f.is_side for f in faces) == {True: 171, False: 30}
    assert sorted(Counter(tuple(f.normal) for f in sides).values()) == [40, 41, 45, 45]
    depths = np.array([face.centre @ face.normal for face in sides])
    assert np.abs(depths[:, None] - [48.0, 64.0, 80.0]).min(axis=1).max() < 1e-3


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
    assert len(points) == 350
    assert len(junctions) == 240
    assert all(p.is_tee for p in junctions), "a running bond admits no other kind"


def test_a_local_colour_window_locates_itself_uniquely_on_the_target():
    """What the recolouring bought: a partial view can say where it is.

    The as-built target was a periodic two-colour bond with markers only at the
    wall ends, so *every* local window matched more than twenty places and
    correspondence had to register a whole wall against a marker column. It also
    meant a wrong anchor reprojected as well as the right one, which is how a
    view could pose 147 degrees from the truth at barely a pixel of residual.

    Now every three-course by four-stud window on every wall is unique, so a
    window identifies its own position outright.
    """
    up = np.array([0.0, 0.0, 1.0])
    walls = defaultdict(list)
    for face in exposed_faces(sides_only=True):
        walls[tuple(np.round(face.normal, 6))].append(face)

    windows = defaultdict(int)
    placements = 0
    for normal, wall in walls.items():
        along = np.cross(up, np.array(normal))
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
    assert max(windows.values()) == 1


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
