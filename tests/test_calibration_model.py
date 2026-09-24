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


def test_the_target_is_two_wide_bricks_from_the_box():
    """35 two-wide bricks: only these have a readable face on both sides of a wall."""
    bricks = load_bricks()
    assert len(bricks) == 35
    assert {b.part for b in bricks} == {"3001.dat", "3002.dat", "3003.dat"}


def test_extent_is_a_whole_number_of_studs_and_bricks():
    """8 x 8 studs by 8 bricks: two walls of 64 mm at a right angle, 76.8 mm high.

    The tolerance is a micron rather than exact because Studio rounds the
    rotation matrices it writes into the `.ldr`, which leaves about 40 nm on a
    144 mm span. That is four orders of magnitude below LEGO's own moulding
    tolerance, so it is a property of the file format and not of the target --
    but it is also why nothing downstream should compare model coordinates for
    exact equality.
    """
    extent = extent_mm()
    assert extent == pytest.approx([8 * STUD_MM, 8 * STUD_MM, 8 * BRICK_MM], abs=1e-3)


def test_no_joint_has_the_same_colour_on_both_sides():
    """The one layout property that has to survive whatever bricks were on hand.

    A joint whose two faces share a colour is not a joint in the image: the blobs
    merge, and a merged pair looks like one brick of the wrong size, which puts
    object points on a lie rather than merely losing a feature.

    The target as built uses eight colours rather than the designed five, and 33
    of its bricks are black -- but it still has zero same-colour joints, and that
    is why it calibrates as well as the design did. Black costs *features*: a
    black face against the 0.25 mm chamfer has no visible edge, so none of its
    126 appearances across eight rendered views is detected. It costs accuracy
    only if the estimator is left short of features, and it is not.
    """
    up = np.array([0.0, 0.0, 1.0])
    walls = defaultdict(list)
    # Keyed by depth as well as direction. On a box the only faces sharing a
    # direction are one wall's; on any other shape -- a corner's inner face and
    # the far end of the other arm -- faces 48 mm apart in depth would otherwise
    # be compared as if they lay side by side.
    for face in exposed_faces(sides_only=True):
        depth = round(float(face.centre @ face.normal), 2)
        walls[(tuple(np.round(face.normal, 6)), depth)].append(face)

    shared = 0
    for (normal, _), wall in walls.items():
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


def test_exposed_faces_are_the_surface_a_camera_can_see():
    """Faces on the target's surface, and none buried between bricks.

    The ray is cast from every corner of a face, not just its centre. On the
    hollow tower this target replaced, casting from the centre alone let two
    cavity faces through, reported as part of the opposite wall 128 mm from its
    plane -- and nothing downstream noticed, because correspondence absorbed them
    as outliers. A corner has no cavity, but the same test separates a wall's
    inner face (open, and visible) from the faces bricks press against each other.
    """
    faces = exposed_faces()
    sides = [f for f in faces if f.is_side]
    assert len(faces) == 94
    assert Counter(f.is_side for f in faces) == {True: 86, False: 8}
    # Outer and inner face of each wall, and each wall's far end.
    assert sorted(Counter(tuple(np.round(f.normal, 6)) for f in sides).values()) == [20, 20, 23, 23]
    depths = np.array([face.centre @ face.normal for face in sides])
    assert np.abs(depths[:, None] - [0.0, 16.0, 64.0]).min(axis=1).max() < 1e-3


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
    assert len(points) == 198
    assert len(junctions) == 68
    assert all(p.is_tee for p in junctions), "a running bond admits no other kind"


def test_a_local_colour_window_is_distinguishable():
    """Few 3-course by 4-stud windows of colour groups repeat anywhere on the target.

    Correspondence does not look windows up; it fits a homography per wall and
    grows it on its own inliers, so the wall needs to be locally distinguishable
    on average, not everywhere. The tower it replaced ran at 19 % ambiguous, worst
    five times, and calibrated correctly; the corner is at 5 %, worst twice. If
    this climbs, the homography starts seeding on the wrong brick, and the failure
    is not subtle: on a periodic layout a wrong anchor reprojects under a pixel
    while posing 147 degrees from the truth.
    """
    from fishsense_wuwnet.target_rules import MAX_AMBIGUOUS, MAX_REPEAT, window_ambiguity

    placements, ambiguous, worst = window_ambiguity()
    assert placements >= 60
    assert ambiguous <= MAX_AMBIGUOUS
    assert worst <= MAX_REPEAT


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
