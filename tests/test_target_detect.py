"""Tests for finding the LEGO target in an image and saying which brick is which.

These run against rendered views rather than photographs, so what they pin down
is the *geometry and correspondence* half of the detector -- which is the half
that is finished. The colour front end is tuned on renders and will need
retuning on real frames; `test_colour_front_end_is_the_part_that_is_not_settled`
says so in the one place a reader will look.

Images are rendered small to keep the suite quick. The detector is scale-free
in its thresholds except for `min_area_px`, which is passed down accordingly.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import fishsense_wuwnet.target_detect as td_module
from fishsense_wuwnet.target_detect import detect_quads, match
from fishsense_wuwnet.target_render import look_at, render

SIZE = (1400, 1050)
CAMERA = np.array([[1000.0, 0.0, 700.0], [0.0, 1000.0, 525.0], [0.0, 0.0, 1.0]])
VIEWS = [(260.0, -330.0, 190.0), (0.0, -520.0, 60.0), (-300.0, -400.0, 150.0)]


def _detect(eye, camera=CAMERA):
    rotation, translation = look_at(eye)
    view = render(camera, rotation, translation, size=SIZE)
    quads = detect_quads(view.image, min_area_px=40.0)
    return view, quads, match(quads, camera, min_faces=4)


def _pose_error(correspondence, camera, rotation, translation):
    ok, rvec, tvec = cv2.solvePnP(
        correspondence.object_points, correspondence.image_points, camera, None
    )
    assert ok
    recovered, _ = cv2.Rodrigues(rvec)
    angle = np.degrees(
        np.arccos(np.clip((np.trace(recovered @ rotation.T) - 1.0) / 2.0, -1.0, 1.0))
    )
    return float(angle), float(np.linalg.norm(tvec.ravel() - translation))


@pytest.mark.parametrize("eye", VIEWS)
def test_a_view_is_detected_matched_and_posed(eye):
    rotation, translation = look_at(eye)
    _, quads, correspondence = _detect(eye)
    assert len(quads) >= 8
    assert len(correspondence) >= 24  # six faces
    angle, offset = _pose_error(correspondence, CAMERA, rotation, translation)
    assert angle < 1.0
    assert offset < 5.0


@pytest.mark.parametrize("eye", VIEWS)
def test_the_camera_is_not_placed_behind_the_wall_it_is_looking_at(eye):
    """The mirror solution, which scored *better* than the truth before it was ruled out.

    The detector orders corners counter-clockwise in image axes, where y points
    down; the model orders them counter-clockwise in the wall's own frame. The
    two differ by a reflection, and the reflected match reprojects under a pixel
    while placing the camera at its mirror image through the wall plane. Nothing
    in the residual reveals it -- only the outward normal does -- so it is
    asserted separately from the pose test above, which it would otherwise pass
    at 180 degrees out.

    The tolerance is loose on purpose: this test is not about accuracy, it is
    about which side of the wall the camera ended up on. A mirrored solution
    puts it roughly twice the standoff away, so a few percent of the range
    separates the two cases by an enormous margin.
    """
    rotation, translation = look_at(eye)
    _, _, correspondence = _detect(eye)
    ok, rvec, tvec = cv2.solvePnP(
        correspondence.object_points, correspondence.image_points, CAMERA, None
    )
    recovered, _ = cv2.Rodrigues(rvec)
    centre = -recovered.T @ tvec.reshape(3)
    standoff = float(np.linalg.norm(eye))
    assert np.linalg.norm(centre - np.asarray(eye)) < 0.05 * standoff


def test_a_staggered_column_of_one_colour_is_not_one_quad():
    """Same-colour bricks in adjacent courses overlap by half a brick.

    They are separated only by the chamfer, so any smoothing before
    classification merges a whole column into a zigzag blob -- which would then
    be fitted as a single enormous brick and put object points on a lie. The
    guard is that no detected quad is much larger than a brick should be.
    """
    view, quads, _ = _detect((260.0, -330.0, 190.0))
    areas = np.array([abs(cv2.contourArea(q.corners.astype(np.float32))) for q in quads])
    assert len(areas) > 0
    assert areas.max() < 6.0 * np.median(areas)


def test_bricks_on_the_targets_vertical_edge_are_recovered_as_two_faces():
    """A corner brick shows two faces and projects to a chevron, not a quad.

    Worth keeping even though the target is no longer black and white: a chevron
    that is not split is silently dropped, and the target's vertical edges are
    where two walls are seen at once, which is what ties their poses together.
    The check is that faces are recovered on more than one wall at a time.
    """
    _, quads, correspondence = _detect((260.0, -330.0, 190.0))
    walls = {tuple(np.round(f.normal, 6)) for f in correspondence.faces}
    assert len(walls) >= 2, "only one wall matched; chevrons are being dropped"
    assert len(quads) >= len(correspondence) // 4


def test_one_seed_colour_suffices_because_local_windows_are_unique():
    """The target's uniqueness is what lets the hypothesis search stay cheap.

    Every three-course by four-stud window on this target is unique, so a wrong
    anchor cannot match much and the first correct one wins. Widening the seed
    search then buys nothing and costs several times the runtime. On the earlier
    periodic target the opposite held, and one colour produced a confident pose
    147 degrees from the truth -- so this is a property of *this* target, and it
    is asserted rather than assumed.
    """
    import functools

    from fishsense_wuwnet.target_detect import _seed_faces

    rotation, translation = look_at((-300.0, -400.0, 150.0))
    view = render(CAMERA, rotation, translation, size=SIZE)
    quads = detect_quads(view.image, min_area_px=40.0)

    narrow = match(quads, CAMERA, min_faces=4)
    original = td_module._seed_faces
    td_module._seed_faces = functools.partial(_seed_faces, colours=4)
    try:
        wide = match(quads, CAMERA, min_faces=4)
    finally:
        td_module._seed_faces = original

    assert len(narrow) >= 0.95 * len(wide)
    angle, _ = _pose_error(narrow, CAMERA, rotation, translation)
    assert angle < 1.0


def test_the_three_dimensional_target_settles_the_focal_length_ratio():
    """The reason for a tower rather than a board.

    A planar target leaves `fx/fy` poorly constrained unless the views are
    rolled, which is exactly why our own pool dataset cannot settle a ~1 %
    anisotropy. Here the truth is anisotropic by 1.09 %, the fit is seeded
    isotropic and 11 % long, and the ratio comes back to a small fraction of a
    percent -- from views that are not rolled at all.
    """
    truth = np.array([[1000.0, 0.0, 712.0], [0.0, 989.0, 520.0], [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(3)
    object_points, image_points = [], []
    for azimuth in np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False):
        radius = rng.uniform(330.0, 520.0)
        elevation = rng.uniform(-0.2, 0.5)
        eye = (
            radius * np.cos(azimuth) * np.cos(elevation),
            radius * np.sin(azimuth) * np.cos(elevation),
            43.2 + radius * np.sin(elevation),
        )
        _, _, correspondence = _detect(eye, camera=truth)
        if len(correspondence) >= 24:
            object_points.append(correspondence.object_points.astype(np.float32))
            image_points.append(correspondence.image_points.astype(np.float32))

    assert len(object_points) >= 8
    guess = np.array([[1150.0, 0.0, 700.0], [0.0, 1150.0, 525.0], [0.0, 0.0, 1.0]])
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )
    rms, fitted, _, _, _ = cv2.calibrateCamera(
        object_points, image_points, SIZE, guess, None, flags=flags
    )
    assert rms < 2.0
    # Tolerances are set by the size these tests render at, not by the method.
    # At 1400x1050 a brick is about 80 by 24 px and the fit lands near -0.8 % on
    # fx; at a full 4000x3000 sensor the same sweep gives +0.033 % on fx,
    # +0.017 % on fy and +0.016 % on the ratio. The ratio is the claim that
    # matters, so it is held an order tighter than the absolute scale.
    ratio = fitted[0, 0] / fitted[1, 1]
    assert ratio == pytest.approx(truth[0, 0] / truth[1, 1], rel=4e-3)
    assert fitted[0, 0] == pytest.approx(truth[0, 0], rel=1.5e-2)


def test_colour_front_end_is_the_part_that_is_not_settled():
    """A renderer's flat colour is not a photograph, and the thresholds know it.

    Raising the saturation floor is what stopped the render's own grey-blue
    background being classified as blue bricks, which produced one blob covering
    the whole image. A photograph's background is arbitrary, so this threshold
    is the first thing to fail on real frames and is kept a parameter for that
    reason.
    """
    from fishsense_wuwnet.target_detect import classify_colours

    rotation, translation = look_at((260.0, -330.0, 190.0))
    view = render(CAMERA, rotation, translation, size=SIZE, background=(70, 60, 48))
    labels = classify_colours(view.image)
    assert (labels == 1).sum() < 0.05 * labels.size
    permissive = classify_colours(view.image, saturation_floor=70)
    assert (permissive == 1).sum() > 0.5 * labels.size
