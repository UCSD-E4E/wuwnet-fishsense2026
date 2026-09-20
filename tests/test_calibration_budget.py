"""How an intrinsics error reaches the length, and why the answer has two regimes.

The target exists to make an in-air calibration good enough to replace the wet
one. "Good enough" needs a number, and the number is not a property of the
calibration alone: it depends on whether the laser's extrinsics are trusted from
the bench or re-fitted every dive, and those two cases have *opposite*
sensitivities. Trusting the bench, range and transverse extent scale together
and a focal-length error cancels out of the length while a principal-point error
is ruinous. Re-fitting the beam pins the range, which absorbs the principal
point almost entirely and in the same motion destroys the cancellation that was
protecting length from the focal length.

Getting this backwards would mean optimising the target for the wrong parameter,
so it is pinned here rather than left in a notebook. The forward model is the
exact flat-port projection; only the inverse is given the wrong intrinsics.
"""

import numpy as np
import pytest

from fishsense_wuwnet.laser import fit_beam_to_ranges
from fishsense_wuwnet.pipelines import back_project_pinax, measure_length
from fishsense_wuwnet.refraction import (
    SWEET_WATER,
    FlatPort,
    alpha_azimuth_to_pixel,
    optimal_d0,
    project_water_points,
    reconstruct_points,
)

CAMERA = np.array([[2876.0, 0.0, 2000.0], [0.0, 2845.0, 1500.0], [0.0, 0.0, 1.0]])
_D0, _VIRTUAL_COP, _ = optimal_d0(0.006, 1.49, SWEET_WATER, np.radians(41.0))
PORT = FlatPort(_D0, 0.006, 1.49, SWEET_WATER)

BEAM_ORIGIN = np.array([-0.04, -0.11, 0.0])
BEAM_AXIS = np.array([0.0, 0.0, 1.0])
LENGTH = 0.30
DOT_FRACTION = 0.52  # where on the body the dot actually lands, from the corpus
CALIBRATION_RANGES = np.array([0.8, 1.2, 1.8, 2.5, 3.5, 4.5])
RANGES = np.array([1.0, 1.5, 2.0, 3.0, 4.0])


def _pixels(points):
    """Image the points with the *true* camera, through the exact port model."""
    alpha, azimuth = project_water_points(points, PORT)
    return alpha_azimuth_to_pixel(alpha, azimuth, CAMERA)


def _fish(orientation):
    axis = {"across": np.array([1.0, 0.0, 0.0]), "along": np.array([0.0, 1.0, 0.0])}[
        orientation
    ]
    centre = BEAM_ORIGIN + RANGES[:, None] * BEAM_AXIS
    return (
        centre + axis * (LENGTH * (1.0 - DOT_FRACTION)),
        centre - axis * (LENGTH * DOT_FRACTION),
        centre,
    )


def _worst_error(used, orientation="across", refit=False, initial=None):
    """Worst relative length and range error over the range sweep, in percent."""
    back = lambda pixels: back_project_pinax(pixels, used, PORT, _VIRTUAL_COP)

    origin, axis = BEAM_ORIGIN, BEAM_AXIS
    if refit:
        rays = back(_pixels(BEAM_ORIGIN + CALIBRATION_RANGES[:, None] * BEAM_AXIS))
        origin, axis = fit_beam_to_ranges(rays[1], rays[0], CALIBRATION_RANGES, initial)

    head, tail, dot = _fish(orientation)
    ray_origin, direction = back(_pixels(dot))
    recovered, _ = reconstruct_points(direction, ray_origin, origin, axis)
    length = measure_length(_pixels(head), _pixels(tail), recovered[:, 2], back)
    return (
        100.0 * np.abs(length / LENGTH - 1.0).max(),
        100.0 * np.abs(recovered[:, 2] / RANGES - 1.0).max(),
    )


def _scaled(fx=1.0, fy=1.0, dcx=0.0, dcy=0.0):
    used = CAMERA.copy()
    used[0, 0] *= fx
    used[1, 1] *= fy
    used[0, 2] += dcx
    used[1, 2] += dcy
    return used


def test_the_floor_with_nothing_wrong_is_pinax_itself():
    """A control, so that every number below is attributable to one perturbation.

    It does not come out at zero, and the residue is not numerical: the forward
    model is the *exact* axial projection and the inverse is Pinax, which is the
    central approximation to it. What is left with perfect intrinsics is
    therefore the model error alone -- three ten-thousandths of a percent of
    length. That is the floor every figure in this file sits on, and it is three
    to four orders below the errors being studied, which is the reason those
    figures can be read as calibration error rather than as anything else.
    """
    for refit in (False, True):
        length, ranged = _worst_error(CAMERA, refit=refit)
        assert length == pytest.approx(2.8e-4, rel=0.5)
        assert ranged < 1e-3


def test_trusting_the_bench_a_focal_error_cancels_but_the_principal_point_does_not():
    """Range and transverse extent scale together, so the focal length divides out.

    The principal point does not, and it is far worse than it looks: the dot's
    offset from the principal point *is* the triangulation baseline in the
    image, and at 4 m that offset is only about eighty pixels. Five pixels of
    error is six percent of it.
    """
    focal_length, focal_range = _worst_error(_scaled(fx=1.05, fy=1.05))
    assert focal_length < 0.2
    assert focal_range == pytest.approx(5.0, abs=0.1)

    centre_length, centre_range = _worst_error(_scaled(dcx=5.0, dcy=5.0))
    assert centre_length > 5.0
    assert centre_range > 5.0


def test_refitting_the_beam_each_dive_reverses_which_parameter_binds():
    """And it is a reversal, not an improvement across the board.

    The beam is re-fitted through the same wrong intrinsics, so it absorbs the
    principal-point error almost completely -- twenty pixels costs a tenth of a
    percent. But pinning the range is exactly what destroys the cancellation
    above, so the focal error now arrives at the length essentially one for one.
    """
    centre_length, _ = _worst_error(_scaled(dcx=20.0, dcy=20.0), refit=True)
    assert centre_length < 0.2

    focal_length, focal_range = _worst_error(_scaled(fx=1.05, fy=1.05), refit=True)
    assert focal_length == pytest.approx(4.75, abs=0.2)
    assert focal_range < 0.05


def test_an_anisotropy_reaches_only_the_axis_it_lies_on():
    """Which is why `fx/fy` is the quantity the three-dimensional target is for.

    A fish held across the frame is measured by `fx` alone, so an error in the
    ratio arrives whole; held along the frame it is measured by `fy` and does
    not arrive at all. A planar target leaves this ratio poorly constrained
    unless the views are rolled.
    """
    across, _ = _worst_error(_scaled(fx=1.01), refit=True, orientation="across")
    along, _ = _worst_error(_scaled(fx=1.01), refit=True, orientation="along")
    assert across == pytest.approx(1.0, abs=0.05)
    assert along < 0.01


def test_the_fitted_beam_is_not_identifiable_but_its_predictions_are():
    """The camera rays are nearly parallel to the beam, so the fit has a valley.

    Started from five very different guesses the recovered origin moves by
    almost half a metre -- and the length it predicts is the same to three
    decimal places. Worth pinning because the natural reaction to an
    unidentifiable fit is to distrust what it is used for, and here that would
    be wrong: the measurement depends only on the part that is determined.
    """
    used = _scaled(fx=1.05, fy=1.05)
    starts = [
        [0.0, -0.1, 0.0, 0.0],
        [-0.04, -0.11, 0.0, 0.0],
        [0.1, 0.05, 0.01, -0.01],
        [-0.2, -0.3, 0.0, 0.0],
        [0.3, 0.2, -0.02, 0.02],
    ]
    origins, lengths = [], []
    for start in starts:
        rays = back_project_pinax(
            _pixels(BEAM_ORIGIN + CALIBRATION_RANGES[:, None] * BEAM_AXIS),
            used,
            PORT,
            _VIRTUAL_COP,
        )
        origin, _ = fit_beam_to_ranges(rays[1], rays[0], CALIBRATION_RANGES, start)
        origins.append(origin[:2])
        lengths.append(_worst_error(used, refit=True, initial=start)[0])

    assert np.ptp(np.array(origins), axis=0).max() > 0.2
    assert np.ptp(np.array(lengths)) < 0.01


def test_the_in_air_calibration_we_measured_is_comfortably_inside_the_budget():
    """The closing number: our own calibration error, propagated to a length.

    The 4000x3000 sweep on the LEGO target returned fx +0.033 %, fy +0.017 % and
    the principal point within a few pixels. Through a per-dive beam fit that is
    worth a twentieth of a percent of length, against a 15 % worst-case budget.
    """
    used = _scaled(fx=1.00033, fy=1.00017, dcx=3.3, dcy=0.6)
    for orientation in ("across", "along"):
        length, ranged = _worst_error(used, orientation=orientation, refit=True)
        assert length < 0.1
        assert ranged < 0.01
