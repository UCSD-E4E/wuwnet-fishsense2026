"""The reachable-envelope numbers section 4.2 hangs on, pinned so they cannot drift.

These once lived only in the paper's prose, computed in a transcript and never
committed -- and a later attempt to re-derive them from the raw differential
magnification, without the range triangulation, came out three times too large
and looked like a contradiction. The module now owns the computation; these
tests own the numbers.

The distinction the fast test guards is the one that caused that scare: the
reachable error is the *delivered* length error, with the range triangulated
through the same uncorrected model, and that cancellation is most of why the
number is small. Measuring the differential magnification alone is a different,
larger quantity that the rig never produces.
"""

import numpy as np
import pytest

from fishsense_wuwnet.envelope import (
    load_corpus_pairs,
    reachable_envelope,
    uncorrected_length_error,
    worst_over_bearing,
)


def test_the_dot_on_axis_makes_the_length_nearly_right():
    """A fish square across the frame with the dot at its centre, at 1 m.

    Near the axis the uncorrected range and magnification errors both run as the
    water index and cancel in the product, so the delivered length is close even
    though neither factor is. This is the mechanism the whole envelope rests on.
    """
    error = uncorrected_length_error(0.30, 1.0, bearing=0.0, along=0.0)
    assert abs(error) < 1.5


def test_a_tip_that_leaves_the_frame_is_not_reachable():
    """A large fish close in has a tip off the sensor, and is excluded, not scored."""
    assert uncorrected_length_error(0.60, 0.30, bearing=np.pi / 2, along=0.0) is None


def test_the_worst_bearing_is_larger_than_the_axis_aligned_one():
    """Bearing is maximised over because the corpus does not record it.

    The worst orientation costs more than a fish lying along one image axis, so
    the reported envelope is a strict upper bound on what a fish in that position
    can cost -- not an average over orientations.
    """
    aligned = abs(uncorrected_length_error(0.30, 1.0, bearing=0.0, along=0.0))
    worst = worst_over_bearing(0.30, 1.0)
    assert worst >= aligned
    assert worst < 15.0  # still reachable, still under budget


def test_the_range_triangulation_is_what_makes_it_small():
    """Without the triangulation the same geometry reads several times larger.

    Re-derives the raw differential magnification -- fish extent against dot
    displacement, no range cancellation -- and confirms it is much bigger, which
    is exactly the quantity that produced a false alarm before this was pinned.
    """
    from fishsense_wuwnet.envelope import _camera_and_port, _project
    from fishsense_wuwnet.pipelines import back_project_pinax, back_project_uncorrected, differential_length_error
    from fishsense_wuwnet.envelope import FOCAL, LASER_ORIGIN, LASER_AXIS, PRINCIPAL

    intrinsics, port = _camera_and_port()
    _, virtual_cop, _ = __import__("fishsense_wuwnet.refraction", fromlist=["optimal_d0"]).optimal_d0(
        0.006, 1.49, 1.333, np.radians(41.37)
    )
    length, range_m = 0.30, 1.0
    offset = float(np.hypot(LASER_ORIGIN[0], LASER_ORIGIN[1]))

    def pixel_radius(transverse):
        from fishsense_wuwnet.refraction import project_water_points, alpha_azimuth_to_pixel

        alpha, azimuth = project_water_points(np.array([[transverse, 0.0, range_m]]), port)
        px = alpha_azimuth_to_pixel(alpha, azimuth, intrinsics)[0]
        return float(np.hypot(px[0] - PRINCIPAL[0], px[1] - PRINCIPAL[1]))

    flatten = lambda d: d[..., :2] / d[..., 2:3]
    unc = lambda px: flatten(back_project_uncorrected(px, intrinsics)[1])
    pin = lambda px: flatten(back_project_pinax(px, intrinsics, port, virtual_cop)[1])
    raw = abs(
        differential_length_error(
            unc, pin, principal_point=np.array(PRINCIPAL),
            radii=np.array([pixel_radius(offset + 0.5 * length)]),
            laser_radius=pixel_radius(offset), extent="radial",
        )[0]
    ) * 100

    delivered = worst_over_bearing(length, range_m)
    assert raw > 2 * delivered


@pytest.mark.slow
def test_the_full_envelope_matches_the_paper():
    """The headline: p50 1.2 %, p90 3.0 %, max 14.3 % over the production corpus.

    Slow (~2 min): it runs the real computation over every distinct (length,
    range) geometry the corpus contains. It is the guard that the number section
    4.2 quotes is the number the code produces, so it is worth the wait; mark
    `slow` deselects it for a quick loop.
    """
    result = reachable_envelope(load_corpus_pairs())
    assert result["n_frames"] > 2800
    assert result["p50"] == pytest.approx(1.2, abs=0.1)
    assert result["p90"] == pytest.approx(3.0, abs=0.2)
    assert result["p95"] == pytest.approx(5.5, abs=0.3)
    assert result["max"] == pytest.approx(14.3, abs=0.5)
    assert result["over_15pct"] == 0.0
