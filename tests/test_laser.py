"""Tests for slate-free per-dive laser calibration.

Everything here is against a synthetic beam projected through an ideal pinhole
-- the locus result is pinhole geometry, and the point of these tests is that
the algebra closes exactly, so tolerances are tight. The real-data numbers live
in the notebook.
"""

import numpy as np
import pytest

from fishsense_wuwnet.laser import (
    beam_angle,
    close_with_one_range,
    close_with_two_ranges,
    fit_locus,
    range_along_beam,
)

O_TRUE = np.array([-0.0336, -0.0983, 0.0])
D_TRUE = np.array([0.00798, -0.00245, 1.0])
RANGES = np.linspace(0.5, 5.0, 30)


def _directions(ranges, origin=O_TRUE, axis=D_TRUE, noise=0.0, seed=0):
    """Normalised directions of the dot at each range, optionally with noise."""
    points = origin[None, :] + ranges[:, None] * axis[None, :]
    u = points[:, :2] / points[:, 2:3]
    if noise:
        u = u + np.random.default_rng(seed).normal(0.0, noise, u.shape)
    return u


def test_locus_is_a_line_through_d_along_o():
    """u(t) = D + O/t: the dots must be collinear, oriented along O, through D."""
    locus = fit_locus(_directions(RANGES), orient_toward=O_TRUE[:2])

    assert locus.rms < 1e-12
    o_hat = O_TRUE[:2] / np.linalg.norm(O_TRUE[:2])
    assert locus.direction == pytest.approx(o_hat, abs=1e-9)
    # D is on the line: its perpendicular offset from the line is zero
    assert float((D_TRUE[:2] - locus.centre) @ locus.normal) == pytest.approx(0.0, abs=1e-12)


def test_locus_alone_cannot_fix_scale():
    """Scaling |O| and every range together leaves the locus unchanged.

    This is the whole reason a range is needed: the line carries two degrees
    of freedom, and this gauge is one of the two it cannot see.
    """
    a = fit_locus(_directions(RANGES), orient_toward=O_TRUE[:2])
    b = fit_locus(_directions(3.0 * RANGES, origin=3.0 * O_TRUE), orient_toward=O_TRUE[:2])
    assert b.direction == pytest.approx(a.direction, abs=1e-12)
    assert float((b.centre - a.centre) @ a.normal) == pytest.approx(0.0, abs=1e-12)


def test_one_range_with_known_o_recovers_the_beam():
    u = _directions(RANGES)
    locus = fit_locus(u, orient_toward=O_TRUE[:2])
    i = 12
    origin, axis = close_with_one_range(locus, u[i], RANGES[i], np.linalg.norm(O_TRUE[:2]))

    assert origin == pytest.approx(O_TRUE, abs=1e-9)
    assert beam_angle(axis, D_TRUE) < 1e-6


def test_two_ranges_recover_the_beam_with_no_prior():
    u = _directions(RANGES)
    locus = fit_locus(u, orient_toward=O_TRUE[:2])
    origin, axis = close_with_two_ranges(locus, u[3], RANGES[3], u[25], RANGES[25])

    assert origin == pytest.approx(O_TRUE, abs=1e-9)
    assert beam_angle(axis, D_TRUE) < 1e-6


def test_two_close_ranges_are_ill_conditioned():
    """Adjacent ranges give 1/Z_a ~ 1/Z_b and the solve blows up under noise."""
    u = _directions(RANGES, noise=1e-4)
    locus = fit_locus(u, orient_toward=O_TRUE[:2])

    far = close_with_two_ranges(locus, u[0], RANGES[0], u[-1], RANGES[-1])
    near = close_with_two_ranges(locus, u[10], RANGES[10], u[11], RANGES[11])

    far_err = abs(np.linalg.norm(far[0][:2]) - np.linalg.norm(O_TRUE[:2]))
    near_err = abs(np.linalg.norm(near[0][:2]) - np.linalg.norm(O_TRUE[:2]))
    assert near_err > 10 * far_err


def test_five_percent_range_error_costs_a_fraction_of_a_degree():
    """The range can be crude: at 2 m, 5% moves the beam by ~|O|/Z^2 * dZ."""
    u = _directions(RANGES)
    locus = fit_locus(u, orient_toward=O_TRUE[:2])
    i = int(np.argmin(np.abs(RANGES - 2.0)))
    o_mag = np.linalg.norm(O_TRUE[:2])

    _, axis = close_with_one_range(locus, u[i], RANGES[i] * 1.05, o_mag)
    assert 0.05 < beam_angle(axis, D_TRUE) < 0.3


def test_recovered_beam_reproduces_ranges():
    """Closing the loop: ranges read back off the recovered beam match the truth."""
    u = _directions(RANGES)
    locus = fit_locus(u, orient_toward=O_TRUE[:2])
    origin, axis = close_with_two_ranges(locus, u[2], RANGES[2], u[27], RANGES[27])
    for k in (0, 7, 15, 29):
        assert range_along_beam(u[k], origin, axis) == pytest.approx(RANGES[k], rel=1e-9)


def test_sign_ambiguity_is_resolved_by_orient_toward():
    u = _directions(RANGES)
    a = fit_locus(u, orient_toward=O_TRUE[:2])
    b = fit_locus(u, orient_toward=-O_TRUE[:2])
    assert b.direction == pytest.approx(-a.direction, abs=1e-12)


def test_wrong_branch_two_ranges_raises():
    """A negative |O| means the directions were mis-assigned; fail loudly."""
    u = _directions(RANGES)
    locus = fit_locus(u, orient_toward=-O_TRUE[:2])  # deliberately flipped
    with pytest.raises(ValueError, match="not positive"):
        close_with_two_ranges(locus, u[3], RANGES[3], u[25], RANGES[25])


# --- range-free closure: the object the dot sits on is the reference ---------

from fishsense_wuwnet.laser import close_with_apparent_size, vanishing_point_budget


def _apparent_sizes(ranges, true_size=0.35, yaw_deg=None):
    """Apparent size of a rigid object of `true_size` at each range, optionally yawed."""
    sizes = true_size / ranges
    if yaw_deg is not None:
        sizes = sizes * np.cos(np.radians(yaw_deg))
    return sizes


def test_apparent_size_recovers_the_beam_with_no_range():
    u = _directions(RANGES)
    locus = fit_locus(u, orient_toward=O_TRUE[:2])
    o_mag = np.linalg.norm(O_TRUE[:2])
    origin, axis, size, rms = close_with_apparent_size(locus, u, _apparent_sizes(RANGES), o_mag)

    assert origin == pytest.approx(O_TRUE, abs=1e-9)
    assert beam_angle(axis, D_TRUE) < 1e-6
    assert size == pytest.approx(0.35, rel=1e-9)  # the object's true size falls out
    assert rms < 1e-12


def test_apparent_size_unit_does_not_matter():
    """Pixels, normalised coordinates, or furlongs: only the intercept is used."""
    u = _directions(RANGES)
    locus = fit_locus(u, orient_toward=O_TRUE[:2])
    o_mag = np.linalg.norm(O_TRUE[:2])
    _, axis_a, _, _ = close_with_apparent_size(locus, u, _apparent_sizes(RANGES), o_mag)
    _, axis_b, _, _ = close_with_apparent_size(locus, u, 1234.5 * _apparent_sizes(RANGES), o_mag)
    assert beam_angle(axis_a, axis_b) < 1e-9


def test_apparent_size_needs_range_spread():
    """Every frame at the same range: the line has no leverage and the fit refuses."""
    u = _directions(np.full(5, 2.0))
    locus = fit_locus(_directions(RANGES), orient_toward=O_TRUE[:2])
    with pytest.raises(ValueError, match="range spread"):
        close_with_apparent_size(locus, u, _apparent_sizes(np.full(5, 2.0)), 0.1)


def test_apparent_size_conditioning_matches_two_ranges():
    """A 1.2x range spread is poorly conditioned, 2x is fine -- same as Tier 2."""
    o_mag = np.linalg.norm(O_TRUE[:2])
    rng = np.random.default_rng(1)

    def error_for(spread):
        ranges = np.linspace(1.0, spread, 8)
        u = _directions(ranges, noise=2e-4)
        sizes = _apparent_sizes(ranges) * (1 + rng.normal(0.0, 0.01, len(ranges)))
        locus = fit_locus(u, orient_toward=O_TRUE[:2])
        _, axis, _, _ = close_with_apparent_size(locus, u, sizes, o_mag)
        return beam_angle(axis, D_TRUE)

    assert error_for(2.0) < 0.1
    assert error_for(1.2) > 2 * error_for(2.0)


def test_yaw_contamination_is_bounded_by_cosine():
    """A fish turning 20 deg between frames changes apparent length by 6% -- inside 15%."""
    u = _directions(RANGES)
    locus = fit_locus(u, orient_toward=O_TRUE[:2])
    o_mag = np.linalg.norm(O_TRUE[:2])
    yaw = np.linspace(0.0, 20.0, len(RANGES))
    _, axis, _, _ = close_with_apparent_size(locus, u, _apparent_sizes(RANGES, yaw_deg=yaw), o_mag)
    p_d_err = float(locus.position_along(axis[:2]) - locus.position_along(D_TRUE[:2]))
    assert abs(p_d_err) < vanishing_point_budget(0.15, o_mag, 4.0)


def test_budget_is_tightest_at_the_farthest_range():
    o_mag = 0.104
    assert vanishing_point_budget(0.15, o_mag, 4.0) < vanishing_point_budget(0.15, o_mag, 2.0)
    # 15% at 4 m with a 104 mm baseline: 3.9 mrad
    assert vanishing_point_budget(0.15, o_mag, 4.0) == pytest.approx(0.0039, rel=1e-6)
