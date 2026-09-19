"""Tests for the two length-error estimators.

The estimators exist because the obvious measurement gives the wrong answer. A
fish's length is read from the camera while its range is read from a laser dot
that lands near the optical axis, so the two are sampled at different field
positions. The error that matters is the *difference* in magnification between
those positions, not the distortion across the fish's own extent -- and the
latter is about fifty times smaller on real data, so measuring it instead makes
an uncorrected flat port look harmless.
"""

import numpy as np
import pytest

from fishsense_wuwnet.pipelines import differential_length_error

PRINCIPAL = (2007.0, 1508.0)
RADII = np.linspace(10.0, 2500.0, 40)
LASER_RADIUS = 330.0  # where the pool laser dot actually sits


def _ideal(pixels):
    """A distortion-free reference: pixel offset straight to normalised direction."""
    return (np.asarray(pixels, dtype=float) - np.asarray(PRINCIPAL)) / 2900.0


def _radial(k):
    """A model with a radial magnification error of the flat-port kind."""

    def model(pixels):
        xy = _ideal(pixels)
        r2 = np.sum(xy**2, axis=1, keepdims=True)
        return xy * (1.0 + k * r2)

    return model


def test_identical_model_has_no_error():
    error = differential_length_error(_ideal, _ideal, PRINCIPAL, RADII, LASER_RADIUS)
    assert error == pytest.approx(np.zeros_like(RADII), abs=1e-12)


def test_pure_scale_error_cancels():
    """The central analytic result: a uniform scale error does not reach length.

    Range and transverse extent scale together, so the factor divides out. It
    lands entirely in *range* instead -- which is why an uncorrected flat port
    costs ~32% in range and far less in length.
    """
    scaled = lambda pixels: 1.7 * _ideal(pixels)
    error = differential_length_error(scaled, _ideal, PRINCIPAL, RADII, LASER_RADIUS)
    assert error == pytest.approx(np.zeros_like(RADII), abs=1e-12)


def test_radial_error_grows_with_field_position():
    """A radial magnification error does not cancel, and grows outward."""
    error = differential_length_error(_radial(0.15), _ideal, PRINCIPAL, RADII, LASER_RADIUS)

    beyond_laser = RADII > LASER_RADIUS
    assert np.all(np.diff(error) > 0)
    assert np.all(error[beyond_laser] > 0)
    assert error[-1] > 0.05


def test_error_is_zero_at_the_laser_radius_for_a_tangential_span():
    """By construction: that is the field position the range was referenced at.

    Only exactly true for a span across the radius, because that is the case
    whose magnification is the same quantity the range was referenced through.
    A span *along* the radius is stretched by the derivative instead, which does
    not equal the position magnification even at the dot -- see the next test.
    """
    radii = np.array([LASER_RADIUS])
    for k in (0.05, 0.15, -0.1):
        error = differential_length_error(
            _radial(k), _ideal, PRINCIPAL, radii, LASER_RADIUS, extent="tangential"
        )
        assert error[0] == pytest.approx(0.0, abs=1e-12)


def test_a_radial_span_is_not_unbiased_even_at_the_laser_radius():
    """The zero of the differential error is not where the dot is, for a radial span.

    Worth pinning because it is easy to assume otherwise: the dot sets the
    range through its position, the fish sets the length through its extent,
    and off the axis those are different magnifications even at the same
    radius. Small, but not zero, and it does not vanish with a better model --
    only with a fish held across the radius rather than along it.
    """
    radii = np.array([LASER_RADIUS])
    error = differential_length_error(
        _radial(0.15), _ideal, PRINCIPAL, radii, LASER_RADIUS, extent="radial"
    )
    assert abs(error[0]) > 1e-6
    assert abs(error[0]) < 0.02


def test_moving_the_laser_outward_shifts_the_zero_crossing():
    """Where the dot lands sets where the measurement is unbiased."""
    model = _radial(0.15)
    for laser_radius in (330.0, 1200.0):
        error = differential_length_error(model, _ideal, PRINCIPAL, RADII, laser_radius, extent="tangential")
        crossing = RADII[np.argmin(np.abs(error))]
        assert crossing == pytest.approx(laser_radius, rel=0.1)


def test_sign_flips_either_side_of_the_reference():
    """Inside the laser radius the length is under-read, outside it is over-read."""
    error = differential_length_error(_radial(0.15), _ideal, PRINCIPAL, RADII, 1200.0, extent="tangential")
    assert error[RADII < 1200.0].max() < 0.0
    assert error[RADII > 1200.0].min() > 0.0


def test_the_axial_model_reconstructs_the_simulation_exactly():
    """The exact model should recover a synthetic point to machine precision.

    The simulation projects water points through the same flat-port physics the
    axial back-projection inverts, so a round trip has nothing left to
    approximate. Pinax, being the central approximation to it, must not be
    exact -- and that difference is the whole content of the comparison, so both
    halves are asserted.
    """
    from fishsense_wuwnet.pipelines import back_project_axial, back_project_pinax
    from fishsense_wuwnet.refraction import (
        FlatPort,
        alpha_azimuth_to_pixel,
        optimal_d0,
        project_water_points,
        reconstruct_points,
        SWEET_WATER,
    )

    K = np.array([[2850.0, 0.0, 2007.0], [0.0, 2850.0, 1508.0], [0.0, 0.0, 1.0]])
    d0, virtual_cop, _ = optimal_d0(0.006, 1.49, SWEET_WATER, np.radians(41.0))
    port = FlatPort(d0, 0.006, 1.49, SWEET_WATER)

    laser_origin = np.array([-0.04, -0.11, 0.0])
    laser_axis = np.array([0.0, 0.0, 1.0])
    ranges = np.linspace(0.6, 4.0, 12)
    truth = laser_origin + ranges[:, None] * laser_axis

    alpha, azimuth = project_water_points(truth, port)
    pixels = alpha_azimuth_to_pixel(alpha, azimuth, K)

    origins, directions = back_project_axial(pixels, K, port)
    recovered, _ = reconstruct_points(directions, origins, laser_origin, laser_axis)
    assert recovered == pytest.approx(truth, abs=1e-9)

    cop, directions = back_project_pinax(pixels, K, port, virtual_cop)
    approx, _ = reconstruct_points(directions, cop, laser_origin, laser_axis)
    residual = np.abs(approx[:, 2] - truth[:, 2]).max()
    assert residual > 1e-9, "Pinax is the central approximation; it cannot be exact"
    assert residual < 1e-3, "but it should still be sub-millimetre over this range"


def test_length_error_depends_on_how_the_fish_is_held():
    """A flat port's magnification is symmetric but not isotropic.

    A span lying along the radius is stretched by the derivative of the radial
    mapping; a span across it by the mapping itself. They differ substantially
    off axis, so reporting one number per field radius describes neither case.
    """
    from fishsense_wuwnet.pipelines import back_project_pinax, back_project_uncorrected, differential_length_error
    from fishsense_wuwnet.refraction import FlatPort, SWEET_WATER, optimal_d0

    K = np.array([[2850.0, 0.0, 2007.0], [0.0, 2850.0, 1508.0], [0.0, 0.0, 1.0]])
    d0, virtual_cop, _ = optimal_d0(0.006, 1.49, SWEET_WATER, np.radians(41.0))
    port = FlatPort(d0, 0.006, 1.49, SWEET_WATER)
    # Image-plane (tan) coordinates, as the notebooks pass: the first two
    # components of the *unit* direction are sin-space, where the refraction is
    # exactly a factor of n_w and so cancels as a pure scale by construction.
    flatten = lambda d: d[..., :2] / d[..., 2:3]
    uncorrected = lambda px: flatten(back_project_uncorrected(px, K)[1])
    pinax = lambda px: flatten(back_project_pinax(px, K, port, virtual_cop)[1])

    radii = np.array([2500.0])
    kw = dict(principal_point=K[:2, 2], radii=radii, laser_radius=330.0)
    radial = float(differential_length_error(uncorrected, pinax, extent="radial", **kw)[0])
    tangential = float(differential_length_error(uncorrected, pinax, extent="tangential", **kw)[0])

    # Both are large and same-signed, but the radial case is the worse one by a
    # factor of roughly three -- the point the single-column table obscured.
    assert abs(radial) > 2 * abs(tangential)
    assert np.sign(radial) == np.sign(tangential)

    with pytest.raises(ValueError, match="radial"):
        differential_length_error(uncorrected, pinax, extent="diagonal", **kw)
