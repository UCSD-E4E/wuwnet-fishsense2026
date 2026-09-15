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


def test_error_is_zero_at_the_laser_radius():
    """By construction: that is the field position the range was referenced at."""
    radii = np.array([LASER_RADIUS])
    for k in (0.05, 0.15, -0.1):
        error = differential_length_error(_radial(k), _ideal, PRINCIPAL, radii, LASER_RADIUS)
        assert error[0] == pytest.approx(0.0, abs=1e-12)


def test_moving_the_laser_outward_shifts_the_zero_crossing():
    """Where the dot lands sets where the measurement is unbiased."""
    model = _radial(0.15)
    for laser_radius in (330.0, 1200.0):
        error = differential_length_error(model, _ideal, PRINCIPAL, RADII, laser_radius)
        crossing = RADII[np.argmin(np.abs(error))]
        assert crossing == pytest.approx(laser_radius, rel=0.1)


def test_sign_flips_either_side_of_the_reference():
    """Inside the laser radius the length is under-read, outside it is over-read."""
    error = differential_length_error(_radial(0.15), _ideal, PRINCIPAL, RADII, 1200.0)
    assert error[RADII < 1200.0].max() < 0.0
    assert error[RADII > 1200.0].min() > 0.0
