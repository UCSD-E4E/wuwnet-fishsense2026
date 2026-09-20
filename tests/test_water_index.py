"""What assuming a water index costs, and whether two presets are enough.

Section 8 lists the assumed index as a limitation. This says how big it is.

The question is narrow and worth stating precisely: the correction needs a
number for `n_w`, the deployment does not measure one, and the paper should say
what that costs rather than leaving a reader to wonder. The answer is that it
costs well under a percent everywhere reachable -- which is the finding, and is
only useful if it is pinned, because "small" is exactly the kind of claim that
rots when a constant is edited.

Two structural results matter more than the magnitudes:

**Only Pinax can adapt.** Its index is a parameter, so a preset switch is a
constant. An in-water SVP calibration bakes the index into its fitted focal
length, so the same switch costs another calibration session in water of the
right salinity. That asymmetry is the paper's argument for calibrating in air,
seen from an angle the rest of the evaluation does not reach.

**The effective index drifts with range**, because attenuation is
wavelength-dependent. That makes it a *range-correlated* term rather than a
constant one, which is the category that reaches the beam fit rather than
cancelling -- so it is worth naming even though it is small.

Geometry throughout is the reachable one: the dot must land on the fish, so the
fish sits near the axis and only its own extent reaches outward. Quoting these
errors at a frame corner would repeat the unreachability mistake §4.2 exists to
correct.
"""

import numpy as np
import pytest

from fishsense_wuwnet import water
from fishsense_wuwnet.pipelines import back_project_pinax, back_project_svp, measure_length
from fishsense_wuwnet.refraction import (
    SALTY_WATER,
    SWEET_WATER,
    FlatPort,
    alpha_azimuth_to_pixel,
    fit_svp_model,
    optimal_d0,
    project_water_points,
)

CAMERA = np.array([[2850.0, 0.0, 2007.0], [0.0, 2850.0, 1508.0], [0.0, 0.0, 1.0]])
HALF_FOV = np.arctan(np.hypot(2007.0, 1508.0) / 2850.0)
GLASS_T, N_GLASS = 0.006, 1.49
LENGTH, DEPTH = 0.30, 2.0

#: |O|, the laser offset. The fish is centred on the dot, so this is where a
#: measurable fish actually sits -- see the module docstring.
OFFSET = 0.104

#: The housing is one piece of hardware; its camera-to-glass gap does not change
#: when the diver moves from a pool to the sea. Only the water does.
D0, _, _ = optimal_d0(GLASS_T, N_GLASS, SALTY_WATER, HALF_FOV)

#: Envelope midpoints, as against the round numbers the paper inherited.
FRESH_CENTRED, SEA_CENTRED = 1.33423, 1.34048

_COP: dict = {}
_SVP: dict = {}


def _cop(n):
    return _COP.setdefault(n, optimal_d0(GLASS_T, N_GLASS, n, HALF_FOV)[1])


def _svp(n):
    return _SVP.setdefault(
        n, fit_svp_model(FlatPort(D0, GLASS_T, N_GLASS, n), HALF_FOV, DEPTH)
    )


def _pixels(n_true, depth=DEPTH):
    """Image the fish's endpoints through the *true* water, exactly."""
    port = FlatPort(D0, GLASS_T, N_GLASS, n_true)
    points = np.array(
        [[OFFSET + LENGTH / 2, 0.0, depth], [OFFSET - LENGTH / 2, 0.0, depth]]
    )
    alpha, azimuth = project_water_points(points, port)
    return alpha_azimuth_to_pixel(alpha, azimuth, CAMERA)


def pinax_signed_error(n_true, n_assumed, depth=DEPTH):
    """Signed length error (%) correcting `n_true` water as though `n_assumed`.

    Signed matters: across the visible band the true index falls either side of
    any single assumed value, so the errors change sign. Take absolute values
    first and the linearity the effective-index construction rests on disappears
    into a fold at zero.
    """
    px = _pixels(n_true, depth)
    port = FlatPort(D0, GLASS_T, N_GLASS, n_assumed)
    measured = measure_length(
        px[0], px[1], depth, lambda q: back_project_pinax(q, CAMERA, port, _cop(n_assumed))
    )
    return (float(measured) - LENGTH) / LENGTH * 100


def pinax_error(n_true, n_assumed, depth=DEPTH):
    """Magnitude of the above, which is what a budget is quoted in."""
    return abs(pinax_signed_error(n_true, n_assumed, depth))


def svp_error(n_true, n_calibrated_in, depth=DEPTH):
    """The same, for an in-water calibration performed in `n_calibrated_in`."""
    px = _pixels(n_true, depth)
    measured = measure_length(
        px[0], px[1], depth, lambda q: back_project_svp(q, CAMERA, _svp(n_calibrated_in))
    )
    return abs(float(measured) - LENGTH) / LENGTH * 100


def _worst(indices, preset):
    return max(pinax_error(n, preset) for n in indices)


# --------------------------------------------------------------------------
# The physics, before any geometry


def test_quan_fry_reproduces_its_own_published_values():
    # Anchors the coefficients. Both are quoted in the literature to 4 decimals.
    assert water.index(0.0, 20.0, 550.0) == pytest.approx(1.3343, abs=5e-4)
    assert water.index(35.0, 20.0, 550.0) == pytest.approx(1.3408, abs=5e-4)


def test_the_two_water_types_do_not_overlap():
    """A two-mode preset is a well-posed classification, not a guess."""
    fresh = water.envelope(water.FRESH_ENVELOPE)
    sea = water.envelope(water.SEA_ENVELOPE)
    assert max(fresh) < min(sea)
    # And the gap is a decent fraction of the step the presets are meant to span.
    assert min(sea) - max(fresh) > 0.003


def test_temperature_matters_as_much_as_salinity():
    """A CTD reading alone does not settle the index."""
    base = water.index(33.0, 15.0)
    from_salinity = abs(water.index(38.0, 15.0) - base)
    from_temperature = abs(water.index(33.0, 25.0) - base)
    assert from_temperature == pytest.approx(from_salinity, rel=0.5)
    assert from_temperature > 0.0009


def test_dispersion_is_comparable_to_the_whole_fresh_to_sea_step():
    """So an index quoted without a wavelength is underspecified.

    Larger than the separation between the two envelopes' midpoints, smaller
    than the gap between the paper's round constants -- the same order as the
    thing a salinity correction is trying to fix, not a rounding detail.
    """
    dispersion = water.index(33.0, 15.0, 450.0) - water.index(33.0, 15.0, 650.0)
    fresh = water.envelope(water.FRESH_ENVELOPE)
    sea = water.envelope(water.SEA_ENVELOPE)
    midpoint_step = sum(sea) / len(sea) - sum(fresh) / len(fresh)

    assert dispersion > midpoint_step
    assert dispersion < SALTY_WATER - SWEET_WATER


# --------------------------------------------------------------------------
# What it costs in length


def test_the_error_is_linear_in_the_index_so_an_effective_index_is_exact():
    """Averaging the index and averaging the errors agree, to rounding.

    This is what licenses `water.effective_index`. Were the mapping curved, a
    single effective number would be an approximation hiding a spread, and the
    whole construction would need replacing with a per-wavelength integral.
    """
    assumed = water.index(33.0, 15.0, 550.0)
    weights, errors, indices = [], [], []
    for lam, absorption in water.PURE_WATER_ABSORPTION.items():
        weight = np.exp(-absorption * 2 * DEPTH)
        n = water.index(33.0, 15.0, lam)
        weights.append(weight)
        indices.append(n)
        errors.append(pinax_signed_error(n, assumed))
    weights = np.array(weights)
    at_mean_index = pinax_signed_error(float(np.average(indices, weights=weights)), assumed)
    mean_of_errors = float(np.average(errors, weights=weights))
    assert at_mean_index == pytest.approx(mean_of_errors, abs=0.01)


def test_assuming_fresh_water_on_a_sea_dive_costs_under_a_percent():
    """The cost of doing nothing at all -- the figure's headline, made reachable."""
    worst = _worst(water.envelope(water.SEA_ENVELOPE), SWEET_WATER)
    assert 0.5 < worst < 1.0
    # Against a 15 % budget this is a twentieth. It is a limitation, not a defect.
    assert worst < 15.0 / 10


def test_two_presets_remove_most_of_it():
    sea = water.envelope(water.SEA_ENVELOPE)
    always_fresh = _worst(sea, SWEET_WATER)
    two_mode = max(
        pinax_error(n, water.nearest_preset(n, (SWEET_WATER, SALTY_WATER))) for n in sea
    )
    assert two_mode < always_fresh / 2


def test_the_inherited_presets_sit_at_the_edge_of_their_envelopes():
    """Re-centring is a two-constant change worth more than adding the second mode.

    1.333 and 1.342 are round numbers from the Pinax paper, not envelope
    midpoints, and each sits near one end of the range it is meant to represent.
    """
    for bounds, inherited, centred in (
        (water.FRESH_ENVELOPE, SWEET_WATER, FRESH_CENTRED),
        (water.SEA_ENVELOPE, SALTY_WATER, SEA_CENTRED),
    ):
        indices = water.envelope(bounds)
        assert _worst(indices, centred) < _worst(indices, inherited) / 1.5


def test_only_pinax_can_switch_presets_without_another_calibration():
    """The SVP baseline needs a salt-water session to reach the same place.

    Both models degrade at much the same rate when the index is wrong -- the
    error is in the physics, not the correction. What differs is the price of
    fixing it, and that is the asymmetry the paper is about.
    """
    sea = water.envelope(water.SEA_ENVELOPE)
    pinax_switched = _worst(sea, SALTY_WATER)
    svp_stuck = max(svp_error(n, SWEET_WATER) for n in sea)
    svp_recalibrated = max(svp_error(n, SALTY_WATER) for n in sea)

    # Switching a constant gets Pinax where a whole pool session gets the SVP.
    assert pinax_switched == pytest.approx(svp_recalibrated, abs=0.15)
    assert svp_stuck > 2 * svp_recalibrated


# --------------------------------------------------------------------------
# The broadband index, and the one term that is range-correlated


def test_the_effective_index_sits_above_the_mid_visible_value():
    """Water strips red, so what survives is blue-shifted, and blue refracts more."""
    for salinity in (0.0, 33.0):
        reference = water.index(salinity, 15.0, water.REFERENCE_WAVELENGTH_NM)
        assert water.effective_index(salinity, 15.0, DEPTH) > reference + 0.002


def test_the_effective_index_climbs_with_range_which_is_the_dangerous_kind():
    """A constant size bias never reaches the beam; a range-correlated one does.

    Clear water only. Coastal water absorbs blue instead and reverses this, so
    the sign is a property of the water, not of the geometry.
    """
    near = water.effective_index(33.0, 15.0, 1.0)
    far = water.effective_index(33.0, 15.0, 4.5)
    assert far > near
    drift = pinax_error(far, near)
    # Small, but it does not average away over a dive the way a constant does.
    assert 0.05 < drift < 0.5


def test_the_laser_dot_needs_no_effective_index_at_all():
    """It is monochromatic, so its index is exact and costs nothing to get right."""
    exact = water.index(33.0, 15.0, water.LASER_WAVELENGTH_NM)
    assumed = water.index(33.0, 15.0, water.REFERENCE_WAVELENGTH_NM)
    # Using the mid-visible value for the dot is a real but tiny error, and
    # unlike the silhouette's it is removable by substituting one constant.
    assert 0.0 < pinax_error(exact, assumed) < 0.1
