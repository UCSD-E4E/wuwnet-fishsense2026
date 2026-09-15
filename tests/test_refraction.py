"""Validation of the flat-port model against Łuczyński et al. (2017).

The point of these tests is that the notebook's conclusions rest on the geometry
being right. Table 1 of the paper is the one published numeric oracle; the rest
are internal-consistency checks that would catch a sign or convention slip.
"""

import numpy as np
import pytest

from fishsense_wuwnet.refraction import (
    SALTY_WATER,
    SWEET_WATER,
    FlatPort,
    LayeredPort,
    exit_ray,
    axis_crossing,
    field_angle,
    fit_svp_model,
    svp_scene_radius,
    focus_section,
    optimal_d0,
    pixel_to_alpha_azimuth,
    project_water_points,
    ray_directions,
    reconstruct_points,
    alpha_azimuth_to_pixel,
    water_radius,
)

HALF_FOV = np.radians(35.0)  # the paper's standard 70 deg example
N_GLASS = 1.5

# Table 1: glass thickness (mm) -> (d0*, virtual centre of projection), in mm.
TABLE_1_SWEET = {
    1: (0.15, 0.06),
    3: (0.45, 0.18),
    5: (0.76, 0.31),
    10: (1.52, 0.61),
    15: (2.28, 0.92),
    20: (3.04, 1.22),
}
TABLE_1_SALTY = {
    1: (0.14, 0.06),
    3: (0.42, 0.17),
    5: (0.70, 0.29),
    10: (1.40, 0.58),
    15: (2.10, 0.87),
    20: (2.80, 1.15),
}


@pytest.mark.parametrize("d1,expected", TABLE_1_SWEET.items())
def test_table_1_sweet_water(d1, expected):
    d0_star, virtual_cop, _ = optimal_d0(d1, N_GLASS, SWEET_WATER, HALF_FOV)
    assert d0_star == pytest.approx(expected[0], rel=0.02, abs=0.01)
    assert virtual_cop == pytest.approx(expected[1], rel=0.03, abs=0.01)


@pytest.mark.parametrize("d1,expected", TABLE_1_SALTY.items())
def test_table_1_salty_water(d1, expected):
    d0_star, virtual_cop, _ = optimal_d0(d1, N_GLASS, SALTY_WATER, HALF_FOV)
    assert d0_star == pytest.approx(expected[0], rel=0.02, abs=0.01)
    assert virtual_cop == pytest.approx(expected[1], rel=0.03, abs=0.01)


def test_optimal_d0_is_independent_of_length_unit():
    """The same housing in mm and in m must give the same answer."""
    in_mm, _, _ = optimal_d0(10.0, N_GLASS, SWEET_WATER, HALF_FOV)
    in_m, _, _ = optimal_d0(0.010, N_GLASS, SWEET_WATER, HALF_FOV)
    assert in_m * 1000.0 == pytest.approx(in_mm, rel=1e-6)


def test_focus_section_collapses_at_the_optimum():
    """The whole premise: at d0* the axial camera is a virtual pinhole."""
    d0_star, _, best = optimal_d0(10.0, N_GLASS, SWEET_WATER, HALF_FOV)
    _, far = focus_section(FlatPort(50.0, 10.0, N_GLASS, SWEET_WATER), HALF_FOV)
    assert best < 0.02
    assert far > 100 * best


def test_optimal_d0_depends_strongly_on_field_of_view():
    """There is no FOV-free optimum -- a wide lens moves it several times over."""
    narrow, _, _ = optimal_d0(10.0, N_GLASS, SWEET_WATER, np.radians(10.0))
    wide, _, _ = optimal_d0(10.0, N_GLASS, SWEET_WATER, np.radians(70.0))
    assert narrow > 3 * wide


def test_field_angle_inverts_water_radius():
    """Forward projection must invert the forward ray trace."""
    port = FlatPort(0.0015, 0.010, N_GLASS, SWEET_WATER)
    alpha = np.linspace(0.001, HALF_FOV, 64)
    for z in (0.5, 2.0, 5.0):
        recovered = field_angle(water_radius(alpha, z, port), z, port)
        assert recovered == pytest.approx(alpha, abs=1e-9)


def test_field_angle_rejects_points_inside_the_port():
    port = FlatPort(0.0015, 0.010, N_GLASS, SWEET_WATER)
    with pytest.raises(ValueError, match="beyond the outer glass"):
        field_angle(0.1, port.d2 * 0.5, port)


def test_no_refraction_reduces_to_a_straight_line():
    """With every index equal the port is invisible: alpha = atan(radius / z)."""
    port = FlatPort(0.0015, 0.010, 1.0, 1.0)
    for radius, z in ((0.3, 2.0), (0.05, 5.0)):
        assert field_angle(radius, z, port) == pytest.approx(np.arctan(radius / z))


def test_water_ray_direction_ignores_the_glass():
    """Glass shifts a ray sideways but cannot change its final angle in water."""
    alpha = np.radians(30.0)
    expected = np.arcsin(np.sin(alpha) / 1.34)
    for n_glass in (1.4, 1.5, 1.6):
        for d1 in (0.005, 0.020):
            _, gamma = __import__(
                "fishsense_wuwnet.refraction", fromlist=["exit_ray"]
            ).exit_ray(alpha, FlatPort(0.002, d1, n_glass, 1.34))
            assert gamma == pytest.approx(expected)


def test_axis_crossing_matches_the_thin_ray_limit():
    """As alpha -> 0 the crossing depth tends to d2 - n_w * (d0 + d1 / n_g)."""
    port = FlatPort(0.0015, 0.010, N_GLASS, SWEET_WATER)
    limit = port.d2 - SWEET_WATER * (port.d0 + port.d1 / N_GLASS)
    assert axis_crossing(1e-7, port) == pytest.approx(limit, rel=1e-6)


def test_svp_focal_scale_tends_to_the_water_index_paraxially():
    """The narrow-lens limit of an in-water SVP calibration is exactly n_w."""
    port = FlatPort(0.0015, 0.010, N_GLASS, SWEET_WATER)
    coeffs = fit_svp_model(port, np.radians(0.5), 2.0)
    assert coeffs[0] == pytest.approx(SWEET_WATER, rel=1e-3)


def test_svp_is_accurate_at_its_calibration_depth():
    """A fitted SVP model reproduces the flat port almost exactly where it was calibrated.

    This is what makes it a fair baseline rather than a strawman -- and it is why
    the paper's comparison is about *depth generalisation*, not about the model
    being wrong everywhere.
    """
    port = FlatPort(0.0015, 0.010, N_GLASS, SWEET_WATER)
    coeffs = fit_svp_model(port, HALF_FOV, 2.0)

    alpha = np.linspace(0.01, HALF_FOV, 32)
    truth = water_radius(alpha, 2.0, port) / 2.0
    assert svp_scene_radius(np.tan(alpha), coeffs) == pytest.approx(truth, rel=2e-3)


def _svp_depth_errors(port, coeffs, depths, half_fov=HALF_FOV):
    alpha = np.linspace(0.01, half_fov, 32)
    errors = []
    for depth in depths:
        truth = water_radius(alpha, depth, port) / depth
        errors.append(
            np.max(np.abs(svp_scene_radius(np.tan(alpha), coeffs) - truth) / truth)
        )
    return np.array(errors)


def test_svp_degrades_away_from_its_calibration_depth():
    """The same fit gets steadily worse at other ranges -- an SVP model has one viewpoint."""
    port = FlatPort(0.0015, 0.010, N_GLASS, SWEET_WATER)
    errors = _svp_depth_errors(
        port, fit_svp_model(port, HALF_FOV, 2.0), (2.0, 1.0, 0.5, 0.25)
    )

    assert np.all(np.diff(errors) > 0)
    assert errors[0] < 1e-3


def test_svp_depth_error_is_driven_by_the_exit_plane_offset():
    """The depth dependence a pinhole cannot absorb is the ray's sideways shift in the port.

    In the limit of a thin port hard against the lens, a flat port is a pure
    *angular* map -- and radial distortion fits an angular map exactly, at every
    range. What breaks that is `r_exit = d0 tan(alpha) + d1 tan(beta)`, a lateral
    offset whose relative size grows as 1/Z. Both terms contribute, so a thick
    pane matters even when d0 is nearly zero, and shrinking d0 alone hits a floor
    set by the glass.

    The practical consequence for FishSense: with a tight housing, in-water SVP
    calibration is nearly as good as Pinax *on geometry alone*, so the case for
    Pinax rests on its practical advantages -- calibrating in air, and retuning
    for salinity without re-entering the water.
    """

    def worst_error(d0, d1):
        port = FlatPort(d0, d1, N_GLASS, SWEET_WATER)
        return _svp_depth_errors(port, fit_svp_model(port, HALF_FOV, 2.0), (0.5,))[0]

    tight_and_thin = worst_error(0.0005, 0.001)  # r_exit ~ 0.8 mm at 35 deg

    # A thin pane hard against the lens is almost perfectly fitted by an SVP model.
    assert tight_and_thin < 5e-4

    # Thickening the glass alone degrades it, even with the lens still against the pane.
    assert worst_error(0.0005, 0.020) > 10 * tight_and_thin

    # And a loose housing is worse by two orders of magnitude.
    assert worst_error(0.050, 0.010) > 20 * tight_and_thin


def test_pixel_round_trip():
    K = np.array([[2850.0, 0, 2007.0], [0, 2850.0, 1508.0], [0, 0, 1.0]])
    alpha = np.radians([5.0, 20.0, 35.0])
    azimuth = np.radians([0.0, 100.0, -140.0])
    a, z = pixel_to_alpha_azimuth(alpha_azimuth_to_pixel(alpha, azimuth, K), K)
    assert a == pytest.approx(alpha)
    assert np.cos(z) == pytest.approx(np.cos(azimuth))


def test_reconstruct_recovers_a_point_on_the_laser_line():
    laser_origin = np.array([-0.04, -0.11, 0.0])
    laser_axis = np.array([0.0, 0.0, 1.0])
    truth = laser_origin + 2.5 * laser_axis

    directions = truth / np.linalg.norm(truth)
    got, denom = reconstruct_points(directions, np.zeros(3), laser_origin, laser_axis)

    assert got == pytest.approx(truth)
    assert 0.0 < denom <= 1.0


def test_reconstruct_handles_an_offset_ray_origin():
    """Pinax puts the virtual centre of projection off the coordinate origin."""
    laser_origin = np.array([-0.04, -0.11, 0.0])
    laser_axis = np.array([0.0, 0.0, 1.0])
    origin = np.array([0.0, 0.0, 0.0006])
    truth = laser_origin + 3.0 * laser_axis

    directions = truth - origin
    got, _ = reconstruct_points(directions, origin, laser_origin, laser_axis)
    assert got == pytest.approx(truth)


def test_project_water_points_is_vectorised():
    port = FlatPort(0.0015, 0.010, N_GLASS, SWEET_WATER)
    points = np.random.default_rng(0).uniform(-0.3, 0.3, (4, 5, 3))
    points[..., 2] = np.random.default_rng(1).uniform(0.5, 5.0, (4, 5))

    alpha, azimuth = project_water_points(points, port)
    assert alpha.shape == (4, 5)
    assert azimuth.shape == (4, 5)

    # The recovered ray must actually reach each point.
    radius = water_radius(alpha, points[..., 2], port)
    assert radius == pytest.approx(np.hypot(points[..., 0], points[..., 1]), abs=1e-12)


def test_ray_directions_are_unit():
    d = ray_directions(np.radians([0.0, 15.0, 40.0]), np.radians([10.0, 200.0, -30.0]))
    assert np.linalg.norm(d, axis=-1) == pytest.approx(1.0)


def test_layered_port_reproduces_a_single_pane():
    """A LayeredPort spelled out as the two-layer case must equal FlatPort exactly."""
    flat = FlatPort(0.0015, 0.010, N_GLASS, SWEET_WATER)
    layered = LayeredPort(((0.0015, 1.0), (0.010, N_GLASS)), SWEET_WATER)

    alpha = np.linspace(0.001, HALF_FOV, 32)
    assert layered.d2 == pytest.approx(flat.d2)
    for a, b in zip(exit_ray(alpha, flat), exit_ray(alpha, layered)):
        assert a == pytest.approx(b)
    assert water_radius(alpha, 2.0, layered) == pytest.approx(water_radius(alpha, 2.0, flat))


def test_stack_cannot_change_the_final_water_direction():
    """However many panes, only the sideways offset accumulates -- gamma is fixed."""
    alpha = np.radians(30.0)
    expected = np.arcsin(np.sin(alpha) / SWEET_WATER)
    for layers in (
        ((0.001, 1.0), (0.003, 1.52)),
        ((0.001, 1.0), (0.003, 1.52), (0.008, 1.0), (0.006, 1.49)),
        ((0.02, 1.0), (0.01, 1.7), (0.01, 1.0), (0.01, 1.4), (0.005, 1.6)),
    ):
        _, gamma = exit_ray(alpha, LayeredPort(layers, SWEET_WATER))
        assert gamma == pytest.approx(expected)


def test_extra_layers_increase_the_exit_offset():
    """Adding the outer enclosure pushes the ray further off axis, which is what matters."""
    alpha = np.radians(30.0)
    bare = LayeredPort(((0.001, 1.0), (0.003, 1.52)), SWEET_WATER)
    enclosed = LayeredPort(
        ((0.001, 1.0), (0.003, 1.52), (0.008, 1.0), (0.006, 1.49)), SWEET_WATER
    )
    assert exit_ray(alpha, enclosed)[0] > exit_ray(alpha, bare)[0]


def test_layered_port_works_through_forward_projection():
    """The whole pipeline must accept a stack, not just a single pane."""
    port = LayeredPort(
        ((0.001, 1.0), (0.003, 1.52), (0.008, 1.0), (0.006, 1.49)), SALTY_WATER
    )
    alpha = np.linspace(0.001, HALF_FOV, 24)
    for z in (0.5, 3.0):
        assert field_angle(water_radius(alpha, z, port), z, port) == pytest.approx(
            alpha, abs=1e-9
        )
