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


# --- the board's measured geometry ----------------------------------------

def test_board_object_points_use_the_measured_pitch():
    """The board's pitch is calipered, not nominal, and the two axes differ slightly.

    Corner-to-corner spans: 549 mm over the 13 pitches of the long axis, 379 mm over
    the 9 of the short. Guarded because the fx/fy discussion turns on these numbers
    and a well-meaning collapse back to one scalar would be silent -- see
    `SQUARE_PITCH_M` for why a scalar cannot express an anisotropic target.
    """
    from fishsense_wuwnet.dataset import (
        NOMINAL_SQUARE_SIZE_M,
        SQUARE_PITCH_M,
        board_object_points,
    )

    long_pitch, short_pitch = SQUARE_PITCH_M
    assert short_pitch / long_pitch == pytest.approx(0.9972, abs=1e-4)
    assert long_pitch > NOMINAL_SQUARE_SIZE_M and short_pitch > NOMINAL_SQUARE_SIZE_M

    objp = board_object_points()
    assert objp[:, 0].max() == pytest.approx(0.549, abs=1e-9)
    assert objp[:, 1].max() == pytest.approx(0.379, abs=1e-9)

    isotropic = board_object_points(square=NOMINAL_SQUARE_SIZE_M)
    assert isotropic[:, 0].max() == pytest.approx(0.546, abs=1e-9)


# --- the dome port, which is the state of practice -------------------------

def test_a_concentric_dome_is_optically_absent():
    """The defining property: pupil at the centre of curvature bends nothing.

    Every ray leaves along a radius and meets both surfaces at normal incidence,
    so gamma == alpha exactly and an in-air calibration needs no correction at
    all. This is the benchmark the flat port has to justify itself against, so
    it is asserted to machine precision rather than to a tolerance.
    """
    from fishsense_wuwnet.refraction import DomePort, dome_exit_ray

    dome = DomePort(radius=0.050, thickness=0.005, n_glass=1.49, n_water=SWEET_WATER)
    alpha = np.radians([0.0, 5.0, 15.0, 30.0, 41.4])
    r_exit, gamma = dome_exit_ray(alpha, dome)

    assert gamma == pytest.approx(alpha, abs=1e-12)
    # and it leaves at the outer radius, along that same radius
    assert r_exit == pytest.approx((0.050 + 0.005) * np.sin(alpha), abs=1e-12)


def test_dome_deviation_grows_with_decentring_and_field_angle():
    from fishsense_wuwnet.refraction import DomePort, dome_exit_ray

    alpha = np.radians(30.0)
    previous = 0.0
    for decentre in (0.002, 0.005, 0.010):
        dome = DomePort(0.050, 0.005, 1.49, SWEET_WATER, decentre=decentre)
        _, gamma = dome_exit_ray(alpha, dome)
        deviation = abs(gamma - alpha)
        assert deviation > previous
        previous = deviation
    # on axis there is nothing to bend, whatever the decentring
    _, gamma_axis = dome_exit_ray(0.0, DomePort(0.050, 0.005, 1.49, SWEET_WATER, decentre=0.010))
    assert gamma_axis == pytest.approx(0.0, abs=1e-12)


def test_a_misaligned_dome_still_beats_an_uncorrected_flat_port():
    """Why a dome is the state of practice: even misaligned it is far better.

    At 30 degrees off axis an uncorrected flat port is 7.97 degrees wrong. A dome
    decentred by 5 mm -- a poor alignment, a tenth of its own radius -- is 0.74
    degrees wrong, and one decentred by 10 mm is 1.48. The flat port's error does
    not depend on assembly at all; the dome's is entirely assembly, and goes to
    zero with it.
    """
    from fishsense_wuwnet.refraction import DomePort, dome_exit_ray

    alpha = np.radians(30.0)
    _, gamma_flat = exit_ray(alpha, FlatPort(0.001, 0.006, 1.49, SWEET_WATER))
    flat_error = abs(gamma_flat - alpha)

    _, gamma_5mm = dome_exit_ray(alpha, DomePort(0.050, 0.005, 1.49, SWEET_WATER, decentre=0.005))
    assert abs(gamma_5mm - alpha) < 0.1 * flat_error

    _, gamma_10mm = dome_exit_ray(alpha, DomePort(0.050, 0.005, 1.49, SWEET_WATER, decentre=0.010))
    assert abs(gamma_10mm - alpha) < 0.2 * flat_error


def test_dome_forward_projection_inverts_the_ray_trace():
    """`dome_field_angle` must undo `dome_water_radius`, and be exact when concentric."""
    from fishsense_wuwnet.refraction import DomePort, dome_field_angle, dome_water_radius

    dome = DomePort(0.050, 0.005, 1.49, SWEET_WATER, decentre=0.008)
    alpha = np.radians([2.0, 10.0, 25.0, 40.0])
    for z in (0.8, 2.0, 4.0):
        radius = dome_water_radius(alpha, z, dome)
        assert dome_field_angle(radius, z, dome) == pytest.approx(alpha, abs=1e-9)

    # a concentric dome is a pinhole: the field angle is just arctan(r / z)
    concentric = DomePort(0.050, 0.005, 1.49, SWEET_WATER)
    radius = np.array([0.1, 0.4, 1.0])
    assert dome_field_angle(radius, 2.0, concentric) == pytest.approx(
        np.arctan(radius / 2.0), abs=1e-9
    )


# --- the LEGO calibration target -------------------------------------------

def test_the_calibration_target_geometry_is_exact_and_metric():
    """The target's dimensions come from the design file, not from a ruler.

    That is the point of using it: a printed board's pitch has to be measured
    before it can be trusted, and ours could not be pinned down well enough to
    settle its own anisotropy. Every dimension below is a whole number of LEGO
    units, which is what makes it checkable at all.
    """
    from fishsense_wuwnet.calibration_model import (
        BRICK_MM,
        STUD_MM,
        extent_mm,
        load_bricks,
        model_points,
    )

    bricks = load_bricks()
    assert len(bricks) == 135
    assert {b.part for b in bricks} == {"3001.dat"}  # all 2x4 bricks

    # Nine courses of brick, four walls: 144 x 128 x 86.4 mm, every dimension a
    # whole number of studs or bricks.
    # 18 studs by 16 studs by 9 bricks. The tolerance is 1 micron, not machine
    # epsilon: Studio writes rotations to finite precision, which leaves tens of
    # nanometres in the extent. Every brick does sit exactly on the LEGO grid.
    extent = extent_mm()
    assert extent == pytest.approx([18 * STUD_MM, 16 * STUD_MM, 9 * BRICK_MM], abs=1e-3)

    points = model_points()
    assert points.shape == (8 * 135, 3)

    # Four marker colours, one per wall, so a detected marker names the face.
    markers = [b for b in bricks if b.is_marker]
    assert len(markers) == 18
    assert len({b.colour for b in markers}) == 4


def test_similarity_fit_separates_shape_error_from_scale():
    """A reconstruction is recovered up to a similarity, so the fit removes one.

    What survives is shape error, which is the model-independent quantity; the
    fitted scale is reported separately because comparing it against the scale
    the laser supplies is what tests the metric chain.
    """
    from fishsense_wuwnet.calibration_model import fit_similarity, model_points

    model = model_points()
    rng = np.random.default_rng(0)
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    observed = 2.5 * model @ rotation.T + np.array([100.0, -20.0, 7.0])

    scale, _, _, rms = fit_similarity(model, observed)
    assert scale == pytest.approx(2.5, rel=1e-9)
    assert rms == pytest.approx(0.0, abs=1e-6)

    # A reconstruction that is wrong in shape cannot be fitted away by a scale:
    # squashing one axis leaves a residual no similarity can absorb.
    squashed = observed * np.array([1.0, 1.0, 0.97])
    _, _, _, rms_squashed = fit_similarity(model, squashed)
    assert rms_squashed > 1.0
