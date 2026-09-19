"""Flat-port refraction geometry for FishSense underwater housings.

Self-contained implementation of the geometry behind the Pinax model:

> Łuczyński, Pfingsthorn & Birk, "The Pinax-model for accurate and efficient
> refraction correction of underwater cameras in flat-pane housings," Ocean
> Engineering 133 (2017) 9-22.

A flat pane makes an underwater camera physically *axial*: rays do not share one
viewpoint. Pinax exploits the fact that a real housing puts the lens very close
to the glass, so near an optimal camera-to-glass distance the axial camera
collapses to a virtual pinhole a short way down the optical axis.

Conventions
-----------
- Camera centre of projection at the origin, optical axis +z into the scene.
- The port is rotationally symmetric about the optical axis, so every ray can be
  described by a field angle `alpha` and an azimuth. All the geometry below is
  therefore scalar in `alpha`; no vector Snell is needed.
- Lens distortion is assumed already removed by the in-air calibration. Both the
  uncorrected and the Pinax pipelines undistort with the same in-air model, so
  it cancels and is not part of the effect under study.
- Every distance may be in any single unit, as long as it is used consistently.
  The notebooks use metres, matching the FishSense laser-reconstruction code.
  Nothing here has a unit-dependent tolerance.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar

# Refraction indices used throughout Łuczyński et al. (2017) for field work.
SWEET_WATER = 1.333
SALTY_WATER = 1.342

# Bisection halves [0, pi/2] sixty times -> ~1.4e-18 rad, past float64 on that
# interval. Bisecting the ANGLE rather than a distance keeps this unit-free.
_BISECT_ITERS = 60


@dataclass(frozen=True)
class FlatPort:
    """Geometry of a single-pane housing, measured along the optical axis."""

    d0: float  # centre of projection -> inner glass surface
    d1: float  # glass thickness
    n_glass: float
    n_water: float
    n_air: float = 1.0

    @property
    def layers(self) -> Tuple[Tuple[float, float], ...]:
        """``((thickness, index), ...)`` from the camera outward."""
        return ((self.d0, self.n_air), (self.d1, self.n_glass))

    @property
    def d2(self) -> float:
        """Depth of the outer glass surface."""
        return self.d0 + self.d1


@dataclass(frozen=True)
class LayeredPort:
    """An arbitrary stack of parallel flat interfaces between the camera and the water.

    FishSense puts an already-waterproof camera (with its own flat window) inside
    a second enclosure, so the real optical path is
    ``air gap -> camera window -> air gap -> housing pane -> water`` — four
    interfaces, not two. `FlatPort` is the two-interface special case.

    Because every surface is parallel, Snell's invariant ``n_i sin(theta_i) =
    sin(alpha)`` holds all the way through, so a stack behaves exactly like a
    single pane whose lateral offset is the sum of each layer's contribution. In
    particular the *final* direction in water is completely unaffected by the
    stack — only the sideways displacement accumulates.
    """

    layers: Tuple[Tuple[float, float], ...]  # ((thickness, index), ...) camera outward
    n_water: float
    n_air: float = 1.0

    @property
    def d2(self) -> float:
        """Depth of the outermost surface."""
        return float(sum(t for t, _ in self.layers))


def snell(alpha, n_from, n_to):
    """Refraction angle for an incident field angle `alpha`. NaN past the critical angle."""
    s = (n_from / n_to) * np.sin(alpha)
    return np.where(np.abs(s) <= 1.0, np.arcsin(np.clip(s, -1.0, 1.0)), np.nan)


def exit_ray(alpha, port) -> Tuple[np.ndarray, np.ndarray]:
    """Where a camera ray at field angle `alpha` leaves the port, and its angle in water.

    Returns ``(r_exit, gamma)``: the radial offset from the optical axis at the
    outermost surface, and the ray's angle to the axis once in water.

    Accepts a `FlatPort` or a `LayeredPort`. Note that `gamma` depends only on the
    water index -- the port shifts a ray sideways but cannot change its final
    direction, however many panes it has.
    """
    gamma = snell(alpha, port.n_air, port.n_water)
    r_exit = sum(
        thickness * np.tan(snell(alpha, port.n_air, index))
        for thickness, index in port.layers
    )
    return r_exit, gamma


@dataclass(frozen=True)
class DomePort:
    """A spherical dome, the alternative to a flat pane and the state of practice.

    A dome is the reason flat-port refraction is a choice rather than a fact. If
    the camera's entrance pupil sits exactly at the dome's centre of curvature,
    every camera ray leaves along a radius, meets both glass surfaces at normal
    incidence, and is not bent at all: the housing is optically absent and an
    in-air calibration is valid underwater with no correction. `decentre` is the
    distance by which the pupil misses that centre, and it is the whole story --
    the deviation scales with it and vanishes with it.

    That ideal is hard to buy. The pupil's position is a property of the lens and
    moves with zoom and focus, domes are sold in discrete sizes with discrete
    extension rings, and the alignment cannot be seen from outside the housing.
    A dome also forms a *virtual image* a short way in front of the port, so the
    lens must focus far closer than the subject, which is why dome users fit
    close-up dioptres and stop down.

    `radius` is the inner radius; `thickness` the glass. `decentre` is positive
    when the centre of curvature lies in front of the pupil.
    """

    radius: float
    thickness: float
    n_glass: float
    n_water: float
    decentre: float = 0.0
    n_air: float = 1.0


def _refract(direction, normal, eta):
    """Vector Snell. `normal` points along the outgoing side; `eta` = n_from / n_to."""
    cos_i = np.sum(direction * normal, axis=-1, keepdims=True)
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    if np.any(k < 0):
        raise ValueError("total internal reflection in the port")
    return eta * direction + (np.sqrt(k) - eta * cos_i) * normal


def dome_exit_ray(alpha, port: DomePort) -> Tuple[np.ndarray, np.ndarray]:
    """Where a camera ray at field angle `alpha` leaves a dome, and its angle in water.

    Returns ``(r_exit, gamma)`` to match `exit_ray`, so the two port types are
    interchangeable in a pipeline. Traced as vectors through both spherical
    surfaces rather than in closed form, because the closed form is only tidy in
    the concentric case -- which is exactly the case that needs no tracing.

    For ``decentre == 0`` this returns ``gamma == alpha`` identically: the dome is
    transparent, and that is the benchmark a flat port has to earn its place
    against.
    """
    alpha_in = np.asarray(alpha, dtype=float)
    alpha = np.atleast_1d(alpha_in)
    u = np.stack([np.sin(alpha), np.cos(alpha)], axis=-1)
    centre = np.array([0.0, float(port.decentre)])

    # first surface: the inner sphere about the centre of curvature
    u_dot_c = u @ centre
    disc = u_dot_c ** 2 - centre @ centre + port.radius ** 2
    if np.any(disc < 0):
        raise ValueError("ray misses the dome; check radius against decentre")
    p1 = (u_dot_c + np.sqrt(disc))[..., None] * u
    n1 = (p1 - centre) / port.radius
    v = _refract(u, n1, port.n_air / port.n_glass)

    # second surface: the outer sphere, same centre
    w = p1 - centre
    v_dot_w = np.sum(v * w, axis=-1)
    outer = port.radius + port.thickness
    s2 = -v_dot_w + np.sqrt(v_dot_w ** 2 + outer ** 2 - port.radius ** 2)
    p2 = p1 + s2[..., None] * v
    n2 = (p2 - centre) / outer
    e = _refract(v, n2, port.n_glass / port.n_water)

    gamma = np.arctan2(e[..., 0], e[..., 1])
    r_exit = p2[..., 0]
    if alpha_in.ndim == 0:  # match exit_ray: a scalar in, a scalar out
        return r_exit[0], gamma[0]
    return r_exit, gamma


def water_radius(alpha, z, port: FlatPort):
    """Radius from the optical axis at which the ray at `alpha` reaches depth `z`."""
    r_exit, gamma = exit_ray(alpha, port)
    return r_exit + (z - port.d2) * np.tan(gamma)


def field_angle(radius, z, port: FlatPort) -> np.ndarray:
    """Field angle of the camera ray that images a water point at ``(radius, z)``.

    This is the axial-camera forward projection -- the computationally hard
    direction, which Agrawal et al. reduce to a 12th-degree polynomial. Because
    `water_radius` is strictly increasing in `alpha`, bisection on ``[0, pi/2)``
    is unconditionally robust and needs no spurious-root filtering.

    Raises if any point is not beyond the outer glass surface, where the mapping
    is not defined.
    """
    radius = np.asarray(radius, dtype=float)
    z = np.asarray(z, dtype=float)

    if np.any(z <= port.d2):
        raise ValueError(
            f"all points must lie beyond the outer glass surface (z > {port.d2}); "
            f"got a minimum z of {np.min(z)}"
        )

    shape = np.broadcast_shapes(radius.shape, z.shape)
    lo = np.zeros(shape)
    hi = np.full(shape, 0.5 * np.pi - 1e-12)

    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        too_small = water_radius(mid, z, port) < radius
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)

    return 0.5 * (lo + hi)


def project_water_points(points, port: FlatPort) -> Tuple[np.ndarray, np.ndarray]:
    """Field angle and azimuth of the camera rays imaging 3D water points.

    `points` has shape ``(..., 3)`` in the camera frame.
    """
    points = np.asarray(points, dtype=float)
    radius = np.hypot(points[..., 0], points[..., 1])
    azimuth = np.arctan2(points[..., 1], points[..., 0])
    return field_angle(radius, points[..., 2], port), azimuth


def axis_crossing(alpha, port: FlatPort):
    """Depth at which a water ray, traced back into the housing, crosses the optical axis.

    For a true pinhole this would be the same depth for every `alpha`. The spread
    over the field of view is the "focus section", and its length is what the
    Pinax model minimises.
    """
    r_exit, gamma = exit_ray(alpha, port)
    return port.d2 - r_exit / np.tan(gamma)


def focus_section(port: FlatPort, half_fov, n_rays=256) -> Tuple[float, float]:
    """``(midpoint, length)`` of the focus section over a field of view.

    The midpoint is the virtual centre of projection: the point the corrected
    camera behaves as though it sees from.
    """
    alpha = np.linspace(half_fov / n_rays, half_fov, n_rays)
    z = axis_crossing(alpha, port)
    return float(0.5 * (z.min() + z.max())), float(z.max() - z.min())


def optimal_d0(d1, n_glass, n_water, half_fov, n_air=1.0, n_rays=256):
    """Camera-to-glass distance that best collapses the axial camera to a pinhole.

    Returns ``(d0_star, virtual_cop, focus_length)``, all in the unit of `d1`.

    The optimum depends on `half_fov` -- there is no field-of-view-free answer, and
    for a wide lens it moves by several times. Solved over the dimensionless ratio
    ``d0 / d1`` so the result does not depend on the length unit.
    """

    def cost(ratio):
        port = FlatPort(ratio * d1, d1, n_glass, n_water, n_air)
        return focus_section(port, half_fov, n_rays)[1] / d1

    res = minimize_scalar(
        cost, bounds=(1e-6, 1.0), method="bounded", options={"xatol": 1e-10}
    )

    d0_star = float(res.x * d1)
    virtual_cop, focus_length = focus_section(
        FlatPort(d0_star, d1, n_glass, n_water, n_air), half_fov, n_rays
    )
    return d0_star, virtual_cop, focus_length


def fit_svp_model(port: FlatPort, half_fov, calibration_depth, n_rays=256, n_radial=2):
    """Fit a single-viewpoint pinhole + radial distortion model to a flat port.

    This is what an in-water calibration actually produces: OpenCV-style
    intrinsics fitted to a target held at one distance. Fitting the radial terms
    as well as the focal length matters for a fair comparison -- real radial
    coefficients absorb most of the refracted mapping at a *fixed* depth, so a
    focal-length-only fit would be a strawman. What an SVP model cannot absorb is
    the depth dependence, and that is precisely the failure Pinax removes.

    Fits ``y = c0 u + c1 u^3 + c2 u^5 + ...`` where ``u`` is the normalised scene
    radius ``R / Z`` and ``y = tan(alpha)`` is the normalised image radius. ``c0``
    is the focal scale; its paraxial value is exactly `n_water`.
    """
    alpha = np.linspace(half_fov / n_rays, half_fov, n_rays)
    u = water_radius(alpha, calibration_depth, port) / calibration_depth
    basis = np.stack([u ** (2 * i + 1) for i in range(n_radial + 1)], axis=-1)
    coeffs, *_ = np.linalg.lstsq(basis, np.tan(alpha), rcond=None)
    return coeffs


def svp_image_radius(u, coeffs):
    """Normalised image radius predicted by a fitted SVP model."""
    u = np.asarray(u, dtype=float)
    return sum(c * u ** (2 * i + 1) for i, c in enumerate(coeffs))


def svp_scene_radius(image_radius, coeffs, u_max=8.0):
    """Invert a fitted SVP model: normalised image radius -> normalised scene radius.

    `svp_image_radius` is monotonically increasing over the fitted range, so
    bisection is unconditionally safe.
    """
    image_radius = np.asarray(image_radius, dtype=float)
    lo = np.zeros_like(image_radius)
    hi = np.full_like(image_radius, u_max)

    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        too_small = svp_image_radius(mid, coeffs) < image_radius
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)

    return 0.5 * (lo + hi)


def ray_directions(alpha, azimuth) -> np.ndarray:
    """Unit ray directions from a field angle and azimuth, shape ``(..., 3)``."""
    sin_a = np.sin(alpha)
    return np.stack(
        [
            sin_a * np.cos(azimuth),
            sin_a * np.sin(azimuth),
            np.cos(alpha) * np.ones_like(azimuth),
        ],
        axis=-1,
    )


def alpha_azimuth_to_pixel(alpha, azimuth, camera_intrinsics) -> np.ndarray:
    """Project a field angle and azimuth to pixel coordinates, shape ``(..., 2)``."""
    r = np.tan(alpha)
    return np.stack(
        [
            camera_intrinsics[0, 0] * r * np.cos(azimuth) + camera_intrinsics[0, 2],
            camera_intrinsics[1, 1] * r * np.sin(azimuth) + camera_intrinsics[1, 2],
        ],
        axis=-1,
    )


def pixel_to_alpha_azimuth(pixels, camera_intrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """Recover field angle and azimuth from pixel coordinates."""
    pixels = np.asarray(pixels, dtype=float)
    x = (pixels[..., 0] - camera_intrinsics[0, 2]) / camera_intrinsics[0, 0]
    y = (pixels[..., 1] - camera_intrinsics[1, 2]) / camera_intrinsics[1, 1]
    return np.arctan(np.hypot(x, y)), np.arctan2(y, x)


def reconstruct_points(directions, ray_origin, laser_origin, laser_axis):
    """Closest point on each camera ray to the laser line.

    Generalises the FishSense laser reconstruction to a camera-ray origin that is
    not the coordinate origin, which Pinax needs -- it places the virtual centre
    of projection a short way down the optical axis. With ``ray_origin = 0`` this
    reduces to the formulation in `fishsense_imwut.camera.reconstruct_points`.

    Parameters
    ----------
    directions : array_like, shape (..., 3)
        Camera ray directions (need not be unit).
    ray_origin : array_like, shape (3,) or (..., 3)
        One origin, or one per ray for a non-central model.
    laser_origin, laser_axis : array_like, shape (3,)

    Returns
    -------
    world_points : ndarray, shape (..., 3)
    denom : ndarray
        ``sin^2`` of the angle between the two lines -- the conditioning metric
        FishSense gates on. It goes to zero as the lines become parallel, which is
        what bounds the usable range.
    """
    u = np.asarray(directions, dtype=float)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)

    ray_origin = np.asarray(ray_origin, dtype=float)
    laser_origin = np.asarray(laser_origin, dtype=float)
    v = np.asarray(laser_axis, dtype=float)
    v = v / np.linalg.norm(v)

    w0 = ray_origin - laser_origin
    d = u @ v
    denom = 1.0 - d**2

    # Row-wise rather than matrix products, so `ray_origin` may be one point (a
    # central model) or one per ray (an axial one, where every ray leaves the
    # port at its own place and no common viewpoint exists).
    s = (d * (w0 @ v) - np.sum(u * w0, axis=-1)) / denom
    return ray_origin + s[..., None] * u, denom
