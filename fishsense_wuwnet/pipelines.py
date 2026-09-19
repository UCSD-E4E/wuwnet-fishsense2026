"""The three ways to back-project a FishSense pixel through a flat port.

All three take pixel coordinates and return ``(ray_origin, directions)``, ready
for `fishsense_wuwnet.refraction.reconstruct_points`. They differ only in what
they believe about the water:

``uncorrected``
    In-air calibration used as-is underwater. The "do nothing" baseline.

``svp``
    In-water calibration under a single-viewpoint pinhole model -- the state of
    practice the Pinax paper compares against. Refits one focal length at one
    calibration distance, which is exact near that distance and on the optical
    axis, and degrades away from both.

``pinax``
    Treats every water ray as leaving a single virtual centre of projection a
    short way down the optical axis, in its true refracted direction.

Only `pinax` is given the housing geometry, and it is given the housing it is
*told* about -- not necessarily the true one. That is deliberate: the gap between
the assumed and true port is what the residual-budget analysis measures.
"""

from typing import Tuple

import numpy as np

from fishsense_wuwnet.refraction import (
    DomePort,
    FlatPort,
    dome_exit_ray,
    exit_ray,
    pixel_to_alpha_azimuth,
    ray_directions,
    svp_scene_radius,
)


def back_project_uncorrected(
    pixels, camera_intrinsics
) -> Tuple[np.ndarray, np.ndarray]:
    """In-air intrinsics applied underwater with no refraction correction."""
    alpha, azimuth = pixel_to_alpha_azimuth(pixels, camera_intrinsics)
    return np.zeros(3), ray_directions(alpha, azimuth)


def back_project_svp(
    pixels, camera_intrinsics, svp_coeffs
) -> Tuple[np.ndarray, np.ndarray]:
    """In-water single-viewpoint calibration: refitted pinhole + radial distortion.

    Exact near the depth it was calibrated at, and progressively wrong away from
    it -- an SVP model has one viewpoint, so it cannot represent a mapping that
    depends on range.
    """
    alpha, azimuth = pixel_to_alpha_azimuth(pixels, camera_intrinsics)
    u = svp_scene_radius(np.tan(alpha), svp_coeffs)
    return np.zeros(3), ray_directions(np.arctan(u), azimuth)


def back_project_pinax(
    pixels, camera_intrinsics, port: FlatPort, virtual_cop
) -> Tuple[np.ndarray, np.ndarray]:
    """Pinax correction: true refracted direction, from the virtual centre of projection."""
    alpha, azimuth = pixel_to_alpha_azimuth(pixels, camera_intrinsics)
    _, gamma = exit_ray(alpha, port)
    return np.array([0.0, 0.0, virtual_cop]), ray_directions(gamma, azimuth)


def back_project_axial(
    pixels, camera_intrinsics, port: FlatPort
) -> Tuple[np.ndarray, np.ndarray]:
    """The exact axial model: every ray leaves the port at its own point.

    This is the flat-port geometry without approximation -- Agrawal et al.'s
    non-central camera, of which Pinax is the central approximation. Each ray
    exits the outer glass surface at radius `r_exit`, which grows with field
    angle, so there is no common centre of projection to place a virtual pinhole
    at. `back_project_pinax` discards exactly that offset.

    It is the more correct model and, on this housing at survey range, worth
    about 3% of the reprojection residual -- the offset reaches a few
    millimetres against ranges of metres. Included so the figures can show that
    the approximation is not what limits the result.
    """
    alpha, azimuth = pixel_to_alpha_azimuth(pixels, camera_intrinsics)
    r_exit, gamma = exit_ray(alpha, port)
    origins = np.stack(
        [r_exit * np.cos(azimuth), r_exit * np.sin(azimuth), np.full_like(r_exit, port.d2)],
        axis=-1,
    )
    return origins, ray_directions(gamma, azimuth)


def back_project_dome(
    pixels, camera_intrinsics, dome: DomePort
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project through a dome port, exactly.

    Included so the figures can compare against the state of practice rather
    than only against other flat-port treatments. A concentric dome needs no
    correction at all, so for ``decentre == 0`` this is the in-air pinhole; what
    it buys over `back_project_uncorrected` is the residual from a dome that was
    not assembled perfectly, which is the only error a dome has.
    """
    alpha, azimuth = pixel_to_alpha_azimuth(pixels, camera_intrinsics)
    r_exit, gamma = dome_exit_ray(alpha, dome)
    z_exit = np.sqrt(np.maximum((dome.radius + dome.thickness) ** 2 - r_exit ** 2, 0.0)) + dome.decentre
    origins = np.stack(
        [r_exit * np.cos(azimuth), r_exit * np.sin(azimuth), np.broadcast_to(z_exit, r_exit.shape)],
        axis=-1,
    )
    return origins, ray_directions(gamma, azimuth)


def measure_length(pixels_head, pixels_tail, depth, back_project) -> np.ndarray:
    """Length of a fish from its head and tail pixels, given a range estimate.

    This is the FishSense measurement itself: the laser gives the range, the two
    endpoints are back-projected to that depth, and the length is the distance
    between them. Errors enter twice -- through the range and through the
    back-projection geometry -- and off-axis it is the geometry that dominates.

    `back_project` is one of the ``back_project_*`` functions above, already bound
    to its calibration.
    """
    head_origin, head = back_project(pixels_head)
    tail_origin, tail = back_project(pixels_tail)

    # Each endpoint carries its own ray origin. For a central model the two are
    # the same point and this is the usual formula; for the axial model they are
    # not, and using one endpoint's origin for both would reintroduce exactly the
    # approximation the axial model exists to avoid.
    depth = np.asarray(depth, dtype=float)
    head_point = head_origin + head * (
        (depth - head_origin[..., 2]) / head[..., 2]
    )[..., None]
    tail_point = tail_origin + tail * (
        (depth - tail_origin[..., 2]) / tail[..., 2]
    )[..., None]

    return np.linalg.norm(head_point - tail_point, axis=-1)


# --------------------------------------------------------------------------
# Two different length errors, which are easy to conflate
#
# FishSense measures a fish by taking a *range* from its laser dot and a
# *transverse extent* from the camera. The dot lands near the optical axis; the
# fish can be anywhere in the frame. So the two halves of the measurement are
# read at different field positions, and any camera model whose angular
# magnification varies across the field scales them differently.
#
# That gives two distinct error terms:
#
# `local_length_error`
#     How much a fish-sized object's own extent is distorted, at one field
#     position. Small, because it is a second difference -- and a homography
#     absorbs the first-order scale entirely.
#
# `differential_length_error`
#     How much the magnification at the fish differs from the magnification at
#     the laser dot. This is first order and dominates: measured on the
#     2024-05-01 pool data it reaches ~22% at the frame corner for an
#     uncorrected flat port, against ~0.4% for the local term.
#
# Measuring only the local term makes an uncorrected port look harmless. It is
# not; the error is in the differential.
# --------------------------------------------------------------------------


def local_length_error(corners, directions, object_points, pairs):
    """Relative length error over a fish-sized baseline, from homography residuals.

    Fits the board's own plane to the corrected directions and asks how much each
    pair's *separation* is distorted relative to what the plane predicts. Because
    it is expressed as a ratio of separations, it is free of pose, tilt and range:

        relative error = |d_a - d_b| / |H(a) - H(b)|

    where ``d`` is the homography residual. Returns ``(relative_error,
    midpoint_radius_px)`` per pair.

    This is the *non-projective* part of the model error only. The homography
    absorbs any global scale, which is exactly the differential term below, so a
    small number here does not mean the measurement is safe.
    """
    import cv2

    homography, _ = cv2.findHomography(object_points[:, :2], directions, method=0)
    if homography is None:
        return None, None

    predicted = cv2.perspectiveTransform(
        object_points[:, :2].reshape(-1, 1, 2).astype(np.float64), homography
    ).reshape(-1, 2)
    residual = directions - predicted

    a, b = pairs[:, 0], pairs[:, 1]
    separation = np.linalg.norm(predicted[a] - predicted[b], axis=1)
    relative = np.linalg.norm(residual[a] - residual[b], axis=1) / np.maximum(
        separation, 1e-12
    )
    return relative, np.linalg.norm(0.5 * (corners[a] + corners[b]), axis=1)


def differential_length_error(
    model, reference, principal_point, radii, laser_radius, azimuth=0.0, extent="radial"
):
    """Relative length error from magnification differing across the field.

    `model` and `reference` map pixel coordinates to normalised directions.
    `reference` stands in for the truth -- on real data the in-water calibration is
    the best empirical description of what the camera actually does underwater, so
    it is the natural choice, with the caveat that it then scores zero by
    construction and cannot show its own error.

    Range is taken at `laser_radius`, where the dot lands, through the *position*
    magnification: what the model does to the dot's own direction is what moves the
    triangulated depth.

    Length is taken at each radius in `radii` through the magnification **along the
    fish's own extent**, which is not the same quantity and is why `extent` exists.
    A flat port's magnification is rotationally symmetric but not isotropic: a span
    lying along the radius is stretched by `d(tan gamma)/d(tan alpha)`, a span across
    it by `tan gamma / tan alpha`, and off axis those differ by about a factor of
    three. A horizontally-held fish meets the first case at the left and right of the
    frame and the second at the top and bottom, so one number per field radius
    describes neither. Pass ``extent="radial"`` for the worse case, ``"tangential"``
    for the better one.

    Returns relative length error, as a fraction.
    """
    if extent not in ("radial", "tangential"):
        raise ValueError(f"extent must be 'radial' or 'tangential', not {extent!r}")
    principal_point = np.asarray(principal_point, dtype=float)
    radii = np.asarray(radii, dtype=float)

    def mapped_radius(r, fn):
        pixels = principal_point + np.column_stack(
            [r * np.cos(azimuth), r * np.sin(azimuth)]
        )
        return np.linalg.norm(fn(pixels), axis=1)

    def position_mag(r):
        return mapped_radius(r, model) / np.maximum(mapped_radius(r, reference), 1e-12)

    def extent_mag(r):
        if extent == "tangential":
            # A span perpendicular to the radius scales as the radius itself does.
            return position_mag(r)
        # A span along the radius scales as the derivative of the mapping.
        # Central difference, with the step scaled to the radius: a fixed
        # absolute step is ill-conditioned near the axis and wasteful far from
        # it, and a one-sided difference leaves first-order error that shows up
        # as a spurious few-parts-in-1e10 offset where the answer should be
        # exactly zero.
        step = np.maximum(np.abs(r), 1.0) * 1e-4
        d_model = mapped_radius(r + step, model) - mapped_radius(r - step, model)
        d_reference = mapped_radius(r + step, reference) - mapped_radius(r - step, reference)
        return d_model / np.where(np.abs(d_reference) > 1e-15, d_reference, 1e-15)

    at_field = extent_mag(radii)
    at_laser = float(position_mag(np.array([laser_radius]))[0])
    return at_field / at_laser - 1.0
