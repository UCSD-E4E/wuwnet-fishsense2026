"""Per-dive laser calibration from the dots alone, plus one or two ranges.

FishSense's laser extrinsics are not stable between dives, so the laser has to
be recalibrated every dive. Today that means a slate: a known planar target,
detected and posed in several frames, so each dot can be placed in 3D and a
line fitted through them. This module replaces the slate with the dots the dive
already produces.

The geometry
------------
Write the beam as ``P(t) = O + t * D`` in the camera frame with ``O = (Ox, Oy, 0)``
and ``D = (Dx, Dy, 1)``. Its image in normalised (direction) coordinates is

    u(t) = D + O / t

which is a **straight line** through ``D`` in the direction of ``O``. So the
dots, undistorted and refraction-corrected, trace a line whose orientation is
the direction of ``O`` and whose position fixes the component of ``D`` across
it. That is two of the four degrees of freedom, and it needs no board, no
range, and no scale -- just the dots. Measured on real data it recovers them to
0.02-0.04 deg from about ten dots.

What the line cannot fix is where along it ``D`` sits (the ``t -> inf``
asymptote) and the magnitude ``|O|``: together they set the parameterisation
``s = |O| / t`` of position along the line, and without at least one *ratio* of
ranges the whole thing is scale-free. Hence:

- **one** range, with ``|O|`` measured once, closes it (``close_with_one_range``);
- **two** ranges at different distances close it with no prior at all
  (``close_with_two_ranges``), since ``p_i = |O| / Z_i + p_D`` is then two linear
  equations in two unknowns;
- **no** range at all, if the dot sits on a rigid object of *unknown* size seen
  in more than one frame (``close_with_apparent_size``): its apparent size
  ``l_i`` is proportional to ``1 / Z_i``, so ``p_i = p_D + (|O| / k) * l_i`` is a
  straight line whose intercept is the vanishing point. The fish the dot is on
  is that object. ``|O|`` measured once then sets the absolute scale.

A 5% range is plenty: at 2 m it moves the beam by ``|O| / Z^2 * dZ`` ~ 0.15 deg,
against the ~2 deg of drift being corrected. The same arithmetic gives the
tolerance the calibration has to meet at all (``vanishing_point_budget``): for a
15% worst-case length error at 4 m it is ~12 px, and the routes above land at
0.3-2 px on real data.

None of this is specific to a flat port -- the locus is pinhole geometry once
the dots are refraction-corrected -- and the flat port's own axial scale signal
sits under the model-error floor for this housing; see the notebook.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Locus:
    """The dot locus in normalised coordinates: a line ``centre + p * direction``."""

    centre: np.ndarray  # (2,) a point on the line
    direction: np.ndarray  # (2,) unit vector along it -- the direction of O
    rms: float  # perpendicular scatter of the dots about the line

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]])

    def position_along(self, u) -> np.ndarray:
        """Signed coordinate ``p`` of point(s) ``u`` along the line."""
        return (np.asarray(u, dtype=float) - self.centre) @ self.direction


def fit_locus(directions, orient_toward=None) -> Locus:
    """Total-least-squares line through undistorted dot directions.

    `directions` is ``(N, 2)`` normalised coordinates -- pixels already put through
    the in-air calibration and the refraction correction. The line's direction is
    ambiguous in sign; pass `orient_toward` (any rough guess at the direction of
    ``O``, e.g. from the previous dive) to resolve it, otherwise the first
    component is made non-negative.
    """
    u = np.asarray(directions, dtype=float)
    centre = u.mean(axis=0)
    direction = np.linalg.svd(u - centre)[2][0]

    if orient_toward is not None:
        if direction @ np.asarray(orient_toward, dtype=float) < 0:
            direction = -direction
    elif direction[0] < 0:
        direction = -direction

    normal = np.array([-direction[1], direction[0]])
    rms = float(np.sqrt(np.mean(((u - centre) @ normal) ** 2)))
    return Locus(centre, direction, rms)


def _beam(locus: Locus, o_mag: float, p_d: float) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble ``(O, D)`` from the locus, a baseline magnitude and D's position on the line."""
    origin = np.append(o_mag * locus.direction, 0.0)
    d_xy = locus.centre + p_d * locus.direction
    return origin, np.append(d_xy, 1.0)


def close_with_one_range(
    locus: Locus, direction_at_range, range_m: float, o_mag: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Full beam from the locus, ``|O|`` (measured once) and one known range.

    `direction_at_range` is the undistorted direction of the dot in the frame
    whose range is known. The dot sits at ``p = p_D + |O| / Z`` along the line,
    which pins ``p_D``. Returns ``(O, D)`` with ``O = (Ox, Oy, 0)``,
    ``D = (Dx, Dy, 1)``.
    """
    p_i = float(locus.position_along(direction_at_range))
    p_d = p_i - o_mag / float(range_m)
    return _beam(locus, o_mag, p_d)


def close_with_two_ranges(
    locus: Locus, direction_a, range_a: float, direction_b, range_b: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Full beam from the locus and two known ranges -- no prior on ``|O|``.

    Solves ``p_i = |O| / Z_i + p_D`` for both unknowns. The pair should be well
    separated in range: the system is ill-conditioned when ``1/Z_a ~ 1/Z_b``, and
    on real data pairs more than 1 m apart recover ``|O|`` to ~0.6% while close
    pairs can be off by tens of percent.
    """
    p_a = float(locus.position_along(direction_a))
    p_b = float(locus.position_along(direction_b))
    system = np.array([[1.0 / float(range_a), 1.0], [1.0 / float(range_b), 1.0]])
    o_mag, p_d = np.linalg.solve(system, np.array([p_a, p_b]))
    if o_mag <= 0:
        raise ValueError(
            f"recovered |O| = {o_mag:.4f} is not positive; the two ranges are "
            "probably too close together, or a direction is on the wrong branch"
        )
    return _beam(locus, float(o_mag), float(p_d))


def close_with_apparent_size(
    locus: Locus, directions, apparent_sizes, o_mag: float
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Full beam from the locus, ``|O|`` and the apparent size of whatever the dot is on.

    `directions` are the undistorted dot directions in frames where the dot sits
    on the *same* rigid object, and `apparent_sizes` its size in those frames in
    any consistent unit (pixels, normalised coordinates -- the unit cancels).
    Since ``l_i = k / Z_i``, the dot position along the line is linear in size,
    ``p_i = p_D + (|O| / k) * l_i``, and the intercept where the object would
    shrink to nothing is the vanishing point. No range is needed; the frames only
    have to span some range so the line has leverage.

    Returns ``(O, D, object_size, rms)``: the beam, the object's true size in the
    units of ``o_mag`` per unit of `apparent_sizes` (a by-product), and the rms of
    the fit in the units of `directions`. On the pool data this recovers ``D`` to
    0.3 px / 0.009 deg from ten frames of the board treated as an unknown object.
    """
    p = locus.position_along(directions)
    l = np.asarray(apparent_sizes, dtype=float)
    if l.ndim != 1 or len(l) != len(p) or len(l) < 2:
        raise ValueError("need one apparent size per direction, and at least two frames")
    if np.ptp(l) <= 0:
        raise ValueError("apparent sizes are all equal: no range spread, the intercept is undefined")
    design = np.column_stack([np.ones_like(l), l])
    (p_d, slope), *_ = np.linalg.lstsq(design, p, rcond=None)
    if slope <= 0:
        raise ValueError(
            f"fitted slope {slope:.3g} is not positive: dots must move toward O as the "
            "object grows, so either the locus is mis-oriented or the object is not rigid"
        )
    rms = float(np.sqrt(np.mean((design @ [p_d, slope] - p) ** 2)))
    origin, axis = _beam(locus, o_mag, float(p_d))
    return origin, axis, float(o_mag / slope), rms


def vanishing_point_budget(length_tolerance: float, o_mag: float, range_m: float) -> float:
    """How well ``D`` must be placed along the locus for a given length tolerance.

    Range is ``Z = |O| / (p - p_D)``, so ``dZ/Z = dp_D * Z / |O|``; for a laser
    near the optical axis the camera scale cancels in length and this is the
    whole length error. Returns the allowed ``p_D`` error in normalised units
    (multiply by the focal length in pixels). Worst case is the farthest range.
    """
    return float(length_tolerance) * float(o_mag) / float(range_m)


def range_along_beam(direction, origin, axis) -> float:
    """Range to the laser spot for a dot seen along `direction`.

    Closest approach of the camera ray to the beam, returning its depth. The
    same computation FishSense uses once the beam is known.
    """
    ray = np.append(np.asarray(direction, dtype=float)[:2], 1.0)
    ray = ray / np.linalg.norm(ray)
    unit_axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)

    w0 = -np.asarray(origin, dtype=float)
    b = float(ray @ unit_axis)
    denom = 1.0 - b * b
    if abs(denom) < 1e-12:
        return float("nan")
    s = (b * (unit_axis @ w0) - (ray @ w0)) / denom
    return float((s * ray)[2])


def beam_angle(axis_a, axis_b) -> float:
    """Angle in degrees between two beam axes."""
    a = np.asarray(axis_a, dtype=float) / np.linalg.norm(axis_a)
    b = np.asarray(axis_b, dtype=float) / np.linalg.norm(axis_b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))
