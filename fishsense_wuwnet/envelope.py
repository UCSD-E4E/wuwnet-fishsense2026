"""The reachable-envelope computation behind section 4.2's headline numbers.

The figure usually quoted for an uncorrected flat port is a *free-field* bound:
the worst length error anywhere in the frame, for a target at any orientation
(+55.6 % radial at the frame corner for this sensor). That number is real and
it is not the one that matters, because a FishSense Lite measurement cannot
happen at the frame corner: the laser dot has to land *on* the fish, and the
beam runs nearly parallel to the optical axis, so a measurable fish sits near
the axis.

This module computes what the uncorrected model actually costs over the fish
the deployment photographed. Three things make it the *reachable* error rather
than the free-field one, and all three matter:

* **The dot is on the fish.** The fish is centred on the beam at its real range,
  not placed freely in the frame, so its field position is set by the laser
  offset and its own length -- not chosen to maximise the error.
* **The range is triangulated through the model under test**, exactly as the rig
  does it: the dot's range comes from ``reconstruct_points`` against the mount,
  through the *same* uncorrected back-projection used for the length. This is
  the cancellation that makes the number small -- the uncorrected range error
  and the uncorrected magnification error are both ~n_w and partly cancel in the
  product, the same mechanism that makes a focal-scale error cancel (section 5).
  Measuring the raw differential magnification instead -- without the
  triangulation -- gives roughly three times this, and answers a question the
  rig does not ask.
* **The geometry is real.** It runs over every (length, range) pair the
  production corpus actually contains, cached in ``data/corpus_length_range.npz``
  so this repo reproduces the number without the corpus itself.

Bearing is *not* observed in the corpus, so it is maximised over -- the worst
orientation at each pair, which is a strict upper bound. Dot placement along the
body *is* observed (0.40-0.65 of body length, median 0.52), so it is swept only
over that measured band rather than freely.
"""

from pathlib import Path

import numpy as np

from .pipelines import back_project_uncorrected, measure_length
from .refraction import (
    SWEET_WATER,
    FlatPort,
    alpha_azimuth_to_pixel,
    optimal_d0,
    project_water_points,
    reconstruct_points,
)

#: The pool/garage camera, its housing, and the deployed laser mount. `FOCAL`
#: and the principal point are the fleet-nominal in-air intrinsics; the mount
#: offset is the bench extrinsic, mostly vertical, which is *why* a horizontally
#: held fish extends tangentially -- the better case (section 4.2). Changing the
#: offset direction, not just modelling the port, is what changes the cost.
FOCAL = 2850.0
PRINCIPAL = (2007.0, 1508.0)
GLASS_THICKNESS_M = 0.006
GLASS_INDEX = 1.49
HALF_FOV = np.radians(41.37)
LASER_ORIGIN = np.array([-0.0336, -0.0983, 0.0])
LASER_AXIS = np.array([0.0, 0.0, 1.0])

#: Measured dot placement along the body, as a signed offset from the centre:
#: 0.40-0.65 of body length is s in [-0.10, +0.15]. Bearing is unobserved.
DOT_ALONG_BODY = np.linspace(-0.10, 0.15, 4)
BEARINGS = np.radians(np.arange(0.0, 180.0, 20.0))

CORPUS_CACHE = Path(__file__).resolve().parent.parent / "data" / "corpus_length_range.npz"


def _camera_and_port(n_water=SWEET_WATER):
    intrinsics = np.array([[FOCAL, 0, PRINCIPAL[0]], [0, FOCAL, PRINCIPAL[1]], [0, 0, 1.0]])
    d0, virtual_cop, _ = optimal_d0(GLASS_THICKNESS_M, GLASS_INDEX, n_water, HALF_FOV)
    port = FlatPort(d0, GLASS_THICKNESS_M, GLASS_INDEX, n_water)
    return intrinsics, port


def _project(points, port, intrinsics):
    alpha, azimuth = project_water_points(np.atleast_2d(points), port)
    return alpha_azimuth_to_pixel(alpha, azimuth, intrinsics)


def uncorrected_length_error(length, range_m, bearing, along, n_water=SWEET_WATER):
    """Percent length error of the uncorrected model for one placed fish, or None off-frame.

    `bearing` is the fish's orientation in the frame; `along` is where the dot
    sits on the body, signed from the centre. Returns ``None`` when either tip
    leaves the sensor -- a fish whose end cannot be seen is not measurable, so it
    is not part of the reachable set.
    """
    intrinsics, port = _camera_and_port(n_water)
    principal = np.array(PRINCIPAL)
    half_diagonal = np.hypot(*principal)

    dot = LASER_ORIGIN + range_m * LASER_AXIS
    direction = np.array([np.cos(bearing), np.sin(bearing), 0.0])
    head = dot + (0.5 - along) * length * direction
    tail = dot - (0.5 + along) * length * direction

    head_px, tail_px = _project(head, port, intrinsics)[0], _project(tail, port, intrinsics)[0]
    if max(np.hypot(*(head_px - principal)), np.hypot(*(tail_px - principal))) > half_diagonal:
        return None

    origin, ray = back_project_uncorrected(_project(dot, port, intrinsics), intrinsics)
    recovered, _ = reconstruct_points(ray, origin, LASER_ORIGIN, LASER_AXIS)
    measured = measure_length(
        _project(head, port, intrinsics),
        _project(tail, port, intrinsics),
        recovered[0, 2],
        lambda px: back_project_uncorrected(px, intrinsics),
    )
    return 100.0 * (float(measured[0]) - length) / length


def worst_over_bearing(length, range_m, n_water=SWEET_WATER):
    """Largest absolute reachable error over bearing and measured dot placement."""
    candidates = [
        uncorrected_length_error(length, range_m, bearing, along, n_water)
        for along in DOT_ALONG_BODY
        for bearing in BEARINGS
    ]
    candidates = [abs(e) for e in candidates if e is not None]
    return max(candidates) if candidates else None


def load_corpus_pairs(path=CORPUS_CACHE):
    """The production corpus's (length_m, range_m) pairs, cached from the sibling repo."""
    data = np.load(path)
    return np.column_stack([data["length_m"], data["range_m"]]).astype(float)


def reachable_envelope(pairs=None, n_water=SWEET_WATER, round_range_m=0.05):
    """Frame-weighted percentiles of the worst reachable uncorrected error.

    Pairs are binned by (length, range) so a geometry photographed many times is
    computed once and weighted by its count -- the envelope describes the frames
    the deployment took, not the distinct geometries. Returns a dict of
    percentiles plus the worst case and the tail fractions.
    """
    if pairs is None:
        pairs = load_corpus_pairs()

    from collections import Counter

    bins = Counter(
        (round(float(length), 3), round(float(range_m) / round_range_m) * round_range_m)
        for length, range_m in pairs
    )
    values, weights = [], []
    for (length, range_m), count in bins.items():
        worst = worst_over_bearing(length, range_m, n_water)
        if worst is not None:
            values.append(worst)
            weights.append(count)

    values = np.array(values)
    weights = np.array(weights, dtype=float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()

    def percentile(q):
        return float(values[np.searchsorted(cumulative, q / 100.0)])

    return {
        "n_frames": int(weights.sum()),
        "n_geometries": len(values),
        "p50": percentile(50),
        "p75": percentile(75),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": float(values.max()),
        "over_5pct": float(weights[values > 5].sum() / weights.sum()),
        "over_15pct": float(weights[values > 15].sum() / weights.sum()),
    }


if __name__ == "__main__":
    result = reachable_envelope()
    print(f"{result['n_frames']} frames over {result['n_geometries']} distinct geometries\n")
    print("uncorrected length error, worst bearing at each real (length, range):")
    for key in ("p50", "p75", "p90", "p95", "p99", "max"):
        print(f"   {key:>4}  {result[key]:5.1f} %")
    print(f"\n   frames over 15 %: {100 * result['over_15pct']:.1f} %"
          f"      over 5 %: {100 * result['over_5pct']:.1f} %")
