"""Figures for sections 4.2 onward, where the simulation is the result.

Two kinds live here and they are deliberately not interchangeable.

**Finished.** Where the simulation *is* the result -- how a calibration error
propagates to a length, how the port's error depends on field angle -- there is
nothing a photograph would add, and those figures are final.

**Placeholder.** Anything that claims the detector works on a photograph is
standing in for a measurement not yet taken, and is marked by
`figstyle.placeholder` so it cannot be mistaken for the other kind in a draft
somebody else reads.

Run: ``uv run python refraction_analysis/make_target_figures.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Committed caches; nothing here opens a photograph.
DATA = Path(__file__).resolve().parent.parent / "data"

from fishsense_wuwnet import figstyle
from fishsense_wuwnet.dataset import (
    PATTERN,
    SQUARE_PITCH_M,
    board_object_points,
    cache_by_session,
    fit_line,
    load_corner_cache,
)
from fishsense_wuwnet.laser import fit_beam_to_ranges, range_along_beam
from fishsense_wuwnet.pipelines import (
    back_project_pinax,
    back_project_uncorrected,
    differential_length_error,
    measure_length,
)
from fishsense_wuwnet.refraction import (
    SWEET_WATER,
    FlatPort,
    alpha_azimuth_to_pixel,
    exit_ray,
    optimal_d0,
    project_water_points,
    reconstruct_points,
)
from fishsense_wuwnet.target_detect import detect_quads, match
from fishsense_wuwnet.target_render import look_at, render

CAMERA = np.array([[2876.0, 0.0, 2000.0], [0.0, 2845.0, 1500.0], [0.0, 0.0, 1.0]])
_D0, _VCOP, _ = optimal_d0(0.006, 1.49, SWEET_WATER, np.radians(41.0))
PORT = FlatPort(_D0, 0.006, 1.49, SWEET_WATER)
#: Bench transverse offset, 104 mm, in the near-vertical direction the mount
#: actually uses -- section 4.2 turns on that direction, because a horizontally
#: held fish then extends tangentially rather than radially. The magnitude was
#: previously a simulation constant of 117 mm carried over from the tests, which
#: put the envelope annotation at 15.0 deg for the wrong reason.
BEAM_ORIGIN = 0.104 * np.array([-0.04, -0.11, 0.0]) / np.hypot(0.04, 0.11)
BEAM_AXIS = np.array([0.0, 0.0, 1.0])
LENGTH = 0.30

#: Where the dot lands along the body, measured over the production corpus:
#: inside the span in 100 % of frames, at 0.40-0.65 of body length from the
#: tail, median 0.52. The median is what a typical measurement looks like; the
#: 0.40 end is what bounds the envelope, because it puts the far tip furthest
#: from the axis. Quoting a centred dot understates that bound by 1.6 deg at
#: 1 m, and section 3 would contradict it.
DOT_FRACTION = 0.52
DOT_FRACTION_WORST = 0.40
CALIBRATION_RANGES = np.array([0.8, 1.2, 1.8, 2.5, 3.5, 4.5])
RANGES = np.array([1.0, 1.5, 2.0, 3.0, 4.0])


# --------------------------------------------------------------------------
# Finished: how a calibration error reaches the length
# --------------------------------------------------------------------------
def _pixels(points):
    alpha, azimuth = project_water_points(points, PORT)
    return alpha_azimuth_to_pixel(alpha, azimuth, CAMERA)


def _length_error(used, refit):
    back = lambda px: back_project_pinax(px, used, PORT, _VCOP)
    origin, axis = BEAM_ORIGIN, BEAM_AXIS
    if refit:
        rays = back(_pixels(BEAM_ORIGIN + CALIBRATION_RANGES[:, None] * BEAM_AXIS))
        origin, axis = fit_beam_to_ranges(rays[1], rays[0], CALIBRATION_RANGES)
    centre = BEAM_ORIGIN + RANGES[:, None] * BEAM_AXIS
    across = np.array([1.0, 0.0, 0.0])
    head = centre + across * (LENGTH * (1.0 - DOT_FRACTION))
    tail = centre - across * (LENGTH * DOT_FRACTION)
    ray_origin, direction = back(_pixels(centre))
    recovered, _ = reconstruct_points(direction, ray_origin, origin, axis)
    length = measure_length(_pixels(head), _pixels(tail), recovered[:, 2], back)
    return 100.0 * np.abs(length / LENGTH - 1.0).max()


def error_budget():
    focal = np.linspace(0.0, 5.0, 11)
    centre = np.linspace(0.0, 20.0, 11)
    curves = {}
    for refit in (False, True):
        curves["focal", refit] = [
            _length_error(_perturb(fx=1 + f / 100, fy=1 + f / 100), refit) for f in focal
        ]
        curves["centre", refit] = [
            _length_error(_perturb(dcx=c, dcy=c), refit) for c in centre
        ]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    labels = {False: "laser trusted from the bench", True: "laser re-fitted each dive"}
    for refit, colour, dash in ((False, figstyle.SERIES[0], "-"), (True, figstyle.SERIES[1], "--")):
        axes[0].plot(focal, curves["focal", refit], dash, color=colour, label=labels[refit])
        axes[1].plot(centre, curves["centre", refit], dash, color=colour, label=labels[refit])
    axes[0].set_xlabel("focal-length error (%)")
    axes[1].set_xlabel("principal-point error (px)")
    axes[0].set_title("Focal length", fontsize=9)
    axes[1].set_title("Principal point", fontsize=9)
    for ax in axes:
        ax.set_ylabel("worst length error (%)")
        ax.axhline(15.0, color="#8a8a85", lw=0.8, ls=":")
        ax.legend(fontsize=7.2, frameon=False)
        # Shared, because the two panels are meant to be compared by eye and
        # independent autoscaling would make the flat curve in one look like
        # the steep curve in the other.
        ax.set_ylim(-0.6, 20.0)
    fig.suptitle("Which intrinsic binds depends on how the laser is calibrated")
    fig.tight_layout(rect=(0, 0, 1, 1))
    return fig, "sim-calibration-error-budget-by-laser-regime"


def _dot_air_angle(range_m=2.0):
    """Where the dot sits, as an air-side angle -- the reference the error is measured from.

    Derived rather than written as a constant: the dot is at
    arctan(|O|/Z) in the water, which at 2 m is 2.98 degrees, and the air-side
    angle the pixel radius encodes is 3.97.
    """
    scene = np.arctan(np.linalg.norm(BEAM_ORIGIN) / range_m)
    return float(np.degrees(np.arcsin(np.clip(SWEET_WATER * np.sin(scene), -1.0, 1.0))))


def _perturb(fx=1.0, fy=1.0, dcx=0.0, dcy=0.0):
    used = CAMERA.copy()
    used[0, 0] *= fx
    used[1, 1] *= fy
    used[0, 2] += dcx
    used[1, 2] += dcy
    return used


# --------------------------------------------------------------------------
# Finished: the port's error is a function of field angle, not of the lens
# --------------------------------------------------------------------------
def _to_scene(air_deg, n_water=SWEET_WATER):
    """Air-side (image) field angle to scene-side (in-water) direction, degrees.

    The two sides of the port are not the same angle and the difference is
    large: 15.9 degrees in the water is 21.4 in the air. Everything this paper
    derives from the rig's geometry -- the laser offset, the fish's extent, the
    ranges -- lives in the water, so the scene side is the one a reader can
    check against the prose, and the one this figure is drawn in.
    """
    return np.degrees(np.arcsin(np.clip(np.sin(np.radians(air_deg)) / n_water, -1.0, 1.0)))


def field_angle_collapse():
    # Computed against the *air-side* angle, because that is what a pixel
    # radius is: r = f tan(alpha). Plotted against the scene side.
    angles = np.linspace(0.5, 35.0, 60)
    flatten = lambda d: d[..., :2] / d[..., 2:3]
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    for i, focal in enumerate((1329.0, 1751.0, 2876.0)):
        K = np.array([[focal, 0, 2000.0], [0, focal, 1500.0], [0, 0, 1.0]])
        unc = lambda px: flatten(back_project_uncorrected(px, K)[1])
        pin = lambda px: flatten(back_project_pinax(px, K, PORT, _VCOP)[1])
        error = 100 * differential_length_error(
            unc, pin, principal_point=K[:2, 2],
            radii=focal * np.tan(np.radians(angles)),
            laser_radius=focal * np.tan(np.radians(_dot_air_angle())), extent="radial")
        ax.plot(_to_scene(angles), error, color=figstyle.SERIES[i], ls=figstyle.DASHES[i], lw=2.0,
                label=f"f = {focal:.0f} px  ({np.degrees(np.arctan(2500/focal)):.0f}° half-FOV)")

    offset = np.linalg.norm(BEAM_ORIGIN)
    # The far tip, not the half-length: the dot is not centred on the body. This
    # is already a scene-side angle -- the offset and the fish are in the water
    # -- so it is plotted directly, with no conversion.
    far_tip = np.degrees(np.arctan((offset + (1.0 - DOT_FRACTION_WORST) * LENGTH) / 1.0))
    ax.axvspan(0, far_tip, color=figstyle.SERIES[2], alpha=0.10, lw=0)
    # Text in the empty wedge under the curve, with a panel behind it: the
    # curve climbs steeply past the shaded edge, so anything placed above it
    # ends up written across the data.
    ax.annotate(f"reachable: the dot must be on the fish,\nso nothing exceeds {far_tip:.1f}° even at 1 m",
                xy=(far_tip, 2.0), xytext=(18.5, 1.0), fontsize=7.4, color="#3f3e3b",
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#d8d6d1", lw=0.7, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color="#8a8a85", lw=0.9,
                                connectionstyle="arc3,rad=-0.2"))
    ax.set_xlabel("scene field angle, in water (degrees)")
    ax.set_ylabel("uncorrected length error (%)")
    ax.legend(fontsize=7.4, frameon=False, loc="upper left")
    ax.set_title("A flat port's error follows the field angle, not the lens")
    fig.tight_layout(rect=(0, 0, 1, 1))
    return fig, "sim-flat-port-error-vs-field-angle-by-focal-length"


# --------------------------------------------------------------------------
# Finished: what the target's build quality is worth
# --------------------------------------------------------------------------
# Measured by rendering the target with the stated defect and calibrating with
# the nominal model; eight views at 1400x1050. Reproduce with the sweep in
# `scratchpad/prove.py`; embedded here because one sweep is about ten minutes
# and the figure should not need that to redraw.
CHAMFER_MM = np.array([0.100, 0.175, 0.250, 0.325])
CHAMFER_FX = np.array([-0.806, -0.959, -0.890, -0.461])
SLOP_MM = np.array([0.00, 0.02, 0.05, 0.10, 0.20])
SLOP_FX = np.array([-0.890, -0.939, -1.025, -1.321, -1.956])
SLOP_RATIO = np.array([0.165, 0.163, 0.144, 0.146, 0.142])


def build_quality():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    axes[0].plot(CHAMFER_MM, CHAMFER_FX, "o-", color=figstyle.SERIES[0])
    axes[0].axvline(0.25, color="#8a8a85", lw=0.8, ls=":")
    axes[0].set_xlabel("true chamfer (mm), model assumes 0.25")
    axes[0].set_ylabel("recovered fx error (%)")
    axes[0].set_title("A constant we only think we know", fontsize=9)

    axes[1].plot(SLOP_MM, SLOP_FX, "o-", color=figstyle.SERIES[0], label="fx")
    axes[1].plot(SLOP_MM, SLOP_RATIO, "s--", color=figstyle.SERIES[1], label="fx/fy ratio")
    axes[1].set_xlabel("per-brick placement error (mm, sd)")
    axes[1].set_ylabel("recovered error (%)")
    axes[1].set_title("How well it is built", fontsize=9)
    axes[1].legend(fontsize=7.4, frameon=False)
    # Shared, and this one matters: autoscaled, the chamfer panel spans 0.4 %
    # and its scatter reads as a strong trend -- which is the opposite of what
    # the panel is there to show.
    for ax in axes:
        ax.set_ylim(-2.2, 0.45)
        ax.axhline(0.0, color="#8a8a85", lw=0.8)

    fig.suptitle("What limits “exact by construction” is assembly, not specification")
    fig.tight_layout(rect=(0, 0, 1, 1))
    return fig, "sim-target-sensitivity-to-build-quality"


# --------------------------------------------------------------------------
# The built target, on photographs
#
# These three were placeholders until the target was built and shot. They run
# on 21 raw frames of it, cached in `data/lego_target_raw.npz` so that they
# redraw without the image archive, as the board and laser caches do.
#
# Raw rather than JPEG, and not for tidiness: the TG-6 applies an in-camera lens
# correction to its JPEGs, so JPEG geometry is not the sensor's geometry. On
# this board the radial term measures k1 = -0.147 in raw against +0.011 in JPEG.
# --------------------------------------------------------------------------
LEGO_CACHE = DATA / "lego_target_raw.npz"
SENSOR = (4014, 3016)
_LEGO: dict = {}

#: Distortion is held at the board's value and only the intrinsic matrix is
#: solved. A tower 144 mm across spans too little depth to condition five
#: distortion terms; fitted freely they come back k2 = -0.38 with k3 = +0.68,
#: the signature of a radial series absorbing what it can reach. Fixing it makes
#: this a statement about K, which is what section 5 is about.
_FIX_DIST = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
             | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_TANGENT_DIST)


def _lego():
    """Cached brick-target correspondences, plus the board calibration to compare with."""
    if _LEGO:
        return _LEGO
    z = np.load(LEGO_CACHE, allow_pickle=False)
    views = sorted(k.split("/", 1)[1] for k in z.files if k.startswith("obj/"))
    objp = board_object_points().astype(np.float32)
    garage = cache_by_session(load_corner_cache(DATA / "board_corners_raw.npz"), "garage_lens")
    names = sorted(garage)
    guess = np.array([[2900.0, 0, SENSOR[0] / 2], [0, 2900.0, SENSOR[1] / 2], [0, 0, 1.0]])
    _, board_K, board_d, _, _ = cv2.calibrateCamera(
        [objp for _ in names],
        [garage[n].reshape(-1, 1, 2).astype(np.float32) for n in names],
        SENSOR, guess, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    _LEGO.update(raw=z, views=views, board_K=board_K, board_d=board_d,
                 obj=[z[f"obj/{v}"] for v in views], img=[z[f"img/{v}"] for v in views])
    return _LEGO


def _lego_calibrate(obj, img, board_K, board_d):
    _, K, _, _, _ = cv2.calibrateCamera(obj, img, SENSOR, board_K.copy(), board_d.copy(),
                                        flags=_FIX_DIST)
    return K


def detection_example():
    """The detector on one photograph of the built target.

    The frame is cached downscaled for display; the outlines are the
    full-resolution detections, which is what the figure is about.
    """
    z = _lego()["raw"]
    image = cv2.imdecode(z["frame/image"], cv2.IMREAD_COLOR)
    scale = float(z["frame/scale"][0])
    quads, matched = z["frame/quads"] * scale, z["frame/matched"] * scale

    canvas = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    centres = matched.reshape(-1, 4, 2).mean(axis=1)
    for quad in quads:
        claimed = np.linalg.norm(centres - quad.mean(axis=0), axis=1).min() < 5.0
        cv2.polylines(canvas, [np.round(quad).astype(np.int32)], True,
                      (233, 30, 160) if claimed else (120, 120, 120),
                      2 if claimed else 1, cv2.LINE_AA)
    for point in matched:
        cv2.circle(canvas, tuple(np.round(point).astype(int)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(np.round(point).astype(int)), 2, (16, 24, 40), -1, cv2.LINE_AA)

    pad = 30
    xs, ys = matched[:, 0], matched[:, 1]
    crop = canvas[max(int(ys.min()) - pad, 0):int(ys.max()) + pad,
                  max(int(xs.min()) - pad, 0):int(xs.max()) + pad]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.imshow(crop)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    faces = len(matched) // 4
    ax.set_title(f"{len(quads)} brick faces found; {faces} matched to the model "
                 f"(magenta), {len(quads) - faces} not (grey)")
    fig.tight_layout()
    return fig, "target-detection-on-one-view"


def focal_ratio_vs_views():
    """Which intrinsic a brick target constrains, as views are added.

    Not a claim that it settles `fx/fy`. It does not, on this capture: every
    frame was shot upright, roll spanning 14.9 degrees, so the tower's vertical
    axis maps to image y in all of them and the anisotropy direction is barely
    observed. Scaling the model's own z axis by +/-1.5 % moves the fitted ratio
    from 0.991 to 1.011 while the reprojection error moves only 5.066 to 5.087
    px -- the images have almost no leverage on that direction, so the estimate
    inherits whatever anisotropy the object model or the detector carries.

    What the figure does show is that the ratio is the better-determined of the
    two: across random subsets the absolute focal length scatters several times
    more widely. Both are drawn against the board's value, so the offset is
    visible beside the scatter rather than hidden by centring.
    """
    lego = _lego()
    obj, img = lego["obj"], lego["img"]
    board_K, board_d = lego["board_K"], lego["board_d"]
    rng = np.random.default_rng(0)

    board_ratio = board_K[0, 0] / board_K[1, 1]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for n in (6, 8, 10, 12, 16, len(obj)):
        for _ in range(12):
            take = rng.choice(len(obj), size=min(n, len(obj)), replace=False)
            K = _lego_calibrate([obj[k] for k in take], [img[k] for k in take], board_K, board_d)
            axes[0].plot(n, 100 * (K[0, 0] / board_K[0, 0] - 1), "o",
                         color=figstyle.SERIES[0], alpha=0.45, ms=4)
            axes[1].plot(n, 100 * ((K[0, 0] / K[1, 1]) / board_ratio - 1), "o",
                         color=figstyle.SERIES[0], alpha=0.45, ms=4)

    for ax, title in ((axes[0], "absolute scale"), (axes[1], "anisotropy")):
        ax.axhline(0.0, color=figstyle.SERIES[1], ls="--", lw=1.4, label="checkerboard")
        ax.set_xlabel("views used")
        ax.set_title(title, fontsize=9)
        # Shared, and that is the whole point: on one scale the ratio's scatter
        # is visibly a fraction of the focal length's. Given separate limits
        # each panel fills its own axes and the comparison disappears. The lower
        # bound clears the widest scale draw (~-10 %) so no point is clipped.
        ax.set_ylim(-11.0, 4.0)
        ax.legend(fontsize=7.2, frameon=False, loc="lower right")
    axes[0].set_ylabel("difference from the checkerboard (%)")
    fig.suptitle("A brick target pins the ratio better than the scale")
    # No baked subcaption: the numbers belong in the LaTeX \caption, where they
    # cannot go stale in the image.
    fig.tight_layout()
    return fig, "target-focal-ratio-vs-view-count"


def target_lego_vs_checkerboard():
    """Section 5: is a tower of bricks as good a reference as a printed board?

    One question, one axis. Both calibrations are taken in air, both run through
    the same correction and the same distortion, and both measure the same
    calipered 312.5 mm decoy on the same pool frames, so the calibration
    reference is the only thing differing.

    This is a different claim from recovering the same intrinsics, and a more
    useful one. The two calibrations disagree by ~2 % on focal length and ~0.8 % on
    the ratio and over a hundred pixels on the principal point; what is reported
    here is what that disagreement is worth once it has passed through a
    per-dive beam fit and arrived at a length.
    """
    import csv

    lego = _lego()
    board_K, board_d = lego["board_K"], lego["board_d"]
    brick_K = _lego_calibrate(lego["obj"], lego["img"], board_K, board_d)
    pool = _pool()
    objp, boards, board_spots, frames = pool["objp"], pool["boards"], pool["spots"], pool["frames"]

    def model(K):
        def correct(px):
            xy = cv2.undistortPoints(np.asarray(px, float).reshape(-1, 1, 2),
                                     K, board_d.ravel()).reshape(-1, 2)
            r = np.hypot(xy[:, 0], xy[:, 1])
            _, gamma = exit_ray(np.arctan(r), FlatPort(0.001, 0.006, 1.49, SWEET_WATER))
            return xy * np.where(r > 1e-9, np.tan(gamma) / np.where(r > 1e-9, r, 1.0), 1.0)[:, None]
        return correct

    def beam(correct):
        points = []
        for n in frames:
            u = correct(boards[n])
            _, rvec, tvec = cv2.solvePnP(objp, u.reshape(-1, 1, 2), np.eye(3), None)
            normal = cv2.Rodrigues(rvec)[0][:, 2]
            d = correct(board_spots[n][:2])[0]
            ray = np.array([d[0], d[1], 1.0]); ray /= np.linalg.norm(ray)
            points.append((normal @ np.asarray(tvec).ravel()) / (normal @ ray) * ray)
        points = np.array(points)
        origin, axis = fit_line(points)
        off = points - origin
        residual = np.linalg.norm(off - (off @ axis)[:, None] * axis, axis=1)
        return fit_line(points[residual <= 3 * np.median(residual)])

    spots = load_corner_cache(DATA / "laser_dots_decoy_raw.npz")
    with open(DATA / "decoy_sam3.csv") as handle:
        masks = {r["frame"]: r for r in csv.DictReader(handle)}
    true_length, true_aspect = 0.3125, 312.5 / 106.1
    kept = [n for n in sorted(set(spots) & set(masks))
            if abs((float(masks[n]["length_px"]) / float(masks[n]["height_px"]))
                   / true_aspect - 1) <= 0.25]

    series = {}
    for label, K in (("checkerboard", board_K), ("brick target", brick_K)):
        correct = model(K)
        origin, axis = beam(correct)
        ranges, errors = [], []
        for n in kept:
            d = correct(spots[n][:2])[0]
            z = range_along_beam(d, origin, axis)
            ends = [correct(np.array([[float(masks[n][f"end{i}_x"]),
                                       float(masks[n][f"end{i}_y"])]]))[0] for i in (0, 1)]
            sep = np.linalg.norm(np.append(ends[0], 1.0) - np.append(ends[1], 1.0))
            ranges.append(z); errors.append(100 * (sep * z / true_length - 1))
        series[label] = (np.array(ranges), np.array(errors))

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.axhline(0.0, color="#4a4a47", lw=1.0, zorder=1)
    # Both plotted at one x per frame, so a frame reads as a vertical pair.
    # Each calibration derives its own range, and drawing each at its own value
    # separates the two points of a pair by more than the difference the figure
    # exists to show.
    shared = 0.5 * (series["checkerboard"][0] + series["brick target"][0])
    for (label, (_, e)), colour, marker in zip(series.items(),
                                               (figstyle.SERIES[1], figstyle.SERIES[0]),
                                               ("s", "o")):
        ax.plot(shared, e, marker, color=colour, ms=6, alpha=0.85,
                label=f"{label}: median {np.median(e):+.2f} %")
    ax.set_xlabel("range to the decoy (m)")
    ax.set_ylabel("length error (%)")
    ax.legend(fontsize=7.4, frameon=False)
    ax.set_title("Same decoy, same correction, two in-air calibration targets")
    # No baked subcaption: the numbers (2.25 % focal, 0.79 % fx/fy, 0.04-0.18 pp
    # delivered) live in the LaTeX \caption, so they cannot go stale in the image
    # -- which is exactly how the old 20-view 0.4/0.9 figures survived here.
    fig.tight_layout()
    return fig, "target-pool-length-lego-vs-checkerboard"


def range_versus_length():
    """Section 4.2: the port's error goes into range, and cancels out of length.

    Simulation, deliberately: truth here is known by construction, where every
    range on real frames is derived through some camera model. The forward model
    is the exact flat-port projection; each pipeline then inverts it and is asked
    two questions about the same dot -- how far, and how long.

    The answers differ completely. An uncorrected port reports range 25 % short
    at every distance, because that is 1/n_w and nothing else, while reporting
    length almost correctly: the same factor inflates the angular extent as it
    shrinks the range. Only the second difference survives, which is why the
    length curve rises toward close range, where the fish subtends a larger
    angle.

    This figure exists to back a disclosure rather than a claim. No number in
    the paper depends on range -- FishSense Lite delivers a length -- but
    single-laser ranging is established practice in ROV survey work, so a reader
    may expect range to be an output of a rig like this one.
    """
    ranges = np.linspace(0.6, 5.0, 45)
    fish, offset = 0.30, 0.104
    beam_origin, beam_axis = np.array([0.0, -offset, 0.0]), np.array([0.0, 0.0, 1.0])
    across = np.array([1.0, 0.0, 0.0])          # this rig: |O| vertical, fish horizontal

    def image(points):
        alpha, azimuth = project_water_points(points, PORT)
        return alpha_azimuth_to_pixel(alpha, azimuth, CAMERA)

    curves = {}
    for name, back in (("Uncorrected", lambda q: back_project_uncorrected(q, CAMERA)),
                       ("Pinax", lambda q: back_project_pinax(q, CAMERA, PORT, _VCOP))):
        got_range, got_length = [], []
        for z in ranges:
            centre = beam_origin + z * beam_axis
            origin, direction = back(image(centre[None, :]))
            recovered, _ = reconstruct_points(direction, origin, beam_origin, beam_axis)
            reported = float(np.ravel(recovered[:, 2])[0])
            got_range.append(100 * (reported / z - 1))
            head, tail = centre + across * fish * 0.48, centre - across * fish * 0.52
            measured = measure_length(image(head[None, :]), image(tail[None, :]),
                                      recovered[:, 2], back)
            got_length.append(100 * (float(np.ravel(measured)[0]) / fish - 1))
        curves[name] = (np.array(got_range), np.array(got_length))

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.axhline(0.0, color="#8a8a85", lw=0.8, zorder=1)
    blue = figstyle.PIPELINE_STYLE["Uncorrected"]["color"]
    aqua = figstyle.PIPELINE_STYLE["Pinax"]["color"]
    ax.plot(ranges, curves["Uncorrected"][0], color=blue, ls="-", lw=2.0,
            label="Uncorrected — range")
    ax.plot(ranges, curves["Uncorrected"][1], color=blue, ls="--", lw=2.0,
            label="Uncorrected — length")
    ax.plot(ranges, curves["Pinax"][0], color=aqua, ls="-.", lw=2.0,
            label="Pinax — range and length")

    exact = 100 * (1 / SWEET_WATER - 1)
    ax.annotate(f"$1/n_w - 1$ = {exact:.1f} %", xy=(4.6, exact), xytext=(0, 7),
                textcoords="offset points", ha="right", fontsize=7.4, color=blue)
    ax.set_xlabel("true range to the dot (m)")
    ax.set_ylabel("error (%)")
    ax.set_ylim(-30.0, 8.0)
    ax.legend(fontsize=7.4, frameon=False, loc="center right")
    ax.set_title("The port's error goes into range, and cancels out of length")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig, "sim-uncorrected-range-error-vs-length-error"


# --------------------------------------------------------------------------
# Pool data, shared by the two end-to-end figures below
# --------------------------------------------------------------------------
_POOL: dict = {}


def _pool():
    """Camera models and the beam each one fits, from the board session.

    Cached because both end-to-end figures need the identical objects: the
    decoy figure is only meaningful if its beam is the one the board figure
    validates, fitted through the same intrinsics.
    """
    if _POOL:
        return _POOL
    objp = board_object_points()
    corners = load_corner_cache(DATA / "board_corners_raw.npz")
    dots = load_corner_cache(DATA / "laser_dots_raw.npz")

    def calibrated(session):
        cd = cache_by_session(corners, session)
        names = sorted(cd)
        _, K, dist, _, _ = cv2.calibrateCamera(
            [objp for _ in names],
            [cd[n].reshape(-1, 1, 2).astype(np.float32) for n in names],
            (4014, 3016), None, None)
        return K, dist.ravel()

    K_air, d_air = calibrated("garage_lens")
    K_water, d_water = calibrated("pool_lens")

    def undistort(px, K, dist):
        return cv2.undistortPoints(np.asarray(px, float).reshape(-1, 1, 2), K, dist).reshape(-1, 2)

    def pinax(xy):
        r = np.hypot(xy[:, 0], xy[:, 1])
        _, gamma = exit_ray(np.arctan(r), FlatPort(0.001, 0.006, 1.49, SWEET_WATER))
        return xy * np.where(r > 1e-9, np.tan(gamma) / np.where(r > 1e-9, r, 1.0), 1.0)[:, None]

    models = {
        "Uncorrected": lambda p: undistort(p, K_air, d_air),
        "In-water SVP": lambda p: undistort(p, K_water, d_water),
        "Pinax": lambda p: pinax(undistort(p, K_air, d_air)),
    }

    boards = cache_by_session(corners, "pool_lens")
    spots = {k: v for k, v in cache_by_session(dots, "water").items() if "244" not in k}
    frames = sorted(set(boards) & set(spots))

    def beam(correct):
        points = []
        for n in frames:
            u = correct(boards[n])
            _, rvec, tvec = cv2.solvePnP(objp, u.reshape(-1, 1, 2), np.eye(3), None)
            normal = cv2.Rodrigues(rvec)[0][:, 2]
            d = correct(spots[n][:2])[0]
            ray = np.array([d[0], d[1], 1.0]); ray /= np.linalg.norm(ray)
            points.append((normal @ np.asarray(tvec).ravel()) / (normal @ ray) * ray)
        points = np.array(points)
        origin, axis = fit_line(points)
        off = points - origin
        residual = np.linalg.norm(off - (off @ axis)[:, None] * axis, axis=1)
        return fit_line(points[residual <= 3 * np.median(residual)])

    _POOL.update(objp=objp, models=models, boards=boards, spots=spots,
                 frames=frames, beams={n: beam(f) for n, f in models.items()})
    return _POOL


MARKERS = {"Uncorrected": "o", "In-water SVP": "s", "Pinax": "^"}


def pool_range_by_model():
    """Section 4.2, on real frames: the same range disagreement, measured.

    There is no model-free range on real data -- every one is derived through
    some camera model -- so this cannot be a plot against truth, and is not
    drawn as one. What it is instead is a check between two *independent*
    routes.

    The in-water SVP calibration is fitted to the underwater frames themselves
    and contains no refraction theory at all. Pinax is derived from the in-air
    calibration plus the port model and never sees an underwater frame. They
    share no data and no assumptions, and they agree on the board's range to
    better than a percent. An uncorrected port departs from both by about a
    quarter, which is the number the simulation predicts for 1/n_w.

    Agreement between two independent routes is not proof that either is right.
    It is, though, the strongest statement this dataset supports, and it is why
    the claim is put as a disclosure in section 4.2 rather than as a result.

    The scale is not assumed. Each range is solved from the board's *calipered*
    geometry, so the metre comes from a physical object and the camera model
    contributes only angles; and the magnification those angles carry was
    measured rather than taken from theory -- section 4.1's in-water fit returns
    a focal ratio of 1.3126 against the 1.333 refraction predicts, from
    underwater frames alone. The uncorrected model assumes that ratio is 1.
    """
    pool = _pool()
    objp, models, boards, spots, frames = (pool["objp"], pool["models"], pool["boards"],
                                           pool["spots"], pool["frames"])

    def board_range(correct, n):
        """Range to the dot from the board's own pose -- metric via its calipered size."""
        u = correct(boards[n])
        _, rvec, tvec = cv2.solvePnP(objp, u.reshape(-1, 1, 2), np.eye(3), None)
        normal = cv2.Rodrigues(rvec)[0][:, 2]
        d = correct(spots[n][:2])[0]
        ray = np.array([d[0], d[1], 1.0]); ray /= np.linalg.norm(ray)
        return float(((normal @ np.asarray(tvec).ravel()) / (normal @ ray) * ray)[2])

    ranges = {name: np.array([board_range(fn, n) for n in frames])
              for name, fn in models.items()}

    # The corrected reference is Pinax. An in-water single-viewpoint fit agrees
    # with it to better than a percent -- an independent check, since it is
    # fitted to the underwater frames and holds no refraction theory -- but the
    # paper never defines that model, so showing it here as a second corrected
    # series only invites the question of why there are two. It is dropped, and
    # the figure makes the one claim it needs: the uncorrected port reads a
    # quarter short of the corrected range.
    reference = ranges["Pinax"]

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    lo, hi = 0.6, 1.08 * reference.max()
    ax.plot([lo, hi], [lo, hi], color="#4a4a47", lw=1.0, zorder=1)
    ax.annotate("Pinax reference", xy=(lo + 0.58 * (hi - lo), lo + 0.58 * (hi - lo)),
                xytext=(7, -4), textcoords="offset points", ha="left", va="top",
                fontsize=7.2, color="#4a4a47", rotation=45, rotation_mode="anchor")

    departure = {}
    for name in models:
        if name == "In-water SVP":
            continue  # a second corrected series the paper never defines; see reference note
        style = figstyle.pipeline(name, line=False)
        ax.plot(reference, ranges[name], marker=MARKERS[name], markersize=6,
                linestyle="none", label=name, zorder=3, markeredgewidth=1.4,
                markerfacecolor="none" if name == "Pinax" else style["color"], **style)
        departure[name] = np.median(100 * (ranges[name] / reference - 1))

    ax.set_xlabel("Pinax range (m)")
    ax.set_ylabel("range from each model (m)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7.4, frameon=False, loc="upper left")
    ax.set_title("Range from the board's own pose:\nthe uncorrected port reads a quarter short", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, "pool-range-by-camera-model"


def pool_end_to_end():
    """Section 6: each camera model calibrates the laser with itself, then measures.

    The experiment that isolates the port. A deployment never mixes camera
    models: whatever intrinsics calibrate the laser are the ones that then
    measure with it, so each model here is run as its own closed pipeline.
    Range comes from the dot, length from the frame, and the known target is the
    board's 549.0 mm span.

    Real data, not a placeholder. Ten pool frames carry both a board and a laser
    dot; frame 244 is excluded, as it was when these numbers were first derived.

    Two details matter and neither is incidental. The beam is fitted by
    `fit_line` through the points where each dot's ray meets its own board
    plane, then refitted after rejecting anything beyond three times the median
    residual -- not by `fit_beam_to_ranges`, which answers a different question.
    And the span is averaged over all ten rows of the board rather than taken
    from one corner pair, because a single pair across a tilted board is
    dominated by the depth difference between its ends: measured that way the
    worst case triples.
    """
    pool = _pool()
    objp, models = pool["objp"], pool["models"]
    boards, spots, frames = pool["boards"], pool["spots"], pool["frames"]
    index = lambda c, r: r * PATTERN[0] + c
    long_pairs = [(index(0, r), index(13, r)) for r in range(PATTERN[1])]
    true_span = 13 * SQUARE_PITCH_M[0]

    measured = {}
    for name, correct in models.items():
        origin, axis = pool["beams"][name]
        per_frame = {}
        for n in frames:
            u = correct(boards[n])
            d = correct(spots[n][:2])[0]
            z = range_along_beam(d, origin, axis)
            rays = np.c_[u, np.ones(len(u))]
            pts = rays * (z / rays[:, 2])[:, None]
            span = np.mean([np.linalg.norm(pts[i] - pts[j]) for i, j in long_pairs])
            per_frame[n] = (z, 100 * (span - true_span) / true_span)
        measured[name] = per_frame

    # One axis here is ground truth and the other is not, and the distinction
    # decides what this figure may claim.
    #
    # The calipered 549.0 mm span is truth: a physical length, measured with
    # calipers, entering the pipeline nowhere else. Length error against it is a
    # real accuracy result.
    #
    # Range is NOT. Every range here is derived through a camera model, and the
    # board pose that might serve as a reference is the very quantity the beam
    # was fitted to -- same expression, same inputs -- so comparing the two
    # measures a fit residual. Fitting leave-one-out instead, each model
    # predicts its *own* board poses to about 0.2 % median, the uncorrected one
    # included. It is wrong about range by a quarter and wrong about
    # magnification by a quarter, and the two are consistent, so nothing inside
    # the laser-plus-board system can tell. Section 4.1's homography test can,
    # which is why it exists and why it needs no range at all.
    #
    # So the range axis is labelled as one model's estimate and claims nothing.
    order = sorted(frames, key=lambda n: measured["Pinax"][n][0])
    x = np.array([measured["Pinax"][n][0] for n in order])

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.axhline(0.0, color="#4a4a47", lw=1.0, zorder=1)
    ax.annotate("calipered 549.0 mm", xy=(0.99, 0.0), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=7.2, color="#4a4a47")
    summary = []
    for name in models:
        err = np.array([measured[name][n][1] for n in order])
        style = figstyle.pipeline(name, line=False)
        ax.plot(x, err, marker=MARKERS[name], markersize=5, linestyle="none",
                label=name, zorder=3, markeredgewidth=1.4,
                markerfacecolor="none" if name == "Pinax" else style["color"], **style)
        summary.append((np.median(err), np.abs(err).max()))

    ax.set_xlabel("range to the board, as Pinax estimates it (m)")
    ax.set_ylabel("length error (%)")
    ax.set_ylim(-3.0, 3.0)
    ax.legend(fontsize=7.4, frameon=False, loc="lower right", ncol=3,
              handletextpad=0.3, columnspacing=1.2)
    ax.set_title("Camera-only end to end: measuring a 549.0 mm span")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig, "pool-end-to-end-board-span-by-camera-model"


def pool_decoy_length():
    """Section 6: the same three models measuring a fish-shaped object.

    The board figure above measures a rigid plane; this measures the decoy, a
    fish-shaped body that swings on its line. Both use the *same* beam, fitted
    on the board session through each model's own intrinsics, so the only thing
    changing is the object being measured.

    The three models are indistinguishable on it -- medians within 0.2 pp of
    each other. Read that carefully: it is **not** evidence that the uncorrected
    model is fine, and it is not evidence that the correction is unnecessary. It
    is evidence that this measurement cannot tell, which is section 4.2's
    prediction arriving on a fish-shaped body. The dot has to land on the decoy,
    so the decoy sits near the optical axis, and there the port's range error and
    its magnification error move together and cancel out of the length. Anything
    claiming the uncorrected model is bad has to come from section 4.1, whose
    test needs no range and therefore has nothing to cancel against.

    Range is not shown here; section 4.2 carries it. What does limit a length is
    the silhouette. One frame's mask has an aspect ratio
    of 4.7 against the decoy's calipered 2.95, and carries a +28 % error on its
    own; the rule that rejects it uses no range, no calibration and no known
    length, only the mask's own proportions, which is the same scale-free
    argument P1 makes for reading pose off an aspect ratio.

    The beam is fitted on the board session and applied across to the decoy
    session, which precedes it by about ten minutes and one handling event, so
    the roughly -2.7 % common offset includes real drift and is not evidence
    about any of the three models.
    """
    import csv

    pool = _pool()
    models = pool["models"]
    spots = load_corner_cache(DATA / "laser_dots_decoy_raw.npz")
    with open(DATA / "decoy_sam3.csv") as handle:
        masks = {r["frame"]: r for r in csv.DictReader(handle)}

    true_length = 0.3125            # calipered, 312-313 mm
    true_aspect = 312.5 / 106.1     # calipered length over calipered body height
    frames = sorted(set(spots) & set(masks))

    def aspect(n):
        return float(masks[n]["length_px"]) / float(masks[n]["height_px"])

    kept = [n for n in frames if abs(aspect(n) / true_aspect - 1) <= 0.25]
    dropped = [n for n in frames if n not in kept]

    # One panel, and deliberately only the length half.
    #
    # An earlier version put range beside it, showing the uncorrected model a
    # quarter short. That panel was not independent evidence and should not have
    # been read as any: the beam each model uses is fitted from the board frames
    # that section 4.2's range figure already plots, and both routes back-project
    # through the same intrinsics, so the same n_w error simply appeared twice.
    # Section 4.2 carries range; this carries what only the decoy can show, which
    # is a fish-shaped body at unconstrained pose.
    #
    # Worth recording from that deleted panel, because it is not obvious: the
    # beam fit does not absorb the magnification error, it passes it through. The
    # uncorrected model recovers the laser's transverse offset as 101.2 mm
    # against Pinax's 103.0 and a bench 104 -- essentially right -- while its
    # ranges come out a quarter short. |O| is transverse and survives the
    # magnification; range goes as |O|/tan(alpha) and takes all of it.
    measured, lengths = {}, {}
    for name, correct in models.items():
        origin, axis = pool["beams"][name]
        rng, err = {}, {}
        for n in kept:
            d = correct(spots[n][:2])[0]
            z = range_along_beam(d, origin, axis)
            ends = [correct(np.array([[float(masks[n][f"end{i}_x"]),
                                       float(masks[n][f"end{i}_y"])]]))[0] for i in (0, 1)]
            sep = np.linalg.norm(np.append(ends[0], 1.0) - np.append(ends[1], 1.0))
            rng[n] = z
            err[n] = 100 * (sep * z / true_length - 1)
        measured[name], lengths[name] = rng, err

    # Every model plotted at one x, so a frame reads as a vertical triplet.
    reference = np.array([0.5 * (measured["In-water SVP"][n] + measured["Pinax"][n])
                          for n in kept])

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.axhline(0.0, color="#4a4a47", lw=1.0, zorder=1)
    ax.annotate("calipered 312.5 mm", xy=(0.99, 0.0), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=7.2, color="#4a4a47")
    summary = []
    for name in models:
        style = figstyle.pipeline(name, line=False)
        ys = np.array([lengths[name][n] for n in kept])
        ax.plot(reference, ys, marker=MARKERS[name], markersize=5, linestyle="none",
                label=name, zorder=3, markeredgewidth=1.4,
                markerfacecolor="none" if name == "Pinax" else style["color"], **style)
        summary.append(np.median(ys))
    ax.set_xlabel("range to the decoy (m)")
    ax.set_ylabel("length error (%)")
    ax.set_ylim(-8.0, 4.0)
    ax.legend(fontsize=7.4, frameon=False, loc="lower right", ncol=3,
              handletextpad=0.3, columnspacing=1.2)
    ax.set_title("The same length from all three camera models")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig, "pool-decoy-length-by-camera-model"


def main():
    figstyle.apply()
    finished = [error_budget, field_angle_collapse, build_quality, range_versus_length,
                pool_range_by_model, pool_end_to_end, pool_decoy_length,
                detection_example, focal_ratio_vs_views, target_lego_vs_checkerboard]
    provisional = []
    for builder in finished + provisional:
        fig, name = builder()
        paths = figstyle.save(fig, name)
        plt.close(fig)
        tag = "placeholder" if builder in provisional else "final"
        print(f"  [{tag:11s}] {paths[0].name}")


if __name__ == "__main__":
    main()
