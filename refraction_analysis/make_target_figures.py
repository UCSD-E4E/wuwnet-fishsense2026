"""Figures for the calibration-target sections, section 5 onward.

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

from fishsense_wuwnet import figstyle
from fishsense_wuwnet.laser import fit_beam_to_ranges
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
    optimal_d0,
    project_water_points,
    reconstruct_points,
)
from fishsense_wuwnet.target_detect import detect_quads, match
from fishsense_wuwnet.target_render import look_at, render

CAMERA = np.array([[2876.0, 0.0, 2000.0], [0.0, 2845.0, 1500.0], [0.0, 0.0, 1.0]])
_D0, _VCOP, _ = optimal_d0(0.006, 1.49, SWEET_WATER, np.radians(41.0))
PORT = FlatPort(_D0, 0.006, 1.49, SWEET_WATER)
BEAM_ORIGIN = np.array([-0.04, -0.11, 0.0])
BEAM_AXIS = np.array([0.0, 0.0, 1.0])
LENGTH, DOT_FRACTION = 0.30, 0.52
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
    fig.text(
        0.5, 0.005,
        "Trusting the bench, range and extent scale together and the focal error cancels; the dot sits "
        "~80 px off axis\nat 4 m, so the principal point is ruinous. Re-fitting the beam absorbs the "
        "principal point and, by pinning the\nrange, destroys that cancellation. Dotted line is the "
        "15 % worst-case budget.",
        ha="center", va="bottom", fontsize=7.2, color="#52514e")
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    return fig, "sim-calibration-error-budget-by-laser-regime"


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
def field_angle_collapse():
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
            laser_radius=focal * np.tan(np.radians(4.0)), extent="radial")
        ax.plot(angles, error, color=figstyle.SERIES[i], ls=figstyle.DASHES[i], lw=2.0,
                label=f"f = {focal:.0f} px  ({np.degrees(np.arctan(2500/focal)):.0f}° half-FOV)")

    offset = np.linalg.norm(BEAM_ORIGIN)
    far_tip = np.degrees(np.arctan((offset + LENGTH / 2) / 1.0))
    ax.axvspan(0, far_tip, color=figstyle.SERIES[2], alpha=0.10, lw=0)
    ax.annotate(f"reachable: the dot must be on the fish,\nso nothing exceeds {far_tip:.0f}° even at 1 m",
                xy=(far_tip, 3.0), xytext=(19.0, 9.0), fontsize=7.4, color="#52514e",
                arrowprops=dict(arrowstyle="->", color="#8a8a85", lw=0.9))
    ax.set_xlabel("field angle (degrees)")
    ax.set_ylabel("uncorrected length error (%)")
    ax.legend(fontsize=7.4, frameon=False, loc="upper left")
    ax.set_title("A flat port's error follows the field angle, not the lens")
    fig.text(0.5, 0.005,
             "Three lenses spanning a threefold change in focal length, same pane. The curves are "
             "identical to better than\n0.01 %, so the correction transfers to any flat-pane housing "
             "— action cameras and phones included.",
             ha="center", va="bottom", fontsize=7.2, color="#52514e")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
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
    fig.text(0.5, 0.005,
             "Misspecifying the chamfer by ±0.15 mm moves fx by less than the estimator's own scatter. "
             "Assembly slop does\nbind — about 1 % of length at 0.2 mm — and it leaves the "
             "fx/fy ratio, the quantity the tower is for, alone.",
             ha="center", va="bottom", fontsize=7.2, color="#52514e")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    return fig, "sim-target-sensitivity-to-build-quality"


# --------------------------------------------------------------------------
# Placeholders: everything that should be a photograph
# --------------------------------------------------------------------------
SIZE = (2200, 1650)
SMALL = np.array([[1570.0, 0.0, 1100.0], [0.0, 1553.0, 825.0], [0.0, 0.0, 1.0]])


def detection_example():
    rotation, translation = look_at((300.0, -380.0, 210.0))
    view = render(SMALL, rotation, translation, size=SIZE)
    quads = detect_quads(view.image, min_area_px=60.0)
    found = match(quads, SMALL, min_faces=4)

    # Matched and unmatched are drawn differently, and in colours the target
    # does not itself contain -- outlining bricks in yellow on a target with
    # yellow bricks hides the thing the figure is about.
    canvas = cv2.cvtColor(view.image, cv2.COLOR_BGR2RGB)
    matched_centres = found.image_points.reshape(-1, 4, 2).mean(axis=1)
    for quad in quads:
        claimed = (
            len(matched_centres)
            and np.linalg.norm(matched_centres - quad.centre, axis=1).min() < 5.0
        )
        cv2.polylines(
            canvas, [quad.corners.round().astype(np.int32)], True,
            (233, 30, 160) if claimed else (120, 120, 120),
            2 if claimed else 1, cv2.LINE_AA)
    for point in found.image_points:
        cv2.circle(canvas, tuple(point.round().astype(int)), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(point.round().astype(int)), 4, (16, 24, 40), -1, cv2.LINE_AA)

    xs, ys = found.image_points[:, 0], found.image_points[:, 1]
    pad = 60
    crop = canvas[max(int(ys.min()) - pad, 0):int(ys.max()) + pad,
                  max(int(xs.min()) - pad, 0):int(xs.max()) + pad]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.imshow(crop)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{len(quads)} brick faces found; {len(found) // 4} matched to the model "
                 f"(magenta), {len(quads) - len(found) // 4} not (grey)")
    figstyle.placeholder(fig, "rendered view — replace with a photograph of the built target")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig, "target-detection-on-one-view"


def focal_ratio_vs_views():
    rng = np.random.default_rng(3)
    object_points, image_points = [], []
    for azimuth in np.linspace(0.0, 2 * np.pi, 14, endpoint=False):
        radius, elevation = rng.uniform(330, 520), rng.uniform(-0.2, 0.5)
        eye = (radius * np.cos(azimuth) * np.cos(elevation),
               radius * np.sin(azimuth) * np.cos(elevation),
               43.2 + radius * np.sin(elevation))
        rotation, translation = look_at(eye)
        view = render(SMALL, rotation, translation, size=SIZE)
        found = match(detect_quads(view.image, min_area_px=60.0), SMALL, min_faces=4)
        if len(found) >= 24:
            object_points.append(found.object_points.astype(np.float32))
            image_points.append(found.image_points.astype(np.float32))

    truth = SMALL[0, 0] / SMALL[1, 1]
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST
             | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    counts, ratios, focals = [], [], []
    for n in range(4, len(object_points) + 1, 2):
        guess = np.array([[SMALL[0, 0] * 1.1, 0, SIZE[0] / 2],
                          [0, SMALL[0, 0] * 1.1, SIZE[1] / 2], [0, 0, 1.0]])
        _, fitted, _, _, _ = cv2.calibrateCamera(
            object_points[:n], image_points[:n], SIZE, guess, None, flags=flags)
        counts.append(n)
        ratios.append(100 * ((fitted[0, 0] / fitted[1, 1]) / truth - 1))
        focals.append(100 * (fitted[0, 0] / SMALL[0, 0] - 1))

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.axhline(0.0, color="#8a8a85", lw=0.8)
    ax.plot(counts, focals, "o-", color=figstyle.SERIES[0], label="fx")
    ax.plot(counts, ratios, "s--", color=figstyle.SERIES[1], label="fx/fy ratio")
    ax.set_xlabel("views used")
    ax.set_ylabel("error (%)")
    ax.legend(fontsize=7.4, frameon=False, loc="lower right")
    # Fixed, and wide enough to read as "flat". Autoscaled, a spread of a
    # tenth of a percent fills the panel and looks like a trend.
    ax.set_ylim(-0.15, 0.15)
    ax.set_title(f"Recovering a {100 * (truth - 1):.2f} % anisotropy from unrolled views")
    fig.text(0.5, 0.005,
             f"Both stay inside \u00b10.1 % from four views on. The anisotropy being recovered is "
             f"{100 * (truth - 1):.2f} % \u2014 about thirty times\nthe residual error \u2014 and no "
             f"view here is rolled, which is the case a planar board cannot settle.",
             ha="center", va="bottom", fontsize=7.2, color="#52514e")
    figstyle.placeholder(fig, "rendered views — replace with photographs of the built target")
    fig.tight_layout(rect=(0, 0.19, 1, 0.92))
    return fig, "target-focal-ratio-vs-view-count"


def pool_fish_by_calibration():
    """Section 6 end-to-end: an in-air LEGO calibration, used to measure the pool fish.

    Holds the slot for the experiment itself. The axes are real -- the range
    spread is the pool session's 1.08-4.48 m and the reference length is the
    decoy's calipered 312.5 mm -- but every plotted value is drawn from a fixed
    seed and is not a measurement.

    The series here are *calibration sources*, not pipelines, so
    `figstyle.PIPELINE_STYLE` deliberately does not apply: mapping "LEGO target"
    onto Pinax's colour would claim an identity between two different things.
    Slots are taken from `SERIES` directly, in order, each with its own dash.
    """
    rng = np.random.default_rng(11)
    ranges = np.linspace(1.08, 4.48, 13)

    # (label, slot, dash, marker, bias %, per-frame scatter %)
    sources = (
        ("LEGO target, in air", 2, "-", "o", -0.4, 0.7),
        ("Checkerboard, in air", 1, "--", "s", 1.6, 1.1),
        ("Checkerboard, in water (SVP)", 0, "-.", "^", 0.9, 0.9),
    )

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.axhline(0.0, color="#8a8a85", lw=0.8, zorder=1)
    for label, slot, dash, marker, bias, scatter in sources:
        errors = bias + rng.normal(0.0, scatter, ranges.size)
        ax.plot(ranges, errors, marker=marker, linestyle=dash, markersize=4,
                color=figstyle.SERIES[slot], label=label, zorder=3)

    ax.set_xlabel("range (m)")
    ax.set_ylabel("length error (%)")
    # Fixed, so the three sources stay separable. The 15 % budget is ten times
    # this half-height; drawing it would flatten every series onto the axis, so
    # it is stated instead of plotted.
    ax.set_ylim(-6.0, 6.0)
    ax.legend(fontsize=7.4, frameon=False, loc="upper right", ncol=1)
    ax.set_title("Pool fish length, by the target the camera was calibrated on")
    fig.text(0.5, 0.005,
             "Thirteen frames of the 312.5 mm decoy over the pool session's range spread, each "
             "measured through the flat-port\ncorrection. The deployment budget is 15 % worst "
             "case \u2014 off scale here, ten times the half-height plotted.",
             ha="center", va="bottom", fontsize=7.2, color="#52514e")
    figstyle.placeholder(
        fig, "illustrative values \u2014 replace with the pool fish measured "
             "against an in-air LEGO calibration")
    fig.tight_layout(rect=(0, 0.17, 1, 0.92))
    return fig, "pool-fish-length-by-in-air-calibration-target"


def main():
    figstyle.apply()
    finished = [error_budget, field_angle_collapse, build_quality]
    provisional = [detection_example, focal_ratio_vs_views, pool_fish_by_calibration]
    for builder in finished + provisional:
        fig, name = builder()
        paths = figstyle.save(fig, name)
        plt.close(fig)
        tag = "placeholder" if builder in provisional else "final"
        print(f"  [{tag:11s}] {paths[0].name}")


if __name__ == "__main__":
    main()
