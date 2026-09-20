"""The 2024-05-01 Garage/Pool flat-port dataset.

An Olympus TG-6 in its FishSense enclosure, photographed against one checkerboard
in a garage and then in a swimming pool, same afternoon, sequential frames. It is
a matched in-air / in-water pair with no confound, which is what makes it usable
as ground truth for the refraction analysis.

Session structure, recovered from EXIF timestamps and from where the laser
appears (the folders themselves record none of this):

===================  ==========  ==============================================
Session              Frames      Contents
===================  ==========  ==============================================
``garage_lens``      001-064     board in air, laser off
(marker)             065         photo of feet -- the break; laser switched on
``garage_laser``     066-086     board in air, green laser dot on the board
``pool_laser``       092-109     fish decoy in water, laser dot on the decoy
(268 s gap)                      rig handling
``pool_lens``        110-261     board in water; laser on again from 239
===================  ==========  ==============================================

The board is 14x10 *inner corners* (15x11 squares) of 42 mm, so 630 x 462 mm.

Reading a frame off the NAS costs several seconds, so corner detection is done
once and cached; see `build_corner_cache`. The notebooks read the cache.
"""

import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

DATA_ROOT = Path(
    os.environ.get("FISHSENSE_DATA_ROOT", "~/mnt/fishsense_data/__unsorted-data")
).expanduser()

GARAGE_DIR = DATA_ROOT / "2024.05.01.FishSense.Chris Garage"
POOL_DIR = DATA_ROOT / "2024.05.01.FishSense.Chris Pool"

PATTERN = (14, 10)  # inner corners

#: Nominal pitch the board was *specified* at. Kept only so results computed against
#: it can be reproduced; it is not the board.
NOMINAL_SQUARE_SIZE_M = 0.042

#: Measured pitch of the physical E4E board used for the 2024-05-01 sessions, from
#: calipered **corner-to-corner** spans: 549 mm over the 13 pitches of the long axis
#: and 379 mm over the 9 of the short one.
#:
#: The board is very nearly square -- the two pitches differ by 0.28% -- and both run
#: about 0.55% over the nominal 42.0 mm. Corner-to-corner is the span these object
#: points represent; an earlier inner-edge-to-inner-edge reading gave 548.75 / 384.0
#: and is superseded.
#:
#: **Do not use this to settle the fx/fy question.** With a square object model an
#: in-air fit returns fx/fy = 0.9931, and all seven cameras of the IMWUT production
#: fleet return 0.99141 +- 0.00044. Backing this board out of that fit leaves a *larger*
#: camera anisotropy, 0.97%, not a smaller one -- so on this reading the target is not
#: the cause. But the conclusion is not stable at the precision a rule affords: the short
#: axis is only 9 pitches, so 1 mm of reading error is 0.26% of it, a reading of 377.5 mm
#: would make the camera exactly square, and two careful attempts at this span have
#: differed by 5 mm. The measurement cannot decide it either way.
#:
#: What can: one calibration session shot in portrait. A camera-side anisotropy is fixed
#: in sensor coordinates and is unchanged by rotating the camera; a target-side one is
#: fixed to the board and inverts about 1.0. That test compares the camera to itself and
#: depends on no length measurement at all.
#:
#: What *is* settled either way: a single scalar square size cannot express an
#: anisotropic target, which is a limitation of every calibration path here that takes
#: one, production included.
SQUARE_PITCH_M = (0.549 / 13, 0.379 / 9)  # (long axis, short axis)

#: Deprecated alias. Prefer `SQUARE_PITCH_M`; this is the isotropic mean and exists
#: only for code that still wants one number.
SQUARE_SIZE_M = float(np.mean(SQUARE_PITCH_M))

# (directory, first frame, last frame) -- inclusive, by the NNNN in PxxxNNNN.JPG.
SESSIONS = {
    "garage_lens": (GARAGE_DIR, 1, 64),
    "garage_laser": (GARAGE_DIR, 66, 86),
    "pool_laser": (POOL_DIR, 92, 109),
    "pool_lens": (POOL_DIR, 110, 261),
}

# Frames whose laser dot was confirmed by eye. The pool lens-calibration session
# has the laser on only from 239 onward, and those are the frames that carry both
# a board pose and a dot -- the only in-water data that can anchor a 3D point on
# the beam.
POOL_LASER_ON_BOARD = (239, 240, 241, 242, 243, 244, 247, 258, 259, 260, 261)


def frame_number(path) -> int:
    """The NNNN in ``PxxxNNNN.JPG``."""
    return int(re.search(r"P\d{2}(\d{5})\.JPG", Path(path).name).group(1)[-4:])


def session_frames(session: str) -> list:
    """Sorted image paths for a named session."""
    directory, first, last = SESSIONS[session]
    return sorted(
        p for p in directory.glob("*.JPG") if first <= frame_number(p) <= last
    )


def board_object_points(pattern=PATTERN, square=SQUARE_PITCH_M) -> np.ndarray:
    """Planar board corner coordinates in metres, shape ``(cols*rows, 3)``, z = 0.

    `square` is the measured, **anisotropic** pitch ``(long_axis, short_axis)`` by
    default; pass a single float for an isotropic board. Passing
    ``NOMINAL_SQUARE_SIZE_M`` reproduces results computed before the board was
    calipered -- see `SQUARE_PITCH_M` for why that is not the same thing.
    """
    cols, rows = pattern
    pitch = np.broadcast_to(np.asarray(square, dtype=float).ravel(), (2,))
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * pitch
    return objp


def detect_corners(image, pattern=PATTERN) -> Optional[np.ndarray]:
    """Detect the full board, returning ``(cols*rows, 2)`` float32 or None.

    Uses OpenCV's sector-based detector: it is far faster than
    `findChessboardCorners` on failures (which matters when a third of the frames
    have no usable board), already subpixel-accurate, and copes with the blur and
    low contrast of the underwater frames.

    Partial boards are rejected. They are usable in principle, but the meta grid
    does not say *which* sub-block was found, so the object-point correspondence
    would be ambiguous.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray, pattern, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    if not found:
        return None
    return corners.reshape(-1, 2).astype(np.float32)


def detect_laser_dot(
    image, locus=(1700, 2250, 700, 1650), min_contrast=35.0, min_pixels=4
) -> Optional[Tuple[float, float, int, float]]:
    """Locate the green laser dot, returning ``(x, y, n_pixels, contrast)`` or None.

    Green excess alone does not work: it saturates on the white board in air and
    collapses against blue-green pool water. What survives both is *local* green
    contrast -- the dot is a compact local maximum relative to its surroundings.

    `locus` bounds the search to where a rigidly-mounted laser can put the dot.
    That is what rejects the two systematic false positives in this dataset: the
    green trash bin in the garage frames, and (only partly) the green
    "Engineers for Exploration" logo printed on the board header. The logo is
    better removed in board coordinates once a pose is known -- see the notebook.
    """
    from scipy import ndimage

    rgb = image.astype(np.float32)
    if rgb.shape[-1] == 3:  # OpenCV gives BGR
        blue, green, red = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    else:
        raise ValueError("detect_laser_dot needs a colour image")

    excess = green - 0.5 * (red + blue)
    local = excess - ndimage.median_filter(excess, size=21)

    mask = (local > 20.0) & (green > 110.0)
    labels, count = ndimage.label(mask)

    x0, x1, y0, y1 = locus
    best = None
    for i in range(1, count + 1):
        ys, xs = np.nonzero(labels == i)
        if not (min_pixels <= len(xs) <= 400):
            continue
        if (xs.max() - xs.min()) > 35 or (ys.max() - ys.min()) > 35:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        if not (x0 < cx < x1 and y0 < cy < y1):
            continue
        contrast = float(local[ys, xs].max())
        if contrast < min_contrast:
            continue
        if best is None or contrast > best[3]:
            best = (cx, cy, int(len(xs)), contrast)
    return best


def build_corner_cache(out_path, sessions: Iterable[str] = None, progress=True) -> Dict:
    """Detect boards across the named sessions once and cache them to ``.npz``.

    Each frame costs several seconds to pull off the NAS, so a full pass is tens
    of minutes. Run this once; the notebooks load the result.
    """
    sessions = list(SESSIONS) if sessions is None else list(sessions)
    found = {}

    for session in sessions:
        paths = session_frames(session)
        if progress:
            print(f"{session}: {len(paths)} frames", flush=True)
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            corners = detect_corners(image)
            if corners is not None:
                found[f"{session}/{path.name}"] = corners
            if progress:
                print(
                    f"  {path.name} {'ok' if corners is not None else '--'}", flush=True
                )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **found)
    if progress:
        print(f"cached {len(found)} boards -> {out_path}", flush=True)
    return found


def load_corner_cache(path) -> Dict[str, np.ndarray]:
    """Load a cache written by `build_corner_cache`, keyed ``session/filename``."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def cache_by_session(cache: Dict[str, np.ndarray], session: str) -> Dict[str, np.ndarray]:
    """The entries of a cache belonging to one session, keyed by filename."""
    prefix = f"{session}/"
    return {k[len(prefix) :]: v for k, v in cache.items() if k.startswith(prefix)}


def laser_frames() -> Dict[str, list]:
    """The frames carrying both a board pose and a laser dot, by medium.

    These are the only frames that can anchor a 3D point on the beam: the pool
    *laser* session (092-109) aims at a fish decoy whose geometry is unknown, so
    despite being the densest laser data it cannot be used for a fit.
    """
    pool_dir, _, _ = SESSIONS["pool_lens"]
    return {
        "air": session_frames("garage_laser"),
        "water": [
            p
            for p in sorted(pool_dir.glob("*.JPG"))
            if frame_number(p) in POOL_LASER_ON_BOARD
        ],
    }


def build_laser_cache(out_path, progress=True) -> Dict[str, Tuple]:
    """Detect the laser dot in every board+dot frame and cache it to ``.npz``.

    Stored as ``medium/filename -> (x, y, n_pixels, contrast)``.
    """
    found = {}
    for medium, paths in laser_frames().items():
        if progress:
            print(f"{medium}: {len(paths)} frames", flush=True)
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            dot = detect_laser_dot(image)
            if dot is not None:
                found[f"{medium}/{path.name}"] = np.asarray(dot, dtype=float)
            if progress:
                print(
                    f"  {path.name} "
                    + ("--" if dot is None else f"({dot[0]:.0f},{dot[1]:.0f}) c={dot[3]:.0f}"),
                    flush=True,
                )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **found)
    if progress:
        print(f"cached {len(found)} laser dots -> {out_path}", flush=True)
    return found


def fit_line(points) -> Tuple[np.ndarray, np.ndarray]:
    """Total-least-squares 3D line through points: ``(origin_at_z0, unit_direction)``.

    The direction is the first principal component; the origin is the point where
    that line crosses ``z = 0``, matching the FishSense laser convention of a
    position plus an axis.
    """
    points = np.asarray(points, dtype=float)
    centroid = points.mean(axis=0)
    direction = np.linalg.svd(points - centroid)[2][0]
    if direction[2] < 0:
        direction = -direction
    origin = centroid - direction * (centroid[2] / direction[2])
    return origin, direction


# --------------------------------------------------------------------------
# Raw-space extraction
#
# The TG-6 applies an in-camera lens correction to its JPEGs. Measured against
# raw on this dataset it is a radial warp of about 1 px median, and the
# displacement correlates with image radius at r ~ +0.85 -- so it grows toward
# the frame edges, which is exactly where the refraction signal lives. JPEG
# geometry is therefore not the sensor's geometry, and a refraction study has to
# work in raw.
#
# `fishsense_core.image.LinearRawImage` is the decode FishSense already uses for
# the laser detector: linear 16-bit BGR, no CLAHE, no EXIF rotation, in sensor
# coordinates. Its `bayer_excess` gives per-super-cell green/red excess computed
# before demosaic, which is what makes a saturated dot recoverable.
#
# Sensor coordinates are the 4014x3016 visible area; the JPEG is the (8, 8)
# crop of it, so `jpeg + 8 == sensor` for position (but NOT for the warp above).
# --------------------------------------------------------------------------

JPEG_CROP_OFFSET = (8, 8)

#: `bayer_excess` is computed at half resolution and lifted back. "repeat" puts
#: each super-cell at its block's top-left, "bilinear" at the block centroid;
#: they differ by up to a full pixel, and fishsense-core prices 1 px of laser-dot
#: error at 0.75% length error. Geometry wins here -- "repeat" exists for
#: checkpoint compatibility, which we do not need.
BAYER_UPSAMPLE = "bilinear"


def load_raw(path, bayer_upsample=BAYER_UPSAMPLE):
    """Open the ORF beside a JPEG as a `LinearRawImage`.

    The import is function-local on purpose. This is the only function in the
    repo that opens a raw file, and the only one needing `fishsense-core`, which
    is why that lives in the optional `raw` dependency group rather than in the
    dependencies proper. Keeping the import here means the module, the analysis,
    the figures and the tests all load with it absent.
    """
    try:
        from fishsense_core.image.linear_raw_image import LinearRawImage
    except ModuleNotFoundError as exc:  # pragma: no cover - platform dependent
        raise ModuleNotFoundError(
            "load_raw needs fishsense-core, which is in the optional 'raw' "
            "dependency group: install it with `uv sync --group raw`. It exists "
            "only to rebuild the caches in data/ from the original ORF "
            "photographs, which are not in this repo, and upstream publishes "
            "Linux x86_64 wheels only. Every analysis here runs from the "
            "committed caches without it."
        ) from exc

    return LinearRawImage(Path(path).with_suffix(".ORF"), bayer_upsample=bayer_upsample)


def raw_gray(raw, low_percentile=1.0, high_percentile=90.0, gamma=2.2) -> np.ndarray:
    """Tone-map a linear raw to uint8 for corner detection.

    The chessboard detector expects display-referred contrast. Linear sensor data
    scaled by its maximum crushes the board: one specular highlight sets the
    scale and the squares collapse into a few levels. So the data is stretched
    between two percentiles and gamma-encoded. Monotonic and pixel-wise, so no
    pixel moves and corner geometry is untouched.

    Both ends matter, and the defaults were measured rather than guessed. Without
    a low floor, and with the high end at the 99.5th percentile, the garage's
    blown-out windows hijack the scale and board detection drops to 36/64 frames
    against 62/64 from the JPEGs. A 1st-percentile floor with the high end at the
    90th recovers 14/14 on the frames tested, and costs nothing underwater (the
    pool scores identically at every percentile tried between 85 and 99.5).

    The high end clips, which is not invertible, so it can in principle bias
    subpixel corners near bright regions -- hence choosing the loosest setting
    that actually recovers the frames rather than the most aggressive one.
    """
    green = raw.data[..., 1].astype(np.float32)
    low, high = np.percentile(green, [low_percentile, high_percentile])
    scaled = np.clip((green - low) / max(float(high - low), 1.0), 0.0, 1.0)
    return ((scaled ** (1.0 / gamma)) * 255.0).astype(np.uint8)


def detect_laser_dot_raw(raw, seed=None, half=40, threshold=0.5):
    """Sub-pixel laser centroid from the Bayer green-excess channel.

    Returns ``(x, y, peak)`` in sensor coordinates, or None.

    The green *excess* is what survives both media: it does not saturate where
    the raw green channel clips (the in-air frames are 30% clipped at the dot,
    having been shot at f/2 against f/8 underwater), and it does not collapse
    against blue-green pool water the way a plain green threshold does.

    `seed` is a rough sensor-coordinate location to search around; without one
    the global maximum of the excess channel is used, which is only safe when
    nothing else in frame is strongly green.
    """
    excess = raw.bayer_excess[..., 0].astype(np.float64)

    if seed is None:
        cy, cx = np.unravel_index(int(np.argmax(excess)), excess.shape)
    else:
        cx, cy = int(round(seed[0])), int(round(seed[1]))

    y0, y1 = max(0, cy - half), min(excess.shape[0], cy + half + 1)
    x0, x1 = max(0, cx - half), min(excess.shape[1], cx + half + 1)
    window = excess[y0:y1, x0:x1]
    if window.size == 0:
        return None

    peak = float(window.max())
    if peak <= 0:
        return None

    weight = np.clip(window - threshold * peak, 0.0, None)
    if weight.sum() <= 0:
        return None

    ys, xs = np.mgrid[0 : window.shape[0], 0 : window.shape[1]]
    return (
        float(x0 + (weight * xs).sum() / weight.sum()),
        float(y0 + (weight * ys).sum() / weight.sum()),
        peak,
    )


def build_raw_caches(
    corner_out="data/board_corners_raw.npz",
    laser_out="data/laser_dots_raw.npz",
    sessions=("garage_lens", "garage_laser", "pool_lens"),
    jpeg_laser_cache="data/laser_dots.npz",
    progress=True,
):
    """Re-extract boards and laser dots in raw sensor coordinates.

    Kept separate from the JPEG caches rather than replacing them: the difference
    between the two is a direct measurement of the TG-6's in-camera rectification,
    which is worth keeping as a cross-check.

    Laser search is seeded from the JPEG detections (offset into sensor
    coordinates) so it does not have to trust a global green maximum.
    """
    seeds = {}
    if jpeg_laser_cache and Path(jpeg_laser_cache).exists():
        jpeg = load_corner_cache(jpeg_laser_cache)
        for key, value in jpeg.items():
            seeds[Path(key).name] = (
                value[0] + JPEG_CROP_OFFSET[0],
                value[1] + JPEG_CROP_OFFSET[1],
            )

    laser_wanted = {p.name for paths in laser_frames().values() for p in paths}
    corners_found, dots_found = {}, {}

    for session in sessions:
        paths = session_frames(session)
        if progress:
            print(f"{session}: {len(paths)} frames", flush=True)
        for path in paths:
            try:
                raw = load_raw(path)
                corners = detect_corners(raw_gray(raw))
            except Exception as exc:  # a corrupt or missing ORF should not stop the pass
                if progress:
                    print(f"  {path.name} RAW-FAIL {exc}", flush=True)
                continue

            if corners is not None:
                corners_found[f"{session}/{path.name}"] = corners

            dot = None
            if path.name in laser_wanted:
                dot = detect_laser_dot_raw(raw, seed=seeds.get(path.name))
                if dot is not None:
                    medium = "air" if session.startswith("garage") else "water"
                    dots_found[f"{medium}/{path.name}"] = np.asarray(dot, dtype=float)

            if progress:
                mark = "ok" if corners is not None else "--"
                extra = "" if dot is None else f" dot=({dot[0]:.1f},{dot[1]:.1f})"
                print(f"  {path.name} {mark}{extra}", flush=True)

    for out, found, label in (
        (corner_out, corners_found, "boards"),
        (laser_out, dots_found, "laser dots"),
    ):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **found)
        if progress:
            print(f"cached {len(found)} {label} -> {out}", flush=True)

    return corners_found, dots_found
