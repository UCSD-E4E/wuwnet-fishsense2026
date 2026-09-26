"""Separate the camera's fx/fy from a printed checkerboard's own, by rolling the camera.

A calibration's `fx/fy` is the sensor's times the target's: the target term is
fixed to the board, so it inverts when the camera is rolled 90 degrees relative to
it, while the sensor term does not. With the board shot at both rolls,

    sensor = sqrt(r_horizontal * r_vertical)     board = sqrt(r_horizontal / r_vertical)

where "horizontal" means the board's long side runs across the image. The roll is
measured from each frame's detected grid, never from EXIF: the TG-6's orientation
sensor cannot tell rolls apart when the camera looks down, and on 2026-09-25 it
tagged every board frame the same.

The 2026-09-22 roll test did this across sessions (2024 upright, 2026 rolled) and
found the sensor at 1.0009 and this board's print at about +0.7 %.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from .dataset import PATTERN, board_object_points, load_raw, raw_gray

SENSOR = (4014, 3016)


def detect_boards(frames: Path, cache: Path, log=print) -> Dict[str, np.ndarray]:
    """Inner corners of every raw frame the full board is found in; cached by frame."""
    found: Dict[str, np.ndarray] = {}
    tried = set()
    if Path(cache).exists():
        z = np.load(cache)
        found = {k: z[k] for k in z.files if not k.startswith("_missed/")}
        tried = set(found) | {k[len("_missed/"):] for k in z.files if k.startswith("_missed/")}
    missed = sorted(tried - set(found))
    for path in sorted(Path(frames).glob("*.ORF")):
        if path.stem in tried:
            continue
        gray = raw_gray(load_raw(path))
        ok, corners = cv2.findChessboardCorners(
            gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            found[path.stem] = corners.reshape(-1, 2).astype(np.float32)
        else:
            missed.append(path.stem)
        log(f"  {path.stem}: {'board found' if ok else 'no full board'}")
        np.savez_compressed(cache, **found, **{f"_missed/{m}": np.zeros(1) for m in missed})
    return found


def long_side_angle(corners: np.ndarray) -> float:
    """Direction of the board's long side in the image, degrees in [0, 180)."""
    cols = PATTERN[0]
    d = corners[cols - 1] - corners[0]
    return float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)


def roll_of(corners: np.ndarray) -> str:
    a = long_side_angle(corners)
    return "horizontal" if min(a, 180.0 - a) < 45.0 else "vertical"


def calibrate(views) -> tuple:
    """``(K, dist, rms)`` from a list of corner arrays; distortion fitted freely."""
    objp = board_object_points().astype(np.float32)
    guess = np.array([[2900.0, 0, SENSOR[0] / 2], [0, 2900.0, SENSOR[1] / 2], [0, 0, 1.0]])
    rms, K, dist, _, _ = cv2.calibrateCamera(
        [objp] * len(views), [v.reshape(-1, 1, 2) for v in views], SENSOR, guess, None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    return K, dist, rms


def split(views_by_roll: Dict[str, list]) -> Optional[dict]:
    """Per-roll calibrations and the sensor/board decomposition, if both rolls exist."""
    out = {}
    for roll, views in views_by_roll.items():
        if len(views) >= 4:
            K, dist, rms = calibrate(views)
            out[roll] = {"K": K, "dist": dist, "rms": rms, "n": len(views), "ratio": K[0, 0] / K[1, 1]}
    if "horizontal" in out and "vertical" in out:
        h, v = out["horizontal"]["ratio"], out["vertical"]["ratio"]
        out["sensor"] = float(np.sqrt(h * v))
        out["board"] = float(np.sqrt(h / v))
    return out
