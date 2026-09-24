"""Calibrate the TG-6 from raw photographs of a brick target.

The path the corner target is shot through: detect on each raw frame, cache the
correspondences, calibrate with bad views rejected, and say which views broke
the two conditions the corner needs.

Those conditions came out of testing the corner against the suite, not from
reasoning about it:

* **From outside.** Seen from the concave side a lone wall can match as its own
  mirror image -- on one rendered view the pose came back 176 degrees out -- and a
  single such view took fx 43 % off. Views are therefore rejected by residual
  (`reject_views`), and any view that matched only one wall is flagged, since that
  is the signature.
* **Close.** A 64 mm target at 30-70 cm spans too small an angle to separate focal
  length from distance: fx came back 4-7 % off while fx/fy held. At 13-31 cm it
  matched the tower (fx/fy +0.04 %, fx +0.24 %). Views beyond `FAR_WIDTHS` target
  widths are flagged.

Distortion is held fixed, as for the tower: a small target spans too little of
the field to condition a radial series. It comes from a checkerboard shot in the
same session if one is given, and otherwise from the committed garage board --
the radial term is a property of the lens, and it matched across sessions to
0.002 in k1.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .dataset import (
    PATTERN,
    board_object_points,
    cache_by_session,
    load_corner_cache,
    load_raw,
    raw_colour,
    raw_gray,
)
from .target_detect import detect_quads, match

SENSOR = (4014, 3016)
DATA = Path(__file__).resolve().parent.parent / "data"
#: Furthest a view should be, in multiples of the target's width. The corner
#: (64 mm) calibrated like the tower out to 31 cm and lost fx beyond; the tower
#: (144 mm) was fine to 70 cm. Both are about five widths.
FAR_WIDTHS = 5.0
MIN_FACES = 6

#: Intrinsics only; distortion is supplied, not fitted (see module docstring).
FIX_DISTORTION = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
                  | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_TANGENT_DIST)


def _guess() -> np.ndarray:
    return np.array([[2900.0, 0, SENSOR[0] / 2], [0, 2900.0, SENSOR[1] / 2], [0, 0, 1.0]])


def board_calibration(frames: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, str]:
    """``(K, dist, source)`` from a checkerboard: a raw folder if given, else the garage cache."""
    objp = board_object_points().astype(np.float32)
    if frames is None:
        garage = cache_by_session(load_corner_cache(DATA / "board_corners_raw.npz"), "garage_lens")
        images = [garage[n].reshape(-1, 1, 2).astype(np.float32) for n in sorted(garage)]
        source = f"garage board, {len(images)} frames (2024)"
    else:
        images = []
        for path in sorted(Path(frames).glob("*.ORF")):
            gray = raw_gray(load_raw(path))
            ok, corners = cv2.findChessboardCorners(
                gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ok:
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
                images.append(corners.astype(np.float32))
        source = f"checkerboard {Path(frames).name}, {len(images)} frames"
    if len(images) < 5:
        raise ValueError(f"only {len(images)} checkerboard frames detected")
    _, K, dist, _, _ = cv2.calibrateCamera([objp] * len(images), images, SENSOR, _guess(), None,
                                           flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    return K, dist, source


def _model_digest(model: Path) -> str:
    return hashlib.sha256(Path(model).read_bytes()).hexdigest()[:16]


def detect_folder(frames: Path, model: Path, cache: Path, K: np.ndarray, log=print) -> Dict[str, dict]:
    """Correspondences for every raw frame, cached; the cache is dropped if the model changes.

    Detection on raw takes several seconds a frame, so a rerun after changing only
    the calibration should not repeat it. The model's hash is stored with the
    cache, because correspondences are only valid against the model they were
    matched to.
    """
    digest = _model_digest(model)
    stored: Dict[str, dict] = {}
    if Path(cache).exists():
        z = np.load(cache)
        if "model" in z.files and str(z["model"]) == digest:
            for key in z.files:
                if key.startswith("obj/"):
                    stem = key[4:]
                    stored[stem] = {"obj": z[f"obj/{stem}"], "img": z[f"img/{stem}"],
                                    "walls": int(z[f"walls/{stem}"])}
    for path in sorted(Path(frames).glob("*.ORF")):
        if path.stem in stored:
            continue
        found = match(detect_quads(raw_colour(load_raw(path)), open_px=5, chroma_blur=2.0), K, path=model)
        walls = len({tuple(np.round(f.normal, 3)) for f in found.faces})
        stored[path.stem] = {"obj": found.object_points.astype(np.float32),
                             "img": found.image_points.astype(np.float32), "walls": walls}
        log(f"  {path.stem}: {len(found) // 4} faces on {walls} wall(s)")
        arrays = {"model": np.array(digest)}
        for stem, v in stored.items():
            arrays.update({f"obj/{stem}": v["obj"], f"img/{stem}": v["img"],
                           f"walls/{stem}": np.array(v["walls"])})
        np.savez_compressed(cache, **arrays)
    return stored


@dataclass
class Calibration:
    K: np.ndarray
    dist: np.ndarray
    rms: float
    views: List[str]
    residual: Dict[str, float]  # per-view rms, px, final fit
    distance: Dict[str, float]  # camera to target, mm
    dropped: Dict[str, float] = field(default_factory=dict)  # view -> its residual when dropped


def reject_views(names, obj, img, K, dist, floor_px: float = 3.0) -> Calibration:
    """Calibrate, drop faces matched to the wrong brick and views gone wrong, repeat.

    Two levels, because the brick target fails at both. A face is dropped when
    any of its corners is off by more than three times the median corner error
    (never below 1 px): a face matched to the wrong brick is wrong at all four
    corners, not at one. On the photographed tower this drops about 30 % of faces
    and takes the residual from 14 px to 5 px -- and fx against the checkerboard
    from -2.25 % to -0.61 %, so leaving them in is not a small matter.

    A view is then dropped if it has fewer than `MIN_FACES` faces left, or if its
    residual is over three times the median view's (never below `floor_px`): the
    mirror-image match seen from inside the corner is wrong everywhere at once.
    """
    names, obj, img = list(names), list(obj), list(img)
    dropped: Dict[str, float] = {}
    while True:
        rms, K_fit, _, rvecs, tvecs = cv2.calibrateCamera(obj, img, SENSOR, K.copy(), dist.copy(),
                                                          flags=FIX_DISTORTION)
        errors = [np.linalg.norm(cv2.projectPoints(o, r, t, K_fit, dist)[0].reshape(-1, 2) - i, axis=1)
                  for o, i, r, t in zip(obj, img, rvecs, tvecs)]
        per = [float(np.sqrt(np.mean(e ** 2))) for e in errors]
        face_limit = max(1.0, 3.0 * float(np.median(np.concatenate(errors))))
        view_limit = max(3.0 * float(np.median(per)), floor_px)
        changed = False
        for k in reversed(range(len(obj))):
            keep = np.repeat(errors[k].reshape(-1, 4).max(axis=1) <= face_limit, 4)
            if per[k] > view_limit or keep.sum() < 4 * MIN_FACES:
                if len(obj) - 1 >= 4:
                    dropped[names[k]] = per[k]
                    del names[k], obj[k], img[k]
                    changed = True
                continue
            if not keep.all():
                obj[k], img[k] = obj[k][keep], img[k][keep]
                changed = True
        if not changed:
            break
    distance = {n: float(np.linalg.norm(t)) for n, t in zip(names, tvecs)}
    return Calibration(K_fit, dist, float(rms), names, dict(zip(names, per)), distance, dropped)


def calibrate_folder(frames: Path, model: Path, checkerboard: Optional[Path] = None,
                     cache: Optional[Path] = None, log=print):
    """Everything a shoot needs: detection, calibration, and the checks on each view."""
    board_K, dist, source = board_calibration(checkerboard)
    cache = Path(cache) if cache else DATA / f"{Path(frames).name.replace(' ', '_').lower()}_raw.npz"
    log(f"distortion from {source}; correspondences cached in {cache}")
    views = detect_folder(frames, model, cache, board_K, log=log)
    usable = {n: v for n, v in views.items() if len(v["obj"]) >= 4 * MIN_FACES}
    if len(usable) < 4:
        raise ValueError(f"only {len(usable)} views with {MIN_FACES}+ faces; need 4")
    names = sorted(usable)
    result = reject_views(names, [usable[n]["obj"] for n in names], [usable[n]["img"] for n in names],
                          board_K, dist)
    from .calibration_model import brick_corners, load_bricks

    corners = np.vstack([brick_corners(b) for b in load_bricks(model) if not b.is_plate])
    far = FAR_WIDTHS * float(np.ptp(corners[:, :2], axis=0).max())
    flags = {n: [] for n in views}
    for n, v in views.items():
        if n not in usable:
            flags[n].append(f"only {len(v['obj']) // 4} faces")
        elif v["walls"] < 2:
            flags[n].append("one wall only: fine from outside, a mirror risk from inside")
        if n in result.distance and result.distance[n] > far:
            flags[n].append(f"too far: reshoot within {far / 10:.0f} cm")
        if n in result.dropped:
            flags[n].append(f"dropped: residual {result.dropped[n]:.1f} px")
    return result, board_K, source, views, flags
