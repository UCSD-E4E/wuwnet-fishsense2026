"""Calibrate the camera by structure-from-motion: no target, just photographs.

    uv sync --group raw --group sfm
    uv run python refraction_analysis/sfm_calibration.py ~/data/2026-09-25/LEGO

Self-calibrating SfM (COLMAP) solves the scene's 3D structure and the camera
together from features matched across views of any static, textured scene. The
question it answers here is whether a citizen scientist needs a calibration target
at all, or only a few dozen photos of an ordinary room.

Raw frames are converted without the TG-6's in-camera lens correction, so the
intrinsics are the sensor's. COLMAP is started from its default focal length guess
(1.2 x the image width -- about 60 % long for this camera, since the frames carry no
EXIF focal length), so the focal length it reports is recovered, not supplied.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fishsense_wuwnet.dataset import load_raw, raw_colour  # noqa: E402


def to_images(frames: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for raw in sorted(frames.glob("*.ORF")):
        image = out / f"{raw.stem}.jpg"
        if not image.exists():
            cv2.imwrite(str(image), raw_colour(load_raw(raw)), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out


def run(images: Path, work: Path, model: str, refine_pp: bool):
    """One shared camera of the given COLMAP model; the largest reconstruction."""
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    if db.exists():
        db.unlink()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = model
    pycolmap.extract_features(db, images, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
    pycolmap.match_exhaustive(db)
    options = pycolmap.IncrementalPipelineOptions()
    options.ba_refine_principal_point = refine_pp
    options.ba_refine_focal_length = True
    options.ba_refine_extra_params = True
    maps = pycolmap.incremental_mapping(db, images, work / "sparse", options=options)
    if not maps:
        return None
    return max(maps.values(), key=lambda r: r.num_reg_images())


def main():
    frames = Path(sys.argv[1]).expanduser()
    work = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp") / f"sfm_{frames.name}"
    images = to_images(frames, work / "images")
    n_images = len(list(images.glob("*.jpg")))
    for model, refine_pp in (("RADIAL", False), ("RADIAL", True), ("OPENCV", True)):
        rec = run(images, work / f"{model}_{'pp' if refine_pp else 'centre'}", model, refine_pp)
        if rec is None:
            print(f"{model:7s} pp {'free' if refine_pp else 'fixed'}: no reconstruction")
            continue
        cam = next(iter(rec.cameras.values()))
        p = cam.params
        if model == "RADIAL":  # f, cx, cy, k1, k2
            f, fx_fy, cx, cy, k1, k2 = p[0], 1.0, p[1], p[2], p[3], p[4]
        else:  # fx, fy, cx, cy, k1, k2, p1, p2
            f, fx_fy, cx, cy, k1, k2 = np.sqrt(p[0] * p[1]), p[0] / p[1], p[2], p[3], p[4], p[5]
        print(f"{model:7s} pp {'free ' if refine_pp else 'fixed'}: {rec.num_reg_images()}/{n_images} images, "
              f"{rec.num_points3D()} points, reproj {rec.compute_mean_reprojection_error():.2f} px | "
              f"f {f:.1f}  fx/fy {fx_fy:.4f}  cx {cx:.0f}  cy {cy:.0f}  k1 {k1:+.4f}  k2 {k2:+.4f}", flush=True)


if __name__ == "__main__":
    main()
