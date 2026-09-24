"""How each brick colour of the photographed build actually photographs, face by face.

    uv run python refraction_analysis/measure_brick_colours.py

Needs the raw frames (`/home/chris/data/Lego Model Horizontal`, not committed).
Writes `data/brick_colour_samples.csv`, the evidence behind `COLOUR_CLASS`.

Every visible face of the model is projected into the frame with the view's pose
and sampled inside its outline, so the label comes from the model rather than
from the detector -- which is what lets this count the faces the detector
*misses*, black above all. Pose comes from the committed matched correspondences,
intrinsics from the in-air checkerboard.
"""

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "refraction_analysis"))

from fishsense_wuwnet.calibration_model import BRICK_MM, MODEL_IO, exposed_faces  # noqa: E402
from fishsense_wuwnet.dataset import load_raw, raw_colour  # noqa: E402
from fishsense_wuwnet.target_detect import classify_colours  # noqa: E402
from make_target_figures import LEGO_CACHE, _lego  # noqa: E402

FRAMES = Path("/home/chris/data/Lego Model Horizontal")
SENSOR = (4014, 3016)
OUT = REPO / "data" / "brick_colour_samples.csv"

#: The cached correspondences predate the fix that puts a brick's body *below* its
#: LDraw origin; their object points sit half a brick high. A uniform shift like
#: this only moves the pose, so it is undone here rather than by regenerating.
CACHE_OFFSET = np.array([0.0, 0.0, BRICK_MM / 2.0])


def main():
    lego = _lego()
    K, dist = lego["board_K"], lego["board_d"]
    cache = np.load(LEGO_CACHE)
    faces = exposed_faces(MODEL_IO, sides_only=True)
    frames = sorted(FRAMES.glob("*.ORF"))
    rows = []
    for index, frame in enumerate(frames[:21:2]):
        if f"obj/{frame.stem}" not in cache.files:
            continue
        obj = cache[f"obj/{frame.stem}"] - CACHE_OFFSET
        ok, rvec, tvec, _ = cv2.solvePnPRansac(obj, cache[f"img/{frame.stem}"], K, dist,
                                               reprojectionError=12.0)
        if not ok:
            continue
        camera = (-cv2.Rodrigues(rvec)[0].T @ tvec).ravel()
        image = raw_colour(load_raw(frame))
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 0] *= 100 / 255
        lab[..., 1:] -= 128
        labels = classify_colours(image, chroma_blur=2.0)
        for face in faces:
            ray = camera - face.centre
            cos_view = float(face.normal @ ray / np.linalg.norm(ray))
            if cos_view < 0.35:
                continue
            inner = face.centre + 0.6 * (face.corners - face.centre)  # clear of the seams
            px = cv2.projectPoints(inner.astype(np.float32), rvec, tvec, K, dist)[0].reshape(-1, 2)
            if (px < 0).any() or (px[:, 0] >= SENSOR[0]).any() or (px[:, 1] >= SENSOR[1]).any():
                continue
            mask = np.zeros(SENSOR[::-1], np.uint8)
            cv2.fillConvexPoly(mask, px.astype(np.int32), 1)
            if mask.sum() < 150:
                continue
            inside = mask.astype(bool)
            L, a, b = np.median(lab[inside], axis=0)
            label = Counter(labels[inside].tolist()).most_common(1)[0][0]
            rows.append((face.colour, L, a, b, cos_view, int(mask.sum()), label, index))
        print(f"{frame.stem}: {len(rows)} faces so far", flush=True)

    header = OUT.read_text().split("model_colour")[0] if OUT.exists() else ""
    with open(OUT, "w") as f:
        f.write(header + "model_colour,L,a,b,cos_view,pixels,classifier_label,frame_index\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.2f},{r[2]:.2f},{r[3]:.2f},{r[4]:.3f},{r[5]},{r[6]},{r[7]}\n")
    print(OUT, len(rows))


if __name__ == "__main__":
    main()
