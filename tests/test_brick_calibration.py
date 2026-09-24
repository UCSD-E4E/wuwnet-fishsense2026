"""The corner target's calibration path, and the two conditions it needs."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from fishsense_wuwnet.brick_calibration import SENSOR, reject_views
from fishsense_wuwnet.calibration_model import MODEL_IO, _ldraw_lines, brick_corners, exposed_faces, load_bricks, visible_corners
from fishsense_wuwnet.target_design import compare
from fishsense_wuwnet.target_render import look_at

CORNER = MODEL_IO
TRUTH = np.array([[2950.0, 0, 2007.0], [0, 2918.2, 1508.0], [0, 0, 1.0]])  # 1.09 % anisotropic
OUTSIDE = np.radians(np.linspace(150.0, 300.0, 8))  # the half-space outside the corner


def _aim():
    c = np.vstack([brick_corners(b) for b in load_bricks(CORNER)])
    return (c.min(axis=0) + c.max(axis=0)) / 2.0


def _eye(aim, azimuth, radius, elevation):
    return aim + radius * np.array([np.cos(azimuth) * np.cos(elevation),
                                    np.sin(azimuth) * np.cos(elevation), np.sin(elevation)])


def test_compare_lines_up_a_turned_export_and_finds_a_recoloured_brick(tmp_path):
    """Studio places and turns the model freely; only the bricks themselves should matter."""
    turn = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])  # quarter turn about LDraw's vertical
    out, first = [], True
    for line in _ldraw_lines(CORNER):
        if not line.startswith("1 "):
            out.append(line)
            continue
        f = line.split()
        p = turn @ np.array(list(map(float, f[2:5]))) + [500.0, 0.0, -300.0]
        r = turn @ np.array(list(map(float, f[5:14]))).reshape(3, 3)
        colour = "0" if first else f[1]
        first = False
        out.append(" ".join(["1", colour, *(f"{v:.1f}" for v in p), *(f"{v:g}" for v in r.ravel()), f[14]]))
    built = tmp_path / "built.ldr"
    built.write_text("\n".join(out) + "\n")

    assert compare(CORNER, CORNER)["matched"] == 35
    diff = compare(built, CORNER)
    assert diff["matched"] == 35 and not diff["missing"] and not diff["extra"]
    assert len(diff["recoloured"]) == 1 and diff["recoloured"][0][3] == 0


def test_one_mismatched_view_is_dropped_and_the_rest_calibrate():
    """The failure seen from inside the corner: a view whose faces are paired with
    the wrong bricks. Left in, it took fx 43 % off.

    Simulated by pairing each face's image corners with the next face's object
    corners. Not by reflecting the geometry: an L reflected across one wall is the
    same L turned a quarter, so a pose fits it exactly and nothing is wrong to find.
    """
    aim = _aim()
    faces = exposed_faces(CORNER, sides_only=True)
    rng = np.random.default_rng(0)
    names, obj, img = [], [], []
    for k, azimuth in enumerate(OUTSIDE):
        eye = _eye(aim, azimuth, rng.uniform(130, 310), rng.uniform(-0.2, 0.5))
        R, t = look_at(eye, aim)
        seen = [f for f in faces if f.normal @ (eye - f.centre) > 0]
        o = np.vstack([visible_corners(f) for f in seen]).astype(np.float32)
        i = cv2.projectPoints(o, cv2.Rodrigues(R)[0], t, TRUTH, None)[0].reshape(-1, 2)
        i = (i + rng.normal(0, 0.3, i.shape)).astype(np.float32)
        if k == 3:
            o = np.roll(o.reshape(-1, 4, 3), 1, axis=0).reshape(-1, 3)  # wrong bricks
        names.append(f"view{k}")
        obj.append(o)
        img.append(i)

    guess = TRUTH.copy()
    guess[0, 0] = guess[1, 1] = 2900.0
    result = reject_views(names, obj, img, guess, np.zeros(5))
    assert list(result.dropped) == ["view3"]
    assert result.K[0, 0] / result.K[1, 1] == pytest.approx(TRUTH[0, 0] / TRUTH[1, 1], rel=1e-3)
    assert result.K[0, 0] == pytest.approx(TRUTH[0, 0], rel=5e-3)


def test_the_corner_calibrates_like_the_tower_from_outside_and_close():
    """Rendered at the TG-6's resolution, shot from outside at 13-31 cm.

    Measured when the corner was designed: fx/fy +0.04 %, fx +0.24 %, against the
    tower's -0.01 % and +0.34 % at its own 30-70 cm. At 30-70 cm the corner's fx
    came back 4-7 % off -- it is too small there -- which is why the distance is
    part of the design, not a detail of the test.
    """
    from fishsense_wuwnet.target_detect import detect_quads, match
    from fishsense_wuwnet.target_render import render

    aim = _aim()
    rng = np.random.default_rng(3)
    names, obj, img = [], [], []
    for k, azimuth in enumerate(OUTSIDE):
        eye = _eye(aim, azimuth, rng.uniform(130, 310), rng.uniform(-0.2, 0.5))
        R, t = look_at(eye, aim)
        view = render(TRUTH, R, t, size=SENSOR, path=CORNER, seed=0)
        found = match(detect_quads(view.image), TRUTH, path=CORNER)
        if len(found) >= 24:
            names.append(f"view{k}")
            obj.append(found.object_points.astype(np.float32))
            img.append(found.image_points.astype(np.float32))
    assert len(obj) >= 6

    guess = TRUTH.copy()
    guess[0, 0] = guess[1, 1] = TRUTH[0, 0] * 1.15
    result = reject_views(names, obj, img, guess, np.zeros(5))
    assert result.rms < 2.0
    assert result.K[0, 0] / result.K[1, 1] == pytest.approx(TRUTH[0, 0] / TRUTH[1, 1], rel=1e-3)
    assert result.K[0, 0] == pytest.approx(TRUTH[0, 0], rel=5e-3)
