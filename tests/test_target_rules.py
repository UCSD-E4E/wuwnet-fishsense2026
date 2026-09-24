"""The design rules: that they measure what they claim, and that the corner design passes."""

from pathlib import Path

import numpy as np
import pytest

from fishsense_wuwnet.calibration_model import MODEL_IO, load_bricks
from fishsense_wuwnet.target_rules import check, load_inventory, window_ambiguity

CORNER = Path(__file__).resolve().parent.parent / "calibration_model" / "corner_10698.ldr"


def _by_rule(path):
    return {r.rule.split()[0]: r for r in check(path)}


def test_the_inventory_is_the_whole_box():
    assert sum(load_inventory().values()) == 790


def test_the_rules_reproduce_what_was_measured_on_the_photographed_build():
    """The joint and window numbers were measured independently before these rules
    existed -- 0 of 407 joints shared a colour, 19 % of windows were ambiguous, worst
    five times -- so agreeing with them is a check on the rules, not on the build."""
    rules = _by_rule(MODEL_IO)
    assert rules["4"].passed and "0 of 407" in rules["4"].detail
    placements, ambiguous, worst = window_ambiguity(MODEL_IO)
    assert ambiguous == pytest.approx(0.19, abs=0.01) and worst == 5
    assert rules["6"].passed
    # And it was not built from one box, nor in readable colours: 33 black bricks.
    assert not rules["1"].passed and not rules["3"].passed


def test_the_corner_design_passes_every_rule():
    failed = [f"{r.rule}: {r.detail}" for r in check(CORNER) if not r.passed]
    assert not failed, failed


def _variant(tmp_path, edit):
    lines = CORNER.read_text().splitlines()
    parts = [i for i, line in enumerate(lines) if line.startswith("1 ")]
    lines = edit(lines, parts)
    out = tmp_path / "variant.ldr"
    out.write_text("\n".join(lines) + "\n")
    return out


def test_an_unreadable_colour_is_caught(tmp_path):
    def blacken(lines, parts):
        fields = lines[parts[0]].split()
        fields[1] = "0"
        lines[parts[0]] = " ".join(fields)
        return lines

    assert not _by_rule(_variant(tmp_path, blacken))["3"].passed


def test_a_same_group_joint_is_caught(tmp_path):
    """Recolour every brick red: every joint then joins one group."""

    def all_red(lines, parts):
        for i in parts:
            fields = lines[i].split()
            fields[1] = "4"
            lines[i] = " ".join(fields)
        return lines

    rules = _by_rule(_variant(tmp_path, all_red))
    assert not rules["4"].passed and not rules["5"].passed


def test_the_corner_design_is_detected_and_posed():
    """The same renderer and detector the as-built target is tested with."""
    cv2 = pytest.importorskip("cv2")
    from fishsense_wuwnet.target_detect import detect_quads, match
    from fishsense_wuwnet.target_render import look_at, render

    w, h = 2000, 1500
    K = np.array([[1470.0, 0, w / 2], [0, 1470.0, h / 2], [0, 0, 1]])
    bricks = load_bricks(CORNER)
    aim = np.mean([b.centre for b in bricks], axis=0)
    for eye in [(-230, -230, 140), (-320, -90, 110), (-90, -320, 110)]:
        R, t = look_at(np.array(eye, float) + aim * [1, 1, 0], aim)
        view = render(K, R, t, size=(w, h), path=CORNER, seed=1)
        c = match(detect_quads(view.image), K, path=CORNER)
        assert len(c) // 4 >= 20
        ok, rvec, _ = cv2.solvePnP(c.object_points, c.image_points, K, None)
        err = np.degrees(np.arccos(np.clip((np.trace(cv2.Rodrigues(rvec)[0] @ R.T) - 1) / 2, -1, 1)))
        assert err < 1.0
