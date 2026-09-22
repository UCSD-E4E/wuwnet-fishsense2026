# fx/fy anisotropy: the roll test (deferred to camera-ready)

**Question.** §8 reports a ~1 % `fx/fy` and cannot say whether it is the *sensor*
(fixed in sensor coordinates, survives a 90° camera roll) or the *target* (fixed
to the board, inverts under a 90° roll). Chris shot the data to settle it; the
computation is deferred past the WUWNet submission.

## The data (all raw ORF, in `/home/chris/data`, not committed)

| folder | frames | roll |
|---|---|---|
| `Lego Model Horizontal`  | 21 | upright |
| `Lego Model Vertical`    | 14 | camera rolled 90° |
| `Checkerboard Vertical`  | 19 | camera rolled 90° |

Missing for a fully clean test: **Checkerboard Horizontal, same session.** Shoot
it for camera-ready — it makes the checkerboard a same-session roll pair too.

## The method

A recovered ratio factors as `sensor × target^(±1)`: the target's contribution
inverts under a 90° roll while the sensor's does not. From the same target at
both rolls:

    sensor = sqrt(ratio_horizontal · ratio_vertical)   # survives roll
    target = sqrt(ratio_horizontal / ratio_vertical)   # inverts under roll

Work in **raw** (the TG-6's JPEG lens-correction is not the sensor's geometry).
Use an **isotropic** board object model for the checkerboard so its assumed pitch
injects no anisotropy. Fix distortion from the checkerboard; solve K only for the
LEGO (3D target, non-planar solve is slow — cap outlier-rejection iterations).

## Partial results (2026-09-21, incomplete)

| target / roll | views | rms px | fx/fy |
|---|---|---|---|
| Checkerboard vertical (isotropic board) | 18 | 0.65 | **1.00801** |
| LEGO horizontal | 21 | 5.4 | **0.99773** |
| LEGO vertical | — | — | *not yet computed* |
| old garage checkerboard (2024, ~horizontal, anisotropic pitch) | 83 | 0.46 | 0.99030 |

**Strong hint, not yet conclusive:** the checkerboard's ratio appears to *flip*
sign with roll — 0.990 horizontal (old garage) vs 1.008 vertical (this session,
isotropic board). A near-inversion is the **target-side** signature: it would
mean the sensor is close to isotropic and the ~1 % is largely the board. But the
two checkerboard numbers are different sessions/possibly different units, so this
is not yet a valid comparison.

**The clean test is the LEGO pair** (same camera, both rolls). LEGO horizontal is
0.99773; the vertical calibration kept timing out (six non-planar solves) and was
not finished. Finish it, apply the decomposition above, and cross-check against a
same-session checkerboard pair.

Script: `scratchpad/roll2.py` / `lego_v.py` in the 2de4213f session, or rebuild
from this note. Reduce LEGO refit to 2 iterations and it completes in time.
