# fx/fy anisotropy: the roll test (result, 2026-09-22)

**Question.** Section 8 reports a ~1 % `fx/fy` and cannot say whether it is the
*sensor* (fixed in sensor coordinates, survives a 90° camera roll) or the
*target* (fixed to the board, inverts under a 90° roll). Section 8 currently
leans camera-side. This test says otherwise, though not yet airtight.

## Data (raw ORF, `/home/chris/data`, not committed)

| set | views | roll |
|---|---|---|
| Lego Model Horizontal | 21 | upright |
| Lego Model Vertical | 13 usable (14 shot) | camera rolled 90° |
| Checkerboard Vertical | 19 boards | camera rolled 90° |

Detections cached to `scratchpad/roll_cache.npz`; analysis in
`scratchpad/roll2b.py`, `roll_balanced.py`. All raw, not JPEG (the TG-6's JPEG
lens-correction is not the sensor's geometry).

## Method

A recovered ratio factors as `sensor × target^(±1)`: the target's own
anisotropy inverts under a 90° roll, the sensor's does not. Decomposition needs
the **same target at both rolls**; a joint calibration over both rolls estimates
the sensor directly (and averages the target term out).

## Result: the sensor is near-isotropic; the ~1 % is the target

Two independent targets, both decomposed across a 90° roll, agree the sensor is
near-isotropic. A recovered ratio factors as `sensor x target^(+/-1)`; the target
term inverts under roll, so `sqrt(up x rolled)` is the sensor and `sqrt(up/rolled)`
the target.

**Checkerboard (well-conditioned; the strong evidence).** Same physical board and
same camera -- confirmed empirically: the raw radial term matches, k1 = -0.1497
(garage 2024, upright) vs -0.1479 (this session, rolled 90°). Roll is a clean
contrast: garage clusters at 0/180°, the recent set at 89°. Isotropic pitch,
outlier-rejected, rms 0.39 and 0.67:

| roll | fx/fy |
|---|---|
| garage, upright | 0.99387 |
| recent, rolled 90° | 1.00807 |

  sensor = sqrt(0.99387 x 1.00807) = **1.00094 (+0.09 %, near-isotropic)**
  board  = sqrt(0.99387 / 1.00807) = **0.99293 (-0.71 %, the print's own anisotropy)**

**LEGO (same session, corroborating).** The same-session tower, calibrated
jointly over both rolls, gives fx/fy **0.998 (-0.2 %)**, robust to balancing the
13+13 roll groups -- near-isotropic, independent of the checkerboard. Its fx/fy is
weakly conditioned (rms 5-6 px, the 3D tower conditions fx/fy within one roll so
the rolled views barely pulled the joint fit), so it is corroboration, not the
lead.

**So the ~1 % that reads as "the camera's" in any single calibration is the
target's**: the checkerboard print is ~0.7 % anisotropic, the LEGO object model
~0.6 %, and the sensor sits within ~0.2 % of square.

**Remaining caveat.** The checkerboard pair is cross-session (2024 vs 2026),
though same board and same camera -- k1 matching to 0.002 says the lens/focus
barely moved, and fx/fy is stable to focus, so this is minor. The one thing not
yet in hand is a single-session same-board both-roll set; it would remove the
cross-session caveat entirely, but the case no longer rests on it.

## Consequence for section 8

Section 8 leans "~1 % camera-side" (corner-to-corner caliper reading). **That is
now contradicted by a direct optical measurement.** Revise it: the sensor is
near-isotropic (fx/fy within ~0.2 %), and the ~1 % anisotropy is target-side --
the board's own print anisotropy, measured here at ~0.7 % by rolling the camera
rather than by rulering the board. The caliper never could resolve it (1 mm on
the 9-pitch short axis is 0.26 %); the roll test does.

Reproduce: detections cached in `scratchpad/roll_cache.npz`; garage corners in
`data/board_corners_raw.npz`; analysis `scratchpad/roll2b.py`.
