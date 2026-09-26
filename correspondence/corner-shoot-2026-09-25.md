# First shoot of the corner target, 2026-09-25

Raw frames in `~/data/2026-09-25/{LEGO,Checkerboard}` (not in the repo); matched
correspondences in `data/corner_2026-09-25_raw.npz`, board corners in
`data/checkerboard_2026-09-25_raw.npz`. Reproduce:

    uv run python refraction_analysis/calibrate_brick_target.py \
        --frames ~/data/2026-09-25/LEGO --checkerboard ~/data/2026-09-25/Checkerboard
    uv run python refraction_analysis/board_roll.py \
        --boards ~/data/2026-09-25/Checkerboard --corner data/corner_2026-09-25_raw.npz

## What was shot

| set | frames | roll (measured from the images) | ISO |
|---|---|---|---|
| corner | 35 | 24 upright, 11 rolled 90 deg (poses and EXIF agree) | 400-800 |
| checkerboard | 40 (37 detected) | 23 with the board's long side across the image, 14 along it | 200-400 |

The horizontal checkerboard set (P9250036-061) was missing from the first copy and
added later. EXIF orientation is no guide for the board: every frame of the first
copy is tagged the same, because the orientation sensor cannot tell rolls apart when
the camera looks down. Rolls here are measured from each frame's detected grid or pose.

## Results

**fx/fy: the corner measures the sensor, in one session.** Both board rolls, same
session (`board_roll.py`):

| | fx | fy | fx/fy |
|---|---|---|---|
| board, long side across (23) | 2948.3 | 2978.4 | 0.98989 |
| board, long side along (14) | 2998.6 | 2967.8 | 1.01037 |
| sensor = sqrt(h x v) | | | **1.00008** |
| corner, 22 views | 2997.1 | | **1.00126** |

The board's print is 1.02 % anisotropic -- each roll lands about 1 % either side of
square -- and the sensor is square to 0.01 %. The corner is **+0.12 %** from it.
Fitting the board's print aspect as a free parameter across both rolls instead
gives aspect -0.89 % on the long axis and sensor fx/fy 0.99897, so the two methods
agree to 0.1 %, and the corner is +0.12 to +0.24 % from the sensor between them.
A printed board at one roll reads its own print; the brick corner does not.

**The corner calibrates on its own.** Fitting its own distortion, with the same
two-level rejection, against the board fitted with its print aspect free:

| | fx | fx/fy | cx, cy | k1 | rms px |
|---|---|---|---|---|---|
| board, both rolls, aspect free | 2969.0 | 0.9990 | 2030, 1490 | -0.152 | 1.0 |
| corner alone, k1 + k2 free | 2966.9 (-0.07 %) | 1.0022 | 2002, 1509 | -0.155 | 6.4 |
| corner alone, k1 free | 2973.9 (+0.16 %) | 1.0019 | 1997, 1515 | -0.127 | 6.3 |
| corner, board's distortion held | 2995.8 (+0.90 %) | 1.0014 | 2002, 1512 | (board) | 6.5 |

- **Focal length: within 0.1-0.2 % of the board**, from bricks alone. The +0.9 %
  appears only when the corner is handed the board's distortion, so it is the fit
  trading distortion for focal length, at least in part. Focus breathing -- the
  corner was focused at ~22 cm, the board at ~1.2 m -- would push the same way, and
  near corner views read 1.4 % longer than far ones, but that split was also fitted
  with the board's distortion. Unresolved; the test is one locked focus.
- **fx/fy: about 0.2 % from the sensor**, where a single-roll board is 1 % off.
- **k1 matches** (-0.155 against -0.152); k2 is not determined by the corner.
- **Principal point is the weak parameter**: about 30 px left and 20 px low of the
  board's, stable across distortion models. A small target does not cover enough of
  the frame to pin it; frames with the target near the image edges would.
- **Per-point residual 6.3 px** against the board's 1.0 -- more views for the same
  certainty; lower ISO should help.

For FishSense this is sufficient: the per-dive beam fit absorbs focal scale and the
principal point, and the corner gets right the one thing a single-roll board gets
wrong, fx/fy.

(The earlier comparison against the vertical boards alone gave fx -0.02 %: those
boards' own print error happened to cancel the offset. A one-K fit across
both rolls gives fx 2933 at rms 1.26 because it forces the board's anisotropy into
one model; don't use it as the reference.)

## Diagnostics

- **Dropped and empty views.** 31 frames matched at least six faces; 7 of those were dropped by the two-level
  rejection, one of them a two-wall view 81 px out -- the mirror match seen from the
  concave side. The 3 frames with no match are P9250010 (far beyond the 32 cm
  limit) and P9250007 and P9250014 (the concave side); the matcher returned
  nothing rather than a wrong answer. P9250029 matched only 4 faces. 6 used views are past 32 cm and flagged.
- **Residual 6.6 px** against the tower's 4.9: likely ISO 400-800 (the tower's
  error budget was dominated by colour-mask edge noise) and the close range, which
  makes every brick larger in pixels.
- **Faces found per colour** (of faces squarely in view, over the 24 views):
  red 71 %, blue 66 %, yellow 63 %, **green 48 %**, all 63 %. Green was the darkest
  colour on the tower and sat nearest the classifier's saturation floor; it is the
  first thing to improve, and a lower ISO should help.
- **Upright against rolled.** Calibrated separately, upright views give fx/fy
  1.0014 (0.9994-1.0051) and rolled 0.9975 (0.9961-0.9998). The intervals barely
  overlap; taken at face value that is a corner-side anisotropy near 0.2 %. But the
  rolled subset is 10 mostly single-wall views and its own fx wanders from -4.6 %
  to +1.7 % across resamples, so it is not settled. About 15 more rolled frames,
  from outside the corner and within 32 cm, would settle it.

## For the next shoot

ISO 100-200; about as many rolled as upright corner frames; both board rolls in
the same copy; from outside the corner, within 32 cm.

## Structure-from-motion: calibration without a target (2026-09-26)

`refraction_analysis/sfm_calibration.py` (COLMAP via `pycolmap`, `uv sync --group
sfm`) self-calibrates from ordinary photographs: features matched across views,
scene and camera solved together, nothing known about any target. Started from
COLMAP's default focal guess, about 60 % long here, so the focal length is recovered.

| | f | fx/fy | principal point | k1 |
|---|---|---|---|---|
| SfM on the checkerboard frames | 2964.5-2965.7 | 0.9999 | (2001-2007, 1512-1522) | -0.162 |
| corner, own distortion | 2966.9 | 1.0022 | (2002, 1509) | -0.155 |
| checkerboard calibration | 2970.5 | sensor 1.0001 | (2030, 1490) | -0.152 |

- **On the 40 checkerboard frames SfM works**: 39 registered, ~8000 points, 1.0 px,
  focal length 0.2 % from the board's, fx/fy 0.9999 against the sensor's 1.0001.
  The board is only texture to it; its geometry is never used.
- **On the 35 corner frames it fails**: 3 images registered. Features are plentiful
  (~7600 per image) but only 50 of 595 pairs verify, 7 with 100+ inliers. Circling
  close around a small object gives neighbouring frames too little shared scene.
  SfM needs overlapping views of one scene, not a walk around an object.
- **The principal point.** SfM and the corner, independent of each other, agree at
  about (2002, 1510); the checkerboard's (2030, 1490) is the outlier. Calling the
  principal point the corner's weak parameter (above) was wrong -- if anything it
  is the board's, likely through its print anisotropy or the tilt coupling of a
  planar target.

What this means for citizen science: the camera can calibrate itself from ~40
overlapping photos of an ordinary textured room, shot at the working distance, with
no target at all. The brick corner is then a check, and a scale, not a requirement.
Untested: how few photos suffice, and whether a room without a checkerboard in it
gives as much texture.
