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
| checkerboard | 14 (12 detected) | all one roll: board's long side along the image | 200-400 |

The horizontal checkerboard set was shot first (frames P9250036-061) but was not
in the copied folder. EXIF orientation is no guide for the board: every frame is
tagged the same, because the orientation sensor cannot tell rolls apart when the
camera looks down. Rolls here are measured from each frame's detected grid or pose.

## Results

**Focal length: the corner agrees with the checkerboard.** 24 views, rms 6.6 px:
fx 2998.2 against the board's 2998.6, **-0.02 %** (bootstrap 68 %: -0.57 to +0.90 %).

**fx/fy: the corner lands on the sensor; the board does not.** The board at this
roll reads 1.0108. Paired with the same board's 2024 upright set (same board and
camera, confirmed by matching distortion), it splits into **sensor 1.0005** and a
**board print error of about 1 %** -- the 2026-09-22 roll test found 1.0009 and
0.7 % by the same method. The corner reads **1.0010** (bootstrap 0.9991-1.0031),
**+0.05 % from the sensor**. A single-session brick target measured the camera's
anisotropy; a single-roll printed board measured its own print.

The cross-session pairing is the remaining caveat. The horizontal board frames
from this session remove it: rerun `board_roll.py` with both folders.

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
