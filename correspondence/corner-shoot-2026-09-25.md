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

**Focal length: +0.9 % above the board, and probably focus, not the corner.**
Against the aspect-free board fit (fx 2969.0) the corner reads fx +0.90 %. The two
were not shot at the same focus: the boards at 0.66-1.98 m (median 1.2 m), the
corner at 0.10-0.46 m (median 22 cm), and the TG-6 refocuses for every frame. A
lens's effective focal length grows as it focuses closer -- a thin 4.5 mm lens by
+1.7 % between those medians. Within the corner set, near views (median 18 cm)
calibrate 1.4 % longer than far ones (31 cm), against +1.1 % predicted, which
points to focus breathing. The board's near and far halves (0.8 vs 1.5 m, +0.26 %
predicted) differ by -0.5 %, below what those subsets resolve. Not proven: the
test is both targets at one locked focus, or the board shot close.

If it holds, a calibration carries the focal length of the distance it was focused
at, and the corner has to be shot close. Fish are at 1-5 m. Length is tolerant --
the per-dive beam fit cancels focal scale -- but it belongs in the paper.

(The earlier comparison against the vertical boards alone gave fx -0.02 %: those
boards' own print error happened to cancel the focus offset. A one-K fit across
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
