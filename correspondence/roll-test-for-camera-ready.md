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

## Result: the sensor is near-isotropic; the ~1 % is largely the target

| estimate | fx/fy | reading |
|---|---|---|
| LEGO joint, both rolls (same session, same tower) | **0.9977** | sensor −0.23 % |
| LEGO joint, balanced 13+13, 40 draws | 0.9977, IQR [0.9971, 0.9981] | robust |
| LEGO horizontal alone | 0.9978 | |
| LEGO vertical alone | 0.9863 | weakly conditioned (13 views) |
| Checkerboard vertical (isotropic pitch, rms 0.67) | 1.0081 | single roll |
| Old garage checkerboard (isotropic pitch, rms 0.39) | 0.9939 | single roll, 2024 |

The **LEGO joint fx/fy is 0.998 (−0.2 %) and robust** to balancing the two roll
groups. That is the same-session, same-target answer, and it is near-isotropic.
Its implied target anisotropy is ~+0.6 %; the checkerboard's own is ~+0.7 %
(sqrt(1.0081/0.9939)). Both targets carry roughly the anisotropy that shows up as
"the camera's" ~1 % in any single calibration.

**Convergent, not airtight.** Three things keep this short of proof:

1. The LEGO fx/fy is weakly conditioned -- rms 5-6 px, and the 90° roll did *not*
   pull the joint fit (the 3D tower already conditions fx/fy within one roll, so
   the rolled views added little). The estimate is 0.998 but its own spread
   allows up to ~−0.8 % (the naive sqrt(h·v) gives 0.992).
2. The two checkerboards cannot be decomposed against each other: different
   physical board *and* different session, so `sqrt(H·V)` across them is not a
   clean sensor estimate. They are only consistent-with, not evidence-for.
3. No same-session, same-board, both-roll, low-rms pair exists in this data --
   which is the one clean measurement.

## To close it (one shoot)

Photograph the **checkerboard at both rolls in one session** -- 15 upright, 15
rolled 90°, same board, locked focus. That is the well-conditioned (rms <1),
same-target, both-roll pair the LEGO can't quite be. `sqrt(H·V)` is then the
sensor and `sqrt(H/V)` the board; if the sensor lands near 1.000 as the LEGO
joint says, section 8's ~1 % is target-side and the section should be revised
from "camera-side" to "target-side, sensor near-isotropic."

**Bottom line for section 8 as written:** the "~1 % is camera-side" lean is not
supported and is mildly contradicted. Soften it to open, citing the LEGO joint
near-isotropic result, until the clean checkerboard pair confirms.
