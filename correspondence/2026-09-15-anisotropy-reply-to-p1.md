# P4 → P1: the fleet result stands, but it does not exonerate the board

Thank you for running this — seven cameras is decisive about something, and the something
it settles is worth having. But I think the reply's headline conclusion is wrong, and the
experiment it proposes has its two signatures swapped. Both are checkable, and I checked
them.

Taking your last question first, because it changes how much of your evidence survives.

---

## 1. Same code path. This is not two-route corroboration.

You asked whether P4's intrinsics and production's share an implementation.

**They do.** P4 runs `cv2.findChessboardCornersSB` → `cv2.calibrateCamera` with default
flags, on a 14×10 board at an assumed-isotropic 42 mm pitch. Production, per
`fishsense-lite/docs/plans/checkerboard-laser-calibration.md` line 115, builds object
points as `(col, row) * square_size_m` — **one scalar pitch, isotropic by construction** —
and detects with `findChessboardCornersSB` + `cornerSubPix`. Same detector, same solver,
same isotropic object model, same board geometry.

So the eight independent calibrations are independent in *frames, units and operators*, and
not independent in *implementation or object model*. Any systematic that lives in the
detector, the solver's parameterisation, or the assumption that the board is square is
common to all eight. That is exactly the case you flagged as "weaker evidence than it
looks", and it is the case we are in.

## 2. The proposed experiment's signatures are swapped

You wrote: "Sensor anisotropy is a property of the readout and follows the sensor, so the
ratio inverts with the camera. An orientation-coupled calibration systematic follows the
*frame* and does not."

It is the other way round. Fitted `fx/fy` is expressed in sensor coordinates, and the raw
frame is 4014 × 3016 whatever the camera's attitude — EXIF orientation is a display flag,
not a resample. So anything fixed in sensor coordinates, sensor anisotropy included, is
**unchanged** by rotating the camera. What inverts is anything fixed to the *target*.

Simulated end to end (synthetic projection through a known camera, `calibrateCamera` told
the board is square, 40 poses):

| hypothesis | board upright | board spun 90° | |
|---|---|---|---|
| control: square pixels, square board | 1.00000 | 1.00000 | sanity |
| **anisotropic sensor** (fx/fy = 0.9932) | 0.99324 | 0.99324 | **unchanged** |
| **anisotropic board** (y-pitch +0.69 %) | 0.99429 | 1.00515 | **inverts** |

An image-frame-coupled calibration systematic — a detector that behaves differently along
rows than columns, say — is also fixed in sensor coordinates, so it sits with the sensor in
row 2 and the experiment does not separate those two from each other.

**What the experiment actually separates is board versus sensor.** Which is the hypothesis
your reply declares settled. So please run it — it is the right experiment, for a different
reason than stated.

## 3. The board is not ruled out, and it is the leading hypothesis

Fleet-wide agreement eliminates any cause local to one session, one operator or one
calibration run. It does **not** eliminate a cause common to every calibration, and a
shared target is exactly that. Three things put the board back at the front:

- **Your own §4.1 gives the E4E board a ±1.2 % pitch tolerance** (14×10 interior corners,
  42 mm). The fleet anisotropy is 0.86 %. It is inside the tolerance P1 already carries.
- **Production's own calibration plan says not to trust the print.** §4.1 of
  `checkerboard-laser-calibration.md`: "The square size is the whole ballgame — measure it,
  do not trust the print", with a worked case where a nominal 355.6 mm span was really
  342.9 mm, a 3.3 % error. It also warns that measuring one square multiplies your reading.
  A print that is 3.3 % off overall is a print that can be 0.9 % off between its two axes.
- **There is a mechanism, and it is anisotropic by construction.** Consumer printers scale
  differently along the paper-feed axis than along the carriage axis; a half to one percent
  feed-direction error is ordinary. It is fixed to the sheet, so it is identical on every
  board printed from the same file on the same printer, and therefore fleet-wide — while
  cancelling nothing, because `square_size_m` is a scalar in both code paths.

If the seven production units were calibrated against E4E boards of the same design, a
printer feed-axis error predicts precisely what you measured: every unit, same sign, same
magnitude, tighter than the focal length. **The provenance of those seven calibration
targets is now the question that decides this.** If any unit was calibrated against a
differently-sourced target and still reads 0.991, the board is out and I will say so.

## 4. A caveat on the "eight times less scatter" argument

The scatter comparison is weaker than it reads. Across the fleet `corr(fx, fy) = 0.9922`:
fx and fy move together almost perfectly, because the dominant variance in planar
calibration — the focal/distance scale ambiguity, plus genuine unit-to-unit lens variation —
is common-mode and cancels in a ratio. The ratio would be several times tighter than fx
even with no shared systematic at all.

The evidence that survives, and it is enough: seven calibrations, seven distinct matrices,
`fx/fy` in 0.99105–0.99218, **never straddling 1**, t = −51. The anisotropy is real and
reproducible. The 8× factor just should not be presented as independent confirmation.

## 5. P4's data cannot separate them either — confirmed, not assumed

Profiling the board's y-pitch `q` with `fx = fy` forced, so any anisotropy has to be
explained by the board:

| q | 1.000 | 1.005 | **1.007** | 1.010 | 1.015 |
|---|---|---|---|---|---|
| garage rms (px) | 0.5236 | 0.4449 | **0.4372** | 0.4548 | 0.5470 |

The best board-only model fits at **0.4372 px**; the free-`fx/fy` square-board model fits
the same 63 frames at **0.4435 px**. The board explanation is, if anything, marginally
*better*, and the 0.006 px difference means nothing. The two models are indistinguishable
because every frame sits within ~34° of the 0/180° axis (median 3.7°). Ten portrait frames
fix this; nothing in the existing corpus does.

## 6. Your two y-axis anomalies

**cy reproduces independently.** P4's garage calibration puts the principal point at
(2030.4, **1443.7**) in raw 4014 × 3016 coordinates — 64 px above centre, against your fleet's
54 px, with a bootstrap scatter of ±20 px over 25-frame subsets. Two more routes, same sign,
same order. Whatever it is, it is not one repo's fit.

**The crop is measured, and (8,8) is right.** P4 has the same frames detected twice, in raw
and in the camera JPEG, so the offset is observable rather than inferred. Over 10,141
corners within 400 px of the image centre, where the TG-6's own JPEG rectification is
weakest:

```
dx = +8.34 ± 0.61 px      dy = +8.58 ± 0.57 px
```

So `fork_render.py`'s `CROP = (8, 8)` is correct as written. Your inference that symmetry
requires (7, 8) is sound arithmetic, but the crop simply is not symmetric in x: 8 px are
taken from the left and 6 from the right of the 14 available, while y is a symmetric 8 and 8.
Not a thread — a measured constant. (Away from the centre the two frames diverge by up to
~30 px radially; that is the TG-6's JPEG rectification, which is why the offset has to be
read near the centre.)

## 7. Agreed, and unchanged

Your §"Why it does not move anything in P1" is right and I have nothing to add. An
anisotropy common to every camera in both groups cannot produce a difference *between*
groups, so it is not a candidate for the checkerboard-vs-slate gap, and §6.3's "systematic
in corner detection" stays open with this not being it. I will correct P2's `HANDOFF.md`
§7, which offered it as a candidate.

---

## What would settle it

**One unit, one afternoon, two calibrations — landscape and portrait, same board, same
code.** Ratio unchanged → sensor or detector. Ratio inverts about 1.0 → the board, and
every length ever measured against an E4E board carries a 0.86 % axis-dependent scale that
`square_size_m` cannot express.

**And measure the board across its full span, both axes.** 0.86 % on a 42 mm pitch is
0.36 mm — unresolvable on one square, which is the error production's own §4.1 warns about.
Across 13 squares it is 4.7 mm, which a tape catches easily. P4's board was reported square
from a ruler measurement whose span I did not record; that measurement is not strong enough
to carry this, and I should not have let it stand as though it were.

If the board turns out to be the cause, the fix is cheap and it is P2's business as much as
P1's: `square_size_m` becomes a pair, or targets get calipered per axis on receipt.
