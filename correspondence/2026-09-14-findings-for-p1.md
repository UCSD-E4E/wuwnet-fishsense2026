Two findings from P4 (`../wuwnet-fishsense2026`) land on this repo's range-trend
audit. Neither is a bug report — both need your judgement about whether they change
anything in `PAPER.md`, and the second one may be a result rather than a problem.
Read `post_labeling_analysis/HANDOFF.md` §0 first; both of these obey its ground
rules, and the second one turns on §0's "known lengths are the validation set".

Reproduce them from `cscw-fishsense2027/annotation_analysis/scalefree_laser_selfcal.ipynb`,
which imports `fishsense_imwut.calibration` by `sys.path` and reads
`fish_model_analysis/data/corpus.csv` unmodified.

---

**1. The range-trend audit has a one-sided bias from the parallax it already documents.**

HANDOFF §6.4 establishes half-thickness parallax: the dot lands on a solid model's
flank, snout and fork lie in the midplane, so `length(z) = L + b/z` with b negative
(trout −1.8 cm, Snook −2.4, Grouper −5.0, Box −0.4 ≈ 0).

Differentiate it. The Theil-Sen pairwise slope between frames at z_i, z_j is
`b(1/z_j − 1/z_i)/(z_j − z_i) = −b/(z_i z_j)`, which is **positive for every b < 0**,
at every range, under a perfectly correct calibration. `range_trend` is therefore
biased positive on any solid model, by an amount set by that model's thickness and
the dive's range distribution.

Computed per (dive, model) cell that passes the existing gates (≥8 frames beyond
0.8 m, ≥2× spread), as a fraction of known length:

| model | b | cells | predicted trend | at/over the 2.0 %/m flag |
|---|---|---|---|---|
| Box | −0.4 cm | 10 | +0.43 to +0.94 %/m | 0 |
| Weasly Fish | −1.8 cm | 11 | +0.94 to +2.27 %/m | 3 |
| Snook | −2.4 cm | 1 | +2.39 %/m | 1 |
| Grouper | −5.0 cm | 0 | — | no cell passes the gates |

So 4 of 22 cells reach the flag threshold on parallax alone. Against the observed
spread (−6.1 to +5.5 %/m) this is not the dominant term, and the flag is two-sided
with a Sen interval that has to clear the threshold — but it is one-sided, it is
largest exactly where the audit is most used (near range, thick models), and the Box
is the only model free of it.

Questions I cannot answer from here:
- Does it move `CORPUS_ACCURACY_DIVES`? 503 was removed by the pre-filter at +5.3 %/m
  and 498 kept at +2.8 %/m; both are Box/Weasly cells whose parallax term is under
  1 %/m, so my guess is no, but the cohort is yours and `accuracy_cohort` should be
  re-run with the term subtracted to see.
- Should `range_trend` take an optional `b` and subtract `−b/(z_i z_j)` per pair, or
  is the honest move a documented caveat that the statistic is positively biased on
  solid models by a known amount? Subtracting makes a scale-free statistic depend on
  a parallax fit that was made with known lengths, which §0 would not like.
- Is it worth a line in `PAPER.md` where the audit is described? It slightly weakens
  "spends no known length" if the correction needs one.

---

**2. The audit statistic run backwards is a calibration, and on the Box it matches yours.**

P4's per-dive work (`../wuwnet-fishsense2026/refraction_analysis/laser_per_dive.ipynb`,
`fishsense_wuwnet/laser.py`) shows the in-plane angle φ — which `calibration.py`'s
docstring correctly says is invisible to the dots — is recoverable with **no known
length** from any rigid object the dot lands on in two frames at different ranges.
Apparent size goes as 1/Z, so the dot's position along its locus is linear in apparent
size and the intercept is the beam's vanishing point. It is your range-trend statistic
used as the estimator instead of the residual: solve φ so the p90 length reads the same
at every range.

Implemented in your parameterisation (`rotate`, `triangulate_depth`, `theil_sen`) and
compared per dive against `fit_phi_joint`, which uses known lengths:

| model | φ_scalefree − φ_anchor | cells |
|---|---|---|
| **Box** | **+0.003° median, 0.016° MAD** | 10 |
| Weasly Fish | +0.080° (→ −0.005° after subtracting §6.4's b/z) | 11 |
| Snook | +0.100° | 1 |
| Shark | +0.268° | 2 |

The disagreement orders by thickness and your own parallax correction removes it,
which is finding 1 seen from the other side: a scale-free estimator on *length* cannot
distinguish a thick object from a rotated laser. On the Box — dot and landmarks
co-planar — it reproduces the known-length calibration to 0.003°, spending nothing.

What I would like your view on:
- Does this belong in P1 at all? It is P4's method, but P1 owns the corpus, the φ
  parameterisation and the claim that φ "has to come from a known length (an anchor),
  or from the same object seen at two well-separated ranges (scale-free)". That
  sentence predicts this result; nobody had run the second branch.
- If it stays out of P1, the citation still runs the wrong way round: P4 and P2 both
  now lean on your `range_trend`, and P2's `HANDOFF.md` §7 proposes your acceptance
  test as the shared contract. Worth a sentence in `PAPER.md` that the statistic is
  also an estimator?
- The per-dive joint fit over mixed models is contaminated (median 0.076° from the
  anchor, validating worse than the recorded calibration). I do not think that number
  should be quoted anywhere; it is parallax, not a property of the method. Agree?

---

Unrelated, for whoever picks up §4.1 in P2: the E4E checkerboard is 4.2 cm on **both**
axes by ruler (measured), yet an in-air `calibrateCamera` on 63 garage frames of it
returns `fx/fy = 0.9932 ± 0.0015` — 5σ from square, unmoved by rational, thin-prism or
tilted-sensor models, and independently 0.9926 through the JPEG path. Either the sensor
readout is anisotropic by 0.7 % or the calibration carries an orientation-coupled
systematic; the P4 sessions are all landscape so they cannot separate the two. It is
inside P4's 15 % budget and irrelevant there, but §6.3 lists "a systematic in corner
detection" as a live candidate for the checkerboard-vs-slate gap, and this is one.
