# Reference-free per-dive laser calibration — draft sections for P2

Extracted from the flat-port paper on 2026-09-19. This material is **P2's**: it
removes the barrier P2's `HANDOFF.md` §1 calls *"calibration target must be
fabricated by measurement"*, for the laser half, by removing the target entirely.
It was drafted inside the flat-port paper before that scope call was made.

Two things to know before reusing it.

**The method does not depend on a flat port.** The locus argument is pinhole
geometry on corrected directions; a dome-ported rig can adopt it unchanged. The
flat-port paper supplies the correction that makes the directions correct, and
nothing more. Section 5.3 below says so.

**CSCW will treat the geometry as machinery.** The locus derivation, the gauge
argument, the exact scale-invariance property and the size-measure comparison are
a technical paper's worth of material. Either they get a technical home — its own
paper, or the laser-extrinsics work — and P2 cites the result, or they are
compressed into a P2 section pointing at `fishsense_wuwnet.laser` and the
notebooks. That is a decision, not a formatting question.

Every number here is reproduced by `refraction_analysis/laser_per_dive.ipynb` and
the tests in `tests/test_laser.py`; the code stays in this repo.

---

## 5. The laser: reference-free per-dive calibration

### 5.1 The locus fixes two degrees of freedom for free

In normalised (direction) coordinates the beam's image is

    u(t) = D + O / t,

a **straight line** through `D` in the direction of `O`. The dots a dive already
produces therefore trace a line whose orientation gives the direction of `O` and whose
offset fixes the component of `D` across it. That is two of four degrees of freedom
from the dots alone, with no target, no range and no scale. Measured on ten dots it
recovers them to 0.02-0.04 deg, with a line-fit residual of 0.31 px.

Those ten dots span only 2.1-7.6 deg of field angle, which invites the objection that the
locus is fitted where the camera model is easiest. The objection does not survive the
data. The Pinax residual is nearly flat across the frame -- 0.37 px median inside 200 px
of the centre, 0.46 px at 1800 px -- and the decoy session behind the end-to-end result of
section 6.1 spans **2.9 to 39.6 deg**, dots landing from 149 px out to 2162 px. The
method is not confined to a central patch; it is exercised across most of the frame.

What the line cannot fix is where `D` sits *along* it — the `t → ∞` asymptote — and the
magnitude `|O|`. Together these set the parameterisation `s = |O| / t` of position
along the line: sliding `D` along the line while rescaling every range leaves every dot
where it was. This is monocular scale ambiguity, and no amount of dot reprojection
refinement can touch it.

### 5.2 Apparent size breaks the gauge

Any *ratio* of two dot ranges breaks it. Known ranges supply one; so does a rigid
object of **unknown** size. Its apparent size obeys `ℓ ∝ 1/Z`, and the dot on it sits
at `p = p_D + |O| / Z`, so

    p_i = p_D + (|O| / k) · ℓ_i,

a straight line in (apparent size, position along locus) whose **intercept is the
vanishing point** `p_D` — the position at which the object would shrink to nothing.
No known range, no known size, no mount prior. `|O|`, measured once on a bench, then
sets absolute scale, and the object's true size falls out as a by-product.

Two frames of one object suffice; a dive's worth averages. The conditioning mirrors the
two-range case: the estimator needs leverage in `ℓ`, i.e. range spread.

The dimension used matters. A fish's *length* is foreshortened by yaw, one-sidedly,
and the estimator reads foreshortening as extra range. Its *height at the dot* is not
changed by yaw. We use height, obtained from a segmentation mask.

### 5.3 Scope: what the flat port contributes, and what it does not

The pairing of these two halves invites a stronger claim than we make, so we state the
limit plainly.

The locus argument of section 5.1 is **pinhole geometry**. It holds for any camera whose
pixels map to known scene rays, and the apparent-size argument of section 5.2 is ordinary
perspective. Neither depends on the port being flat. What section 4's correction supplies
is the pixel-to-ray map: without a correct map the dots do not lie on a straight line and
the estimator has nothing to fit. A dome port needs such a map too, and being very nearly
central it obtains one more easily.

**So the flat port is load-bearing for the camera half and not for the laser half.** We
tested the one route by which it could have been load-bearing for both — an axial camera
is not scale-invariant, so in principle the port itself supplies range and the method
becomes entirely self-contained — and it fails for this housing by two orders of
magnitude (section 7).

This is a limitation of the framing and a generality of the method: a dome-ported rig can
adopt section 5 unchanged, and still needs its own answer to section 4.

## 6. Evaluation

### 6.1 End to end: no reference object in the water

The claim of the paper is that a rig can be calibrated without putting a reference object
in the water. We test it directly. The camera is calibrated from the in-air session alone
and corrected with Pinax. The laser is calibrated from the **decoy session alone**, using
only the apparent body height of a swinging fish-shaped object whose size was never
measured. Both are then used to measure the **checkerboard** -- 549 mm along its long axis,
calipered corner to corner, which appears in neither *laser* calibration -- over ten
frames at 1.1-4.5 m.

**What this does and does not put under test.** The board is held out of the laser
calibration, but it is the object the camera was calibrated on and, through the garage
beam fit's poses, the origin of `|O|`'s scale. So an error in the board's assumed size
propagates into `|O|`, into every range, and into the measured span -- where it cancels
against a ground truth that scaled with it. We verified this is exact rather than
approximate: scaling the assumed board pitch from 0.95x to 1.10x moves `|O|` from 98.98 mm
to 114.61 mm and leaves the error at +2.24% throughout.

This experiment therefore measures the laser's **angular** calibration and the refraction
correction, and is blind to absolute metric scale. That is the right target -- angle is
what drifts between dives, and section 3's budget is a bound on angle -- but it is not a
metric accuracy result, and section 6.2 supplies the scale test separately.

Three baselines bracket the result. *No per-dive calibration* uses the bench beam measured
in air 48 minutes earlier: what a rig does if it trusts its extrinsics. *Conventional* is
the board-referenced beam fit, the slate equivalent -- scored **in-sample**, on the very
frames it was fitted to, so it is an optimistic bound rather than a fair competitor. The
reference-free row is scored two ways: across sessions (the decoy precedes the board frames
by ten minutes and one handling event, so it carries real drift) and leave-one-frame-out
within a session, which isolates the estimator from drift but calibrates and tests on the
same object, and is therefore an optimistic bound of its own rather than an independent
result.

| laser calibration | median | p90 abs | max abs |
|---|---|---|---|
| none -- bench beam, 48 min earlier | -5.4% | 8.4% | 15.8% |
| **reference-free, cross-session** | **+0.0%** | 1.2% | 2.6% |
| **reference-free, leave-one-out** | **+1.0%** | 1.8% | 2.7% |
| conventional slate (in-sample, optimistic) | -0.3% | 1.5% | 1.6% |

The board's pitch is measured, not assumed: 42.2308 mm along the long axis and 42.1111 mm
along the short, so the object model is the calipered one and the target span is 549 mm
rather than the nominal 546.0. It is the same physical board in both sessions. The result
does not rest on that: across three assumed geometries spanning 1.4% -- nominal, an
earlier edge-to-edge reading, and this one -- these rows move by at most 0.5 pp.

Because a deployment's tolerance is a choice, we report the same result against tolerance
rather than against a fixed line -- the fraction of frames within each requirement:

| laser calibration | 2% | 5% | 10% | 15% |
|---|---|---|---|---|
| none -- bench beam | 0% | 40% | 80% | 90% |
| **reference-free, cross-session** | 90% | **100%** | 100% | 100% |
| **reference-free, leave-one-out** | 90% | **100%** | 100% | 100% |
| conventional slate (in-sample) | 100% | 100% | 100% | 100% |

The reference-free pipeline meets a 5% requirement on every frame, including across a
handling event it was given no chance to observe; the same rig without a per-dive
calibration meets it on two frames in five.

The reference-free row matches the in-sample slate, which deserves scepticism rather than
celebration, so we separate the two things it contains. The beam *direction* recovered
from the decoy still differs from the board session's by 0.261 deg -- real drift over ten
minutes and a handling event, which no estimator can undo. What the size measure fixes is
the *estimation* error in the vanishing point, from 3.7 px to 0.2 px. The drift that
remains lies mostly across the locus rather than along it, and only the along-locus
component reaches the range. Not all drift costs a measurement.

One systematic appeared in the reference-free and conventional rows alike when the board
was assumed square: its long-axis span read 0.4 percentage points differently from its
short-axis span. Section 8 reports what that turned out to be.

### 6.2 Absolute scale, against an independently calipered object

Section 6.1 is blind to scale by construction. The decoy supplies the missing test,
because its dimensions were calipered directly -- 312.5 mm long, 106.1 mm deep just aft of
the dorsal fin -- and so are independent of the board and of everything in the calibration
chain. Reading the decoy's own metric length back out of the reference-free pipeline:

**311.2 mm, against 312.5 mm calipered: -0.4%.**

Aggregated at p90 over the thirteen frames rather than by mean, because yaw foreshortens
one-sidedly and a mean would measure the decoy's pose distribution rather than the decoy.
Unlike section 6.1 this chain -- calipered object, ranges, beam, `|O|`, the garage board's
poses -- responds one-for-one to an error in the board's assumed size, so it is a genuine
test of absolute scale, and the rig passes it at half a percent.

The same reconstruction recovers the decoy's body-height profile, which peaks at 109.4 mm;
the calipered 106.1 mm falls on it 8% of a body length aft of the deepest point, which is
where "just behind the dorsal fin" should land. Shape and scale are both recovered, from a
calibration that never saw a ruler in the water.

### 6.3 Pool: a board of unknown size

Treating the checkerboard as an anonymous rigid object (√ convex-hull area of its
corners in corrected coordinates, tilt left in, its 42 mm pitch never used), ten frames
at 1.08–4.47 m give:

- vanishing point **0.31 px** from the slate-calibrated reference (fit RMS 1.15 px);
- beam direction **0.009°**;
- recovered object size 0.460 m against a true 0.454 m.

Against the 12 px budget of §3 this is a 40× margin. Pairwise conditioning:

| range ratio | pairs | median `p_D` error | in budget |
|---|---|---|---|
| 1.0–1.3 | 15 | 10.3 px | 9/15 |
| 1.3–1.7 | 15 | 4.5 px | 15/15 |
| 1.7–2.5 | 8 | 1.9 px | 8/8 |
| > 2.5 | 7 | 0.5 px | 7/7 |

### 6.4 Pool: a fish-shaped object that moves

The same dive carries a second laser session on a fish decoy (312.5 mm, measured)
hanging on a line, free to yaw: 13 frames, 0.8-3.3 m, the dots spanning 2.9-39.6 degrees
of field. Masks come from a text-prompted
segmentation model ("fish") on a crop centred on the dot -- the same backend the
deployment already runs for head/tail keypointing, so it is not a new dependency here,
though it is a heavy one for a method sold on deployability. Two lighter attempts failed
first and are worth recording: chroma thresholding against the pool wall, and GrabCut
seeded by the dot, both bloated into the background by 10-20% because the camera's
underwater white balance leaves the flank and the water the same hue. A fish will present
the same problem. Comparing the two size
measures:

| size measure | `p_D` vs reference | leave-one-out sd | fit residual |
|---|---|---|---|
| **sqrt(mask area)** | **0.2 px** | 0.5 px | 2.4 px |
| body height at the dot | 3.7 px | 2.1 px | 15.6 px |
| snout-to-tail length | 12.2 px | 1.7 px | 10.6 px |

The area wins because it uses every pixel of the silhouette, while any chord inherits
whatever the chord is drawn through. Height at the dot inherits the dot's wander along the
body -- 0.107 of a body length between frames here, worth 6.5% of the height on its own --
and that noise propagates straight into the intercept. Measured frame-to-frame consistency:
1.1% for the area against 6.5% for the height at the dot.

The caveat is the failure mode named below. Silhouette area falls off with yaw, so if a
diver's approach angle were correlated with range -- head-on far away, side-on close in --
the area would carry exactly the range-correlated bias that reaches the vanishing point.
Here it is not, and the noise reduction dominates; on a survey dive it should be checked.

Height closes the beam inside budget; length fails exactly as predicted, because the far
frames are the oblique ones. Per-frame scatter is 10-16 px -- the decoy swings -- yet the
intercept is stable to 2 px: the line fit averages pose noise.

**The size by-product is the method's noisiest output, and it is not how the pipeline
measures anything.** This needs saying because the paper reports two very different numbers
for the decoy's length. The **by-product** is the single parameter `k = |O|/slope` fitted
across all frames: 274 mm, 12% short. The **measurement** is the per-frame reconstruction
`l * Z` aggregated at p90, which is what a deployment actually computes and what section
6.2 reports: 311.2 mm, 0.4% short. In an exact model the two coincide; they differ here
because the model is not exact and a least-squares slope is a far noisier estimator than a
robust aggregate of per-frame products. Validate the pipeline on the measurement, never on
the by-product.

The same gap explains the depth numbers. The decoy's body height, calipered afterwards, is
106.1 mm; the by-product returns 99.7 mm from height at the dot and 112.4 mm from maximum
body height. Neither is a calibration error, because the estimator is **exactly invariant to a
constant scale error in the size measurement**: in `p = p_D + (|O|/k) * l`, replacing `l`
by `c*l` for any constant `c` leaves the intercept `p_D` untouched and moves only the
slope, hence only the recovered `k`. Scaling every measured size by 1.06, 0.85 or 2.5
moves `p_D` by less than 0.001 px. So whether the segmentation mask includes the dorsal
fin, and where along the body one chooses to measure, are absorbed entirely by the
by-product.

What is *not* absorbed is a **range-correlated** bias, and that distinction turns out to
matter more than goodness of fit. Maximum body height is the more self-consistent measure
-- 5.8 px fit residual against 15.6 px, because the dot wanders over 0.107 of the body
length between frames and takes the local height with it -- but it is the *worse*
calibrator, giving +5.1% median and 11.4% worst end to end against +2.2% and 3.5% for
height at the dot (section 6.1). A size measure that is consistent frame to frame is not
thereby free of a trend with range, and only the trend reaches the vanishing point. Fit
residual is therefore not a model-selection criterion here; a held-out target is.

### 6.5 An independent production corpus

We re-ran the estimator on 2,927 frames over 32 dives from an independent deployment of
the same measurement pipeline, where per-dive calibrations had been produced
conventionally and lengths validated against known references. This is a **cross-check on
independently collected data, not an independent validation**: our estimator and the
reference fit run on the same frames and the same objects, so what it establishes is
agreement between two estimators, one of which spends known lengths and one of which does
not. The housings in that deployment are not all of the type studied in section 4, which
makes the section a test of section 5 alone -- consistent with its being port-independent
(section 5.3). Expressing our estimator
in that pipeline's own in-plane-angle parameterisation φ and comparing against its
known-length fit:

| reference object | φ (ours) − φ (known-length) | cells |
|---|---|---|
| **Box** (slab) | **+0.003° median, 0.016° MAD** | 10 |
| trout model | +0.080° → −0.005° after thickness correction | 11 |
| Snook model | +0.100° | 1 |
| Shark model | +0.268° | 2 |

On a slab, spending no known length, the estimator reproduces a known-length
calibration to 0.003°. On the other targets it disagrees, and the disagreement orders
with the targets' thickness.

### 6.6 The failure mode, stated

A size measured off a surface the dot does not lie on carries a term in `1/z`. The dot
lands on a solid object's **flank** while the landmarks used for size lie nearer its
**midplane**, so the measured size is short by roughly the half thickness over the
range. A scale-free estimator operating on size cannot distinguish such a term from a
rotated laser, and will rotate the laser to cancel it.

**We do not claim the observed disagreement is that term.** The ordering is consistent
with thickness, but each target in the corpus was measured on its own sessions, so a
per-target `1/z` coefficient cannot be separated from whatever range dependence those
sessions' calibrations retain -- the point that corpus's own authors make, and they
decline the attribution for the same reason. Their fit is also quantitatively awkward
for a pure thickness story: the solid trout's coefficient is about five times smaller
than its measured 58.7 mm across the body predicts, while a flat plate shows a larger
one. We report the disagreement, name the mechanism it is consistent with, and leave
the attribution open.

What survives without the attribution is the design rule, which is what the method
needs: **the size fed to the estimator must be measured on the surface the dot lands
on, and must not drift with range.** Any size measure that is systematically wrong in
a way correlated with distance -- for whatever reason -- moves the vanishing point,
and no amount of averaging removes it. Section 6.4 gives the measured case: a size
measure that is more self-consistent can still be the worse calibrator.

Our own decoy test is not exposed to the session confound in the same way, because
there the same object calibrates and is measured within a single session; but it is a
single object, so it cannot separate the mechanisms either.


## Negative result, carried over from the flat-port draft

**A prior on the mount cannot replace the object.** The locus fixes two of the beam's
four degrees of freedom, so a mount constrained to one degree of freedom would close it
outright — no range, no object, nothing but the dots. This is the one alternative to
section 5.2 that would need no scene content at all, so it has to be ruled out before
the apparent-size route is justified.

It does not survive the check. Constraining `D` to the circle the design axis implies —
parallel to the optical axis, radius fixed by one bench measurement — places `D` 12 px
off at best, which is 15.6 % at 4 m: over budget, with no margin. The two sessions we
have disagree about that radius by far more than their own fit noise, so the prior is
not merely imprecise but unsupported by the data available to us.

We stop there deliberately. Establishing *how* the mount actually moves — whether its
freedom is one degree or several, and with what distribution — is a characterisation of
the hardware, it needs the per-dive extrinsics of many units rather than the dots of
one dive, and it is the subject of separate work. The conclusion this paper needs is
only the negative one: no mount prior available to us closes the beam, so the size of
something in the scene is doing real work. The design axis remains useful within this
paper as a branch-resolver for the locus's sign ambiguity and as an outlier gate.
