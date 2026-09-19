# Calibrating an Underwater Laser-Camera Rig Without an In-Water Reference

*Draft, WUWNet 2026. Sections marked [TODO] are placeholders.*

## Abstract

Laser–camera rigs measure fish length underwater by reading range from a projected
dot and transverse extent from the image. Both halves must be calibrated, and both
are conventionally calibrated against a reference object placed in the water: a
checkerboard for the camera and a dive slate for the laser. Underwater calibration
is the dominant cost of operating such a rig, and it is repeated because the laser's
extrinsics are not stable between dives.

We show that neither reference is necessary, by two independent arguments. For the
camera, a flat port is an axial imaging system whose refraction is a pure function of
the water refractive index, so an in-air calibration plus a per-pixel correction
(Pinax) replaces the in-water session; this half is specific to the port. For the
laser, the dot locus through a correctly-modelled camera is a straight line whose two
free parameters are fixed by the dots alone, and the remaining scale gauge — the
position of the beam's vanishing point along that line — is broken by the *apparent
size* of any rigid object the dot happens to land on, whose true size need not be
known. The dive's own subjects are the calibration target. **This second argument is
not specific to a flat port**: it is pinhole geometry applied to corrected directions,
and a dome-ported rig can adopt it unchanged. What the two halves share is the
corrected directions, and what their combination buys is the system property we test:
no reference object of any kind enters the water.

On a matched in-air/in-water dataset we confirm the refraction physics directly
(water/air focal ratio 1.3214 against a predicted 1.333) and quantify the cost of
ignoring it: an uncorrected flat port over-reads a fish at the frame corner by 22%
while under-reading one near the optical axis, an error no single scale factor can
absorb. We also report a correction to the usual case for Pinax: on geometry alone, a
properly fitted in-water calibration is indistinguishable from it, and the real
advantages are calibration noise (2.1× worse in water) and the removal of the
in-water session entirely.

End to end -- camera calibrated in air, laser calibrated from a moving fish decoy, both
then used to measure a 549 mm target held out of the laser calibration -- the
reference-free pipeline reads **+0.0% median, 2.6% worst case** over 1.1-4.5 m, against
**-5.4% median and 15.8% worst** for the same rig with no per-dive laser calibration, and
**-0.3%** for a conventional slate calibration scored in-sample. That experiment isolates
angle, which is what drifts, and is blind to absolute scale by construction; scale is
tested separately against an independently calipered object and comes back to **-0.4%**. Component-wise, the
estimator recovers the beam direction to 0.009 deg against a slate reference, and on 2,927
frames of an independent production corpus it reproduces a known-length calibration to
0.003 deg (median, 10 dives) where the reference object is a slab. We give the failure
mode -- a size measure that drifts with range moves the vanishing point, and no averaging
removes it -- and three negative results, including that the flat port's own scale
signal is too small to supply range for this housing.

## 1. Introduction

[TODO: motivation — fish stock assessment, citizen science, why length matters]

A stereo rig measures length from disparity; a laser–camera rig measures it from one
camera plus a projected dot, which is cheaper, smaller and easier to make watertight.
The price is calibration. The camera's intrinsics must account for the refraction at
the housing port, and the laser's position and direction in the camera frame — its
extrinsics — must be known. Neither survives a dive unchanged: the port is a
refractive element whose effect depends on the water, and the laser is a mechanical
assembly that moves.

Our measurements on a production system put numbers on the second of these. Two
independent calibrations of the same beam, 48 minutes and one hand-carry apart,
differ by 0.378° in direction and 2.5 mm in origin — about 5% in range. Across a
season the spread is of order 2°. This is why the field procedure includes a dive
slate: a known planar target, photographed several times per dive, whose pose places
each dot in three dimensions so that a line can be fitted through them. It is
also why the procedure is expensive, and why an untrained operator cannot run it.

This paper removes both references. Our contributions are:

1. **A direct, ground-truth-free measurement of flat-port refraction error** on a
   matched in-air/in-water dataset, using a homography-residual test that requires no
   external reference, and a decomposition of the resulting length error into a small
   *local* term and a dominant *differential* term that prior treatments conflate
   (§4).
2. **A correction to the case for the Pinax model**: on geometry alone it is
   indistinguishable from a well-fitted in-water calibration. Its advantages are
   elsewhere (§4.3).
3. **A per-dive laser calibration that needs no target**, spending no known length and
   no known range, using the apparent size of arbitrary rigid objects in the scene
   (section 5). It requires a correct camera model but not a flat port, so it transfers
   to other housings; we say so explicitly in section 5.3 rather than let the pairing
   imply otherwise.
4. **An end-to-end demonstration**: both halves calibrated with no in-water reference,
   then used to measure a target held out of the laser calibration, at +0.0% median against
   -5.4% for the same rig uncalibrated -- with an explicit account of what that experiment
   can and cannot see (section 6.1), and a separate absolute-scale check against a
   calipered object at -0.4% (section 6.2).
5. **Validation on an independent production corpus** of 2,927 frames over 32 dives,
   and the estimator's one systematic failure mode (section 6).
6. **Three negative results** that constrain the design space: the flat port cannot
   supply range at this housing's geometry; a one-degree-of-freedom mount prior is
   rejected by the data; and the nominal laser axis is too loose to close the
   calibration (§7).

## 2. Related work

[TODO: Treibitz et al. on flat refractive geometry; Agrawal et al. axial cameras;
Łuczyński et al. 2017 Pinax; Jordt-Sedlazeck & Koch; underwater SfM; laser-camera
ranging in fisheries — Rochet et al., Dunbrack; stereo-video alternatives.]

The Pinax model [Łuczyński et al. 2017] treats a flat port as an axial camera and
computes a virtual pinhole at an optimal offset, from which a per-pixel refraction map
is derived. Its central practical claim is that the camera can be calibrated in air
and corrected analytically for any water salinity. We adopt the model and test that
claim, and we separate the parts of it that hold on geometry from the parts that hold
for other reasons.

## 3. System and problem statement

The rig is an Olympus TG-6 in a commercial flat-port housing with a 532 nm laser
module mounted alongside, offset by |O| ≈ 104 mm from the optical axis and nominally
parallel to it. Images are processed in raw sensor coordinates (4014 × 3016); the
camera's own JPEG pipeline applies a radial lens correction which, measured against
raw, displaces corners by ~1 px with radius correlation +0.85 — the same functional
form as the refraction signal, so measuring one through the other would confound them.

Write the beam as `P(t) = O + tD` in the camera frame, with `O = (O_x, O_y, 0)` and
`D = (D_x, D_y, 1)`. Length is obtained as `L = ℓ_px · Z / f`, where `Z` is the range
read off the beam. The calibration problem is thus:

- **camera**: recover the map from pixels to scene rays through the port;
- **laser**: recover `(O, D)`, four degrees of freedom, once per dive.

The tolerance follows from the application. For a worst-case length error ε at range
Z, differentiating `Z = |O| / (p − p_D)` — where `p` is the dot's position along its
locus in the corrected image and `p_D` the locus position of `D` — gives

    |δp_D| ≤ ε · |O| / Z.

The budget is therefore a property of the rig, not of this paper: it scales as `|O|/Z`,
so a shorter baseline or a longer working range tightens it proportionally. For our
`|O|` = 104 mm at 4 m it is 3.9 mrad -- **12 px** at our focal length -- for a 15%
tolerance, and 1.3 mrad or 4 px for 5%. The same expression bounds `|δ|O||` at
`ε·|O|`. Because a deployment's tolerance is a choice and not ours to make, section 6.1
reports achieved error against a range of tolerances rather than against a fixed line.

## 4. The camera: refraction without an in-water reference

### 4.1 A ground-truth-free test

A rigid planar board, viewed through a *correct* camera model, is related to its own
plane by a homography. If the model is wrong, the relation is not a homography and the
best-fit homography's residual measures the model error — with no external ground
truth, no known pose, and no known scale. Flat-port refraction has a distinctive
signature under this test: because the refraction angle grows with field angle, the
residual should grow with radial distance from the image centre, whereas a plain
focal-length error would be uniform.

We photographed one checkerboard (14 × 10 interior corners, 42 mm pitch) with one
camera in air and then in a pool the same afternoon — a matched pair with no confound
in camera, target, or operator. 63 in-air and 121 in-water frames yielded corners.

**Three pipelines are compared throughout.** *Uncorrected* uses the in-air
calibration underwater unchanged -- the do-nothing baseline. *In-water SVP* is an
ordinary OpenCV calibration fitted to the underwater frames under a **single-viewpoint**
model: pinhole plus radial distortion, which assumes every ray passes through one centre
of projection. That assumption is precisely what a flat port destroys -- refraction makes
the camera *axial*, its rays crossing the optical axis over a spread of points rather than
at one -- but the radial terms absorb most of the refraction anyway, which is what makes
it a fair baseline rather than a straw man. It is also the state of practice: calibrate
underwater and treat the result as a pinhole. *Pinax* is the in-air calibration plus the
analytic refraction correction. Only the first needs no wet session; only the third needs
no wet session *and* claims to be right.

**The physics is confirmed in kind, with a residual we do not model.** The ratio of
in-water to in-air fitted focal
length is **1.3126**, against a water refractive index of 1.333 -- the right effect, of
the right size, 1.5% low. We report the gap rather than call it agreement: magnification
approximately equal to `n_w` is a paraxial statement, while a focal length fitted over a
plus-or-minus 20 degree field with radial terms is a weighted compromise, so a
percent-level offset is expected and ought eventually to be predicted rather than
absorbed. (With a nominally square object model the same ratio reads 1.3214; the board is
not square -- section 8 -- and 1.3126 is the value from the measured geometry.)

**The error is radial.** Over the well-sampled radial range the uncorrected
homography residual climbs **5.4×**, monotonically, from the image centre to 1500 px.
Both corrections are nearly flat over the same span and end well below uncorrected.
(Radial bins containing fewer than 200 corners are not used: a board reaches the
extreme corners only occasionally, and a median over a handful of corners is noise.)

### 4.2 What it costs in length, and why the obvious measurement understates it

Converting the residual to a length error requires care, because there are two such
errors and they differ by about fifty times.

A laser–camera rig reads *range* from a dot that sits near the optical axis, and
*extent* from a fish that can be anywhere in the frame. The two halves of one
measurement are therefore sampled at different field positions.

The **local** error is the distortion across the fish's own extent at one field
position. It is a second difference, and a homography absorbs the first-order scale,
so it is small: 0.37% uncorrected on a 297 mm fish.

The **differential** error is the difference in magnification between the fish's
field position and the laser dot's. It is first order and it dominates:

| fish field radius | uncorrected | Pinax |
|---|---|---|
| 500 px | +1.3% | +0.9% |
| 1000 px | +4.1% | +1.9% |
| 1500 px | +7.7% | +2.1% |
| 2000 px | +13.0% | +2.1% |
| 2500 px | **+20.9%** | +1.6% |

An uncorrected flat port over-reads a fish at the frame corner by 21% and *under*-reads
one closer to the axis than the laser dot. The sign change is as important as the
magnitude: this is not a bias a single scale factor could absorb, because it depends on
where in the frame the fish happened to be. Two images of the same fish, framed
differently, give different lengths.

Reporting only the local term -- which a board-diagonal test measures -- makes an
uncorrected flat port look harmless. It is not.

**Both columns are differences from the in-water SVP calibration**, which is the
reference here and therefore reads zero by construction. We flag this because the Pinax
column invites a reading it does not support: it is not an accuracy, it is a
*disagreement between two corrections*, and this experiment cannot say which of the two
carries it. The uncorrected column is established regardless -- 21% is far larger than
any plausible disagreement between two corrections that agree to 2% -- so the cost of
ignoring refraction is measured, while the residual 2% between SVP and Pinax is bounded
but unattributed. Settling that needs an external length reference spanning the field,
which this dataset does not have.

### 4.3 A correction to the case for Pinax

Median homography residuals are: uncorrected 2.85e-4, in-water SVP 1.17e-4, Pinax
1.26e-4. The two corrections track each other within the line
width at every radius. Note that this metric, unlike the length comparison above,
privileges neither: a homography residual is computed against the board's own plane, so
each model is scored on its own.

**On geometry alone, a properly fitted in-water calibration is as good as Pinax.** We
state this plainly because it is easy to construct a comparison in which Pinax wins:
if the in-water baseline is a focal-length-only fit, it loses badly, but a fit that
includes radial terms absorbs most of the refraction, and the gap closes.

The case for Pinax rests elsewhere, and our data support two such arguments.
First, **in-water calibration is noisier**: calibration RMS was 0.438 px in air against
0.929 px in water -- 2.1x worse, same camera, same board, same afternoon. The penalty
comes from blur and lost views, not from jitter on well-detected corners. Second, and
operationally decisive, **the in-water session disappears**. Pinax's angular correction
`γ = asin(sin α / n_w)` depends only on the water index; the port geometry enters only
through a small lateral ray offset, negligible at survey range.

### 4.4 How a dome compares, and why accept a refractive port at all

A flat pane is not the usual choice, and the paper would be evading its own question if it
compared only flat-port treatments. A **dome port** whose centre of curvature coincides
with the camera's entrance pupil bends nothing: every ray leaves along a radius, meets both
glass surfaces at normal incidence, and an in-air calibration is valid underwater with no
correction whatsoever. That is exact, not approximate, and we assert it to machine
precision in test rather than argue it.

So a dome sets the benchmark, and what it costs is entirely assembly. Scene-direction
error at the frame corner (41.4 degrees off axis), and the resulting range error on our
laser geometry:

| port | error at frame corner | range error, 0.5-5 m |
|---|---|---|
| flat pane, uncorrected | -11.87 deg | ~25% |
| dome, concentric | **0.00 deg, exactly** | 0% |
| dome, 2 mm decentred | +0.40 deg | ~1% |
| dome, 5 mm decentred | +1.00 deg | **+2.4 to +2.6%** |
| dome, 10 mm decentred | +2.00 deg | ~5% |
| flat pane, Pinax | -- | ~0% |

A dome decentred by 5 mm -- a tenth of its own radius, a poor build -- still beats an
uncorrected pane by an order of magnitude, and its range error is nearly flat with distance
because the decentring is angular and the laser dot sits near the axis at every range.

Three reasons to accept a pane anyway, and only the third is optical. A dome wide enough
for a wide lens is bulky, fragile and expensive. Its alignment depends on the entrance
pupil's position, which moves with zoom and focus and cannot be inspected once the housing
is sealed -- so its error is an assembly tolerance that differs between units and changes
when the lens refocuses, where a pane's is a fixed function of the water index, identical
on every unit and every dive. And a dome forms a virtual image a few centimetres in front
of the port, so the lens must focus far nearer than the subject.

The measured comparison is the one that matters for this system: **a corrected flat pane
sits inside a realistically-built dome.** That is a stronger claim than "the correction
recovers most of what the pane costs", and it is the reason the choice is defensible rather
than merely forced.

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

## 7. Negative results

**The flat port cannot supply the range.** An axial camera is not scale-invariant, so
in principle a board of unknown size yields its own range and the method becomes entirely
self-contained. We tested it: with the board's scale free, the residual-versus-scale
profile is flat to 0.004-0.05 px against a **2.2 px** floor, and a pinhole control is
exactly flat as it must be.

That 2.2 px is the residual of this particular fit -- seven parameters, a log-scale and a
rigid pose -- and it should not be confused with the accuracy of the camera model. Scored
by a homography, which has eight parameters and absorbs an affine map of the plane, the
same Pinax model leaves a **0.37 px** median residual. The gap between the two is what
the extra degrees of freedom absorb, not a change in the optics. Sweeping the port offset
`d0` from 0.7 to 30 mm never separates them. The signal is real (0.05 px at working
range) but it sits two orders of magnitude under the model-mismatch floor for this
housing. Frames under 0.5 m, or a deliberately loose port, would change this — the
latter in direct tension with Pinax.

**A one-degree-of-freedom mount prior is rejected.** The laser body rolls in its mount,
which would make `D` sweep a cone of fixed half-angle about the mount axis; if true,
the locus would close the beam with no range and no object. Fitting 114 per-dive loci
across seven units as a closed loop gives χ² = 12 with 104 px of residual against 109
px of spread, and an equal-weighted fit explains no more variance than a Gaussian blob
of matched spread. Two of seven units show loop structure at 3–5σ; five do not.

**The nominal axis is too loose.** The laser is designed parallel to the optical axis,
so `D` should lie on a circle about the principal point whose radius one bench
measurement fixes. Our in-air and in-water sessions give circle radii of 0.478° and
0.278° — 13σ apart. Intersecting the locus with the design circle places `D` 12 px off
at best, which is 15.6% at 4 m: over budget, with no margin. The design axis remains
useful as a branch-resolver and an outlier gate.

## 8. Limitations

- **One camera, one housing, one pool.** The refraction results are a matched pair with
  no confound, but `n = 1` in every other sense, and the pool's index was assumed, never
  measured.
- **Small dot counts.** Ten and thirteen dots for the two pool sessions, and the
  best-conditioned row of section 6.2 rests on seven pairs drawn from those ten, which are
  not independent. The production corpus is larger but was collected for another purpose.
- **The decoy comparison has no independent truth.** Its reference is a slate
  calibration from the same dive ten minutes and one handling event later, so the
  3.8 px includes real drift and is an upper bound, not the method's error.
- **`|O|` is still measured once.** The method is reference-free per dive, not
  calibration-free. `|O|` error enters range 1:1; we observe 1.2–2.5 mm of movement
  across a handling event, inside the 15 mm budget but not negligible at a tighter one.
- **Range spread is required, and we have not measured how often it occurs.** A dive whose
  objects were all photographed at one range cannot be calibrated this way; section 6.2
  shows conditioning becoming reliable past a range ratio of about 1.3 (section 6.3). Our pool sessions
  were arranged to span 1.1-4.5 m. Whether an unscripted survey dive supplies one rigid
  object at two clearly separated ranges is a protocol question we state rather than
  answer, and it is the main obstacle to deploying this unsupervised.
- **Fish are not rigid.** Our fish-shaped validation object is a rigid lure. Body
  height at the dot is robust to yaw but not to a bending body.
- **A 1% `fx/fy` anisotropy we can characterise but not yet attribute.** With a nominal
  square object model our in-air calibrations return `fx/fy` = 0.9931, and the production
  fleet returns 0.99141 +- 0.00044 across seven cameras from seven independent
  calibrations -- real, reproducible, and eight times better determined than the focal
  length those calibrations were measuring. Two explanations fit: an anisotropic sensor
  or detector, fixed in sensor coordinates; or an anisotropic printed target, fixed to the
  board. The fleet-wide agreement does not separate them, because all seven share a target
  design.

  We calipered the board. Corner to corner its pitches are 42.2308 mm (long axis) and
  42.1111 mm (short), a 0.28% anisotropy in the *opposite* direction, which makes the
  camera-side term larger rather than smaller: 0.97%. On that reading the target is not
  the cause. We do not present it as settled. The short axis spans only 9 pitches, so a
  1 mm reading error is 0.26% of it; a reading of 377.5 mm would make the camera exactly
  square; and two careful attempts at that span differed by 5 mm. The question lives
  inside the precision of a rule.

  The test that decides it needs no length measurement: one calibration session shot in
  portrait. A sensor-side anisotropy is fixed in sensor coordinates and is unchanged by
  rotating the camera; a target-side one inverts about 1.0. We have not run it.

  Two things are already firm. A single scalar square size cannot express an anisotropic
  target, which is a limitation of every calibration path here and in the production
  pipeline. And none of our results turn on the answer: across the nominal board, an
  earlier edge-to-edge reading, and the corner-to-corner one -- a 1.4% spread in assumed
  geometry -- the end-to-end rows of section 6.1 move by at most 0.5 percentage points.

## 9. Conclusion

A laser-camera rig can be calibrated without putting any reference object in the water,
by two arguments that hold for different reasons. The camera half follows from the axial
geometry of a flat port: an in-air calibration plus an analytic correction, whose
advantage over an in-water calibration is not geometric accuracy but the removal of the
in-water session and a 2.1x noise penalty. The laser half follows from the structure of
the dot locus, which is pinhole geometry and therefore not specific to any port: two of
four degrees of freedom are free, and the remaining scale gauge is broken by the apparent
size of whatever the dot lands on -- the dive's own subjects, of unknown size.
End to end, with no reference object ever entering the water, the rig measures a held-out
549 mm target to +0.0% median and 2.6% worst case across 1.1-4.5 m -- inside a 5%
requirement on every frame -- where the same rig without a per-dive laser calibration reads
-5.4% median and 15.8% worst. That comparison isolates the angular calibration, which is
what drifts between dives; absolute scale, tested separately against a calipered object,
comes back within 0.4%. Component-wise the estimator recovers the beam to 0.009 deg in a
pool and to 0.003 deg on a production corpus where the reference object is a slab. Where
it fails, it fails for a stated reason with a stated remedy.

[TODO: implications for citizen-science deployment; future work — dome ports, in-situ
salinity, bending bodies.]
