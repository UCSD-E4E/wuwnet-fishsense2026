# Calibrating an Underwater Camera in Air: Removing the Corrective Optic from a Laser–Camera Rig

*Draft. Sections marked [TODO] are placeholders.*

## Abstract

A laser–camera rig measures fish length from one image: range from a projected dot,
transverse extent from the camera. Behind the flat port of a dive housing the camera is
not a pinhole -- refraction at the pane makes it an *axial* system, whose rays cross the
optical axis over a spread of points rather than at one -- so the deployed reference
implementation restores the air path in hardware, with a wide-angle air lens fitted at
the port, and calibrates the camera in the water.

We replace the optic with a model. An in-air calibration plus an analytic per-pixel
correction reproduces what the in-water calibration achieves, which lets the lens come
off and the calibration target stay dry. The consequence we think matters most is not
accuracy but what it unlocks: **in air the camera is central**, so ordinary multi-view
calibration applies, and the target can be a commodity three-dimensional object rather
than a fabricated, measured plane. No equivalent is available underwater without solving
refractive structure-from-motion first.

We also correct the usual case for the correction. Measured against a rigid target with
a ground-truth-free test, an uncorrected flat port costs far less than the free-field
analysis suggests -- **1.2 % at the median frame and 14.3 % at worst** over every
geometry the production corpus actually photographed -- because the laser dot must land
on the fish, which pins the fish near the optical axis. And a properly fitted in-water
single-viewpoint calibration is geometrically indistinguishable from the analytic
correction, so the case for the latter rests on the wet session it removes, not on the
error it corrects. We report both, together with what a dome port would cost instead.

## 1. Introduction

[TODO: motivation — fish stock assessment, citizen science, why length matters]

A stereo rig measures length from disparity; a laser–camera rig measures it from one
camera plus a projected dot, which is cheaper, smaller and easier to make watertight.
The price is refraction. A dive housing presents a flat pane to the water, and a flat
pane is not optically neutral: rays bend at the interface by an amount that grows with
field angle, and the camera behind it has no single centre of projection at all.

The deployed system solves this in hardware. It fits a wide-angle **air lens** at the
port, restoring an air path so the camera behaves as a pinhole again, and calibrates it
in the water. Both choices cost something operationally. The lens is a part to carry,
fit, and purge of trapped air before the dive; that last step is invisible to the diver
if it goes wrong, and when it does, the result is an occlusion across part of the frame
that has cost this deployment tens of measurements and some repeated calibration work.
And calibrating in the water means the target goes in the water.

This paper removes both by modelling the port instead. That is not a new model -- the
correction is Łuczyński et al.'s, and we implement it unchanged -- and our contribution
is not the model but what follows from being able to calibrate dry.

**The keystone is that in air the camera is central.** Underwater behind a flat pane it
is axial, so multi-view methods that assume a single viewpoint do not apply; recovering
intrinsics there means solving refractive structure-from-motion, which is a research
problem rather than something an untrained operator runs. Move the calibration into air
and that constraint lifts entirely. Ordinary tooling applies, and the target no longer
has to be a printed plane whose pitch must be measured before it can be trusted -- it can
be a commodity three-dimensional object, which is both more accessible and better
conditioned than the plane it replaces.

Our contributions are:

1. **A direct, ground-truth-free measurement of flat-port refraction error**, using a
   homography-residual test that needs no external reference, no known pose and no known
   scale, and which predicts the error's *shape* rather than only its size (§4.1).
2. **A corrected account of what that error costs**, which is much less than the
   free-field figure usually quoted, because the laser geometry constrains where a
   measurable fish can be (§4.2).
3. **A correction to the case for the analytic model**: on geometry alone a well-fitted
   in-water calibration matches it, so its advantage is the wet session it removes (§4.3).
4. **The consequence of calibrating in air** -- centrality, and the commodity
   three-dimensional targets it permits (§5).
5. **Two negative results** that constrain the design space: the flat port cannot supply
   range at this housing's geometry, and the cost of the port is not a property of the
   port alone but of the port together with the laser mount (§4.2, §7).

## 2. Related work

[TODO: Treibitz et al. on flat refractive geometry; Agrawal et al. axial cameras;
Łuczyński et al. 2017; Jordt-Sedlazeck & Koch on refractive SfM; dome ports; underwater
stereo. The SVP-vs-model comparison in §4.3 is the contested claim and must engage
Łuczyński's own evaluation directly.]

## 3. System and problem statement

The rig is an Olympus TG-6 in a commercial flat-port housing with a 532 nm laser module
mounted alongside, offset by `|O|` ≈ 104 mm from the optical axis and nominally parallel
to it. Images are processed in raw sensor coordinates (4014 × 3016); the camera's own
JPEG pipeline applies a radial lens correction which, measured against raw, displaces
corners by ~1 px with radius correlation +0.85 — the same functional form as the
refraction signal, so measuring one through the other would confound them.

Length is obtained as `L = ℓ_px · Z / f`, with `Z` the range read off the laser. The
tolerance the deployment works to is **15 % on length, worst case**, so that is the bar
every number here is held against.

One feature of the geometry does more work than anything else in this paper, and it is
easy to miss: **the dot has to land on the fish**, because that is how the fish is
ranged. The dot sits at scene radius `|O|` from the optical axis at every distance, so a
measurable fish is always near the axis, and only its own extent reaches outward. We
verified this on 227 production frames carrying both a dot and head/tail landmarks: the
dot falls inside the body span in **100 %** of them, at 0.40–0.65 of the body length
from the tail, median 0.52. Section 4.2 shows what that does to the error.

## 4. What a flat port costs

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

### 4.2 What it costs in length, once the laser geometry is accounted for

[TODO: rewrite in full. The structure is settled and the numbers are computed; what
remains is prose.

 - The local/differential decomposition stays: range is read at the dot, extent at the
   fish, and the second difference is small while the first-order term is not.
 - Magnification is rotationally symmetric but **not isotropic**: a span along the radius
   is stretched by the derivative of the radial mapping, a span across it by the mapping
   itself, and the two differ by about a factor of three. `differential_length_error`
   takes `extent="radial"|"tangential"`.
 - The free-field table (+55.6 % radial, +22.1 % tangential at 2500 px, measured against
   the in-water calibration) is **not reachable**, and saying so is the section's point.
   The dot must be on the fish; 2500 px would put the surface ~12 cm from the port.
 - The reachable envelope, over every (length, range) pair in the production corpus with
   the measured dot placement and the worst bearing at each:
       p50 1.2 %   p75 1.4 %   p90 3.0 %   p95 5.5 %   p99 11.5 %   max 14.3 %
       over 5 %: 5.6 % of frames.  over 15 %: none.
 - Real data agrees: the camera-only end-to-end of §6 reads +0.3 % median and 1.8 % worst
   uncorrected, consistent with the typical rows.
 - The design finding: this is small **for this mount**. `O` is offset near-vertically,
   so a horizontally-held fish extends tangentially, the better case by three. A
   horizontally offset laser would put that extent radially and be roughly three times
   worse. The port's cost is a property of port *and* mount, which is actionable and, as
   far as we know, unstated.]

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

### 4.4 The corrective optic this replaces

The reference implementation of this system does not leave the port bare. It mounts a
Backscatter M52 Underwater 81-degree Wide Air Lens at the housing port, restoring an air
path so the camera behaves as a pinhole again, and the companion system paper quantifies
the failure that optic prevents rather than the accuracy it achieves -- its own figure is
the refraction model with the index step removed, which it correctly calls close to
tautological.

**We make no claim about that optic's performance, and do not measure it.** Our
configuration is the bare housing -- the water-to-air focal ratio we report in section 4.1
is the signature of an uncorrected port, and would be near unity with the lens fitted --
and our claim is about what that configuration achieves, not about what the lens fails to.
The two are alternative paths to the same place: one buys the air path in glass, the other
models the water. Which is preferable is a question about availability and cost for the
people who have to assemble the rig, not about optics, and it is not settled here.

### 4.5 How a dome compares, and why accept a refractive port at all

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

## 5. What calibrating in air enables

[TODO: write. This is the paper's keystone and currently exists only as an argument, not
as text or evidence.

 - Underwater behind a flat pane the camera is axial: no single viewpoint. Multi-view
   calibration and structure-from-motion assume a central camera, so they do not apply
   without solving refractive SfM.
 - In air the camera is central, exactly. So every in-air method is available, and the
   choice of target is no longer constrained to a plane whose pitch has been measured.
 - A commodity three-dimensional target — interlocking bricks — is *better conditioned*
   than a checkerboard, not merely cheaper: a planar target carries known degeneracies,
   which is precisely why our own `fx/fy` question (§8) cannot be settled from this
   dataset, every frame sitting within 34° of one axis. A 3D target breaks that by
   construction.
 - It is also more accurate as a reference. A printed board carries a ±1.2 % pitch
   tolerance and ours measured 0.5 % oversize with a disputed anisotropy; moulded bricks
   hold ~±0.01 mm on an 8 mm pitch, published and identical worldwide, and need no
   measuring at all.
 - Status: a pilot is being shot. Report it or state it as enabled-but-untested; do not
   claim it works until it has.
 - Building and grading the target itself belongs to the deployability paper; what is
   claimed here is the enablement.]

## 6. Evaluation

[TODO: rewrite around the camera-only end-to-end, which is the experiment that isolates
the port. Each camera model calibrates the laser with itself and then measures the
board's 549.0 mm span, so the pipeline is internally consistent, as a deployment's would
be. Ten frames, 1.08–4.48 m:

    median   Uncorrected +0.3 %   In-water SVP -0.2 %   Pinax -0.3 %
    worst    1.8 %                1.6 %                 1.6 %

  The honest reading is that on *this* data the correction buys little, because the board
  sits at 175–843 px where the differential term is small — which is the same fact §4.2
  establishes, seen from the other side, not a contradiction of it. Say so. The claim the
  data supports is that the correction removes a term that is small in the median and
  reaches a third of the budget in the close-range tail, not that it rescues the system.]

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

**The port's cost is not the port's alone.** Section 4.2's envelope is small partly
because this rig's laser is offset almost vertically, so a horizontally-held fish extends
across the radius rather than along it. The same port with a horizontally offset laser
would be roughly three times worse. A negative result for anyone reading a flat port's
cost off the port alone.

## 8. Limitations

- **One camera, one housing, one pool.** The refraction results are a matched pair with
  no confound, but `n = 1` in every other sense, and the pool's index was assumed, never
  measured.
- **The reachable envelope is computed, not measured, at its tail.** The p99 and maximum
  of §4.2 come from simulating each real (length, range) pair; our own photographs never
  put a known target in the close-range oblique geometry where the error is largest, and
  the rig cannot be made to — the dot would have to land on a surface 12 cm away. The
  median and p90 are corroborated by §6; the tail is not.
- **Fish bearing is unobserved.** The envelope maximises over it because the corpus does
  not record it. A distribution over real poses would narrow the tail, probably a lot.
- **The water index was assumed.** Fresh water throughout, never measured, and the
  salinity adaptability the model is often credited with is untested here.

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

[TODO: rewrite. The shape: a flat port's refraction can be modelled rather than
cancelled in glass, which removes a part, a step and a silent failure class from the
dive, and moves the calibration target out of the water. What it costs is modest and now
quantified against the geometry the rig can actually produce. What it buys, beyond that,
is centrality: in air the camera is a pinhole, and the methods and targets that follow
from that are not available underwater at all.]
