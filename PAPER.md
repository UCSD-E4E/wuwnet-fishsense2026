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
It is also, because it must come off to be flooded, threaded back on by hand before every
dive -- so the calibration that included it describes one particular mounting, and a
calibration taken with an optic that is regularly disturbed is not the same asset as one
taken through a pane that never moves. And calibrating in the water means the target goes
in the water.

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

**The second consequence is that the result is not about our camera.** The model's
parameters belong to the *housing* -- pane thickness, pane index, and the standoff
between pane and entrance pupil -- together with the water; the camera behind the pane
enters only through its intrinsics, which are calibrated separately and in air. The
refraction error is therefore a function of field angle alone, identical to within 0.01 %
across a threefold change in focal length (section 4.2). Action cameras and phones in
dive housings sit behind flat panes just as our housing does, so the correction transfers
to them unchanged, while the air lens it replaces does not: a corrective optic is a
housing-specific accessory, and for most consumer housings no equivalent part exists. The
hardware this makes measurable is considerably more accessible than the camera we tested.

Our contributions are:

1. **A direct, ground-truth-free measurement of flat-port refraction error**, using a
   homography-residual test that needs no external reference, no known pose and no known
   scale, and which predicts the error's *shape* rather than only its size (§4.1).
2. **A corrected account of what that error costs**, which is much less than the
   free-field figure usually quoted, because the laser geometry constrains where a
   measurable fish can be (§4.2).
3. **A correction to the case for the analytic model**: on geometry alone a well-fitted
   in-water calibration matches it, so its advantage is the wet session it removes (§4.3).
4. **The consequences of calibrating in air** -- centrality, the commodity
   three-dimensional targets it permits, and the transfer of the whole result to any
   flat-pane housing rather than only to ours (§5).
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

Length is obtained by back-projecting the snout and tail-fork pixels through the camera
model, placing both at the laser-derived range `Z`, and taking the Euclidean distance
between them. It is *not* obtained by similar triangles: the scalar form
`L = ℓ_px · Z / f` is a first-order approximation the companion system paper rejects
explicitly, because it assumes the fish lies along an image axis, is exact only at the
principal point, and requires `fx = fy`. All three matter here — §4.2 covers both the
off-axis error and how the fish is held, and §8 reports a measured ~1 % `fx/fy`
anisotropy. Underwater the back-projection runs through the port model rather than
`K⁻¹`, and that substitution is what this paper is about.

The tolerance the deployment works to is **15 % on length, worst case**, so that is the
bar every number here is held against.

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
in camera, board, or operator. 63 in-air and 121 in-water frames yielded corners.

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
homography residual climbs **2.4×**, monotonically, from the image centre to 1370 px.
Both corrections are nearly flat over the same span and end well below uncorrected.

Two exclusions set that range, and the second matters more than the first. Radial bins
holding fewer than 200 corners are not used: a board reaches the extreme corners only
occasionally, and a median over a handful of corners is noise. But the frames that *do*
reach those corners are also not a random sample. A board fills the frame's corners only
when it is close, and close is where a flat port departs furthest from the single
viewpoint a homography assumes — so those 15 of 121 frames sit roughly six times higher
at *every* radius, the image centre included. Pooling them in makes part of the curve's
rise a change in which frames are being averaged rather than a change in radius: the
1369–1598 px bin drew 85 % of its corners from them. The profile is therefore taken over
the 106 frames that stay inside the well-sampled range. Pooled over all 121 the same
curve reads 5.1×, and the difference between that and 2.4× is composition, not signal.
The close-range frames are not discarded from the study — they are where §4.2 and §6
live — they are simply not averaged into a radial profile they would dominate.

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
   far as we know, unstated.
 - Close the section by tabulating against **field angle rather than pixel radius**, and
   say why: with the pane fixed, the error is a function of field angle alone. Held at
   f = 1329, 1751 and 2876 px the same table comes back identical to better than 0.01 %:
       2 deg  -0.03 %    5 deg  0.40 %    8 deg  1.19 %
      12 deg   2.87 %   20 deg  8.70 %   35 deg 33.68 %   (radial extent)
   The envelope above is angular for the same reason -- the dot sits at atan(|O|/Z) and
   the fish's far tip no further than atan((|O| + L/2)/Z), neither involving the lens. So
   these numbers describe the housing, not the camera, which is what §5 then trades on.
   Presenting the table this way costs nothing and is what makes the transfer claim
   checkable rather than rhetorical.]

**And what it costs in range, which is not the same answer.** The cancellation above is
specific to length. Read the *range* out of the same reconstruction and an uncorrected
flat port is **25.0 % short**, flat with distance — it is `1/n_w - 1`, and on the pool
frames it puts the board at 0.88 m where both corrected models put it at 1.20 m. Nothing
in a length measurement reveals this, because the same factor inflates the angular extent
by exactly as much as it shrinks the range, and the two cancel.

It does not affect any number in this paper. FishSense Lite delivers a length; range is an
intermediate, and §6 shows the three camera models agreeing on length to within 1.8 %
precisely because of that cancellation. We state it because single-laser ranging is
established practice in ROV survey work, so a reader may reasonably expect range to be an
output of a rig like this one — and if it ever becomes one here, an uncorrected port is
wrong by a quarter from the first frame.

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

One asymmetry between them is structural rather than optical, and it does not depend on
how well the optic performs. **The lens is not a fixed part of the camera.** It threads on,
and it has to come off to be flooded, so its pose relative to the entrance pupil is
re-established by hand before every dive. Whatever calibration was taken with it fitted
therefore describes *that* mounting. A flat pane is bonded into the housing and is never
removed; its geometry is fixed for the life of the housing, which is what makes a single
in-air calibration valid indefinitely rather than until the next time the optic is
disturbed.

This matters more than it first appears, because it interacts with which intrinsic the
measurement is sensitive to (section 5). Under the per-dive beam re-fit the system
actually uses, a shift in the *principal point* is absorbed almost entirely -- twenty
pixels costs a tenth of a percent of length -- so a refit that merely decentres the optic
is nearly free. What is not absorbed is a change in effective *focal length*, which
reaches the length one for one. So the demanding axis is the one along the optical
axis, and on a threaded mount that is the axis set by how hard the ring was done up.

**We have not measured this, and it should not be asserted without measuring.** It is not
a dive experiment: fit the optic, calibrate in air, remove it, refit it, calibrate again,
ten times over. The scatter in the recovered focal length *is* the repeatability, and the
budget in section 5 converts it to a length error directly. We report the sensitivity and
flag the quantity as open (section 8).

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
 - Building and grading the brick target itself belongs to the deployability paper; what is
   claimed here is the enablement.

 CLOSE THE LOOP. "Good enough to replace the wet session" now has a number, from a
 simulation that shares no code between the forward model and the inverse: the exact
 axial projection images the scene, Pinax inverts it, and only the inverse is given
 the wrong intrinsics. Written up as tests/test_calibration_budget.py.

 - Report the two regimes, because they have **opposite** sensitivities and getting
   them the wrong way round would mean optimising the brick target for the wrong parameter.
   Trusting the laser's bench extrinsics, range and extent scale together so a focal
   error cancels out of the length (5 % of focal length -> 0.1 % of length) while the
   principal point is ruinous (5 px -> 5.5 %), because the dot's offset from the
   principal point *is* the triangulation baseline in the image and at 4 m that offset
   is only ~80 px. Re-fitting the beam every dive -- which is what §6 actually does --
   absorbs the principal point almost entirely (20 px -> 0.1 %) and in the same motion
   destroys the cancellation, so the focal error arrives one for one (5 % -> 4.8 %).
 - So the in-air calibration's binding parameter is the **focal length**, and an
   anisotropy reaches only the axis it lies on: `fx` 1 % high costs 1.0 % on a fish
   held across the frame and 0.001 % on one held along it. That is the cleanest
   statement of why the brick target is three-dimensional.
 - The closing number: the measured in-air calibration (fx +0.033 %, fy +0.017 %,
   principal point within a few px) propagates to **0.04 % of length**, against the
   15 % worst-case budget of §8. The floor underneath every figure here is Pinax's own
   approximation against the exact model, 0.0003 %.
 - Two sensitivities of the brick target, same method. Misspecifying the brick chamfer by
   +/-0.15 mm moves fx by less than the estimator's own scatter, so the one constant
   that is not published precisely is not binding. Assembly slop is: about 1 % of
   length at 0.2 mm of per-brick placement error, and it leaves the fx/fy ratio alone.
   That is the real limit on "exact by construction" -- how well it is *built*, not how
   well it is known.

 SECOND ENABLEMENT: the hardware it opens up. Argument and supporting numbers settled;
 needs writing.

 - The model is parameterised by the *port*, not the camera: pane thickness, pane index,
   standoff, water index. The camera enters only through intrinsics, which are now
   calibrated in air by ordinary means. Nothing in the correction is specific to a TG-6.
 - Verified, and worth stating as a result rather than an assertion: with the pane held
   fixed, the length error is a function of field angle alone. Tabulated against field
   angle it is identical across f = 1329, 1751 and 2876 px to better than 0.01 % --
   -0.03 % at 2 deg, 1.19 % at 8 deg, 2.87 % at 12 deg. A wider lens does not make the
   correction larger; it maps the same angles onto fewer pixels.
 - Nor does a wider lens enlarge the reachable envelope of §4.2, because that envelope is
   angular too: the dot sits at atan(|O|/Z) and the far tip of the fish no further than
   atan((|O| + L/2)/Z), neither of which involves the lens. For our mount that is 6.7 deg
   and 15.0 deg at 1 m, falling to 1.5 and 4.7 deg at 4.5 m -- the same envelope, and the
   same error inside it, for any camera behind the same pane.
 - So the conclusion transfers to action cameras and to phones in dive housings, which
   are flat-pane almost without exception. This matters because the thing being removed
   is the part that does *not* transfer: the M52 air lens of §4.4 is an accessory for one
   housing, and for most consumer housings no equivalent exists. A dome is likewise a
   specialist part (§4.5). Modelling the pane is the only one of the three routes
   available to a diver who already owns a camera.
 - Scope honestly. What is demonstrated is one housing; what is argued is transfer, and
   the argument rests on the model's parameterisation plus the field-angle collapse, not
   on having tested a second camera. Say that. The per-housing quantities -- thickness,
   index, standoff -- still have to be obtained for each housing, and section 4.1's
   homography-residual test is what obtains them without a reference.]

## 6. Evaluation

[TODO: rewrite around the camera-only end-to-end. It does **not** isolate the port —
saying so was backwards, and the figure now contradicts it. What it shows is that at
reachable geometry the port is nearly invisible in length: each camera model calibrates
the laser with itself and then measures the board's 549.0 mm span, so each pipeline is
internally consistent, and all three land. Ten frames, 1.08–4.48 m:

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
  put a known-size object in the close-range oblique geometry where the error is largest, and
  the rig cannot be made to — the dot would have to land on a surface 12 cm away. The
  median and p90 are corroborated by §6; the tail is not.
- **Fish bearing is unobserved.** The envelope maximises over it because the corpus does
  not record it. A distribution over real poses would narrow the tail, probably a lot.
- **The water index was assumed, and this is what that can cost.** Fresh water
  throughout, never measured, and we hold no salt-water imagery at all -- so the salinity
  adaptability the model is often credited with is untested on anything we photographed.
  What can be done is bound it, and `tests/test_water_index.py` does, against Quan &
  Fry's empirical index over salinity, temperature and wavelength. Calibrating in fresh
  water and diving in the sea costs **under 0.8 %** of length at §4.2's reachable field
  position, worst case across 30-35 PSU and 0-30 °C. Two presets cut that to **under
  0.3 %**, and re-centring them halves it again: 1.333 and 1.342 are round numbers
  inherited from the Pinax paper, and each sits nearer one end of the envelope it stands
  for than the middle of it (the midpoints are 1.3344 and 1.3406).

  The asymmetry is the part worth keeping. Pinax takes the index as a parameter, so
  adapting is a change of constant; an in-water calibration absorbs it into the fitted
  focal length and reaches the same accuracy only by way of another session in water of
  the right salinity. Both degrade at much the same rate when the index is wrong -- the
  error is in the physics, not in the correction -- and what separates them is the price
  of being right, which is the whole of this paper's argument seen from one more angle.

  Two things cap how well any of this can be known, and both are unmeasured here. The
  index depends on wavelength as much as on salinity: across the visible band it moves by
  0.0082, the same order as the entire fresh-to-sea step, so an index quoted without a
  wavelength is underspecified and refining one below about 0.002 is not meaningful. And
  because water attenuates red faster than blue, the effective index for a broadband
  silhouette climbs with range -- about 0.0018 from 1 m to 4.5 m in clear water, worth
  0.13 pp, and *range-correlated* rather than constant, which is the category that
  reaches the beam fit instead of cancelling against it. The laser dot is monochromatic
  and has no such problem. In turbid water, which absorbs blue rather than red, the drift
  reverses sign.
- **The corrective optic's refit repeatability is unmeasured.** Section 4.4 argues that a
  threaded optic which must be removed to be flooded cannot be in the same place twice,
  so a calibration taken with it fitted describes one mounting. That argument is
  structural and we believe it; the *size* of the effect we do not know, and we are
  careful not to imply otherwise. It is cheap to settle and needs no water -- fit,
  calibrate in air, remove, refit, repeat -- and until someone does, the comparison in
  section 4.4 is a statement about what has to be re-established, not about how much it
  costs. Our own route has no removable element, so the question does not arise for it,
  which is the only reason we can leave it open.

- **A 1% `fx/fy` anisotropy we can characterise but not yet attribute.** With a nominal
  square object model our in-air calibrations return `fx/fy` = 0.9931, and the production
  fleet returns 0.99141 +- 0.00044 across seven cameras from seven independent
  calibrations -- real, reproducible, and eight times better determined than the focal
  length those calibrations were measuring. Two explanations fit: an anisotropic sensor
  or detector, fixed in sensor coordinates; or an anisotropic printed board, fixed in the
  board's own frame. The fleet-wide agreement does not separate them, because all seven share a board
  design.

  We calipered the board. Corner to corner its pitches are 42.2308 mm (long axis) and
  42.1111 mm (short), a 0.28% anisotropy in the *opposite* direction, which makes the
  camera-side term larger rather than smaller: 0.97%. On that reading the board is not
  the cause. We do not present it as settled. The short axis spans only 9 pitches, so a
  1 mm reading error is 0.26% of it; a reading of 377.5 mm would make the camera exactly
  square; and two careful attempts at that span differed by 5 mm. The question lives
  inside the precision of a rule.

  The test that decides it needs no length measurement: one calibration session shot in
  portrait. A sensor-side anisotropy is fixed in sensor coordinates and is unchanged by
  rotating the camera; a board-side one inverts about 1.0. We have not run it.

  Two things are already firm. A single scalar square size cannot express an anisotropic
  board, which is a limitation of every calibration path here and in the production
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

## Figures

All are written by `uv run python refraction_analysis/make_figures.py` and
`refraction_analysis/make_target_figures.py`, as PDF and PNG with the same stem.
Captions live in this table and not inside the images, so that setting the paper
does not mean re-rendering a figure to reword a sentence; each becomes a
`\caption{}` as it stands.
This paper's land in `figures/`; figures belonging to a sibling paper are written
straight into that paper's handover folder, so ownership is where the generator
puts the file rather than a note someone has to remember. **Placeholder** means the figure stands in for a measurement not
yet taken; each carries a visible banner saying so, and must not go to camera-ready
in that state.

| section | file stem | status | caption |
|---|---|---|---|
| §3 | `sim-flat-port-angular-compression-2-m-salt-water` | final | What a flat port does to the image. A direction out in the water is imaged as though it lay at a steeper angle, so the scene is compressed into the frame: at this camera's 41.4 deg corner the true direction is 29.5 deg. The ratio is 1/n_w to three decimals near the axis, which is why the effect passes for ordinary radial distortion and why an in-water calibration absorbs most of it. Drawn at 2 m; the mapping moves only in the fourth decimal between 0.5 and 5 m, and that movement is the part no single-viewpoint model can represent. |
| §3 | `sim-uncorrected-radial-error-at-the-image-plane-2-m-salt-water` | final | The same compression in pixels rather than degrees: how far an in-air calibration misplaces a point at 2 m. 131 px at 10 deg off axis, 286 px at 20 deg, 898 px at the frame corner. The monotonic radial shape is the signature §4.1 then looks for on real frames, where it appears as a homography residual growing with radius. |
| §4.1 | `pool-homography-residual-vs-radial-position-underwater-frames` | final | The ground-truth-free test, on the underwater frames. Each frame's corrected corners are fitted to the board's own plane by a homography: a correct camera model leaves only detection noise, a wrong one leaves radial structure. Uncorrected climbs 2.4x from the image centre to 1370 px while both corrections stay flat and well below it. Computed over the 106 of 121 frames that stay inside the well-sampled range -- the 15 reaching further are close-range frames sitting about six times higher at every radius, so pooling them would make part of the rise a change of frame population rather than of radius. Bins holding fewer than 200 corners are not used. |
| §4.2 | `pool-differential-magnification-at-fish-vs-at-laser-dot-line-fish-along-the` | final | Why a small local distortion is not a small length error. Right: the non-projective distortion across a fish-sized 295 mm span is under half a percent for every pipeline, which taken alone would say the correction is barely worth applying. Left: the term that actually reaches the measurement, the difference in magnification between the fish and the laser dot that ranges it, which for an uncorrected port grows without limit off axis. Two cautions. In-water SVP is the reference for this comparison and so scores zero by construction rather than by merit; and the wide-radius end of the left panel is not reachable, because the dot has to land on the fish, which pins it near the dotted line. |
| §4.2 | `sim-flat-port-error-vs-field-angle-by-focal-length` | final | Three lenses spanning a threefold change in focal length, same pane. The curves are identical to better than 0.01 %, so the correction transfers to any flat-pane housing — action cameras and phones included. |
| §4.2 | `sim-length-error-vs-field-position-0-3-m-fish-at-2-0-m` | final | A 0.30 m fish at 2 m, moved out from the optical axis. Left: every pipeline, on a scale the uncorrected curve sets. Right: the three corrected treatments alone -- in-water SVP, Pinax and the exact axial model -- at a scale where they can be told apart. The same reachability caution applies as to the figure above: the far end of the axis is not attainable with the dot on the fish. |
| §4.2 | `sim-uncorrected-range-error-vs-length-error` | final | A 300 mm fish with the dot on it, laser offset 104 mm. The uncorrected range error is flat because it is 1/n_w; the length error is what survives the cancellation, and it grows toward close range where the fish subtends a larger angle. Simulation, so the truth is known rather than derived through a camera model. |
| §4.2 | `pool-range-by-camera-model` | final | Ten pool frames. Every range is absolute, solved from the board's calipered geometry — the object supplies the metric scale, the camera model only the angles. The reference is the mean of the two corrected models, which share no data and differ by 0.6 %: one is fitted to underwater frames and holds no refraction theory, the other comes from the in-air calibration and never sees one. An uncorrected port sits 26 % below both. |
| §4.3 | `sim-in-water-svp-length-error-vs-range-by-housing-geometry` | final | What an in-water single-viewpoint calibration cannot absorb. A flat port hard against the lens is almost a pure angular map, and radial distortion fits an angular map at every range; what survives is the sideways offset a ray accumulates crossing the pane, whose relative size grows as 1/Z. So an SVP model is exact only at its calibration distance (dotted) and degrades with the thickness of the pane and the air gap behind it. Log scale. |
| §4.3 | `sim-length-error-from-calibration-noise-300-trials` | final | Calibration noise, 300 trials per model, as violins. Each model is given the pixel noise its own session actually shows -- the in-water calibration the larger figure measured underwater, Pinax the smaller one measured in air -- so the comparison carries the real penalty of calibrating in water rather than assuming both are equally precise. The result is the one we did not expect going in: noise is not what separates these two models. |
| §4.3 | `sim-range-error-vs-distance-noise-free` | final | Range error against true distance, noise-free, log scale. The number this panel exists to produce is the dome's: decentred by 5 mm it reads 2.4 % to 2.6 % long and essentially flat across 0.5-5 m, because a decentring is an angular error and the laser dot sits near the axis at every range. That is the state of practice with a poor build, and a corrected flat port sits well inside it. |
| §4.3 | `sim-range-error-vs-labeler-noise-by-direction` | final | Range error against dot-placement noise, where the direction of the noise matters more than its size: displaced along the epipolar line (left) against across it (right). The noise model is the labeler, because the dot is placed by a human or by a detector a human audits. Production measures those labels collinear to 2.9 px at worst, with accepted detector predictions a median 1 px off the dive line, so the shaded band marks the range the delivered corpus actually occupies and everything to its right is an open-ended sweep rather than a deployment condition. |
| §4.5 | `sim-what-an-in-air-calibration-costs-by-port-type` | final | Why accept a refractive port at all. A dome whose centre of curvature sits at the camera's entrance pupil bends nothing: every ray meets both glass surfaces at normal incidence and an in-air calibration is valid underwater exactly, not approximately. Left: both port types on one scale, where the flat pane's error is an order of magnitude larger than any dome's. Right: the domes alone, zoomed, at decentrings of 0, 2, 5 and 10 mm -- a dome's only error is the error of not assembling it perfectly. |
| §5 | `sim-calibration-error-budget-by-laser-regime` | final | Trusting the bench, range and extent scale together and the focal error cancels; the dot sits ~80 px off axis at 4 m, so the principal point is ruinous. Re-fitting the beam absorbs the principal point and, by pinning the range, destroys that cancellation. Dotted line is the 15 % worst-case budget. |
| §5 | `sim-target-sensitivity-to-build-quality` | final | Misspecifying the chamfer by ±0.15 mm moves fx by less than the estimator's own scatter. Assembly slop does bind — about 1 % of length at 0.2 mm — and it leaves the fx/fy ratio, the quantity the tower is for, alone. |
| §5 | `target-detection-on-one-view` | **placeholder** | The brick target found in a single view: brick faces matched to the model outlined in magenta, faces found but not matched in grey, and the lattice points the pose is solved from marked as dots. Placeholder -- this is a rendered view, and nothing in this paper has yet been run against a photograph of the built target. |
| §5 | `target-focal-ratio-vs-view-count` | **placeholder** | Both stay inside ±0.1 % from four views on. The anisotropy being recovered is 1.09 % — about thirty times the residual error — and no view here is rolled, which is the case a planar board cannot settle. Placeholder: these are real fits, but to rendered views rather than photographs of the built target, and `classify_colours` is the part that will need retuning on real frames. |
| §5 | `target-pool-length-lego-vs-checkerboard` | **placeholder** | Placeholder: this experiment has not been run, and the plotted values come from a fixed seed. It will show the 312.5 mm decoy measured against two in-air calibrations — one on the brick target, one on the checkerboard — corrected identically, so the calibration target is the only thing differing. §4 settles the correction; this settles the reference. |
| §6 | `pool-end-to-end-board-span-by-camera-model` | final | Ten pool frames; each model calibrates the laser with itself, so every pipeline is internally consistent. Median +0.3 / -0.2 / -0.3 %, worst 1.8 / 1.6 / 1.6 %, in legend order. All three land on the calipered span, including the uncorrected one: its range and magnification errors are both about a quarter, and they cancel. Length cannot separate these models — §4.1's ground-truth-free test is what does. |
| §6 | `pool-decoy-length-by-camera-model` | final | 12 of 13 decoy frames; 1 rejected for a mask whose aspect ratio misses the calipered 2.95 by more than a quarter. The uncorrected model ranges 25 % short of the in-water calibration and Pinax sits within 0.6 %, yet all three read the same length (median -2.6 / -2.8 / -2.7 %, in legend order). Length cannot see the error that range shows plainly. |
| §8 | `sim-length-error-from-a-water-index-mismatch` | final | What assuming the water index costs. Both models are calibrated for fresh water and the true index is then swept across it; both degrade at the same rate, because the error is in the physics and not in either correction. What differs is the price of fixing it: Pinax takes the index as a parameter, while an in-water calibration needs another session in water of the right salinity. Note the geometry -- the fish sits at half the frame half-width here, further off axis than §4.2's reachable envelope allows, so these are upper bounds; §8 gives the reachable figures. |

Three of 21 are placeholders. Two await photographs of the built calibration target:
the detector currently runs on rendered views only. The third is §5's, which compares
the two in-air targets against each other and so cannot be drawn until the brick
target has been shot; its axes are real -- the pool session's range spread, the
decoy's calipered 312.5 mm -- and its values come from a fixed seed.

§5's and §6's figures are deliberately one question each. §5's varies only the target
the camera was calibrated on, both calibrations in air and both corrected identically,
because §5's claim is that a tower of bricks is an adequate reference. §6's varies only
the correction, because §6's claim is about the stack end to end. An earlier draft
varied both at once, which left the port correction as an unnamed constant in two
series and replaced it in the third.

Three finished figures about the laser's drift and its per-dive recalibration have
been handed to **P1, the rig paper**, and now live in `correspondence/p1/figures/`:
mount movement between sessions is hardware behaviour, and this paper is about the
port. `laser_calibration.ipynb` and `laser_per_dive.ipynb` stay here and write
there. §6 still leans on a per-dive laser calibration for the end-to-end
measurement, so it cites that work rather than plotting it.
