# Critique of the draft

Written as a program-committee review of my own draft, then a fix list. I have tried to
argue against the paper rather than for it.

---

## A. One structural problem that outranks everything else

### A1. ~~The draft quietly drops a conclusion my own analysis reached~~ — RESOLVED 2026-09-15

`laser_per_dive.ipynb` concludes, in as many words:

> **So this is not a flat-port result.** The locus is pinhole geometry and the range must
> come from outside the imagery — the same with a dome.

The draft never says this. It is titled "Flat-Port Laser–Camera Rig", its abstract binds
the two halves with "For the camera... For the laser...", and §5 describes the locus
without once noting that a dome would give the same line. A reader finishes §5 believing
the flat port enabled the laser method. It did not. What the flat port supplies is the
*correction* that makes the locus straight, and a dome needs a correction too.

This was the most serious thing in the draft, because it is not an error of analysis but
of presentation, and it is the kind a reviewer who knows the field will catch immediately
and read as overselling.

**Fixed.** The title drops "Flat-Port" and now reads *Calibrating an Underwater
Laser-Camera Rig Without an In-Water Reference*, which is the claim actually supported.
The abstract says the laser argument "is not specific to a flat port... a dome-ported rig
can adopt it unchanged". A new section 5.3, *Scope: what the flat port contributes, and
what it does not*, states the limit in full and notes that the one route by which the port
could have been load-bearing for both halves was tested and fails by two orders of
magnitude. The contributions list separates the port-dependent and port-independent
claims, and the conclusion presents them as two arguments that hold for different reasons.

Worth noting: stating the limit *strengthened* the paper. Section 5 now reads as a method
with a broader domain than the title suggests, rather than as a result the title is
trying to borrow authority for.

### A2. The two halves are weakly joined — PARTLY ADDRESSED

Strip A1 away and §4 and §5 share a dataset and a rig but not a result. §4 could be a
short note; §5 could be its own paper. A reviewer will ask why this is one submission.
Two defensible answers: (i) the contribution is the *end-to-end* claim that no reference
object need enter the water, which needs both halves; (ii) WUWNet short-paper scope
favours a system result over two fragments.

**(i) is now taken and demonstrated** — section 6.1 is the experiment that joins the
halves, and the title states the joint claim rather than a port-based one. What remains
is presentational: sections 4 and 5 are still written as two self-contained studies that
meet only in 6.1. A reviewer may still ask for a tighter thread, and the honest answer is
that the halves are joined by the *corrected directions* and by the deployment claim, not
by a shared mechanism.

---

## B. Major: claims the evidence does not reach

### B1. ~~There is no end-to-end validation~~ — RESOLVED 2026-09-15

The paper claims a rig can be calibrated with no in-water reference. It never once
measures an object of known length using **in-air Pinax + reference-free laser** and
reports the error. Every result is component-wise: refraction residuals here, beam
angles there. The implied experiment is a one-liner —

> take the pool session; calibrate the camera from the garage frames only; calibrate the
> laser from the decoy's apparent height only; measure the *board* (0.454 m, never used
> in either calibration) and report the error distribution against range and field
> position.

— and the data to run it already existed.

**Run.** Camera from the garage frames only + Pinax; laser from the decoy's apparent body
height only; board (546.0 mm, in neither calibration) measured over ten frames at
1.1-4.6 m. Reference-free **+2.4% median / 3.6% worst**, against **-5.7% / 16.4%** for the
same rig with no per-dive calibration and **-0.3%** for the slate scored in-sample. Folded
into the draft as section 6.1, and it is now the abstract's lead.

Three things the run added that the critique did not anticipate. (i) The reference-free
error is *bias*, not scatter, and it matches the 0.270 deg of beam drift between the decoy
and board sessions — so a leave-one-out variant within a session, which removes the drift,
lands at +1.0% / 2.7%. Both are reported; the cross-session number is the honest headline
because a real dive cannot observe its own drift either. (ii) The conventional baseline had
to be labelled in-sample, because it is fitted on the frames it is scored on; presenting it
as a competitor would have flattered it. (iii) The board's long- and short-axis spans
disagree by 0.4 pp in *every* row, which is the section 8 anisotropy showing up end to end
— a camera-side term neither laser calibration can be blamed for.

### B2. ~~"Indistinguishable from SVP" vs "Pinax differs by 2.9%"~~ — RESOLVED 2026-09-15

§4.3 reports homography residuals of 1.16 vs 1.26 × 10⁻⁴ and calls the two corrections
indistinguishable. §4.2's table then shows Pinax disagreeing with SVP by up to **2.9%**
in differential length. Both cannot be casually true. Either the residual metric is
insensitive to the thing that matters for length, or the 2.9% is SVP's error rather than
Pinax's — and the paper cannot tell, because **SVP is the reference and reads zero by
construction**.

The draft inherits this from the notebook, which states the caveat; the draft's table
does not. As written, the Pinax column invites the reading "Pinax is accurate to 3%",
which is unsupported. **Fixed** by taking the first option: section 4.2 now states that both columns are
differences from SVP, that the Pinax column is a disagreement rather than an accuracy, and
that the experiment cannot attribute it. The uncorrected column survives the caveat on its
own terms (21% dwarfs a 2% inter-model disagreement), so the paper still measures the cost
of ignoring refraction; what it now declines to claim is Pinax's accuracy. Section 4.3
additionally notes that the homography metric, unlike the length comparison, privileges
neither model. An external length reference spanning the field would settle it and this
dataset has none.

### B3. ~~The only externally checkable number is wrong by 12%~~ — RESOLVED, and it changed the claim

§6.2 is honest that the decoy's `p_D` reference is a slate calibration from ten minutes
later, so 3.8 px is an upper bound rather than an error. Granted. But then note what is
left: the *one* quantity in §5–§6 with true external ground truth is the recovered decoy
size, and it comes back **274 mm against a measured 312.5 mm, −12%**.

The paper explains this correctly (yaw foreshortening on the oblique far frames) and
uses it to motivate height over length. A hostile reviewer will still write: "the
authors' only ground-truthed output is 12% wrong." The defence has to be made explicitly
and early — that height, the measure actually recommended, has no independent size check
in this dataset *because the decoy's height was never measured* — and then the obvious
question is why not. **Measure the decoy's body height.** It is a tape measure and it
converts the paper's weakest number into its strongest.

### B4. ~~The tolerance budget is never stressed~~ — PARTLY RESOLVED

ε = 15% appears in §3 and every subsequent "within budget" / "40× margin" claim depends
on it. 15% of a 30 cm fish is 4.5 cm. A reviewer from fisheries will ask whether that is
scientifically useful, and if the answer for some application is 5%, then Tier 1 at 4 m
(2.7%) is fine but the design-axis route is out by 3×, and the near-range-ratio pairs in
§6.1 fail. The paper should **report results against ε rather than fix ε**: one figure of
achieved error versus required tolerance, with the methods as curves, and let the reader
locate their own application.

**Done for the end-to-end result** (section 6.1 now tabulates the fraction of frames inside
2 / 5 / 10 / 15%), and it changes the story for the better: the reference-free pipeline
clears **5%** on every frame, so the claim no longer depends on 15% being acceptable. Still
to do: the same treatment for the component results in 6.2-6.4 and the negative results in
section 7, several of which are still stated against a fixed 15%.

### B5. ~~The production corpus is doing ambiguous work~~ — RESOLVED 2026-09-15

Three problems. (i) The draft says "the same measurement pipeline" without saying whether
those rigs use flat ports — if they do not, the section is evidence *against* the
paper's framing (see A1) and should be presented that way. (ii) Our estimator and the
reference `fit_phi_joint` run on the *same frames and the same objects*; that is a
consistency check between two estimators, not independent validation, and "validation"
overstates it. (iii) The comparison is against a known-length fit, so the strong claim —
"spends no known length" — is about our side only; the reference is as good as its
references.

**Fixed.** It is now called a cross-check on independently collected data, explicitly not
an independent validation, with the reason given: both estimators run on the same frames
and objects, so what it shows is agreement between two estimators, one of which spends
known lengths. The housing question is settled the other way from what I expected --
those rigs are not all flat-ported, which makes the section a test of section 5 alone and
is consistent with section 5.3's claim that the laser half is port-independent.

---

## C. Internal inconsistencies a careful reader will find

### C1. ~~A 2.2 px model floor and a 0.31 px locus fit~~ — RESOLVED, and my explanation was wrong

§7 reports "a 2.2 px floor of structured model error" for the flat-port model on this
housing. §5.1 reports a locus line-fit residual of **0.31 px**. A reader will ask how a
0.31 px fit is possible inside a 2.2 px model error.

I claimed the answer was that the ten dots span only 2-8 deg of field where model error is
small, while 2.2 px is a whole-frame number driven by the corners. **Measured, that is
false.** The Pinax homography residual is essentially flat with radius: 0.37 px median
inside 200 px of the centre, 0.46 px at 1800 px, 0.37 px over the whole frame. There is no
corner blow-up to appeal to.

The real reconciliation is that the two numbers are **different fits**, not different field
positions. The 2.2 px comes from a seven-parameter fit (log-scale plus rigid pose) and the
0.37 px from an eight-parameter homography that absorbs an affine map of the plane; the gap
is what the extra freedom absorbs. Section 7 now says so.

And the consequence I wanted the paper to own is also wrong, in the paper's favour. The
board-session dots do sit in a central patch (109-394 px), but the **decoy session behind
the end-to-end headline spans 149-2162 px, 2.9-39.6 deg** -- most of the frame. Section 5.1
now gives both numbers. The method is exercised across the field, not in a favourable
corner of it.

### C2. "Agreement to 0.9%" may be a systematic, not agreement

1.3214 against 1.333 is presented as confirmation. It is also a 0.9% discrepancy with no
error bar and no model. Magnification ≈ `n_w` is a paraxial statement; a focal length
fitted over a ±20° field is a weighted compromise, so a percent-level offset is expected
and ought to be *predicted*, not treated as noise. Predicting it would strengthen the
section considerably; leaving it invites "your headline agreement is the same size as
your unexplained anisotropy" (§8), even though the anisotropy in fact cancels in a
water/air ratio.

### C3. The 5.4× radial ramp has a self-inflicted conservatism the paper mentions but does not quantify

Board detection failed on 31 of 152 underwater frames, disproportionately at high radius
— exactly where the effect is largest — and partial boards are rejected. The draft notes
this makes the curves conservative. A reviewer will want the *size* of the conservatism,
or an analysis accepting partial boards (a homography absorbs the unknown offset, so
this is cheap).

> **Updated 2026-09-20, and the ramp is now 2.4×, not 5.4×.** Looking for the size of
> this conservatism turned up a second selection effect running the other way, and a
> larger one. The frames that reach high radius are the close ones, and a close board is
> where the port departs furthest from the single viewpoint a homography assumes; they
> sit about six times higher at *every* radius, centre included. Only 15 of 121 frames
> reach past 1598 px, and they supplied 85 % of the 1369–1598 px bin — so the pooled
> curve's outer half was measuring a change of population, not of radius. §4.1 now takes
> the profile over the 106 frames that stay inside the well-sampled range, which gives
> 2.4× over 0–1370 px; pooled over all 121 it reads 5.1×.
>
> C3's original question stands and is unaffected: detection failure still censors high
> radius, and accepting partial boards is still cheap. But it is now a question about a
> 2.4× ramp, and the two effects push opposite ways — censoring makes the curve
> conservative, pooling made it look steeper than it is.

---

## D. Minor, but they will be noticed

- **Title/abstract overclaim "reference-free."** `|O|` is measured once on a bench and the
  camera needs an in-air board. The accurate phrase is *no in-water reference*. Fix the
  title to say so; it costs nothing and removes an easy attack.
- **n = 1 everywhere.** One camera, one housing, one pool, one salinity, one laser. §8
  says so, which helps, but the abstract does not.
- **Ten dots.** The headline pool result rests on ten points; §6.1's best-conditioned row
  rests on seven pairs drawn from those ten (not independent).
- **Segmentation dependency.** A text-prompted segmentation model with gated weights is a
  heavy and fragile dependency for a method sold on deployability. Show the method works
  with a cruder mask (the paper has the negative result — chroma and GrabCut both failed —
  and should report it as a limitation rather than omit it).
- **Availability of range spread is assumed, not measured.** The estimator needs ≥1.7×
  range spread on one object. How often do real survey dives supply that? The production
  corpus can answer it and does not.
- **`|O| = 104 mm` is baked into every budget number.** State the budget as a formula in
  |O| so other rigs can use it.
- **Fish bend.** Acknowledged in §8, but height at the dot is defended only against yaw.
  Roll and body flexion both change it; no data.
- **Related work is a TODO** and the Pinax comparison is the paper's most contested claim,
  so it must engage Łuczyński et al.'s own evaluation directly, not just cite it.

---

## E. What is actually strong, and should be led with harder

For balance — these are the parts a reviewer should end up defending:

1. **The homography-residual test.** Ground-truth-free, pose-free, with a *shape*
   prediction (radial growth) rather than a magnitude, so it is falsifiable in a way
   magnitude comparisons are not. It is also immune to the board-anisotropy worry of §8,
   because a homography absorbs an affine scaling of the plane. Say that.
2. **The local/differential decomposition.** The observation that the obvious measurement
   understates the error by ~50× because range and extent are sampled at different field
   positions is a genuinely useful correction to how this class of system is evaluated,
   and it generalises beyond flat ports.
3. **The negative result on Pinax vs SVP.** Publishing "the standard justification for
   this model does not hold on geometry; here is what does" is more valuable than another
   confirmation, and it is the kind of thing that gets cited.
4. **The apparent-size gauge-breaking argument.** Short, exact, and it converts an
   unknown-size object into a calibration target. The failure mode is characterised and
   has a one-line remedy.
5. **The three negative results in §7.** Each closes a route a reader would otherwise try.

---

## F. Priority fix list

| # | fix | cost |
|---|---|---|
| 1 | ~~Run the end-to-end experiment (B1)~~ | **done** — section 6.1 |
| 2 | ~~State the limitation and retitle (A1)~~ | **done** — new section 5.3, new title |
| 3 | ~~Measure the decoy's body height (B3)~~ | **done** -- 106.1 mm; changed the claim, see B3 |
| 4 | Replace fixed ε with error-vs-tolerance | **done for 6.1**; sections 6.2-7 remain |
| 5 | ~~Reconcile the 2.2 px floor with the 0.31 px locus fit (C1)~~ | **done** -- and the premise was wrong, see C1 |
| 6 | ~~Reframe the corpus section as a cross-check (B5)~~ | **done** |
| 7 | ~~De-privilege SVP in the differential table (B2)~~ | **done** |
| 8 | Write related work, engaging Łuczyński's evaluation directly | a day |
| 9 | Re-run the radial curves accepting partial boards (C3) | a day |

Items 1–3 change what the paper claims. The rest change how well it defends it.

## G. The one-sentence verdict

The physics is sound, two of the results are genuinely worth publishing, and the draft
currently claims a system-level contribution it demonstrates only in pieces — with a
framing that implies the flat port is load-bearing for a result that does not need it.
B1 and A1 are both fixed. The headline B1 produced (+2.4% against -5.7% uncalibrated, 5%
met on every frame) is stronger than what the draft led with before, and stating A1's
limitation made section 5 more general rather than less. The top remaining item is the related-work section, which must engage Luczynski's own evaluation directly because the SVP comparison
is the paper's most contested claim.

---

# Round 2 — 2026-09-15

Re-read after the A1/B1/B2/B5/C1 fixes. The paper is much stronger and the old items stay
closed. This round found one thing that outranks anything in Round 1, because it concerns
what the headline experiment actually measures.

## H1. ~~Section 6.1 does not test absolute scale~~ — FIXED 2026-09-15

The end-to-end experiment reads: calibrate the camera in air on the board, take `|O|` from
a garage beam fit that is posed against the board, calibrate the laser from the decoy, then
measure **the board's** 549 mm span. The scale chain and the ground truth are the same
object.

Tested directly by scaling the assumed board pitch and re-running everything:

| assumed board scaled by | `\|O\|` recovered | end-to-end error |
|---|---|---|
| 0.95 | 98.98 mm | **+2.24%** |
| 1.00 | 104.19 mm | **+2.24%** |
| 1.05 | 109.40 mm | **+2.24%** |
| 1.10 | 114.61 mm | **+2.24%** |

Not approximately invariant — *exactly* invariant across a 15% swing. A board-scale error
propagates into `|O|`, hence into every range, hence into the measured span, and cancels
against a ground truth that scaled with it.

This is not a flaw in the method. It is a flaw in the claim. What section 6.1 measures is
the laser's **angular** calibration and the refraction correction — which is the right
target, since angle is what drifts between dives and scale is not — but a reader given
"measures a held-out 549 mm target to +2.2%" will take it as a metric accuracy result, and
it is not one. The fix is a sentence saying what is and is not under test, plus H2.

Note also that "held out of both calibrations" is true of the *laser* calibration only. The
board is the object the camera was calibrated on and the source of `|O|`'s scale. The
phrase oversells.

## H2. ~~The absolute-scale test is missing~~ — FIXED 2026-09-15, now section 6.2

The decoy was calipered independently of the board: 312.5 mm long, 106.1 mm deep. Running
the reference-free pipeline and reading the decoy's own metric length back gives

**311.2 mm (p90 over 13 frames) against 312.5 calipered — −0.4%.**

p90 rather than mean because yaw foreshortens one-sidedly, per the paper's own aggregation
rule. That chain — calipered decoy ← ranges ← beam ← `|O|` ← garage board PnP — *does*
respond 1:1 to a board-scale error, so unlike section 6.1 it tests absolute metric scale.
It is the paper's only such test and it is currently absent.

Adding it costs a paragraph and converts "we measured a target held out of the laser
calibration" into "and the metric scale is independently confirmed to 0.4%".

## H3. ~~Two numbers both called "the decoy's length"~~ — FIXED 2026-09-15

Section 6.3 reports the length-based Tier 3 recovering **274 mm against 312.5 (−12%)**.
H2 reports **311.2 mm (−0.4%)**. Both are correct and they are different quantities — the
first is the estimator's single-parameter by-product `|O|/slope`, the second is the
per-frame reconstruction `ℓ·Z` aggregated at p90 — but the paper does not distinguish them
and a careful reader will think one of them is wrong.

This also sharpens a result the paper already half-states: the by-product is the method's
**noisiest** output, and nobody should validate the calibration against it. Say so once,
plainly, and give both numbers with their definitions.

## H4. ~~The leave-one-out row calibrates and tests on the same object~~ — FIXED 2026-09-15

Section 6.1's leave-one-frame-out row fits the beam from nine frames of the board's
apparent size and measures the board in the tenth. `|O|` pins absolute scale so it is not
circular in the strong sense, but a range-correlated error in the board's apparent size
produces a compensating range error that partly cancels in the measurement — the same
mechanism H1 describes, one level down. It should be labelled the way the conventional row
already is, as an optimistic bound on the drift-free case rather than an independent result.

## What I checked and found sound

- The C1 reconciliation holds: Pinax residual 0.37 px at the centre, 0.46 px at 1800 px,
  flat enough that no corner blow-up exists to appeal to.
- The decoy session really does span 2.9–39.6 degrees of field, so the port-independence
  and field-coverage claims are supported by the data behind the headline.
- The scale-invariance argument of section 6.3 is exact, not approximate; I re-derived it
  and it holds for any constant factor.
- The end-to-end result is stable across all three board geometries tried (±0.5 pp).

## Priority now

| # | fix | cost |
|---|---|---|
| 1 | ~~State what section 6.1 tests and does not (H1)~~ | **done** |
| 2 | ~~Add the absolute-scale check (H2)~~ | **done** -- new section 6.2, -0.4% |
| 3 | ~~Distinguish the two decoy-length numbers (H3)~~ | **done** |
| 4 | ~~Label the leave-one-out row as optimistic (H4)~~ | **done** |
| 5 | Write related work, engaging Luczynski's evaluation directly | a day |
| 6 | Figures, then ACM LaTeX | — |

Items 1–4 were all corrections to what the paper *claims*, not to what it did, and are
done. **Related work is now the only substantive gap** before figures and formatting.


---

# Round 3 — 2026-09-19: scope reset

The paper was restructured after two scope decisions and one measurement that
invalidated its headline. Most of Rounds 1 and 2 no longer applies, because the
sections they were about have moved out.

**The laser work is P2's.** It removes the barrier P2's `HANDOFF.md` §1 calls
"calibration target must be fabricated by measurement", for the laser half, by
removing the target. Sections 5 and 6 of the old draft, and the mount-prior
negative result, are extracted to `correspondence/p2/LASER_SECTIONS_FOR_P2.md`. The code, tests and
notebooks stay here. Rounds 1–2 items A2, B3, H2, H3 and H4 went with them.

**P1 is the rig paper.** This one is about the port, and motivation that reaches
past the port — extrinsics drift, per-dive recalibration — belongs elsewhere.

**The headline was unreachable.** The old §4.2 quoted +22 % to +55 % length error
off-axis. The laser dot has to land on the fish, which pins the fish near the
optical axis; measured on 227 production frames, the dot falls inside the body
span in 100 % of them, mid-body. Over every (length, range) pair the corpus
actually photographed, the uncorrected error is **p50 1.2 %, p95 5.5 %, max
14.3 %** — it never breaches the 15 % budget on any geometry the rig can produce.
The real-data end-to-end that I had read as contradicting the paper agrees with
this exactly.

## What that leaves, and what is now open

The paper's claim is smaller and better founded: the port's cost is modest, it is
bounded by the laser geometry rather than by the optics alone, and correcting it
is free. The motivation moves from accuracy to operability, and the keystone is
**centrality** — in air the camera is a pinhole, which is what makes commodity
3D targets and ordinary tooling possible, and has no underwater equivalent.

| # | open item | note |
|---|---|---|
| 1 | §5 "What calibrating in air enables" is an outline | the keystone, and currently the weakest-written part |
| 2 | §4.2 rewrite | numbers computed, prose not written |
| 3 | ~~§6 rewrite around the camera-only end-to-end~~ | **done 2026-09-20.** Written, and it says so: all three camera models land on both objects, which is §4.2 seen from the other side rather than a contradiction. It also gained a second experiment -- the calipered decoy at unconstrained pose -- and the structural point that only that one can test absolute scale, since the board calibrates the camera, poses the beam and is then the thing measured. Two figures, both from committed code. |
| 4 | §9 conclusion | follows 1–3 |
| 5 | Related work | unchanged from Round 1: the longest pole, and §4.3 is the contested claim |
| 6 | LEGO pilot | in progress; would turn §5 from enabled-but-untested into demonstrated, and would settle §8's `fx/fy` question if shot with 90° rolls |
| 7 | Figures, then ACM LaTeX | unchanged |

## Carried over and still true

- **B2** — §4.2's table scores against the in-water calibration, which reads zero by
  construction. Still holds for the free-field numbers that remain.
- **C2** — the water/air focal ratio sits 1.5 % below `n_w` and the paper says the
  offset "ought eventually to be predicted rather than absorbed". Still unpredicted.
- **C3** — the radial curves reject partial boards and are conservative by an
  unquantified amount.
- **The `fx/fy` anisotropy** is unresolved and a rule cannot resolve it. Item 6 can.
