# wuwnet-fishsense2026

Calibrating an underwater laser–camera rig without an in-water reference
(WUWNet 2026). The paper draft is [`PAPER.md`](PAPER.md); its standing
self-review, including what is still open, is [`REVIEW.md`](REVIEW.md).

A rig that measures fish length from one camera and one laser dot has to
calibrate two things, and conventionally both are calibrated against a reference
object put in the water — a checkerboard for the camera, a dive slate for the
laser, the latter every dive because the laser's extrinsics move. This repo is
the analysis behind removing both.

- **The camera** is calibrated in air. A flat port is an axial imaging system
  whose refraction depends only on the water index, so an in-air calibration plus
  a per-pixel correction replaces the in-water session.
- **The laser** is calibrated from the dive's own subjects. The dot locus is a
  straight line whose two free parameters come from the dots alone; the remaining
  scale gauge is broken by the *apparent size* of any rigid object the dot lands
  on, whose true size need not be known. This half is not specific to a flat
  port — it is pinhole geometry on corrected directions.

End to end, with nothing of known size in the water, the rig measures a held-out
549 mm target to +0.0 % median and 2.6 % worst case over 1.1–4.5 m, against
−5.4 % and 15.8 % for the same rig trusting its bench extrinsics. Absolute scale,
checked separately against an independently calipered object, comes back within
0.4 %.

Two results are corrections to expectations rather than confirmations, and are
the parts most worth reading: on geometry alone a well-fitted **in-water
single-viewpoint calibration is indistinguishable from the Pinax model** (§4.3),
and the system turns out to be limited by **how well the apparent size of the
object is measured** — roughly forty times more than by dot detection.

```bash
uv sync
uv run pytest                                      # 62 tests
uv run jupyter lab                                 # refraction_analysis/
uv run python refraction_analysis/make_figures.py  # figures/ as PDF and PNG
```

Detected board corners and laser dots are cached under `data/`, so every notebook
runs without access to the image archive. `figures/` is generated, never edited.

## Layout

| path | what it is |
|---|---|
| `fishsense_wuwnet/refraction.py` | flat-port and dome-port geometry, Pinax, the exact axial model |
| `fishsense_wuwnet/pipelines.py` | the back-projections, and the two length errors that differ by ~50× |
| `fishsense_wuwnet/laser.py` | the dot locus, and three ways to close the beam |
| `refraction_analysis/` | simulation, pool validation, drift, per-dive calibration |
| `correspondence/` | notes exchanged with the sibling papers |

The refraction model is self-contained — no dependency on `fishsense-pinax` — so
this repo runs on the same Python 3.13 / OpenCV 4 stack as `fishsense-core`.

> Łuczyński, Pfingsthorn & Birk, *"The Pinax-model for accurate and efficient
> refraction correction of underwater cameras in flat-pane housings,"* Ocean
> Engineering 133 (2017) 9–22. `uv run pytest` pins our implementation against
> that paper's own Table 1.

## Sibling repos

| repo | paper | relationship |
|---|---|---|
| `imwut_2026_fishsense_lite` | the system paper | quantifies the failure this repo's corrections address; shares the forward model verbatim |
| `cscw-fishsense2027` | deployability for citizen scientists | consumes the target-free laser calibration; see its `HANDOFF.md` §7 |
