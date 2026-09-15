# fishsense-wuwnet2026

Flat-port refraction analysis for FishSense (WUWNet 2026).

Quantifies what an uncorrected flat-pane housing costs FishSense laser
reconstruction and fish-length measurement, and how much of that the Pinax model
recovers.

> Łuczyński, Pfingsthorn & Birk, *"The Pinax-model for accurate and efficient
> refraction correction of underwater cameras in flat-pane housings,"* Ocean
> Engineering 133 (2017) 9–22.

```bash
uv sync
uv run pytest          # validates the refraction model against the paper
uv run jupyter lab     # refraction_analysis/
```

The refraction model in `fishsense_wuwnet/refraction.py` is self-contained (no
dependency on `fishsense-pinax`), so this repo runs on the same Python 3.13 /
OpenCV 4 stack as `fishsense-core`.
