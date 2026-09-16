# Question for the data-worker: does SAM3 load clean in production?

Short version: **when the GPU worker loads `sam3.1_multiplex.pt`, does the log show
missing keys?** One line from a pod log settles it.

## Context

P4 (`wuwnet-fishsense2026`) needed fish masks for a laser-calibration experiment and
reused the head/tail stage's backend rather than inventing one — same pinned SAM3 commit
(`8e451d5eb43c817b64ae7577fb7b9ae223db88a9`), same `build_sam3_image_model` →
`Sam3Processor` path, same concept prompt `"fish"`, same 1800x1350 crop centred on the
laser dot, run against `sam3.1_multiplex.pt`. Masks came out clean on all 13 frames
(scores 0.62-0.91) and the experiment worked.

But the load printed this, every time:

```
loaded /path/sam3.1_multiplex.pt and found missing and/or unexpected keys:
missing_keys=['backbone.vision_backbone.convs.3.conv_1x1.weight',
              'backbone.vision_backbone.convs.3.conv_1x1.bias',
              'backbone.vision_backbone.convs.3.conv_3x3.weight',
              'backbone.vision_backbone.convs.3.conv_3x3.bias']
```

Four keys, one neck convolution block (`convs.3`), left at initialisation.

## Why it is worth a minute of your time

`headtail_predictor.py` already documents exactly this hazard, in its own words:

> upstream ships two SAM3 checkpoints that are not interchangeable, and
> `build_sam3_image_model`'s `_load_checkpoint` loads either with `strict=False` —
> missing keys are *printed*, not raised — so the wrong one degrades every mask without
> failing anything.

That reasoning is about checkpoint *choice*. This is the same failure surface one level
down: a checkpoint that is nominally correct, loading into a model whose neck has one
more block than the weights provide. `strict=False` means nothing raises either way.

Two possibilities, and they have different consequences:

1. **Production prints the same four keys.** Then this is a property of the pinned
   commit against this checkpoint, the §0.2b benchmark (35.0% usable vs 23.8% for Mask
   R-CNN) was measured with it, and nothing is wrong — but it is worth a sentence in
   `headtail_predictor.py` so the next person who sees it does not spend an afternoon on
   it, as I nearly did.
2. **Production loads clean.** Then P4's run and production are not the same model, my
   masks came from a partially-initialised backbone, and — more importantly — the
   difference is something in the environment rather than the weights, which would be
   worth understanding before it moves the other way and quietly degrades production.

## What I am asking for

From a GPU-worker pod that has loaded the checkpoint:

```
kubectl logs <data-worker-gpu-pod> | grep -i -A5 "missing_keys\|loaded .*\.pt"
```

If the loader line is not in the retained logs, a cold start reproduces it — the
`_load_checkpoint` print happens on every load, and `get_segmenter` logs
`loading SAM3 checkpoint=%s` immediately before it.

Also useful if it is cheap: the exact checkpoint bytes in use. P4 used a 3,502,755,717-byte
`sam3.1_multiplex.pt`; if `model-weights/sam3/3.1/sam3.1_multiplex.pt` in Garage differs in
size or digest, that alone explains it.

## What P4 used, for comparison

- `sam3` at the pinned commit above, installed from git, in a standalone uv project.
- `torch` 2.14.0+cu130, CUDA 13.2 driver, RTX 3060 Laptop 6 GB, bf16 autocast.
- `build_sam3_image_model(checkpoint_path=..., load_from_HF=False, device="cuda")` —
  `load_from_HF=False` deliberately, so the loader cannot silently fall back to
  `facebook/sam3`'s `sam3.pt`.
- `Sam3Processor(model, confidence_threshold=0.3)`.
- Undeclared imports SAM3 needs and does not list, in case the worker image ever
  rebuilds: `setuptools<80` (for `pkg_resources`), `einops`, `psutil`, `pycocotools`,
  `pandas`, `scikit-image`, `scikit-learn`, `matplotlib`, `scipy`, `open_clip_torch`.

No action needed on P4's side either way — the masks are cached and the calibration
result does not depend on the last neck block. This is a production-hygiene question,
not a blocker.
