"""Calibrate from a shoot of the brick target, and say which views to reshoot.

    uv run python refraction_analysis/calibrate_brick_target.py \\
        --frames "/home/chris/data/Corner Target" \\
        --model calibration_model/calibration_model.io \\
        [--checkerboard "/home/chris/data/Corner Checkerboard"]

Without --frames it only checks the model -- worth running on a Studio export
before a shoot. Raw ORF only; the TG-6's JPEGs carry an in-camera lens correction.
Before any image is touched the model is checked against the design rules, and
against --design if one is given, so a swapped brick shows up here rather than
as a bad calibration.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fishsense_wuwnet.brick_calibration import calibrate_folder  # noqa: E402
from fishsense_wuwnet.calibration_model import MODEL_IO  # noqa: E402
from fishsense_wuwnet.target_design import compare  # noqa: E402
from fishsense_wuwnet.target_rules import report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, help="folder of raw .ORF frames of the target")
    parser.add_argument("--model", type=Path, default=MODEL_IO, help="the target as built (Studio .io or .ldr)")
    parser.add_argument("--design", type=Path, help="a design to check the model against, brick for brick")
    parser.add_argument("--checkerboard", type=Path, help="same-session checkerboard frames, for distortion")
    parser.add_argument("--cache", type=Path, help="where to cache correspondences")
    args = parser.parse_args()

    if args.design is not None:
        diff = compare(args.model, args.design)
        print(f"model vs design: {diff['matched']} bricks match")
        for kind in ("recoloured", "missing", "extra"):
            for item in diff[kind]:
                print(f"  {kind}: {item}")
    print(report(args.model))
    if args.frames is None:
        return
    print()

    result, board_K, source, views, flags = calibrate_folder(args.frames, args.model,
                                                             args.checkerboard, args.cache)
    K = result.K
    ratio, board_ratio = K[0, 0] / K[1, 1], board_K[0, 0] / board_K[1, 1]
    print(f"\n{len(result.views)} views used, rms {result.rms:.2f} px")
    print(f"  fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}  fx/fy {ratio:.5f}")
    print(f"  against the checkerboard ({source}): fx {100 * (K[0, 0] / board_K[0, 0] - 1):+.2f} %, "
          f"fx/fy {100 * (ratio / board_ratio - 1):+.2f} %")
    print("\nview        faces walls  dist cm  resid px  notes")
    for name in sorted(views):
        v = views[name]
        dist = result.distance.get(name)
        res = result.residual.get(name)
        print(f"{name:11s} {len(v['obj']) // 4:5d} {v['walls']:5d}  "
              f"{f'{dist / 10:7.1f}' if dist else '      -'}  {f'{res:8.2f}' if res else '       -'}  "
              f"{'; '.join(flags[name])}")


if __name__ == "__main__":
    main()
