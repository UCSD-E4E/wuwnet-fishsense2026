"""Execute every analysis notebook and write its figures to figures/.

    uv run python refraction_analysis/make_figures.py

Each notebook is run in a fresh kernel with a preamble injected that applies the
paper's print style and registers an autosave hook, so every figure a notebook
draws is written as both PDF (for the paper) and PNG (for review), named from
its own title. The notebooks on disk are not modified -- the preamble exists
only in the executed copy -- so a figure can never drift from the analysis that
produced it, and re-running this script is the only way figures are made.
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT.parent / "figures"

#: Figures belonging to a sibling paper are written into that paper's handover
#: folder rather than this one's `figures/`, so ownership is a property of where
#: the generator puts the file and not of a note someone has to remember. The
#: laser's drift and its per-dive recalibration are P1's, the rig paper: mount
#: movement between sessions is hardware behaviour, and REVIEW.md takes
#: extrinsics drift and per-dive recalibration out of this paper's scope.
P1_FIGURE_DIR = ROOT.parent / "correspondence" / "p1" / "figures"

#: Notebook -> (filename prefix, destination). The prefix says which analysis a
#: figure came from; the destination says which paper it belongs to.
NOTEBOOKS = {
    "flat_port_refraction.ipynb": ("sim-", FIGURE_DIR),
    "pool_validation.ipynb": ("pool-", FIGURE_DIR),
    "laser_calibration.ipynb": ("drift-", P1_FIGURE_DIR),
    "laser_per_dive.ipynb": ("perdive-", P1_FIGURE_DIR),
}

#: The notebooks apply the style themselves, so all this adds is the headless
#: backend and the harvest hook.
PREAMBLE = """
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from fishsense_wuwnet import figstyle
figstyle.install_autosave(prefix={prefix!r}, directory=Path({directory!r}))
"""


def main() -> int:
    destinations = list(dict.fromkeys(d for _, d in NOTEBOOKS.values()))
    for directory in destinations:
        directory.mkdir(parents=True, exist_ok=True)
    before = {(d, p.name) for d in destinations for p in d.glob("*")}
    failed = []

    for name, (prefix, directory) in NOTEBOOKS.items():
        path = ROOT / name
        if not path.exists():
            print(f"  {name}: missing, skipped")
            continue
        nb = nbformat.read(path, as_version=4)
        nb.cells.insert(
            0,
            nbformat.v4.new_code_cell(
                PREAMBLE.format(prefix=prefix, directory=str(directory))
            ),
        )
        print(f"  {name}: executing ...", flush=True)
        try:
            NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
        except Exception as exc:  # keep going; report at the end
            failed.append((name, f"{type(exc).__name__}: {exc}"[:200]))
            print(f"    FAILED: {type(exc).__name__}")

    for directory in destinations:
        after = sorted(p.name for p in directory.glob("*"))
        made = [n for n in after if (directory, n) not in before]
        rel = directory.relative_to(ROOT.parent)
        print(f"\n{len(after)} files in {rel} ({len(made)} new)")
        stems = sorted({n.rsplit(".", 1)[0] for n in after})
        for stem in stems:
            have = [s for s in ("pdf", "png") if (directory / f"{stem}.{s}").exists()]
            print(f"  {stem:58s} {'+'.join(have)}")
    if failed:
        print("\nfailed notebooks:")
        for name, err in failed:
            print(f"  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
