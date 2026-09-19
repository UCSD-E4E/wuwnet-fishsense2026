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

#: Prefixed onto every figure name so a file says which analysis it came from.
NOTEBOOKS = {
    "flat_port_refraction.ipynb": "sim-",
    "pool_validation.ipynb": "pool-",
    "laser_calibration.ipynb": "drift-",
    "laser_per_dive.ipynb": "perdive-",
}

PREAMBLE = """
import matplotlib
matplotlib.use("Agg")
from fishsense_wuwnet import figstyle
figstyle.apply()
figstyle.install_autosave(prefix={prefix!r})
"""


def main() -> int:
    FIGURE_DIR.mkdir(exist_ok=True)
    before = {p.name for p in FIGURE_DIR.glob("*")}
    failed = []

    for name, prefix in NOTEBOOKS.items():
        path = ROOT / name
        if not path.exists():
            print(f"  {name}: missing, skipped")
            continue
        nb = nbformat.read(path, as_version=4)
        nb.cells.insert(0, nbformat.v4.new_code_cell(PREAMBLE.format(prefix=prefix)))
        print(f"  {name}: executing ...", flush=True)
        try:
            NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
        except Exception as exc:  # keep going; report at the end
            failed.append((name, f"{type(exc).__name__}: {exc}"[:200]))
            print(f"    FAILED: {type(exc).__name__}")

    after = sorted(p.name for p in FIGURE_DIR.glob("*"))
    made = [n for n in after if n not in before]
    print(f"\n{len(after)} files in {FIGURE_DIR} ({len(made)} new)")
    stems = sorted({n.rsplit(".", 1)[0] for n in after})
    for stem in stems:
        have = [s for s in ("pdf", "png") if (FIGURE_DIR / f"{stem}.{s}").exists()]
        print(f"  {stem:58s} {'+'.join(have)}")
    if failed:
        print("\nfailed notebooks:")
        for name, err in failed:
            print(f"  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
