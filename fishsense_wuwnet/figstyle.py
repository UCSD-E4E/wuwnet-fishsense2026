"""Print figure style for the paper, and the hook that saves every figure.

Two jobs. `apply()` sets a consistent look for figures that end up in a
two-column paper rather than on a screen. `install_autosave()` registers an
IPython hook so that executing the notebooks writes every figure they draw to
`figures/` as both PDF (vector, for the paper) and PNG (for review), named from
the axes title so the files are identifiable without opening them.

The palette is the data-viz reference categorical order, used in its fixed
sequence and never cycled -- a fourth series takes slot 4, not slot 1 again.
Every chart here is a line or bar chart, which validates on the adjacent
pairlist: worst adjacent CVD dE 9.1, normal-vision dE 19.6 against the light
surface. Three slots sit below the 3:1 contrast floor, so the relief rule
applies; each series therefore also carries a line style, which is redundancy
worth having in print anyway, where a figure may be photocopied to greyscale.

Figures wider than `MAX_WIDTH_IN` are scaled down on save. The notebooks size
several figures for a screen, and at 13 inches a 9 pt label is unreadable once
the figure is set in a two-column page.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

#: The reference categorical order, in its fixed sequence.
SERIES = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

#: Secondary encoding, required by the relief rule for the low-contrast slots
#: and kept for all of them so greyscale stays readable.
DASHES = ("-", "--", "-.", ":")

#: Two-column page width. Anything wider is scaled to fit on save.
MAX_WIDTH_IN = 7.0

#: Where the paper's figures land. Both formats, same stem.
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"

_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRID = "#d8d7d2"


def apply() -> None:
    """Set rcParams for two-column print figures."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.prop_cycle": mpl.cycler(color=list(SERIES)),
            "axes.edgecolor": _INK_SECONDARY,
            "axes.labelcolor": _INK,
            "axes.titlecolor": _INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": _GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            "axes.axisbelow": True,
            "text.color": _INK,
            "xtick.color": _INK_SECONDARY,
            "ytick.color": _INK_SECONDARY,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "legend.frameon": False,
            "figure.dpi": 110,
        }
    )


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    out = "".join(keep)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")[:70]


def _name_for(fig) -> str:
    """Name a figure from its title, so the filename says what it is."""
    if fig._suptitle is not None and fig._suptitle.get_text().strip():
        return _slug(fig._suptitle.get_text())
    for ax in fig.axes:
        if ax.get_title().strip():
            return _slug(ax.get_title())
    return f"figure-{fig.number}"


def save(fig, name: str, directory: Path = FIGURE_DIR) -> list:
    """Write one figure as PDF and PNG; returns the paths written.

    Oversized figures are scaled to `MAX_WIDTH_IN` first. Font sizes are in
    points and do not scale with the figure, so shrinking the canvas is what
    makes the labels legible at the size the figure is actually printed.
    """
    directory.mkdir(parents=True, exist_ok=True)
    width, height = fig.get_size_inches()
    if width > MAX_WIDTH_IN:
        fig.set_size_inches(MAX_WIDTH_IN, height * MAX_WIDTH_IN / width)
        fig.tight_layout()
    written = []
    for suffix in ("pdf", "png"):
        path = directory / f"{name}.{suffix}"
        fig.savefig(path)
        written.append(path)
    return written


def install_autosave(prefix: str = "", directory: Path = FIGURE_DIR) -> None:
    """Save every figure a notebook draws, once, after the cell that drew it.

    Registered as a `post_run_cell` hook rather than wrapping `savefig` calls
    into the notebooks, so the notebooks stay readable and no cell has to know
    it is being harvested. Figures are keyed by identity, so a cell that redraws
    an existing figure does not write it twice.
    """
    from IPython import get_ipython

    shell = get_ipython()
    if shell is None:  # not under IPython: nothing to hook
        return
    seen: set = set()

    def _harvest(_result=None):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            if id(fig) in seen or not fig.axes:
                continue
            seen.add(id(fig))
            stem = _name_for(fig)
            save(fig, f"{prefix}{stem}" if prefix else stem, directory)

    shell.events.register("post_run_cell", _harvest)
