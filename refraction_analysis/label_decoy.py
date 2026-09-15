"""Hand-label the decoy's body height at the laser dot, one frame at a time.

Run:  uv run python refraction_analysis/label_decoy.py

For each on-decoy frame a crop around the dot opens. Click the DORSAL edge of the
body (top of the dark back) and then the VENTRAL edge (belly), both on the line
through the dot perpendicular to the body axis. Right-click to redo the frame,
close the window to skip it. Points are saved in raw sensor coordinates to
data/decoy_heights.csv as they go, so the script can be stopped and resumed.
"""

import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from fishsense_wuwnet.dataset import JPEG_CROP_OFFSET, POOL_DIR, load_corner_cache

ON_DECOY = (92, 93, 94, 96, 97, 102, 103, 104, 105, 106, 107, 108, 109)
OUT = Path(__file__).resolve().parent.parent / "data" / "decoy_heights.csv"
HALF = 900  # crop half-width in pixels around the dot

dots = load_corner_cache(Path(__file__).resolve().parent.parent / "data" / "laser_dots_decoy_raw.npz")
done = {}
if OUT.exists():
    with open(OUT) as f:
        for row in csv.DictReader(f):
            done[row["frame"]] = row

with open(OUT, "a", newline="") as f:
    writer = csv.writer(f)
    if not done:
        writer.writerow(["frame", "dorsal_x", "dorsal_y", "ventral_x", "ventral_y"])
    for fr in ON_DECOY:
        name = f"P5010{fr:03d}.JPG"
        if name in done:
            continue
        img = cv2.cvtColor(cv2.imread(str(POOL_DIR / name)), cv2.COLOR_BGR2RGB)
        x, y = dots[name][:2] - np.array(JPEG_CROP_OFFSET)
        x0, y0 = int(max(0, x - HALF)), int(max(0, y - 0.7 * HALF))
        crop = img[y0 : y0 + int(1.4 * HALF), x0 : x0 + 2 * HALF]

        clicks = []
        fig, ax = plt.subplots(figsize=(13, 9))
        ax.imshow(crop)
        ax.plot(x - x0, y - y0, "o", mfc="none", mec="yellow", ms=14, mew=2)
        ax.set_title(f"{name}: click DORSAL edge, then VENTRAL edge, through the dot  (right-click = redo)")
        ax.set_axis_off()

        def on_click(event, clicks=clicks, ax=ax):
            if event.inaxes is not ax:
                return
            if event.button == 3:
                clicks.clear()
                [l.remove() for l in ax.lines[1:]]
                fig.canvas.draw()
                return
            clicks.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, "r+", ms=16, mew=2)
            if len(clicks) == 2:
                ax.plot([clicks[0][0], clicks[1][0]], [clicks[0][1], clicks[1][1]], "r-", lw=1.5)
            fig.canvas.draw()
            if len(clicks) == 2:
                plt.pause(0.4)
                plt.close(fig)

        fig.canvas.mpl_connect("button_press_event", on_click)
        plt.show()
        if len(clicks) == 2:
            (dx, dy), (vx, vy) = clicks
            writer.writerow([name, dx + x0 + JPEG_CROP_OFFSET[0], dy + y0 + JPEG_CROP_OFFSET[1],
                             vx + x0 + JPEG_CROP_OFFSET[0], vy + y0 + JPEG_CROP_OFFSET[1]])
            f.flush()
            print(f"{name}: height {np.hypot(dx - vx, dy - vy):.0f} px")
        else:
            print(f"{name}: skipped")
print(f"labels in {OUT}")
