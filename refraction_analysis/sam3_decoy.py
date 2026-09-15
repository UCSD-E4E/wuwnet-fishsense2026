"""SAM3 masks of the pool decoy, one per on-decoy frame, and the body size at the dot.

Produces data/decoy_sam3_masks.npz and data/decoy_sam3.csv, which
laser_per_dive.ipynb consumes; the notebook does not need SAM3 itself.

Mirrors fishsense-lite's head/tail stage (predict_headtail_image.py): an
1800x1350 crop centred on the laser dot, text prompt "fish", the mask that
contains the dot. Checkpoint is SAM 3.1's `sam3.1_multiplex.pt` (not `sam3.pt`;
the loader takes either silently). Loading it prints four missing keys under
`backbone.vision_backbone.convs.3`; the masks were produced with that warning.

SAM3 is not a dependency of this project. It ran in a separate uv project:

    [project]  requires-python = ">=3.13,<3.14"
    dependencies = ["sam3 @ git+https://github.com/facebookresearch/sam3.git@8e451d5eb43c817b64ae7577fb7b9ae223db88a9",
                    "torch", "torchvision", "numpy>=2.2.6", "opencv-python-headless", "pillow",
                    "setuptools<80", "einops", "psutil", "pycocotools", "pandas", "scikit-image",
                    "scikit-learn", "matplotlib", "scipy", "open_clip_torch"]
    [tool.uv]  package = false;  override-dependencies = ["numpy"]

(setuptools<80 for pkg_resources; the rest are imports SAM3 does not declare; the
numpy override is the one fishsense-lite uses.) On NixOS torch needs
LD_LIBRARY_PATH=/run/opengl-driver/lib to see the GPU. Then:

    LD_LIBRARY_PATH=/run/opengl-driver/lib uv run --project <env> python sam3_decoy.py --checkpoint sam3.1_multiplex.pt
    uv run --project <env> python sam3_decoy.py --stub     # geometry only, no weights

Run on an RTX 3060 (6 GB) in bf16; 13 frames in under a minute after load.
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw

POOL = Path.home() / "mnt/fishsense_data/__unsorted-data/2024.05.01.FishSense.Chris Pool"
DATA = Path(__file__).resolve().parent.parent / "data"
DOTS = DATA / "laser_dots_decoy_raw.npz"
JPEG_OFF = np.array([8.0, 8.0]); CROP_W, CROP_H = 1800, 1350
ON = [92, 93, 94, 96, 97, 102, 103, 104, 105, 106, 107, 108, 109]

def crop_around(img, x, y):
    x0 = int(np.clip(x - CROP_W / 2, 0, img.shape[1] - CROP_W)); y0 = int(np.clip(y - CROP_H / 2, 0, img.shape[0] - CROP_H))
    return img[y0:y0 + CROP_H, x0:x0 + CROP_W], x0, y0

def walk(mask, x, y, dx, dy, maxlen=3000):
    for k in range(1, maxlen):
        xi, yi = int(round(x + k * dx)), int(round(y + k * dy))
        if not (0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0]) or not mask[yi, xi]: return k
    return None

def measure(mask, cx, cy):
    """Height at the dot (shortest chord near-perpendicular to the body axis), length along the axis."""
    ys, xs = np.nonzero(mask); pts = np.c_[xs, ys].astype(float); c = pts.mean(0)
    w, v = np.linalg.eigh(np.cov((pts - c).T)); axis = v[:, 1]; perp = np.array([-axis[1], axis[0]])
    s = (pts - c) @ axis; L = np.percentile(s, 99.5) - np.percentile(s, 0.5)
    ends = (c + np.percentile(s, 0.5) * axis, c + np.percentile(s, 99.5) * axis)
    best = None
    for th in np.radians(np.arange(-25, 26, 5)):
        d = np.cos(th) * perp + np.sin(th) * axis
        a, b = walk(mask, cx, cy, d[0], d[1]), walk(mask, cx, cy, -d[0], -d[1])
        if a and b and (best is None or a + b < best[0]): best = (a + b, d, a, b)
    if best is None: return None
    h, d, a, b = best
    return dict(height=h, top=np.array([cx, cy]) + a * d, bot=np.array([cx, cy]) - b * d, length=L, ends=ends, area=int(mask.sum()))

class Stub:
    def segment(self, rgb, cx, cy):
        m = np.zeros(rgb.shape[:2], bool); cv2.ellipse(m.view(np.uint8), (int(cx) + 40, int(cy) - 20), (300, 110), -25, 0, 360, 1, -1); return [m], [1.0]

class Sam3:
    def __init__(self, ckpt):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch; self.dev = dev
        model = build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=False, device=dev); model.eval()
        self.proc = Sam3Processor(model, device=dev, confidence_threshold=0.3)
    def segment(self, rgb, cx, cy):
        with self.torch.inference_mode(), self.torch.autocast(self.dev, dtype=self.torch.bfloat16, enabled=self.dev == "cuda"):
            st = self.proc.set_image(Image.fromarray(rgb)); st = self.proc.set_text_prompt("fish", st)
            masks = [m.detach().cpu().numpy().squeeze().astype(bool) for m in st["masks"]]; scores = [float(s) for s in st["scores"]]
            if not any(m[int(cy), int(cx)] for m in masks):        # nothing on the dot: box prompt around it as a fallback
                st = self.proc.add_geometric_prompt([cx / rgb.shape[1], cy / rgb.shape[0], 0.5, 0.5], True, st)
                masks += [m.detach().cpu().numpy().squeeze().astype(bool) for m in st["masks"]]; scores += [float(s) for s in st["scores"]]
        return masks, scores

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint"); ap.add_argument("--stub", action="store_true"); a = ap.parse_args()
    seg = Stub() if a.stub else Sam3(a.checkpoint)
    dots = {k: v for k, v in np.load(DOTS).items()}
    rows, masks_out, tiles = [], {}, []
    for fr in ON:
        n = f"P5010{fr:03d}.JPG"; bgr = cv2.imread(str(POOL / n)); x, y = dots[n][:2] - JPEG_OFF
        crop, x0, y0 = crop_around(bgr, x, y); cx, cy = x - x0, y - y0
        masks, scores = seg.segment(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), cx, cy)
        on_dot = [(s, m) for s, m in zip(scores, masks) if m[int(cy), int(cx)]]
        if not on_dot: print(f"{n[4:8]}: {len(masks)} masks, none on the dot"); continue
        score, mask = max(on_dot, key=lambda t: t[0])
        nl, lab = cv2.connectedComponents(mask.astype(np.uint8)); mask = lab == lab[int(cy), int(cx)]
        m = measure(mask, cx, cy)
        if m is None: print(f"{n[4:8]}: chord failed"); continue
        lift = lambda p: np.asarray(p) + [x0, y0] + JPEG_OFF
        rows.append([n, score, m["height"], m["length"], m["area"], *lift(m["top"]), *lift(m["bot"]), *lift(m["ends"][0]), *lift(m["ends"][1])])
        masks_out[n] = np.packbits(mask); masks_out[n + "_origin"] = np.array([x0, y0])
        print(f"{n[4:8]}: score {score:.2f}  area {m['area']:7d}  height {m['height']:5.0f} px  length {m['length']:5.0f} px  L/h {m['length']/m['height']:.2f}")
        ov = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).copy(); ov[mask] = (0.55 * ov[mask] + [0, 110, 0]).astype(np.uint8)
        k = 0.3; im = Image.fromarray(ov).resize((int(CROP_W * k), int(CROP_H * k))); dr = ImageDraw.Draw(im)
        dr.line([tuple(m["top"] * k), tuple(m["bot"] * k)], fill="red", width=3); dr.line([tuple(m["ends"][0] * k), tuple(m["ends"][1] * k)], fill="cyan", width=2)
        dr.ellipse([(cx * k - 4, cy * k - 4), (cx * k + 4, cy * k + 4)], outline="yellow", width=2); dr.text((5, 5), f"{n[4:8]} {score:.2f}", fill="yellow"); tiles.append(im)
    out = DATA; tag = "stub" if a.stub else "sam3"
    with open(out / f"decoy_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "score", "height_px", "length_px", "area", "top_x", "top_y", "bot_x", "bot_y", "end0_x", "end0_y", "end1_x", "end1_y"]); w.writerows(rows)
    np.savez(out / f"decoy_{tag}_masks.npz", **masks_out)
    if tiles:
        W_, H_ = tiles[0].size; sheet = Image.new("RGB", (W_ * 5, H_ * 3))
        for i, t in enumerate(tiles): sheet.paste(t, ((i % 5) * W_, (i // 5) * H_))
        sheet.save(out / f"decoy_{tag}.jpg", quality=80)
    print(f"{len(rows)} frames -> decoy_{tag}.csv")

if __name__ == "__main__":
    main()
