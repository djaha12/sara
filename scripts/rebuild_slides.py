#!/usr/bin/env python3
"""Strip the baked-in logos off the five slides and stamp the real one back on.

Every slide currently carries a *different* mangled logo - the ray counts, wave
shapes and wordmark fonts all disagree, and slide 5's has collapsed into a green
blob.  So none of them can be cleaned up in place; each one gets removed and the
traced vector logo (see build_logo.py) is drawn fresh at the same coordinates on
every slide.

Removal is done with harmonic inpainting: solve Laplace's equation inside the
mask with the surrounding pixels as boundary conditions.  Because the solution
matches the boundary exactly, there is no seam, and because the sky behind every
logo is a smooth gradient, there is nothing to smear.
"""

import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from PIL import Image

import build_logo_v2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "source")
OUT = os.path.join(ROOT, "output")

# Every coordinate below is expressed against this canvas and rescaled to
# whatever the inputs actually are, so dropping in higher-resolution originals
# needs no edits here.
CANVAS = 1254

# One shared logo placement for every slide: top-right, small, sitting just
# above the headline.  The right margin mirrors the headline's left margin
# (104 px) and the box bottom clears the highest headline - slide 2's, which
# starts at y=196 - by 18 px.
LOGO_W = 160
LOGO_Y = 69
LOGO_MARGIN_RIGHT = 104

# Slide 3 is the exception: its top-right corner is the Paris building, not
# sky, so the shared spot lands the logo on balconies and foliage.  Set this to
# a (x, y) pair to park slide 3 over its own sky instead.
SLIDE3_OVERRIDE = None

# White card behind the logo.  The supplied artwork's royal blue sits at 1.38:1
# against these skies - effectively invisible - and slide 3's top-right corner
# is a building rather than sky.  The plate fixes both without touching a single
# logo colour, which is why it is the default.  Set to None to drop it.
PLATE = dict(pad=15, radius=20, opacity=0.94)

# Region to hunt for the old logo in.  Generous - the detector finds the actual
# ink, this only keeps it from mistaking headline text or clouds for a logo.
ROIS = {
    "slide1_karlovy": (995, 25, 1250, 215),
    "slide2_paris": (995, 25, 1250, 215),
    "slide3_why": (55, 20, 285, 195),
    "slide4_office": (995, 25, 1250, 210),
    "slide5_office_broken": (995, 25, 1250, 210),
}

# Slide 5 is slide 4 re-rendered with a corrupted logo: outside the damaged
# corner the two are identical to within JPEG noise, so the corner is
# transplanted from slide 4 rather than invented.
S5_PATCH = (855, 90, 1254, 425)


# --------------------------------------------------------------------------

def detect_logo(img, roi, scale=1.0):
    """Mask the logo ink inside `roi`, judged against the local sky colour."""
    x0, y0, x1, y1 = roi
    ring_w = max(3, int(round(6 * scale)))
    sub = img[y0:y1, x0:x1].astype(np.float32)
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    lum = sub.mean(2)

    # Sky reference taken from the ROI's own border ring, which the logo never
    # reaches, so it tracks each slide's own gradient instead of a fixed guess.
    ring = np.concatenate([lum[:ring_w].ravel(), lum[-ring_w:].ravel(),
                           lum[:, :ring_w].ravel(), lum[:, -ring_w:].ravel()])
    sky = float(np.median(ring))

    gold = (r - b > 25) & (r > 130)          # sun
    bright = lum > sky + 38                  # white waves / wordmark
    dark = lum < sky - 35                    # navy waves
    mask = gold | bright | dark

    mask = grow(mask, max(1, int(round(3 * scale))))   # close speckles
    mask = fill_holes(mask, max(3, int(round(9 * scale)) | 1))
    mask = grow(mask, max(3, int(round(8 * scale))))   # cover the AA halo

    full = np.zeros(img.shape[:2], bool)
    full[y0:y1, x0:x1] = mask
    return full


def grow(mask, radius):
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def fill_holes(mask, size=9):
    import cv2
    m = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k).astype(bool)


# --------------------------------------------------------------------------

def harmonic_inpaint(img, mask, pad=10):
    """Solve nabla^2 u = 0 over `mask`, pinned to the surrounding pixels."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return img
    y0 = max(0, ys.min() - pad)
    y1 = min(img.shape[0], ys.max() + 1 + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(img.shape[1], xs.max() + 1 + pad)

    sub = img[y0:y1, x0:x1].astype(np.float64)
    m = mask[y0:y1, x0:x1]
    h, w = m.shape

    idx = -np.ones((h, w), np.int64)
    idx[m] = np.arange(int(m.sum()))
    n = int(m.sum())

    rows, cols, vals = [], [], []
    rhs = np.zeros((n, sub.shape[2]))
    ry, rx = np.nonzero(m)
    for p, (r, c) in enumerate(zip(ry, rx)):
        deg = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue          # Neumann at the sub-image edge
            deg += 1
            if m[nr, nc]:
                rows.append(p); cols.append(idx[nr, nc]); vals.append(-1.0)
            else:
                rhs[p] += sub[nr, nc]
        rows.append(p); cols.append(p); vals.append(float(deg))

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    sol = np.stack([spl.spsolve(A, rhs[:, ch]) for ch in range(sub.shape[2])], 1)

    out = img.copy()
    patch = sub.copy()
    patch[m] = sol
    out[y0:y1, x0:x1] = np.clip(patch, 0, 255)
    return out


def add_matched_grain(img, orig, mask, seed=7):
    """Put back the sky's own fine noise so the filled area isn't glassy."""
    import cv2
    ys, xs = np.nonzero(mask)
    y0, y1 = max(0, ys.min() - 40), min(img.shape[0], ys.max() + 41)
    x0, x1 = max(0, xs.min() - 40), min(img.shape[1], xs.max() + 41)

    ring = orig[y0:y1, x0:x1].astype(np.float32)
    ring_mask = ~mask[y0:y1, x0:x1]
    hp = ring - cv2.GaussianBlur(ring, (0, 0), 2.0)
    sigma = float(hp[ring_mask].std())

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, img.shape[:2] + (1,))
    noise = cv2.GaussianBlur(noise[..., 0], (0, 0), 0.7)[..., None]

    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3.0)[..., None]
    return np.clip(img + noise * soft, 0, 255)


# --------------------------------------------------------------------------

def feather_patch(dst, src, rect, feather=28):
    """Copy `rect` from src into dst with a soft edge and matched exposure."""
    import cv2
    x0, y0, x1, y1 = rect
    d = dst.astype(np.float32)
    s = src.astype(np.float32)

    m = np.zeros(dst.shape[:2], np.float32)
    m[y0:y1, x0:x1] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), feather)

    # Exposure match on the ring just outside the patch, so slide 4's slightly
    # different JPEG rendition doesn't show up as a rectangle.
    ring = np.zeros(dst.shape[:2], bool)
    ring[max(0, y0 - 40):min(dst.shape[0], y1 + 40),
         max(0, x0 - 40):min(dst.shape[1], x1 + 40)] = True
    ring[y0:y1, x0:x1] = False
    delta = d[ring].mean(0) - s[ring].mean(0)
    s = s + delta

    return np.clip(d * (1 - m[..., None]) + s * m[..., None], 0, 255)


# --------------------------------------------------------------------------

def rounded_mask(w, h, radius, ss=4):
    from PIL import ImageDraw
    im = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(im).rounded_rectangle(
        [0, 0, w * ss - 1, h * ss - 1], radius=radius * ss, fill=255)
    return np.asarray(im.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0


def composite(img, logo_rgb, logo_a, x, y, plate=None, scale=1.0):
    out = img.astype(np.float32).copy()
    h, w = logo_a.shape

    if plate:
        pad = int(round(plate["pad"] * scale))
        rad = int(round(plate["radius"] * scale))
        pw, ph = w + 2 * pad, h + 2 * pad
        pxx, pyy = x - pad, y - pad
        m = rounded_mask(pw, ph, rad) * plate.get("opacity", 1.0)
        reg = out[pyy:pyy + ph, pxx:pxx + pw]
        colour = np.asarray(plate.get("colour", (255.0, 255.0, 255.0)), np.float32)
        out[pyy:pyy + ph, pxx:pxx + pw] = colour * m[..., None] + reg * (1 - m[..., None])

    region = out[y:y + h, x:x + w]
    a = logo_a[..., None]
    out[y:y + h, x:x + w] = logo_rgb * a + region * (1 - a)
    return out


def run(variant_suffix="", write_debug=False, logo_variant="exact"):
    os.makedirs(OUT, exist_ok=True)

    originals = {n: np.asarray(Image.open(os.path.join(SRC, n + ".png"))
                               .convert("RGB")).astype(np.float32)
                 for n in ROIS}

    sizes = {img.shape[1] for img in originals.values()}
    if len(sizes) != 1:
        raise SystemExit(f"slides differ in width: {sizes}")
    sizes_w = sizes.pop()
    scale = sizes_w / CANVAS
    px = lambda v: int(round(v * scale))
    box = lambda b: tuple(px(v) for v in b)

    logo_rgb, logo_a = build_logo_v2.render(px(LOGO_W), logo_variant)
    lw, lh = logo_a.shape[1], logo_a.shape[0]
    logo_x = int(round(sizes_w - px(LOGO_MARGIN_RIGHT))) - lw
    logo_y = px(LOGO_Y)
    print(f"logo: {lw}x{lh} at ({logo_x},{logo_y})  [canvas scale {scale:g}]")

    # Slide 5: heal the corrupted corner from slide 4 before anything else.
    originals["slide5_office_broken"] = feather_patch(
        originals["slide5_office_broken"], originals["slide4_office"],
        box(S5_PATCH), feather=max(8, px(28)))

    results = {}
    for name, roi in ROIS.items():
        img = originals[name]
        mask = detect_logo(img, box(roi), scale)
        clean = harmonic_inpaint(img, mask)
        clean = add_matched_grain(clean, img, mask)
        if write_debug:
            Image.fromarray((mask * 255).astype(np.uint8)).save(
                os.path.join(OUT, f"_mask_{name}.png"))
            Image.fromarray(clean.astype(np.uint8)).save(
                os.path.join(OUT, f"_clean_{name}.png"))
        at = (logo_x, logo_y)
        if name == "slide3_why" and SLIDE3_OVERRIDE:
            at = (px(SLIDE3_OVERRIDE[0]), px(SLIDE3_OVERRIDE[1]))
        final = composite(clean, logo_rgb, logo_a, *at, plate=PLATE, scale=scale)
        results[name] = final.astype(np.uint8)
        print(f"  {name}: masked {int(mask.sum()):6d} px")

    suffix = variant_suffix
    order = ["slide1_karlovy", "slide2_paris", "slide3_why",
             "slide4_office", "slide5_office_broken"]
    for i, name in enumerate(order, 1):
        img = Image.fromarray(results[name])
        base = os.path.join(OUT, f"{i:02d}_{name.replace('_broken','')}{suffix}")
        img.save(base + ".png")
        img.save(base + ".jpg", quality=95, subsampling=0)
    print("wrote", len(order), "slides ->", OUT)
    return results


if __name__ == "__main__":
    import sys
    run(write_debug="--debug" in sys.argv)
