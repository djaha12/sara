#!/usr/bin/env python3
"""Rebuild the GRAND VOYAGE logo as a clean, resolution-independent asset.

The only usable source we have is a phone screenshot of the Instagram avatar,
which is small (~352x220 px of actual artwork) and JPEG-mushy.  Scaling that up
directly would look soft, so instead we:

  1. unmix each pixel against the white avatar background to get a real alpha,
  2. split the artwork into its two ink colours (gold sun / navy waves+wordmark),
  3. trace both masks into vector contours with potrace,
  4. keep a smoothed colour *field* per ink so the sun's gradient survives,
  5. rasterise the contours at any requested size with 4x supersampling.

Step 5 is what actually gets called by the compositor, so the logo is drawn
fresh at its final pixel size on every slide - never resampled, never soft.
"""

import json
import os

import cv2
import numpy as np
import potrace
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SCREENSHOT = os.path.join(ROOT, "logo", "logo_source_screenshot.png")
OUT_DIR = os.path.join(ROOT, "logo")

# Tight crop around the artwork inside the circular avatar (measured by hand).
CROP = (110, 425, 480, 665)
# Work at 6x so potrace sees smooth edges instead of the source's blocky ones.
TRACE_SCALE = 6

WHITE = np.array([251.0, 252.0, 251.0])  # measured avatar background


# --------------------------------------------------------------------------
# 1. segmentation + alpha unmixing
# --------------------------------------------------------------------------

def load_artwork():
    im = Image.open(SCREENSHOT).convert("RGB").crop(CROP)
    w, h = im.size
    im = im.resize((w * TRACE_SCALE, h * TRACE_SCALE), Image.LANCZOS)
    return np.asarray(im).astype(np.float32)


def classify(rgb):
    """Split into gold / navy candidate regions using hue, not brightness."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = rgb.mean(2)

    # Navy is the only ink where blue outruns red.  Gold is the opposite.
    navy = (b - r > 10) & (lum < 215)
    gold = (r - b > 30) & (r > 140)

    navy = clean(navy)
    gold = clean(gold)
    # A pixel can't be both; navy wins because the waves sit on top of the sun.
    gold &= ~navy
    return gold, navy


def clean(mask, k=5):
    m = mask.astype(np.uint8)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern)
    return m.astype(bool)


def colour_field(rgb, mask):
    """Smooth colour of an ink, extended past its own edges.

    Sampling only the *core* of a stroke avoids picking up white fringing from
    the JPEG, and inpainting outward means the field is still defined for the
    anti-aliased rim and for any pixel the vector outline adds back.
    """
    core = cv2.erode(mask.astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if core.sum() < 50:  # stroke too thin to erode - fall back to the mask
        core = mask.astype(np.uint8)
    w = core.astype(np.float32)

    # Normalised-convolution push-out: repeatedly blur the known colours and
    # their weights together, so colour spreads outward while the already-known
    # interior keeps its own value.  Much faster than cv2.inpaint and, for a
    # field this smooth, indistinguishable from it.
    num = rgb * w[..., None]
    den = w.copy()
    for sigma in (4, 8, 16, 32, 64):
        num = cv2.GaussianBlur(num, (0, 0), sigma)
        den = cv2.GaussianBlur(den, (0, 0), sigma)
        known = den > 1e-4
        filled = np.where(known[..., None], num / np.maximum(den, 1e-4)[..., None], 0)
        num = np.where(w[..., None] > 0, rgb * w[..., None], filled * known[..., None])
        den = np.where(w > 0, w, known.astype(np.float32))

    field = num / np.maximum(den, 1e-4)[..., None]
    return cv2.GaussianBlur(field.astype(np.float32), (0, 0), 6)


def alpha_from_unmix(rgb, field, mask):
    """How much ink is in each pixel, assuming ink-over-white compositing.

    I = a*F + (1-a)*W  =>  a = <W-I, W-F> / |W-F|^2
    """
    d_img = WHITE[None, None, :] - rgb
    d_fg = WHITE[None, None, :] - field
    denom = (d_fg * d_fg).sum(2)
    denom = np.maximum(denom, 1e-3)
    a = (d_img * d_fg).sum(2) / denom
    a = np.clip(a, 0.0, 1.0)
    a[~mask] = 0.0
    return a


# --------------------------------------------------------------------------
# 2. vector tracing
# --------------------------------------------------------------------------

def pt(p):
    return (float(p.x), float(p.y))


def trace(binary, turdsize):
    # potrace.Bitmap takes *image* polarity (dark = ink) and inverts internally,
    # so a mask has to go in negated or it traces the background instead.
    bmp = potrace.Bitmap(np.logical_not(binary.astype(bool)))
    path = bmp.trace(turdsize=turdsize, alphamax=1.0, opticurve=True,
                     opttolerance=0.2)
    contours = []
    for curve in path:
        pts = [pt(curve.start_point)]
        for seg in curve:
            if seg.is_corner:
                pts.append(pt(seg.c))
                pts.append(pt(seg.end_point))
            else:
                pts.extend(flatten_bezier(pts[-1], pt(seg.c1),
                                          pt(seg.c2), pt(seg.end_point)))
        contours.append(np.asarray(pts, dtype=np.float64))
    return contours


def flatten_bezier(p0, p1, p2, p3, steps=24):
    t = np.linspace(0.0, 1.0, steps + 1)[1:][:, None]
    mt = 1.0 - t
    pts = (mt ** 3) * np.asarray(p0) + 3 * (mt ** 2) * t * np.asarray(p1) \
        + 3 * mt * (t ** 2) * np.asarray(p2) + (t ** 3) * np.asarray(p3)
    return [tuple(p) for p in pts]


# --------------------------------------------------------------------------
# 3. rasterisation
# --------------------------------------------------------------------------

def rasterise(contours, bbox, out_w, out_h, ss=4):
    """Even-odd fill of the contours, supersampled then box-downsampled."""
    x0, y0, x1, y1 = bbox
    sx = (out_w * ss) / (x1 - x0)
    sy = (out_h * ss) / (y1 - y0)

    acc = np.zeros((out_h * ss, out_w * ss), dtype=bool)
    for c in contours:
        if len(c) < 3:
            continue
        pts = [((px - x0) * sx, (py - y0) * sy) for px, py in c]
        layer = Image.new("1", (out_w * ss, out_h * ss), 0)
        ImageDraw.Draw(layer).polygon(pts, fill=1)
        acc ^= np.asarray(layer, dtype=bool)  # even-odd handles holes

    a = acc.astype(np.float32).reshape(out_h, ss, out_w, ss).mean((1, 3))
    return a


def crop_field(field, bbox, shrink=4):
    """Store only the bbox, at reduced resolution - the field is smooth enough
    that this is lossless in practice and keeps the asset ~20x smaller."""
    x0, y0, x1, y1 = bbox
    crop = field[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))]
    crop = np.clip(crop, 0, 255).astype(np.uint8)
    w = max(1, crop.shape[1] // shrink)
    h = max(1, crop.shape[0] // shrink)
    return np.asarray(Image.fromarray(crop).resize((w, h), Image.LANCZOS))


def sample_field(field, out_w, out_h):
    return np.asarray(
        Image.fromarray(field.astype(np.uint8)).resize((out_w, out_h), Image.LANCZOS)
    ).astype(np.float32)


# --------------------------------------------------------------------------

def build():
    rgb = load_artwork()
    gold_m, navy_m = classify(rgb)

    gold_field = colour_field(rgb, gold_m)
    navy_field = colour_field(rgb, navy_m)

    gold_a = alpha_from_unmix(rgb, gold_field, gold_m)
    navy_a = alpha_from_unmix(rgb, navy_field, navy_m)

    # Binarise at the 50% blend point - that is where the true outline sits.
    gold_bin = gold_a > 0.5
    navy_bin = navy_a > 0.5

    # Let the sun bleed a hair under the waves so no hairline seam can open up
    # where the two vector outlines meet.
    gold_bin |= cv2.dilate(gold_bin.astype(np.uint8),
                           np.ones((3, 3), np.uint8)).astype(bool) & navy_bin

    gold_c = trace(gold_bin, turdsize=64)
    navy_c = trace(navy_bin, turdsize=48)

    ink = gold_bin | navy_bin
    ys, xs = np.nonzero(ink)
    bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))

    payload = {
        "bbox": bbox,
        "gold": [np.round(c, 2).tolist() for c in gold_c],
        "navy": [np.round(c, 2).tolist() for c in navy_c],
    }
    with open(os.path.join(OUT_DIR, "logo_vector.json"), "w") as fh:
        json.dump(payload, fh)

    np.savez_compressed(os.path.join(OUT_DIR, "logo_fields.npz"),
                        gold=crop_field(gold_field, bbox),
                        navy=crop_field(navy_field, bbox))

    print("bbox", bbox, "aspect", (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]))
    print("gold contours", len(gold_c), "navy contours", len(navy_c))
    return payload


# --------------------------------------------------------------------------
# public API used by the compositor
# --------------------------------------------------------------------------

def render(width, variant="brand", ss=4):
    """Draw the logo fresh at `width` px.  Returns an RGBA float array.

    variant "brand" keeps the original palette exactly as it is in the source
    artwork.  variant "white" knocks the navy out to white for dark backdrops
    and leaves the gold sun untouched.
    """
    with open(os.path.join(OUT_DIR, "logo_vector.json")) as fh:
        data = json.load(fh)
    fields = np.load(os.path.join(OUT_DIR, "logo_fields.npz"))

    x0, y0, x1, y1 = data["bbox"]
    aspect = (y1 - y0) / (x1 - x0)
    w = int(round(width))
    h = int(round(width * aspect))

    gold_a = rasterise([np.asarray(c) for c in data["gold"]], data["bbox"], w, h, ss)
    navy_a = rasterise([np.asarray(c) for c in data["navy"]], data["bbox"], w, h, ss)

    gold_rgb = sample_field(fields["gold"], w, h)
    if variant == "white":
        navy_rgb = np.full((h, w, 3), 255.0, np.float32)
    else:
        navy_rgb = sample_field(fields["navy"], w, h)

    # navy over gold
    out_rgb = gold_rgb * gold_a[..., None]
    out_a = gold_a.copy()
    out_rgb = navy_rgb * navy_a[..., None] + out_rgb * (1 - navy_a[..., None])
    out_a = navy_a + out_a * (1 - navy_a)

    return out_rgb, out_a


if __name__ == "__main__":
    build()
    for variant in ("brand", "white"):
        rgb, a = render(600, variant)
        img = np.dstack([rgb, a * 255.0]).clip(0, 255).astype(np.uint8)
        Image.fromarray(img, "RGBA").save(
            os.path.join(OUT_DIR, f"grand_voyage_{variant}_600.png"))
        print("wrote preview", variant)
