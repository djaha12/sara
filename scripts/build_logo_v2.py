#!/usr/bin/env python3
"""Cut the supplied GRAND VOYAGE logo out of its black background.

The artwork is clean vector-quality, but it sits on solid black *and* the sun's
disc and the waves carry a black stroke, so "delete the black" would eat the
strokes.  The two are separated structurally instead of by colour:

  * emblem - a morphological closing of the coloured shapes re-attaches any
    black that is sandwiched between them (the disc ring, the wave outlines)
    while leaving the wide black wedges between the sun's rays as background;
  * wordmark - the letters carry no black stroke, so plain alpha-vs-black works
    and the letter counters fall out transparent on their own.

Nothing is redrawn: every surviving pixel keeps its original colour, which is
what "1 to 1" requires.
"""

import os

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "logo", "logo_source_v2_black.png")
OUT_DIR = os.path.join(ROOT, "logo")

# Row that separates the emblem from the wordmark (measured: the two are
# divided by a 44 px band with no coloured pixels at all, rows 816-859).
SPLIT_ROW = 838
# Bridges the ~10 px disc ring and the wave outlines without closing the
# much wider wedges of background between the sun's rays.
CLOSE_R = 13


def load():
    return np.asarray(Image.open(SRC).convert("RGB")).astype(np.float32)


def coloured(rgb):
    mx, mn = rgb.max(2), rgb.min(2)
    return (mx > 70) & ((mx - mn) > 35)


def alpha_vs_black(rgb, ref):
    """I = a*F over black  =>  a = <I,F>/<F,F>, per pixel against ink colour F."""
    num = (rgb * ref).sum(2)
    den = float((ref * ref).sum()) or 1.0
    return np.clip(num / den, 0.0, 1.0)


def build(close_r=CLOSE_R):
    rgb = load()
    C = coloured(rgb)

    emblem = np.zeros_like(C)
    emblem[:SPLIT_ROW] = True

    # --- emblem: close the coloured shapes to recover their black strokes ----
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_r * 2 + 1,) * 2)
    solid = cv2.morphologyEx((C & emblem).astype(np.uint8), cv2.MORPH_CLOSE, k)
    a_emb = cv2.GaussianBlur(solid.astype(np.float32), (0, 0), 0.8)

    # The rays' own edges have no stroke, so let their colour-derived alpha win
    # there - it is properly anti-aliased, the closing is not.
    yellow = rgb[C & emblem & (rgb[..., 0] > rgb[..., 2])]
    ref_y = yellow.mean(0) if len(yellow) else np.array([255., 220., 30.])
    a_col = alpha_vs_black(rgb, ref_y)
    a_emb = np.maximum(a_emb, np.where(C & emblem, a_col, 0.0))
    a_emb[~emblem] = 0.0

    # --- wordmark: no stroke, so alpha straight off the blue ----------------
    word = ~emblem
    blue = rgb[C & word]
    ref_b = blue.mean(0) if len(blue) else np.array([60., 90., 175.])
    a_word = alpha_vs_black(rgb, ref_b)
    a_word[~word] = 0.0
    # kill the faint JPEG haze in the empty margins
    a_word[a_word < 0.06] = 0.0

    alpha = np.clip(a_emb + a_word, 0.0, 1.0)

    # Un-premultiply: pixels are ink composited over black, so the stored
    # colour has to be divided back out or edges look dark when re-composited.
    safe = np.maximum(alpha, 1e-3)[..., None]
    out_rgb = np.clip(rgb / safe, 0, 255)
    out_rgb[alpha < 0.02] = 0.0

    ys, xs = np.nonzero(alpha > 0.02)
    bbox = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return out_rgb, alpha, bbox


def save_master():
    rgb, alpha, (x0, y0, x1, y1) = build()
    img = np.dstack([rgb, alpha * 255.0])[y0:y1, x0:x1]
    img = np.clip(img, 0, 255).astype(np.uint8)
    path = os.path.join(OUT_DIR, "grand_voyage_v2_master.png")
    Image.fromarray(img, "RGBA").save(path)
    print("master", img.shape[1], "x", img.shape[0], "->", os.path.basename(path))
    return path


def render(width, variant="exact"):
    """Draw the logo at `width` px, downscaled from the master in one step.

    variant "exact" is the supplied artwork untouched.  variant "white" swaps
    only the blue ink for white - for dark or blue backdrops where the original
    royal blue vanishes - and leaves the sun and the black strokes alone.
    """
    master = Image.open(os.path.join(OUT_DIR, "grand_voyage_v2_master.png"))
    h = int(round(width * master.height / master.width))
    im = master.resize((int(width), h), Image.LANCZOS)
    arr = np.asarray(im).astype(np.float32)
    rgb, a = arr[..., :3], arr[..., 3] / 255.0

    if variant == "white":
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        blueness = np.clip((b - r) / 90.0, 0.0, 1.0)   # 0 for yellow/black
        rgb = rgb * (1 - blueness[..., None]) + 255.0 * blueness[..., None]
    return rgb, a


if __name__ == "__main__":
    save_master()
    for w in (150, 300, 600, 1200):
        rgb, a = render(w)
        img = np.dstack([rgb, a * 255.0]).clip(0, 255).astype(np.uint8)
        Image.fromarray(img, "RGBA").save(
            os.path.join(OUT_DIR, f"grand_voyage_v2_{w}.png"))
        print("  ", f"grand_voyage_v2_{w}.png", f"{w}x{img.shape[0]}")
