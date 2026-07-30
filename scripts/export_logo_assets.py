#!/usr/bin/env python3
"""Export the reconstructed logo as reusable assets.

Produces transparent PNGs at a few common sizes (drawn fresh at each size, so
none of them is a resample of another) plus an SVG for anything that needs to
scale further - print, large banners, a website header.
"""

import json
import os

import numpy as np
import potrace
from PIL import Image

import build_logo as B

OUT = os.path.join(B.ROOT, "logo")
SIZES = [205, 410, 820, 1600]


def png_exports():
    for variant in ("brand", "white"):
        for w in SIZES:
            rgb, a = B.render(w, variant)
            img = np.dstack([rgb, a * 255.0]).clip(0, 255).astype(np.uint8)
            path = os.path.join(OUT, f"grand_voyage_{variant}_{w}.png")
            Image.fromarray(img, "RGBA").save(path)
            print("  ", os.path.basename(path), f"{w}x{img.shape[0]}")


# --------------------------------------------------------------------------

def svg_path_data(binary, turdsize, x0, y0, sx, sy):
    """Trace to real cubic beziers (not the flattened polylines the raster
    pipeline uses) so the SVG stays small and smooth at any zoom."""
    bmp = potrace.Bitmap(np.logical_not(binary.astype(bool)))
    path = bmp.trace(turdsize=turdsize, alphamax=1.0, opticurve=True,
                     opttolerance=0.2)

    def P(p):
        return f"{(p.x - x0) * sx:.2f},{(p.y - y0) * sy:.2f}"

    out = []
    for curve in path:
        d = ["M" + P(curve.start_point)]
        for seg in curve:
            if seg.is_corner:
                d.append("L" + P(seg.c))
                d.append("L" + P(seg.end_point))
            else:
                d.append(f"C{P(seg.c1)} {P(seg.c2)} {P(seg.end_point)}")
        d.append("Z")
        out.append(" ".join(d))
    return " ".join(out)


def sun_fill_datauri(field, bbox, px_w=260):
    """The sun's colour ramp, as a small PNG to be clipped by the vector path.

    A 2-stop linear gradient cannot describe this artwork - the disc is
    saturated while the rays fade unevenly to the left - and fitting one
    bleaches the disc.  Clipping the real colour field to the traced outline
    keeps the edges vector-sharp and the colour exact; the fill itself is so
    smooth that a low-resolution raster is indistinguishable.
    """
    import base64
    import io

    x0, y0, x1, y1 = bbox
    crop = field[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))]
    h = max(1, int(round(px_w * crop.shape[0] / crop.shape[1])))
    img = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).resize(
        (px_w, h), Image.LANCZOS)  # noqa: E501
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def svg_export():
    rgb = B.load_artwork()
    gold_m, navy_m = B.classify(rgb)
    gold_f = B.colour_field(rgb, gold_m)
    navy_f = B.colour_field(rgb, navy_m)
    gold_bin = B.alpha_from_unmix(rgb, gold_f, gold_m) > 0.5
    navy_bin = B.alpha_from_unmix(rgb, navy_f, navy_m) > 0.5

    with open(os.path.join(OUT, "logo_vector.json")) as fh:
        x0, y0, x1, y1 = json.load(fh)["bbox"]

    w, h = 1000.0, 1000.0 * (y1 - y0) / (x1 - x0)
    sx, sy = w / (x1 - x0), h / (y1 - y0)

    gold_d = svg_path_data(gold_bin, 64, x0, y0, sx, sy)
    navy_d = svg_path_data(navy_bin, 48, x0, y0, sx, sy)

    sun_uri = sun_fill_datauri(gold_f, (x0, y0, x1, y1))
    navy_hex = "#%02X%02X%02X" % tuple(
        int(round(v)) for v in np.median(rgb[navy_bin], 0))

    for variant, ink in (("brand", navy_hex), ("white", "#FFFFFF")):
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}">
  <defs>
    <clipPath id="sunclip" clipPathUnits="userSpaceOnUse">
      <path clip-rule="evenodd" d="{gold_d}"/>
    </clipPath>
  </defs>
  <g clip-path="url(#sunclip)">
    <image xlink:href="{sun_uri}" x="0" y="0" width="{w:.0f}" height="{h:.0f}" preserveAspectRatio="none"/>
  </g>
  <path fill="{ink}" fill-rule="evenodd" d="{navy_d}"/>
</svg>
"""
        path = os.path.join(OUT, f"grand_voyage_{variant}.svg")
        with open(path, "w") as fh:
            fh.write(svg)
        print("  ", os.path.basename(path), f"{len(svg)//1024} KB  ink {ink}")



if __name__ == "__main__":
    print("PNG:")
    png_exports()
    print("SVG:")
    svg_export()
