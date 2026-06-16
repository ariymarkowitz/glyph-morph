"""Example: align two text glyphs by maximum cross-correlation and show them.

Renders two glyphs (default 'e' and 'd') as antialiased alpha images, computes
the displacement that best overlays the second onto the first, and draws the
result. Run with no GUI backend and it falls back to saving a PNG.

    python example.py            # glyphs 'e' and 'd'
    python example.py A V        # any pair of characters
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from glyph_align import align_glyphs, overlay

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_glyph(char: str, font: ImageFont.FreeTypeFont, pad: int = 8) -> np.ndarray:
    """Render a single character to a tightly-cropped alpha array in [0, 1]."""
    # Measure the glyph's tight bounding box.
    tmp = Image.new("L", (1, 1))
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), char, font=font)
    x0, y0, x1, y1 = bbox
    w, h = (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad

    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((pad - x0, pad - y0), char, fill=255, font=font)
    return np.asarray(img, dtype=np.float64) / 255.0


def main() -> None:
    char_a = sys.argv[1] if len(sys.argv) > 1 else "e"
    char_b = sys.argv[2] if len(sys.argv) > 2 else "d"

    font = load_font(220)
    alpha_a = render_glyph(char_a, font)
    alpha_b = render_glyph(char_b, font)

    result = align_glyphs(alpha_a, alpha_b)
    print(f"Best displacement for '{char_b}' onto '{char_a}': "
          f"(dx={result.dx}, dy={result.dy})  score={result.score:.4f}")

    a_canvas, b_canvas = overlay(alpha_a, alpha_b, result.dx, result.dy)

    # Compose an RGB image: A in red, B in blue, overlap shows as magenta.
    H, W = a_canvas.shape
    rgb = np.ones((H, W, 3))
    rgb[..., 1] -= np.maximum(a_canvas, b_canvas)          # remove green where either
    rgb[..., 2] -= a_canvas                                # A subtracts blue -> red
    rgb[..., 0] -= b_canvas                                # B subtracts red  -> blue
    rgb = np.clip(rgb, 0.0, 1.0)

    try:
        import matplotlib
        try:
            matplotlib.use("MacOSX")
        except Exception:
            pass
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(rgb)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_axis_off()
        fig.tight_layout(pad=0)

        if matplotlib.get_backend().lower() in ("agg",):
            out = "overlay.png"
            fig.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0)
            print(f"No interactive backend; saved {out}")
        else:
            plt.show()
    except ImportError:
        out = "overlay.png"
        Image.fromarray((rgb * 255).astype(np.uint8)).save(out)
        print(f"matplotlib unavailable; saved {out}")


if __name__ == "__main__":
    main()
