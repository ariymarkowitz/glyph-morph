"""Example: harmonic morph between the words 'minimum' and 'maximum'.

Renders each word as a single antialiased alpha image, aligns them by maximum
cross-correlation, and animates a harmonic (Laplace-field) morph back and forth,
pausing on each end. Shows interactively, or saves a GIF under a headless
backend.

    python morph_words_example.py                  # 'minimum' -> 'maximum'
    python morph_words_example.py hello world       # any pair of words
"""

from __future__ import annotations

import sys

import numpy as np

from glyph_align import align_glyphs, overlay
from glyph_morph import GlyphMorph, ease_in_out

from example import load_font, render_glyph

N_FRAMES = 36   # frames per transition
HOLD = 18       # frames to pause on each end


def main() -> None:
    word_a = sys.argv[1] if len(sys.argv) > 1 else "minimum"
    word_b = sys.argv[2] if len(sys.argv) > 2 else "maximum"

    font = load_font(220)
    alpha_a = render_glyph(word_a, font)
    alpha_b = render_glyph(word_b, font)

    # Align B onto A, then place both on a shared canvas.
    result = align_glyphs(alpha_a, alpha_b)
    a_canvas, b_canvas = overlay(alpha_a, alpha_b, result.dx, result.dy)
    morpher = GlyphMorph(a_canvas, b_canvas, field="harmonic")

    # Pause on A, morph A->B, pause on B, morph back: a seamless loop.
    seq = np.linspace(0.0, 1.0, N_FRAMES)
    ts = np.concatenate([
        np.zeros(HOLD), seq, np.ones(HOLD), seq[::-1],
    ])

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    H, W = a_canvas.shape
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))  # fill the figure, no margins
    ax.axis("off")
    im = ax.imshow(morpher.mask(0.0), cmap="gray_r", vmin=0.0, vmax=1.0)

    def update(frame: int):
        im.set_data(morpher.mask(ease_in_out(float(ts[frame]))))
        return (im,)

    anim = animation.FuncAnimation(
        fig, update, frames=len(ts), interval=40, blit=True
    )

    backend = matplotlib.get_backend().lower()
    if backend == "agg":
        out = f"morph_{word_a}_{word_b}_harmonic.gif"
        anim.save(out, writer=animation.PillowWriter(fps=25))
        print(f"No interactive backend; saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
