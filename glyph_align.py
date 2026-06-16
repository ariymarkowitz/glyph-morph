"""Align two glyph images by maximum cross-correlation.

A glyph is given as a single-colour pixel image with a continuous alpha
channel (the alpha encodes coverage / antialiasing).  We treat the alpha
channel as a scalar signal and find the integer pixel displacement that, when
applied to the second glyph, maximises its cross-correlation with the first.

The core routine returns a displacement vector ``(dx, dy)`` measured in pixels,
where ``dx`` is the horizontal (column) shift and ``dy`` the vertical (row)
shift to apply to glyph *B* so that it best overlays glyph *A*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class Alignment:
    """Result of aligning glyph B onto glyph A.

    Attributes:
        dx: Horizontal shift (columns) to apply to B. Positive = rightwards.
        dy: Vertical shift (rows) to apply to B. Positive = downwards.
        score: Raw correlation score (total overlapping ink) at the chosen
            displacement.
    """

    dx: int
    dy: int
    score: float

    @property
    def vector(self) -> tuple[int, int]:
        return (self.dx, self.dy)


def load_glyph_alpha(path: str) -> np.ndarray:
    """Load an image and return its alpha channel as a float array in [0, 1].

    If the image has no alpha channel, the (inverted) luminance is used so that
    ink -> high value and background -> 0, matching the alpha convention.
    """
    from PIL import Image

    img = Image.open(path)
    if "A" in img.getbands():
        alpha = np.asarray(img.convert("RGBA"))[..., 3]
    else:
        # No alpha: assume dark ink on light background -> invert luminance.
        alpha = 255 - np.asarray(img.convert("L"))
    return alpha.astype(np.float64) / 255.0


def _cross_correlate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full 2-D cross-correlation of ``a`` and ``b`` computed via FFT.

    ``out[i, j] = sum_{m,n} a[m, n] * b[m - lag_row, n - lag_col]`` where
    ``lag_row = i - (b.rows - 1)`` and ``lag_col = j - (b.cols - 1)``.
    """
    # Correlation is convolution with a flipped kernel.
    return fftconvolve(a, b[::-1, ::-1], mode="full")


def align_glyphs(
    alpha_a: np.ndarray,
    alpha_b: np.ndarray,
) -> Alignment:
    """Find the displacement of B that maximises cross-correlation with A.

    Uses the raw (unnormalised) correlation sum, which is the total overlapping
    ink at each displacement -- so the chosen alignment is the one that overlaps
    the two glyphs the most. Unlike a normalised score (NCC), this won't slide
    one glyph almost off the other to line up a single strongly-correlated
    sliver, which matters for glyphs of repeated strokes (e.g. whole words).

    Args:
        alpha_a: 2-D float array (alpha of glyph A), values in [0, 1].
        alpha_b: 2-D float array (alpha of glyph B), values in [0, 1].

    Returns:
        An ``Alignment`` with the integer shift ``(dx, dy)`` for glyph B.
    """
    a = np.asarray(alpha_a, dtype=np.float64)
    b = np.asarray(alpha_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("alpha images must be 2-D arrays")

    hb, wb = b.shape

    # Sum of a*b over the overlapping region at each displacement.
    score_map = _cross_correlate(a, b)

    pi, pj = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
    lag_row = pi - (hb - 1)
    lag_col = pj - (wb - 1)

    return Alignment(dx=int(lag_col), dy=int(lag_row),
                     score=float(score_map[pi, pj]))


def overlay(
    alpha_a: np.ndarray,
    alpha_b: np.ndarray,
    dx: int,
    dy: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Place A and B on a common canvas with B shifted by ``(dx, dy)``.

    Returns two alpha arrays of identical shape (``a_canvas``, ``b_canvas``)
    suitable for compositing or visualisation.
    """
    ha, wa = alpha_a.shape
    hb, wb = alpha_b.shape

    # B's top-left corner in A's coordinate frame, then a global offset so the
    # union of both rectangles fits with a non-negative origin.
    b_top, b_left = dy, dx
    top = min(0, b_top)
    left = min(0, b_left)
    bottom = max(ha, b_top + hb)
    right = max(wa, b_left + wb)

    H, W = bottom - top, right - left
    a_canvas = np.zeros((H, W), dtype=alpha_a.dtype)
    b_canvas = np.zeros((H, W), dtype=alpha_b.dtype)

    a_canvas[-top:-top + ha, -left:-left + wa] = alpha_a
    bo_r, bo_c = b_top - top, b_left - left
    b_canvas[bo_r:bo_r + hb, bo_c:bo_c + wb] = alpha_b

    return a_canvas, b_canvas
