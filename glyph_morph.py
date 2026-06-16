"""Morph between two aligned glyph images (binary, no antialiasing).

Steps:

1. Threshold each glyph's alpha at 0.5 to get its filled (binary) region. Form
   the *intersection* (in both), and the two *exclusive* regions ``A-only`` and
   ``B-only``.

2. On each exclusive region compute a depth field running from 0 on the shared
   core to 1 on the glyph's outline. By default this is the geodesic distance
   from the core (normalised to [0, 1]); its level sets are wavefronts at
   constant arc length, so along an extended segment (the stem of a 'd', say)
   the morph front advances at a constant rate when ``t`` is swept linearly.
   A harmonic (Laplace) field is also available via ``field="harmonic"``.

3. Turn that into a per-pixel switch schedule ``T(x)`` -- the value of ``t`` at
   which the pixel flips between glyphs -- and at each ``t`` emit the hard
   silhouette: core, plus not-yet-eroded A-only, plus already-grown B-only.

The output is a 0/1 mask. ``mask(0)`` is glyph A's threshold mask and
``mask(1)`` is glyph B's. No antialiasing is applied.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))
# 8-connected steps with Euclidean weights, for isotropic geodesic distance.
_STEPS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, 2.0 ** 0.5), (-1, 1, 2.0 ** 0.5),
    (1, -1, 2.0 ** 0.5), (1, 1, 2.0 ** 0.5),
)


def geodesic_distance(region: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Geodesic distance from ``core`` measured through ``region`` (Dijkstra).

    Returns a float array of distances; ``inf`` where unreachable.
    """
    H, W = region.shape
    dist = np.full((H, W), np.inf)

    seeds = region & ndimage.binary_dilation(core)
    heap = [(0.0, int(y), int(x)) for y, x in zip(*np.nonzero(seeds))]
    for _, y, x in heap:
        dist[y, x] = 0.0
    heapq.heapify(heap)

    while heap:
        d, y, x = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        for dy, dx, w in _STEPS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and region[ny, nx]:
                nd = d + w
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(heap, (nd, ny, nx))
    return dist


def _normalised_distance(region: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Per-component schedule field for ``region``, in (0, 1).

    Each connected component is normalised by its own maximum distance so it
    sweeps its full range as ``t`` runs 0->1::

        D = (d + 0.5) / (dmax_c + 1)

    The ``+ 0.5`` / ``+ 1`` offset places the seed (d=0) and the tip (d=dmax_c)
    half a step inside (0, 1) rather than at 0 and 1. Two consequences:

    * No pixel switches exactly at t=0 or t=1, so there is no jump between a held
      end frame and the next -- yet the endpoints of mask() stay exact (every D
      is strictly inside (0, 1)).
    * A single-pixel component (dmax_c = 0) maps to D = 0.5, i.e. it flips at the
      middle of the morph instead of at an end -- which is where the lone specks
      the 0.5 threshold leaves along coincident outlines belong.

    D is affine in d, so an extended segment (e.g. a stem) stays linear.
    """
    dist = geodesic_distance(region, core)
    out = np.zeros(region.shape)
    labels, n = ndimage.label(region)
    for lab in range(1, n + 1):
        comp = labels == lab
        d = dist[comp]
        fin = d[np.isfinite(d)]
        dmax_c = fin.max() if fin.size else 0.0
        # Unreachable pixels (a component not touching the core) -> 0.5.
        out[comp] = np.where(np.isfinite(d), (d + 0.5) / (dmax_c + 1.0), 0.5)
    return out


def solve_harmonic(unknown: np.ndarray, one_mask: np.ndarray) -> np.ndarray:
    """Solve Laplace's equation on ``unknown`` with Dirichlet data.

    ``unknown`` pixels are solved for; ``one_mask`` pixels are held at 1 and all
    other non-unknown pixels at 0. Image borders use a no-flux (Neumann)
    condition. Returns the harmonic field over the whole grid.

    By the maximum principle the solution on ``unknown`` is strictly in (0, 1),
    so it can be used directly as a depth field. Note that for a thin stroke the
    u=1 condition on the two nearby long sides makes the field saturate across
    the width, so unlike the geodesic distance it is *not* linear along a stem.
    """
    H, W = unknown.shape
    u = np.where(one_mask, 1.0, 0.0)

    ys, xs = np.nonzero(unknown)
    n = ys.size
    if n == 0:
        return u

    idx = -np.ones((H, W), dtype=np.int64)
    idx[ys, xs] = np.arange(n)

    diag = np.zeros(n)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    rhs = np.zeros(n)

    for dy, dx in _NEIGHBOURS:
        yy, xx = ys + dy, xs + dx
        inb = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        diag[inb] += 1.0  # each in-bounds neighbour adds to the stencil

        k = np.nonzero(inb)[0]
        nyy, nxx = yy[inb], xx[inb]
        nb_unknown = unknown[nyy, nxx]

        rows.append(k[nb_unknown])
        cols.append(idx[nyy[nb_unknown], nxx[nb_unknown]])
        data.append(-np.ones(int(nb_unknown.sum())))

        nb_one = one_mask[nyy, nxx] & ~nb_unknown
        np.add.at(rhs, k[nb_one], 1.0)

    rows.append(np.arange(n))
    cols.append(np.arange(n))
    data.append(diag)

    A = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    u[ys, xs] = spsolve(A, rhs)
    return u


class GlyphMorph:
    """Precomputed binary morph between two aligned alpha images.

    Both inputs must share the same shape (align them first, e.g. with
    :func:`glyph_align.align_glyphs` + :func:`glyph_align.overlay`).

    Args:
        field: How to build the depth field on each exclusive region.
            ``"geodesic"`` (default) is the geodesic distance from the shared
            core -- linear along an extended segment such as a stem.
            ``"harmonic"`` solves Laplace's equation (0 on the core, 1 on the
            glyph outline); smooth, but it saturates across thin strokes so the
            front is not linear along a stem.
    """

    def __init__(
        self,
        alpha_a: np.ndarray,
        alpha_b: np.ndarray,
        *,
        threshold: float = 0.5,
        field: str = "geodesic",
    ) -> None:
        a = np.asarray(alpha_a)
        b = np.asarray(alpha_b)
        if a.shape != b.shape:
            raise ValueError("images must have the same shape")

        mask_a = a >= threshold
        mask_b = b >= threshold
        self.inter = mask_a & mask_b
        self.a_only = mask_a & ~mask_b
        self.b_only = mask_b & ~mask_a

        if field == "geodesic":
            depth_a = _normalised_distance(self.a_only, self.inter)
            depth_b = _normalised_distance(self.b_only, self.inter)
        elif field == "harmonic":
            depth_a = solve_harmonic(self.a_only, one_mask=~mask_a)
            depth_b = solve_harmonic(self.b_only, one_mask=~mask_b)
        else:
            raise ValueError("field must be 'geodesic' or 'harmonic'")

        # Switch schedule T(x): the value of t at which a pixel flips.
        #   A-only: erode from the outline (depth=1 -> T=0) toward the core.
        #   B-only: grow from the core (depth=0 -> T=0) toward the outline.
        T = np.zeros(a.shape)
        T[self.a_only] = 1.0 - depth_a[self.a_only]
        T[self.b_only] = depth_b[self.b_only]
        self.T = T

    def mask(self, t: float) -> np.ndarray:
        """Binary morph silhouette at parameter ``t`` in [0, 1] (0/1 float).

        ``mask(0)`` is glyph A's mask and ``mask(1)`` is glyph B's. Apply easing
        to ``t`` before calling for eased motion.
        """
        present = (
            self.inter
            | (self.a_only & (self.T >= t))
            | (self.b_only & (self.T <= t))
        )
        return present.astype(float)


def ease_in_out(s: float) -> float:
    """Smootherstep in-out easing: flat slope at both ends, 0->1."""
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (s * (s * 6.0 - 15.0) + 10.0)


def morph(alpha_a: np.ndarray, alpha_b: np.ndarray, t: float, **kwargs) -> np.ndarray:
    """Convenience one-shot morph: binary mask at parameter ``t``."""
    return GlyphMorph(alpha_a, alpha_b, **kwargs).mask(t)
