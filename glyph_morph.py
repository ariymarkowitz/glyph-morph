"""Morph between two aligned glyph images, with anti-aliased output.

Steps:

1. Threshold each glyph's alpha at 0.5 to get its filled (binary) region. Form
   the *intersection* (in both), and the two *exclusive* regions ``A-only`` and
   ``B-only``.

2. On each exclusive region compute a depth field running from 0 on the shared
   core to 1 on the glyph's outline. By default this is the geodesic distance
   from the core (normalised to [0, 1]); its level sets are wavefronts at
   constant arc length, so along an extended segment (the stem of a 'd', say)
   the morph front advances at a constant rate when ``t`` is swept linearly.
   A harmonic (Laplace) field is also available via ``field="harmonic"``; it is
   reparametrised per component (see :func:`equalize_area`) so each component's
   visible area still changes linearly in ``t`` despite the field saturating.

3. Turn that into a per-pixel switch schedule ``T(x)`` -- the value of ``t`` at
   which the pixel flips between glyphs.

4. Anti-alias the output. The hard silhouette would flip whole pixels on at
   once, giving jagged edges that crawl pixel-by-pixel as ``t`` is swept. Two
   things produce edges in a frame: the *static outline* (where the glyph meets
   the background) and the *moving front* (the level set ``T = t`` sweeping
   across an exclusive region). The static outline is anti-aliased by reusing
   the glyphs' own (already anti-aliased) alpha. The moving front is given a
   ~1px-wide soft edge: ``(T - t)`` divided by the local spacing of ``T``'s
   level sets (precomputed, ~ ``|grad T|``) is the signed distance from the
   pixel centre to the front in pixel units, which maps straight to a coverage
   fraction. The front therefore advances continuously and sub-pixel as ``t``
   changes, so the animation is smooth and free of crawling/popping, while the
   shared interior stays fully solid.

The output is a coverage mask in [0, 1]. ``mask(0)`` is glyph A's anti-aliased
alpha and ``mask(1)`` is glyph B's.
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


def equalize_area(depth: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Reparametrise ``depth`` per component so swept area is linear in level.

    Each connected component of ``region`` has its depth values replaced by
    their empirical CDF (rank-based histogram equalisation): the pixel with the
    ``k``-th smallest depth (of ``m`` pixels) is mapped to ``(k + 0.5) / m``.

    This preserves the *ordering* of the original field -- the same pixels
    switch in the same sequence -- but rescales the timing. Because the values
    are now uniform on (0, 1), the number of pixels with ``depth <= x`` is
    exactly ``x * m``, so the component's visible area grows/shrinks at a
    constant rate as ``t`` is swept linearly. This straightens out fields that
    are non-linear in area (notably the harmonic field, which saturates across
    thin strokes and so otherwise makes components vanish/appear suddenly at one
    end of the morph and crawl at the other).

    The ``+ 0.5`` / ``m`` offset keeps every value strictly inside (0, 1), as in
    :func:`_normalised_distance`, so no pixel switches exactly at ``t=0`` / ``1``.
    """
    out = np.array(depth, dtype=float)
    labels, n = ndimage.label(region)
    for lab in range(1, n + 1):
        comp = labels == lab
        d = depth[comp]
        m = d.size
        # rank[i] = number of component pixels with a strictly smaller depth
        # (ties broken arbitrarily but stably); the empirical CDF is then
        # (rank + 0.5) / m, uniform on (0, 1).
        order = np.argsort(d, kind="stable")
        rank = np.empty(m, dtype=np.int64)
        rank[order] = np.arange(m)
        out[comp] = (rank + 0.5) / m
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


def _level_spacing(field: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Local spacing of ``field``'s level sets on ``region`` (~ ``|grad field|``).

    For each pixel, the mean absolute difference of ``field`` to its in-region
    4-neighbours. On a linear ramp this is exactly the per-pixel slope, i.e. how
    much ``field`` changes between adjacent pixels; its reciprocal is how many
    pixels the ``field = c`` level set moves when ``c`` changes by one. Computed
    only from in-region neighbours so the discontinuity at the region's border
    (where ``field`` jumps to its out-of-region value) never leaks in. Zero
    outside ``region`` and on isolated single pixels.
    """
    total = np.zeros_like(field, dtype=float)
    count = np.zeros_like(field, dtype=float)
    for dy, dx in _NEIGHBOURS:
        nbr = np.roll(np.roll(field, -dy, axis=0), -dx, axis=1)
        nbr_in = np.roll(np.roll(region, -dy, axis=0), -dx, axis=1)
        valid = region & nbr_in
        total[valid] += np.abs(field - nbr)[valid]
        count[valid] += 1.0
    spacing = np.zeros_like(field, dtype=float)
    nz = count > 0
    spacing[nz] = total[nz] / count[nz]
    return spacing


class GlyphMorph:
    """Precomputed anti-aliased morph between two aligned alpha images.

    Both inputs must share the same shape (align them first, e.g. with
    :func:`glyph_align.align_glyphs` + :func:`glyph_align.overlay`).

    Args:
        field: How to build the depth field on each exclusive region.
            ``"geodesic"`` (default) is the geodesic distance from the shared
            core -- linear along an extended segment such as a stem.
            ``"harmonic"`` solves Laplace's equation (0 on the core, 1 on the
            glyph outline); smooth, but it saturates across thin strokes so the
            front is not linear along a stem.
        equalize: For the ``"harmonic"`` field, reparametrise each component's
            depth via :func:`equalize_area` so its visible area changes at a
            constant rate as ``t`` is swept (default ``True``). Without this the
            saturating harmonic field makes components shrink/grow suddenly at
            one end of the morph and crawl at the other. Ignored for
            ``"geodesic"``, which is already built to be linear along a stem.
        aa_width: Width, in pixels, of the soft edge given to the moving front
            (default ``1.0``). Larger values blur the front more; smaller values
            sharpen it toward the hard binary silhouette.
    """

    def __init__(
        self,
        alpha_a: np.ndarray,
        alpha_b: np.ndarray,
        *,
        threshold: float = 0.5,
        field: str = "geodesic",
        equalize: bool = True,
        aa_width: float = 1.0,
    ) -> None:
        a = np.asarray(alpha_a, dtype=float)
        b = np.asarray(alpha_b, dtype=float)
        if a.shape != b.shape:
            raise ValueError("images must have the same shape")

        # Keep the glyphs' own anti-aliased alpha to render the static outlines.
        self.alpha_a = np.clip(a, 0.0, 1.0)
        self.alpha_b = np.clip(b, 0.0, 1.0)
        self.aa_width = float(aa_width)

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
            if equalize:
                depth_a = equalize_area(depth_a, self.a_only)
                depth_b = equalize_area(depth_b, self.b_only)
        else:
            raise ValueError("field must be 'geodesic' or 'harmonic'")

        # Every frame is a per-pixel blend  (1 - s)*alpha_a + s*alpha_b , where
        # the switch fraction s(x, t) sweeps 0 -> 1 as the morph front crosses x.
        # We build a switch *centre* Ts(x) -- the t at which the pixel is half
        # switched -- and a *width* W(x) in t, then s is a clamped ramp of
        # (t - Ts) / W. Because s makes a sharp (~1px) spatial front, only a
        # one-pixel band is ever a partial mix of the two glyphs, so there is no
        # ghosting; everywhere else the pixel is wholly one glyph or the other.
        #   A-only erodes from the outline (Ts ~ 0) inward to the core (Ts ~ 1).
        #   B-only grows from the core (Ts ~ 0) outward to the outline (Ts ~ 1).
        #   Intersection is solid throughout: Ts = 0.5, W = 1, so it just
        #     cross-fades alpha_a -> alpha_b (invisible, both are ~opaque).
        Ts = np.full(a.shape, 0.5)
        Ts[self.a_only] = 1.0 - depth_a[self.a_only]
        Ts[self.b_only] = depth_b[self.b_only]

        spacing = (
            _level_spacing(Ts, self.a_only) + _level_spacing(Ts, self.b_only)
        )
        W = np.ones(a.shape)
        excl = self.a_only | self.b_only
        # Real edges set their width from the level-set spacing, giving a ~1px
        # spatial front; the front's pixels switch in sequence, so it glides.
        # An isolated single-pixel speck (spacing 0: no in-region neighbour) has
        # no spatial front, so it would hard-pop on/off in one frame. Fade those
        # over a short t-window (W_SPECK) instead. These specks are the sub-pixel
        # leftovers of thresholding coincident outlines; softened, they vanish.
        W_SPECK = 0.08
        w_excl = spacing[excl] * self.aa_width
        w_excl[w_excl <= 0.0] = W_SPECK
        W[excl] = w_excl

        # A glyph's sub-threshold outline fringe (alpha in (0, 0.5)) lies in
        # neither filled region, so it has no schedule of its own. Give each such
        # pixel the schedule of the nearest filled pixel, so the fringe switches
        # together with the outline it borders -- not all at once with global t.
        # Without this a near-vertical edge sitting just below threshold (e.g. a
        # 'd' stem) lights up as a faint line along its whole length mid-morph.
        filled = mask_a | mask_b
        iy, ix = ndimage.distance_transform_edt(
            ~filled, return_distances=False, return_indices=True
        )
        Ts = Ts[iy, ix]
        W = W[iy, ix]

        # Normalise each pixel's ramp by its own endpoint values so that s is
        # *exactly* 0 at t=0 and *exactly* 1 at t=1 (clamping the raw ramp would
        # leave the first/last pixels to switch slightly off). This makes mask(0)
        # and mask(1) reproduce alpha_a and alpha_b to the last fringe pixel.
        self.Ts = Ts
        self.W = W
        self._s0 = np.clip((0.0 - Ts) / W + 0.5, 0.0, 1.0)
        s1 = np.clip((1.0 - Ts) / W + 0.5, 0.0, 1.0)
        self._sden = np.maximum(s1 - self._s0, 1e-6)

    def mask(self, t: float) -> np.ndarray:
        """Anti-aliased morph coverage at parameter ``t`` in [0, 1].

        Returns a float coverage mask in [0, 1]. ``mask(0)`` is glyph A's
        anti-aliased alpha *exactly* and ``mask(1)`` is glyph B's. Apply easing
        to ``t`` before calling for eased motion.
        """
        s_raw = np.clip((t - self.Ts) / self.W + 0.5, 0.0, 1.0)
        s = np.clip((s_raw - self._s0) / self._sden, 0.0, 1.0)
        return (1.0 - s) * self.alpha_a + s * self.alpha_b


def ease_in_out(s: float) -> float:
    """Smootherstep in-out easing: flat slope at both ends, 0->1."""
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (s * (s * 6.0 - 15.0) + 10.0)


def morph(alpha_a: np.ndarray, alpha_b: np.ndarray, t: float, **kwargs) -> np.ndarray:
    """Convenience one-shot morph: binary mask at parameter ``t``."""
    return GlyphMorph(alpha_a, alpha_b, **kwargs).mask(t)
