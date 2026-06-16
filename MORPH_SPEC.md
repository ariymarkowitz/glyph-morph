# Glyph Morphing — Implementation Specification

A description of the algorithm that animates a smooth,
anti-aliased morph between two glyph images (e.g. `e` → `d`), controlled by a
parameter `t ∈ [0, 1]` with `t = 0` showing glyph A and `t = 1` showing glyph B.

This is a rough outline of the ideas and the non-obvious design decisions involved.
All of the work was vibe-coded (including most of this document) so it's probably more
worth it to read this document than the source code.

---

## 1. Core idea

The morph is a *spatial
sweep*: the regions where the two glyphs differ are switched on/off
progressively as a soft "front" sweeps across them, while the shared interior
stays solid throughout. The result reads as one shape growing/retracting into
the other.

Two properties are engineered in:

1. **Constant-rate motion** — the front advances at steady speed (and optionally
   each disconnected piece changes its *area* at a steady rate) as `t` is swept
   linearly, so nothing crawls then jumps.
2. **Sub-pixel anti-aliasing** — edges are soft and move continuously, with no
   pixel popping or staircase crawl.

A glyph image is a 2-D array of **alpha** (coverage) in `[0, 1]`: `0` =
background, `1` = inked, fractional = the renderer's own edge anti-aliasing.
Only alpha is used. The two inputs must have identical shape.

The pipeline: **align** the two glyphs onto a shared canvas (§2), **precompute**
a per-pixel switch schedule once (§3–6), then cheaply **render** any frame (§7).

---

## 2. Alignment (preprocessing)

Translate glyph B so it best overlaps A, then place both on a common canvas of
equal shape.

Find the integer pixel shift of B that **maximises the raw 2-D cross-correlation**
of the two alpha images — i.e. the shift maximising total overlapping ink:

```
score[shift] = Σ_pixels A · (B shifted by `shift`)
```

Then paste both glyphs onto a canvas
sized to fit their union, B at the chosen offset.

Alignment is just a convenience; the morph proper only requires two equal-shaped
alpha images.

---

## 3. Region decomposition

Threshold each glyph at `0.5` to get binary filled regions `maskA`, `maskB`, and
split the plane into:

- **Intersection** = `maskA AND maskB` — the shared core; stays solid.
- **A-only** = filled in A but not B — must *disappear* as `t: 0 → 1`.
- **B-only** = filled in B but not A — must *appear* as `t: 0 → 1`.

---

## 4. Depth field

On each exclusive region build a scalar **depth field** `D ∈ (0, 1)` encoding
*the order in which pixels switch*: low depth switches early, high depth late.
`D = 0` at the boundary the front grows/erodes *from* (the shared core), `D = 1`
at the boundary it ends *at* (the glyph outline). Two field types:

**Geodesic (default).** The geodesic distance of each region pixel from the
shared core, measured *through the region only* (shortest in-region path; use
8-connected steps with Euclidean weights for near-isotropy). Then normalise
**per connected component** by that component's own maximum distance so each
sweeps its full range. Because distance is linear along an extended segment
(e.g. a stem), the front advances at constant speed there — the main reason this
is the default.

**Harmonic (alternative).** Solve Laplace's equation on the region with
Dirichlet data `0` on the core side and `1` on the outer outline (the pixels
outside the glyph's own filled region), Neumann at image borders. Standard
5-point stencil, solved as a sparse linear system; the maximum principle keeps
the solution strictly in `(0, 1)`. Smoother than geodesic, but it **saturates
across thin strokes**, so it is *not* linear along a stem — pair it with §5.

**Normalisation offset.** When normalising a component, place the seed and tip
*strictly inside* `(0, 1)` (e.g. map distances via `(d + 0.5)/(dmax + 1)`)
rather than at the exact ends. This ensures no pixel switches exactly at `t = 0`
or `t = 1` (avoiding a jump out of a held end frame), and makes a single-pixel
component map to `0.5` — flipping mid-morph, where the stray specks left by
thresholding coincident outlines belong.

---

## 5. Area equalisation (recommended for harmonic)

Reparametrise a depth field, per connected component, so its **visible area
changes at a constant rate** as `t` sweeps linearly. Replace each pixel's depth
by its rank-based empirical CDF within the component: the pixel with the `k`-th
smallest depth of `m` becomes `(k + 0.5)/m`. This **preserves switching order**
but makes the values uniform on `(0, 1)`, so the count of pixels below any level
`x` is exactly `x·m` → linear area.

Apply this to the harmonic field (otherwise its saturation makes components
appear/vanish suddenly at one end and crawl at the other). Skip it for geodesic,
which is already linear along a stroke.

---

## 6. Switch schedule

Each frame is, per pixel, a blend `(1 − s)·alpha_a + s·alpha_b`, where the
**switch fraction** `s(x, t)` sweeps `0 → 1` as the front crosses the pixel. `s`
is a clamped linear ramp defined by two precomputed fields:

- **Switch centre `Ts`** — the `t` at which the pixel is half-switched:
  - `Ts = 0.5` by default (covers the intersection — a harmless invisible
    cross-fade between two opaque glyphs),
  - `Ts = 1 − D` on **A-only** (erodes outline→core: outline switches off first),
  - `Ts = D` on **B-only** (grows core→outline: core switches on first).
- **Switch width `W`** — the transition width *in `t`*, chosen so the front has a
  fixed spatial thickness of ~1 pixel regardless of the local steepness of `Ts`.

The width comes from the **local spacing of `Ts`'s level sets**: per pixel, the
mean absolute difference of `Ts` to its in-region neighbours (in-region only, so
the jump at the region border doesn't leak in). On a ramp this is the per-pixel
slope; multiply by the desired front width in pixels (default `1`) to get `W`.
This converts a *t*-width into a fixed *pixel*-width, so the soft front is ~1px
wide everywhere.

**Specks.** An isolated single-pixel piece has no in-region neighbour → spacing
`0` → it would hard-pop on/off in one frame. Give such pixels a small fixed
`t`-width instead so they fade quickly rather than pop.

---

## 7. Anti-aliasing, rendering, and exact endpoints

**Two edge types.** *Static outlines* (glyph meeting background) are
anti-aliased for free: the blend carries the shown glyph's own fractional alpha.
The *moving front* gets its softness from the ramp: `(t − Ts)/W` is the signed
distance from the pixel centre to the front in pixel units, mapping directly to
a coverage fraction, so the front advances sub-pixel and smoothly.

**Fringe inheritance.** A glyph's sub-threshold outline fringe (alpha in
`(0, 0.5)`) is in neither filled region and so has no schedule; left alone it
would switch with the global `t` and light up as a faint line mid-morph. Give
every non-filled pixel the schedule (`Ts`, `W`) of its **nearest filled pixel**
(via a distance transform returning nearest-feature indices) so the fringe
switches with the outline it borders.

**Rendering a frame:**

```
s = clamp( (t − Ts) / W + 0.5 , 0, 1)
mask = (1 − s) · alpha_a + s · alpha_b
```

**Exact endpoints.** Because every `Ts` is strictly inside `(0, 1)`, a raw ramp
leaves the first/last pixels slightly off at the ends. Normalise each pixel's
ramp by its own values at `t = 0` and `t = 1` (precompute `s0 = s(0)`,
`s1 = s(1)` per pixel; remap `s ← (s − s0)/(s1 − s0)`, clamped) so that
`mask(0) = alpha_a` and `mask(1) = alpha_b` **exactly**, down to the last fringe
pixel.

`mask(t)` uses only precomputed fields, so animation is cheap. `t` may be eased
(e.g. smootherstep) before rendering for nicer motion.

---

## 8. Summary

**Precompute once:** align (§2) → threshold into intersection / A-only / B-only
(§3) → depth field per exclusive region (§4) → optional area equalisation (§5) →
switch centre `Ts` and width `W`, with speck handling and fringe inheritance
(§6–7) → per-pixel endpoint-normalisation constants.

**Per frame:** ramp `Ts`, `W` against `t`, apply endpoint normalisation, output
the `alpha_a`/`alpha_b` blend (§7).

**Guarantees:** `mask(0) = alpha_a` and `mask(1) = alpha_b` exactly; the front
moves at constant speed (geodesic) and/or constant area rate (equalised); only a
~1px band is ever a partial mix (no ghosting); edges anti-aliased and sub-pixel
(no popping/crawl).

**Parameters:** threshold (`0.5`), field (`geodesic`/`harmonic`), equalise
(on for harmonic), front width in pixels (`1.0`).
