# Glyph Morphing — Implementation Specification

A language-independent description of the algorithm that produces a smooth,
anti-aliased animated morph between two glyph images. It is detailed enough to
reconstruct an equivalent implementation in any language with basic image and
sparse-linear-algebra facilities; it deliberately avoids
language/library-specific details.

---

## 1. Overview

The goal is to animate a transition between two single-colour glyph images
(e.g. the letter `e` turning into `d`) controlled by a parameter `t ∈ [0, 1]`,
where `t = 0` shows glyph A exactly and `t = 1` shows glyph B exactly.

The morph is **not** a cross-fade and **not** a vector interpolation. Instead it
is a *spatial sweep*: the parts of the image that differ between the two glyphs
are switched on or off progressively, pixel region by pixel region, as a "front"
sweeps across them. The shared interior of the two glyphs stays solid the whole
time. The result reads as one shape continuously growing/retracting into the
other.

Two properties are engineered in deliberately:

1. **Constant-rate motion** — the sweeping front advances at a steady speed (and,
   optionally, each disconnected piece changes its visible *area* at a steady
   rate) as `t` is swept linearly, so nothing crawls then jumps.
2. **Sub-pixel anti-aliasing** — edges are soft and move continuously, with no
   pixel popping or staircase crawling.

The pipeline has two stages:

- **Stage 1 — Alignment (preprocessing).** Translate one glyph so it best
  overlaps the other, and place both on a common canvas. (Section 3.)
- **Stage 2 — Morph construction + rendering.** Precompute a per-pixel switch
  schedule once, then cheaply evaluate any frame `mask(t)`. (Sections 4–9.)

---

## 2. Data types and conventions

- A **glyph image** is a 2-D array of **alpha** (coverage) values in `[0, 1]`.
  `0` = background, `1` = fully inked. Fractional values are the font
  renderer's own anti-aliasing along edges. Colour is irrelevant; only alpha is
  used.
- All arrays are indexed `[row, col]` = `[y, x]`. "Shape" means `(height,
  width)`.
- The two morph inputs, `alpha_a` and `alpha_b`, **must have identical shape**
  (guarantee this via Stage 1).
- Output of a frame is a **coverage mask** in `[0, 1]`, same shape, to be drawn
  as a single-colour image (alpha = mask value).

---

## 3. Stage 1 — Alignment

Purpose: choose an integer pixel translation of glyph B so that it overlaps
glyph A as much as possible, then lay both onto a shared canvas.

### 3.1 Best displacement (maximum cross-correlation)

Treat each alpha image as a scalar signal. Compute the full 2-D
**cross-correlation** of A with B:

```
score[lag_y, lag_x] = Σ_{y,x} A[y, x] · B[y − lag_y, x − lag_x]
```

over all integer lags where the supports overlap. This equals the total
overlapping ink when B is shifted by `(lag_x, lag_y)`. (Efficiently computed via
FFT as a convolution of A with a point-reflected B, but a direct sum is
equivalent.)

Choose the displacement `(dx, dy)` = the lag that **maximises** `score`.

Use the **raw, unnormalised** correlation sum (total overlapping ink), *not* a
normalised correlation (NCC). Normalisation tends to slide one glyph nearly off
the other to line up a single strongly-correlated sliver; the raw sum keeps the
bulk of the ink overlapping, which matters for glyphs with repeated strokes
(e.g. whole words). `dx` is the column (horizontal, positive = right) shift,
`dy` the row (vertical, positive = down) shift applied to B.

### 3.2 Place on a common canvas

Create a canvas large enough to hold both rectangles: A at its origin and B
offset by `(dx, dy)`, with a global offset so the union has a non-negative
origin. Concretely, with A of size `(ha, wa)` and B of size `(hb, wb)`:

```
top    = min(0, dy)      left  = min(0, dx)
bottom = max(ha, dy+hb)  right = max(wa, dx+wb)
H, W   = bottom − top, right − left
```

Paste A at `(−top, −left)` and B at `(dy − top, dx − left)` into two zero-filled
`H×W` canvases. The two resulting equal-shaped alpha arrays are the inputs to
Stage 2.

> Alignment is a convenience/preprocessing step. Any method that yields two
> equal-shaped, sensibly-overlapping alpha images is acceptable; the morph
> algorithm itself only requires equal shapes.

---

## 4. Stage 2 — Region decomposition

Given the two aligned alpha images, threshold each at `τ = 0.5` to get binary
**filled regions**:

```
maskA = alpha_a ≥ τ
maskB = alpha_b ≥ τ
```

Partition the plane into three regions of interest:

- **Intersection** `inter = maskA AND maskB` — pixels filled in *both* glyphs.
  This is the shared core; it stays solid through the whole morph.
- **A-only** `a_only = maskA AND NOT maskB` — must *disappear* as `t: 0 → 1`.
- **B-only** `b_only = maskB AND NOT maskA` — must *appear* as `t: 0 → 1`.

Pixels filled in neither glyph remain background (with one fringe exception, see
§8.3).

---

## 5. Depth field

On each exclusive region we build a scalar **depth field** `D ∈ (0, 1)` that
encodes *the order in which pixels switch*. Low depth switches early in the
sweep, high depth switches late. Two field types are offered.

The two regions use opposite orientations (handled later in §7), but each field
is defined the same way: `0` at the boundary it grows/erodes *from*, `1` at the
boundary it ends *at*.

### 5.1 Geodesic field (default — linear along a stroke)

For a region `R` with a "core" boundary `C` (the intersection region), compute
the **geodesic distance** of every pixel of `R` from `C`, measured *through* `R`
only (the distance of the shortest in-region path).

- Use a Dijkstra shortest-path / fast-marching style propagation on the pixel
  grid.
- **8-connected** neighbours with Euclidean step weights: orthogonal steps cost
  `1`, diagonal steps cost `√2`. (This keeps the distance approximately
  isotropic.)
- **Seeds** (distance `0`): the pixels of `R` that are adjacent to the core
  `C` (i.e. `R ∩ dilate(C)`). Propagate distances outward through `R`; pixels
  unreachable through `R` get `+∞`.

Then **normalise per connected component**. Label `R`'s connected components.
For each component `c` with finite max distance `dmax_c`:

```
D = (d + 0.5) / (dmax_c + 1)        for reachable pixels
D = 0.5                             for unreachable pixels (component not
                                    touching the core)
```

Notes on the `+0.5 / +1` offset and the unreachable case:

- It places the seed (`d = 0`) and the tip (`d = dmax_c`) **strictly inside**
  `(0, 1)` rather than exactly at the ends. Consequently **no pixel switches
  exactly at `t = 0` or `t = 1`**, avoiding a visible jump between a held end
  frame and the first moving frame. (Exact endpoints of the rendered mask are
  still guaranteed separately — see §9.)
- A single-pixel component (`dmax_c = 0`) maps to `D = 0.5`, i.e. it flips at the
  *middle* of the morph. This is intentional: lone specks left by thresholding
  coincident outlines belong in the middle, not at an end.
- `D` is **affine in `d`**, so along an extended segment (e.g. the stem of a
  `d`) depth grows linearly with arc length — the front advances at constant
  speed when `t` is swept linearly. This is the key reason geodesic is the
  default.

### 5.2 Harmonic field (alternative — smooth)

Solve **Laplace's equation** `∇²u = 0` on the exclusive region with Dirichlet
boundary data:

- The exclusive-region pixels are the **unknowns**.
- Held at `1`: every pixel **not in the glyph's own filled region** (i.e.
  outside `maskA` for the A-only field; outside `maskB` for the B-only field) —
  this is the outer outline the front travels *to*.
- Held at `0`: all other non-unknown pixels (effectively the shared core side).
- **Image borders**: no-flux (Neumann) — a missing neighbour simply doesn't
  contribute to the stencil.

Discretisation (standard 5-point Laplacian). For each unknown pixel `p`, with
`N(p)` its in-bounds 4-neighbours:

```
|N(p)| · u[p] − Σ_{q ∈ N(p), q unknown} u[q] = Σ_{q ∈ N(p), q fixed at 1} 1
```

i.e. the diagonal is the count of in-bounds neighbours, each unknown neighbour
contributes `−1` off-diagonal, and each neighbour fixed at `1` contributes `+1`
to the right-hand side (neighbours fixed at `0` contribute nothing). Assemble
the sparse linear system and solve it. By the maximum principle the solution is
strictly in `(0, 1)`, so `u` is used directly as the depth field `D`.

Caveat: across a **thin stroke**, the `u = 1` condition on both nearby long
sides makes `u` saturate across the width, so unlike the geodesic field the
harmonic field is **not** linear along a stem. This is corrected by §6.

### 5.3 Choosing the field

`geodesic` is the default and needs no area correction. `harmonic` is smoother
but should be paired with the area equalisation of §6, otherwise components
appear/vanish suddenly at one end and crawl at the other.

---

## 6. Area equalisation (optional; recommended for harmonic)

Reparametrise a depth field per component so that **visible area changes at a
constant rate** as `t` is swept linearly.

For each connected component of the exclusive region (with `m` pixels):

1. Rank the component's pixels by depth value (stable sort; ties broken
   arbitrarily but consistently). Let `rank[p] ∈ {0, …, m−1}` be the number of
   component pixels with strictly smaller depth.
2. Replace each pixel's depth by its **empirical CDF**:

   ```
   D'[p] = (rank[p] + 0.5) / m
   ```

This preserves the **switching order** (same pixels switch in the same
sequence) but rescales the *timing* so the values are uniform on `(0, 1)`. Then
the count of pixels with `D' ≤ x` is exactly `x·m`, so the component's area
grows/shrinks linearly in the sweep. The `+0.5 / m` offset again keeps every
value strictly inside `(0, 1)` (no pixel switches at `t = 0` or `1`).

Apply to both exclusive regions' fields. For the geodesic field this is normally
**skipped** (it is already built to be linear along a stroke); for the harmonic
field it is normally **applied**.

---

## 7. Per-pixel switch schedule

Each rendered frame is, per pixel, a blend `(1 − s)·alpha_a + s·alpha_b`, where
the **switch fraction** `s(x, t)` sweeps `0 → 1` as the front crosses the pixel.
We precompute two fields that define `s`:

- a **switch centre** `Ts(x)` — the value of `t` at which the pixel is
  half-switched, and
- a **switch width** `W(x)` — the width, *in `t`*, of the soft transition.

Then `s` is a clamped linear ramp of `(t − Ts) / W + 0.5`.

### 7.1 Switch centre `Ts`

```
Ts = 0.5                       everywhere by default (covers the intersection)
Ts = 1 − D_a                   on A-only   (depth field of the A-only region)
Ts = D_b                       on B-only   (depth field of the B-only region)
```

Interpretation:

- **A-only** erodes from the outline inward: pixels near the outline
  (`D_a ≈ 1 → Ts ≈ 0`) switch *off* first; pixels near the core
  (`D_a ≈ 0 → Ts ≈ 1`) switch off last.
- **B-only** grows from the core outward: pixels near the core
  (`D_b ≈ 0 → Ts ≈ 0`) switch *on* first; pixels near the outline last.
- **Intersection** has `Ts = 0.5`, `W = 1` — it just cross-fades
  `alpha_a → alpha_b`, which is invisible since both are ~opaque there. It is
  effectively solid throughout.

### 7.2 Switch width `W` and the level-set spacing

The width is chosen so that the moving front has a **fixed spatial thickness of
~1 pixel**, regardless of how steep the depth field is locally. The conversion
factor is the local **spacing of the level sets** of `Ts`.

Define, on a region `R`, the level-set spacing of a field `f` at each pixel as
the **mean absolute difference of `f` to its in-region 4-neighbours**:

```
spacing[p] = mean over in-region 4-neighbours q of |f[p] − f[q]|
```

(Computed only from neighbours that are also in `R`, so the discontinuity at
`R`'s border — where `f` jumps to its out-of-region value — never leaks in.
Pixels with no in-region neighbour get spacing `0`.)

On a linear ramp this is exactly the per-pixel slope of `f`: how much `f` changes
between adjacent pixels. Its reciprocal is how many pixels the `f = c` level set
moves when `c` changes by one — i.e. it converts a *t*-width into a *pixel*-width.

Compute the spacing of `Ts` over the A-only region and over the B-only region
and sum them (each pixel is in at most one, so this just unions the two). Then:

```
W = 1                          everywhere by default
W[excl] = spacing[excl] · aa_width      on the exclusive regions
```

where `excl = a_only OR b_only` and `aa_width` (default `1.0`) is the desired
front thickness in pixels. Larger `aa_width` → blurrier front; smaller →
sharper, toward a hard binary edge.

**Speck handling.** A pixel with spacing `0` (an isolated single-pixel piece
with no in-region neighbour) has no spatial front, so a width derived from it
would be `0` and the pixel would hard-pop on/off in a single frame. Replace any
non-positive width on the exclusive regions with a small constant
`W_SPECK = 0.08` so such specks **fade over a short `t`-window** instead. (These
specks are the sub-pixel leftovers of thresholding coincident outlines; softened,
they disappear cleanly.)

---

## 8. Anti-aliasing details

Two kinds of edges appear in any frame, and each is anti-aliased differently.

### 8.1 Static outline (glyph-meets-background)

Where a glyph meets the background and is *not* moving, reuse the glyph's **own
already-anti-aliased alpha**. This is automatic: the per-pixel blend
`(1 − s)·alpha_a + s·alpha_b` carries the fractional alpha of whichever glyph is
currently shown, so static outlines keep the font renderer's anti-aliasing for
free.

### 8.2 Moving front

The moving front is the level set `Ts = t` sweeping across an exclusive region.
The soft ramp from §7 gives it a fixed ~1px-wide soft edge: `(t − Ts)` divided by
the local level-set spacing is the signed distance, *in pixels*, from the pixel
centre to the front, which maps straight to a coverage fraction. Because this is
continuous, the front advances **sub-pixel** as `t` changes — smooth, with no
crawling or popping — while the shared interior stays fully solid.

### 8.3 Sub-threshold fringe inheritance

A glyph's outline fringe (alpha strictly between `0` and `0.5`) lies in *neither*
filled region, so it has no schedule of its own and would otherwise switch with
the global `t` (lighting up as a faint line along, e.g., a `d`'s stem mid-morph).

Fix: give every non-filled pixel the schedule (`Ts`, `W`) of its **nearest
filled pixel**. Compute, over the complement of `filled = maskA OR maskB`, the
indices of the nearest filled pixel (a Euclidean distance transform that returns
nearest-feature indices), and gather `Ts` and `W` from those indices. Now each
fringe pixel switches together with the outline it borders.

---

## 9. Rendering a frame `mask(t)` and exact endpoints

We want `mask(0)` to reproduce `alpha_a` **exactly** and `mask(1)` to reproduce
`alpha_b` **exactly**, down to the last fringe pixel. Because every `Ts` is
strictly inside `(0, 1)` (§5–6), a raw clamped ramp would leave the first/last
pixels to switch *slightly* off at the endpoints. Correct this by normalising
each pixel's ramp by its own endpoint values.

Precompute (once):

```
s0   = clamp( (0 − Ts) / W + 0.5 , 0, 1)      # raw s at t = 0
s1   = clamp( (1 − Ts) / W + 0.5 , 0, 1)      # raw s at t = 1
sden = max(s1 − s0, ε)                         # ε small, e.g. 1e-6
```

Then each frame:

```
s_raw = clamp( (t − Ts) / W + 0.5 , 0, 1)
s     = clamp( (s_raw − s0) / sden , 0, 1)
mask  = (1 − s) · alpha_a + s · alpha_b
```

This remaps each pixel's switch fraction so it is exactly `0` at `t = 0` and
exactly `1` at `t = 1`, guaranteeing exact endpoints while keeping the smooth
interior sweep.

`mask(t)` is `O(number of pixels)` and uses only the precomputed fields, so
animation is cheap.

---

## 10. Easing and looping (presentation)

`t` may be eased before calling `mask`. A convenient choice is **smootherstep**
(zero slope at both ends):

```
ease(s) = s³ · (s · (s · 6 − 15) + 10)      for s ∈ [0, 1]
```

A seamless back-and-forth loop is, e.g.: hold on A for some frames, sweep
`t: 0 → 1`, hold on B, sweep `t: 1 → 0`, applying `ease` to each swept value.

---

## 11. End-to-end summary

**Precompute (once per glyph pair):**

1. Align B onto A by maximum raw cross-correlation; place both on a common
   `H×W` canvas → `alpha_a`, `alpha_b`. *(§3)*
2. Threshold at `0.5` → `inter`, `a_only`, `b_only`. *(§4)*
3. Build depth fields `D_a`, `D_b` on the two exclusive regions
   (geodesic distance from the core, or harmonic). *(§5)*
4. If using harmonic, equalise area per component on each field. *(§6)*
5. Form `Ts` (`0.5` baseline; `1 − D_a` on A-only; `D_b` on B-only). *(§7.1)*
6. Form `W` from the level-set spacing of `Ts` × `aa_width`; replace zero-width
   specks with `W_SPECK`. *(§7.2)*
7. Propagate `Ts`, `W` into the sub-threshold fringe by nearest-filled-pixel.
   *(§8.3)*
8. Precompute endpoint-normalisation constants `s0`, `sden`. *(§9)*

**Per frame:**

9. Optionally ease `t`. *(§10)*
10. Compute `s` from the ramp and endpoint normalisation; output
    `mask = (1 − s)·alpha_a + s·alpha_b`. *(§9)*

**Guarantees:** `mask(0) = alpha_a` and `mask(1) = alpha_b` exactly; the front
moves at constant speed (geodesic) and/or constant area rate (equalised); only a
~1px band is ever a partial mix of the two glyphs (no ghosting); edges are
anti-aliased and move sub-pixel (no popping or crawling).

---

## 12. Parameters

| Parameter   | Default      | Meaning |
|-------------|--------------|---------|
| `threshold` | `0.5`        | Alpha cutoff separating filled from background. |
| `field`     | `"geodesic"` | Depth field type: `"geodesic"` or `"harmonic"`. |
| `equalize`  | `true`       | Area-equalise the field (applies to harmonic; ignored for geodesic). |
| `aa_width`  | `1.0`        | Soft-edge width of the moving front, in pixels. |
| `W_SPECK`   | `0.08`       | `t`-window over which isolated single-pixel specks fade. |
| `ε`         | `1e-6`       | Floor on the endpoint-normalisation denominator. |
