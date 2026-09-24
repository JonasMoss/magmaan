# third_party/quadpack — Vendored QUADPACK `qagi` adaptive integrator

This directory holds the f2c-translated C source for the QUADPACK routines that
perform adaptive Gauss-Kronrod integration over a semi-infinite interval
(`qagi`). magmaan uses exactly one entry point, `dqagie_`, inside
`robust::weighted_chisq::imhof_upper` to evaluate Imhof's (1961) integral for the
tail probability of a weighted sum of chi-squares — the kernel of every
Satorra-Bentler / pEBA / pOLS p-value and the robust nested tests.

QUADPACK is **public domain** (Piessens, de Doncker-Kapenga, Überhuber, Kahaner;
SLATEC / netlib), so it is used directly rather than reimplemented. R's own
`integrate()` (`Rdqagi`) and `CompQuadForm::imhof` use the same routine; matching
it gives near-exact agreement with both. The sources are committed (not fetched
at build time) because upstream has been stable for decades and vendoring keeps
the build self-contained.

## Vendor manifest

| File | Upstream | Source | License | Purpose |
|---|---|---|---|---|
| `dqagie.c` | netlib `quadpack/dqagie.f` | <https://www.netlib.org/quadpack/> @ 2026-05-30 | Public domain | Adaptive driver over `[0,inf)`: bisect-worst-panel + epsilon extrapolation |
| `dqk15i.c` | netlib `quadpack/dqk15i.f` | same | Public domain | 15-point Gauss-Kronrod rule for infinite intervals (does the `x=(1-t)/t` transform internally) |
| `dqelg.c` | netlib `quadpack/dqelg.f` | same | Public domain | Epsilon (Wynn) extrapolation table |
| `dqpsrt.c` | netlib `quadpack/dqpsrt.f` | same | Public domain | Maintains the error-ordered subdivision list |
| `d1mach.c` | written here | netlib `d1mach` semantics, via `<float.h>` | project license | IEEE double machine constants; dependency-free replacement for netlib's bit-table `d1mach.f` (no COMMON block) |
| `pow_dd.c` | written here | libf2c semantics | project license | The one `x**y` helper f2c emits for `dqk15i`; libf2c is not linked |
| `f2c.h` | f2c distribution | copied from `third_party/port/cport/f2c.h` | freely distributable (f2c) | f2c runtime types/macros (`integer`=`int`, `doublereal`=`double`) |
| `upstream/*.f` | netlib `quadpack/` | same | Public domain | Verbatim Fortran sources the `.c` were translated from |

## Local changes

- **Translation.** `f2c -A` (ANSI prototypes), f2c version 20240504 — the same
  translator used for `third_party/port`. The `.c` are otherwise verbatim f2c
  output.
- **`dqagi` dropped.** The simple wrapper `dqagi` only sizes the work arrays and
  calls `dqagie`, then invokes Fortran `xerror` on abnormal return. We call
  `dqagie` directly from C++ (sizing the work arrays there), which avoids the
  `xerror` / libI77 dependency entirely. `upstream/dqagi.f` is kept for
  reference; it is not built.
- **`d1mach` rewritten.** Replaced netlib's bit-table `d1mach.f` (COMMON block,
  platform-conditional DATA statements) with a tiny `<float.h>` version. The
  netlib `d1mach.f` is intentionally not vendored.
- **Symbol isolation.** `dqagie`/`dqk15i`/`dqelg`/`dqpsrt` are unique to
  QUADPACK, but `d1mach_` and `pow_dd` also exist in `third_party/port`. The
  QUADPACK target renames them to magmaan-private symbols at compile time
  (`-Dd1mach_=magmaan_qpk_d1mach_ -Dpow_dd=magmaan_qpk_pow_dd`, see
  `cmake/QuadpackVendor.cmake`) so the two vendored libraries never collide.

## Re-vendoring

```sh
# fetch the cone (d1mach 404s under quadpack/ — we hand-write it)
for f in dqagie dqk15i dqelg dqpsrt; do
  curl -fsSL "https://www.netlib.org/quadpack/$f.f" -o upstream/$f.f
done
# translate
for f in dqagie dqk15i dqelg dqpsrt; do f2c -A upstream/$f.f && mv $f.c .; done
```

## References

- R. Piessens, E. de Doncker-Kapenga, C. W. Überhuber, D. K. Kahaner.
  *QUADPACK: A Subroutine Package for Automatic Integration.* Springer, 1983.
- J. P. Imhof. *Computing the distribution of quadratic forms in normal
  variables.* Biometrika 48(3/4), 419-426, 1961.
