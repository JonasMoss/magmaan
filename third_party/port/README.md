# third_party/port — Vendored PORT (Bell Labs) optimizer routines

This directory holds the f2c-translated C source for the Bell Labs PORT
optimizer routines that back R's `nlminb` (general unconstrained-or-bounded
minimization, TOMS 611) and R's `nls` (NL2SOL with bounds, TOMS 573). The
routines are committed into the repository rather than fetched at build time
because the upstream sources are stable (no upstream changes in decades) and
because vendoring keeps the build self-contained and reproducible.

This vendoring strategy is documented as the magmaan "Phase 1" optimizer
roadmap; see `docs/architecture/roadmap.md`.

## Vendor manifest

| File / directory | Upstream | Upstream commit / tag | License | Purpose |
|---|---|---|---|---|
| `cport/*.c`, `cport/f2c.h` (75 files) | [`ampl/asl`](https://github.com/ampl/asl) at `src/examples/PORT/cport/` | master @ 2025-11-23 | BSD-3-Clause (see `LICENSE-AMPL`) | NL2SOL + SUMSL/HUMSL bounded explicit-gradient cores, helper library, `f2c.h` |
| `drmnfb_routines.c` | [`fermi-lat/optimizers`](https://github.com/fermi-lat/optimizers) at `src/drmnfb_routines.c` | master @ 2026-04-14 | BSD-3-Clause-equivalent (see `LICENSE-fermi-lat`) | SUMSL bounded finite-difference variant: `drmnfb_` + `ds3grd_` |
| `port_io.c` | written here | — | project license | No-op stubs for `s_wsfe`, `e_wsfe`, `do_fio`, `do_lio`, replacing libf2c's libI77 so iteration prints silently |
| `f2c_intrinsics.c` | written here, helpers per netlib `libf2c/libF77/` | — | project license + AT&T permissive notice for the algorithmic content | `pow_dd`, `s_copy` — the subset of libF77 PORT actually calls |

## Why two upstream sources

The publicly-available subset of Netlib PORT is split functionally:

- AMPL/ASL `cport/` ships the **NL2SOL** family (`dn2gb_`, `drn2gb_`, etc.)
  and the SUMSL/HUMSL **explicit-gradient** bounded variants (`drmngb_`,
  `drmnhb_`), but **not** the finite-difference (`drmnfb_`) driver that
  computes its own gradient by central differences.
- Fermi-LAT's `optimizers/src/drmnfb_routines.c` ships exactly `drmnfb_` +
  `ds3grd_` (the finite-difference gradient helper). It is what NASA uses
  to drive the Fermi-LAT likelihood model when gradients are not available
  in closed form.

magmaan supplies analytic gradients, so the *explicit-gradient* `drmngb_` is
what `Backend::Port` calls in practice. `drmnfb_` is kept in the vendored set
for two reasons: it costs us nothing additional (one ~17 KB file, no extra
helper symbols), and it gives future code a clean fall-back when a model
without analytic gradients needs PORT-style trust-region behaviour.

The AMPL and Fermi-LAT f2c translations share helper routines
(`da7sst_`, `dl7itv_`, …) — those live exclusively in the AMPL set; the
Fermi-LAT file calls them, so we only build the AMPL versions.

## Why no libf2c

`libf2c` is the runtime f2c-translated Fortran code expects at link time. Two
parts:

- `libF77` — small (~20 KB), provides intrinsic helpers (`pow_dd`, `d_sign`,
  `s_copy`, …). We need a tiny subset (PORT uses only `pow_dd` and `s_copy`),
  inlined here as `f2c_intrinsics.c`.
- `libI77` — bulky (~500 KB compiled), provides Fortran formatted I/O
  (`s_wsfe`, `do_fio`, …). PORT uses I/O only to print iteration traces and
  parameter-check error messages. magmaan never wants these prints, so we
  stub the I/O entry points to no-ops in `port_io.c` and the PORT routines'
  output paths become silent. Iteration history is recovered post-fit from
  the IV/V arrays (PORT's structured solver-state output), not from prints.

## License chain

The PORT routines themselves originate at Bell Labs (David M. Gay, Linda
Kaufman, et al., 1970s–1990s). Netlib's `port/readme` explicitly carves out
the NL2SOL (TOMS 573) and SUMSL/HUMSL (TOMS 611) families as **publicly
available** outside the rest of PORT3, which is under a (now-defunct) Bell
Labs limited-use agreement. From the verbatim quote in
`https://netlib.org/port/readme`:

> "Also available are the up-to-date versions of NL2SOL (TOMS algorithm 573)
> … `dn[s2][fgp][b ]` are the double-precision versions … Similarly available
> are current versions of SMSNO, SUMSL, HUMSL (TOMS algorithm 611) for
> general unconstrained minimization … `dmn[fgh][b ]` are double-precision
> versions."

Every meaningful downstream — AMPL's ASL, NASA Fermi-LAT's optimizers,
Sandia's Dakota framework, R Core — treats the carve-out as effectively
public-domain code. AMPL and Fermi-LAT have explicitly relicensed their
f2c-translated packagings as BSD-3-Clause; magmaan inherits both BSD-3
packagings directly.

References:

- TOMS 573 — Dennis, Gay, Welsch (1981), "An Adaptive Nonlinear
  Least-Squares Algorithm", ACM TOMS 7(3):348–368.
- TOMS 611 — Gay (1983), "Algorithm 611: Subroutines for Unconstrained
  Minimization Using a Model/Trust-Region Approach", ACM TOMS 9(4):503–524.
- Netlib `port/readme` — <https://netlib.org/port/readme>.
- AMPL/ASL — <https://github.com/ampl/asl> (BSD-3, see `LICENSE-AMPL`).
- Fermi-LAT optimizers — <https://github.com/fermi-lat/optimizers> (BSD-3-equivalent, see `LICENSE-fermi-lat`).

## Local modifications

Tracked to keep an audit trail; the AT&T/Bell Labs copyright notices inside
each `.c` file are preserved as-is.

- `drmnfb_routines.c` — patched `#include "f2c/f2c.h"` to `#include "f2c.h"`
  (consolidated include path; we vendor `cport/f2c.h` rather than depending
  on a system libf2c).

No other modifications. Re-vendoring (from a fresh upstream pull) is a
mechanical copy-plus-this-patch.
