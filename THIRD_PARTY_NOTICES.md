# Third-party notices

magmaan itself is released under the MIT License (see [LICENSE](LICENSE)).

This repository additionally vendors third-party sources under `third_party/`.
They are committed verbatim rather than fetched at build time, and each retains
its own license. Nothing here restricts magmaan's own MIT terms; this file
records what the vendored code is and under what terms it ships.

## `third_party/port/` — PORT optimizer routines

f2c-translated C sources for the Bell Labs PORT routines that back R's
`nlminb` (TOMS 611) and R's `nls` (NL2SOL with bounds, TOMS 573).

- **License:** BSD 3-Clause.
- **Copyright:** AMPL Optimization, Inc. (2017-2023) and the Fermi-LAT
  Collaboration (2019).
- **Texts:** [`third_party/port/LICENSE-AMPL`](third_party/port/LICENSE-AMPL),
  [`third_party/port/LICENSE-fermi-lat`](third_party/port/LICENSE-fermi-lat).
- **Provenance and per-file manifest:**
  [`third_party/port/README.md`](third_party/port/README.md).

## `third_party/quadpack/` — QUADPACK `qagi` adaptive integrator

f2c-translated C sources for the QUADPACK Gauss-Kronrod routines used to
evaluate Imhof's integral, the kernel behind the Satorra-Bentler, pEBA, and
robust nested tests.

- **License:** public domain. The QUADPACK routines (`dqagie`, `dqk15i`,
  `dqelg`, `dqpsrt`, and the verbatim Fortran under `upstream/`) carry no
  copyright or license conditions and have been distributed without
  restriction via netlib and SLATEC for decades.
- **Attribution:** R. Piessens, E. de Doncker-Kapenga, C. W. Überhuber, and
  D. K. Kahaner, *QUADPACK: A Subroutine Package for Automatic Integration*,
  Springer-Verlag, 1983.
- **`f2c.h`:** part of the f2c distribution by AT&T Bell Laboratories
  (S. I. Feldman et al.), freely redistributable. See
  <https://www.netlib.org/f2c/>.
- **Locally written:** `d1mach.c` and `pow_dd.c` were written for magmaan and
  are covered by magmaan's MIT license.
- **Full text and provenance:**
  [`third_party/quadpack/LICENSE`](third_party/quadpack/LICENSE),
  [`third_party/quadpack/README.md`](third_party/quadpack/README.md).
