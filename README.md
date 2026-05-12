# latva

A C++23 toolkit for **linear structural equation modeling** under
normal-theory ML. Built to match
[lavaan](https://lavaan.ugent.be/) numerically while exposing every
extension seam (estimator, optimizer, SE method, fit index) as
first-class API for methods developers.

> **Status:** pre-alpha. The skeleton builds; the parser is being written.

## What it is and isn't

- **Is:** a C++ core for SEM tool builders — parser, partable, ML fit, SEs.
  No exceptions, no RTTI, `std::expected` everywhere fallible.
- **Isn't:** an end-user tool. There is no `cfa(model, data)` facade in C++.
  R / Python bindings are out of scope for v0.

## Building

Requirements: CMake ≥ 3.28, GCC ≥ 13 / Clang ≥ 17 / MSVC ≥ 19.37.

```sh
cmake --preset asan
cmake --build --preset asan
ctest --preset asan
```

Other presets: `default` (Debug), `release`, `ubsan`.

## Layout

See [`AGENTS.md`](AGENTS.md). The design rationale is in the planning
document referenced from there.

## License

TBD.
