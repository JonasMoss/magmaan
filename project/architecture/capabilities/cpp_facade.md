### Staged C++ facade

- `magmaan::api` provides a tested staged facade over the currently
  implemented core primitives. Model construction, data construction, fitting,
  standard SEs, robust reporting, tests, modification indices, score tests,
  Wald/z tests, standardization, defined parameters, fit measures, and nested
  tests remain explicit calls rather than a lavaan-style one-shot summary.
- The facade supports complete-data ML, raw continuous FIML, continuous
  ULS/GLS/explicit-weight WLS, and ordinal/mixed DWLS/WLS through the same
  underlying fit and post-fit primitives described above. Unsupported
  combinations fail with `api::ErrorStage::UnsupportedCombination`.
