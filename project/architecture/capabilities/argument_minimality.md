### Argument-minimality sweep

One core architecture rule is that functions should take exactly the data they
need, and no more. The current C++ core mostly follows this:

- Fit results are plain `Estimates` and carry no back-pointer to the model.
- Continuous fitters take the estimable structure, matrix representation,
  sample/raw data, starts, discrepancy, and optimizer. They do not take
  parameter names.
- Information, covariance, SE, chi-square, degrees of freedom, Wald/z tests,
  robust U-Gamma reducers, both casewise-Zc and caller-supplied-Gamma robust
  test reductions, both-bread robust test trace-moment reducers, the
  materialized empirical-Gamma reference reducer, and fit measures are exposed
  as separate primitives rather than as methods on one fitted-object bundle.
- Satorra-Bentler-family reducers take scalar test statistics, degrees of
  freedom, and eigenvalues or trace summaries rather than fit objects.
- The Satorra-2000 C++ helper takes the model, reparameterization matrices,
  raw-data pieces, chi-square values, degrees of freedom, and an explicit
  `A.method` option (`exact` default, `delta` for lavaan-style
  moment-Jacobian column-space restrictions). Its empirical-Gamma computation
  can run in the default streaming casewise-reduced mode, in a materialized
  full-Gamma reference mode, or in a dense full-`UΓ` diagnostic mode for
  benchmarks. The same computation flag is honored by the FIML lavaan-
  convention restriction-map path: streaming projects through the lavaan-style
  expected `WLS.V` columns at the fitted larger model before contracting the
  retained saturated EM ACOV, without materializing another full Gamma.
  Materialized builds each `n_g * acov_g` block and dense eigendecomposes the
  full moment-space product. The numeric kernels fail closed on non-finite or
  singular pencils:
  model-implied covariance blocks, the pooled expected-information `P`, and
  restriction companion `C` are finite/SPD-gated; reduced meats are
  finite/PSD-gated; the FIML sandwich route allows finite nonsingular observed
  bread that is indefinite in finite samples, but still rejects singular bread
  and degenerate `C` before any generalized eigensolver is called. Continuous
  pairwise expected-bread U-factors also avoid materializing `Gamma_NT^pw`: the
  raw/pairwise `build_u_factor` overload builds the residual subspace from the
  model Jacobian, applies the pattern-grouped pairwise NT metric as an operator,
  and whitens only the reduced `df x df` Gram matrix. Observed-bread pairwise
  inference still materializes the pairwise NT metric because that
  representation stores `A = L_Gamma^{-1} Delta`.

The remaining pressure point is the R boundary. Several exported R wrappers
currently accept the transparent fit list for convenience and then unpack the
partable, sample statistics, and estimates internally. That is acceptable as an
interactive convenience while the R package is exploratory, but it is not the
final architectural ideal. Thin R wrappers should gradually mirror the C++
primitive signatures; fit-list helpers can remain as explicit convenience
adapters layered on top.
