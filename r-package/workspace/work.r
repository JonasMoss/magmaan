model <- " visual  =~ x1 + x2 + x3
              textual =~ x4 + x5 + x6
              speed   =~ x7 + x8 + x9 "

data <- as.matrix(na.omit(lavaan::HolzingerSwineford1939[, 7:(7+8)]))

f <- \(model, data) {
  ss <- magmaan::data_sample_stats_from_raw(data)
  partable <- magmaan::lavaan_lavaanify(model)
  magmaan::fit_fit(partable, ss)
}

g <- \(model, data) lavaan::cfa(model,data = data, estimator = "ML")

f(model, data)
g(model, data)

microbenchmark::microbenchmark(f(model, data), g(model, data))


mod <- lavaan::cfa(model, data = data, estimator = "ML")


model <- " visual  =~ x1 + x2 + x3
              textual =~ x4 + x5 + x6
              speed   =~ x7 + x8 + x9 "

data <- as.matrix(na.omit(lavaan::HolzingerSwineford1939[, 7:(7+8)]))

ss       <- magmaan::data_sample_stats_from_raw(data)        # N-divisor S, mean, nobs
partable <- magmaan::lavaan_lavaanify(model)
fit      <- magmaan::fit_fit(partable, ss)               # fit on the same N-divisor moments
uf       <- magmaan::infer_build_u_factor(fit)             # bread = "expected", moments = "structured"

Zc <- magmaan::infer_casewise_contributions(partable, data)   # N x p*  centred vech contributions
M  <- magmaan::infer_reduced_gamma_sample(uf, Zc, ss$nobs)    # df x df ;  denom = N_total
ev <- magmaan::infer_ugamma_eigenvalues(M)                 # ascending eigenvalues of UΓ̂

T_ml  <- magmaan::infer_chi2_stat(magmaan::fit_sample_stats(fit), fit$fmin)
df_ml <- magmaan::infer_df_stat(fit$partable, magmaan::fit_sample_stats(fit))
magmaan::infer_satorra_bentler(T_ml, df_ml, ev)            # -> list(chi2_scaled, scale_c, df)
