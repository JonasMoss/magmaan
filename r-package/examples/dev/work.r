model <- " visual  =~ x1 + x2 + x3
              textual =~ x4 + x5 + x6
              speed   =~ x7 + x8 + x9 "

data <- as.matrix(na.omit(lavaan::HolzingerSwineford1939[, 7:(7+8)]))

f <- \(model, data) {
  ss <- magmaanlab::data_sample_stats_from_raw(data)
  partable <- magmaanlab::lavaan_lavaanify(model)
  magmaanlab::fit_fit(partable, ss)
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

ss       <- magmaanlab::data_sample_stats_from_raw(data)        # N-divisor S, mean, nobs
partable <- magmaanlab::lavaan_lavaanify(model)
fit      <- magmaanlab::fit_fit(partable, ss)               # fit on the same N-divisor moments
uf       <- magmaanlab::infer_build_u_factor(fit)             # bread = "expected", moments = "structured"

Zc <- magmaanlab::infer_casewise_contributions(partable, data)   # N x p*  centred vech contributions
M  <- magmaanlab::infer_reduced_gamma_sample(uf, Zc, ss$nobs)    # df x df ;  denom = N_total
ev <- magmaanlab::infer_ugamma_eigenvalues(M)                 # ascending eigenvalues of UΓ̂

T_ml  <- magmaanlab::infer_chi2_stat(magmaanlab::fit_sample_stats(fit), fit$fmin)
df_ml <- magmaanlab::infer_df_stat(fit$partable, magmaanlab::fit_sample_stats(fit))
magmaanlab::infer_satorra_bentler(T_ml, df_ml, ev)            # -> list(chi2_scaled, scale_c, df)
