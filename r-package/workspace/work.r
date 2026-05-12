model <- " visual  =~ x1 + x2 + x3
              textual =~ x4 + x5 + x6
              speed   =~ x7 + x8 + x9 "

data <- as.matrix(na.omit(lavaan::HolzingerSwineford1939[, 7:(7+8)]))

f <- \(model, data) {
  n <- nrow(data)
  S <- cov(data) * (n-1) / n

  partable <- latva::latva_lavaanify(hs_model)
  latva::latva_fit(partable, S, n)
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

ss       <- latva::latva_sample_stats_from_raw(data)        # N-divisor S, mean, nobs
partable <- latva::latva_lavaanify(model)
fit      <- latva::latva_fit(partable, ss$S, ss$nobs)    # fit on the same N-divisor moments
uf       <- latva::latva_build_u_factor(fit)             # bread = "expected", moments = "structured"

Zc <- latva::latva_casewise_contributions(fit, data)        # N x p*  centred vech contributions
M  <- latva::latva_reduced_gamma_sample(uf, Zc, nrow(Zc))   # df x df ;  denom = N_total
ev <- latva::latva_ugamma_eigenvalues(M)                 # ascending eigenvalues of UΓ̂

se <- latva::latva_se_expected(fit)                      # supplies T_ML and df
latva::latva_satorra_bentler(se$chi2, se$df, ev)         # -> list(chi2_scaled, scale_c, df)
