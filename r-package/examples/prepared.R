# Reusable preparation: parity, changed datasets, inference and invalidation.
suppressPackageStartupMessages(library(magmaan))
ctrl <- list(max_iter = 3000, ftol = 1e-12, gtol = 1e-8)
syntax <- 'f =~ x1 + x2 + x3 + x4'
make_data <- function(seed, n = 400, shift = 0) {
  set.seed(seed)
  z <- matrix(rnorm(n * 4), n, 4) + rnorm(n)
  colnames(z) <- paste0('x', 1:4)
  as.data.frame(z + shift)
}
close_fit <- function(a, b, tol = 1e-7) {
  stopifnot(max(abs(a$theta - b$theta)) < tol,
            abs(a$fmin - b$fmin) < tol,
            identical(a$converged, b$converged),
            identical(a$partable[, c('lhs', 'op', 'rhs', 'free')],
                      b$partable[, c('lhs', 'op', 'rhs', 'free')]),
            !is.null(a$audit), !is.null(a$diagnostics))
}
reject <- function(expr, pattern) {
  e <- tryCatch({ force(expr); NULL }, error = identity)
  stopifnot(inherits(e, 'error'), grepl(pattern, conditionMessage(e)))
}
m <- prepare_model(syntax)
spec <- model_spec(syntax)
for (seed in 1:2) {
  x <- make_data(seed, 400 + seed * 20, seed / 3)
  d <- prepare_data(m, x)
  for (method in c('ML', 'ULS', 'GLS')) {
    a <- estimate(m, d, method, control = ctrl)
    b <- magmaan(spec, x, estimator = method, control = ctrl)
    close_fit(a, b)
  }
  W <- diag(10)
  w <- prepare_weight(d, 'WLS', W = W)
  close_fit(estimate(m, d, weight = w, control = ctrl),
            magmaan:::fit_wls(spec, df_to_data(x, spec), W, control = ctrl))
  reject(estimate(m, prepare_data(m, x), weight = w), 'another dataset')
  a <- estimate(m, d, control = ctrl)
  stopifnot(max(abs(vcov(a, data = x) - vcov(magmaan(spec, x, control = ctrl), data = x))) < 1e-7)
}
reject({m$kind <- 'raw'}, 'locked')
reject(estimate(unserialize(serialize(m, NULL)), d), 'prepare it again')
reject(prepare_data(m, make_data(1)[, -1]), 'missing model variables')

# FIML: missingness packs change with the dataset; estimation does not compute H1.
mf <- prepare_model(syntax, meanstructure = TRUE)
for (seed in 3:4) {
  x <- make_data(seed)
  x[seq(seed, nrow(x), 7), 2] <- NA
  d <- prepare_data(mf, x, 'raw')
  a <- estimate(mf, d, control = ctrl)
  b <- magmaan:::fit_fiml(mf$spec, x, control = ctrl)
  close_fit(a, b)
  stopifnot(is.null(a$fiml_h1), !is.null(a$fiml_pack))
}

ordinal_data <- function(seed, shift = 0) {
  x <- make_data(seed, 600, shift)
  x[] <- lapply(x, function(z) ordered(cut(z, c(-Inf, -.5, .6, Inf)),
                                      levels = levels(cut(z, c(-Inf, -.5, .6, Inf)))))
  x
}
proto <- ordinal_data(20)
for (parameterization in c('delta', 'theta')) {
  mo <- prepare_model(syntax, ordered = names(proto), prototype = proto,
                      parameterization = parameterization)
  original <- model_spec(syntax, ordered = names(proto), parameterization = parameterization)
  for (seed in 21:22) {
    x <- ordinal_data(seed, (seed - 20) / 4)
    d <- prepare_data(mo, x)
    stats <- magmaan:::data_ordinal_stats_from_df(x, original)
    for (method in c('ULS', 'DWLS', 'WLS')) {
      a <- estimate(mo, d, method, control = ctrl)
      b <- switch(method, ULS = magmaan:::fit_uls_ordinal(original, stats, control = ctrl),
                  DWLS = magmaan:::fit_dwls_ordinal(original, stats, control = ctrl),
                  WLS = magmaan:::fit_wls_ordinal(original, stats, control = ctrl))
      close_fit(a, b, 1e-6)
      if (method == 'DWLS') {
        w <- prepare_weight(d, 'DWLS', full = FALSE)
        close_fit(estimate(mo, d, weight = w, control = ctrl), a, 1e-6)
        stopifnot(max(abs(magmaan_core$robust_ordinal(a, a$ordinal_stats)$vcov -
                         magmaan_core$robust_ordinal(b, b$ordinal_stats)$vcov)) < 1e-6)
      }
    }
  }
  bad <- proto; levels(bad$x1) <- rev(levels(bad$x1))
  reject(prepare_data(mo, bad), 'category levels/order')
}
# Mixed continuous/ordinal data uses the same construction.
x <- make_data(30, 600)
x$x1 <- ordinal_data(30)$x1
mm <- prepare_model(syntax, meanstructure = TRUE, ordered = 'x1', prototype = x)
dm <- prepare_data(mm, x)
for (method in c('DWLS', 'WLS')) {
  a <- estimate(mm, dm, method, control = ctrl)
  b <- magmaan(syntax, x, meanstructure = TRUE, ordered = 'x1', estimator = method, control = ctrl)
  close_fit(a, b, 1e-6)
}
cat('Prepared interface checks passed.\n')

# Precomputed moments, named-column reordering and empirical continuous weights.
x <- make_data(51, 700)
d <- prepare_data(m, x)
ss <- df_to_data(x, spec)
close_fit(estimate(m, prepare_data(m, ss), control = ctrl), estimate(m, d, control = ctrl))
close_fit(estimate(m, prepare_data(m, x[, 4:1]), control = ctrl), estimate(m, d, control = ctrl))
for (method in c('DWLS', 'WLS')) {
  w <- prepare_weight(d, method)
  stopifnot(length(w$W) == 1L, all(dim(w$W[[1]]) == c(10L, 10L)))
  close_fit(estimate(m, d, weight = w, control = ctrl),
            magmaan:::fit_wls(spec, ss, W = w$W, control = ctrl))
}
reject(prepare_weight(d, 'WLS', W = diag(-1, 10)), 'positive definite')
reject(prepare_weight(d, 'WLS', W = diag(9)), 'moment dimensions')
reject(prepare_weight(prepare_data(m, ss), 'WLS'), 'raw data or explicit W')

# Shared categorical Gamma can be requested for inference without refitting.
mo <- prepare_model(syntax, ordered = names(proto), prototype = proto)
do <- prepare_data(mo, proto)
u <- estimate(mo, do, 'ULS', control = ctrl)
wu <- prepare_weight(do, 'ULS', full = TRUE)
u_full <- estimate(mo, do, weight = wu, control = ctrl)
close_fit(u, u_full, 1e-6)
stopifnot(max(abs(magmaan_core$robust_ordinal(u, wu$stats)$vcov - vcov(u_full))) < 1e-7)

# Multi-group threshold invariance, two different datasets with one schema.
for (seed in 61:62) {
  xg <- rbind(transform(ordinal_data(seed), g = 'a'),
              transform(ordinal_data(seed + 10, .4), g = 'b'))
  if (seed == 61) {
    sg <- model_spec(syntax, ordered = names(proto), parameterization = 'theta',
                     group = 'g', group_labels = c('a', 'b'),
                     group_equal = c('thresholds', 'loadings'))
    mg <- prepare_model(sg, prototype = xg)
  }
  a <- estimate(mg, prepare_data(mg, xg), control = ctrl)
  b <- magmaan(sg, xg, estimator = 'DWLS', control = ctrl)
  close_fit(a, b, 2e-6)
}
# Prepared fitting cannot fall back to any R model/ordinal construction path.
for (fn in c('model_matrix_rep', 'augment_ordinal_partable', 'augment_mixed_ordinal_partable'))
  trace(fn, tracer = quote(stop('unexpected repeated model preparation')),
        where = asNamespace('magmaan'), print = FALSE)
tryCatch({
  close_fit(estimate(mo, do, weight = wu, control = ctrl), u_full, 1e-6)
  close_fit(estimate(m, d, control = ctrl), magmaan:::prepared_estimate_impl(
    m$native, d$native, NULL, 'ML', NULL, ctrl, NULL))
}, finally = {
  for (fn in c('model_matrix_rep', 'augment_ordinal_partable', 'augment_mixed_ordinal_partable'))
    untrace(fn, where = asNamespace('magmaan'))
})
cat('Prepared schema, weight and construction checks passed.\n')
