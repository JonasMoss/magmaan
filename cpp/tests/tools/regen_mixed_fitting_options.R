#!/usr/bin/env Rscript
# Public HS summary fixtures; fitting outputs come from installed lavaan.
stopifnot(as.character(utils::packageVersion("lavaan")) == "0.7.2")
stopifnot(gsub("-", ".", trimws(readLines("cpp/tests/fixtures/lavaan_version.txt"))) == "0.7.2")
.fitting_attempts <- list()
trace("lav_model_est", where = asNamespace("lavaan"), print = FALSE, exit = quote({
  out <- returnValue()
  .GlobalEnv$.fitting_attempts[[length(.GlobalEnv$.fitting_attempts) + 1L]] <- list(
    simple = start == "simple", standardized = lavoptions$optim.parscale != "none",
    start = as.numeric(attr(out, "start")), parameter_scale = as.numeric(attr(out, "parscale")),
    port_scale = as.numeric(scale_1), gradient = as.numeric(attr(out, "dx")),
    theta = as.numeric(out), accepted = isTRUE(attr(out, "converged")),
    iterations = attr(out, "iterations"))
}))
d <- lavaan::HolzingerSwineford1939
ord <- paste0("x", 1:3)
for (v in ord) d[[v]] <- ordered(cut(d[[v]], quantile(d[[v]], c(0, 1/3, 2/3, 1)),
                                   include.lowest = TRUE, labels = FALSE))
cases <- list()
for (p in c("delta", "theta")) for (g in 1:2) for (restricted in c(FALSE, TRUE)) {
  s <- "f =~ x1+x2+x3\ng =~ x4+x5+x6"
  if (g == 1 && restricted) s <- paste(s, "f ~~ 0*g", sep = "\n")
  eq <- if (g == 2 && restricted) "loadings" else NULL
  .fitting_attempts <- list()
  # Estimate-only fitting excludes the auxiliary baseline search from the trace.
  lv <- lavaan::cfa(s, d, ordered = ord, estimator = "WLSMV", parameterization = p,
    group = if (g == 2) "school" else NULL, group.label = if (g == 2) levels(d$school) else NULL,
    group.equal = eq, se = "none", test = "none")
  pt <- lavaan::parTable(lv)
  cases[[length(cases) + 1L]] <- list(parameterization = p, groups = g,
    group_equal = eq, model = s,
    partable = pt[, c("id", "lhs", "op", "rhs", "user", "block", "group", "free", "exo", "ustart", "label", "plabel")],
    parameters = pt[pt$free > 0, c("lhs", "op", "rhs", "group", "start", "est")],
    R = lapply(lv@SampleStats@cov, unname), mean = lapply(lv@SampleStats@mean, as.numeric),
    # lavaan's th slot appends continuous means after the six thresholds.
    thresholds = lapply(lv@SampleStats@th, function(x) as.numeric(head(x, 2L * length(ord)))),
    moments = lapply(lv@SampleStats@WLS.obs, as.numeric),
    weight = lapply(lv@SampleStats@WLS.VD, as.numeric), n_obs = as.integer(unlist(lv@SampleStats@nobs)),
    nacov = lapply(lv@SampleStats@NACOV, unname), explicit_start = numeric(),
    fmin = as.numeric(lv@optim$fx), converged = isTRUE(lv@optim$converged),
    basis = unname(lv@Model@eq.constraints.K), offset = as.numeric(lv@Model@eq.constraints.k0),
    attempts = .fitting_attempts)
}
untrace("lav_model_est", where = asNamespace("lavaan"))
jsonlite::write_json(list(source = "regen_mixed_fitting_options.R; public HS pooled-tertile mixed two-factor CFA",
  lavaan_version = "0.7.2", cases = cases), "cpp/tests/fixtures/fitting/lavaan_mixed_0_7_2.json",
  auto_unbox = TRUE, digits = 17, pretty = TRUE, na = "null", null = "null")
