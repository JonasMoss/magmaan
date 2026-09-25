# Coverage map: inputs the ordinary route accepts but the sphere route may
# not. The requirement is a clear error, never a silent ordinary fit that
# claims to be a sphere fit.

coverage_items <- function(data) {
  hs <- data$hs
  hs_ord <- hs
  for (v in c("x1", "x2", "x3")) hs_ord[[v]] <- as.integer(cut(hs[[v]], 4))
  m1 <- "visual =~ x1 + x2 + x3"
  two <- lavaan::Demo.twolevel
  m2 <- "level: 1\nfw =~ y1 + y2 + y3\nlevel: 2\nfb =~ y1 + y2 + y3"
  list(
    list(feature = "Ordinal indicators (DWLS)",
         ordinary = function() fit_model(m1, hs_ord, ordered = c("x1", "x2", "x3")),
         sphere = function() frontier_fit_sphere(m1, hs_ord, ordered = c("x1", "x2", "x3"))),
    list(feature = "Two-stage ML (ML2S)",
         ordinary = function() fit_model(hs3, data$hs_miss, estimator = "ML2S"),
         sphere = function() frontier_fit_sphere(hs3, data$hs_miss, estimator = "ML2S")),
    list(feature = "Two-level model",
         ordinary = function() fit_model(m2, two, cluster = "cluster"),
         sphere = function() frontier_fit_sphere(m2, two, cluster = "cluster")),
    list(feature = "PSD-constrained ULS",
         ordinary = function() frontier_fit_uls_psd(hs3, hs),
         sphere = function() frontier_fit_sphere(hs3, hs, estimator = "ULS", psd = TRUE)),
    list(feature = "Composite (<~)",
         ordinary = function() fit_model("C <~ x1 + x2 + x3\nvisual =~ x4 + x5 + x6\nvisual ~ C", hs),
         sphere = function() frontier_fit_sphere("C <~ x1 + x2 + x3\nvisual =~ x4 + x5 + x6\nvisual ~ C", hs)),
    list(feature = "Inequality constraint",
         ordinary = function() fit_model(paste0(hs3, "\n", "visual ~~ v*visual\nv > 0.1"), hs),
         sphere = function() frontier_fit_sphere(paste0(hs3, "\n", "visual ~~ v*visual\nv > 0.1"), hs)))
}

run_coverage <- function(data) {
  do.call(rbind, lapply(coverage_items(data), function(it) {
    o <- timed(it$ordinary())$value
    s <- timed(it$sphere())$value
    data.frame(feature = it$feature,
               ordinary = if (is_fit(o) || inherits(o, "magmaan_fit") || is.list(o) &&
                              !inherits(o, "condition")) "fits" else "error",
               sphere = if (inherits(s, "condition")) "error" else "fits",
               sphere_is_sphere = !inherits(s, "condition") &&
                 identical(s$options$chart, "sphere"),
               sphere_message = err_msg(s), ordinary_message = err_msg(o),
               stringsAsFactors = FALSE)
  }))
}
