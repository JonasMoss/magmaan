hs <- function() {
  skip_if_not_installed("lavaan")
  lavaan::HolzingerSwineford1939
}
cfa <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"

# Every ordinary model has a mean structure, so lavaan references use one too.
lav_cfa <- function(model, data, ...) lavaan::cfa(model, data, meanstructure = TRUE, ...)

# A parameter row's key: lhs, op, rhs and group label, so models whose groups
# are numbered in different orders compare correctly.
.key <- function(pt, labels) {
  group <- if (is.null(pt$group) || !length(labels)) "" else labels[pt$group]
  paste(pt$lhs, pt$op, pt$rhs, group)
}
.lav_labels <- function(lav) {
  if (lavaan::lavInspect(lav, "ngroups") > 1L) lavaan::lavInspect(lav, "group.label") else character()
}

# Same parameter rows as lavaan, and the same free estimates.
expect_lavaan_estimates <- function(fit, lav, tolerance = 1e-4) {
  ours <- fit$lab$partable
  theirs <- lavaan::parTable(lav)
  keep <- function(pt) pt[pt$op != "==", , drop = FALSE]
  ours_key <- .key(keep(ours), fit$lab$group_labels)
  expect_setequal(ours_key, .key(keep(theirs), .lav_labels(lav)))
  theirs <- theirs[theirs$free > 0L, , drop = FALSE]
  idx <- match(.key(theirs, .lav_labels(lav)), .key(ours, fit$lab$group_labels))
  expect_false(anyNA(idx))
  expect_equal(ours$est[idx], theirs$est, tolerance = tolerance)
}

# Standard errors of every lavaan free row.
expect_lavaan_se <- function(fit, lav, tolerance = 1e-4) {
  p <- coef(summary(fit))
  theirs <- lavaan::parTable(lav)
  theirs <- theirs[theirs$free > 0L, , drop = FALSE]
  idx <- match(.key(theirs, .lav_labels(lav)), .key(p, fit$lab$group_labels))
  expect_false(anyNA(idx))
  expect_equal(p$se[idx], theirs$se, tolerance = tolerance)
}

ordinal_hs <- function() {
  d <- hs()
  for (v in paste0("x", 1:6)) {
    d[[v]] <- ordered(cut(d[[v]], breaks = stats::quantile(d[[v]], c(0, 1 / 3, 2 / 3, 1)),
                          include.lowest = TRUE, labels = FALSE), levels = 1:3)
  }
  d
}
