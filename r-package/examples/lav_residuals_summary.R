## lavResiduals()-style residual $summary table.
##
## lav_residuals(fit)$summary mirrors lavaan::lavResiduals(fit)$summary: per
## block, the cor.bentler SRMR family (SRMR, its asymptotic SE, an exact-fit
## z-test against 0, and the bias-corrected USRMR with a close-fit confidence
## interval and a close-fit z-test against 0.05). With a mean structure the
## `mean` and `total` sections are populated alongside `cov`. This is straight
## lavaan parity; the underlying residual ACOV is the same projection lavaan
## uses, rescaled into the correlation metric.

suppressMessages({library(lavaan); library(magmaan)})

HS.model <- "visual  =~ x1 + x2 + x3
             textual =~ x4 + x5 + x6
             speed   =~ x7 + x8 + x9"

## Compare one magmaan summary data frame to lavaan's, column by column. Finite
## entries must agree tightly; non-finite (degenerate-section NA) entries must
## agree on which cells are NA.
compare_summary <- function(mag, lav, tol = 1e-4) {
  for (cc in colnames(lav)) {
    a <- mag[[cc]]
    b <- as.numeric(lav[[cc]])
    stopifnot(!is.null(a), length(a) == length(b))
    fin <- is.finite(a) & is.finite(b)
    stopifnot(all(is.finite(a) == is.finite(b)))
    if (any(fin)) stopifnot(max(abs(a[fin] - b[fin])) < tol)
  }
  invisible(TRUE)
}

## --- single group, no mean structure -----------------------------------------
f1 <- cfa(HS.model, data = HolzingerSwineford1939)
m1 <- magmaan(HS.model, HolzingerSwineford1939, estimator = "ML")
s1 <- lav_residuals(m1)$summary[[1]]
compare_summary(s1, lavResiduals(f1)$summary)
stopifnot(identical(colnames(s1), "cov"))

cat("no mean structure - $summary cov column:\n")
print(round(s1$cov, 5))

## --- single group, with mean structure (cov / mean / total) ------------------
f2 <- cfa(HS.model, data = HolzingerSwineford1939, meanstructure = TRUE)
m2 <- magmaan(HS.model, HolzingerSwineford1939, estimator = "ML",
              meanstructure = TRUE)
s2 <- lav_residuals(m2)$summary[[1]]
compare_summary(s2, lavResiduals(f2)$summary)
stopifnot(identical(colnames(s2), c("cov", "mean", "total")))

## --- two groups (configural) with mean structure -----------------------------
f3 <- cfa(HS.model, data = HolzingerSwineford1939, group = "school",
          meanstructure = TRUE)
m3 <- magmaan(HS.model, HolzingerSwineford1939, estimator = "ML",
              groups = "school", meanstructure = TRUE)
s3 <- lav_residuals(m3)$summary
## magmaan and lavaan may order groups differently; align by label.
lav_labels <- lavInspect(f3, "group.label")
for (g in seq_along(s3)) {
  li <- match(m3$group_labels[g], lav_labels)
  compare_summary(s3[[g]], lavResiduals(f3)[[li]]$summary, tol = 1e-3)
}

cat("\nlav_residuals() $summary matches lavaan::lavResiduals(): ok\n")
