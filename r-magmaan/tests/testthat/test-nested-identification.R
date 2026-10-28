# The nested LR reference must not depend on how the factors are identified:
# marker and std.lv fit the same models, so they share the statistic, and the
# reference spectrum is built so they share the p-value too (TASK-168). With
# the exact nesting rows used before, an equality on this model's visual
# loadings gave ML p-values 4% apart (relative); the tolerances leave room only
# for optimizer and finite-difference noise.
test_that("nested LR p-values are invariant to marker versus std.lv identification", {
  hs <- lavaan::HolzingerSwineford1939[, paste0("x", 1:9)]
  h1 <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\nspeed =~ x7 + a*x8 + b*x9"
  h0 <- paste0(h1, "\na == 0.5*b")
  cut4 <- function(x) ordered(cut(x, c(-Inf, stats::quantile(x, c(.25, .5, .75)), Inf)))
  set.seed(168)
  missing <- hs
  for (v in names(missing)) missing[[v]][sample.int(nrow(missing), 25)] <- NA
  ordinal <- hs
  for (v in names(ordinal)) ordinal[[v]] <- cut4(ordinal[[v]])
  mixed <- hs
  for (v in paste0("x", 1:3)) mixed[[v]] <- cut4(mixed[[v]])
  cases <- list(ML = list(hs, "ML", 1e-6), FIML = list(missing, "ML", 1e-6),
                DWLS = list(ordinal, "DWLS", 1e-4), mixed_DWLS = list(mixed, "DWLS", 1e-4),
                association_ML = list(ordinal, "ML", 1e-4))
  for (name in names(cases)) {
    data <- cases[[name]][[1]]
    ordered <- names(data)[vapply(data, is.ordered, logical(1))]
    lr <- lapply(c("marker", "std.lv"), function(identification) {
      spec <- function(syntax) {
        if (length(ordered)) magmaan_model(syntax, identification = identification,
                                           prototype = data, ordered = ordered)
        else magmaan_model(syntax, identification = identification, prototype = data)
      }
      f1 <- magmaan(spec(h1), data, estimator = cases[[name]][[2]])
      f0 <- magmaan(spec(h0), data, estimator = cases[[name]][[2]])
      a <- as.data.frame(anova(f0, f1))
      a[a$test != "score", c("statistic", "pvalue")][1, ]
    })
    expect_true(is.finite(lr[[1]]$pvalue), label = name)
    expect_equal(lr[[2]]$statistic, lr[[1]]$statistic, tolerance = 1e-6, label = name)
    expect_equal(lr[[2]]$pvalue, lr[[1]]$pvalue, tolerance = cases[[name]][[3]], label = name)
  }
})
