test_that("anova and robust nested LR preserve units in a two-group path model", {
  set.seed(6106)
  n <- 300L
  d <- data.frame(x = rnorm(2L*n), group = rep(c("a", "b"), each = n))
  d$y <- 0.6*d$x + rnorm(2L*n)
  h1 <- "y ~ c(b1, b2)*x"
  h0 <- paste(h1, "b1 == b2", sep = "\n")
  compare <- function(units) {
    changed <- d
    changed$y <- units*changed$y
    f1 <- magmaan(magmaan_model(h1, prototype = changed, group = "group"), changed,
                  inference = FALSE)
    f0 <- magmaan(magmaan_model(h0, prototype = changed, group = "group"), changed,
                  inference = FALSE)
    ordinary <- anova(f0, f1, references = c("sb", "peba4"))
    expect_length(attr(ordinary, "unavailable"), 0L)
    observed <- as_lab_fit(f1)$ov_names
    if (is.list(observed)) observed <- observed[[1L]]
    robust <- magmaanlab::robust_nested_lrt(as_lab_fit(f1), as_lab_fit(f0),
                                             data = split(changed[observed], changed$group))
    lr <- ordinary$test == "lr"
    expect_equal(ordinary$statistic[lr & ordinary$reference %in% "sb"] /
      magmaanlab::policy_nested(as_lab_fit(f1), as_lab_fit(f0))$lr$sb_scale,
                 robust$T_scaled, tolerance = 1e-6)
    list(ordinary = ordinary, robust = robust,
         policy_scale = magmaanlab::policy_nested(as_lab_fit(f1), as_lab_fit(f0))$lr$sb_scale)
  }
  reference <- compare(1)
  # Includes the approximately 300-fold variance contrast from the handoff,
  # then the explicitly requested 100-fold change of measurement units.
  for (units in c(sqrt(300), 100)) {
    changed <- compare(units)
    expect_equal(changed$ordinary$statistic, reference$ordinary$statistic,
                 tolerance = 1e-5)
    expect_equal(changed$policy_scale, reference$policy_scale, tolerance = 1e-5)
    expect_equal(changed$ordinary$pvalue[changed$ordinary$reference %in% "sb"], reference$ordinary$pvalue[reference$ordinary$reference %in% "sb"], tolerance = 1e-5)
    expect_equal(changed$ordinary$pvalue[changed$ordinary$reference %in% "peba4"], reference$ordinary$pvalue[reference$ordinary$reference %in% "peba4"],
                 tolerance = 1e-5)
    expect_equal(changed$robust$T_scaled, reference$robust$T_scaled,
                 tolerance = 1e-5)
    expect_equal(changed$robust$eigenvalues, reference$robust$eigenvalues,
                 tolerance = 1e-5)
  }
})
