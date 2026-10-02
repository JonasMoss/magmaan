test_that("saved lavaan specifications supply their structural choices", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  m <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"
  spec <- magmaanlab::model_spec(m)
  expect_equal(coef(magmaan(spec, d, inference = FALSE)),
               coef(magmaan(m, d, inference = FALSE)))
  expect_s3_class(magmaan(spec, d, options = list(start = "fabin3"), inference = FALSE), "magmaan")

  ord <- paste0("x", 1:6)
  for (v in ord) {
    d[[v]] <- as.integer(cut(d[[v]], quantile(d[[v]], c(0, 1/3, 2/3, 1)),
                            include.lowest = TRUE))
  }
  spec <- magmaanlab::model_spec(m, ordered = ord, parameterization = "theta")
  f <- magmaan(spec, d, estimator = "DWLS", inference = FALSE)
  direct <- magmaan(magmaan_model(m, prototype = d, ordered = ord, parameterization = "theta"),
                    d, estimator = "DWLS", inference = FALSE)
  expect_equal(coef(f), coef(direct))
  expect_identical(as_lab_fit(f)$parameterization, "theta")
  expect_equal(coef(magmaan(magmaan_model(spec, prototype = d, ordered = rev(ord),
                                          parameterization = "theta"),
                            d, estimator = "DWLS", inference = FALSE)), coef(f))
  expect_error(magmaan(spec, d), "treats every variable as continuous")
  expect_error(magmaan_model(spec, prototype = d, ordered = ord[-1]), "conflicts")
  expect_error(magmaan_model(spec, prototype = d, parameterization = "delta"), "conflicts")
  expect_error(magmaan(spec, d, estimator = "DWLS", options = list(start = "fabin3")),
               "not available")
})

test_that("saved grouped specifications retain their groups and row accounting", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  d$x1[1:5] <- NA
  m <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"
  appearance <- unique(as.character(d$school))
  spec <- magmaanlab::model_spec(m, group = "school", group_labels = appearance)
  f <- magmaan(spec, d, inference = FALSE)
  expect_identical(f$rows$group, appearance)
  chr <- d
  chr$school <- as.character(chr$school)
  direct <- magmaan(magmaan_model(m, prototype = chr, group = "school"), d, inference = FALSE)
  expect_identical(f$rows, direct$rows)
  expect_equal(coef(f), coef(direct))
  expect_equal(sum(f$rows$deleted), 5L)
  expect_equal(sum(f$rows$used), nobs(f))
  expect_equal(f$rows$group, names(fitted(f)))
  expect_error(magmaan_model(spec, prototype = d, group = "sex"), "conflicts")
})

test_that("EQS and partable-only specifications stay in the lab", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  eqs <- magmaanlab::eqs_model("/EQU V1=1*F1+E1; V2=.8*F1+E2; V3=.7*F1+E3; /VAR F1=1; E1-E3=.5*;",
                              observed_names = paste0("x", 1:3))
  expect_error(magmaan(eqs, d), "use magmaanlab::fit_model")
  spec <- magmaanlab::model_spec("visual =~ x1 + x2 + x3")
  spec$syntax <- NULL
  expect_error(magmaan(spec, d), "must carry lavaan syntax")
})

test_that("defined estimates survive deferred and unsupported inference", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  m <- paste("visual =~ x1 + a*x2 + b*x3", "textual =~ x4 + x5 + x6",
             "ab := a*b", "twice := 2*ab", sep = "\n")
  for (est in c("ML", "ULS", "FIML")) {
    f <- magmaan(m, d, estimator = est, inference = FALSE)
    p <- coef(summary(f))
    defs <- p[p$op == ":=", ]
    lav <- lavaan::parameterEstimates(lavaan::cfa(m, d, estimator = if (est == "FIML") "ML" else est,
                                                missing = if (est == "FIML") "ml" else "listwise",
                                                se = "none"))
    expect_equal(defs$est, lav$est[lav$op == ":="], tolerance = 1e-4)
    expect_true(all(is.na(defs$se)))
    later <- coef(summary(infer(f)))
    expect_equal(later$est[later$op == ":="], defs$est)
    if (est == "ML") expect_true(all(is.finite(later$se[later$op == ":="])))
    else expect_true(all(is.na(later$se[later$op == ":="])))
  }
  fixed <- "visual =~ 1*x1 + a*0.8*x2 + b*0.6*x3\nvisual ~~ 1*visual\nx1 ~~ 1*x1\nx2 ~~ 1*x2\nx3 ~~ 1*x3\nab := a*b"
  p <- coef(summary(magmaan(fixed, d, inference = FALSE)))
  expect_equal(p$est[p$op == ":="], 0.48)
  expect_true(is.na(p$se[p$op == ":="]))
})

test_that("interval arguments fail clearly and numeric selection preserves order", {
  skip_if_not_installed("lavaan")
  f <- magmaan("visual =~ x1 + x2 + x3", lavaan::HolzingerSwineford1939)
  for (level in list(0, 1, 2, NA_real_, Inf, c(.8, .95), "0.95", .95 + 0i)) {
    expect_error(confint(f, level = level), "strictly between zero and one")
    expect_error(summary(f, level = level), "strictly between zero and one")
  }
  expect_error(confint(f, parm = "typo"), "unknown parameter")
  for (parm in list(0, -1, 1.5, 100, NA_integer_, TRUE, 1 + 0i)) {
    expect_error(confint(f, parm = parm), "positive integer indices")
  }
  expect_equal(confint(f, parm = c(2L, 1L)), confint(f)[c(2L, 1L), ])
  expect_equal(confint(f, parm = names(coef(f))[2:3]), confint(f)[2:3, ])
  expect_equal(dim(confint(f, parm = integer())), c(0L, 2L))
})

test_that("unchecked convergence and retained fits remain inspectable", {
  skip_if_not_installed("lavaan")
  f <- magmaan("visual =~ x1 + x2 + x3", lavaan::HolzingerSwineford1939)
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path))
  saveRDS(f, path)
  restored <- readRDS(path)
  expect_identical(coef(restored), coef(f))
  expect_identical(vcov(restored), vcov(f))
  expect_equal(fitted(restored), fitted(f))
  expect_equal(infer(restored)$inference, f$inference)
  f$lab$converged <- NA
  f$lab$verdict$status <- "unchecked"
  expect_output(print(f), "converged: +unchecked")
  f$lab$converged <- FALSE
  expect_output(print(f), "converged: +no")
})
