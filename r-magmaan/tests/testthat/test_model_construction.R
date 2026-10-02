hs <- function() {
  skip_if_not_installed("lavaan")
  lavaan::HolzingerSwineford1939
}
cfa <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"
ord <- paste0("x", 1:6)
ordinal_hs <- function() {
  d <- hs()
  for (v in ord) {
    d[[v]] <- ordered(cut(d[[v]], breaks = stats::quantile(d[[v]], c(0, 1 / 3, 2 / 3, 1)),
                          include.lowest = TRUE, labels = FALSE), levels = 1:3)
  }
  d
}
schema_reason <- function(expr) {
  tryCatch({
    force(expr)
    "no error"
  }, magmaan_schema_error = function(e) e$reason)
}

test_that("a constructed model is reused unchanged across datasets", {
  d <- hs()
  m <- magmaan_model(cfa)
  expect_s3_class(m, "magmaan_model")
  expect_identical(coef(magmaan(m, d, inference = FALSE)),
                   coef(magmaan(cfa, d, inference = FALSE)))
  before <- m
  set.seed(1)
  for (i in 1:3) {
    resample <- d[sample(nrow(d), replace = TRUE), ]
    fit <- magmaan(m, resample, inference = FALSE)
    expect_equal(nobs(fit), nrow(d))
    expect_equal(unname(coef(fit)),
                 magmaanlab::fit_model(cfa, resample, meanstructure = TRUE, fixed_x = FALSE)$theta)
  }
  expect_identical(m, before)
  expect_identical(magmaan_model(m), m)
  expect_error(magmaan_model(m, group = "school"), "already constructed")
  expect_output(print(m), "observed: +x1, x2, x3, x4, x5, x6")
})

test_that("every model has a mean structure and random covariates", {
  m <- magmaan_model("x1 ~ x2 + x3")
  pt <- m$spec$partable
  expect_setequal(pt$lhs[pt$op == "~1"], c("x1", "x2", "x3"))
  expect_true(all(pt$free[pt$lhs %in% c("x2", "x3") & pt$op == "~~"] > 0L))
  expect_false(any(pt$exo == 1L))
})

test_that("grouped models freeze their groups and order", {
  d <- hs()
  skeleton <- d[0, c(ord, "school")]
  m <- magmaan_model(cfa, prototype = skeleton, group = "school")
  expect_identical(m$groups, levels(d$school))
  fit <- magmaan(m, d, inference = FALSE)
  expect_identical(as_lab_fit(fit)$group_labels, levels(d$school))
  expect_equal(fit$rows$rows, as.integer(table(d$school)))
  # Appearance order for a character column.
  chr <- d
  chr$school <- as.character(chr$school)
  appearance <- magmaan_model(cfa, prototype = chr, group = "school")
  expect_identical(appearance$groups, unique(chr$school))
  # The same estimates whatever the row order of the data.
  shuffled <- d[rev(seq_len(nrow(d))), ]
  expect_equal(coef(magmaan(m, shuffled, inference = FALSE)), coef(fit), tolerance = 1e-6)
  expect_error(magmaan_model(cfa, group = "school"), "needs `prototype`")
  expect_error(magmaan_model(cfa, prototype = d, group = "district"), "not in `prototype`")
  expect_identical(schema_reason(magmaan(m, chr[chr$school == "Pasteur", ])), "empty_group")
  other <- chr
  other$school[1] <- "Elsewhere"
  expect_identical(schema_reason(magmaan(m, other)), "undeclared_group")
})

test_that("ordinal models freeze their categories", {
  o <- ordinal_hs()
  m <- magmaan_model(cfa, prototype = o[0, ord], ordered = ord)
  expect_identical(m$categories$x1, as.character(1:3))
  fit <- magmaan(m, o, estimator = "DWLS", inference = FALSE)
  # Integer codes with the declared categories give the same fit.
  codes <- o
  codes[ord] <- lapply(codes[ord], as.integer)
  expect_equal(coef(magmaan(m, codes, estimator = "DWLS", inference = FALSE)), coef(fit))
  empty <- o
  empty$x1[empty$x1 == "3"] <- "2"
  expect_identical(schema_reason(magmaan(m, empty, estimator = "DWLS")), "empty_category")
  reordered <- o
  reordered$x1 <- factor(reordered$x1, levels = 3:1)
  expect_identical(schema_reason(magmaan(m, reordered, estimator = "DWLS")), "changed_levels")
  extra <- codes
  extra$x2[1] <- 4L
  expect_identical(schema_reason(magmaan(m, extra, estimator = "DWLS")), "undeclared_category")
  # A category without complete rows in one group fails, even if others have it.
  grouped <- magmaan_model(cfa, prototype = o[0, c(ord, "school")], ordered = ord,
                           group = "school")
  sparse <- o
  sparse$x1[sparse$school == "Pasteur" & sparse$x1 == "3"] <- NA
  err <- tryCatch(magmaan(grouped, sparse, estimator = "DWLS"),
                  magmaan_schema_error = function(e) e)
  expect_identical(err$reason, "empty_category")
  expect_match(conditionMessage(err), "in group Pasteur")
  expect_error(magmaan_model(cfa, ordered = ord), "need `prototype`")
  expect_error(magmaan_model(cfa, prototype = o, ordered = "x9"), "must occur in the model")
})

test_that("ordered factors must be declared", {
  o <- ordinal_hs()
  expect_error(magmaan(cfa, o, estimator = "DWLS"), "ordered factor in `prototype`")
  m <- magmaan_model(cfa)
  expect_error(magmaan(m, o), "ordered factor in `data`")
  expect_error(magmaan_model(cfa, prototype = o, ordered = "x1"), "ordered factor in `prototype`")
})

test_that("lab specifications are constructor input under the ordinary contract", {
  d <- hs()
  spec <- magmaanlab::model_spec(cfa)
  expect_equal(coef(magmaan(spec, d, inference = FALSE)), coef(magmaan(cfa, d, inference = FALSE)))
  std <- magmaanlab::model_spec(cfa, std_lv = TRUE)
  expect_identical(magmaan_model(std)$identification, "std.lv")
  expect_error(magmaan_model(std, identification = "marker"), "conflicts")
  # Fixed covariates are rejected rather than converted.
  expect_error(magmaan_model(magmaanlab::model_spec("x1 ~ x2")), "fixes its exogenous covariates")
  expect_s3_class(magmaan_model(magmaanlab::model_spec("x1 ~ x2", fixed_x = FALSE)),
                  "magmaan_model")
})

test_that("removed arguments name their replacement", {
  d <- hs()
  expect_error(magmaan(cfa, d, psd = TRUE), "covariance = \"psd\"")
  expect_error(magmaan(cfa, d, start = "fabin3"), "options = list\\(start")
  expect_error(magmaan(cfa, d, fixed.x = FALSE), "random")
  expect_error(magmaan(cfa, d, meanstructure = TRUE), "every model has a mean structure")
  expect_error(magmaan(cfa, d, missing = "pairwise"), "magmaanlab::fit_model")
  expect_error(magmaan(cfa, d, cluster = "school"), "two-level")
  for (arg in c("ordered", "group", "group.equal", "group.partial",
                "identification", "parameterization")) {
    expect_error(do.call(magmaan, stats::setNames(list(cfa, d, "x"), c("model", "data", arg))),
                 paste0("`", arg, "` is an argument of magmaan_model"), fixed = TRUE)
  }
  expect_error(magmaan(cfa, d, group = undefined_variable), "magmaan_model")
  expect_error(magmaan(cfa, d, typo = 1), "unused argument `typo`")
  expect_error(magmaan(cfa, d, options = list(starts = "fabin3")), "options\\$start")
  expect_error(magmaan(cfa, d, options = list(solver = "port")), "unknown option")
  expect_error(magmaan(cfa, d, options = list(optimizer = "nlopt-slsqp")), "must be one of")
})

test_that("barrier fits are experimental, penalized and recorded", {
  d <- hs()
  assign("barrier_shown", FALSE, envir = getFromNamespace(".session", "magmaan"))
  expect_message(fit <- magmaan(cfa, d, covariance = "barrier"), "experimental")
  expect_no_message(again <- magmaan(cfa, d, covariance = barrier(0.25)))
  expect_identical(coef(again), coef(fit))
  expect_true(fit$experimental)
  expect_identical(fit$covariance$lambda, 0.25)
  lab <- as_lab_fit(fit)
  expect_identical(lab$composition$model_penalty, "joint")
  expect_identical(lab$composition$penalty_weight, 0.25)
  expect_true(is.finite(lab$penalty$penalized_fmin))
  expect_true(all(fit$inference$status$reason == "penalized"))
  expect_error(vcov(fit), class = "magmaan_inference_unavailable")
  expect_output(print(fit), "barrier\\(lambda = 0.25\\), experimental")
  expect_output(print(fit), "unavailable \\(penalized\\)")
  # An interior estimate moves by O(lambda / N).
  plain <- magmaan(cfa, d)
  expect_lt(max(abs(coef(fit) - coef(plain))), 0.05)
  stronger <- magmaan(cfa, d, covariance = barrier(4), inference = FALSE)
  expect_gt(max(abs(coef(stronger) - coef(plain))), max(abs(coef(fit) - coef(plain))))
  # barrier(0) is the unrestricted fit, with its inference.
  zero <- magmaan(cfa, d, covariance = barrier(0))
  expect_identical(coef(zero), coef(plain))
  expect_equal(zero$inference$status, plain$inference$status)
  expect_false(zero$experimental)
  expect_output(print(zero), "barrier\\(0\\): the unrestricted fit")
  # Nested tests between penalized fits are unavailable too.
  wider <- magmaan(paste(cfa, "x1 ~~ x4", sep = "\n"), d, covariance = "barrier", inference = FALSE)
  nested <- anova(magmaan(cfa, d, covariance = "barrier", inference = FALSE), wider)
  expect_true(all(grepl("penalized", attr(nested, "unavailable"))))
  expect_error(anova(plain, fit), "covariance policy")
  for (bad in list(-1, NA_real_, Inf, c(1, 2), "1")) expect_error(barrier(bad), "finite non-negative")
  expect_error(magmaan(cfa, d, covariance = "barier"), "barrier\\(lambda\\)")
  expect_error(magmaan(cfa, d, covariance = "barrier", options = list(preset = "lavaan-0.7.2")),
               "unrestricted covariance")
})

test_that("barrier fits cover the primary estimators", {
  d <- hs()
  d$x1[1:10] <- NA
  fiml <- suppressMessages(magmaan(cfa, d, estimator = "FIML", covariance = "barrier"))
  expect_true(all(fiml$inference$status$reason == "penalized"))
  o <- ordinal_hs()
  m <- magmaan_model(cfa, prototype = o, ordered = ord)
  dwls <- suppressMessages(magmaan(m, o, estimator = "DWLS", covariance = "barrier"))
  expect_identical(as_lab_fit(dwls)$composition$model_penalty, "joint")
  expect_true(all(dwls$inference$status$reason == "penalized"))
})
