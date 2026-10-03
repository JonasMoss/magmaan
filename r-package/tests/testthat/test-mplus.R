mplus_cases <- list(
  list(names = "x1 x2 x3", model = "F BY X1 X2 X3;",
       lavaan = "F =~ 1*x1 + x2 + x3; F ~~ F; F ~ 0*1"),
  list(names = "x1 x2 x3 ageyr", model = "F BY x1-x3; F ON ageyr;",
       lavaan = "F =~ 1*x1 + x2 + x3; F ~ ageyr; F ~~ F; F ~ 0*1"),
  list(names = "x1 x2 x3", model = "x2 x3 ON x1;",
       lavaan = "x2 ~ x1; x3 ~ x1; x2 ~~ x3"),
  list(names = "x1 x2 x3", model = "F BY x1 x2*0.8 x3 (1);\nx2 x3 (v);",
       lavaan = "F =~ 1*x1 + start(.8)*x2 + .eq1.*x2 + .eq1.*x3; F ~~ F; F ~ 0*1; x2 ~~ v*x2; x3 ~~ v*x3")
)
mplus_input <- function(case) paste0("TITLE: lab test\nDATA: FILE=x;\nVARIABLE: NAMES=", case$names,
    ";\nMODEL: ", case$model, "\nOUTPUT: TECH1;\n")
mplus_reference <- function(case, data, ...) {
  observed <- strsplit(case$names, " ")[[1]]
  # Explicit residual variances and means; x moments are supplied by fixed.x.
  ys <- if (grepl("age", case$names)) observed[1:3] else if (grepl("x2 x3 ON", case$model)) observed[2:3] else observed
  syntax <- paste(case$lavaan, paste(ys, "~~", ys, collapse="; "),
                  paste(ys, "~ 1", collapse="; "), sep="; ")
  lavaan::lavaan(syntax, data=data, fixed.x=TRUE, meanstructure=TRUE,
    auto.var=FALSE, auto.cov.lv.x=FALSE, auto.cov.y=FALSE,
    auto.fix.first=FALSE, auto.fix.single=FALSE, information="expected", ...)
}
mplus_key <- function(pt) {
  lhs <- pt$lhs; rhs <- pt$rhs
  swap <- pt$op == "~~" & lhs > rhs
  lhs[swap] <- pt$rhs[swap]; rhs[swap] <- pt$lhs[swap]
  paste(lhs, pt$op, rhs)
}

test_that("Mplus rows and ML inference match independent live lavaan models", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  for (case in mplus_cases) {
    m <- mplus_model(mplus_input(case))
    ref <- mplus_reference(case, d)
    pt <- lavaan::parTable(ref)
    # Generated equality rows use row-number plabels, which depend on ordering.
    expect_equal(sum(m$partable$op == "=="), sum(pt$op == "=="))
    pt <- pt[pt$op != "==", ]
    rows <- m$partable[m$partable$op != "==", ]
    idx <- match(mplus_key(rows), mplus_key(pt))
    expect_false(anyNA(idx))
    expect_equal(rows$free > 0, pt$free[idx] > 0)
    expect_equal(rows$ustart, pt$ustart[idx], tolerance=1e-12)
    fit <- fit_model(m, d, estimator="ML")
    expect_true(fit$converged)
    idx <- match(mplus_key(pt), mplus_key(fit$partable))
    expect_false(anyNA(idx))
    # Conditioned x moments are sample quantities, not fitted parameters.
    fitted <- pt$free > 0 | is.finite(pt$ustart)
    expect_equal(fit$partable$est[idx[fitted]], pt$est[fitted], tolerance=1e-5)
    se <- magmaan_core$inference_se(magmaan_core$inference_vcov(magmaan_core$inference_information_expected(fit), fit))
    free <- pt$free > 0
    expect_equal(se[fit$partable$free[idx[free]]], pt$se[free], tolerance=1e-5, ignore_attr=TRUE)
    measures <- fit_measures(fit)
    expect_equal(measures$chisq, unname(lavaan::fitMeasures(ref,"chisq")), tolerance=1e-5)
    expect_equal(measures$df, unname(lavaan::fitMeasures(ref,"df")))
    prepared <- prepare_model(m)
    reused <- estimate(prepared, prepare_data(prepared,d))
    expect_equal(reused$partable$est, fit$partable$est, tolerance=1e-5)
    rebuilt <- magmaanlab:::.rebuild_model_spec(m)
    expect_identical(rebuilt$mplus_source, m$mplus_source)
    expect_equal(fit_model(rebuilt,d)$partable$est, fit$partable$est, tolerance=1e-5)
  }
})

test_that("Mplus inputs retain notes, spelling, starts and serialize across processes", {
  skip_if_not_installed("lavaan")
  case <- mplus_cases[[1]]
  case$names <- "X1 X2 X3"
  m <- mplus_model(mplus_input(case))
  d <- lavaan::HolzingerSwineford1939
  names(d)[match(c("x1","x2","x3"), names(d))] <- c("X1","X2","X3")
  fit <- fit_model(m,d)
  expect_true(all(c("X1","X2","X3") %in% m$partable$lhs))
  expect_named(m$mplus_notes, c("class","rule","line","col","message"))
  expect_output(print(m), "reported but not imported; see \\$mplus_notes")
  inp <- tempfile(fileext=".inp"); writeLines(m$mplus_source, inp)
  expect_equal(mplus_model(file=inp)$partable, m$partable)
  expect_identical(mplus_model(file=inp)$mplus_source, paste0(m$mplus_source, "\n"))
  saved <- tempfile(fileext=".rds"); result <- tempfile(fileext=".rds")
  saveRDS(list(spec=m, fit=fit, data=d), saved)
  script <- tempfile(fileext=".R")
  writeLines(c("library(magmaanlab)",
    sprintf("x <- readRDS(%s)", deparse(saved)),
    "a <- fit_model(x$spec,x$data)",
    "b <- fit_model(x$fit$model,x$data)",
    sprintf("saveRDS(list(a=a$partable$est,b=b$partable$est),%s)", deparse(result))), script)
  status <- system2(file.path(R.home("bin"),"Rscript"), shQuote(script), stdout=TRUE, stderr=TRUE)
  expect_null(attr(status,"status"))
  got <- readRDS(result)
  expect_equal(got$a,fit$partable$est,tolerance=1e-5)
  expect_equal(got$b,fit$partable$est,tolerance=1e-5)
})

test_that("Mplus rejects unsupported inputs and construction overrides explicitly", {
  base <- mplus_input(mplus_cases[[1]])
  for (item in list(c(paste0(base,"DEFINE: x1=2;\n"),"CL16"),
                    c(sub("NAMES=", "GROUPING=g(1=a 2=b); NAMES=",base),"CL11"),
                    c(sub("F BY X1 X2 X3;", "x2 ON x1; x1;",base),"MS08"),
                    c(paste0(base,"ANALYSIS: MODEL=NOMEANSTRUCTURE;\n"),"MS11"),
                    c(sub("x1 x2 x3", "a1b-a3b",base),"NM02"))) {
    expect_error(mplus_model(item[1]), item[2])
    expect_error(mplus_model(item[1]), "instead|remove|write|use", ignore.case=TRUE)
  }
  expect_error(mplus_model(), "exactly one")
  expect_error(mplus_model(base,file="x"), "exactly one")
  m <- mplus_model(base)
  expect_error(magmaanlab:::.rebuild_model_spec(m, overrides=list(fixed_x=FALSE)), "cannot be overridden")
  expect_error(magmaanlab:::.rebuild_model_spec(m, group="g"), "only one group")
})
