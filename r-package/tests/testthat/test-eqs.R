eqs_four <- paste(
  "/EQU V1=1*F1+E7; V2=.8*F1+E8; V3=.7*F1+E9; V4=.9*F1+E10;",
  "/VAR F1=1; E7-E10=.5*; /END")
lavaan_four <- paste(
  "F1 =~ start(1)*x1 + start(.8)*x2 + start(.7)*x3 + start(.9)*x4",
  "F1 ~~ 1*F1; x1 ~~ start(.5)*x1; x2 ~~ start(.5)*x2;",
  "x3 ~~ start(.5)*x3; x4 ~~ start(.5)*x4", sep="\n")

eqs_lavaan <- function(syntax, data, ...) {
  lavaan::lavaan(syntax, data=data, auto.var=FALSE, auto.cov.lv.x=FALSE,
                 auto.cov.y=FALSE, auto.fix.first=FALSE, auto.fix.single=FALSE,
                 fixed.x=FALSE, meanstructure=FALSE, ...)
}

test_that("EQS model rows and start hints match an independently written lavaan model", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  m <- eqs_model(eqs_four, paste0("x",1:4))
  ref <- eqs_lavaan(lavaan_four,d,do.fit=FALSE)
  key <- function(pt) paste(pt$lhs,pt$op,pt$rhs)
  idx <- match(key(m$partable),key(lavaan::parTable(ref)))
  expect_false(anyNA(idx))
  expect_equal(m$partable$free > 0L,lavaan::parTable(ref)$free[idx] > 0L)
  expect_equal(m$partable$ustart,lavaan::parTable(ref)$ustart[idx],tolerance=1e-12)
  expect_identical(m$eqs_source,eqs_four)
  rebuilt <- do.call(model_spec,c(list(syntax=m$syntax),m$options))
  expect_equal(m$partable,rebuilt$partable,ignore_attr=TRUE)
})

test_that("EQS ML fits, standard errors and chi-square agree with live lavaan", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  m <- eqs_model(eqs_four,paste0("x",1:4))
  fit <- fit_model(m,d)
  ref <- eqs_lavaan(lavaan_four,d,information="expected")
  expect_true(fit$converged)
  pt <- lavaan::parTable(ref)
  key <- function(pt) paste(pt$lhs,pt$op,pt$rhs)
  idx <- match(key(pt),key(fit$partable))
  expect_false(anyNA(idx))
  expect_equal(fit$partable$est[idx],pt$est,tolerance=1e-5)
  ours_se <- infer_se(infer_vcov(infer_information_expected(fit), fit))
  free <- pt$free > 0L
  expect_equal(ours_se[fit$partable$free[idx[free]]],pt$se[free],tolerance=1e-5,ignore_attr=TRUE)
  measures <- fit_measures(fit)
  expect_equal(measures$chisq,unname(lavaan::fitMeasures(ref,"chisq")),tolerance=1e-5)
  expect_equal(measures$df,unname(lavaan::fitMeasures(ref,"df")))
  prepared <- prepare_model(m)
  expect_identical(prepared$spec$eqs_source,eqs_four)
  reused <- estimate(prepared,prepare_data(prepared,d))
  expect_equal(reused$partable$est,fit$partable$est,tolerance=1e-5)
})

test_that("EQS unsupported sections and structural errors are explicit", {
  expect_error(eqs_model("/MODEL (V1,V2) ON F1;"),"unsupported EQS section")
  expect_error(eqs_model("/EQU V1=F1+*E1;"),"unit")
  expect_error(eqs_model("/EQU V1=F1+E1; /COV F1,E1=*;"),"cross-covariances")
  expect_error(eqs_model("/EQU V1=.7F1+.2V2+E1; V2=.8F1+E2;"),"indicators participating")
  expect_error(eqs_model("/VAR V1=1;",observed_names=NA_character_),"column names")
})

test_that("EQS covariance defaults do not acquire free factor covariances", {
  m <- eqs_model("/EQU V1=F1+E1; V2=*F1+E2; V3=F2+E3; V4=*F2+E4;")
  expect_false(any(m$partable$op == "~~" & m$partable$lhs != m$partable$rhs))
})
