test_that("ordinary fits accept the shared EQS constructor", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  source <- paste("/EQU V1=1*F1+E1; V2=.8*F1+E2; V3=.7*F1+E3; V4=.9*F1+E4;",
                  "/VAR F1=1; E1-E4=.5*;")
  model <- eqs_model(source, paste0("x",1:4))
  fit <- magmaan(model,d,inference=FALSE)
  ref <- magmaanlab::fit_model(model,d)
  expect_s3_class(fit,"magmaan")
  expect_equal(as_lab_fit(fit)$partable,ref$partable)
  inferred <- magmaan(model,d)
  expect_s3_class(inferred,"magmaan")
  expect_equal(as_lab_fit(inferred)$partable$est,ref$partable$est)
  reported <- coef(summary(inferred))
  expect_true(all(is.finite(reported$se[reported$free > 0L])))
  expect_identical(eqs_model,magmaanlab::eqs_model)
  expect_error(magmaan(model,d,identification="marker"),"model constructor")
  expect_error(magmaan(model,d,fixed.x=TRUE),"model constructor")
})
