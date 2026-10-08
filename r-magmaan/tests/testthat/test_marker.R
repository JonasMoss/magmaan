.marker_sample <- function() {
  set.seed(1291); f <- rnorm(600)
  data.frame(x1=.01*f+rnorm(600),x2=.8*f+.6*rnorm(600),x3=.7*f+.7*rnorm(600),x4=.6*f+.8*rnorm(600))
}
test_that("ordinary switched fits report and infer from their fitted partable", {
  skip_if_not_installed("lavaan")
  skip_if(as.character(packageVersion("lavaan")) != "0.7.2")
  d <- .marker_sample(); model <- magmaan_model("f =~ x1+x2+x3+x4")
  fit <- magmaan(model,d,inference=FALSE,options=list(preset="lavaan-0.7.2"))
  lab <- as_lab_fit(fit); info <- lab$fitting$marker_switch
  expect_equal(nrow(info),1L); expect_false(info$reverted)
  expect_identical(fit$fitting,lab$fitting)
  expect_equal(model$spec$options$marker,NULL)
  expect_equal(fit$model$spec$options$marker,setNames(info$new,"f"))
  warm <- magmaan(model,d,inference=FALSE,options=list(preset="lavaan-0.7.2",start=fit))
  expect_equal(coef(warm),coef(fit),tolerance=1e-5)
  fit <- infer(fit)
  expect_true(is.numeric(coef(fit)))
  expect_true(is.matrix(vcov(fit)))
  expect_true(is.data.frame(coef(summary(fit))))
  expect_true(is.list(fitted(fit)))
  expect_true(is.data.frame(coef(summary(fit,standardized=TRUE))))
  expect_true(is.data.frame(modindices(fit)))
  expect_true(is.data.frame(fit_measures(fit)))
  expect_output(print(fit),"Marker switched as lavaan 0.7.2 does")
  expect_output(print(summary(fit)),"Marker switched as lavaan 0.7.2 does")
  expect_error(anova(fit,fit),class="magmaan_unsupported_model")
  expect_error(magmaan(model,d,options=list(marker="bad")),"options\\$marker")
  off <- magmaan(model,d,inference=FALSE,options=list(preset="lavaan-0.7.2",marker="default"))
  expect_equal(nrow(off$lab$fitting$marker_switch),0L)
})

test_that("ordinary FIML and categorical DWLS retain the switched specification", {
  skip_if_not_installed("lavaan")
  skip_if(as.character(packageVersion("lavaan")) != "0.7.2")
  for (route in c("FIML", "ordinal", "mixed")) {
    d <- .marker_sample()
    ordered <- if (route == "ordinal") names(d) else if (route == "mixed") names(d)[1:3] else character()
    for (v in ordered) d[[v]] <- ordered(cut(d[[v]], c(-Inf, -.4, .4, Inf)))
    if (route == "FIML") d[seq(1L, 600L, 10L), "x4"] <- NA_real_
    model <- magmaan_model("f =~ x1+x2+x3+x4", prototype = d, ordered = ordered)
    fit <- suppressWarnings(magmaan(model, d, estimator = if (route == "FIML") "FIML" else "DWLS",
      inference = FALSE, options = list(preset = "lavaan-0.7.2")))
    lab <- as_lab_fit(fit)
    info <- lab$fitting$marker_switch
    expect_equal(info$old, "x1"); expect_false(info$reverted)
    expect_equal(fit$model$spec$options$marker, setNames(info$new, "f"))
    expect_identical(fit$fitting, lab$fitting)
    expect_output(print(fit), "Marker switched as lavaan 0.7.2 does")
  }
})
