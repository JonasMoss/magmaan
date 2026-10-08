.marker_oracle <- function() {
  skip_if_not_installed("lavaan")
  skip_if(as.character(packageVersion("lavaan")) != "0.7.2")
}
.marker_data <- function(seed = 1291L, n = 600L) {
  set.seed(seed); f <- rnorm(n)
  data.frame(x1 = .01*f+rnorm(n), x2 = .8*f+.6*rnorm(n),
             x3 = .7*f+.7*rnorm(n), x4 = .6*f+.8*rnorm(n))
}
.marker_compare <- function(fit, lv) {
  a <- fit$partable; b <- lavaan::parTable(lv)
  key <- function(p) paste(p$lhs,p$op,p$rhs,p$group)
  ix <- match(key(a), key(b))
  expect_equal(a$free, b$free[ix])
  parameters <- !a$op %in% c("==", "<", ">", ":=")
  expect_equal(a$est[parameters], b$est[ix][parameters], tolerance = 1e-5)
  expect_equal(fit$converged, lavaan::lavInspect(lv,"converged"))
  expect_equal(fit$fmin, as.numeric(lv@optim$fx), tolerance = 1e-9)
}
test_that("complete ML switches and retains the actual fitted specification", {
  .marker_oracle(); d <- .marker_data(); syntax <- "f =~ x1+x2+x3+x4"
  warnings <- character()
  lv <- withCallingHandlers(lavaan::cfa(syntax,d,meanstructure=TRUE,fixed.x=FALSE),
    warning=function(w) {warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning")})
  expect_true(any(grepl("marker",warnings)))
  fit <- fit_model(model_spec(syntax,meanstructure=TRUE,fixed_x=FALSE),d,
                   options=list(preset="lavaan-0.7.2"))
  .marker_compare(fit,lv)
  info <- fit$fitting$marker_switch
  expect_equal(info$lv,"f"); expect_equal(info$old,"x1")
  expect_false(info$reverted)
  expected <- lavaan:::lav_pt_marker_adapt(lavaan::lavaanify(syntax,auto.fix.first=TRUE),
    list(implied=list(cov=list(stats::cov(d)))))$info
  expect_equal(info$new,expected$new)
  expect_equal(info$r_old,expected$r.old); expect_equal(info$r_new,expected$r.new)
  expect_equal(fit$model$options$marker, setNames(info$new,"f"))
  expect_identical(fit$fitting$effective$marker,"lavaan-0.7.2")
  fixed <- fit$model$partable$op == "=~" & fit$model$partable$free == 0
  expect_equal(fit$model$partable$rhs[fixed],info$new)
  own <- fit_model(model_spec(syntax,meanstructure=TRUE,fixed_x=FALSE),d,
                   options=list(preset="lavaan-0.7.2",marker="default"))
  off <- suppressWarnings(lavaan::cfa(syntax,d,meanstructure=TRUE,fixed.x=FALSE,bad.marker.crit=0))
  .marker_compare(own,off)
  expect_equal(nrow(own$fitting$marker_switch),0L)
  expect_true(own$fitting$modified_preset)
  expect_error(case_rerun(fit),class="magmaan_unsupported_model")
  expect_error(refit_from_null(fit,fit),class="magmaan_unsupported_model")
  expect_error(policy_nested(fit,fit),class="magmaan_unsupported_model")
  expect_error(convention_nested(fit,fit,"ML"),class="magmaan_unsupported_model")
  expect_error(robust_nested_lrt(fit,fit,method="restriction_map"),class="magmaan_unsupported_model")
  expect_error(nested_score_test(fit,fit),class="magmaan_unsupported_model")
  expect_true(is.list(policy_inference(fit)))
  expect_true(is.list(convention_inference(fit,"ML")))
  expect_true(is.matrix(vcov(fit)))
  expect_true(is.list(magmaan_core$model_implied(fit)))
  expect_true(is.list(residuals(fit)))
  expect_output(print(fit),"magmaan fit")
  expect_true(is.data.frame(modification_indices(fit, data=d)))
  expect_true(is.data.frame(score_tests(fit, data=d)))
  expect_true(is.data.frame(policy_modification_indices(fit)))
  expect_true(is.data.frame(policy_fit_measures(fit)))
})
test_that("grouped loading equalities rebuild in the switched coordinates", {
  .marker_oracle(); d <- rbind(.marker_data(),.marker_data(1292)); d$g <- rep(c("A","B"),each=600)
  syntax <- "f =~ x1+x2+x3+x4"
  spec <- model_spec(syntax,meanstructure=TRUE,fixed_x=FALSE,group="g",group_labels=c("A","B"),group_equal="loadings")
  fit <- fit_model(spec,d,options=list(preset="lavaan-0.7.2"))
  lv <- suppressWarnings(lavaan::cfa(syntax,d,group="g",group.equal="loadings",meanstructure=TRUE,fixed.x=FALSE))
  .marker_compare(fit,lv)
  expect_equal(nrow(fit$fitting$marker_switch),1L)
})
test_that("marker validation and deferred routes are explicit", {
  .marker_oracle(); d <- .marker_data()
  spec <- model_spec("f =~ x1+x2+x3+x4",meanstructure=TRUE,fixed_x=FALSE)
  expect_error(fit_model(spec,d,options=list(marker="unknown")),"marker")
  expect_error(fit_model(spec,d,options=list(marker=NA_character_)),"nonmissing")
  expect_error(fit_model(spec,d,estimator="FIML",options=list(marker="lavaan-0.7.2")),"unsupported_model")
  comp <- model_spec("f <~ x1+x2+x3\nx4 ~ f",meanstructure=TRUE,fixed_x=FALSE)
  expect_error(fit_model(comp,d,options=list(preset="lavaan-0.7.2")),"composite")
  expect_error(fit_model(spec$partable,d,options=list(preset="lavaan-0.7.2")),
               class="magmaan_unsupported_model")
  prepared <- prepare_model(spec); data <- prepare_data(prepared,d)
  fit <- estimate(prepared,data,options=list(preset="lavaan-0.7.2"))
  .marker_compare(fit,suppressWarnings(lavaan::cfa(spec$syntax,d,meanstructure=TRUE,fixed.x=FALSE)))
  expect_equal(nrow(fit$fitting$marker_switch),1L)
})
test_that("a rejected switched fit retains the original verdict on reversion", {
  info <- data.frame(lv="f",old="x1",new="x2",r_old=.01,r_new=.4,reverted=FALSE)
  original <- list(converged=FALSE,theta=c(1,2),fitting=list())
  selected <- .marker_select(list(converged=FALSE,theta=c(3,4)),function() original,info)
  expect_identical(selected$theta,original$theta)
  expect_false(selected$converged)
  expect_true(selected$fitting$marker_switch$reverted)
})

test_that("the retained small-N oracle witness reverts to the original model", {
  .marker_oracle(); d <- .marker_data(2L, 10L)
  # Seed 2 exhausts both searches. Compare the reversion effects, since
  # non-converged optimizer endpoints are not a fitting-rule contract.
  attempt_counts <- integer()
  native_run <- .marker_run
  local_mocked_bindings(.marker_run = function(run) {
    result <- native_run(run)
    attempt_counts <<- c(attempt_counts, length(result$fitting$attempts))
    result
  })
  oracle_attempts <- 0L
  oracle_estimate <- getFromNamespace("lav_model_est", "lavaan")
  local_mocked_bindings(lav_model_est = function(...) {
    oracle_attempts <<- oracle_attempts + 1L
    oracle_estimate(...)
  }, .package = "lavaan")
  notes <- character()
  lv <- withCallingHandlers(lavaan::cfa("f =~ x1+x2+x3+x4",d,
    meanstructure=TRUE,fixed.x=FALSE,se="none",test="none"),
    warning=function(w) invokeRestart("muffleWarning"),
    message=function(m) {notes <<- c(notes,conditionMessage(m));invokeRestart("muffleMessage")})
  expect_true(any(grepl("reverting to",notes)))
  fit <- suppressWarnings(fit_model(model_spec("f =~ x1+x2+x3+x4",
    meanstructure=TRUE,fixed_x=FALSE),d,options=list(preset="lavaan-0.7.2")))
  expect_true(fit$fitting$marker_switch$reverted)
  expect_false(fit$converged)
  expect_false(lavaan::lavInspect(lv,"converged"))
  expect_equal(attempt_counts, c(4L, 4L))
  expect_equal(oracle_attempts, sum(attempt_counts))
  expect_equal(length(fit$fitting$attempts), 4L)
  expect_equal(fit$model$partable, fit$requested_model$partable)
  original <- lavaan::lavaanify("f =~ x1+x2+x3+x4", auto=TRUE,
    meanstructure=TRUE, fixed.x=FALSE)
  key <- function(p) paste(p$lhs,p$op,p$rhs,p$group)
  ix <- match(key(fit$partable),key(original))
  expect_equal(fit$partable$free, original$free[ix])
  expect_equal(fit$partable$ustart, original$ustart[ix])
})
