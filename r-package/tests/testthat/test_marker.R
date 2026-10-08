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
  if (isTRUE(fit$ordinal) || isTRUE(fit$mixed_ordinal))
    expect_equal(abs(fit$fmin - as.numeric(lv@optim$fx)), 0, tolerance = 1e-9)
  else expect_equal(fit$fmin, as.numeric(lv@optim$fx), tolerance = 1e-9)
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
test_that("marker validation and unsupported models are explicit", {
  .marker_oracle(); d <- .marker_data()
  spec <- model_spec("f =~ x1+x2+x3+x4",meanstructure=TRUE,fixed_x=FALSE)
  expect_error(fit_model(spec,d,options=list(marker="unknown")),"marker")
  expect_error(fit_model(spec,d,options=list(marker=NA_character_)),"nonmissing")
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

.marker_route_case <- function(route, grouped = FALSE) {
  .marker_oracle()
  d <- .marker_data()
  if (grouped) {
    d <- rbind(d, .marker_data(1292L))
    d$g <- rep(c("A", "B"), each = 600L)
  }
  ordered <- if (route == "ordinal") paste0("x", 1:4) else if (route == "mixed") paste0("x", 1:3) else character()
  for (v in ordered) d[[v]] <- ordered(cut(d[[v]], c(-Inf, -.4, .4, Inf)))
  if (route == "FIML") {
    set.seed(1293L)
    for (v in paste0("x", 1:4)) d[sample(nrow(d), 60L), v] <- NA_real_
  }
  spec <- model_spec("f =~ x1+x2+x3+x4", meanstructure = TRUE, fixed_x = FALSE,
    ordered = ordered, group = if (grouped) "g" else "", group_labels = if (grouped) c("A", "B") else NULL)
  oracle <- function(off = FALSE) suppressWarnings(lavaan::cfa(spec$syntax, d,
    meanstructure = TRUE, fixed.x = FALSE, ordered = ordered,
    group = if (grouped) "g" else NULL,
    estimator = if (route == "FIML") "ML" else "DWLS",
    missing = if (route == "FIML") "ml" else "listwise",
    bad.marker.crit = if (off) 0 else .1))
  oracle_h1_converged <- NULL
  if (route == "FIML") {
    em <- getFromNamespace("lav_em_squarem", "lavaan")
    local_mocked_bindings(lav_em_squarem = function(...) {
      result <- em(...)
      oracle_h1_converged <<- result$converged
      result
    }, .package = "lavaan")
  }
  lv <- oracle()
  fit <- suppressWarnings(fit_model(spec, d, estimator = if (route == "FIML") "FIML" else "DWLS",
    options = list(preset = "lavaan-0.7.2")))
  if (route == "FIML") {
    # Stalled EM endpoints are outside the H1 compatibility contract. Only
    # compare switch decisions when both saturated fits have converged.
    expect_true(all(fit$fitting$h1$converged))
    expect_true(isTRUE(oracle_h1_converged))
  }
  .marker_compare(fit, lv)
  info <- fit$fitting$marker_switch
  expect_equal(info$old, "x1"); expect_false(info$reverted)
  adapt <- lavaan:::lav_pt_marker_adapt(lavaan::lavaanify(spec$syntax, auto.fix.first = TRUE), lv@h1, lavdata = lv@Data)
  expect_equal(info$new, adapt$info$new)
  expect_equal(info$r_old, adapt$info$r.old); expect_equal(info$r_new, adapt$info$r.new)
  off <- suppressWarnings(fit_model(spec, d, estimator = if (route == "FIML") "FIML" else "DWLS",
    options = list(preset = "lavaan-0.7.2", marker = "default")))
  oracle_off <- oracle(TRUE)
  if (grouped) {
    # This retained weak-marker control exhausts both searches. As in the
    # small-N reversion witness above, failed optimizer endpoints are outside
    # the estimate-parity contract; retain the original marker and verdict.
    expect_false(off$converged)
    expect_false(lavaan::lavInspect(oracle_off, "converged"))
    expect_length(off$fitting$attempts, 4L)
    expect_equal(off$partable$rhs[off$partable$op == "=~" & off$partable$free == 0L], rep("x1", 2L))
  } else .marker_compare(off, oracle_off)
  expect_equal(nrow(off$fitting$marker_switch), 0L)
  model <- prepare_model(spec, prototype = if (length(ordered)) d else NULL)
  data <- prepare_data(model, d, kind = if (route == "FIML") "raw" else NULL)
  prepared <- suppressWarnings(estimate(model, data, options = list(preset = "lavaan-0.7.2")))
  .marker_compare(prepared, lv)
  expect_equal(prepared$fitting$marker_switch, info)
  direct <- if (route == "FIML") fit_fiml(spec, d, options = list(preset = "lavaan-0.7.2")) else {
    stats <- if (route == "ordinal") data_ordinal_stats_from_df(d, spec) else data_mixed_ordinal_stats_from_df(d, spec)
    if (route == "ordinal") fit_dwls_ordinal(spec, stats, options = list(preset = "lavaan-0.7.2")) else
      fit_dwls_mixed_ordinal(spec, stats, options = list(preset = "lavaan-0.7.2"))
  }
  .marker_compare(direct, lv)
  expect_equal(direct$fitting$marker_switch, info)
}
test_that("FIML marker adaptation reads the EM H1 covariance", {
  .marker_route_case("FIML")
})
test_that("ordinal DWLS marker adaptation reads polychoric correlations", {
  .marker_route_case("ordinal")
})
test_that("mixed DWLS marker adaptation reads mixed H1 covariances", {
  .marker_route_case("mixed")
})
test_that("grouped ordinal DWLS averages the H1 item-rest correlations", {
  .marker_route_case("ordinal", grouped = TRUE)
})

test_that("route fits without a weak marker keep their original coordinates", {
  .marker_oracle()
  for (route in c("FIML", "ordinal", "mixed")) {
    d <- .marker_data()
    d$x1 <- d$x2 + .3*d$x1
    ordered <- if (route == "ordinal") names(d) else if (route == "mixed") names(d)[1:3] else character()
    for (v in ordered) d[[v]] <- ordered(cut(d[[v]], c(-Inf, -.4, .4, Inf)))
    if (route == "FIML") d[seq(1L, 600L, 10L), "x4"] <- NA_real_
    spec <- model_spec("f =~ x1+x2+x3+x4", meanstructure = TRUE, fixed_x = FALSE, ordered = ordered)
    fit <- fit_model(spec, d, estimator = if (route == "FIML") "FIML" else "DWLS", options = list(preset = "lavaan-0.7.2"))
    off <- fit_model(spec, d, estimator = if (route == "FIML") "FIML" else "DWLS", options = list(preset = "lavaan-0.7.2", marker = "default"))
    expect_equal(nrow(fit$fitting$marker_switch), 0L)
    expect_equal(fit$theta, off$theta)
    expect_equal(fit$fmin, off$fmin)
    expect_equal(fit$converged, off$converged)
  }
})

test_that("marker adaptation refuses conditional-x residual H1 matrices explicitly", {
  d <- .marker_data()
  d$z <- seq_len(nrow(d)) / nrow(d)
  for (v in paste0("x", 1:4)) d[[v]] <- ordered(cut(d[[v]], c(-Inf, -.4, .4, Inf)))
  spec <- model_spec("f =~ x1+x2+x3+x4\nf ~ z", ordered = paste0("x", 1:4),
    meanstructure = TRUE, fixed_x = TRUE)
  expect_error(fit_model(spec, d, estimator = "DWLS", options = list(preset = "lavaan-0.7.2")),
    class = "magmaan_unsupported_model")
})
