# Global goodness-of-fit arms for one complete-data ML fit. Every arm tests the
# fitted model against the saturated model and is calibrated with SB (mean
# scaling) and PEBA4 from its own robust spectrum; std is the uncorrected
# chi-square reference.
#
# Score arms differ in the two information choices of the projected score:
#   sensitivity - the information used to project the score off the model
#                 tangent (expected Fisher, or the observed H0 Hessian);
#   metric      - the information in the score quadratic ("the weight in the
#                 score"): expected at H0, observed at H0, or observed at the
#                 saturated H1 optimum.
# LR arms share the statistic n F and differ only in the U used for the UGamma
# spectrum: expected information (magmaan's current policy geometry) or
# observed information.

SCORE_ARMS <- list(
  score_EE = c(sensitivity = "expected", metric = "expected"),
  score_OE = c(sensitivity = "observed", metric = "expected"),
  score_OO = c(sensitivity = "observed", metric = "observed"),
  score_OH = c(sensitivity = "observed", metric = "observed-h1"),
  score_EH = c(sensitivity = "expected", metric = "observed-h1")
)

.arm_row <- function(arm, statistic, df, p_std, p_sb, p_peba4, n_negative = 0L,
                     error = NA_character_) {
  data.frame(arm = arm, statistic = statistic, df = df, p_std = p_std,
             p_sb = p_sb, p_peba4 = p_peba4, n_negative = n_negative,
             error = error, stringsAsFactors = FALSE)
}

.failed <- function(arm, e) .arm_row(arm, NA_real_, NA_integer_, NA_real_,
                                     NA_real_, NA_real_, NA_integer_,
                                     conditionMessage(e))

.from_spectrum <- function(arm, statistic, df, ev) {
  fmg <- function(method, param) {
    magmaanlab:::infer_fmg_test(statistic, as.integer(df), ev, method = method,
                                param = param, truncate_negative = TRUE)$p_value
  }
  .arm_row(arm, statistic, df,
           p_std = stats::pchisq(statistic, df, lower.tail = FALSE),
           p_sb = fmg("sb", 0), p_peba4 = fmg("peba", 4),
           n_negative = sum(utils::head(sort(ev, decreasing = TRUE), df) < 0))
}

# All arms for one fitted model. `X` is the raw data matrix with model
# variable names as column names.
fit_arms <- function(fit, X) {
  ctx <- magmaanlab::prepare_inference(fit)
  rows <- list()

  lr <- magmaanlab::inference_quadratic(ctx, "lr")
  statistic <- lr$statistic
  df <- lr$df
  rows$lr_E <- tryCatch({
    ref <- magmaanlab::score_spectrum(lr)
    .from_spectrum("lr_E", statistic, df, ref$eigenvalues)
  }, error = function(e) .failed("lr_E", e))
  rows$lr_O <- tryCatch({
    core <- magmaanlab::magmaan_core
    uf <- core$robust_build_u_factor(fit, bread = "observed", moments = "structured")
    Zc <- magmaanlab:::infer_casewise_contributions(fit$partable, X)
    M <- magmaanlab:::infer_reduced_gamma_sample(uf, Zc, as.numeric(fit$nobs))
    ev <- magmaanlab:::infer_ugamma_eigenvalues(M)
    .from_spectrum("lr_O", statistic, df, ev)
  }, error = function(e) .failed("lr_O", e))

  for (arm in names(SCORE_ARMS)) {
    a <- SCORE_ARMS[[arm]]
    rows[[arm]] <- tryCatch({
      sc <- magmaanlab::score_components(ctx, sensitivity = a[["sensitivity"]],
                                         metric = a[["metric"]])
      pr <- magmaanlab::project_scores(sc)
      ref <- magmaanlab::score_spectrum(pr)
      .from_spectrum(arm, ref$statistic, ref$df, ref$eigenvalues)
    }, error = function(e) .failed(arm, e))
  }
  do.call(rbind, rows)
}
