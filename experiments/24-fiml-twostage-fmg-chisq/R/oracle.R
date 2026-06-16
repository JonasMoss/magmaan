# lavaan correctness oracle (rep 1 of each h0 cell). Anchors both estimators'
# base and scaled statistics to lavaan:
#
#   FIML LRT base   <- lavaan estimator="MLR", missing="ml" naive chisq.
#   FIML mean-scaled <- lavaan yuan.bentler with h1.information="unstructured"
#                       (the saturated-H1 convention magmaan's FIML FMG spectrum
#                       follows; validated in experiment 21). magmaan's FIML "sb"
#                       is NOT lavaan's MLR/Mplus default, which uses structured
#                       H1 + observed information + the Mplus trace approximation.
#   ML2S base        <- lavaan missing="two.stage" naive chisq (machine precision).
#
# Only the ML2S BASE is anchored to lavaan. lavaan's two.stage "scaled" test uses
# a different reference-law convention (its UGamma trace differs from magmaan's on
# incomplete data: e.g. 22.1 vs 33.0 on one MAR cell), the two-stage analogue of
# the FIML MLR structured/observed-info vs unstructured split. magmaan's two-stage
# reference law is instead validated from first principles by the consistency
# identity trace(UGamma) = E[T]: under normal data ncp_hat = mean(base) -
# mean(trace) ~ 0 for both estimators (see noncentrality.csv). So we do not gate
# magmaan against lavaan's two.stage scaling.
#
# No magmaan-internal calls; the magmaan side is the public fmg_tests() battery.

lavaan_fiml_mlr_fit <- function(syntax, df) {
  lavaan::cfa(syntax, data = df, missing = "ml", estimator = "MLR",
              meanstructure = TRUE)
}

lavaan_fiml_yb_unstructured_fit <- function(syntax, df) {
  lavaan::cfa(syntax, data = df, missing = "ml", estimator = "ML",
              test = "yuan.bentler", h1.information = "unstructured",
              meanstructure = TRUE)
}

lavaan_ml2s_fit <- function(syntax, df) {
  lavaan::sem(syntax, data = df, missing = "two.stage", estimator = "ML",
              meanstructure = TRUE)
}

lav_measures <- function(fit) {
  fm <- tryCatch(lavaan::fitMeasures(
    fit, c("chisq", "df", "pvalue", "chisq.scaled", "pvalue.scaled")),
    error = function(e) NULL)
  if (is.null(fm)) return(NULL)
  list(chisq = unname(fm["chisq"]), df = unname(fm["df"]),
       p_naive = unname(fm["pvalue"]),
       chisq_scaled = unname(fm["chisq.scaled"]),
       p_scaled = unname(fm["pvalue.scaled"]))
}

.parity_row <- function(base, metric, magmaan, lavaan) {
  data.frame(c(base, list(metric = metric, magmaan = magmaan, lavaan = lavaan,
                          abs_diff = abs(magmaan - lavaan))),
             stringsAsFactors = FALSE)
}

# Parity rows for one h0 cell. `mag` carries the magmaan-side numbers pulled from
# the rep-1 battery: FIML base + FIML "sb" p, ML2S base + ML2S "sb" p.
parity_rows_for_cell <- function(syntax, df, dist, mech, rate, mag) {
  base <- list(dist = dist, mech = mech, rate = rate)
  rows <- list()
  add <- function(r) if (!is.null(r)) rows[[length(rows) + 1L]] <<- r

  mlr <- tryCatch(lav_measures(lavaan_fiml_mlr_fit(syntax, df)),
                  error = function(e) NULL)
  if (!is.null(mlr)) {
    add(.parity_row(base, "fiml_lrt_chisq", mag$fiml_base, mlr$chisq))
  }
  yb <- tryCatch(lav_measures(lavaan_fiml_yb_unstructured_fit(syntax, df)),
                 error = function(e) NULL)
  if (!is.null(yb)) {
    add(.parity_row(base, "fiml_sb_pvalue", mag$fiml_sb_p, yb$p_scaled))
  }
  ts <- tryCatch(lav_measures(lavaan_ml2s_fit(syntax, df)),
                 error = function(e) NULL)
  if (!is.null(ts)) {
    add(.parity_row(base, "ml2s_chisq", mag$ml2s_base, ts$chisq))
  }
  rows
}
