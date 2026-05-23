library(magmaan)
stopifnot(requireNamespace("jsonlite", quietly = TRUE))

# Frontier auto-identification primitives: marker <-> std_lv swap with a
# structural admissibility check and a backconvert pass for parameter
# estimates. Composes manually; the pattern is:
#
#   marker_spec <- model_spec(syntax, std_lv = FALSE, auto_fix_first = TRUE)
#   std_lv_spec <- model_spec(syntax, std_lv = TRUE,  auto_fix_first = FALSE)
#   check <- magmaan_core$frontier_is_std_lv_admissible(marker_spec, std_lv_spec)
#   if (check$admissible) {
#     fit <- fit_ml(std_lv_spec, data)
#     marker_est <- magmaan_core$frontier_backconvert_std_lv_to_marker(fit,
#                                                                      marker_spec$partable)
#   } else {
#     fit <- fit_ml(marker_spec, data)
#   }

# --- Case 1: clean 3-factor CFA from the textbook corpus ------------------

# geiser::cfa_three_factor — 3 latents with 2 indicators each, validated as
# admissible in the latent_metric_identification experiment.
ref_path <- system.file("..", "..", "tests", "fixtures", "geiser",
                        "gls_reference.json", package = "magmaan")
if (!file.exists(ref_path)) {
  # Running from the repo (the more common case during package development).
  ref_path <- file.path("tests", "fixtures", "geiser", "gls_reference.json")
}
stopifnot(file.exists(ref_path))
ref <- jsonlite::read_json(ref_path, simplifyVector = FALSE)
case <- Filter(function(x) identical(x$id, "cfa_three_factor"),
               ref$cases)[[1L]]
cfa_syntax <- case$model
S <- do.call(rbind, lapply(case$sample_cov,
                            function(r) as.numeric(unlist(r))))
n <- as.integer(case$n_obs)
samp <- list(S = list(S), nobs = n)

marker_spec <- model_spec(cfa_syntax, std_lv = FALSE, auto_fix_first = TRUE)
std_lv_spec <- model_spec(cfa_syntax, std_lv = TRUE,  auto_fix_first = FALSE)

check <- magmaan_core$frontier_is_std_lv_admissible(marker_spec, std_lv_spec)
stopifnot(isTRUE(check$admissible))

marker_fit <- magmaan_core$fit_ml(marker_spec, samp)
std_lv_fit <- magmaan_core$fit_ml(std_lv_spec, samp)
stopifnot(isTRUE(marker_fit$converged), isTRUE(std_lv_fit$converged))

backed <- magmaan_core$frontier_backconvert_std_lv_to_marker(
  std_lv_fit, marker_spec$partable
)

# Align backconverted estimates (in std_lv partable order) to marker partable.
mkey <- paste(marker_fit$partable$lhs, marker_fit$partable$op,
              marker_fit$partable$rhs)
skey <- paste(std_lv_fit$partable$lhs, std_lv_fit$partable$op,
              std_lv_fit$partable$rhs)
ord <- match(mkey, skey)
diffs <- marker_fit$partable$est - backed[ord]
max_diff <- max(abs(diffs), na.rm = TRUE)
cat(sprintf("CFA: max |marker_est - backconvert(std_lv_est)| = %.3e\n",
            max_diff))
stopifnot(max_diff < 1e-3)

# --- Case 2: longitudinal weak-invariance model — must be rejected --------

invariance_syntax <- "
  f1 =~ y1 + eq*y2 + eq*y3
  f2 =~ y4 + eq*y5 + eq*y6
"

m2 <- model_spec(invariance_syntax, std_lv = FALSE, auto_fix_first = TRUE)
s2 <- model_spec(invariance_syntax, std_lv = TRUE,  auto_fix_first = FALSE)
check2 <- magmaan_core$frontier_is_std_lv_admissible(m2, s2)
stopifnot(isFALSE(check2$admissible))
stopifnot(grepl("duplicated label", check2$reason, fixed = TRUE) ||
          grepl("==", check2$reason, fixed = TRUE))
cat(sprintf("invariance model rejected: %s\n", check2$reason))

# --- Case 3: latent variance fixed by user — must be rejected -------------

fixed_var_syntax <- "
  f =~ y1 + y2 + y3
  f ~~ 1*f
"
m3 <- model_spec(fixed_var_syntax, std_lv = FALSE, auto_fix_first = TRUE)
s3 <- model_spec(fixed_var_syntax, std_lv = TRUE,  auto_fix_first = FALSE)
check3 <- magmaan_core$frontier_is_std_lv_admissible(m3, s3)
stopifnot(isFALSE(check3$admissible))
cat(sprintf("user-fixed variance rejected: %s\n", check3$reason))

# --- Frontier registration ------------------------------------------------

stopifnot("frontier_is_std_lv_admissible" %in%
          attr(magmaan_core, "groups")$frontier)
stopifnot("frontier_backconvert_std_lv_to_marker" %in%
          attr(magmaan_core, "groups")$frontier)
stopifnot("frontier_fit_ml_auto_identification" %in%
          attr(magmaan_core, "groups")$frontier)

# --- Case 4: wrapper end-to-end, SE round-trip ---------------------------

wrapped <- magmaan_core$frontier_fit_ml_auto_identification(cfa_syntax, samp)
stopifnot(identical(wrapped$auto_identification$path, "std_lv_swap"))
stopifnot(isTRUE(wrapped$converged))

# Estimates from the wrapped (std_lv-fit + backconverted) result must match
# a direct marker fit.
wrap_diff <- max(abs(wrapped$partable$est - marker_fit$partable$est),
                 na.rm = TRUE)
cat(sprintf("wrapper estimates vs direct marker: max diff = %.3e\n",
            wrap_diff))
stopifnot(wrap_diff < 1e-3)

# SE round-trip: information + vcov computed on the wrapped fit should match
# the corresponding quantities on a direct marker fit (same partable + same
# theta + same data => same Hessian).
info_wrapped <- magmaan_core$inference_information_observed_analytic(wrapped)
info_marker <- magmaan_core$inference_information_observed_analytic(marker_fit)
info_diff <- max(abs(info_wrapped - info_marker), na.rm = TRUE)
cat(sprintf("wrapper info vs direct marker info: max diff = %.3e\n",
            info_diff))
stopifnot(info_diff < 1e-3)

vcov_wrapped <- magmaan_core$inference_vcov_fit(info_wrapped, wrapped)
vcov_marker <- magmaan_core$inference_vcov_fit(info_marker, marker_fit)
se_wrapped <- magmaan_core$inference_se(vcov_wrapped)
se_marker <- magmaan_core$inference_se(vcov_marker)
se_diff <- max(abs(se_wrapped - se_marker), na.rm = TRUE)
cat(sprintf("wrapper SE vs direct marker SE:     max diff = %.3e\n",
            se_diff))
stopifnot(se_diff < 1e-4)

# Wrapper on an inadmissible model: should silently fall back to marker.
inadmissible_wrapped <- magmaan_core$frontier_fit_ml_auto_identification(
  invariance_syntax,
  list(S = list(diag(6) + 0.3),
       nobs = 100L)
)
# (We don't assert convergence; the random sample stats above are bogus —
# we only care that the wrapper recognized the model as inadmissible and
# took the marker_direct path.)
stopifnot(identical(inadmissible_wrapped$auto_identification$path,
                    "marker_direct"))
stopifnot(isFALSE(inadmissible_wrapped$auto_identification$admissible))
cat(sprintf("inadmissible model -> path: %s (reason: %s)\n",
            inadmissible_wrapped$auto_identification$path,
            inadmissible_wrapped$auto_identification$reason))

# --- Headline timing ------------------------------------------------------

case_b <- Filter(function(x) identical(x$id,
                                       "latent_ar_cross_lagged_extended"),
                 ref$cases)[[1L]]
Sb <- do.call(rbind, lapply(case_b$sample_cov,
                             function(r) as.numeric(unlist(r))))
samp_b <- list(S = list(Sb), nobs = as.integer(case_b$n_obs))
mb <- model_spec(case_b$model, std_lv = FALSE, auto_fix_first = TRUE)
sb <- model_spec(case_b$model, std_lv = TRUE,  auto_fix_first = FALSE)
stopifnot(isTRUE(magmaan_core$frontier_is_std_lv_admissible(mb, sb)$admissible))

reps <- 50L
t_marker <- replicate(reps, {
  st <- Sys.time()
  magmaan_core$fit_ml(mb, samp_b)
  as.numeric(difftime(Sys.time(), st, units = "secs"))
})
t_std_lv <- replicate(reps, {
  st <- Sys.time()
  f <- magmaan_core$fit_ml(sb, samp_b)
  magmaan_core$frontier_backconvert_std_lv_to_marker(f, mb$partable)
  as.numeric(difftime(Sys.time(), st, units = "secs"))
})
cat(sprintf("latent_ar_cross_lagged_extended (textbook corpus), %d reps:\n",
            reps))
cat(sprintf("  marker direct           median = %6.0f us\n",
            stats::median(t_marker) * 1e6))
cat(sprintf("  std_lv + backconvert    median = %6.0f us\n",
            stats::median(t_std_lv) * 1e6))
cat(sprintf("  pipeline ratio          %.3f\n",
            stats::median(t_std_lv) / stats::median(t_marker)))
