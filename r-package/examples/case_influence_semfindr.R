# Case-level influence diagnostics: parity against semfindr (the oracle).
#
# semfindr (Cheung & Lai, 2026) is the reference implementation of the
# Pek & MacCallum (2011) leave-one-out framework. This script fits the same
# models in magmaan and in lavaan, runs both engines, and checks that magmaan's
# case_rerun()/est_change_raw()/est_change()/fit_measures_change()/
# mahalanobis_rerun() match semfindr to a tight tolerance.
#
# semfindr is not a magmaan dependency; skip cleanly if it is absent.
if (!requireNamespace("semfindr", quietly = TRUE) ||
    !requireNamespace("lavaan", quietly = TRUE)) {
  message("case_influence_semfindr.R: semfindr/lavaan not installed; skipping.")
  quit(save = "no", status = 0)
}

suppressMessages({library(magmaan); library(lavaan); library(semfindr)})

# Compare two case-keyed matrices on their shared rows/cols.
agree <- function(tag, m, s, tol) {
  rc <- intersect(rownames(m), rownames(s))
  cc <- intersect(colnames(m), colnames(s))
  if (!length(cc)) stop(tag, ": no shared columns (",
                        paste(colnames(m), collapse = ","), " vs ",
                        paste(colnames(s), collapse = ","), ")")
  d <- max(abs(m[rc, cc, drop = FALSE] - s[rc, cc, drop = FALSE]), na.rm = TRUE)
  cat(sprintf("  %-26s cols %2d  max|diff| = %.2e\n", tag, length(cc), d))
  if (!is.finite(d) || d > tol) {
    stop(tag, ": max|diff| ", signif(d, 4), " exceeds tol ", tol)
  }
  invisible(d)
}

check_model <- function(label, mod, dat, estimator = "ML",
                         fm_measures = c("chisq", "cfi", "rmsea", "tli"),
                         cases = 1:25) {
  cat("\n##", label, "(n =", nrow(dat), ")\n")
  lfit <- lavaan::sem(mod, dat)
  mfit <- magmaan::magmaan(mod, dat, estimator = estimator)

  rr <- semfindr::lavaan_rerun(lfit, to_rerun = cases)
  cr <- magmaan::case_rerun(mfit, to_rerun = cases)
  stopifnot(sum(cr$converged) == length(cases))

  agree("est_change_raw", magmaan::est_change_raw(cr),
        semfindr::est_change_raw(rr), tol = 1e-4)
  agree("est_change (DFTHETAS+gcd)", magmaan::est_change(cr),
        semfindr::est_change(rr), tol = 1e-3)
  agree("fit_measures_change",
        magmaan::fit_measures_change(cr, fit_measures = fm_measures),
        semfindr::fit_measures_change(rr, fit_measures = fm_measures), tol = 1e-3)

  s_md <- as.matrix(semfindr::mahalanobis_rerun(lfit))
  m_md <- magmaan::mahalanobis_rerun(mfit)
  colnames(s_md) <- colnames(m_md)            # semfindr labels it differently
  agree("mahalanobis_rerun", m_md, s_md, tol = 1e-6)

  # Approximate one-step engine (no refit; all cases).
  agree("est_change_raw_approx", magmaan::est_change_raw_approx(mfit),
        semfindr::est_change_raw_approx(lfit), tol = 1e-6)
  agree("est_change_approx", magmaan::est_change_approx(mfit),
        semfindr::est_change_approx(lfit), tol = 1e-5)
}

## CFA: no exogenous predictors, so CFI/TLI baselines match lavaan exactly.
mod_cfa <- "f1 =~ x1 + x2 + x3
            f2 =~ x4 + x5 + x6
            f1 ~~ f2"
check_model("CFA / HolzingerSwineford1939", mod_cfa,
            lavaan::HolzingerSwineford1939[, paste0("x", 1:6)])

## Path model with exogenous predictors (fixed.x). chisq and rmsea match
## exactly; cfi/tli are gated OUT here because magmaan's baseline (independence)
## model does not yet free the exogenous covariance the way lavaan does under
## fixed.x, so baseline.df differs (a pre-existing fit-index gap, not a
## case-influence issue). See dev notes.
mod_pa <- "m1 ~ a1 * iv1 + a2 * iv2
           dv ~ b * m1
           a1b := a1 * b
           a2b := a2 * b"
check_model("Path / pa_dat", mod_pa, semfindr::pa_dat,
            fm_measures = c("chisq", "rmsea"))

cat("\ncase_influence_semfindr.R: all parity checks passed.\n")
