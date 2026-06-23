#!/usr/bin/env Rscript
# Generate frozen semfindr-parity fixtures for the case-influence suite.
#
# semfindr (Cheung & Lai, 2026) is the reference implementation of the
# Pek & MacCallum (2011) leave-one-out case-influence framework. Each case here
# fits a model with lavaan, runs semfindr's exact (leave-one-out) engine, and
# freezes its outputs alongside the raw data so a test can re-fit in magmaan and
# compare without semfindr installed:
#
#   tests/fixtures/case_influence/<id>.json
#       model, estimator, raw data, the rerun case set, and semfindr's
#       est_change_raw / est_change (incl. gcd) / fit_measures_change /
#       mahalanobis_rerun outputs.
#
# The live R-side parity check is r-package/examples/case_influence_semfindr.R
# (run by `just r-examples`); this generator produces the frozen artifact and
# is a manual developer step. CI never runs R.
#
# Usage:
#   Rscript tests/tools/regen_semfindr_fixtures.R              # all cases
#   Rscript tests/tools/regen_semfindr_fixtures.R cfa_hs ...   # named cases
#
# Requires the pinned semfindr (tests/fixtures/semfindr_version.txt) and the
# pinned lavaan (tests/fixtures/lavaan_version.txt).

suppressMessages({
  library(lavaan)
  library(semfindr)
  library(jsonlite)
})

repo <- normalizePath(file.path(dirname(normalizePath(sub(
  "^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))), "..", ".."))

# --- version pins ----------------------------------------------------------
check_pin <- function(pkg, file) {
  pin_path <- file.path(repo, "tests", "fixtures", file)
  pinned <- trimws(readLines(pin_path, warn = FALSE)[1])
  inst <- as.character(utils::packageVersion(pkg))
  norm <- function(s) gsub("-", ".", s, fixed = TRUE)
  if (!identical(norm(pinned), norm(inst))) {
    stop(sprintf("%s version mismatch: pinned %s, installed %s.", pkg, pinned, inst),
         call. = FALSE)
  }
  inst
}
semfindr_ver <- check_pin("semfindr", "semfindr_version.txt")
lavaan_ver   <- check_pin("lavaan", "lavaan_version.txt")

out_dir <- file.path(repo, "tests", "fixtures", "case_influence")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# === case table ============================================================
# Each case: a model, an estimator, the data, and the rerun case set. The CFA
# has no exogenous predictors (its CFI/TLI baseline matches lavaan exactly); the
# path model exercises regressions with labels and defined parameters.
cases <- list(
  cfa_hs = list(
    model = "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6\nf1 ~~ f2",
    estimator = "ML",
    data = lavaan::HolzingerSwineford1939[, paste0("x", 1:6)],
    to_rerun = 1:25,
    fit_measures = c("chisq", "cfi", "rmsea", "tli")),

  path_pa = list(
    model = "m1 ~ a1 * iv1 + a2 * iv2\ndv ~ b * m1\na1b := a1 * b\na2b := a2 * b",
    estimator = "ML",
    data = semfindr::pa_dat,
    to_rerun = 1:25,
    # cfi/tli omitted: magmaan's fixed.x baseline df differs from lavaan's.
    fit_measures = c("chisq", "rmsea"))
)

# === serialization helpers =================================================
mat_json <- function(m) {
  m <- as.matrix(m)
  list(
    rownames = I(rownames(m)),
    colnames = I(colnames(m)),
    values   = I(lapply(seq_len(nrow(m)), function(i) unname(as.numeric(m[i, ]))))
  )
}

build_one <- function(id, case) {
  dat <- as.data.frame(case$data)
  ov <- colnames(dat)
  lfit <- lavaan::sem(case$model, dat)
  rr <- semfindr::lavaan_rerun(lfit, to_rerun = case$to_rerun)

  payload <- list(
    `_meta` = list(
      format_version = 1L,
      fixture_kind = "case_influence",
      corpus_id = id,
      tool = "semfindr",
      semfindr_version = semfindr_ver,
      lavaan_version = lavaan_ver),
    model = case$model,
    estimator = case$estimator,
    ov_names = I(ov),
    to_rerun = I(as.integer(case$to_rerun)),
    fit_measures = I(case$fit_measures),
    data = I(lapply(seq_len(nrow(dat)), function(i) unname(as.numeric(dat[i, ])))),
    est_change_raw = mat_json(semfindr::est_change_raw(rr)),
    est_change = mat_json(semfindr::est_change(rr)),
    fit_measures_change = mat_json(
      semfindr::fit_measures_change(rr, fit_measures = case$fit_measures)),
    mahalanobis = mat_json(as.matrix(semfindr::mahalanobis_rerun(lfit)))
  )
  out_path <- file.path(out_dir, paste0(id, ".json"))
  write_json(payload, out_path, pretty = TRUE, auto_unbox = TRUE,
             null = "null", na = "null", digits = NA)
  cat("wrote", out_path, "\n")
}

# === main ==================================================================
args <- commandArgs(trailingOnly = TRUE)
ids <- if (length(args)) args else names(cases)
for (id in ids) {
  if (is.null(cases[[id]])) stop("unknown case: ", id, call. = FALSE)
  build_one(id, cases[[id]])
}
