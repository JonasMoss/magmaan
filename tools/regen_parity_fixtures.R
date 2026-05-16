#!/usr/bin/env Rscript
# Generate self-contained lavaan-parity fixtures for the real-data benchmark
# cases. Each case gets:
#
#   tests/fixtures/parity/<case>/reference.json  -- lavaan oracle outputs
#   tests/fixtures/parity/<case>/data.json       -- committed raw data, when the
#                                                   dataset is redistributable
#
# The C++ golden test tests/golden/lavaan_parity_golden_test.cpp consumes these
# with no R at run time. This is a manual developer step; CI never runs R. It
# is the sibling of tools/regen_oracle.R (synthetic corpus fixtures) and
# supersedes the earlier convergence-only fixture generator.
#
# Usage:
#   Rscript tools/regen_parity_fixtures.R                       # all cases
#   Rscript tools/regen_parity_fixtures.R hs_3factor_cfa ...    # named cases
#
# Case definitions and prepared-data loading are reused from benchmarks/r/.
# Run the benchmark data pipeline first if prepared data is missing:
#   Rscript benchmarks/r/prepare_case.R all

suppressMessages({
  library(lavaan)
  library(magmaan)
  library(jsonlite)
})

repo <- normalizePath(file.path(dirname(normalizePath(sub(
  "^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))), ".."))

source(file.path(repo, "benchmarks", "r", "common.R"))
source(file.path(repo, "benchmarks", "r", "cases.R"))

# common.R derives benchmark paths from the invoked script's location; when it
# is sourced (not run) that points at this file. Pin it back at benchmarks/r/.
bench_script_dir <- function() file.path(repo, "benchmarks", "r")

out_root <- file.path(repo, "tests", "fixtures", "parity")

# Whether a case's raw observations may be committed to the repository. The
# lavaan and psychTools datasets are GPL and ship with those packages, so the
# derived per-case data frame is freely redistributable. The Mplus User's Guide
# example data is not cleared -- it gets reference.json (summary statistics)
# only, and the parity test still covers it for ML estimates via sample_cov.
data_redistributable <- c(
  hs_3factor_cfa       = TRUE,   # lavaan::HolzingerSwineford1939 (GPL-3)
  bollen_democracy_sem = TRUE,   # lavaan::PoliticalDemocracy     (GPL-3)
  demo_growth_linear   = TRUE,   # lavaan::Demo.growth            (GPL-3)
  bfi_5factor          = TRUE,   # psychTools::bfi                (GPL-2 | GPL-3)
  mplus_ex5_1          = FALSE   # Mplus User's Guide example -- licensing TBD
)

# lavaan fit-measure names mirrored by the C++ parity assertions.
fit_measure_keys <- c(
  npar = "npar", logl = "logl", unrestricted_logl = "unrestricted.logl",
  aic = "aic", bic = "bic", bic2 = "bic2",
  cfi = "cfi", tli = "tli", rmsea = "rmsea",
  rmsea_ci_lower = "rmsea.ci.lower", rmsea_ci_upper = "rmsea.ci.upper",
  rmsea_pvalue = "rmsea.pvalue", srmr = "srmr")

fit_case <- function(case, model, data) {
  fun <- switch(
    case$lavaan_function %||% "sem",
    cfa = lavaan::cfa, growth = lavaan::growth, sem = lavaan::sem,
    lavaan::sem)
  fun(model = model, data = data,
      estimator = case$estimator %||% "ML",
      meanstructure = isTRUE(case$meanstructure),
      se = "standard", test = "standard", missing = "listwise")
}

emit <- function(id) {
  case <- get_case(id)
  if (!isTRUE(case$supported_now)) {
    stop("case is not supported_now: ", id, call. = FALSE)
  }
  model <- read_model(id)
  raw_data <- read_prepared_data(id)
  fit <- fit_case(case, model, raw_data)

  ov <- lavaan::lavNames(fit, type = "ov")
  meanstructure <- isTRUE(case$meanstructure)

  ## Complete-case rows in observed-variable order: exactly what lavaan fit
  ## under missing = "listwise", and what the C++ complete-data ML path sees.
  data <- raw_data[stats::complete.cases(raw_data[, ov, drop = FALSE]),
                   ov, drop = FALSE]

  ## lavaan's free parameters, and magmaan's free-parameter order -- the order
  ## the C++ theta vector uses. Aligned via the group|op|lhs|rhs key.
  lpt <- lavaan::parTable(fit)
  lfree <- lpt[lpt$free > 0, , drop = FALSE]
  mspec <- magmaan::model_spec(model, auto_cov_y = TRUE,
                               meanstructure = meanstructure)
  mfree <- mspec$partable[mspec$partable$free > 0, , drop = FALSE]
  mfree <- mfree[order(mfree$free), , drop = FALSE]

  key <- function(d) paste(d$group, d$op, d$lhs, d$rhs, sep = "\r")
  idx <- match(key(mfree), key(lfree))
  aligned <- nrow(mfree) == nrow(lfree) && !anyNA(idx)
  if (!aligned) {
    warning(id, ": magmaan free set does not align with lavaan ",
            "(magmaan ", nrow(mfree), " vs lavaan ", nrow(lfree),
            " free params); fixture marked magmaan_aligned = false",
            call. = FALSE)
  }

  S <- lavaan::lavInspect(fit, "sampstat")$cov
  mean_vec <- if (meanstructure) lavaan::lavInspect(fit, "sampstat")$mean else NULL
  fm <- lavaan::fitMeasures(fit)
  optim <- lavaan::lavInspect(fit, "optim")

  pull <- function(k) if (k %in% names(fm)) unname(fm[[k]]) else NA_real_
  fit_measures <- lapply(fit_measure_keys, pull)

  payload <- list(
    case_id = id,
    model = model,
    lavaan_version = as.character(utils::packageVersion("lavaan")),
    lavaan_function = case$lavaan_function %||% "sem",
    estimator = case$estimator %||% "ML",
    meanstructure = meanstructure,
    auto_cov_y = TRUE,
    n_groups = 1L,
    n_obs = unname(lavaan::lavInspect(fit, "nobs")),
    ov_names = ov,
    n_free = nrow(mfree),
    magmaan_aligned = aligned,
    sample_cov = list(names = colnames(S), values = unname(as.matrix(S))),
    sample_mean = if (meanstructure)
      list(names = names(mean_vec), values = unname(mean_vec)) else NULL,
    lavaan_fmin = pull("fmin"),
    chisq = pull("chisq"),
    df = as.integer(pull("df")),
    pvalue = pull("pvalue"),
    fit_measures = fit_measures)

  if (aligned) {
    payload$param_lhs <- mfree$lhs
    payload$param_op <- mfree$op
    payload$param_rhs <- mfree$rhs
    payload$theta_hat <- unname(lfree$est[idx])
    payload$se <- unname(lfree$se[idx])
    payload$theta_start_lavaan <- unname(lfree$start[idx])
  }

  case_dir <- file.path(out_root, id)
  dir.create(case_dir, recursive = TRUE, showWarnings = FALSE)
  ref_path <- file.path(case_dir, "reference.json")
  write_json(payload, ref_path, pretty = TRUE, auto_unbox = TRUE,
             null = "null", na = "null", digits = NA)

  data_note <- "skipped (not cleared for redistribution)"
  if (isTRUE(data_redistributable[[id]])) {
    data_payload <- list(
      case_id = id,
      ov_names = ov,
      ## One block; mask omitted -> the C++ loader treats X as complete data,
      ## which sample_stats_from_raw() requires.
      raw = list(list(X = unname(as.matrix(data)))))
    data_path <- file.path(case_dir, "data.json")
    write_json(data_payload, data_path, pretty = TRUE, auto_unbox = TRUE,
               digits = NA)
    data_note <- sprintf("data.json (%d x %d)", nrow(data), ncol(data))
  }

  cat(sprintf("wrote %-22s n=%-5d n_free=%-3d aligned=%-5s %s\n",
              id, payload$n_obs, payload$n_free, tolower(as.character(aligned)),
              data_note))
}

args <- commandArgs(trailingOnly = TRUE)
ids <- if (length(args) == 0) {
  names(Filter(function(x) isTRUE(x$supported_now), benchmark_cases))
} else {
  args
}

for (id in ids) emit(id)
