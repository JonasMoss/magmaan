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
# Cases are grouped into four families, each fitted and serialized differently:
#   ML       complete-data normal-theory ML (the original parity tranche)
#   FIML     raw-data full-information ML over genuine missingness
#   LS       continuous ULS / GLS / WLS on complete data
#   ordinal  ordinal DWLS / WLS on integer Likert data
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
# Shared lavaan-to-JSON fixture helpers, also used by tools/regen_oracle.R.
source(file.path(repo, "benchmarks", "r", "fixture_json.R"))

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
  mplus_ex5_1          = FALSE,  # Mplus User's Guide example -- licensing TBD
  bfi_fiml             = TRUE,   # psychTools::bfi                (GPL-2 | GPL-3)
  hs_3factor_ls        = TRUE,   # lavaan::HolzingerSwineford1939 (GPL-3)
  bfi_ordinal_dwls     = TRUE    # psychTools::bfi                (GPL-2 | GPL-3)
)

# lavaan fit-measure names mirrored by the C++ parity assertions.
fit_measure_keys <- c(
  npar = "npar", logl = "logl", unrestricted_logl = "unrestricted.logl",
  aic = "aic", bic = "bic", bic2 = "bic2",
  cfi = "cfi", tli = "tli", rmsea = "rmsea",
  rmsea_ci_lower = "rmsea.ci.lower", rmsea_ci_upper = "rmsea.ci.upper",
  rmsea_pvalue = "rmsea.pvalue", srmr = "srmr")

# Load an R-package dataset (NAs retained), restricted to `vars` if given.
load_dataset <- function(package, name, vars = NULL) {
  env <- new.env(parent = emptyenv())
  utils::data(list = name, package = package, envir = env)
  df <- as.data.frame(get(name, envir = env))
  if (!is.null(vars)) df <- df[, vars, drop = FALSE]
  df
}

# magmaan's free-parameter order, and the index map from magmaan order into a
# lavaan free-parameter table, keyed by group|op|lhs|rhs.
align_magmaan_free <- function(model, lavaan_free, meanstructure,
                               auto_cov_y = TRUE) {
  mspec <- magmaan::model_spec(model, auto_cov_y = auto_cov_y,
                               meanstructure = meanstructure)
  mfree <- mspec$partable[mspec$partable$free > 0, , drop = FALSE]
  mfree <- mfree[order(mfree$free), , drop = FALSE]
  key <- function(d) paste(d$group, d$op, d$lhs, d$rhs, sep = "\r")
  idx <- match(key(mfree), key(lavaan_free))
  list(mfree = mfree, idx = idx,
       aligned = nrow(mfree) == nrow(lavaan_free) && !anyNA(idx))
}

# === the parity case table =================================================
# Decoupled from the rigid 1:1 benchmark_cases registry: a case here is a
# (dataset, model, estimator-family) triple. The five ML rows defer to the
# benchmark registry; the FIML/LS/ordinal rows are self-contained.
parity_cases <- list(
  hs_3factor_cfa       = list(family = "ML"),
  bollen_democracy_sem = list(family = "ML"),
  demo_growth_linear   = list(family = "ML"),
  bfi_5factor          = list(family = "ML"),
  mplus_ex5_1          = list(family = "ML"),

  # FIML over genuine bfi missingness. A 2-factor 10-item model keeps the
  # heaviest fit in the suite tractable; the full 5-factor model is a noted
  # follow-up (docs/todo.md).
  bfi_fiml = list(
    family = "FIML",
    model  = paste("neuro =~ N1 + N2 + N3 + N4 + N5",
                   "extra =~ E1 + E2 + E3 + E4 + E5", sep = "\n"),
    package = "psychTools", dataset = "bfi",
    ov = c(paste0("N", 1:5), paste0("E", 1:5))),

  # Continuous ULS / GLS / WLS on the Holzinger-Swineford 3-factor CFA.
  hs_3factor_ls = list(
    family = "LS",
    model  = paste("visual =~ x1 + x2 + x3",
                   "textual =~ x4 + x5 + x6",
                   "speed =~ x7 + x8 + x9", sep = "\n"),
    package = "lavaan", dataset = "HolzingerSwineford1939",
    ov = paste0("x", 1:9),
    estimators = c("ULS", "GLS", "WLS")),

  # Ordinal DWLS / WLS on five bfi Neuroticism items (all six Likert
  # categories well populated). Items are renamed x1..x5 so the C++ ordinal
  # syntax builder (shared with the ordinal corpus goldens) applies verbatim.
  bfi_ordinal_dwls = list(
    family = "ordinal",
    model  = "f =~ x1 + x2 + x3 + x4 + x5",
    package = "psychTools", dataset = "bfi",
    items = paste0("N", 1:5),
    ov = paste0("x", 1:5),
    estimators = c("DWLS", "WLS"))
)

# === ML family (the original parity tranche) ===============================

fit_case_ml <- function(case, model, data) {
  fun <- switch(
    case$lavaan_function %||% "sem",
    cfa = lavaan::cfa, growth = lavaan::growth, sem = lavaan::sem,
    lavaan::sem)
  fun(model = model, data = data,
      estimator = case$estimator %||% "ML",
      meanstructure = isTRUE(case$meanstructure),
      se = "standard", test = "standard", missing = "listwise")
}

emit_ml <- function(id) {
  case <- get_case(id)
  if (!isTRUE(case$supported_now)) {
    stop("case is not supported_now: ", id, call. = FALSE)
  }
  model <- read_model(id)
  raw_data <- read_prepared_data(id)
  fit <- fit_case_ml(case, model, raw_data)

  ov <- lavaan::lavNames(fit, type = "ov")
  meanstructure <- isTRUE(case$meanstructure)

  ## Complete-case rows in observed-variable order: exactly what lavaan fit
  ## under missing = "listwise", and what the C++ complete-data ML path sees.
  data <- raw_data[stats::complete.cases(raw_data[, ov, drop = FALSE]),
                   ov, drop = FALSE]

  lpt <- lavaan::parTable(fit)
  lfree <- lpt[lpt$free > 0, , drop = FALSE]
  al <- align_magmaan_free(model, lfree, meanstructure)
  mfree <- al$mfree
  idx <- al$idx
  aligned <- al$aligned
  if (!aligned) {
    warning(id, ": magmaan free set does not align with lavaan ",
            "(magmaan ", nrow(mfree), " vs lavaan ", nrow(lfree),
            " free params); fixture marked magmaan_aligned = false",
            call. = FALSE)
  }

  S <- lavaan::lavInspect(fit, "sampstat")$cov
  mean_vec <- if (meanstructure) lavaan::lavInspect(fit, "sampstat")$mean else NULL
  fm <- lavaan::fitMeasures(fit)

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

  write_case(id, payload, data, ov, sprintf("ML n=%d", payload$n_obs))
}

# === FIML family ===========================================================

emit_fiml <- function(id) {
  case <- parity_cases[[id]]
  model <- case$model
  ov <- case$ov
  raw <- load_dataset(case$package, case$dataset, ov)
  ## lavaan missing = "ml" drops only all-missing rows; keep every row with at
  ## least one observed value, NAs preserved for the raw-data FIML path.
  raw <- raw[rowSums(!is.na(raw)) > 0, , drop = FALSE]

  fit <- lavaan::cfa(model, data = raw, missing = "ml",
                     meanstructure = TRUE, std.lv = FALSE)
  fit_mlr <- lavaan::cfa(model, data = raw, missing = "ml",
                         estimator = "MLR", meanstructure = TRUE,
                         std.lv = FALSE)

  ov_fit <- lavaan::lavNames(fit, type = "ov")
  lpt <- lavaan::parTable(fit)
  lfree <- lpt[lpt$free > 0, , drop = FALSE]
  al <- align_magmaan_free(model, lfree, meanstructure = TRUE)
  aligned <- al$aligned
  idx <- al$idx
  mfree <- al$mfree
  if (!aligned) {
    warning(id, ": magmaan FIML free set does not align with lavaan; ",
            "fixture marked magmaan_aligned = false", call. = FALSE)
  }

  fm <- lavaan::fitMeasures(fit)
  pull <- function(k) if (k %in% names(fm)) unname(fm[[k]]) else NA_real_

  ## Robust MLR reporting (lavaan missing = "ml", estimator = "MLR").
  robust <- NULL
  if (lavaan::lavInspect(fit_mlr, "converged")) {
    pt_mlr <- lavaan::parTable(fit_mlr)
    free_mlr <- pt_mlr[pt_mlr$free > 0, , drop = FALSE]
    free_mlr <- free_mlr[order(free_mlr$free), , drop = FALSE]
    test_mlr <- lavaan::lavInspect(fit_mlr, "test")$yuan.bentler.mplus
    robust <- list(
      se_robust_huberwhite = unname(free_mlr$se[idx]))
    if (!is.null(test_mlr)) {
      robust$mlr_chisq_scaled <- as.numeric(test_mlr$stat)
      robust$mlr_scaling_factor <- as.numeric(test_mlr$scaling.factor)
      robust$mlr_trace_ugamma <- as.numeric(test_mlr$trace.UGamma)
      robust$mlr_trace_ugamma_h1 <- as.numeric(attr(test_mlr$stat, "h1"))
      robust$mlr_trace_ugamma_h0 <- as.numeric(attr(test_mlr$stat, "h0"))
    }
  }

  payload <- list(
    case_id = id,
    model = model,
    family = "FIML",
    lavaan_version = as.character(utils::packageVersion("lavaan")),
    lavaan_function = "cfa",
    estimator = "ML",
    missing = "ml",
    meanstructure = TRUE,
    auto_cov_y = TRUE,
    n_groups = 1L,
    n_obs = as.integer(lavaan::lavInspect(fit, "ntotal")),
    ov_names = ov_fit,
    n_free = nrow(mfree),
    magmaan_aligned = aligned,
    df = as.integer(pull("df")),
    chisq = pull("chisq"),
    logl = pull("logl"),
    unrestricted_logl = pull("unrestricted.logl"),
    baseline_chisq = pull("baseline.chisq"),
    baseline_df = as.integer(pull("baseline.df")),
    aic = pull("aic"),
    bic = pull("bic"),
    bic2 = pull("bic2"),
    npar = as.integer(pull("npar")),
    cfi = pull("cfi"),
    tli = pull("tli"),
    rmsea = pull("rmsea"),
    rmsea_ci_lower = pull("rmsea.ci.lower"),
    rmsea_ci_upper = pull("rmsea.ci.upper"),
    rmsea_pvalue = pull("rmsea.pvalue"),
    rmsea_close_h0 = pull("rmsea.close.h0"),
    rmsea_notclose_pvalue = pull("rmsea.notclose.pvalue"),
    rmsea_notclose_h0 = pull("rmsea.notclose.h0"),
    robust = robust)

  if (aligned) {
    payload$param_lhs <- mfree$lhs
    payload$param_op <- mfree$op
    payload$param_rhs <- mfree$rhs
    payload$theta_hat <- unname(lfree$est[idx])
    payload$theta_start_lavaan <- unname(lfree$start[idx])
  }

  ## data.json carries the raw matrix with the missingness mask.
  data_payload <- list(
    case_id = id,
    ov_names = ov_fit,
    raw = list(raw_block_json(raw, ov_fit)))
  write_case_raw(id, payload, data_payload,
                 sprintf("FIML n=%d missing=%d", payload$n_obs, sum(is.na(raw))))
}

# === continuous LS family ==================================================

emit_ls <- function(id) {
  case <- parity_cases[[id]]
  model <- case$model
  ov <- case$ov
  raw <- load_dataset(case$package, case$dataset, ov)
  data <- raw[stats::complete.cases(raw), ov, drop = FALSE]

  fits <- list()
  align <- NULL
  for (est in case$estimators) {
    ## Pure lavaan defaults — passing se=/test= changes lavaan's reported LS
    ## chi-square (e.g. ULS default = Browne residual NT statistic, which
    ## magmaan's continuous_ls_chisq reproduces).
    fit <- lavaan::cfa(model, data = data, estimator = est, std.lv = FALSE)
    lpt <- lavaan::parTable(fit)
    lfree <- lpt[lpt$free > 0, , drop = FALSE]
    if (is.null(align)) {
      align <- align_magmaan_free(model, lfree, meanstructure = FALSE,
                                  auto_cov_y = FALSE)
    }
    fj <- ls_fit_json(fit)
    fj$theta_hat <- fj$theta_hat[align$idx]   # reorder into magmaan free order
    if (est == "ULS") {
      fit_rob <- lavaan::cfa(model, data = data, estimator = "ULS",
                             se = "robust.sem",
                             test = c("satorra.bentler", "mean.var.adjusted",
                                      "scaled.shifted"))
      rj <- ls_robust_json(fit_rob)
      rj$se <- rj$se[align$idx]               # reorder into magmaan free order
      fj$robust <- rj
    }
    fits[[est]] <- fj
  }

  if (!align$aligned) {
    warning(id, ": magmaan LS free set does not align with lavaan; ",
            "fixture marked magmaan_aligned = false", call. = FALSE)
  }

  payload <- list(
    case_id = id,
    model = model,
    family = "LS",
    lavaan_version = as.character(utils::packageVersion("lavaan")),
    lavaan_function = "cfa",
    meanstructure = FALSE,
    auto_cov_y = FALSE,
    n_groups = 1L,
    n_obs = nrow(data),
    ov_names = ov,
    n_free = nrow(align$mfree),
    magmaan_aligned = align$aligned,
    estimators = case$estimators,
    fits = fits)
  if (align$aligned) {
    payload$param_lhs <- align$mfree$lhs
    payload$param_op <- align$mfree$op
    payload$param_rhs <- align$mfree$rhs
  }

  write_case(id, payload, data, ov,
             sprintf("LS n=%d estimators=%s", nrow(data),
                     paste(case$estimators, collapse = "/")))
}

# === ordinal family ========================================================

emit_ordinal <- function(id) {
  case <- parity_cases[[id]]
  model <- case$model
  ov <- case$ov                       # x1..xp (renamed)
  raw <- load_dataset(case$package, case$dataset, case$items)
  data <- raw[stats::complete.cases(raw), , drop = FALSE]
  colnames(data) <- ov
  for (j in seq_along(ov)) data[[j]] <- ordered(data[[j]])

  fits <- list()
  for (est in case$estimators) {
    fit <- lavaan::cfa(model, data = data, ordered = ov,
                       estimator = est, parameterization = "delta")
    fj <- ordinal_fit_json(fit)
    if (est == "DWLS") {
      fit_rob <- lavaan::cfa(model, data = data, ordered = ov,
                             estimator = "DWLS", parameterization = "delta",
                             se = "robust.sem",
                             test = c("satorra.bentler", "mean.var.adjusted",
                                      "scaled.shifted"))
      fj$robust <- ordinal_robust_json(fit_rob)
    }
    fits[[est]] <- fj
  }

  payload <- list(
    case_id = id,
    input = model,
    family = "ordinal",
    lavaan_version = as.character(utils::packageVersion("lavaan")),
    lavaan_function = "cfa",
    ordered = ov,
    n_groups = 1L,
    fits = fits)

  data_payload <- list(
    case_id = id,
    ordered = ov,
    blocks = list(list(block = 0L, label = "",
                       matrix = ordered_to_int_matrix(data, ov))))
  write_case_raw(id, payload, data_payload,
                 sprintf("ordinal n=%d estimators=%s", nrow(data),
                         paste(case$estimators, collapse = "/")))
}

# === writers ===============================================================

# Write reference.json plus an X-only data.json (complete-data families).
write_case <- function(id, payload, data, ov, note) {
  case_dir <- file.path(out_root, id)
  dir.create(case_dir, recursive = TRUE, showWarnings = FALSE)
  write_json(payload, file.path(case_dir, "reference.json"), pretty = TRUE,
             auto_unbox = TRUE, null = "null", na = "null", digits = NA)

  data_note <- "skipped (not cleared for redistribution)"
  if (isTRUE(data_redistributable[[id]])) {
    data_payload <- list(
      case_id = id,
      ov_names = ov,
      raw = list(list(X = unname(as.matrix(data)))))
    write_json(data_payload, file.path(case_dir, "data.json"), pretty = TRUE,
               auto_unbox = TRUE, digits = NA)
    data_note <- sprintf("data.json (%d x %d)", nrow(data), ncol(data))
  }
  cat(sprintf("wrote %-20s %-40s %s\n", id, note, data_note))
}

# Write reference.json plus a caller-built data.json (FIML / ordinal).
write_case_raw <- function(id, payload, data_payload, note) {
  case_dir <- file.path(out_root, id)
  dir.create(case_dir, recursive = TRUE, showWarnings = FALSE)
  write_json(payload, file.path(case_dir, "reference.json"), pretty = TRUE,
             auto_unbox = TRUE, null = "null", na = "null", digits = NA)
  data_note <- "skipped (not cleared for redistribution)"
  if (isTRUE(data_redistributable[[id]])) {
    write_json(data_payload, file.path(case_dir, "data.json"), pretty = TRUE,
               auto_unbox = TRUE, null = "null", na = "null", digits = NA)
    data_note <- "data.json committed"
  }
  cat(sprintf("wrote %-20s %-40s %s\n", id, note, data_note))
}

emit <- function(id) {
  case <- parity_cases[[id]]
  if (is.null(case)) stop("unknown parity case: ", id, call. = FALSE)
  switch(case$family,
         ML      = emit_ml(id),
         FIML    = emit_fiml(id),
         LS      = emit_ls(id),
         ordinal = emit_ordinal(id),
         stop("unknown parity family: ", case$family, call. = FALSE))
}

args <- commandArgs(trailingOnly = TRUE)
ids <- if (length(args) == 0) names(parity_cases) else args
for (id in ids) emit(id)
