#!/usr/bin/env Rscript

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script <- if (length(file_arg)) {
    normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    normalizePath("run_experiment.R", mustWork = FALSE)
  }
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)
set_single_threaded_math()

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--audit|--dry-run-plan] [options]\n\n",
  "Reconstruct the Savalei-Falk (2014) test definitions and run a\n",
  "deterministic witness showing that the paper-era observed-information\n",
  "target differs from today's nearby MLR/FMG/robust.two.stage routes.\n\n",
  "Modes:\n",
  "  --audit             run the deterministic configuration probe (default)\n",
  "  --dry-run-plan      print the published 192-cell design without fitting\n\n",
  "Options:\n",
  "  --seed-base N       deterministic probe seed (default 20140279)\n",
  "  --results-dir PATH  output directory (default results)\n",
  sep = ""
)

parse_args <- function(args) {
  out <- list(audit = TRUE, dry_run_plan = FALSE,
              seed_base = 20140279L, results_dir = NULL)
  i <- 1L
  take <- function() {
    i <<- i + 1L
    if (i > length(args)) {
      stop("missing value after ", args[[i - 1L]], call. = FALSE)
    }
    args[[i]]
  }
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0L)
    } else if (arg == "--audit") {
      out$audit <- TRUE
    } else if (arg == "--dry-run-plan") {
      out$dry_run_plan <- TRUE
      out$audit <- FALSE
    } else if (arg == "--seed-base") {
      out$seed_base <- as.integer(take())
    } else if (arg == "--results-dir") {
      out$results_dir <- take()
    } else {
      stop("unknown argument: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }
  if (is.na(out$seed_base)) stop("seed-base must be an integer", call. = FALSE)
  out
}

published_design <- function() {
  out <- expand.grid(
    model = c("Model 1", "Model 2"),
    n = c(200L, 400L, 600L),
    kurtosis = c(7L, 15L),
    missing_proportion = c(0.15, 0.30),
    pattern_regime = c("FP", "MP"),
    missing_mechanism = c("MCAR", "MAR-L", "MAR-L2", "MAR-NL"),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  out$p <- ifelse(out$model == "Model 1", 14L, 21L)
  out$df <- ifelse(out$model == "Model 1", 76L, 186L)
  out$replications <- 1000L
  out$fits_per_replication <- 2L
  out$cell_id <- sprintf("sf2014-%03d", seq_len(nrow(out)))
  out[, c("cell_id", "model", "p", "df", "n", "kurtosis",
          "missing_proportion", "pattern_regime", "missing_mechanism",
          "replications", "fits_per_replication")]
}

missingness_sets <- function() data.frame(
  model = c(rep("Model 1", 8L), rep("Model 2", 8L)),
  pattern_regime = c(rep("FP", 2L), rep("MP", 6L),
                     rep("FP", 2L), rep("MP", 6L)),
  conditioning_variable = c(
    "V1", "V8", "V1", "V3", "V5", "V8", "V10", "V12",
    "V1", "V15", "V1", "V3", "V5", "V15", "V17", "V19"
  ),
  deleted_set = c(
    "V2,V4,V6,V7", "V9,V11,V13,V14", "V2", "V4", "V6,V7", "V9",
    "V11", "V13,V14", "V2,V4,V6,V7,V8,V9",
    "V13,V14,V16,V18,V20,V21", "V2,V8,V9", "V4", "V6,V7",
    "V13,V14,V16", "V18", "V20,V21"
  ),
  stringsAsFactors = FALSE
)

published_parameters <- function() {
  loadings <- c(
    0.743, 0.252, 0.604, 0.540, 0.201, 0.578, 0.729,
    0.536, 0.528, 0.692, 0.694, 0.509, 0.698, 0.562,
    0.426, 0.530, 0.637, 0.755, 0.564, 0.647, 0.825
  )
  factor <- rep(c("F1", "F2", "F3"), each = 7L)
  data.frame(
    model = "Model 2",
    variable = paste0("V", seq_along(loadings)),
    factor = factor,
    loading = loadings,
    residual_variance = 1 - loadings^2,
    mean = 0,
    variance = 1,
    stringsAsFactors = FALSE
  )
}

implementation_map <- function() data.frame(
  route = c(
    "Savalei-Falk robust FIML",
    "magmaan/lavaan current MLR",
    "magmaan FIML FMG-SB",
    "Savalei-Falk robust two-stage",
    "magmaan/lavaan current ML2S"
  ),
  base_statistic = c(
    "observed-data H1-H0 likelihood ratio",
    "observed-data H1-H0 likelihood ratio",
    "observed-data H1-H0 likelihood ratio",
    "(n-1) times Stage-2 ML discrepancy",
    "(n-1) times Stage-2 ML discrepancy"
  ),
  correction_geometry = c(
    "analytic observed information at structured H0",
    "Mplus/Yuan-Bentler H1-minus-H0 trace approximation",
    "saturated-H1 observed information",
    "observed Stage-1 sandwich plus observed complete-data H",
    "observed Stage-1 sandwich plus unstructured-H1 NT metric"
  ),
  paper_target = c("yes", "no", "no", "yes", "no"),
  reason = c(
    "Equation 6 and EQS SE=EXACT/MISSING=ML",
    "modern trace-difference target used by experiment 77",
    "common saturated-H1 metric, not Equation 6",
    "Equations 2-4 and EQS SE=EXACT/MISSING=TS",
    "matches current lavaan robust.two.stage, which forces unstructured H1"
  ),
  stringsAsFactors = FALSE
)

make_probe_data <- function(seed, n = 300L) {
  set.seed(seed)
  lambda <- c(0.80, 0.72, 0.66, 0.77, 0.62, 0.70)
  eta <- stats::rnorm(n)
  errors <- matrix(stats::rnorm(n * length(lambda)), nrow = n)
  x <- sweep(errors, 2L, sqrt(1 - lambda^2), `*`) +
    tcrossprod(eta, lambda)
  x <- exp(0.45 * x)
  x <- scale(x)
  colnames(x) <- paste0("x", seq_along(lambda))
  dat <- as.data.frame(x)

  # Two MAR deletion sets produce multiple incomplete-data patterns while
  # retaining enough observations for a stable, fast same-data witness.
  eligible_1 <- which(dat$x1 > stats::quantile(dat$x1, 0.35))
  eligible_2 <- which(dat$x3 < stats::quantile(dat$x3, 0.65))
  miss_1 <- sample(eligible_1, size = floor(0.25 * n), replace = FALSE)
  miss_2 <- sample(eligible_2, size = floor(0.20 * n), replace = FALSE)
  dat[miss_1, c("x2", "x4")] <- NA_real_
  dat[miss_2, c("x5", "x6")] <- NA_real_
  dat
}

lavaan_test_row <- function(fit, test_name, route, target_relation) {
  test <- lavaan::lavInspect(fit, "test")[[test_name]]
  if (is.null(test)) stop("lavaan test not found: ", test_name, call. = FALSE)
  base <- lavaan::lavInspect(fit, "test")$standard$stat
  data.frame(
    route = route,
    implementation = paste0("lavaan ", as.character(utils::packageVersion("lavaan"))),
    target_relation = target_relation,
    base_statistic = as.numeric(base),
    corrected_statistic = as.numeric(test$stat),
    df = as.numeric(test$df),
    scaling_factor = as.numeric(test$scaling.factor),
    trace = as.numeric(test$trace.UGamma %||% NA_real_),
    p_value = as.numeric(test$pvalue),
    stringsAsFactors = FALSE
  )
}

magmaan_row <- function(route, target_relation, base, corrected, df, scale,
                        trace = NA_real_, p_value) {
  data.frame(
    route = route,
    implementation = paste0("magmaan ", as.character(utils::packageVersion("magmaan"))),
    target_relation = target_relation,
    base_statistic = as.numeric(base),
    corrected_statistic = as.numeric(corrected),
    df = as.numeric(df),
    scaling_factor = as.numeric(scale),
    trace = as.numeric(trace),
    p_value = as.numeric(p_value),
    stringsAsFactors = FALSE
  )
}

run_probe <- function(seed) {
  require_pkg("lavaan", "install lavaan to run the configuration probe")
  require_pkg("magmaan", "install the current R package first")
  suppressPackageStartupMessages(library(magmaan))
  dat <- make_probe_data(seed)
  model <- "f =~ x1 + x2 + x3 + x4 + x5 + x6"

  lav_mlr <- lavaan::cfa(
    model, data = dat, estimator = "MLR", missing = "ml",
    meanstructure = TRUE
  )
  lav_paper_fiml <- lavaan::cfa(
    model, data = dat, estimator = "ML", missing = "ml",
    meanstructure = TRUE, se = "robust.huber.white", test = "yuan.bentler",
    information = "observed", observed.information = "h1",
    h1.information = "structured", h1.information.meat = "structured",
    omega.information = "observed", omega.h1.information = "structured",
    omega.h1.information.meat = "structured"
  )
  lav_ml2s <- lavaan::cfa(
    model, data = dat, estimator = "ML", missing = "robust.two.stage",
    meanstructure = TRUE
  )
  fits <- list(lav_mlr, lav_paper_fiml, lav_ml2s)
  if (!all(vapply(fits, lavaan::lavInspect, logical(1), what = "converged"))) {
    stop("one or more lavaan probe fits did not converge", call. = FALSE)
  }

  fit_fiml <- magmaan::magmaan(
    model, dat, estimator = "FIML", meanstructure = TRUE,
    se = "none", test = "none"
  )
  fit_ml2s <- magmaan::magmaan(
    model, dat, estimator = "ML2S", meanstructure = TRUE,
    se = "none", test = "none"
  )
  mlr <- magmaan::magmaan_core$estimate_fiml_robust_mlr(fit_fiml)
  fmg <- magmaan::fmg_tests(fit_fiml, tests = "SB")

  rows <- list(
    lavaan_test_row(
      lav_mlr, "yuan.bentler.mplus", "current MLR",
      "experiment 77 comparator; not the paper statistic"
    ),
    lavaan_test_row(
      lav_paper_fiml, "yuan.bentler", "paper-FIML configuration proxy",
      "closest current lavaan reconstruction; secondary sentinel only"
    ),
    lavaan_test_row(
      lav_ml2s, "satorra.bentler", "current robust.two.stage",
      "current ML2S oracle; not the paper observed-H configuration"
    ),
    magmaan_row(
      "current MLR", "experiment 77 comparator; not the paper statistic",
      mlr$chisq, mlr$chisq_scaled, mlr$df, mlr$scaling_factor,
      mlr$trace_ugamma %||% NA_real_,
      stats::pchisq(mlr$chisq_scaled, mlr$df, lower.tail = FALSE)
    ),
    magmaan_row(
      "FIML FMG-SB", "saturated-H1 metric; not the paper statistic",
      fmg$base_statistic[[1L]], fmg$chi2_equiv[[1L]], fmg$df[[1L]],
      fmg$base_statistic[[1L]] / fmg$chi2_equiv[[1L]],
      sum(fmg$eigenvalues[[1L]]), fmg$p_value[[1L]]
    ),
    magmaan_row(
      "current ML2S", "current robust.two.stage oracle; not exact paper target",
      fit_ml2s$chisq, fit_ml2s$chisq_scaled, fit_ml2s$df,
      fit_ml2s$scaling_factor, fit_ml2s$ml2s$trace_ugamma,
      stats::pchisq(fit_ml2s$chisq_scaled, fit_ml2s$df, lower.tail = FALSE)
    )
  )
  out <- do.call(rbind, rows)
  out$missing_rate <- mean(is.na(as.matrix(dat)))
  out$n <- nrow(dat)
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
design <- published_design()
if (args$dry_run_plan) {
  print(design, row.names = FALSE)
  cat("\n", nrow(design), " cells x 1,000 replications x 2 estimators = ",
      nrow(design) * 1000L * 2L, " fits.\n", sep = "")
  quit(save = "no", status = 0L)
}

out_dir <- args$results_dir %||% experiment_path("results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
probe <- run_probe(args$seed_base)
write_csv(design, file.path(out_dir, "published_design.csv"))
write_csv(missingness_sets(), file.path(out_dir, "missingness_sets.csv"))
write_csv(published_parameters(), file.path(out_dir, "published_parameters.csv"))
write_csv(implementation_map(), file.path(out_dir, "implementation_map.csv"))
write_csv(probe, file.path(out_dir, "configuration_probe.csv"))
write_metadata(
  file.path(out_dir, "metadata.csv"),
  values = list(
    experiment = "79-savalei-falk-2014-test-map",
    mode = "audit",
    seed_base = args$seed_base,
    probe_n = unique(probe$n),
    probe_missing_rate = unique(probe$missing_rate),
    published_cells = nrow(design),
    published_replications_per_cell = 1000L,
    paper = "Savalei and Falk (2014), doi:10.1080/10705511.2014.882692",
    git_head = git_scalar(c("rev-parse", "HEAD")),
    git_dirty = git_dirty()
  ),
  packages = c("magmaan", "lavaan")
)

lav_mlr <- subset(probe, implementation == paste0("lavaan ",
                  as.character(utils::packageVersion("lavaan"))) &
                  route == "current MLR")
mag_mlr <- subset(probe, grepl("^magmaan ", implementation) &
                  route == "current MLR")
lav_ts <- subset(probe, route == "current robust.two.stage")
mag_ts <- subset(probe, route == "current ML2S")
paper_proxy <- subset(probe, route == "paper-FIML configuration proxy")
fmg <- subset(probe, route == "FIML FMG-SB")

stopifnot(
  abs(lav_mlr$corrected_statistic - mag_mlr$corrected_statistic) < 5e-3,
  abs(lav_ts$corrected_statistic - mag_ts$corrected_statistic) < 5e-3,
  abs(paper_proxy$corrected_statistic - lav_mlr$corrected_statistic) > 1e-5,
  abs(paper_proxy$corrected_statistic - fmg$corrected_statistic) > 1e-5
)

message("Wrote:")
for (name in c("published_design.csv", "missingness_sets.csv",
               "published_parameters.csv", "implementation_map.csv",
               "configuration_probe.csv", "metadata.csv")) {
  message("  ", file.path(out_dir, name))
}
message("Current-route parity and paper-target separation checks passed.")
