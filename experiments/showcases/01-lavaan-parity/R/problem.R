# Problem construction for a corpus case: the built magmaan model spec, sample
# data, and (for ADF/WLS) the empirical Browne weight. Experiment-local copy so
# this experiment does not source a paper.

make_problem <- function(case, scaling = "n-1") {
  args <- c(list(syntax = case$model), case$model_spec_args)
  spec <- do.call(magmaan::model_spec, args)
  dat <- if (!is.null(case$sample_stats)) {
    case$sample_stats
  } else {
    # case$group_var is NULL for single-group cases, which df_to_data() reads
    # as a single group; multi-group corpus cases carry the grouping column.
    magmaan::df_to_data(case$data, spec, group = case$group_var,
                        scaling = scaling)
  }
  problem <- list(case = case, spec = spec, dat = dat)
  if (toupper(case$estimator) %in% c("ADF", "WLS")) {
    if (!is.null(case$W)) {
      problem$W <- case$W
    } else {
      problem$W <- adf_weight(
        case$data,
        include_means = isTRUE(case$model_spec_args$meanstructure)
      )
    }
  }
  problem
}

# Empirical Browne (ADF/WLS) weight: the inverse of the empirical fourth-moment
# NACOV. Deliberately inverted in R (chol2inv(chol(.))) so this harness studies
# the raw ADF conditioning pathology (see report.qmd); magmaan's own ADF weight
# builder gates rank-deficient NACOVs.
adf_weight <- function(data, include_means = FALSE) {
  if (is.null(data)) {
    stop("ADF/WLS cases require raw data or an explicit W matrix", call. = FALSE)
  }
  X <- as.matrix(data)
  if (include_means) {
    gamma_full <- magmaan::magmaan_core$robust_empirical_gamma_with_means(X)
    return(chol2inv(chol(gamma_full)))
  }
  gamma <- magmaan::magmaan_core$robust_empirical_gamma(X)
  chol2inv(chol(gamma))
}
