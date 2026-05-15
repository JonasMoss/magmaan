#!/usr/bin/env Rscript
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))), "common.R"))
source(file.path(bench_script_dir(), "cases.R"))

require_pkg("lavaan")
require_pkg("jsonlite")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("usage: Rscript benchmarks/r/make_lavaan_reference.R <case_id> [<case_id> ...] | all", call. = FALSE)
}
ids <- if (length(args) == 1 && identical(args, "all")) {
  names(Filter(function(x) isTRUE(x$supported_now), benchmark_cases))
} else {
  args
}

matrix_payload <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.list(x) && !is.data.frame(x)) return(lapply(x, matrix_payload))
  m <- as.matrix(x)
  rn <- rownames(m)
  cn <- colnames(m)
  list(
    rows = if (is.null(rn)) character(0) else rn,
    cols = if (is.null(cn)) character(0) else cn,
    values = unname(m)
  )
}

fit_case <- function(case, model, data) {
  fun <- switch(
    case$lavaan_function %||% "sem",
    cfa = lavaan::cfa,
    growth = lavaan::growth,
    sem = lavaan::sem,
    lavaan::sem
  )

  fun(
    model = model,
    data = data,
    estimator = case$estimator %||% "ML",
    meanstructure = isTRUE(case$meanstructure),
    se = "standard",
    test = "standard",
    missing = "listwise"
  )
}

for (id in ids) {
  case <- get_case(id)
  if (!isTRUE(case$supported_now)) {
    stop("case is not marked supported_now for reference generation: ", id, call. = FALSE)
  }

  data <- read_prepared_data(id)
  model <- read_model(id)
  fit <- fit_case(case, model, data)

  pe <- lavaan::parameterEstimates(fit)
  pt <- lavaan::parTable(fit)
  free <- pt[pt$free > 0, , drop = FALSE]
  free_names <- ifelse(
    nzchar(free$label),
    free$label,
    sprintf("%s%s%s[%d]", free$lhs, free$op, free$rhs, free$free)
  )
  fitted_values <- lavaan::fitted(fit)
  fit_measures <- lavaan::fitMeasures(fit, c("chisq", "df", "pvalue", "fmin"))
  optim <- lavaan::lavInspect(fit, "optim")

  payload <- list(
    case_id = id,
    estimator = case$estimator %||% "ML",
    reference_engine = "lavaan",
    reference_version = as.character(utils::packageVersion("lavaan")),
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    model = model,
    n_obs = unname(lavaan::lavInspect(fit, "nobs")),
    variables = lavaan::lavNames(fit, type = "ov"),
    converged = isTRUE(lavaan::lavInspect(fit, "converged")),
    iterations = as.integer(optim$iterations %||% NA_integer_),
    objective = unname(fit_measures[["fmin"]]),
    chisq = unname(fit_measures[["chisq"]]),
    df = unname(fit_measures[["df"]]),
    pvalue = unname(fit_measures[["pvalue"]]),
    theta_names = unname(free_names),
    theta_hat = stats::setNames(as.numeric(free$est), free_names),
    parameter_table = pe,
    sigma_hat = matrix_payload(fitted_values$cov),
    mu_hat = matrix_payload(fitted_values$mean)
  )

  write_json_file(payload, reference_path(id, "lavaan"))
  cat(sprintf("wrote lavaan reference %-28s -> %s\n", id, reference_path(id, "lavaan")))
}
