#!/usr/bin/env Rscript

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) return(y)
  if (length(x) == 1L && is.na(x)) return(y)
  x
}

script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    return(normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE))
  }
  normalizePath(sys.frames()[[1L]]$ofile %||% "run_experiment.R", mustWork = FALSE)
}

parse_args <- function(args) {
  out <- list(
    reps = 10L,
    backend = "nlopt-lbfgs",
    case_regex = NULL,
    include_observed = TRUE,
    corpus_dir = NULL
  )
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript run_experiment.R [--reps N] [--backend NAME] [--cases REGEX] [--latent-only]\n",
        "\n",
        "Defaults: --reps 10 --backend nlopt-lbfgs --corpus ../../corpus/textbook-corpus\n",
        sep = ""
      )
      quit(save = "no", status = 0L)
    } else if (arg == "--reps") {
      i <- i + 1L
      if (i > length(args)) stop("--reps requires a value", call. = FALSE)
      out$reps <- as.integer(args[[i]])
    } else if (startsWith(arg, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", arg))
    } else if (arg == "--backend") {
      i <- i + 1L
      if (i > length(args)) stop("--backend requires a value", call. = FALSE)
      out$backend <- args[[i]]
    } else if (startsWith(arg, "--backend=")) {
      out$backend <- sub("^--backend=", "", arg)
    } else if (arg == "--cases") {
      i <- i + 1L
      if (i > length(args)) stop("--cases requires a value", call. = FALSE)
      out$case_regex <- args[[i]]
    } else if (startsWith(arg, "--cases=")) {
      out$case_regex <- sub("^--cases=", "", arg)
    } else if (arg == "--include-observed") {
      out$include_observed <- TRUE
    } else if (arg == "--latent-only") {
      out$include_observed <- FALSE
    } else if (arg == "--corpus") {
      i <- i + 1L
      if (i > length(args)) stop("--corpus requires a value", call. = FALSE)
      out$corpus_dir <- args[[i]]
    } else if (startsWith(arg, "--corpus=")) {
      out$corpus_dir <- sub("^--corpus=", "", arg)
    } else {
      stop("unknown argument: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }
  if (!is.finite(out$reps) || out$reps < 1L) {
    stop("--reps must be a positive integer", call. = FALSE)
  }
  out$backend <- trimws(out$backend)
  if (!nzchar(out$backend)) stop("--backend must be nonempty", call. = FALSE)
  out
}

require_pkg <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop("required R package is not installed: ", package, call. = FALSE)
  }
  invisible(TRUE)
}

read_matrix_csv <- function(path) {
  df <- utils::read.csv(
    path, stringsAsFactors = FALSE, check.names = FALSE, row.names = 1
  )
  out <- as.matrix(df)
  storage.mode(out) <- "double"
  out
}

read_vector_csv <- function(path) {
  df <- utils::read.csv(
    path, stringsAsFactors = FALSE, check.names = FALSE, row.names = 1
  )
  out <- df[[1L]]
  names(out) <- rownames(df)
  as.numeric(out)
}

matrix_from_json <- function(x) {
  if (is.null(x)) return(NULL)
  out <- do.call(rbind, lapply(x, function(row) as.numeric(unlist(row, use.names = FALSE))))
  storage.mode(out) <- "double"
  out
}

vector_from_json <- function(x) {
  if (is.null(x)) return(NULL)
  as.numeric(unlist(x, use.names = FALSE))
}

collapse_tags <- function(tags) {
  tags <- as.character(tags %||% character())
  tags <- tags[nzchar(tags)]
  if (!length(tags)) return("")
  paste(tags, collapse = ";")
}

primary_tag <- function(tags) {
  tags <- as.character(tags %||% character())
  tags <- tags[nzchar(tags)]
  if (!length(tags)) return("")
  tags[[1L]]
}

has_latent_variable <- function(model) {
  is.character(model) && length(model) == 1L && grepl("=~", model, fixed = TRUE)
}

case_observed_only <- function(model) {
  !has_latent_variable(model)
}

case_regex_match <- function(row, case_regex) {
  if (is.null(case_regex)) return(TRUE)
  fields <- c(row$case_id %||% "", row$book %||% "", row$primary_class %||% "",
              row$tags %||% "", row$case_dir %||% "")
  any(grepl(case_regex, fields))
}

fixture_find_case <- function(path, id) {
  ref <- jsonlite::read_json(path, simplifyVector = FALSE)
  hits <- Filter(function(x) identical(x$id %||% "", id), ref$cases %||% list())
  if (!length(hits)) return(NULL)
  hits[[1L]]
}

fixture_value <- function(case, field) {
  value <- case[[field]]
  if (!is.null(value)) return(value)
  ml <- case$fits$ML
  if (!is.null(ml)) return(ml[[field]])
  NULL
}

normalize_fixture_case <- function(manifest_case, fixture_case) {
  sample_cov <- fixture_value(fixture_case, "sample_cov")
  n_obs <- fixture_value(fixture_case, "n_obs")
  tag <- manifest_case$family %||% fixture_case$family %||%
    fixture_case$model_kind %||% ""
  list(
    id = manifest_case$id,
    source = manifest_case$source,
    subcorpus = manifest_case$source,
    source_case_id = manifest_case$source_case_id,
    label = manifest_case$label %||% fixture_case$name %||% fixture_case$title %||% fixture_case$label %||% manifest_case$id,
    family = tag,
    primary_tag = tag,
    tags = tag,
    data_kind = "summary",
    fidelity_tier = "",
    n_groups = 1L,
    observed_only = isTRUE(manifest_case$observed_only),
    strict_parity = isTRUE(manifest_case$strict_parity),
    model = fixture_case$model,
    meanstructure = isTRUE(fixture_case$meanstructure),
    fixed_x = isTRUE(fixture_case$fixed_x %||% TRUE),
    model_type = if (identical(fixture_case$lavaan_function %||% "", "growth") ||
                     identical(fixture_case$model_kind %||% "", "growth")) "growth" else "sem",
    n_obs = as.integer(n_obs),
    group_var = NULL,
    group_labels = NULL,
    sample_cov = list(matrix_from_json(sample_cov)),
    sample_mean = {
      mean <- vector_from_json(fixture_value(fixture_case, "sample_mean"))
      if (is.null(mean)) NULL else list(mean)
    },
    raw = NULL
  )
}

usable_case <- function(case, include_observed = FALSE) {
  !is.null(case) && !is.null(case$model) &&
    (!is.null(case$raw) || !is.null(case$sample_cov)) &&
    length(case$n_obs) > 0L && !anyNA(case$n_obs) &&
    (include_observed || !isTRUE(case$observed_only)) &&
    (include_observed || grepl("=~", case$model, fixed = TRUE))
}

coverage_row <- function(case_id, label, source, primary_tag, tags, data_kind,
                         n_groups, n_obs_total, lavaan_function, fidelity_tier,
                         status, reason = "") {
  data.frame(
    case_id = case_id %||% "",
    label = label %||% "",
    source = source %||% "",
    primary_tag = primary_tag %||% "",
    tags = tags %||% "",
    data_kind = data_kind %||% "",
    n_groups = as.integer(n_groups %||% NA_integer_),
    n_obs_total = as.integer(n_obs_total %||% NA_integer_),
    lavaan_function = lavaan_function %||% "",
    fidelity_tier = fidelity_tier %||% "",
    coverage_status = status,
    skip_reason = reason,
    stringsAsFactors = FALSE
  )
}

load_fixture_experiment_cases <- function(fixtures_dir, case_regex = NULL,
                                          include_observed = FALSE) {
  manifest_path <- file.path(fixtures_dir, "textbook_corpus", "manifest.json")
  manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
  candidates <- Filter(function(x) {
    identical(x$measurement_kind %||% "", "continuous")
  }, manifest$cases)
  if (!is.null(case_regex)) {
    candidates <- Filter(function(x) grepl(case_regex, x$id), candidates)
  }

  out <- list()
  coverage <- list()
  for (mc in candidates) {
    loaded <- NULL
    for (rel in mc$oracle_files %||% character()) {
      path <- file.path(fixtures_dir, rel)
      if (!file.exists(path)) next
      fc <- fixture_find_case(path, mc$source_case_id)
      if (!is.null(fc)) {
        candidate <- normalize_fixture_case(mc, fc)
        if (usable_case(candidate, include_observed)) {
          loaded <- candidate
          break
        }
        loaded <- loaded %||% candidate
      }
    }
    if (usable_case(loaded, include_observed)) {
      out[[loaded$id]] <- loaded
      coverage[[length(coverage) + 1L]] <- coverage_row(
        loaded$id, loaded$label, loaded$source, loaded$primary_tag, loaded$tags,
        loaded$data_kind, loaded$n_groups, sum(loaded$n_obs),
        loaded$model_type, loaded$fidelity_tier, "timed_candidate"
      )
    } else {
      coverage[[length(coverage) + 1L]] <- coverage_row(
        mc$id, mc$label %||% mc$id, mc$source, mc$family %||% "",
        mc$family %||% "", "summary", 1L, NA_integer_,
        "", "", "skipped_preload", "no fixture sample statistics"
      )
    }
  }
  list(
    cases = out,
    coverage = if (length(coverage)) do.call(rbind, coverage) else data.frame(),
    source = "tests/fixtures/textbook_corpus"
  )
}

load_corpus_case <- function(corpus_dir, row, include_observed = TRUE) {
  case_dir <- file.path(corpus_dir, row$case_dir)
  meta_path <- file.path(case_dir, "meta.json")
  model_path <- file.path(case_dir, "model.lav")
  meta <- jsonlite::read_json(meta_path, simplifyVector = FALSE)
  model <- paste(readLines(model_path, warn = FALSE), collapse = "\n")
  tags <- as.character(meta$tags %||% character())
  options <- meta$model_options
  data <- meta$data
  data_kind <- data$kind %||% row$data_kind %||% ""
  n_groups <- as.integer(data$n_groups %||% row$n_groups %||% 1L)
  n_obs <- as.integer(unlist(data$n_obs %||% row$n_obs_total, use.names = FALSE))
  if (!length(n_obs)) n_obs <- as.integer(row$n_obs_total %||% NA_integer_)

  base_coverage <- function(status, reason = "") {
    coverage_row(
      meta$case_id %||% row$case_id,
      meta$label %||% row$case_id,
      meta$book %||% row$book,
      primary_tag(tags),
      collapse_tags(tags),
      data_kind,
      n_groups,
      sum(n_obs, na.rm = TRUE),
      meta$lavaan_function %||% row$lavaan_function,
      meta$fidelity_tier %||% row$fidelity_tier,
      status,
      reason
    )
  }

  if (isTRUE(meta$out_of_scope %||% FALSE)) {
    return(list(case = NULL, coverage = base_coverage("skipped_preload", "out of scope")))
  }
  if (length(options$ordered %||% character())) {
    return(list(case = NULL, coverage = base_coverage("skipped_preload", "ordered indicators")))
  }
  if (!identical(options$missing %||% "none", "none")) {
    return(list(case = NULL, coverage = base_coverage("skipped_preload", "not complete data")))
  }
  if (!include_observed && case_observed_only(model)) {
    return(list(case = NULL, coverage = base_coverage("skipped_preload", "observed-only model")))
  }

  files <- data$files %||% list()
  sample_cov <- NULL
  sample_mean <- NULL
  raw <- NULL
  if (identical(data_kind, "summary")) {
    cov_files <- as.character(unlist(files$sample_cov %||% character()))
    mean_files <- as.character(unlist(files$sample_mean %||% character()))
    if (!length(cov_files)) {
      return(list(case = NULL, coverage = base_coverage("skipped_preload", "no sample covariance")))
    }
    sample_cov <- lapply(cov_files, function(path) read_matrix_csv(file.path(case_dir, path)))
    sample_mean <- lapply(mean_files, function(path) read_vector_csv(file.path(case_dir, path)))
    if (!length(sample_mean)) sample_mean <- NULL
  } else if (identical(data_kind, "raw")) {
    raw_path <- files$raw %||% ""
    if (!nzchar(raw_path)) {
      return(list(case = NULL, coverage = base_coverage("skipped_preload", "no raw data file")))
    }
    raw <- utils::read.csv(
      file.path(case_dir, raw_path),
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  } else {
    return(list(case = NULL, coverage = base_coverage("skipped_preload", paste("unsupported data kind:", data_kind))))
  }

  group_var <- data$group_var %||% NULL
  group_labels <- NULL
  if (n_groups > 1L) {
    if (identical(data_kind, "raw") && !is.null(group_var) &&
        nzchar(group_var) && group_var %in% names(raw)) {
      group <- raw[[group_var]]
      group_labels <- if (is.factor(group)) levels(group) else unique(as.character(group))
    } else {
      group_labels <- paste0("g", seq_len(n_groups))
    }
  }

  case <- list(
    id = meta$case_id %||% row$case_id,
    source = meta$book %||% row$book,
    subcorpus = meta$book %||% row$book,
    source_case_id = meta$case_id %||% row$case_id,
    label = meta$label %||% row$case_id,
    family = primary_tag(tags),
    primary_tag = primary_tag(tags),
    tags = collapse_tags(tags),
    data_kind = data_kind,
    fidelity_tier = meta$fidelity_tier %||% row$fidelity_tier %||% "",
    observed_only = case_observed_only(model),
    strict_parity = identical(meta$fidelity_tier %||% "", "book_verified"),
    model = model,
    meanstructure = isTRUE(options$meanstructure),
    fixed_x = isTRUE(options$fixed_x %||% TRUE),
    model_type = if (identical(meta$lavaan_function %||% "", "growth")) "growth" else "sem",
    n_groups = n_groups,
    n_obs = n_obs,
    group_var = group_var,
    group_labels = group_labels,
    sample_cov = sample_cov,
    sample_mean = sample_mean,
    raw = raw
  )
  list(case = case, coverage = base_coverage("timed_candidate"))
}

load_corpus_experiment_cases <- function(corpus_dir, case_regex = NULL,
                                         include_observed = TRUE) {
  manifest_path <- file.path(corpus_dir, "manifest.csv")
  manifest <- utils::read.csv(manifest_path, stringsAsFactors = FALSE)
  keep <- vapply(seq_len(nrow(manifest)), function(i) {
    case_regex_match(manifest[i, , drop = FALSE], case_regex)
  }, logical(1))
  manifest <- manifest[keep, , drop = FALSE]

  cases <- list()
  coverage <- list()
  for (i in seq_len(nrow(manifest))) {
    row <- manifest[i, , drop = FALSE]
    loaded <- tryCatch(
      load_corpus_case(corpus_dir, row, include_observed = include_observed),
      error = function(e) {
        coverage <- coverage_row(
          row$case_id, row$case_id, row$book, row$primary_class, row$tags,
          row$data_kind, row$n_groups, row$n_obs_total, row$lavaan_function,
          row$fidelity_tier, "skipped_preload", conditionMessage(e)
        )
        list(case = NULL, coverage = coverage)
      }
    )
    coverage[[length(coverage) + 1L]] <- loaded$coverage
    if (!is.null(loaded$case) && usable_case(loaded$case, include_observed)) {
      cases[[loaded$case$id]] <- loaded$case
    }
  }
  list(
    cases = cases,
    coverage = if (length(coverage)) do.call(rbind, coverage) else data.frame(),
    source = corpus_dir
  )
}

load_experiment_cases <- function(repo_dir, corpus_dir = NULL, case_regex = NULL,
                                  include_observed = TRUE) {
  if (is.null(corpus_dir)) {
    corpus_dir <- file.path(repo_dir, "corpus", "textbook-corpus")
  }
  corpus_manifest <- file.path(corpus_dir, "manifest.csv")
  if (file.exists(corpus_manifest)) {
    return(load_corpus_experiment_cases(
      normalizePath(corpus_dir, mustWork = TRUE),
      case_regex = case_regex,
      include_observed = include_observed
    ))
  }
  load_fixture_experiment_cases(
    file.path(repo_dir, "tests", "fixtures"),
    case_regex = case_regex,
    include_observed = include_observed
  )
}

sample_stats <- function(case) {
  out <- list(S = case$sample_cov, nobs = as.integer(case$n_obs))
  if (!is.null(case$sample_mean)) out$mean <- case$sample_mean
  out
}

build_spec <- function(case) {
  args <- list(
    syntax = case$model,
    fixed_x = case$fixed_x,
    meanstructure = case$meanstructure,
    model_type = case$model_type
  )
  if (!is.null(case$group_var) && nzchar(case$group_var %||% "")) {
    args$group <- case$group_var
  }
  if (!is.null(case$group_labels) && length(case$group_labels)) {
    args$group_labels <- case$group_labels
  }
  do.call(magmaan::model_spec, args)
}

build_data <- function(case, spec) {
  if (!is.null(case$raw)) {
    return(magmaan::df_to_data(
      case$raw,
      spec,
      group = case$group_var,
      missing = "error",
      scaling = "n"
    ))
  }
  sample_stats(case)
}

fit_function <- function(estimator) {
  switch(estimator,
         NT = magmaan::magmaan_core$fit_ml,
         ULS = magmaan::magmaan_core$fit_uls,
         GLS = magmaan::magmaan_core$fit_gls,
         stop("unknown estimator: ", estimator, call. = FALSE))
}

safe_chisq <- function(data, fit) {
  tryCatch(
    as.numeric(magmaan::magmaan_core$infer_chi2_stat(data, fit$fmin)),
    error = function(e) NA_real_
  )
}

safe_df <- function(data, fit) {
  tryCatch(
    as.integer(magmaan::magmaan_core$infer_df_stat(fit$partable, data)),
    error = function(e) NA_integer_
  )
}

fit_case_once <- function(case, spec, data, estimator, rep, backend, control) {
  fn <- fit_function(estimator)
  start <- Sys.time()
  fit <- tryCatch(
    fn(spec, data, optimizer = backend, control = control),
    error = identity
  )
  elapsed <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  if (inherits(fit, "error")) {
    return(list(
      row = new_fit_row(case, estimator, rep, "ERROR",
                        error = conditionMessage(fit),
                        elapsed_sec = elapsed),
      fit = NULL,
      implied = NULL
    ))
  }
  implied <- tryCatch(magmaan::magmaan_core$model_implied(fit), error = identity)
  if (inherits(implied, "error")) {
    return(list(
      row = new_fit_row(case, estimator, rep, "ERROR",
                        error = paste("model_implied:", conditionMessage(implied)),
                        fit = fit, data = data, elapsed_sec = elapsed),
      fit = fit,
      implied = NULL
    ))
  }
  list(
    row = new_fit_row(case, estimator, rep, "OK", fit = fit, data = data,
                      elapsed_sec = elapsed),
    fit = fit,
    implied = implied
  )
}

scalar_int <- function(x, default = NA_integer_) {
  if (is.null(x) || !length(x)) return(as.integer(default))
  as.integer(x[[1L]])
}

scalar_num <- function(x, default = NA_real_) {
  if (is.null(x) || !length(x)) return(as.numeric(default))
  as.numeric(x[[1L]])
}

scalar_chr <- function(x, default = NA_character_) {
  if (is.null(x) || !length(x)) return(as.character(default))
  as.character(x[[1L]])
}

scalar_lgl <- function(x, default = NA) {
  if (is.null(x) || !length(x)) return(as.logical(default))
  as.logical(x[[1L]])
}

all_lgl <- function(x, default = NA) {
  if (is.null(x) || !length(x)) return(as.logical(default))
  x <- as.logical(x)
  if (all(is.na(x))) return(as.logical(default))
  all(x, na.rm = TRUE)
}

new_fit_row <- function(case, estimator, rep, status, error = "",
                        fit = NULL, data = NULL, elapsed_sec = NA_real_) {
  diag <- fit$diagnostics %||% list()
  data.frame(
    case_id = case$id,
    source = case$source,
    source_case_id = case$source_case_id,
    label = case$label,
    family = case$family,
    primary_tag = case$primary_tag,
    tags = case$tags,
    data_kind = case$data_kind,
    n_groups = as.integer(case$n_groups %||% 1L),
    fidelity_tier = case$fidelity_tier,
    strict_parity = case$strict_parity,
    estimator = estimator,
    replicate = as.integer(rep),
    status = status,
    error = error,
    elapsed_sec = elapsed_sec,
    fmin = as.numeric(fit$fmin %||% NA_real_),
    naive_chisq = if (is.null(fit) || is.null(data)) NA_real_ else safe_chisq(data, fit),
    df = if (is.null(fit) || is.null(data)) NA_integer_ else safe_df(data, fit),
    iterations = scalar_int(fit$iterations),
    f_evals = scalar_int(fit$f_evals),
    g_evals = scalar_int(fit$g_evals),
    optimizer_status = scalar_chr(fit$optimizer_status),
    converged = scalar_lgl(fit$converged),
    grad_norm = scalar_num(fit$grad_norm),
    npar = scalar_int(fit$npar),
    ntotal = scalar_int(fit$ntotal, sum(as.integer(case$n_obs), na.rm = TRUE)),
    sigma_pd_all = all_lgl(diag$sigma_pd_all),
    stringsAsFactors = FALSE
  )
}

free_estimates <- function(fit) {
  pt <- fit$partable
  keep <- pt$free > 0L & is.finite(pt$est)
  data.frame(
    key = paste(pt$group[keep], pt$op[keep], pt$lhs[keep], pt$rhs[keep], sep = "\r"),
    est = as.numeric(pt$est[keep]),
    stringsAsFactors = FALSE
  )
}

parameter_diff <- function(nt_fit, other_fit) {
  nt <- free_estimates(nt_fit)
  other <- free_estimates(other_fit)
  idx <- match(nt$key, other$key)
  ok <- !is.na(idx)
  if (!any(ok)) {
    return(c(n_common = 0, max_abs = NA_real_, rms = NA_real_,
             max_rel = NA_real_))
  }
  d <- other$est[idx[ok]] - nt$est[ok]
  c(
    n_common = sum(ok),
    max_abs = max(abs(d), na.rm = TRUE),
    rms = sqrt(mean(d^2, na.rm = TRUE)),
    max_rel = max(abs(d) / pmax(1, abs(nt$est[ok])), na.rm = TRUE)
  )
}

max_abs_nested_diff <- function(a, b) {
  if (is.null(a) || is.null(b) || length(a) != length(b)) return(NA_real_)
  vals <- Map(function(x, y) {
    x <- as.matrix(x)
    y <- as.matrix(y)
    if (!all(dim(x) == dim(y))) return(Inf)
    if (!length(x)) return(0)
    max(abs(x - y), na.rm = TRUE)
  }, a, b)
  max(as.numeric(vals), na.rm = TRUE)
}

ratio_or_na <- function(num, den) {
  if (!is.finite(num) || !is.finite(den) || den == 0) return(NA_real_)
  num / den
}

pair_row <- function(case, rep, nt, other) {
  ntr <- nt$row
  oth <- other$row
  complete <- identical(ntr$status, "OK") && identical(oth$status, "OK")
  pd <- c(n_common = 0, max_abs = NA_real_, rms = NA_real_, max_rel = NA_real_)
  sigma_diff <- mu_diff <- NA_real_
  if (complete) {
    pd <- parameter_diff(nt$fit, other$fit)
    sigma_diff <- max_abs_nested_diff(nt$implied$sigma, other$implied$sigma)
    mu_diff <- max_abs_nested_diff(nt$implied$mu, other$implied$mu)
  }
  data.frame(
    case_id = case$id,
    source = case$source,
    source_case_id = case$source_case_id,
    label = case$label,
    family = case$family,
    primary_tag = case$primary_tag,
    tags = case$tags,
    data_kind = case$data_kind,
    n_groups = as.integer(case$n_groups %||% 1L),
    fidelity_tier = case$fidelity_tier,
    strict_parity = case$strict_parity,
    estimator = oth$estimator,
    replicate = as.integer(rep),
    status = if (complete) "OK" else "INCOMPLETE",
    nt_status = ntr$status,
    estimator_status = oth$status,
    nt_optimizer_status = ntr$optimizer_status,
    estimator_optimizer_status = oth$optimizer_status,
    nt_converged = ntr$converged,
    estimator_converged = oth$converged,
    nt_elapsed_sec = ntr$elapsed_sec,
    estimator_elapsed_sec = oth$elapsed_sec,
    elapsed_ratio_estimator_over_nt = ratio_or_na(oth$elapsed_sec, ntr$elapsed_sec),
    nt_fmin = ntr$fmin,
    estimator_fmin = oth$fmin,
    fmin_diff_estimator_minus_nt = oth$fmin - ntr$fmin,
    nt_naive_chisq = ntr$naive_chisq,
    estimator_naive_chisq = oth$naive_chisq,
    naive_chisq_diff_estimator_minus_nt = oth$naive_chisq - ntr$naive_chisq,
    abs_naive_chisq_diff = abs(oth$naive_chisq - ntr$naive_chisq),
    iterations_diff_estimator_minus_nt = oth$iterations - ntr$iterations,
    f_evals_diff_estimator_minus_nt = oth$f_evals - ntr$f_evals,
    g_evals_diff_estimator_minus_nt = oth$g_evals - ntr$g_evals,
    grad_norm_ratio_estimator_over_nt = ratio_or_na(oth$grad_norm, ntr$grad_norm),
    param_n_common = as.integer(pd[["n_common"]]),
    param_max_abs_diff = as.numeric(pd[["max_abs"]]),
    param_rms_diff = as.numeric(pd[["rms"]]),
    param_max_rel_diff = as.numeric(pd[["max_rel"]]),
    implied_sigma_max_abs_diff = sigma_diff,
    implied_mu_max_abs_diff = mu_diff,
    stringsAsFactors = FALSE
  )
}

median_na <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  stats::median(x)
}

mean_logical <- function(x) {
  x <- x[!is.na(x)]
  if (!length(x)) return(NA_real_)
  mean(x)
}

case_summary <- function(fits, pairs) {
  labels <- unique(fits[, c("case_id", "source", "label", "family",
                            "primary_tag", "tags", "data_kind", "n_groups",
                            "fidelity_tier", "strict_parity")])
  labels <- labels[order(labels$label, labels$case_id), , drop = FALSE]
  rows <- lapply(seq_len(nrow(labels)), function(i) {
    row <- labels[i, , drop = FALSE]
    f <- fits[fits$case_id == row$case_id, ]
    p <- pairs[pairs$case_id == row$case_id & pairs$status == "OK", ]
    med_time <- function(est) median_na(f$elapsed_sec[f$estimator == est & f$status == "OK"])
    med_chi <- function(est) median_na(f$naive_chisq[f$estimator == est & f$status == "OK"])
    med_pair <- function(est, col) median_na(p[p$estimator == est, col])
    data.frame(
      case_id = row$case_id,
      source = row$source,
      label = row$label,
      family = row$family,
      primary_tag = row$primary_tag,
      tags = row$tags,
      data_kind = row$data_kind,
      n_groups = row$n_groups,
      fidelity_tier = row$fidelity_tier,
      strict_parity = row$strict_parity,
      nt_time_sec = med_time("NT"),
      uls_time_sec = med_time("ULS"),
      gls_time_sec = med_time("GLS"),
      uls_elapsed_ratio_over_nt = med_pair("ULS", "elapsed_ratio_estimator_over_nt"),
      gls_elapsed_ratio_over_nt = med_pair("GLS", "elapsed_ratio_estimator_over_nt"),
      nt_ok = sum(f$estimator == "NT" & f$status == "OK", na.rm = TRUE),
      uls_ok = sum(f$estimator == "ULS" & f$status == "OK", na.rm = TRUE),
      gls_ok = sum(f$estimator == "GLS" & f$status == "OK", na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

geo_mean <- function(x) {
  x <- x[is.finite(x) & x > 0]
  if (!length(x)) return(NA_real_)
  exp(mean(log(x)))
}

quantile_na <- function(x, p) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  unname(stats::quantile(x, p, names = FALSE))
}

one_group_summary <- function(x, group_kind, group) {
  rows <- lapply(split(x, x$estimator), function(y) {
    ratio <- y$elapsed_ratio_estimator_over_nt
    data.frame(
      group_kind = group_kind,
      group = group,
      estimator = y$estimator[[1L]],
      n_pairs = nrow(y),
      n_cases = length(unique(y$case_id)),
      geometric_mean_ratio = geo_mean(ratio),
      median_ratio = median_na(ratio),
      p25_ratio = quantile_na(ratio, 0.25),
      p75_ratio = quantile_na(ratio, 0.75),
      share_faster_than_nt = mean(ratio < 1, na.rm = TRUE),
      median_nt_ms = 1000 * median_na(y$nt_elapsed_sec),
      median_estimator_ms = 1000 * median_na(y$estimator_elapsed_sec),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

row_has_tag <- function(tags, tag) {
  parts <- strsplit(as.character(tags %||% ""), ";", fixed = TRUE)
  vapply(parts, function(x) tag %in% x, logical(1))
}

group_summary <- function(pairs) {
  ok <- pairs[pairs$status == "OK" &
                is.finite(pairs$elapsed_ratio_estimator_over_nt) &
                pairs$elapsed_ratio_estimator_over_nt > 0, , drop = FALSE]
  if (!nrow(ok)) return(data.frame())
  rows <- list(one_group_summary(ok, "Total", "All"))
  rows <- c(rows, lapply(split(ok, ok$source), function(x) {
    one_group_summary(x, "Subcorpus", x$source[[1L]])
  }))
  tags <- sort(unique(unlist(strsplit(as.character(ok$tags), ";", fixed = TRUE))))
  tags <- tags[nzchar(tags)]
  rows <- c(rows, lapply(tags, function(tag) {
    one_group_summary(ok[row_has_tag(ok$tags, tag), , drop = FALSE], "Tag", tag)
  }))
  out <- do.call(rbind, rows)
  out[order(out$group_kind, out$group, out$estimator), , drop = FALSE]
}

write_csv <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(x, path, row.names = FALSE, na = "")
  invisible(path)
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
require_pkg("jsonlite")
require_pkg("magmaan")

experiment_dir <- dirname(script_path())
repo_dir <- normalizePath(file.path(experiment_dir, "..", ".."), mustWork = TRUE)
results_dir <- file.path(experiment_dir, "results")
control <- list(max_iter = 6000L, ftol = 1e-12, gtol = 1e-8, history = 10L)
estimators <- c("NT", "ULS", "GLS")

loaded <- load_experiment_cases(
  repo_dir,
  corpus_dir = args$corpus_dir,
  case_regex = args$case_regex,
  include_observed = args$include_observed
)
cases <- loaded$cases
coverage_df <- loaded$coverage
cat(sprintf(
  "complete-data estimator experiment: source=%s, corpus rows=%d, timed candidates=%d, estimators=%s, reps=%d, backend=%s\n",
  loaded$source,
  nrow(coverage_df),
  length(cases),
  paste(estimators, collapse = ","),
  args$reps,
  args$backend
))

fits <- list()
pairs <- list()

for (case in cases) {
  spec <- tryCatch(build_spec(case), error = identity)
  if (inherits(spec, "error")) {
    for (rep in seq_len(args$reps)) {
      for (estimator in estimators) {
        fits[[length(fits) + 1L]] <- new_fit_row(
          case, estimator, rep, "SKIP",
          error = paste("model_spec:", conditionMessage(spec))
        )
      }
    }
    cat(sprintf("  %-48s SKIP model_spec: %s\n", case$id, conditionMessage(spec)))
    next
  }

  data <- tryCatch(build_data(case, spec), error = identity)
  if (inherits(data, "error")) {
    for (rep in seq_len(args$reps)) {
      for (estimator in estimators) {
        fits[[length(fits) + 1L]] <- new_fit_row(
          case, estimator, rep, "SKIP",
          error = paste("data:", conditionMessage(data))
        )
      }
    }
    cat(sprintf("  %-48s SKIP data: %s\n", case$id, conditionMessage(data)))
    next
  }

  for (rep in seq_len(args$reps)) {
    results <- setNames(vector("list", length(estimators)), estimators)
    for (estimator in estimators) {
      results[[estimator]] <- fit_case_once(
        case, spec, data, estimator, rep, args$backend, control
      )
      results[[estimator]]$estimator <- estimator
      fits[[length(fits) + 1L]] <- results[[estimator]]$row
    }
    for (estimator in c("ULS", "GLS")) {
      pairs[[length(pairs) + 1L]] <- pair_row(
        case, rep, results$NT, results[[estimator]]
      )
    }
  }

  latest <- tail(do.call(rbind, pairs), 2L)
  cat(sprintf(
    "  %-48s ULS ratio=%s GLS ratio=%s ULS dpar=%s GLS dpar=%s\n",
    case$id,
    formatC(latest$elapsed_ratio_estimator_over_nt[latest$estimator == "ULS"], format = "g", digits = 3),
    formatC(latest$elapsed_ratio_estimator_over_nt[latest$estimator == "GLS"], format = "g", digits = 3),
    formatC(latest$param_max_abs_diff[latest$estimator == "ULS"], format = "g", digits = 3),
    formatC(latest$param_max_abs_diff[latest$estimator == "GLS"], format = "g", digits = 3)
  ))
}

fits_df <- if (length(fits)) do.call(rbind, fits) else data.frame()
pairs_df <- if (length(pairs)) do.call(rbind, pairs) else data.frame()
summary_df <- if (nrow(fits_df)) case_summary(fits_df, pairs_df) else data.frame()
group_df <- if (nrow(pairs_df)) group_summary(pairs_df) else data.frame()

write_csv(fits_df, file.path(results_dir, "fits.csv"))
write_csv(pairs_df, file.path(results_dir, "pairs_vs_nt.csv"))
write_csv(summary_df, file.path(results_dir, "case_summary.csv"))
write_csv(group_df, file.path(results_dir, "group_summary.csv"))
write_csv(coverage_df, file.path(results_dir, "coverage.csv"))

cat(sprintf(
  "\nwrote %s, %s, %s, %s, %s\n",
  file.path(results_dir, "fits.csv"),
  file.path(results_dir, "pairs_vs_nt.csv"),
  file.path(results_dir, "case_summary.csv"),
  file.path(results_dir, "group_summary.csv"),
  file.path(results_dir, "coverage.csv")
))
