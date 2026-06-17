# Real-data SEM corpus loading from the textbook-corpus submodule
# (corpus/textbook-corpus/). Experiment-local copy so this experiment is a sink
# and does not source a paper. `corpus_root()` comes from experiments/_support.
#
# Each case lives at cases/<book>/<case_id>/ with meta.json (schema-v2),
# model.lav (lavaan syntax), and either data/raw.csv or
# data/sample_cov.csv (+ sample_mean.csv). Top-level manifest.csv lists every
# case with lavaan_status; "ok"/"ok_with_warnings" are kept.

# ---- low-level readers ----------------------------------------------------

.corpus_read_vector <- function(path) {
  x <- utils::read.csv(path, check.names = FALSE)
  stats::setNames(x[[2L]], x[[1L]])
}

.corpus_read_matrix <- function(path) {
  as.matrix(utils::read.csv(path, row.names = 1L, check.names = FALSE))
}

.corpus_read_raw <- function(path) {
  utils::read.csv(path, check.names = FALSE)
}

.corpus_read_meta <- function(case_dir) {
  meta_path <- file.path(case_dir, "meta.json")
  if (!file.exists(meta_path)) {
    stop("Missing meta.json at ", case_dir, call. = FALSE)
  }
  jsonlite::fromJSON(meta_path, simplifyVector = TRUE,
                     simplifyDataFrame = FALSE, simplifyMatrix = FALSE)
}

# ---- meta.json -> model_spec arguments ------------------------------------

.corpus_meta_spec_args <- function(meta) {
  mo <- meta$model_options %||% list()
  spec_args <- list()
  spec_args$meanstructure <-
    if (!is.null(mo$meanstructure)) isTRUE(mo$meanstructure) else TRUE
  if (!is.null(mo$fixed_x)) {
    spec_args$fixed_x <- isTRUE(mo$fixed_x)
  }
  # Mirror lavaan's auto.cov.y table (TRUE for sem/growth, FALSE for cfa) so
  # magmaan and lavaan agree on the parameter set; growth also flips fixed_x.
  if (identical(meta$lavaan_function, "growth")) {
    spec_args$model_type <- "growth"
    if (is.null(spec_args$fixed_x)) spec_args$fixed_x <- FALSE
    spec_args$auto_cov_y <- TRUE
  } else if (identical(meta$lavaan_function, "sem")) {
    spec_args$auto_cov_y <- TRUE
  }
  spec_args
}

# ---- per-case prep --------------------------------------------------------

corpus_prepare <- function(case_dir, meta = NULL) {
  if (is.null(meta)) meta <- .corpus_read_meta(case_dir)
  model_path <- file.path(case_dir, "model.lav")
  if (!file.exists(model_path)) {
    stop("Missing model.lav at ", case_dir, call. = FALSE)
  }
  syntax <- paste(readLines(model_path, warn = FALSE), collapse = "\n")

  dspec <- meta$data %||% list()
  kind  <- dspec$kind %||% "raw"
  group_var <- dspec$group_var
  if (!is.character(group_var) || length(group_var) != 1L ||
      !nzchar(group_var)) group_var <- ""
  files <- dspec$files %||% list()
  nobs  <- as.integer(dspec$n_obs %||% integer())

  if (identical(kind, "summary")) {
    cov_files <- unlist(files$sample_cov %||% character())
    if (!length(cov_files)) {
      stop("Summary case ", basename(case_dir),
           " lists no sample_cov files", call. = FALSE)
    }
    cov_paths <- file.path(case_dir, cov_files)
    miss <- !file.exists(cov_paths)
    if (any(miss)) {
      stop("Missing summary cov file(s): ",
           paste(cov_paths[miss], collapse = ", "), call. = FALSE)
    }
    sample_cov <- lapply(cov_paths, .corpus_read_matrix)
    mean_files <- unlist(files$sample_mean %||% character())
    sample_mean <- if (length(mean_files)) {
      mean_paths <- file.path(case_dir, mean_files)
      Map(function(mp, cov) {
        if (file.exists(mp)) .corpus_read_vector(mp)[colnames(cov)]
        else NULL
      }, mean_paths, sample_cov)
    } else {
      rep(list(NULL), length(sample_cov))
    }
    if (!length(nobs)) {
      stop("Summary case ", basename(case_dir),
           " missing data.n_obs in meta.json", call. = FALSE)
    }
    if (length(nobs) != length(sample_cov)) {
      stop("Summary case ", basename(case_dir), ": ", length(nobs),
           " n_obs value(s) for ", length(sample_cov), " group(s)",
           call. = FALSE)
    }
    multigroup <- nzchar(group_var) || length(sample_cov) > 1L
    group_labels <- if (multigroup) paste0("g", seq_along(sample_cov))
                    else character()
    return(list(kind = "summary", syntax = syntax, data = NULL,
                sample_cov = sample_cov, sample_mean = sample_mean,
                nobs = nobs, group_var = group_var,
                group_labels = group_labels))
  }

  # raw
  raw_file <- files$raw %||% "data/raw.csv"
  raw_path <- file.path(case_dir, raw_file)
  if (!file.exists(raw_path)) {
    stop("Missing raw data file: ", raw_path, call. = FALSE)
  }
  raw <- .corpus_read_raw(raw_path)
  if (nzchar(group_var)) {
    if (!group_var %in% names(raw)) {
      stop("Raw case ", basename(case_dir), ": grouping column `", group_var,
           "` not found in ", raw_path, call. = FALSE)
    }
    return(list(kind = "raw", syntax = syntax, data = raw,
                nobs = integer(), group_var = group_var,
                group_labels = unique(as.character(raw[[group_var]])),
                sample_cov = NULL, sample_mean = NULL))
  }
  # Single-group raw: subset to the observed variables magmaan needs so a
  # downstream ADF fourth-moment weight lines up with the moment vector.
  spec_args <- c(list(syntax = syntax), .corpus_meta_spec_args(meta))
  spec <- do.call(magmaan::model_spec, spec_args)
  mdat <- magmaan::df_to_data(raw, spec, scaling = "n-1")
  data <- as.data.frame(mdat$X[[1L]], check.names = FALSE)
  list(kind = "raw", syntax = syntax, data = data, nobs = nrow(data),
       group_var = "", group_labels = character(),
       sample_cov = NULL, sample_mean = NULL)
}

# Build one case for (case, weight). Returns NULL when the pair is infeasible
# (ADF on summary or multi-group data has no empirical fourth-moment weight).
corpus_case <- function(meta, weight, case_dir, prep = NULL) {
  weight <- toupper(weight)
  if (is.null(prep)) prep <- corpus_prepare(case_dir, meta)
  multigroup <- nzchar(prep$group_var) || length(prep$group_labels) > 1L
  if (identical(weight, "ADF") &&
      (identical(prep$kind, "summary") || multigroup)) {
    return(NULL)
  }
  spec_args <- .corpus_meta_spec_args(meta)
  if (multigroup) {
    spec_args$group <- if (nzchar(prep$group_var)) prep$group_var else "group"
    spec_args$group_labels <- prep$group_labels
  }
  case_id <- meta$case_id
  case <- list(
    id = paste0(case_id, "__", tolower(weight)),
    label = meta$label %||% case_id,
    corpus = meta$book %||% basename(dirname(case_dir)),
    geiser_id = case_id,         # back-compat field name used by scripts
    catalogue_id = case_id,
    family = meta$primary_class %||% meta$lavaan_function %||% "",
    data_kind = prep$kind,
    estimator = weight,
    model = prep$syntax,
    model_spec_args = spec_args,
    group_var = if (nzchar(prep$group_var)) prep$group_var else NULL,
    published = list(full = NA_integer_, snlls = NA_integer_),
    note = ""
  )
  if (identical(prep$kind, "raw")) {
    case$data <- prep$data
  } else {
    stats <- list(S = prep$sample_cov, nobs = prep$nobs)
    if (!all(vapply(prep$sample_mean, is.null, logical(1L)))) {
      stats$mean <- prep$sample_mean
    }
    case$sample_stats <- stats
  }
  class(case) <- c("corpus_case", "list")
  case
}

# ---- top-level: cases for a corpus root -----------------------------------

corpus_cases <- function(root = corpus_root(),
                         weights = c("GLS", "ULS", "ADF"),
                         books = NULL,
                         ids = NULL) {
  manifest_path <- file.path(root, "manifest.csv")
  if (!file.exists(manifest_path)) {
    stop("Missing corpus manifest: ", manifest_path,
         "\n(The textbook-corpus real-data dependency is private and optional; ",
         "it is not part of the public repository. Mount it at ",
         "corpus/textbook-corpus/ to run corpus-dependent experiments. ",
         "See corpus/README.md.)",
         call. = FALSE)
  }
  manifest <- utils::read.csv(manifest_path, stringsAsFactors = FALSE,
                              check.names = FALSE)
  ok_status <- c("ok", "ok_with_warnings")
  manifest <- manifest[manifest$lavaan_status %in% ok_status, , drop = FALSE]
  if (!is.null(books)) {
    manifest <- manifest[manifest$book %in% books, , drop = FALSE]
  }
  if (!is.null(ids)) {
    manifest <- manifest[manifest$case_id %in% ids, , drop = FALSE]
  }
  weights <- toupper(weights)
  cases <- list()
  skipped <- list()
  for (i in seq_len(nrow(manifest))) {
    case_id  <- manifest$case_id[i]
    case_dir <- file.path(root, manifest$case_dir[i])
    meta <- tryCatch(.corpus_read_meta(case_dir),
                     error = function(e) {
                       for (w in weights) {
                         skipped[[length(skipped) + 1L]] <<- data.frame(
                           case_id = case_id, weight = w,
                           reason = conditionMessage(e),
                           stringsAsFactors = FALSE)
                       }
                       NULL
                     })
    if (is.null(meta)) next
    prep <- tryCatch(corpus_prepare(case_dir, meta),
                     error = function(e) {
                       for (w in weights) {
                         skipped[[length(skipped) + 1L]] <<- data.frame(
                           case_id = case_id, weight = w,
                           reason = conditionMessage(e),
                           stringsAsFactors = FALSE)
                       }
                       NULL
                     })
    if (is.null(prep)) next
    for (w in weights) {
      case <- corpus_case(meta, w, case_dir, prep = prep)
      if (is.null(case)) {
        skipped[[length(skipped) + 1L]] <- data.frame(
          case_id = case_id, weight = w,
          reason = "ADF needs single-group raw data",
          stringsAsFactors = FALSE)
        next
      }
      cases[[case$id]] <- case
    }
  }
  attr(cases, "skipped") <- if (length(skipped)) {
    do.call(rbind, skipped)
  } else {
    data.frame(case_id = character(), weight = character(),
               reason = character(), stringsAsFactors = FALSE)
  }
  cases
}
