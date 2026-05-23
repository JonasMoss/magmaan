#!/usr/bin/env Rscript

# Regenerate paper-corpus fixtures from ignored local raw downloads.
#
# Raw public-repository files live under external/paper_corpus/. The checked-in
# fixture stores only derived model/data summaries, sample statistics, and
# lavaan oracle quantities.

suppressPackageStartupMessages({
  library(jsonlite)
  library(lavaan)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", ".."))
fixtures <- file.path(repo_root, "tests", "fixtures")

out_dir <- file.path(fixtures, "paper_corpus")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

paper_root <- Sys.getenv("PAPER_CORPUS_ROOT", unset = "")
if (!nzchar(paper_root)) {
  paper_root <- file.path(repo_root, "external", "paper_corpus")
}

`%||%` <- function(x, y) if (is.null(x)) y else x

md5_file <- function(path) unname(tools::md5sum(path))

check_file <- function(path, md5) {
  if (!file.exists(path)) {
    stop("Missing raw paper-corpus file: ", path, call. = FALSE)
  }
  got <- md5_file(path)
  if (!identical(got, md5)) {
    stop("Hash mismatch for ", path, ": expected ", md5, ", got ", got,
         call. = FALSE)
  }
}

as_plain_matrix <- function(x) unname(as.matrix(x))
as_plain_vector <- function(x, names_ref = NULL) {
  if (!is.null(names_ref)) x <- x[names_ref]
  unname(as.numeric(x))
}

matrix_or_null <- function(x) {
  if (is.null(x)) NULL else as_plain_matrix(x)
}

vector_or_null <- function(x, names_ref = NULL) {
  if (is.null(x)) NULL else as_plain_vector(x, names_ref)
}

sem_data_zxqvn <- function(path) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  wide <- as.data.frame(env$data, check.names = FALSE)

  measured <- grep("[.](PEBT1|PEBT2|positive|negative|intention)$",
                   names(wide), value = TRUE)
  behaviors <- unique(sub("[.](PEBT1|PEBT2|positive|negative|intention)$",
                          "", measured))
  long <- do.call(rbind, lapply(behaviors, function(b) {
    data.frame(
      participant_id = wide$participant_id,
      Gender = wide$Gender,
      Age = wide$Age,
      Education = wide$Education,
      income = wide$income,
      HHsize = wide$HHsize,
      behavior = b,
      PEBT1 = wide[[paste0(b, ".PEBT1")]],
      PEBT2 = wide[[paste0(b, ".PEBT2")]],
      positive = wide[[paste0(b, ".positive")]],
      negative = wide[[paste0(b, ".negative")]],
      intention = wide[[paste0(b, ".intention")]],
      check.names = FALSE
    )
  }))

  long$participant_id <- factor(long$participant_id)
  long$behavior <- factor(long$behavior)
  long$Gender <- factor(long$Gender)
  long$PEBT1 <- (long$PEBT1 - 1) / 4
  long$PEBT2 <- (long$PEBT2 - 1) / 4
  long$intention <- (long$intention - 1) / 4
  long$positive <- long$positive / 100
  long$negative <- long$negative / 100
  long <- long[long$Gender != 3, , drop = FALSE]
  row.names(long) <- NULL

  list(
    wide_rows = nrow(wide),
    long_rows = nrow(long),
    behavior_count = length(behaviors),
    cluster_count = length(unique(long$participant_id)),
    behaviors = behaviors,
    data = long
  )
}

free_rows_json <- function(fit) {
  pt <- lavaan::parTable(fit)
  free <- pt[pt$free > 0L, , drop = FALSE]
  free <- free[order(free$free), , drop = FALSE]
  lapply(seq_len(nrow(free)), function(i) {
    list(lhs = as.character(free$lhs[i]),
         op = as.character(free$op[i]),
         rhs = as.character(free$rhs[i]),
         group = as.integer(free$group[i]),
         free = as.integer(free$free[i]),
         est = as.numeric(free$est[i]))
  })
}

fit_json <- function(fit) {
  pt <- lavaan::parTable(fit)
  free <- pt[pt$free > 0L, , drop = FALSE]
  free <- free[order(free$free), , drop = FALSE]
  fm <- lavaan::fitMeasures(fit)
  ss <- lavaan::lavInspect(fit, "sampstat")
  im <- lavaan::lavInspect(fit, "implied")
  ov <- lavaan::lavNames(fit, type = "ov")
  list(
    converged = isTRUE(lavaan::lavInspect(fit, "converged")),
    theta_hat = as.numeric(free$est),
    free_rows = free_rows_json(fit),
    fmin = as.numeric(fm["fmin"]),
    chisq = as.numeric(fm["chisq"]),
    df = as.integer(fm["df"]),
    npar = as.integer(fm["npar"]),
    sample_cov = as_plain_matrix(ss$cov),
    sample_mean = vector_or_null(ss$mean, ov),
    n_obs = as.integer(lavaan::lavInspect(fit, "nobs")),
    sigma = as_plain_matrix(im$cov[ov, ov, drop = FALSE]),
    mu = vector_or_null(im$mean, ov)
  )
}

zxqvn_case <- function() {
  root <- file.path(paper_root, "zxqvn", "raw")
  script <- file.path(root, "analysis_code.R")
  data_path <- file.path(root, "data.rdata")
  check_file(script, "ba3af148608f42bba127dcd68ef44fe6")
  check_file(data_path, "9d778b146e383e40f84cfe82d3e2ede3")

  model <- paste(
    "affect =~ positive + negative",
    "PEBT2 ~ intention + affect + PEBT1",
    "intention ~ affect + PEBT1",
    "affect ~ PEBT1",
    sep = "\n"
  )
  transformed <- sem_data_zxqvn(data_path)
  dat <- transformed$data
  fit <- suppressWarnings(lavaan::sem(model, data = dat, warn = FALSE))
  clustered <- suppressWarnings(lavaan::sem(
    model, data = dat, cluster = "participant_id", warn = FALSE))
  if (!isTRUE(lavaan::lavInspect(fit, "converged"))) {
    stop("zxqvn core ML fit did not converge", call. = FALSE)
  }
  if (!isTRUE(lavaan::lavInspect(clustered, "converged"))) {
    stop("zxqvn clustered lavaan fit did not converge", call. = FALSE)
  }

  list(
    id = "zxqvn_affect_mediation",
    source = "zxqvn",
    source_title = "Positive anticipated affective reactions increase pro-environmental behavior",
    source_url = "https://osf.io/zxqvn/",
    model = model,
    lavaan_function = "sem",
    estimator = "ML",
    meanstructure = FALSE,
    fixed_x = TRUE,
    observed_variables = lavaan::lavNames(fit, type = "ov"),
    cluster_variable = "participant_id",
    core_surface = "complete-data ML point estimates without clustered SEs",
    unsupported_surfaces = list(
      list(kind = "clustered_standard_errors",
           note = paste(
             "The source script fits sem(..., cluster = 'participant_id').",
             "Cluster-robust SEs/intercepts are catalogued but not tested in",
             "the core complete-data ML parity fixture."
           ))
    ),
    source_files = list(
      list(name = "analysis_code.R",
           url = "https://osf.io/download/9zxmr/",
           md5 = "ba3af148608f42bba127dcd68ef44fe6",
           sha256 = "61ce7473bf50cf679af2428328bd7c633facb4c71d05e05dd8afd411ddc47778"),
      list(name = "data.rdata",
           url = "https://osf.io/download/3z8t2/",
           md5 = "9d778b146e383e40f84cfe82d3e2ede3",
           sha256 = "b7506610f5b1e1e3516bd1c7def2b36e74bb12c650ae85868f38a8ca940750aa")
    ),
    data_summary = list(
      wide_rows = transformed$wide_rows,
      long_rows = transformed$long_rows,
      behavior_count = transformed$behavior_count,
      cluster_count = transformed$cluster_count,
      behaviors = transformed$behaviors,
      transformation = paste(
        "Base-R equivalent of source pivot_longer over behavior-specific",
        "PEBT1/PEBT2/positive/negative/intention columns, 0-1 rescaling,",
        "and Gender != 3 filtering."
      )
    ),
    fits = list(ML = fit_json(fit)),
    clustered_reference = list(
      converged = isTRUE(lavaan::lavInspect(clustered, "converged")),
      npar = as.integer(lavaan::fitMeasures(clustered, "npar")),
      df = as.integer(lavaan::fitMeasures(clustered, "df")),
      chisq = as.numeric(lavaan::fitMeasures(clustered, "chisq")),
      point_estimates_match_core = isTRUE(all.equal(
        lavaan::parameterEstimates(fit)$est[seq_len(14L)],
        lavaan::parameterEstimates(clustered)$est[seq_len(14L)],
        tolerance = 1e-6
      ))
    )
  )
}

payload <- list(
  `_meta` = list(
    format_version = 1L,
    fixture_kind = "paper_corpus.reference",
    corpus_id = "magmaan_paper_corpus_zxqvn_v1",
    tool = "tests/tools/regen_paper_corpus_fixtures.R",
    generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
    lavaan_version = as.character(utils::packageVersion("lavaan")),
    raw_storage = "external/paper_corpus/zxqvn/raw"
  ),
  cases = list(zxqvn_case())
)

jsonlite::write_json(payload, file.path(out_dir, "zxqvn_reference.json"),
                     pretty = TRUE, auto_unbox = TRUE, null = "null",
                     na = "null", digits = NA)
cat("Wrote ", file.path(out_dir, "zxqvn_reference.json"), "\n", sep = "")
