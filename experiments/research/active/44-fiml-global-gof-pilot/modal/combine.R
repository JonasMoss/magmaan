#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  index <- match(flag, args)
  if (is.na(index) || index == length(args)) default else args[[index + 1L]]
}
out_dir <- get_arg("--out-dir")
profile <- get_arg("--profile")
expected_cells <- as.integer(get_arg("--expected-cells", NA_character_))
allow_incomplete <- "--allow-incomplete" %in% args
skip_summaries <- "--skip-summaries" %in% args
if (is.null(out_dir) || is.null(profile) ||
    !profile %in% c("focus", "stress")) {
  stop("provide --out-dir and --profile focus|stress", call. = FALSE)
}

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
exp_path <- normalizePath(file.path(script_dir, ".."))
source(file.path(exp_path, "..", "..", "..", "_support", "R", "helpers.R"))
source(file.path(exp_path, "R", "sem_summaries.R"))

slices <- file.path(out_dir, "slices")
bind_file <- function(name) {
  files <- list.files(
    slices, pattern = paste0("^", name, "\\.csv$"),
    recursive = TRUE, full.names = TRUE)
  if (!length(files)) return(NULL)
  do.call(rbind, lapply(files, function(path) {
    read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  }))
}

bind_raw_chunks <- function() {
  files <- list.files(
    slices, pattern = "^cell_[0-9]+_reps_[0-9]+_[0-9]+\\.csv$",
    recursive = TRUE, full.names = TRUE)
  if (!length(files)) return(NULL)
  do.call(rbind, lapply(files, function(path) {
    read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  }))
}

design <- bind_file("design")
replications <- if (allow_incomplete) bind_raw_chunks() else {
  bind_file("replications")
}
if (is.null(design) || is.null(replications)) {
  stop("completed slices must contain design.csv and replications.csv",
       call. = FALSE)
}
design <- unique(design)
design <- design[order(design$cell_id, design$estimator), ]
replications <- replications[order(
  replications$cell_id, replications$rep, replications$estimator), ]
if (anyDuplicated(design[c("cell_id", "estimator")]) ||
    anyDuplicated(replications[c("cell_id", "estimator", "rep")])) {
  stop("overlapping Modal slices detected", call. = FALSE)
}
if (is.finite(expected_cells) && length(unique(design$cell_id)) != expected_cells) {
  stop("combined ", length(unique(design$cell_id)), " cells; expected ",
       expected_cells, call. = FALSE)
}
replication_key <- interaction(
  replications[c("cell_id", "estimator")], drop = TRUE, lex.order = TRUE)
counts <- table(replication_key)
if (!allow_incomplete && length(unique(as.integer(counts))) != 1L) {
  stop("Modal slices have unequal replication counts", call. = FALSE)
}
expected_reps <- max(as.integer(counts))
coverage <- data.frame(
  cell_id = design$cell_id,
  estimator = design$estimator,
  stringsAsFactors = FALSE)
design_key <- interaction(
  coverage[c("cell_id", "estimator")], drop = TRUE, lex.order = TRUE)
coverage$available_replications <- as.integer(counts[as.character(design_key)])
coverage$expected_replications <- expected_reps
coverage$available_replications[is.na(coverage$available_replications)] <- 0L
coverage$missing_replications <-
  coverage$expected_replications - coverage$available_replications
coverage$complete <- coverage$missing_replications == 0L
if (!allow_incomplete && any(!coverage$complete)) {
  stop("Modal slices have incomplete cells", call. = FALSE)
}

write_csv(design, file.path(out_dir, "design.csv"))
write_csv(replications, file.path(out_dir, "replications.csv"))
write_csv(coverage, file.path(out_dir, "coverage.csv"))
for (name in c("generator_calibration", "timing_summary")) {
  value <- bind_file(name)
  if (!is.null(value)) write_csv(value, file.path(out_dir, paste0(name, ".csv")))
}
write_metadata(file.path(out_dir, "metadata.csv"), list(
  profile = profile,
  cells = length(unique(design$cell_id)),
  complete_cells = sum(coverage$complete),
  partial_cells = sum(!coverage$complete & coverage$available_replications > 0L),
  empty_cells = sum(coverage$available_replications == 0L),
  estimator_rows = nrow(replications),
  expected_reps_per_cell = expected_reps,
  available_reps_min = min(coverage$available_replications),
  available_reps_max = max(coverage$available_replications),
  missing_estimator_rows = sum(coverage$missing_replications),
  incomplete_collection = any(!coverage$complete),
  estimators = unique(replications$estimator),
  truth = unique(replications$truth),
  regions = unique(replications$analysis_region),
  alpha_convention = "p <= 0.05; p < 0.05 retained as sensitivity",
  modal_slices = length(unique(dirname(list.files(
    slices, pattern = "^design\\.csv$", recursive = TRUE,
    full.names = TRUE))))))
if (!skip_summaries) sem_write_method_summaries(replications, out_dir)
cat(sprintf("combined %d cells and %d estimator rows into %s\n",
            length(unique(design$cell_id)), nrow(replications), out_dir))
