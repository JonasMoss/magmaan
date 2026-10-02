# Maintainer step, run once where the textbook corpus is mounted:
#   Rscript experiments/showcases/09-robust-test-calibration/lanes/battery/R/build_populations.R
# Fits each case's null (H0, else H1) by lavaan ML to the book's data and writes
# the implied moments to populations/<id>.json. Those derived moments are the
# simulation populations; the run itself never touches the corpus.
suppressPackageStartupMessages({
  library(jsonlite)
  library(lavaan)
})
filearg <- grep("^--file=", commandArgs(), value = TRUE)
exp_dir <- dirname(dirname(normalizePath(sub("^--file=", "", filearg[1]))))
source(file.path(exp_dir, "R", "cases.R"))
root <- Sys.getenv("MAGMAAN_TEXTBOOK_CORPUS",
                   file.path(exp_dir, "..", "..", "..", "..", "..", "external", "textbook-corpus"))
if (!file.exists(file.path(root, "manifest.csv"))) stop("textbook corpus not mounted at ", root)
root <- normalizePath(root)
manifest <- read.csv(file.path(root, "manifest.csv"))

fit_case <- function(case, syntax) {
  args <- textbookcorpus:::.lavaan_args(case, "ML")
  args$model <- syntax
  do.call(textbookcorpus:::.lavaan_fn(case$meta$lavaan_function), args)
}

build_one <- function(spec) {
  case <- textbookcorpus::load_case(manifest$case_dir[manifest$case_id == spec$corpus], root = root)
  h1 <- spec$h1(case$model)
  h0 <- if (is.null(spec$h0)) NULL else spec$h0(h1)
  pop_fit <- fit_case(case, h0 %||% h1)
  stopifnot(lavInspect(pop_fit, "converged"), lavInspect(pop_fit, "post.check"))
  h1_fit <- fit_case(case, h1)
  implied <- lavInspect(pop_fit, "implied")
  if (lavInspect(pop_fit, "ngroups") == 1L) implied <- list(implied)
  nobs <- lavInspect(pop_fit, "nobs")
  labels <- lavInspect(pop_fit, "group.label")
  groups <- lapply(seq_along(implied), function(g) {
    S <- implied[[g]]$cov
    list(label = if (length(labels)) labels[[g]] else "all",
         proportion = nobs[[g]] / sum(nobs),
         ov = colnames(S), sigma = unclass(S),
         mu = if (is.null(implied[[g]]$mean)) rep(0, ncol(S)) else unname(implied[[g]]$mean))
  })
  ms <- isTRUE(case$meta$model_options$meanstructure)
  fm <- function(f) as.list(fitMeasures(f, c("chisq", "df", "cfi", "rmsea")))
  list(id = spec$id, corpus = spec$corpus, type = spec$type,
       restriction = spec$restriction %||% NA,
       disc_pattern = spec$disc_pattern %||% "alternate",
       lavaan_function = case$meta$lavaan_function, meanstructure = ms,
       h1 = h1, h0 = h0, source_n = sum(nobs), groups = groups,
       source_fit = list(h1 = fm(h1_fit), population_model = fm(pop_fit)))
}

`%||%` <- function(a, b) if (is.null(a)) b else a
dir.create(file.path(exp_dir, "populations"), showWarnings = FALSE)
for (spec in case_specs()) {
  pop <- build_one(spec)
  path <- file.path(exp_dir, "populations", paste0(spec$id, ".json"))
  writeLines(toJSON(pop, auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null"), path)
  cat(sprintf("%-14s p=%2d groups=%d  source chisq(H1)=%.1f df=%d  population df=%d\n", spec$id,
              length(pop$groups[[1]]$ov), length(pop$groups), pop$source_fit$h1$chisq,
              as.integer(pop$source_fit$h1$df), as.integer(pop$source_fit$population_model$df)))
}
