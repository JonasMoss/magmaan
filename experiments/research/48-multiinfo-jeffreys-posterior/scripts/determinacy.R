#!/usr/bin/env Rscript
# Secondary runner: refit every dataset of a finished profile with barrier
# variants (the latent-determinacy barrier log det Var(eta | y), a two-weight
# multi-information barrier, and an R refit of the default barrier as a check
# on this fitter). Scored against the stored posterior summaries; no MCMC.

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script <- normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  file.path(dirname(dirname(dirname(script))), "..", "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)
set_single_threaded_math()
require_pkg("magmaan", "install the current R package first (just r-dev)")
suppressPackageStartupMessages(library(magmaan))

exp_dir <- dirname(dirname(normalizePath(sub("^--file=", "", grep("^--file=",
  commandArgs(trailingOnly = FALSE), value = TRUE)[[1L]]))))
for (f in c("design", "structure", "estimators", "barriers")) {
  source(file.path(exp_dir, "R", paste0(f, ".R")))
}

usage <- function() cat(
  "Usage: Rscript scripts/determinacy.R [--profile full] [--cores N] [--resume]\n\n",
  "Refits every dataset listed in results/<profile>/posterior.csv with the\n",
  "latent-determinacy barrier (lambda = 0.25, 0.5, 1, 2), a two-weight\n",
  "multi-information barrier (items 1, latent 0.5), and an R refit of the\n",
  "default barrier. Writes results/<profile>/{fits,params}_barriers.csv.\n",
  "Default 4 cores; about 15 minutes for the 3600-dataset bank.\n", sep = "")

opts <- list(profile = "full", cores = 4L, resume = FALSE)
args <- commandArgs(trailingOnly = TRUE)
i <- 1L
while (i <= length(args)) {
  a <- args[[i]]
  if (a %in% c("-h", "--help")) { usage(); quit(save = "no") }
  else if (a == "--profile") { i <- i + 1L; opts$profile <- args[[i]] }
  else if (a == "--cores") { i <- i + 1L; opts$cores <- as.integer(args[[i]]) }
  else if (a == "--resume") opts$resume <- TRUE
  else stop("unknown argument: ", a, call. = FALSE)
  i <- i + 1L
}

res <- file.path(exp_dir, "results", opts$profile)
post <- read.csv(file.path(res, "posterior.csv"), stringsAsFactors = FALSE)
paths <- list(fits = file.path(res, "fits_barriers.csv"),
              params = file.path(res, "params_barriers.csv"))
done <- if (opts$resume && file.exists(paths$fits)) {
  unique(read.csv(paths$fits, stringsAsFactors = FALSE)[, c("design", "n")])
} else {
  unlink(unlist(paths))
  data.frame(design = character(0), n = integer(0))
}
designs <- all_designs()
cells <- unique(post[, c("design", "n")])
cells <- cells[!paste(cells$design, cells$n) %in% paste(done$design, done$n), , drop = FALSE]
configs <- barrier_configs()

one_dataset <- function(d, n, rep, seed, truth) {
  st <- d$structure
  df <- simulate_data(d, n, seed)
  S <- stats::cov(as.matrix(df)) * (n - 1) / n
  spec <- model_spec(d$syntax)
  dat <- df_to_data(df, spec)
  as_phi <- function(m) {
    r <- run_point(point_methods()[[m]], spec, dat, st)
    if (is.null(r$fit)) return(NULL)
    pt <- r$fit$partable
    phi <- marker_to_phi(st, marker_builder(pt, st)(pt$est[free_order(pt)]))
    if (is.null(phi) || !build_std(st, phi)$ok) NULL else phi
  }
  # Variants start at the default barrier's estimate. The R refit of the
  # default barrier starts elsewhere (lambda = 1), so it tests this fitter.
  phi0 <- as_phi("pen_l025")
  phi_check <- as_phi("pen_l100")
  fits <- list()
  params <- list()
  for (m in names(configs)) {
    start <- if (m == "rfit_multi_l025") phi_check else phi0
    fit <- if (is.null(start)) NULL else fit_barrier(st, S, n, start, configs[[m]])
    fits[[m]] <- data.frame(design = d$key, n = n, rep = rep, seed = seed, method = m,
                            status = if (is.null(fit)) "error" else "ok",
                            converged = if (is.null(fit)) NA else fit$converged,
                            max_grad = if (is.null(fit)) NA_real_ else fit$max_grad,
                            seconds = if (is.null(fit)) NA_real_ else fit$seconds,
                            stringsAsFactors = FALSE)
    if (!is.null(fit)) {
      inv <- invariants_std(st, fit$phi)
      q <- names(truth)
      params[[m]] <- data.frame(design = d$key, n = n, rep = rep, seed = seed, method = m,
                                quantity = q, key = q %in% d$key_params,
                                est = unname(inv[q]), truth = unname(truth),
                                stringsAsFactors = FALSE)
    }
  }
  list(fits = do.call(rbind, fits), params = do.call(rbind, params))
}

cat(sprintf("%d cells to fit | %d configs | %d cores\n", nrow(cells), length(configs), opts$cores))
t_start <- proc.time()[["elapsed"]]
for (ci in seq_len(nrow(cells))) {
  d <- designs[[cells$design[[ci]]]]
  n <- cells$n[[ci]]
  rows <- post[post$design == d$key & post$n == n, c("rep", "seed")]
  truth <- population_invariants(d)
  chunks <- split(seq_len(nrow(rows)), cut(seq_len(nrow(rows)), opts$cores, labels = FALSE))
  out <- parallel::mclapply(chunks, function(ix) {
    lapply(ix, function(k) one_dataset(d, n, rows$rep[[k]], rows$seed[[k]], truth))
  }, mc.cores = opts$cores, mc.preschedule = TRUE)
  bad <- vapply(out, inherits, logical(1), "try-error")
  if (any(bad)) stop("worker failed: ", as.character(out[bad][[1L]]), call. = FALSE)
  out <- unlist(out, recursive = FALSE)
  append_csv(do.call(rbind, lapply(out, `[[`, "fits")), paths$fits)
  append_csv(do.call(rbind, lapply(out, `[[`, "params")), paths$params)
  cat(sprintf("[%2d/%d] %-11s N=%-4d %7.1fs elapsed\n", ci, nrow(cells), d$key, n,
              proc.time()[["elapsed"]] - t_start))
}
cat("Wrote:\n", paste0("  ", unlist(paths), "\n"), sep = "")
