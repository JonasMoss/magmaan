# The Ernst et al. (2023) latent-regression stress design, a collinear latent
# predictor design, and the four textbook-corpus cases on which ordinary ML is
# improper or fails.

ernst_design <- function() {
  l <- c(1, 0.8, 0.6)
  beta <- 0.25
  Phi <- matrix(c(1, beta, beta, beta^2 + 1), 2)
  Lam <- rbind(cbind(l, 0), cbind(0, l))
  Sigma <- Lam %*% Phi %*% t(Lam) + diag(6)
  nm <- c(paste0("x", 1:3), paste0("y", 1:3))
  dimnames(Sigma) <- list(nm, nm)
  list(key = "ernst", syntax = "X =~ x1 + x2 + x3\nY =~ y1 + y2 + y3\nY ~ X",
       Sigma = Sigma)
}

# f1 and f2 correlate .95 and both predict f3 with b = .3, .3.
fold_design <- function() {
  Lam <- kronecker(diag(3), matrix(0.7, 3, 1))
  r <- 0.95
  b <- c(0.3, 0.3)
  Phi <- matrix(0, 3, 3)
  Phi[1:2, 1:2] <- matrix(c(1, r, r, 1), 2)
  Phi[3, 1:2] <- Phi[1:2, 3] <- Phi[1:2, 1:2] %*% b
  Phi[3, 3] <- 1
  Sigma <- Lam %*% Phi %*% t(Lam)
  diag(Sigma) <- 1
  nm <- paste0("x", 1:9)
  dimnames(Sigma) <- list(nm, nm)
  list(key = "fold",
       syntax = "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6\nf3 =~ x7 + x8 + x9\nf3 ~ f1 + f2",
       Sigma = Sigma)
}

corpus_case_ids <- c(
  "geiser_2013_ch4_growth_quadratic",
  "geiser_2013_ch3_cfa_second_order",
  "little_2013_ch8_fig3a_unconstrained_4wave_negaff",
  "little_2013_ch8_fig4c_gc_linear_4wave_negaff")

# One corpus case as a model spec plus data object, read from the mounted
# textbook corpus (raw data, lavaan syntax, and model options).
load_corpus_case <- function(id) {
  meta_path <- list.files(file.path(corpus_root(), "cases"), pattern = "^meta\\.json$",
                          recursive = TRUE, full.names = TRUE)
  dir <- dirname(meta_path[basename(dirname(meta_path)) == id])
  if (length(dir) != 1L) stop("corpus case not found: ", id, call. = FALSE)
  meta <- jsonlite::fromJSON(file.path(dir, "meta.json"), simplifyVector = TRUE)
  syntax <- paste(readLines(file.path(dir, "model.lav"), warn = FALSE), collapse = "\n")
  raw <- utils::read.csv(file.path(dir, meta$data$files$raw), check.names = FALSE)
  opts <- meta$model_options
  spec <- model_spec(syntax, meanstructure = isTRUE(opts$meanstructure),
                     fixed_x = isTRUE(opts$fixed_x))
  list(key = id, spec = spec, data = df_to_data(raw, spec))
}
