stress_distributions <- c("normal", "vm", "ig", "pl")
stress_missingness <- c(
  "complete", "mcar15", "mcar30", "mar30", "strong_mar30")
stress_ranks <- c(1L, 3L, 5L)

stress_population <- function() {
  lambda <- c(1.00, 0.80, 0.90, 0.70, 1.10, 0.85)
  nu <- c(0.50, 0.30, 0.40, 0.20, 0.60, 0.35)
  theta <- c(0.50, 0.60, 0.55, 0.65, 0.50, 0.60)
  psi <- c(1.00, 1.30)
  alpha <- c(0.00, 0.30)
  Sigma <- lapply(psi, function(phi)
    phi * tcrossprod(lambda) + diag(theta))
  mu <- lapply(seq_along(psi), function(g) nu + lambda * alpha[[g]])
  list(p = 6L, groups = 2L, ov = paste0("x", 1:6), lambda = lambda,
       Sigma = Sigma, mu = mu)
}

stress_group_sizes <- function(n_group1) {
  c(as.integer(n_group1), as.integer(round(0.7 * n_group1)))
}

stress_syntax <- function(rank = 0L) {
  rank <- as.integer(rank)
  if (rank < 0L || rank > 5L) stop("rank must be in 0..5", call. = FALSE)
  rhs <- "x1"
  for (j in 2:6) {
    term <- if (j <= rank + 1L) paste0("L", j, "*x", j) else paste0("x", j)
    rhs <- paste(rhs, term, sep = " + ")
  }
  paste("f =~", rhs)
}

stress_model_specs <- function(ranks = stress_ranks) {
  args <- list(group = "school", group_labels = c("A", "B"),
               meanstructure = TRUE)
  make <- function(rank) do.call(
    magmaan::model_spec,
    c(list(syntax = stress_syntax(rank)), args))
  list(H1 = make(0L), H0 = stats::setNames(lapply(ranks, make), ranks))
}

stress_full_design <- function() {
  out <- expand.grid(
    distribution = stress_distributions,
    n_group1 = c(50L, 100L, 200L, 400L),
    missingness = stress_missingness,
    rank = stress_ranks,
    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  out$distribution <- factor(out$distribution, levels = stress_distributions)
  out$missingness <- factor(out$missingness, levels = stress_missingness)
  out <- out[order(out$distribution, out$n_group1, out$missingness, out$rank), ]
  out$distribution <- as.character(out$distribution)
  out$missingness <- as.character(out$missingness)
  row.names(out) <- NULL
  sizes <- lapply(out$n_group1, stress_group_sizes)
  out$n_group2 <- vapply(sizes, `[[`, integer(1L), 2L)
  out$n_total <- out$n_group1 + out$n_group2
  out$missing_rate <- ifelse(out$missingness == "complete", 0,
                             ifelse(out$missingness == "mcar15", .15, .30))
  out$df <- out$rank
  out$cell_id <- seq_len(nrow(out))
  base_key <- interaction(out$distribution, out$n_group1, out$missingness,
                          drop = TRUE, lex.order = TRUE)
  key_levels <- unique(as.character(base_key))
  out$base_id <- match(as.character(base_key), key_levels)
  out[c("cell_id", "base_id", "distribution", "n_group1", "n_group2",
        "n_total", "missingness", "missing_rate", "rank", "df")]
}

stress_confirmation_filter <- function(grid) {
  is_pl_corner <- grid$distribution == "pl" & grid$n_group1 == 100L &
    ((grid$missingness == "mcar15" & grid$rank == 5L) |
       (grid$missingness == "mcar30" & grid$rank %in% c(1L, 3L, 5L)))
  is_small_mcar <- grid$distribution %in% c("vm", "ig", "pl") &
    grid$n_group1 == 50L & grid$missingness == "mcar30" & grid$rank == 5L
  is_strong_mar <- grid$distribution %in% stress_distributions &
    grid$n_group1 == 100L & grid$missingness == "strong_mar30" &
    grid$rank == 5L
  is_pl_corner | is_small_mcar | is_strong_mar
}

stress_design <- function(profile = c("smoke", "screen", "confirm")) {
  profile <- match.arg(profile)
  grid <- stress_full_design()
  if (profile == "smoke") {
    keep <- grid$distribution %in% c("normal", "pl") &
      grid$n_group1 == 50L &
      grid$missingness %in% c("complete", "strong_mar30") &
      grid$rank %in% c(1L, 5L)
    grid <- grid[keep, ]
  } else if (profile == "confirm") {
    grid <- grid[stress_confirmation_filter(grid), ]
  }
  row.names(grid) <- NULL
  grid
}

stress_base_design <- function(grid) {
  fields <- c("base_id", "distribution", "n_group1", "n_group2", "n_total",
              "missingness", "missing_rate")
  out <- unique(grid[fields])
  out <- out[order(out$base_id), ]
  row.names(out) <- NULL
  out
}

stress_validate_design <- function() {
  pop <- stress_population()
  stopifnot(pop$p == 6L, pop$groups == 2L,
            all(vapply(pop$Sigma, function(S)
              min(eigen(S, symmetric = TRUE, only.values = TRUE)$values) > 0,
              logical(1L))))
  grid <- stress_full_design()
  stopifnot(nrow(grid) == 240L, length(unique(grid$base_id)) == 80L,
            identical(sort(unique(grid$rank)), stress_ranks),
            nrow(stress_design("smoke")) == 8L,
            nrow(stress_design("confirm")) == 11L)
  invisible(TRUE)
}
