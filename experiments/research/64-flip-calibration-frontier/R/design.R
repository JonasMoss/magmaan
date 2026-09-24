# Fixed-rank follow-up to Foldnes, Gronneberg and Moss (2026). The published
# null and Study 3 G=8 loading patterns are owned by this experiment leaf.

frontier_groups <- 8L
frontier_tested_items <- 2:5
frontier_null_loadings <- c(
  .8, .9, 1.1, 1.4, .7, 1.4, 1.4, 1.2, 1.1, .6,
  .7, .7, 1.2, .9, 1.3, 1, 1.2, 1.5, .9, 1.3)

frontier_paper_power <- list(
  `5` = rbind(
    c(.88, .72, .62, .62, .78),
    c(.92, .72, .62, .58, .82),
    c(.92, .72, .62, .58, .82),
    c(.88, .72, .62, .58, .78),
    c(.88, .68, .62, .62, .78),
    c(.92, .72, .58, .62, .82),
    c(.88, .68, .62, .58, .82),
    c(.88, .72, .58, .58, .82)),
  `20` = rbind(
    c(1.06, .46, .76, .76, 1.34, .84, 1.16, 1.36, 1.44, .64,
      1.26, .84, .56, 1.46, .86, 1.04, 1.46, 1.14, 1.54, 1.26),
    c(1.14, .46, .76, .84, 1.34, .76, 1.24, 1.36, 1.44, .64,
      1.34, .84, .56, 1.54, .86, 1.04, 1.54, 1.14, 1.54, 1.34),
    c(1.06, .46, .76, .76, 1.34, .84, 1.24, 1.44, 1.44, .64,
      1.34, .84, .64, 1.54, .86, 1.04, 1.54, 1.06, 1.46, 1.34),
    c(1.14, .46, .76, .76, 1.34, .84, 1.24, 1.36, 1.36, .56,
      1.34, .76, .56, 1.54, .86, 1.04, 1.54, 1.14, 1.46, 1.34),
    c(1.06, .54, .84, .76, 1.26, .84, 1.24, 1.44, 1.44, .64,
      1.26, .76, .64, 1.46, .86, 1.04, 1.46, 1.14, 1.54, 1.34),
    c(1.06, .46, .84, .76, 1.26, .84, 1.16, 1.36, 1.44, .56,
      1.26, .76, .56, 1.46, .86, .96, 1.54, 1.14, 1.54, 1.34),
    c(1.14, .54, .76, .76, 1.26, .76, 1.16, 1.36, 1.36, .56,
      1.34, .84, .56, 1.46, .86, .96, 1.46, 1.14, 1.46, 1.26),
    c(1.06, .46, .84, .76, 1.34, .84, 1.16, 1.36, 1.36, .64,
      1.26, .76, .64, 1.54, .86, 1.04, 1.46, 1.06, 1.54, 1.26)))

frontier_group_weights <- c(.50, .65, .80, .95, 1.05, 1.20, 1.35, 1.50)

frontier_group_sizes <- function(n_avg, allocation) {
  stopifnot(n_avg > 0L, allocation %in% c("balanced", "unbalanced"))
  if (allocation == "balanced") return(rep(as.integer(n_avg), frontier_groups))
  out <- as.integer(round(n_avg * frontier_group_weights))
  out[[frontier_groups]] <- out[[frontier_groups]] +
    frontier_groups * as.integer(n_avg) - sum(out)
  out
}

frontier_loading_matrix <- function(p, truth = c("null", "power"),
                                    multiplier = 0) {
  truth <- match.arg(truth)
  stopifnot(p %in% c(5L, 20L), is.finite(multiplier), multiplier >= 0)
  baseline <- frontier_null_loadings[seq_len(p)]
  out <- matrix(rep(baseline, frontier_groups), nrow = frontier_groups,
                byrow = TRUE)
  if (truth == "power") {
    paper <- frontier_paper_power[[as.character(p)]]
    delta <- sweep(paper[, frontier_tested_items, drop = FALSE], 2L,
                   colMeans(paper[, frontier_tested_items, drop = FALSE]), "-")
    out[, frontier_tested_items] <-
      out[, frontier_tested_items, drop = FALSE] + multiplier * delta
  }
  out
}

frontier_population <- function(p, information = c("homogeneous", "geometry"),
                                truth = c("null", "power"), multiplier = 0) {
  information <- match.arg(information)
  truth <- match.arg(truth)
  loadings <- frontier_loading_matrix(p, truth, multiplier)
  phi <- if (information == "homogeneous") rep(1, frontier_groups) else
    rep(c(3, .45), length.out = frontier_groups)
  base_theta <- rep(c(.4, 2.2), length.out = p)
  theta <- lapply(seq_len(frontier_groups), function(g) {
    if (information == "homogeneous") rep(1, p)
    else if (g %% 2L == 1L) base_theta else rev(base_theta)
  })
  sigmas <- lapply(seq_len(frontier_groups), function(g)
    phi[[g]] * tcrossprod(loadings[g, ]) + diag(theta[[g]], p))
  list(p = p, groups = frontier_groups, information = information,
       truth = truth, multiplier = multiplier, loadings = loadings,
       phi = phi, theta = theta, Sigma = sigmas, df = 28L)
}

frontier_model_syntax <- function(p) {
  paste0("f =~ ", paste0("x", seq_len(p), collapse = " + "))
}

frontier_model_specs <- function(p) {
  labels <- paste0("g", seq_len(frontier_groups))
  args <- list(syntax = frontier_model_syntax(p), std_lv = FALSE,
               meanstructure = FALSE, group = "group", group_labels = labels)
  h1 <- do.call(magmaan::model_spec, args)
  partial <- if (p > 5L) paste("f =~", paste0("x", 6:p)) else character()
  h0_args <- c(args, list(group_equal = "loadings"))
  if (length(partial)) h0_args$group_partial <- partial
  list(H1 = h1, H0 = do.call(magmaan::model_spec, h0_args))
}

frontier_read_power_calibration <- function(path = NULL) {
  if (is.null(path) || !nzchar(path)) return(c(`5` = 1, `20` = 1))
  x <- read.csv(path, stringsAsFactors = FALSE)
  if (!all(c("p", "multiplier") %in% names(x)))
    stop("power calibration needs p and multiplier columns", call. = FALSE)
  out <- setNames(x$multiplier, as.character(x$p))
  if (!all(c("5", "20") %in% names(out)) || any(!is.finite(out[c("5", "20")])))
    stop("power calibration must contain finite p=5 and p=20 multipliers",
         call. = FALSE)
  out[c("5", "20")]
}

frontier_canonical_grid <- function(power_calibration = c(`5` = 1, `20` = 1)) {
  out <- expand.grid(
    p = c(5L, 20L), n_avg = c(50L, 100L, 200L, 400L),
    information = c("homogeneous", "geometry"),
    allocation = c("balanced", "unbalanced"),
    distribution = c("normal", "vm", "ig", "pl"),
    truth = c("null", "power"), KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE)
  out$groups <- frontier_groups
  out$df <- 28L
  out$n_total <- frontier_groups * out$n_avg
  out$n_min <- vapply(seq_len(nrow(out)), function(i)
    min(frontier_group_sizes(out$n_avg[[i]], out$allocation[[i]])), integer(1))
  out$base_multiplier <- unname(power_calibration[as.character(out$p)])
  out$power_multiplier <- ifelse(out$truth == "power",
    out$base_multiplier * sqrt(100 / out$n_avg), 0)
  pair_fields <- c("p", "n_avg", "information", "allocation", "distribution")
  pair_key <- do.call(paste, c(out[pair_fields], sep = "|"))
  out$pair_id <- match(pair_key, unique(pair_key))
  out$cell_id <- seq_len(nrow(out))
  out
}

frontier_design <- function(profile = c("smoke", "screen", "focus"),
                            power_calibration = c(`5` = 1, `20` = 1)) {
  profile <- match.arg(profile)
  grid <- frontier_canonical_grid(power_calibration)
  if (profile == "screen") {
    return(grid[grid$truth == "null", , drop = FALSE])
  }
  if (profile == "focus") {
    regime <- (grid$information == "homogeneous" & grid$allocation == "balanced") |
      (grid$information == "geometry" & grid$allocation == "unbalanced")
    keep <- grid$n_avg %in% c(50L, 100L) &
      grid$distribution %in% c("normal", "pl") & regime
    return(grid[keep, , drop = FALSE])
  }
  keep <- grid$truth == "null" & grid$n_avg == 50L &
    grid$information == "homogeneous" & grid$allocation == "balanced"
  grid[keep, , drop = FALSE]
}

frontier_validate_design <- function() {
  stopifnot(length(frontier_null_loadings) == 20L,
            identical(dim(frontier_paper_power$`5`), c(8L, 5L)),
            identical(dim(frontier_paper_power$`20`), c(8L, 20L)),
            abs(sum(frontier_group_weights) - frontier_groups) < 1e-12)
  stopifnot(nrow(frontier_design("smoke")) == 8L,
            nrow(frontier_design("screen")) == 128L,
            nrow(frontier_design("focus")) == 32L)
  for (n in c(50L, 100L, 200L, 400L)) {
    for (allocation in c("balanced", "unbalanced")) {
      sizes <- frontier_group_sizes(n, allocation)
      stopifnot(length(sizes) == frontier_groups, sum(sizes) == frontier_groups * n,
                min(sizes) > 20L)
    }
  }
  for (p in c(5L, 20L)) {
    for (information in c("homogeneous", "geometry")) {
      for (truth in c("null", "power")) {
        pop <- frontier_population(p, information, truth,
                                   if (truth == "power") 1 else 0)
        stopifnot(pop$df == 28L, nrow(pop$loadings) == 8L,
                  ncol(pop$loadings) == p,
                  all(vapply(pop$Sigma, function(S)
                    min(eigen(S, symmetric = TRUE, only.values = TRUE)$values) > 0,
                    logical(1))))
      }
    }
  }
  invisible(TRUE)
}
