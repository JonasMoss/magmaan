# Design and population definitions for experiment 61.

recycle_pattern <- function(x, n) rep(x, length.out = as.integer(n))

n_per_group <- function(n_total, ratio) {
  n_total <- as.integer(n_total)
  if (identical(ratio, "1:1")) return(rep(n_total %/% 2L, 2L))
  if (identical(ratio, "1:3")) return(c(n_total %/% 4L, n_total - n_total %/% 4L))
  stop("unknown allocation ratio: ", ratio, call. = FALSE)
}

reference_generator_catalog <- function() {
  data.frame(
    generator = c("normal", "ig1", "ig2",
                  "ordinal_3_symmetric", "ordinal_5_symmetric",
                  "ordinal_5_asymmetric", "ordinal_7_symmetric",
                  "ordinal_7_asymmetric"),
    data_type = c(rep("continuous", 3L), rep("categorical", 5L)),
    categories = c(NA, NA, NA, 3L, 5L, 5L, 7L, 7L),
    thresholds = c(rep(NA, 3L), "symmetric", "symmetric", "asymmetric",
                   "symmetric", "asymmetric"),
    stringsAsFactors = FALSE)
}

reference_delta <- function(step) {
  # Starting values for pilot/modal runs. Paper runs replace these through
  # --delta-metric/--delta-scalar/--delta-strict after power calibration.
  switch(step, metric = 0.20, scalar = 0.30, strict = 0.35,
         stop("unknown invariance step: ", step, call. = FALSE))
}

reference_cells <- function(mode = c("smoke", "modal", "sensitivity", "full")) {
  mode <- match.arg(mode)
  generators <- reference_generator_catalog()
  if (mode == "smoke") {
    generators <- generators[generators$generator %in%
                               c("normal", "ig1", "ordinal_5_asymmetric"), ]
    base <- merge(
      expand.grid(p = 8L, n_total = 120L, ratio = "1:1",
                  step = c("metric", "scalar", "strict"),
                  truth = c("null", "alternative"), stringsAsFactors = FALSE),
      generators, all = TRUE)
  } else if (mode == "modal") {
    generators <- generators[generators$generator %in%
                               c("normal", "ig1", "ig2",
                                 "ordinal_5_symmetric", "ordinal_5_asymmetric"), ]
    base <- merge(
      expand.grid(p = 8L, n_total = c(220L, 440L), ratio = c("1:1", "1:3"),
                  step = c("metric", "scalar", "strict"),
                  truth = c("null", "alternative"), stringsAsFactors = FALSE),
      generators, all = TRUE)
  } else if (mode == "sensitivity") {
    generators <- generators[generators$generator %in% c("normal", "ig1", "ig2"), ]
    base <- merge(
      expand.grid(p = 8L, n_total = 440L, ratio = c("1:1", "1:3"),
                  step = c("metric", "scalar", "strict"), truth = "null",
                  stringsAsFactors = FALSE),
      generators, all = TRUE)
  } else {
    base <- merge(
      expand.grid(p = c(8L, 16L), n_total = c(220L, 440L, 1760L),
                  ratio = c("1:1", "1:3"),
                  step = c("metric", "scalar", "strict"),
                  truth = c("null", "alternative"), stringsAsFactors = FALSE),
      generators, all = TRUE)
  }
  base$null_kind <- "heterogeneous"
  base$delta <- vapply(base$step, reference_delta, numeric(1))
  base$delta[base$truth == "null"] <- 0

  control_n <- if (mode == "smoke") 120L else 440L
  control_generators <- if (mode == "smoke") {
    c("normal", "ig1", "ordinal_5_asymmetric")
  } else if (mode == "sensitivity") {
    c("normal", "ig1", "ig2")
  } else c("normal", "ig2", "ordinal_5_symmetric")
  control_ratios <- if (mode == "sensitivity") c("1:1", "1:3") else "1:1"
  controls <- subset(base, p == 8L & n_total == control_n &
                       ratio %in% control_ratios & truth == "null" &
                       generator %in% control_generators)
  if (nrow(controls)) {
    controls$null_kind <- "exchangeable"
    base <- rbind(base, controls)
  }
  base <- base[order(match(base$truth, c("null", "alternative")),
                     base$null_kind, base$p, base$n_total,
                     base$ratio, base$generator, base$step), ]
  rownames(base) <- NULL
  base$study <- "reference"
  base$allocation <- base$ratio
  ns <- lapply(seq_len(nrow(base)), function(i)
    n_per_group(base$n_total[[i]], base$ratio[[i]]))
  base$n1 <- vapply(ns, `[[`, integer(1), 1L)
  base$n2 <- vapply(ns, `[[`, integer(1), 2L)
  base$pattern <- "single_item"
  base$affected_fraction <- NA_real_
  base$cell_id <- seq_len(nrow(base))
  base
}

chen_chao_pattern <- function(step, pattern, affected_fraction) {
  items <- if (affected_fraction == "1/3") c(2L, 4L) else c(2L, 3L, 4L, 5L)
  n_affected <- length(items)
  # Metric/scalar values reproduce Chen--Chao. Strict uses the same patterns
  # as proportional residual-variance changes, the experiment's extension.
  magnitude <- switch(pattern,
    small = rep(-.20, n_affected),
    large = rep(-.40, n_affected),
    mixed = rep(c(-.30, -.50), length.out = n_affected),
    nonuniform = rep(c(-.30, .30), length.out = n_affected),
    stop("unknown Chen--Chao violation pattern: ", pattern, call. = FALSE))
  list(items = items, magnitude = magnitude)
}

imbalance_cells <- function(mode = c("smoke", "modal", "full")) {
  mode <- match.arg(mode)
  if (mode == "smoke") {
    n_b <- 120L
    patterns <- "mixed"
    fractions <- "1/3"
    n_a <- 60L
  } else if (mode == "modal") {
    n_b <- c(400L, 1000L, 3000L)
    patterns <- c("small", "large", "mixed", "nonuniform")
    fractions <- c("1/3", "2/3")
    n_a <- 200L
  } else {
    n_b <- c(400L, 600L, 800L, 1000L, 1400L, 2000L, 3000L)
    patterns <- c("small", "large", "mixed", "nonuniform")
    fractions <- c("1/3", "2/3")
    n_a <- 200L
  }
  allocation <- do.call(rbind, lapply(n_b, function(nb) {
    total <- n_a + nb
    rbind(data.frame(allocation = "unbalanced", n1 = n_a, n2 = nb),
          data.frame(allocation = "matched_balanced", n1 = total %/% 2L,
                     n2 = total - total %/% 2L))
  }))
  null <- merge(allocation,
                expand.grid(step = c("metric", "scalar", "strict"),
                            truth = "null", pattern = "none",
                            affected_fraction = NA_character_,
                            stringsAsFactors = FALSE), all = TRUE)
  alt <- merge(allocation,
               expand.grid(step = c("metric", "scalar", "strict"),
                           truth = "alternative", pattern = patterns,
                           affected_fraction = fractions,
                           stringsAsFactors = FALSE), all = TRUE)
  out <- rbind(null, alt)
  out <- out[order(match(out$truth, c("null", "alternative")),
                   out$n1 + out$n2, out$allocation,
                   out$step, out$pattern, out$affected_fraction), ]
  rownames(out) <- NULL
  out$study <- "imbalance"
  out$p <- 6L
  out$n_total <- out$n1 + out$n2
  out$ratio <- paste0(out$n1, ":", out$n2)
  out$generator <- "normal"
  out$data_type <- "continuous"
  out$categories <- NA_integer_
  out$thresholds <- NA_character_
  out$null_kind <- ifelse(out$truth == "null", "exchangeable", "not_applicable")
  out$delta <- 0
  out$cell_id <- seq_len(nrow(out))
  out
}

build_reference_population <- function(p) {
  p <- as.integer(p)
  if (!p %in% c(8L, 16L)) stop("reference study supports p = 8 or 16", call. = FALSE)
  m <- p %/% 2L
  ov <- paste0("x", seq_len(p))
  factor_of <- rep(seq_len(2L), each = m)
  marker <- c(1L, m + 1L)
  lambda <- recycle_pattern(c(.80, .70, .90, .75, .85, .72, .88, .78), p)
  theta <- recycle_pattern(c(.50, .60, .55, .65, .58, .52, .62, .57), p)
  nu <- recycle_pattern(c(.20, .40, .30, .50, .35, .45, .25, .42), p)
  lambda[marker] <- 1
  Lambda <- matrix(0, p, 2L)
  Lambda[cbind(seq_len(p), factor_of)] <- lambda
  Phi <- list(
    matrix(c(1.00, .50, .50, 1.00), 2L),
    matrix(c(1.45, .50 * sqrt(1.45 * .85),
             .50 * sqrt(1.45 * .85), .85), 2L))
  list(name = "reference", p = p, n_factor = 2L, ov = ov,
       factor_names = c("f1", "f2"), factor_of = factor_of, marker = marker,
       Lambda = Lambda, theta = theta, nu = nu, Phi = Phi,
       alpha = list(c(0, 0), c(.30, -.20)))
}

build_imbalance_population <- function() {
  p <- 6L
  list(name = "chen_chao", p = p, n_factor = 1L, ov = paste0("x", seq_len(p)),
       factor_names = "f", factor_of = rep(1L, p), marker = 1L,
       Lambda = matrix(rep(.60, p), ncol = 1L), theta = rep(1, p),
       nu = rep(0, p),
       Phi = list(matrix(1, 1L), matrix(1, 1L)),
       alpha = list(0, 0))
}

population_for_cell <- function(cell) {
  pop <- if (cell$study[[1L]] == "reference") {
    build_reference_population(cell$p[[1L]])
  } else build_imbalance_population()

  components <- list(
    list(Lambda = pop$Lambda, theta = pop$theta, nu = pop$nu,
         Phi = pop$Phi[[1L]], alpha = pop$alpha[[1L]]),
    list(Lambda = pop$Lambda, theta = pop$theta, nu = pop$nu,
         Phi = pop$Phi[[2L]], alpha = pop$alpha[[2L]]))
  if (cell$truth[[1L]] == "alternative") {
    if (cell$study[[1L]] == "reference") {
      items <- pop$p
      magnitude <- cell$delta[[1L]]
    } else {
      violation <- chen_chao_pattern(cell$step[[1L]], cell$pattern[[1L]],
                                     cell$affected_fraction[[1L]])
      items <- violation$items
      magnitude <- violation$magnitude
    }
    target_group <- if (cell$study[[1L]] == "reference") 2L else 1L
    if (cell$step[[1L]] == "metric") {
      components[[target_group]]$Lambda[cbind(items, pop$factor_of[items])] <-
        components[[target_group]]$Lambda[cbind(items, pop$factor_of[items])] + magnitude
    } else if (cell$step[[1L]] == "scalar") {
      components[[target_group]]$nu[items] <-
        components[[target_group]]$nu[items] + magnitude
    } else if (cell$step[[1L]] == "strict") {
      components[[target_group]]$theta[items] <-
        components[[target_group]]$theta[items] * (1 + magnitude)
    }
  }
  if (cell$null_kind[[1L]] == "exchangeable" && cell$truth[[1L]] == "null") {
    components[[2L]] <- components[[1L]]
  }
  moments <- lapply(components, function(x) {
    list(Sigma = x$Lambda %*% x$Phi %*% t(x$Lambda) + diag(x$theta),
         mean = x$nu + drop(x$Lambda %*% x$alpha))
  })
  if (any(!is.finite(unlist(moments)))) stop("non-finite population moments", call. = FALSE)
  pop$group <- components
  pop$moments <- moments
  pop
}

measurement_block <- function(pop, equal_loadings) {
  items <- split(seq_len(pop$p), pop$factor_of)
  paste(vapply(seq_along(items), function(j) {
    rhs <- pop$ov[items[[j]]]
    if (equal_loadings) {
      nonmarker <- !items[[j]] %in% pop$marker
      rhs[nonmarker] <- paste0("l", items[[j]][nonmarker], "*", rhs[nonmarker])
    }
    paste0(pop$factor_names[[j]], " =~ ", paste(rhs, collapse = " + "))
  }, character(1)), collapse = "\n")
}

invariance_syntax <- function(pop, level) {
  equal_loadings <- level %in% c("metric", "metric_chart", "scalar", "strict")
  measurement <- measurement_block(pop, equal_loadings)
  free_intercepts <- paste0(pop$ov, " ~ 1", collapse = "\n")
  fixed_factor_means <- paste0(pop$factor_names, " ~ 0*1", collapse = "\n")
  reference_means <- paste0(pop$factor_names, " ~ c(0, NA)*1", collapse = "\n")
  marker_intercepts <- paste0(pop$ov[pop$marker], " ~ i", pop$marker, "*1",
                              collapse = "\n")
  nonmarker <- setdiff(seq_len(pop$p), pop$marker)
  chart_intercepts <- paste(c(marker_intercepts,
                              paste0(pop$ov[nonmarker], " ~ 1")), collapse = "\n")
  equal_intercepts <- paste0(pop$ov, " ~ i", seq_len(pop$p), "*1", collapse = "\n")
  equal_residuals <- paste0(pop$ov, " ~~ e", seq_len(pop$p), "*", pop$ov,
                            collapse = "\n")
  switch(level,
    configural = paste(measurement, free_intercepts, fixed_factor_means, sep = "\n"),
    metric = paste(measurement, free_intercepts, fixed_factor_means, sep = "\n"),
    metric_chart = paste(measurement, chart_intercepts, reference_means, sep = "\n"),
    scalar = paste(measurement, equal_intercepts, reference_means, sep = "\n"),
    strict = paste(measurement, equal_intercepts, reference_means,
                   equal_residuals, sep = "\n"),
    stop("unknown model level: ", level, call. = FALSE))
}

step_models <- function(step) {
  switch(step,
    metric = c(h1 = "configural", h0 = "metric"),
    scalar = c(h1 = "metric_chart", h0 = "scalar"),
    strict = c(h1 = "scalar", h0 = "strict"),
    stop("unknown invariance step: ", step, call. = FALSE))
}
