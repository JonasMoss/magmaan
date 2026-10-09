basis_stats <- function(record) {
  list(S = lapply(record$groups, function(g) {
    s <- do.call(rbind, lapply(g$sigma, unlist))
    dimnames(s) <- list(unlist(g$ov), unlist(g$ov)); s
  }), mean = lapply(record$groups, function(g) unlist(g$mu)),
  nobs = vapply(record$groups, function(g) g$proportion * 100000, 0.))
}

population_spec <- function(record, syntax = record$h1, means = record$meanstructure) {
  magmaanlab::model_spec(syntax, meanstructure = means, fixed_x = FALSE,
    group = if (length(record$groups) > 1) "study_group" else NULL,
    group_labels = if (length(record$groups) > 1)
      vapply(record$groups, `[[`, "", "label") else NULL)
}

# Total and disturbance variances are needed to set standardized perturbations.
# This is the TASK-141 sizing calculation; implied observed moments use core.
variance_blocks <- function(fit) {
  lapply(seq_len(fit$ngroups), function(g) {
    pt <- fit$partable
    pt <- pt[pt$group == g & pt$op %in% c("=~", "~", "~~") & pt$rhs != "1", ]
    vars <- unique(c(pt$lhs, pt$rhs))
    b <- p <- matrix(0, length(vars), length(vars), dimnames = list(vars, vars))
    for (j in seq_len(nrow(pt))) {
      r <- pt[j, ]; value <- if (r$free > 0) fit$theta[r$free] else r$est
      if (r$op == "=~") b[r$rhs, r$lhs] <- value
      if (r$op == "~") b[r$lhs, r$rhs] <- value
      if (r$op == "~~") p[r$lhs, r$rhs] <- p[r$rhs, r$lhs] <- value
    }
    a <- solve(diag(length(vars)) - b)
    list(total = a %*% p %*% t(a), disturbance = p)
  })
}

build_population <- function(cell) {
  record <- basis_model(cell$model)
  stats <- basis_stats(record)
  model <- population_spec(record)
  omissions <- read.csv(support_path("basis", "sem_score", "misspecification.csv"))
  rows <- omissions[omissions$id == cell$model, ]
  labels <- paste0("study_extra_", seq_len(nrow(rows)))
  group_labels <- vapply(record$groups, `[[`, "", "label")
  extras <- vapply(seq_len(nrow(rows)), function(j) {
    coefficient <- if (length(group_labels) == 1) paste0(labels[j], "*") else {
      terms <- ifelse(group_labels == rows$group[j] | rows$group[j] == "all", labels[j], "0")
      paste0("c(", paste(terms, collapse = ","), ")*")
    }
    paste(rows$lhs[j], rows$op[j], paste0(coefficient, rows$rhs[j]))
  }, "")
  expanded <- population_spec(record, paste(c(record$h1, extras), collapse = "\n"))
  generating <- magmaanlab::fit_model(expanded, stats)
  if (!isTRUE(generating$converged) || abs(generating$fmin) > 1e-8)
    stop("Expanded population reconstruction failed: ", cell$model)
  df <- with(magmaanlab::structural_identification(model), n_moments - n_parameters)
  evaluate <- function(delta) {
    gen <- generating
    keys <- which(gen$partable$label %in% labels & gen$partable$free > 0)
    for (iteration in seq_len(200)) {
      blocks <- variance_blocks(gen)
      old <- gen$theta
      for (k in keys) {
        r <- gen$partable[k, ]; z <- blocks[[r$group]]
        factor <- if (r$op == "~~") {
          sqrt(z$disturbance[r$lhs, r$lhs] * z$disturbance[r$rhs, r$rhs])
        } else {
          outcome <- if (r$op == "=~") r$rhs else r$lhs
          predictor <- if (r$op == "=~") r$lhs else r$rhs
          sqrt(z$total[outcome, outcome]/z$total[predictor, predictor])
        }
        gen$theta[r$free] <- delta * factor
      }
      if (max(abs(gen$theta - old)) < 1e-10) break
    }
    blocks <- variance_blocks(gen)
    implied <- magmaanlab::magmaan_core$model_implied(gen)
    if (iteration == 200 || any(!is.finite(gen$theta)) ||
        min(vapply(implied$sigma, function(s) min(eigen(s, symmetric = TRUE)$values), 0.)) <= 1e-9 ||
        min(vapply(blocks, function(z) min(eigen(z$disturbance, symmetric = TRUE)$values), 0.)) < -1e-9)
      stop("Inadmissible generating population: ", cell$model, " delta=", delta)
    ss <- stats; ss$S <- implied$sigma
    if (record$meanstructure) ss$mean <- implied$mu
    # Verify the RAM calculation against the core and the requested standardization.
    for (g in seq_along(ss$S)) {
      ov <- unlist(record$groups[[g]]$ov)
      dimnames(ss$S[[g]]) <- list(ov, ov)
      if (max(abs(blocks[[g]]$total[ov, ov] - ss$S[[g]])) > 1e-6)
        stop("Core/variance-block disagreement")
    }
    fitted <- magmaanlab::fit_model(model, ss)
    if (!isTRUE(fitted$converged)) stop("Population h1 fit failed: ", cell$model)
    list(record = record, stats = ss, model = model, generating = gen,
         fitted = fitted, df = df, delta = delta,
         rmsea = sqrt(length(record$groups) * max(fitted$fmin, 0)/df))
  }
  if (is.finite(cell$delta)) return(evaluate(cell$delta))
  if (cell$target_rmsea == 0) return(evaluate(0))
  # The catalog sign is the TASK-141 validated choice; try the other sign if needed.
  for (sign in unique(c(rows$sign[1], -rows$sign[1]))) {
    grid <- seq(0, .4, by = .05)
    values <- lapply(grid, function(x) tryCatch(evaluate(sign*x), error = function(e) NULL))
    reach <- which(vapply(values, function(x) !is.null(x) && x$rmsea >= cell$target_rmsea, FALSE))
    if (length(reach)) {
      upper <- grid[reach[1]]
      root <- uniroot(function(x) evaluate(sign*x)$rmsea - cell$target_rmsea,
                      c(0, upper), tol = 1e-7)$root
      return(evaluate(sign*root))
    }
  }
  stop("Target RMSEA cannot be reached at admissible |delta| <= .4: ", cell$model)
}
