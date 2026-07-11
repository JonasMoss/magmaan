# Literature-based continuous main-study primitives for experiment 61.

main_study_cells <- function() {
  base <- expand.grid(
    data_type = c("continuous", "categorical"),
    p = c(8L, 16L),
    n_total = c(220L, 440L, 1760L),
    ratio = c("1:1", "1:3"),
    generator = c("normal", "ig1", "ig2"),
    step = c("metric", "scalar", "strict"),
    truth = c("null", "alternative"),
    stringsAsFactors = FALSE)
  base$control <- FALSE
  control <- subset(base,
                    p == 8L & n_total == 440L & ratio == "1:1" &
                    generator %in% c("normal", "ig2") & truth == "null")
  control$control <- TRUE
  # The categorical generator is latent-Gaussian ordinal projection; the
  # continuous IG labels do not apply to it.
  base$generator[base$data_type == "categorical"] <- "ordinal_probit"
  control$generator[control$data_type == "categorical"] <- "ordinal_probit"
  out <- unique(rbind(base, control))
  out$cell_id <- seq_len(nrow(out))
  out
}

n_per_group <- function(n_total, ratio) {
  if (identical(ratio, "1:1")) return(rep(as.integer(n_total / 2L), 2L))
  c(as.integer(n_total / 4L), as.integer(n_total - n_total / 4L))
}

build_main_population <- function(p) {
  p <- as.integer(p)
  if (!p %in% c(8L, 16L)) stop("main study supports p = 8 or 16", call. = FALSE)
  m <- p / 2L
  ov <- paste0("x", seq_len(p))
  factor_of <- rep(seq_len(2L), each = m)
  marker <- c(1L, m + 1L)
  lambda_pattern <- c(.80, .70, .90, .75, .85, .72, .88, .78)
  theta_pattern <- c(.50, .60, .55, .65, .58, .52, .62, .57)
  nu_pattern <- c(.20, .40, .30, .50, .35, .45, .25, .42)
  lambda <- lambda_pattern[seq_len(p)]
  lambda[marker] <- 1
  theta <- theta_pattern[seq_len(p)]
  nu <- nu_pattern[seq_len(p)]
  Lambda <- matrix(0, p, 2L)
  Lambda[cbind(seq_len(p), factor_of)] <- lambda
  Phi <- list(
    matrix(c(1.00, .50, .50, 1.00), 2L),
    matrix(c(1.45, .50 * sqrt(1.45 * .85), .50 * sqrt(1.45 * .85), .85), 2L))
  list(p = p, m = m, ov = ov, factor_of = factor_of, marker = marker,
       Lambda = Lambda, theta = theta, nu = nu, Phi = Phi,
       alpha = list(c(0, 0), c(.30, -.20)))
}

main_group_moments <- function(pop, group, step = NULL, delta = 0) {
  Lambda <- pop$Lambda
  theta <- pop$theta
  nu <- pop$nu
  target <- pop$p
  if (!is.null(step) && group == 2L) {
    switch(step,
      metric = Lambda[target, pop$factor_of[target]] <- Lambda[target, pop$factor_of[target]] + delta,
      scalar = nu[target] <- nu[target] + delta,
      strict = theta[target] <- theta[target] * (1 + delta),
      stop("unknown invariance step: ", step, call. = FALSE))
  }
  list(Sigma = Lambda %*% pop$Phi[[group]] %*% t(Lambda) + diag(theta),
       mean = nu + drop(Lambda %*% pop$alpha[[group]]))
}

main_syntax <- function(pop, level) {
  block <- function(factor, items, labels = FALSE) {
    rhs <- pop$ov[items]
    if (labels) rhs[-1L] <- paste0("L", items[-1L], "*", rhs[-1L])
    paste0("f", factor, " =~ ", paste(rhs, collapse = " + "))
  }
  items <- split(seq_len(pop$p), pop$factor_of)
  configural <- paste(vapply(seq_along(items), function(j)
    block(j, items[[j]], FALSE), character(1)), collapse = "\n")
  metric <- paste(vapply(seq_along(items), function(j)
    block(j, items[[j]], TRUE), character(1)), collapse = "\n")
  means <- paste0("f", 1:2, " ~ c(0, NA)*1", collapse = "\n")
  anchors <- paste0(pop$ov[pop$marker], " ~ i", pop$marker, "*1", collapse = "\n")
  intercepts <- paste0(pop$ov, " ~ i", seq_len(pop$p), "*1", collapse = "\n")
  residuals <- paste0(pop$ov, " ~~ e", seq_len(pop$p), "*", pop$ov, collapse = "\n")
  switch(level,
    configural = configural,
    metric = metric,
    metric_chart = paste(metric, means, anchors, sep = "\n"),
    scalar = paste(metric, means, intercepts, sep = "\n"),
    strict = paste(metric, means, intercepts, residuals, sep = "\n"),
    stop("unknown model level: ", level, call. = FALSE))
}

step_models <- function(step) {
  switch(step,
    metric = c(h1 = "configural", h0 = "metric"),
    scalar = c(h1 = "metric_chart", h0 = "scalar"),
    strict = c(h1 = "scalar", h0 = "strict"),
    stop("unknown invariance step: ", step, call. = FALSE))
}
