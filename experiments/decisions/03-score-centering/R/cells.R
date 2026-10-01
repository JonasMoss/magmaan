# Populations are local to this decision; no sibling-study inputs.
centering_cells <- function() {
  families <- data.frame(
    family = c('global_normal', 'global_skew', 'nested_normal', 'nested_skew',
               'grouped_mean', 'fiml_global_mcar', 'fiml_global_mar',
               'fiml_nested_mcar', 'fiml_nested_mar'),
    lane = c(rep('ml', 5), rep('fiml', 4)),
    test = c('global', 'global', 'nested', 'nested', 'nested',
             'global', 'global', 'nested', 'nested'),
    innovation = c('normal', 'skew', 'normal', 'skew', rep('normal', 5)),
    missing = c(rep('none', 5), 'mcar', 'mar', 'mcar', 'mar'))
  out <- do.call(rbind, lapply(seq_len(nrow(families)), function(i) {
    grid <- expand.grid(n = c(80L, 300L), role = c('null', 'power'),
                        stringsAsFactors = FALSE)
    cbind(families[rep(i, nrow(grid)), ], grid)
  }))
  rownames(out) <- NULL
  out$cell_id <- seq_len(nrow(out))
  out
}

centering_population <- function(cell, seed) {
  set.seed(seed)
  n <- cell$n
  shift <- if (cell$role == 'power') .15 else 0
  if (cell$family == 'grouped_mean') {
    group <- rep(c('a', 'b'), c(n / 4, 3 * n / 4))
    data <- data.frame(x1 = rnorm(n) + shift + ifelse(group == 'a', .6, -.2),
                       x2 = rnorm(n) + .3, group = group)
    syntax <- paste('x1 ~ c(m,m)*1', 'x2 ~ c(u,u)*1',
                    'x1 ~~ c(1,1)*x1', 'x2 ~~ c(1,1)*x2',
                    'x1 ~~ c(0,0)*x2', sep = '\n')
    return(list(data = data, syntax = syntax, null = paste(syntax, 'm == 0', sep = '\n'),
                label = 'm', target = shift, oracle_variance = 1 / n, group = 'group'))
  }
  innovation <- function(size) {
    if (cell$innovation == 'skew') (rchisq(size, 3) - 3) / sqrt(6) else rnorm(size)
  }
  lambda <- c(.65, .75, .75, .60, .80, .70)
  if (cell$test == 'nested' && cell$role == 'power') lambda[3] <- .90
  errors <- matrix(innovation(n * 6), n, 6)
  if (cell$test == 'global' && cell$role == 'power')
    errors[, 4] <- .18 * errors[, 1] + sqrt(1 - .18^2) * errors[, 4]
  x <- outer(innovation(n), lambda) + sweep(errors, 2, sqrt(1 - lambda^2), '*')
  data <- as.data.frame(x)
  names(data) <- paste0('x', 1:6)
  if (cell$missing != 'none') {
    p2 <- if (cell$missing == 'mar') plogis(-.9 + .8 * data$x1) else rep(.25, n)
    p5 <- if (cell$missing == 'mar') plogis(-1 + .6 * data$x1) else rep(.30, n)
    data$x2[runif(n) < p2] <- NA_real_
    data$x5[runif(n) < p5] <- NA_real_
  }
  syntax <- 'f =~ x1 + a*x2 + b*x3 + x4 + x5 + x6'
  list(data = data, syntax = syntax, null = paste(syntax, 'a == b', sep = '\n'),
       label = 'a', target = if (cell$test == 'global' && cell$role == 'power') NA_real_ else .75,
       oracle_variance = NA_real_, group = NULL)
}
