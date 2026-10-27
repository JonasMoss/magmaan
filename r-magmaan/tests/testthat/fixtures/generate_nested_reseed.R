# Regenerate with lavaan 0.7.2 only; TASK-164.1. Run from the repository root.
stopifnot(utils::packageVersion("lavaan") == "0.7.2")
# Independently reconstruct decisions/04 cell 17, rep 542, seed 726270583.
# The test owns its formulas and never reads the experiment leaf.
nested_reseed_data <- function(seed = 726270583L, skewed = FALSE) {
  loading <- rep(c(.7, .8, .75, .65), 2)
  residual <- 1 - loading^2
  residual[5] <- residual[5] - .4^2 - 2*.4*loading[5]*.4
  population <- paste(c(
    paste0('f1 =~ ', paste0(loading[1:4], '*x', 1:4, collapse = ' + ')),
    paste0('f2 =~ ', paste0(loading[5:8], '*x', 5:8, collapse = ' + ')),
    'f1 =~ .4*x5', 'f1 ~~ 1*f1', 'f2 ~~ 1*f2', 'f1 ~~ .4*f2',
    paste0('x', 1:8, ' ~~ ', residual, '*x', 1:8),
    paste0('x1 ~~ ', .3 * sqrt(residual[1]*residual[5]), '*x5')), collapse = '\n')
  set.seed(seed)
  do.call(rbind, lapply(c('a', 'b'), function(g) {
    x <- lavaan::simulateData(population, sample.nobs = 100,
      skewness = if (skewed) rep(2, 8) else NULL,
      kurtosis = if (skewed) rep(7, 8) else NULL)
    x$group <- g
    x
  }))
}

options(digits = 17)
data <- nested_reseed_data()
path <- "r-magmaan/tests/testthat/fixtures/nested_reseed.csv"
serialized <- data
serialized[vapply(data, is.numeric, logical(1))] <- lapply(
  data[vapply(data, is.numeric, logical(1))], function(x) sprintf("%.17g", x))
write.table(serialized, path, sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
stopifnot(identical(read.csv(path), data))
