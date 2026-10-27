# Keep live categorical inference on the lavaan-0.7.2 preset's bread.
lavaan_at_least_073 <- function() {
  requireNamespace("lavaan", quietly = TRUE) &&
    utils::packageVersion("lavaan") >= "0.7.3"
}

lavaan_expected_bread <- function() {
  if (lavaan_at_least_073()) list(information.bread = "expected") else list()
}

lavaan_categorical_fit <- function(fun, ...) {
  do.call(fun, c(list(...), lavaan_expected_bread()))
}

skip_lavaan_mixed_gamma <- function() {
  testthat::skip_if(lavaan_at_least_073(),
    "lavaan >= 0.7-3 changed its mixed DWLS weights (TASK-164); DWLS parity targets lavaan 0.7-2")
}
