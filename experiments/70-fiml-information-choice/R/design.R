fiml_information_model <- paste(
  "f1 =~ 1*y1 + y2 + y3",
  "f2 =~ 1*y4 + y5 + y6",
  sep = "\n"
)

fiml_information_parameters <- data.frame(
  parameter = c("loading_y5", "factor_covariance"),
  lhs = c("f2", "f1"),
  op = c("=~", "~~"),
  rhs = c("y5", "f2"),
  target = c(0.9, 0.224),
  stringsAsFactors = FALSE
)

fiml_information_methods <- data.frame(
  method = c(
    "expected_model", "expected_sandwich",
    "observed_h1_model", "observed_h1_sandwich",
    "observed_hessian_model", "observed_hessian_sandwich"
  ),
  information = rep(c("expected", "observed_h1", "observed_hessian"), each = 2L),
  covariance = rep(c("model", "sandwich"), 3L),
  lavaan_information = rep(c("expected", "observed", "observed"), each = 2L),
  lavaan_observed_information =
    rep(c("hessian", "h1", "hessian"), each = 2L),
  lavaan_se = rep(c("standard", "robust.huber.white"), 3L),
  stringsAsFactors = FALSE
)

fiml_information_design <- function(profile = "overnight") {
  grid <- expand.grid(
    distribution = c("normal", "t5"),
    specification = c("correct", "omitted_crossloading"),
    missingness = c("complete", "mcar30", "mar30"),
    n = c(150L, 300L, 600L, 1200L),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  grid <- grid[order(
    grid$n, grid$distribution, grid$specification, grid$missingness), ]
  row.names(grid) <- NULL
  grid$cell_id <- seq_len(nrow(grid))
  grid$crossloading <- ifelse(
    grid$specification == "correct", 0, 0.25)

  if (profile == "smoke") {
    keep <- with(grid,
      (distribution == "normal" & specification == "correct" &
         missingness == "complete" & n == 300L) |
      (distribution == "normal" & specification == "correct" &
         missingness == "mar30" & n == 300L) |
      (distribution == "t5" & specification == "omitted_crossloading" &
         missingness == "mcar30" & n == 300L) |
      (distribution == "t5" & specification == "correct" &
         missingness == "mar30" & n == 150L))
    grid <- grid[keep, , drop = FALSE]
  } else if (profile == "pilot") {
    keep <- grid$n %in% c(150L, 600L) &
      ((grid$specification == "correct" &
          grid$missingness %in% c("complete", "mar30")) |
       (grid$specification == "omitted_crossloading" &
          grid$missingness == "mcar30"))
    grid <- grid[keep, , drop = FALSE]
  } else if (profile != "overnight") {
    stop("unknown design profile: ", profile, call. = FALSE)
  }
  row.names(grid) <- NULL
  grid
}

fiml_population_covariance <- function(crossloading = 0) {
  lambda <- rbind(
    c(1.0, 0.0),
    c(0.9, 0.0),
    c(0.8, 0.0),
    c(crossloading, 1.0),
    c(0.0, 0.9),
    c(0.0, 0.8)
  )
  phi <- matrix(c(0.64, 0.224, 0.224, 0.64), 2L, 2L)
  systematic <- lambda %*% phi %*% t(lambda)
  residual <- 1 - diag(systematic)
  if (any(residual <= 0)) {
    stop("cross-loading makes a residual variance non-positive",
         call. = FALSE)
  }
  sigma <- systematic + diag(residual)
  dimnames(sigma) <- list(paste0("y", 1:6), paste0("y", 1:6))
  sigma
}

validate_fiml_information_design <- function() {
  grid <- fiml_information_design("overnight")
  stopifnot(
    nrow(grid) == 48L,
    identical(sort(unique(grid$n)), c(150L, 300L, 600L, 1200L)),
    all(vapply(grid$crossloading, function(x) {
      min(eigen(
        fiml_population_covariance(x), symmetric = TRUE,
        only.values = TRUE)$values) > 1e-8
    }, logical(1))),
    nrow(fiml_information_methods) == 6L,
    nrow(fiml_information_parameters) == 2L
  )
  invisible(TRUE)
}
