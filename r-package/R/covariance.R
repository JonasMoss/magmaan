# Covariance domain and penalty are distinct internal choices. The friendly
# policy selects one; legacy psd= remains a compatible spelling.
.fixed_weight_method <- function(weight) {
  key <- match.arg(tolower(weight), c("uls", "nt", "gls", "dwls", "wls", "adf", "dls", "custom"))
  switch(key, uls = "ULS", nt = "GLS", gls = "GLS", dwls = "DWLS", dls = "DLS", "WLS")
}

.fixed_weight_options <- function(method, W = NULL, dls_a = 0.5) {
  if (!is.null(W) && !method %in% c("WLS", "DWLS")) stop("supplied W requires WLS or DWLS")
  if (method == "DWLS" && !is.null(W)) {
    blocks <- if (is.matrix(W)) list(W) else W
    if (!is.list(blocks) || any(vapply(blocks, function(x)
      !is.matrix(x) || nrow(x) != ncol(x) || any(x[row(x) != col(x)] != 0), logical(1))))
      stop("DWLS requires diagonal W")
  }
  if (method == "DLS" && (!is.numeric(dls_a) || length(dls_a) != 1L ||
      !is.finite(dls_a) || dls_a < 0 || dls_a > 1)) stop("dls_a must be finite and in [0, 1]")
  invisible(NULL)
}

.covariance_options <- function(covariance, psd = FALSE, psd_supplied = FALSE, barrier = NULL) {
  if (!is.logical(psd) || length(psd) != 1L || is.na(psd)) stop("psd must be TRUE or FALSE")
  if (is.null(covariance)) covariance <- if (psd) "psd" else "unrestricted"
  covariance <- match.arg(covariance, c("unrestricted", "psd", "barrier"))
  if (psd_supplied && !identical(psd, identical(covariance, "psd")))
    stop("covariance and psd specify different policies")
  if (!identical(covariance, "barrier") && !is.null(barrier))
    stop("barrier options require covariance = 'barrier'")
  if (identical(covariance, "barrier")) {
    if (is.null(barrier)) barrier <- list()
    if (!is.list(barrier) || anyDuplicated(names(barrier)) || length(setdiff(names(barrier), c("target", "weight"))) ||
        (length(barrier) && (is.null(names(barrier)) || any(!nzchar(names(barrier))))))
      stop("barrier must be a list with target and/or weight")
    barrier <- list(target = match.arg(barrier$target %||% "joint", c("joint", "determinacy")),
                    weight = barrier$weight %||% 0.25)
    if (!is.numeric(barrier$weight) || length(barrier$weight) != 1L ||
        !is.finite(barrier$weight) || barrier$weight < 0)
      stop("barrier weight must be a finite non-negative number")
  }
  list(covariance = covariance, psd = identical(covariance, "psd"), barrier = barrier)
}

.finish_covariance_fit <- function(fit, source, covariance, barrier = NULL, algorithm = NULL) {
  composition <- fit$composition %||% list()
  composition$moment_source <- source
  composition$moment_target <- if (isTRUE(fit$ordinal)) "correlation" else if (isTRUE(fit$fiml))
    "observed_patterns" else "covariance_and_means"
  composition$discrepancy <- if (identical(source, "saturated_fiml")) {
    switch(fit$stage2_weight %||% "nt", nt = "ML", uls = "ULS", dwls = "DWLS", "WLS")
  } else fit$estimator
  if (fit$estimator %in% c("ULS", "GLS", "DWLS", "WLS", "DLS")) {
    composition$weight <- fit$moment_weight %||% switch(fit$estimator,
        ULS = "uls", GLS = "nt", DWLS = "dwls", WLS = "adf", DLS = "dls")
    composition$weight_frozen <- TRUE
    if (identical(composition$weight, "dls")) composition$dls_a <- fit$stage2_dls_a
  }
  composition$covariance_domain <- if (identical(covariance, "barrier")) {
    if (barrier$weight == 0) "unrestricted" else "barrier_interior"
  } else covariance
  composition$model_penalty <- if (is.null(barrier)) "none" else barrier$target
  composition$algorithm <- composition$algorithm %||% algorithm
  if (!is.null(barrier)) {
    composition$penalty_weight <- barrier$weight
    composition$penalty_scale <- "weight_over_n_total"
    composition$inference <- "not_validated"
    fit$penalty_inference <- "not_validated"
  }
  if (identical(source, "pairwise_mcar")) composition$inference <- "not_validated"
  fit$composition <- composition
  fit$covariance_policy <- covariance
  fit
}
