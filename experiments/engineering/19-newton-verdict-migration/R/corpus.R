# One continuous textbook-corpus case as a magmaanlab model, its sample
# moments (prepared by lavaan, so moments match the book's lavaan setup), and
# its raw data when the case has it.
read_case <- function(root, dir) {
  meta <- jsonlite::fromJSON(file.path(root, dir, "meta.json"), simplifyVector = TRUE)
  mo <- meta$model_options
  skip <- function(why) stop("excluded: ", why, call. = FALSE)
  if (isTRUE(meta$out_of_scope)) skip("out of scope")
  if (length(mo$ordered)) skip("categorical")
  if (length(mo$mimic) && mo$mimic != "lavaan") skip("non-lavaan mimic")
  syntax <- paste(readLines(file.path(root, dir, "model.lav"), warn = FALSE), collapse = "\n")
  if (grepl("<~|level:", syntax)) skip("composite or multilevel")
  args <- list(model = syntax, meanstructure = isTRUE(mo$meanstructure),
               fixed.x = isTRUE(mo$fixed_x), do.fit = FALSE)
  for (key in c("group_equal", "group_partial"))
    if (length(mo[[key]])) args[[gsub("_", ".", key)]] <- mo[[key]]
  raw <- NULL
  missing_data <- identical(mo$missing, "fiml")
  if (identical(meta$data$kind, "raw")) {
    raw <- utils::read.csv(file.path(root, dir, meta$data$files$raw), check.names = FALSE)
    args$data <- raw
    if (length(meta$data$group_var)) args$group <- meta$data$group_var
    args$missing <- if (missing_data) "ml" else "listwise"
  } else {
    if (missing_data) skip("missing data without raw data")
    args$sample.cov <- lapply(meta$data$files$sample_cov, function(f)
      as.matrix(utils::read.csv(file.path(root, dir, f), row.names = 1L, check.names = FALSE)))
    args$sample.nobs <- as.integer(meta$data$n_obs)
    if (length(meta$data$files$sample_mean)) args$sample.mean <-
      lapply(meta$data$files$sample_mean, function(f) {
        z <- utils::read.csv(file.path(root, dir, f), check.names = FALSE)
        stats::setNames(z[[2]], z[[1]])
      })
    if (length(args$sample.cov) == 1L) {
      args$sample.cov <- args$sample.cov[[1]]
      if (length(args$sample.mean)) args$sample.mean <- args$sample.mean[[1]]
    }
  }
  fun <- getExportedValue("lavaan", meta$lavaan_function %||% "sem")
  pre <- suppressWarnings(do.call(fun, args))
  ng <- lavaan::lavInspect(pre, "ngroups")
  ss <- lavaan::lavInspect(pre, "sampstat")
  if (ng == 1L) ss <- list(ss)
  if (any(vapply(ss, function(s) min(eigen(s$cov, TRUE, only.values = TRUE)$values) <= 0,
                 logical(1)))) skip("sample covariance not positive definite")
  spec <- list(syntax = syntax, meanstructure = isTRUE(mo$meanstructure),
               fixed_x = isTRUE(mo$fixed_x),
               auto_cov_y = meta$lavaan_function %in% c("sem", "cfa", "growth"))
  if (identical(meta$lavaan_function, "growth")) spec$model_type <- "growth"
  if (ng > 1L) {
    spec$group <- "group"
    spec$group_labels <- as.character(seq_len(ng))
  }
  for (key in c("group_equal", "group_partial")) if (length(mo[[key]])) spec[[key]] <- mo[[key]]
  model <- do.call(magmaanlab::model_spec, spec)
  # Only models whose free-parameter pattern matches lavaan's are compared.
  mp <- model$partable
  lp <- lavaan::parTable(pre)
  key <- function(p) {
    l <- p$lhs; r <- p$rhs; cv <- p$op == "~~"
    l[cv] <- pmin(p$lhs[cv], p$rhs[cv]); r[cv] <- pmax(p$lhs[cv], p$rhs[cv])
    paste(l, p$op, r, p$group, sep = "|")
  }
  rows <- mp$op %in% c("=~", "~~", "~", "~1")
  idx <- match(key(mp[rows, ]), key(lp))
  if (anyNA(idx) || any((mp$free[rows] > 0) != (lp$free[idx] > 0)))
    skip("free-parameter pattern differs from lavaan")
  list(model = model, missing_data = missing_data, n_groups = ng,
       sample = list(S = lapply(ss, `[[`, "cov"), nobs = as.integer(lavaan::lavInspect(pre, "nobs")),
                     mean = if (isTRUE(mo$meanstructure)) lapply(ss, `[[`, "mean") else NULL),
       raw = if (!is.null(raw) && ng == 1L) raw[, lavaan::lavNames(pre, "ov"), drop = FALSE] else NULL)
}
