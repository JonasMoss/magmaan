# Value semantics preserve dataset equality checks used by nested inference.
.deletion_object <- function(policy, known, per_group) {
  structure(list(policy = policy, known = known, per_group = per_group),
            class = "magmaan_deletion")
}

`$<-.magmaan_deletion` <- function(x, name, value) {
  stop("deletion provenance is read-only", call. = FALSE)
}
`[[<-.magmaan_deletion` <- function(x, ..., value) {
  stop("deletion provenance is read-only", call. = FALSE)
}
`[<-.magmaan_deletion` <- function(x, ..., value) {
  stop("deletion provenance is read-only", call. = FALSE)
}

# Row accounting is inspection metadata, independent of SEM calculations.
# Reason counts are exclusive: group, cluster, then model-variable missingness.
.deletion_record <- function(x, ov, group_var = "", labels = character(),
                             policy = "listwise", cluster = NULL) {
  if (!is.list(ov)) ov <- list(ov)
  grouped <- length(group_var) && nzchar(group_var)
  g <- if (grouped) as.character(x[[group_var]]) else rep("all", nrow(x))
  if (!grouped) labels <- "all"
  if (!length(labels)) labels <- unique(g[!is.na(g)])
  rows <- lapply(labels, function(label) which(!is.na(g) & g == label))
  reason <- rep("", nrow(x))
  reason[is.na(g)] <- "missing_group_label"
  reason[!is.na(g) & !g %in% labels] <- "unknown_group_label"
  deletes <- identical(policy, "listwise")
  if (deletes) {
    if (!is.null(cluster)) reason[reason == "" & is.na(cluster)] <- "missing_cluster_id"
    for (b in seq_along(rows)) {
      ii <- rows[[b]]
      bad <- !stats::complete.cases(x[ii, ov[[b]], drop = FALSE])
      reason[ii[bad & reason[ii] == ""]] <- "missing_model_values"
    }
  }
  # Unassignable group labels cannot be attributed to a model group.
  unassigned <- which(is.na(g) | !g %in% labels)
  if (length(unassigned)) {
    rows <- c(rows, list(unassigned))
    labels <- c(labels, NA_character_)
  }
  per_group <- do.call(rbind, Map(function(ii, label) {
    deleted <- if (deletes) sum(nzchar(reason[ii])) else 0L
    data.frame(group = label, rows_supplied = length(ii),
      rows_used = length(ii) - deleted, rows_deleted = deleted,
      missing_model_values = if (deletes) sum(reason[ii] == "missing_model_values") else 0L,
      missing_group_label = if (deletes) sum(reason[ii] == "missing_group_label") else 0L,
      unknown_group_label = if (deletes) sum(reason[ii] == "unknown_group_label") else 0L,
      missing_cluster_id = if (deletes) sum(reason[ii] == "missing_cluster_id") else 0L)
  }, rows, labels))
  .deletion_object(policy = policy, known = TRUE, per_group = per_group)
}

.retain_deletion <- function(fit, data, policy = NULL) {
  record <- if (is.list(data) || is.environment(data)) data$deletion else NULL
  structured <- is.list(data) || is.environment(data)
  if (is.null(record) && structured && !is.null(data$X)) {
    X <- data$X
    labels <- data$group_labels
    if (!length(labels)) labels <- if (length(X) == 1L) "all" else as.character(seq_along(X))
    per_group <- data.frame(group = labels, rows_supplied = vapply(X, nrow, integer(1)),
                            rows_used = vapply(X, nrow, integer(1)), rows_deleted = 0L)
    # Matrix inputs describe only the rows supplied to this call; earlier
    # deletion outside the package cannot be recovered.
    for (reason in c("missing_model_values", "missing_group_label",
                     "unknown_group_label", "missing_cluster_id")) per_group[[reason]] <- 0L
    record <- .deletion_object(policy = policy %||% if (inherits(data, "magmaan_fiml_data")) "fiml" else "supplied_rows",
                               known = TRUE, per_group = per_group)
  }
  if (is.null(record)) {
    per_group <- data.frame(group = as.character(seq_along(fit$nobs)),
      rows_supplied = NA_integer_, rows_used = as.integer(fit$nobs),
      rows_deleted = NA_integer_)
    for (reason in c("missing_model_values", "missing_group_label",
                     "unknown_group_label", "missing_cluster_id")) per_group[[reason]] <- NA_integer_
    record <- .deletion_object(policy = "summary_input", known = FALSE, per_group = per_group)
  }
  fit$deletion <- record
  # Retained statistical inputs are compared by nested inference. Their
  # equality must depend on observations, not preparation history.
  for (field in c("raw_data", "ordinal_stats", "mixed_ordinal_stats")) {
    if (is.list(fit[[field]])) fit[[field]]$deletion <- NULL
  }
  fit
}
