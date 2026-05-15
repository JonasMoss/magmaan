model_spec <- function(syntax,
                       ...,
                       auto_var = TRUE,
                       auto_cov_lv_x = TRUE,
                       auto_cov_y = FALSE,
                       auto_fix_first = TRUE,
                       std_lv = FALSE,
                       effect_coding = FALSE,
                       fixed_x = TRUE,
                       meanstructure = FALSE,
                       ordered = NULL,
                       parameterization = "delta",
                       group = NULL,
                       group_labels = NULL) {
  dots <- list(...)
  if (length(dots)) {
    stop("model_spec(): unused arguments: ", paste(names(dots), collapse = ", "))
  }
  parameterization <- match.arg(parameterization, c("delta", "theta"))
  ordered <- if (is.null(ordered)) character() else as.character(ordered)
  group_var <- if (is.null(group)) "" else as.character(group)[1L]
  n_groups <- if (is.null(group_labels)) 1L else length(group_labels)
  if (nzchar(group_var) && n_groups < 1L) n_groups <- 1L

  opts <- list(
    auto_var = auto_var,
    auto_cov_lv_x = auto_cov_lv_x,
    auto_cov_y = auto_cov_y,
    auto_fix_first = auto_fix_first,
    std_lv = std_lv,
    effect_coding = effect_coding,
    fixed_x = fixed_x,
    meanstructure = meanstructure,
    ordered = ordered,
    parameterization = parameterization
  )
  partable <- lavaan_lavaanify(
    syntax,
    auto_var = auto_var,
    auto_cov_lv_x = auto_cov_lv_x,
    auto_cov_y = auto_cov_y,
    auto_fix_first = auto_fix_first,
    std_lv = std_lv,
    effect_coding = effect_coding,
    fixed_x = fixed_x,
    meanstructure = meanstructure,
    n_groups = n_groups,
    group_var = group_var,
    group_labels = group_labels
  )
  attr(partable, "magmaan.ordered") <- ordered
  attr(partable, "magmaan.parameterization") <- parameterization

  out <- list(
    syntax = syntax,
    partable = partable,
    options = opts,
    ordered = ordered,
    parameterization = parameterization,
    group_var = group_var,
    group_labels = group_labels
  )
  class(out) <- c("magmaan_model_spec", "list")
  out
}

as_magmaan_model_spec <- function(model) {
  if (inherits(model, "magmaan_model_spec")) return(model)
  if (is.character(model) && length(model) == 1L) return(model_spec(model))
  if (is.data.frame(model)) {
    out <- list(
      syntax = NULL,
      partable = model,
      options = list(),
      ordered = attr(model, "magmaan.ordered", exact = TRUE) %||% character(),
      parameterization = attr(model, "magmaan.parameterization", exact = TRUE) %||% "delta",
      group_var = attr(model, "magmaan.group_var", exact = TRUE) %||% "",
      group_labels = attr(model, "magmaan.group_labels", exact = TRUE)
    )
    class(out) <- c("magmaan_model_spec", "list")
    return(out)
  }
  stop("expected a magmaan model spec, model syntax string, or partable data.frame")
}

lavaan_compare_partable <- function(x, reference, est_tolerance = NULL,
                                    ustart_tolerance = 1e-12) {
  as_pt <- function(z, arg) {
    if (inherits(z, "magmaan_model_spec")) return(z$partable)
    if (is.list(z) && !is.data.frame(z) && !is.null(z$partable)) return(z$partable)
    if (is.data.frame(z)) return(z)
    stop("lavaan_compare_partable(): `", arg, "` must be a partable data.frame, ",
         "magmaan model spec, or magmaan fit list")
  }
  got <- as_pt(x, "x")
  exp <- as_pt(reference, "reference")

  required <- c("lhs", "op", "rhs", "group", "free", "ustart", "label", "plabel")
  missing_got <- setdiff(required, names(got))
  missing_exp <- setdiff(required, names(exp))
  if (length(missing_got)) {
    stop("lavaan_compare_partable(): `x` is missing columns: ",
         paste(missing_got, collapse = ", "))
  }
  if (length(missing_exp)) {
    stop("lavaan_compare_partable(): `reference` is missing columns: ",
         paste(missing_exp, collapse = ", "))
  }

  text_col <- function(pt, col) {
    if (!col %in% names(pt)) return(rep("", nrow(pt)))
    out <- as.character(pt[[col]])
    out[is.na(out)] <- ""
    out
  }
  int_col <- function(pt, col) {
    if (!col %in% names(pt)) return(rep(0L, nrow(pt)))
    out <- suppressWarnings(as.integer(pt[[col]]))
    out[is.na(out)] <- 0L
    out
  }
  num_col <- function(pt, col) {
    if (!col %in% names(pt)) return(rep(NA_real_, nrow(pt)))
    suppressWarnings(as.numeric(pt[[col]]))
  }

  normalize <- function(pt) {
    data.frame(
      .row = seq_len(nrow(pt)),
      lhs = text_col(pt, "lhs"),
      op = text_col(pt, "op"),
      rhs = text_col(pt, "rhs"),
      user = int_col(pt, "user"),
      block = int_col(pt, "block"),
      group = int_col(pt, "group"),
      free = int_col(pt, "free"),
      exo = int_col(pt, "exo"),
      ustart = num_col(pt, "ustart"),
      label = text_col(pt, "label"),
      plabel = text_col(pt, "plabel"),
      est = num_col(pt, "est"),
      stringsAsFactors = FALSE
    )
  }

  g <- normalize(got)
  e <- normalize(exp)
  key <- function(pt, i, include_group = TRUE) {
    prefix <- if (include_group) paste0("group=", pt$group[[i]], "|") else ""
    paste0(prefix, "op=", pt$op[[i]], "|lhs=", pt$lhs[[i]], "|rhs=", pt$rhs[[i]])
  }
  row_label <- function(pt, i) {
    paste0("[", key(pt, i), ", free=", pt$free[[i]],
           ", label=", pt$label[[i]], ", plabel=", pt$plabel[[i]], "]")
  }
  same_num <- function(a, b, tol) {
    if (is.na(a) || is.na(b)) return(is.na(a) && is.na(b))
    abs(a - b) <= tol
  }
  fixed_zero <- function(pt, i) {
    pt$free[[i]] == 0L &&
      ((!is.na(pt$ustart[[i]]) && abs(pt$ustart[[i]]) <= ustart_tolerance) ||
         (!is.na(pt$est[[i]]) && abs(pt$est[[i]]) <= (est_tolerance %||% 1e-12)))
  }
  score <- function(gi, ei) {
    sum(g[gi, c("user", "block", "group", "free", "exo")] ==
          e[ei, c("user", "block", "group", "free", "exo")]) +
      2L * (g$label[[gi]] == e$label[[ei]]) +
      2L * (g$plabel[[gi]] == e$plabel[[ei]]) +
      as.integer(same_num(g$ustart[[gi]], e$ustart[[ei]], ustart_tolerance))
  }
  best_match <- function(ei, used, include_group) {
    target <- key(e, ei, include_group)
    candidates <- which(!used & vapply(seq_len(nrow(g)), function(gi) {
      identical(key(g, gi, include_group), target)
    }, logical(1)))
    if (!length(candidates)) return(NA_integer_)
    candidates[[which.max(vapply(candidates, score, numeric(1), ei = ei))]]
  }

  failures <- data.frame(category = character(), detail = character(),
                         stringsAsFactors = FALSE)
  add_failure <- function(category, detail) {
    failures <<- rbind(failures, data.frame(category = category, detail = detail,
                                            stringsAsFactors = FALSE))
  }
  add_drift <- function(category, field, gi, ei) {
    add_failure(category, paste0(field, " for ", row_label(e, ei),
                                 ": got=", as.character(g[[field]][[gi]]),
                                 " expected=", as.character(e[[field]][[ei]])))
  }

  used <- rep(FALSE, nrow(g))
  for (ei in seq_len(nrow(e))) {
    gi <- best_match(ei, used, include_group = TRUE)
    if (is.na(gi)) {
      gi <- best_match(ei, used, include_group = FALSE)
      if (is.na(gi)) {
        add_failure("missing row", row_label(e, ei))
        next
      }
      add_failure("group drift", paste0(row_label(e, ei), ": got group=",
                                        g$group[[gi]], " expected group=", e$group[[ei]]))
    }
    used[[gi]] <- TRUE

    if (!identical(g$label[[gi]], e$label[[ei]])) add_drift("label drift", "label", gi, ei)
    if (!identical(g$plabel[[gi]], e$plabel[[ei]])) add_drift("plabel drift", "plabel", gi, ei)
    if (!identical(g$free[[gi]], e$free[[ei]])) add_drift("free-index drift", "free", gi, ei)
    if (!same_num(g$ustart[[gi]], e$ustart[[ei]], ustart_tolerance)) {
      add_failure("ustart drift", paste0(row_label(e, ei), ": got=", g$ustart[[gi]],
                                         " expected=", e$ustart[[ei]]))
    }
    for (field in c("user", "block", "exo")) {
      if (!identical(g[[field]][[gi]], e[[field]][[ei]])) {
        add_drift("metadata drift", field, gi, ei)
      }
    }
    if (!is.null(est_tolerance) && "est" %in% names(got) && "est" %in% names(exp) &&
        !same_num(g$est[[gi]], e$est[[ei]], est_tolerance)) {
      add_failure("estimate drift", paste0(row_label(e, ei), ": got=", g$est[[gi]],
                                           " expected=", e$est[[ei]]))
    }
  }

  for (gi in which(!used)) {
    add_failure(if (fixed_zero(g, gi)) "extra fixed-zero row" else "extra row",
                row_label(g, gi))
  }

  counts <- if (nrow(failures)) table(failures$category) else integer()
  structure(list(ok = nrow(failures) == 0L,
                 failures = failures,
                 counts = counts),
            class = "magmaan_partable_comparison")
}

data_ordinal_stats_from_df <- function(x, model, ordered = NULL, group = NULL,
                                       missing = c("listwise", "error")) {
  missing <- match.arg(missing)
  if (!is.data.frame(x)) stop("data_ordinal_stats_from_df(): `x` must be a data.frame")
  model <- as_magmaan_model_spec(model)
  ordered <- if (is.null(ordered)) model$ordered else as.character(ordered)
  if (!length(ordered)) stop("data_ordinal_stats_from_df(): `ordered` must name ordered variables")

  group_var <- if (is.null(group)) model$group_var else as.character(group)[1L]
  if (is.null(group_var) || !nzchar(group_var)) group_var <- ""
  labels <- character()
  if (nzchar(group_var)) {
    if (!group_var %in% names(x)) stop("data_ordinal_stats_from_df(): grouping column not found: ", group_var)
    g <- x[[group_var]]
    if (anyNA(g)) stop("data_ordinal_stats_from_df(): grouping column contains missing values")
    labels <- model$group_labels
    if (is.null(labels) || !length(labels)) labels <- if (is.factor(g)) levels(g) else unique(as.character(g))
    labels <- as.character(labels)
  }

  rep <- model_matrix_rep(model$partable)
  ov_by_group <- rep$ov_names
  if (!is.list(ov_by_group)) ov_by_group <- list(ov_by_group)
  if (length(labels) && length(ov_by_group) != length(labels)) {
    stop("data_ordinal_stats_from_df(): model/data group count mismatch")
  }

  make_block <- function(rows, ov, label = NULL) {
    if (!setequal(ordered, ov)) {
      stop("data_ordinal_stats_from_df(): v1 ordinal LS requires all observed variables to be ordered")
    }
    miss_cols <- setdiff(ov, names(x))
    if (length(miss_cols)) stop("data_ordinal_stats_from_df(): data is missing observed variables: ", paste(miss_cols, collapse = ", "))
    block <- x[rows, ov, drop = FALSE]
    cc <- stats::complete.cases(block)
    if (!all(cc)) {
      if (missing == "error") {
        suffix <- if (is.null(label)) "" else paste0(" in group '", label, "'")
        stop("data_ordinal_stats_from_df(): missing observed values", suffix)
      }
      block <- block[cc, , drop = FALSE]
    }
    if (nrow(block) < 2L) stop("data_ordinal_stats_from_df(): fewer than 2 complete rows")
    mat <- matrix(NA_real_, nrow(block), length(ov), dimnames = list(NULL, ov))
    for (j in seq_along(ov)) {
      v <- block[[ov[[j]]]]
      if (is.factor(v)) {
        counts <- tabulate(as.integer(v), nbins = nlevels(v))
        if (any(counts == 0L)) {
          suffix <- if (is.null(label)) "" else paste0(" in group '", label, "'")
          stop("data_ordinal_stats_from_df(): empty ordinal category for ", ov[[j]], suffix)
        }
        mat[, j] <- as.integer(v)
      } else {
        vals <- sort(unique(v))
        mat[, j] <- match(v, vals)
      }
    }
    mat
  }

  if (nzchar(group_var)) {
    g_chr <- as.character(x[[group_var]])
    if (!all(g_chr %in% labels)) {
      stop("data_ordinal_stats_from_df(): data contains groups outside `group_labels`: ",
           paste(setdiff(unique(g_chr), labels), collapse = ", "))
    }
    X <- Map(function(label, ov) make_block(g_chr == label, ov, label), labels, ov_by_group)
    names(X) <- labels
  } else {
    X <- list(make_block(rep(TRUE, nrow(x)), ov_by_group[[1L]]))
  }
  out <- data_ordinal_stats_from_raw_impl(X)
  out$X <- X
  out$ov_names <- ov_by_group
  out$ordered <- ordered
  out$group_var <- group_var
  out$group_labels <- labels
  for (b in seq_along(out$R)) {
    nm <- ov_by_group[[b]]
    dimnames(out$R[[b]]) <- list(nm, nm)
  }
  class(out) <- c("magmaan_ordinal_data", "list")
  out
}

data_mixed_ordinal_stats_from_df <- function(x, model, ordered = NULL, group = NULL,
                                             missing = c("listwise", "error")) {
  missing <- match.arg(missing)
  if (!is.data.frame(x)) stop("data_mixed_ordinal_stats_from_df(): `x` must be a data.frame")
  model <- as_magmaan_model_spec(model)
  ordered <- if (is.null(ordered)) model$ordered else as.character(ordered)
  if (!length(ordered)) stop("data_mixed_ordinal_stats_from_df(): `ordered` must name ordered variables")

  group_var <- if (is.null(group)) model$group_var else as.character(group)[1L]
  if (is.null(group_var) || !nzchar(group_var)) group_var <- ""
  labels <- character()
  if (nzchar(group_var)) {
    if (!group_var %in% names(x)) stop("data_mixed_ordinal_stats_from_df(): grouping column not found: ", group_var)
    g <- x[[group_var]]
    if (anyNA(g)) stop("data_mixed_ordinal_stats_from_df(): grouping column contains missing values")
    labels <- model$group_labels
    if (is.null(labels) || !length(labels)) labels <- if (is.factor(g)) levels(g) else unique(as.character(g))
    labels <- as.character(labels)
  }

  rep <- model_matrix_rep(model$partable)
  ov_by_group <- rep$ov_names
  if (!is.list(ov_by_group)) ov_by_group <- list(ov_by_group)
  if (length(labels) && length(ov_by_group) != length(labels)) {
    stop("data_mixed_ordinal_stats_from_df(): model/data group count mismatch")
  }

  make_block <- function(rows, ov, label = NULL) {
    if (!all(ordered %in% ov)) {
      stop("data_mixed_ordinal_stats_from_df(): ordered variables must be observed model variables")
    }
    if (setequal(ordered, ov)) {
      stop("data_mixed_ordinal_stats_from_df(): use data_ordinal_stats_from_df() for all-ordinal data")
    }
    miss_cols <- setdiff(ov, names(x))
    if (length(miss_cols)) stop("data_mixed_ordinal_stats_from_df(): data is missing observed variables: ", paste(miss_cols, collapse = ", "))
    block <- x[rows, ov, drop = FALSE]
    cc <- stats::complete.cases(block)
    if (!all(cc)) {
      if (missing == "error") {
        suffix <- if (is.null(label)) "" else paste0(" in group '", label, "'")
        stop("data_mixed_ordinal_stats_from_df(): missing observed values", suffix)
      }
      block <- block[cc, , drop = FALSE]
    }
    if (nrow(block) < 2L) stop("data_mixed_ordinal_stats_from_df(): fewer than 2 complete rows")
    mat <- matrix(NA_real_, nrow(block), length(ov), dimnames = list(NULL, ov))
    mask <- integer(length(ov))
    for (j in seq_along(ov)) {
      v <- block[[ov[[j]]]]
      if (ov[[j]] %in% ordered) {
        mask[[j]] <- 1L
        if (is.factor(v)) {
          counts <- tabulate(as.integer(v), nbins = nlevels(v))
          if (any(counts == 0L)) {
            suffix <- if (is.null(label)) "" else paste0(" in group '", label, "'")
            stop("data_mixed_ordinal_stats_from_df(): empty ordinal category for ", ov[[j]], suffix)
          }
          mat[, j] <- as.integer(v)
        } else {
          vals <- sort(unique(v))
          mat[, j] <- match(v, vals)
        }
      } else {
        if (!is.numeric(v)) stop("data_mixed_ordinal_stats_from_df(): continuous variable is not numeric: ", ov[[j]])
        mat[, j] <- as.numeric(v)
      }
    }
    list(X = mat, ordered_mask = mask)
  }

  if (nzchar(group_var)) {
    g_chr <- as.character(x[[group_var]])
    if (!all(g_chr %in% labels)) {
      stop("data_mixed_ordinal_stats_from_df(): data contains groups outside `group_labels`: ",
           paste(setdiff(unique(g_chr), labels), collapse = ", "))
    }
    blocks <- Map(function(label, ov) make_block(g_chr == label, ov, label), labels, ov_by_group)
    X <- lapply(blocks, `[[`, "X")
    ordered_mask <- lapply(blocks, `[[`, "ordered_mask")
    names(X) <- labels
  } else {
    blk <- make_block(rep(TRUE, nrow(x)), ov_by_group[[1L]])
    X <- list(blk$X)
    ordered_mask <- list(blk$ordered_mask)
  }
  out <- data_mixed_ordinal_stats_from_raw_impl(X, ordered_mask)
  out$X <- X
  out$ov_names <- ov_by_group
  out$ordered <- ordered
  out$group_var <- group_var
  out$group_labels <- labels
  for (b in seq_along(out$R)) {
    nm <- ov_by_group[[b]]
    dimnames(out$R[[b]]) <- list(nm, nm)
    names(out$mean[[b]]) <- nm
  }
  class(out) <- c("magmaan_mixed_ordinal_data", "list")
  out
}

augment_ordinal_partable <- function(model, ordinal_stats) {
  parameterization <- attr(partable_arg(model), "magmaan.parameterization",
                           exact = TRUE) %||% "delta"
  if (identical(parameterization, "theta")) {
    stop("augment_ordinal_partable(): theta parameterization is not supported yet; ",
         "use parameterization = 'delta'")
  }
  fix_delta_variances <- function(pt, ov_by_group) {
    for (b in seq_along(ov_by_group)) {
      idx <- pt$op == "~~" &
        pt$lhs == pt$rhs &
        pt$lhs %in% ov_by_group[[b]] &
        pt$group == b
      pt$free[idx] <- 0L
      pt$ustart[idx] <- 1.0
    }
    free_old <- pt$free
    vals <- sort(unique(free_old[free_old > 0L]))
    if (length(vals)) {
      map <- setNames(seq_along(vals), vals)
      pt$free[free_old > 0L] <- unname(map[as.character(free_old[free_old > 0L])])
    }
    pt
  }
  reorder_delta_free <- function(pt) {
    append_unique <- function(ids, add) unique(c(ids, add[!is.na(add) & add > 0L]))
    ids <- integer()
    groups <- sort(unique(pt$group[pt$group > 0L]))
    for (g in groups) {
      in_group <- pt$group == g
      ids <- append_unique(ids, pt$free[in_group & pt$op == "=~"])
      ids <- append_unique(ids, pt$free[in_group & pt$op == "|"])
      ids <- append_unique(ids, pt$free[in_group & !(pt$op %in% c("=~", "|"))])
    }
    if (length(ids)) {
      map <- setNames(seq_along(ids), ids)
      old <- pt$free
      pt$free[old > 0L] <- unname(map[as.character(old[old > 0L])])
    }
    pt
  }

  pt <- partable_arg(model)
  ov_by_group <- ordinal_stats$ov_names
  if (!is.list(ov_by_group)) ov_by_group <- list(ov_by_group)
  pt <- fix_delta_variances(pt, ov_by_group)
  if (any(pt$op == "|")) return(reorder_delta_free(pt))
  required <- names(pt)
  n_new <- sum(vapply(ordinal_stats$n_levels, function(z) sum(as.integer(z) - 1L) + length(z), integer(1)))
  rows <- pt[rep(NA_integer_, n_new), required, drop = FALSE]
  n0 <- nrow(pt)
  next_free <- if (length(pt$free)) max(pt$free, na.rm = TRUE) else 0L
  rr <- 0L
  for (b in seq_along(ov_by_group)) {
    ov <- ov_by_group[[b]]
    th <- ordinal_stats$thresholds[[b]]
    th_pos <- 1L
    for (j in seq_along(ov)) {
      for (lev in seq_len(ordinal_stats$n_levels[[b]][[j]] - 1L)) {
        rr <- rr + 1L
        next_free <- next_free + 1L
        rows[rr, ] <- pt[1L, required, drop = FALSE]
        rows$id[rr] <- n0 + rr
        rows$lhs[rr] <- ov[[j]]
        rows$op[rr] <- "|"
        rows$rhs[rr] <- paste0("t", lev)
        rows$user[rr] <- 0L
        rows$block[rr] <- b
        rows$group[rr] <- b
        rows$free[rr] <- next_free
        rows$exo[rr] <- 0L
        rows$ustart[rr] <- th[[th_pos]]
        rows$label[rr] <- ""
        rows$plabel[rr] <- paste0(".p", n0 + rr, ".")
        th_pos <- th_pos + 1L
      }
      rr <- rr + 1L
      rows[rr, ] <- pt[1L, required, drop = FALSE]
      rows$id[rr] <- n0 + rr
      rows$lhs[rr] <- ov[[j]]
      rows$op[rr] <- "~*~"
      rows$rhs[rr] <- ov[[j]]
      rows$user[rr] <- 0L
      rows$block[rr] <- b
      rows$group[rr] <- b
      rows$free[rr] <- 0L
      rows$exo[rr] <- 0L
      rows$ustart[rr] <- 1.0
      rows$label[rr] <- ""
      rows$plabel[rr] <- ""
    }
  }
  out <- reorder_delta_free(rbind(pt, rows))
  attr(out, "magmaan.group_var") <- attr(pt, "magmaan.group_var", exact = TRUE)
  attr(out, "magmaan.group_labels") <- attr(pt, "magmaan.group_labels", exact = TRUE)
  attr(out, "magmaan.ordered") <- ordinal_stats$ordered
  attr(out, "magmaan.parameterization") <- "delta"
  out
}

augment_mixed_ordinal_partable <- function(model, mixed_stats) {
  pt <- partable_arg(model)
  parameterization <- attr(pt, "magmaan.parameterization", exact = TRUE) %||% "delta"
  if (identical(parameterization, "theta")) {
    stop("augment_mixed_ordinal_partable(): theta parameterization is not supported yet; ",
         "use parameterization = 'delta'")
  }
  if (!any(pt$op == "~1")) {
    stop("augment_mixed_ordinal_partable(): mixed categorical fitting requires model_spec(..., meanstructure = TRUE)")
  }
  ov_by_group <- mixed_stats$ov_names
  if (!is.list(ov_by_group)) ov_by_group <- list(ov_by_group)
  ordered <- mixed_stats$ordered
  fix_delta_variances <- function(pt) {
    for (b in seq_along(ov_by_group)) {
      ord <- intersect(ordered, ov_by_group[[b]])
      idx <- pt$op == "~~" &
        pt$lhs == pt$rhs &
        pt$lhs %in% ord &
        pt$group == b
      pt$free[idx] <- 0L
      pt$ustart[idx] <- 1.0
      idx_i <- pt$op == "~1" &
        pt$lhs %in% ord &
        pt$group == b
      pt$free[idx_i] <- 0L
      pt$ustart[idx_i] <- 0.0
    }
    free_old <- pt$free
    vals <- sort(unique(free_old[free_old > 0L]))
    if (length(vals)) {
      map <- setNames(seq_along(vals), vals)
      pt$free[free_old > 0L] <- unname(map[as.character(free_old[free_old > 0L])])
    }
    pt
  }
  reorder_delta_free <- function(pt) {
    append_unique <- function(ids, add) unique(c(ids, add[!is.na(add) & add > 0L]))
    ids <- integer()
    groups <- sort(unique(pt$group[pt$group > 0L]))
    for (g in groups) {
      in_group <- pt$group == g
      ids <- append_unique(ids, pt$free[in_group & pt$op == "=~"])
      ids <- append_unique(ids, pt$free[in_group & pt$op == "|"])
      ids <- append_unique(ids, pt$free[in_group & !(pt$op %in% c("=~", "|"))])
    }
    if (length(ids)) {
      map <- setNames(seq_along(ids), ids)
      old <- pt$free
      pt$free[old > 0L] <- unname(map[as.character(old[old > 0L])])
    }
    pt
  }
  pt <- fix_delta_variances(pt)
  if (any(pt$op == "|")) return(reorder_delta_free(pt))
  required <- names(pt)
  n_new <- sum(vapply(seq_along(ov_by_group), function(b) {
    sum(as.integer(mixed_stats$n_levels[[b]][mixed_stats$ordered_mask[[b]] != 0L]) - 1L) +
      sum(mixed_stats$ordered_mask[[b]] != 0L)
  }, integer(1)))
  rows <- pt[rep(NA_integer_, n_new), required, drop = FALSE]
  n0 <- nrow(pt)
  next_free <- if (length(pt$free)) max(pt$free, na.rm = TRUE) else 0L
  rr <- 0L
  for (b in seq_along(ov_by_group)) {
    ov <- ov_by_group[[b]]
    th <- mixed_stats$thresholds[[b]]
    th_pos <- 1L
    for (j in seq_along(ov)) {
      if (!mixed_stats$ordered_mask[[b]][[j]]) next
      for (lev in seq_len(mixed_stats$n_levels[[b]][[j]] - 1L)) {
        rr <- rr + 1L
        next_free <- next_free + 1L
        rows[rr, ] <- pt[1L, required, drop = FALSE]
        rows$id[rr] <- n0 + rr
        rows$lhs[rr] <- ov[[j]]
        rows$op[rr] <- "|"
        rows$rhs[rr] <- paste0("t", lev)
        rows$user[rr] <- 0L
        rows$block[rr] <- b
        rows$group[rr] <- b
        rows$free[rr] <- next_free
        rows$exo[rr] <- 0L
        rows$ustart[rr] <- th[[th_pos]]
        rows$label[rr] <- ""
        rows$plabel[rr] <- paste0(".p", n0 + rr, ".")
        th_pos <- th_pos + 1L
      }
      rr <- rr + 1L
      rows[rr, ] <- pt[1L, required, drop = FALSE]
      rows$id[rr] <- n0 + rr
      rows$lhs[rr] <- ov[[j]]
      rows$op[rr] <- "~*~"
      rows$rhs[rr] <- ov[[j]]
      rows$user[rr] <- 0L
      rows$block[rr] <- b
      rows$group[rr] <- b
      rows$free[rr] <- 0L
      rows$exo[rr] <- 0L
      rows$ustart[rr] <- 1.0
      rows$label[rr] <- ""
      rows$plabel[rr] <- ""
    }
  }
  out <- reorder_delta_free(rbind(pt, rows))
  attr(out, "magmaan.group_var") <- attr(pt, "magmaan.group_var", exact = TRUE)
  attr(out, "magmaan.group_labels") <- attr(pt, "magmaan.group_labels", exact = TRUE)
  attr(out, "magmaan.ordered") <- mixed_stats$ordered
  attr(out, "magmaan.parameterization") <- "delta"
  out
}

df_to_data <- function(x, model, group = NULL, missing = c("listwise", "error"),
                       scaling = c("n", "n-1")) {
  missing <- match.arg(missing)
  scaling <- match.arg(scaling)
  if (!is.data.frame(x)) {
    stop("df_to_data(): `x` must be a data.frame")
  }
  model <- as_magmaan_model_spec(model)
  group_var <- if (is.null(group)) model$group_var else as.character(group)[1L]
  if (is.null(group_var) || !nzchar(group_var)) group_var <- ""

  if (nzchar(group_var)) {
    if (!group_var %in% names(x)) {
      stop("df_to_data(): grouping column not found: ", group_var)
    }
    g <- x[[group_var]]
    if (anyNA(g)) stop("df_to_data(): grouping column contains missing values")
    labels <- model$group_labels
    if (is.null(labels) || !length(labels)) {
      labels <- if (is.factor(g)) levels(g) else unique(as.character(g))
    }
    labels <- as.character(labels)
    if (!all(as.character(g) %in% labels)) {
      stop("df_to_data(): data contains groups outside `group_labels`: ",
           paste(setdiff(unique(as.character(g)), labels), collapse = ", "))
    }
    if (!is.null(model$syntax)) {
      model <- do.call(
        model_spec,
        c(list(syntax = model$syntax,
               group = group_var,
               group_labels = labels),
          model$options)
      )
    }
  } else {
    labels <- character()
  }

  rep <- model_matrix_rep(model$partable)
  ov_by_group <- rep$ov_names
  if (!is.list(ov_by_group)) ov_by_group <- list(ov_by_group)

  make_block <- function(rows, ov, label = NULL) {
    miss_cols <- setdiff(ov, names(x))
    if (length(miss_cols)) {
      stop("df_to_data(): data is missing observed variables: ",
           paste(miss_cols, collapse = ", "))
    }
    block <- x[rows, ov, drop = FALSE]
    bad_type <- names(block)[!vapply(block, is.numeric, logical(1))]
    if (length(bad_type)) {
      stop("df_to_data(): observed variables must be numeric: ",
           paste(bad_type, collapse = ", "))
    }
    cc <- stats::complete.cases(block)
    if (!all(cc)) {
      if (missing == "error") {
        suffix <- if (is.null(label)) "" else paste0(" in group '", label, "'")
        stop("df_to_data(): missing observed values", suffix)
      }
      block <- block[cc, , drop = FALSE]
    }
    if (nrow(block) < 2L) {
      suffix <- if (is.null(label)) "" else paste0(" in group '", label, "'")
      stop("df_to_data(): fewer than 2 complete rows", suffix)
    }
    as.matrix(block)
  }

  if (nzchar(group_var)) {
    if (length(ov_by_group) != length(labels)) {
      stop("df_to_data(): model has ", length(ov_by_group),
           " group block(s), but data has ", length(labels), " group label(s)")
    }
    g_chr <- as.character(x[[group_var]])
    X <- Map(function(label, ov) {
      make_block(g_chr == label, ov, label)
    }, labels, ov_by_group)
    names(X) <- labels
  } else {
    X <- list(make_block(rep(TRUE, nrow(x)), ov_by_group[[1L]]))
  }

  ss <- data_sample_stats_from_raw(X)
  if (scaling == "n-1") {
    for (i in seq_along(ss$S)) {
      ss$S[[i]] <- ss$S[[i]] * ss$nobs[[i]] / (ss$nobs[[i]] - 1L)
    }
  }
  for (i in seq_along(ss$S)) {
    nm <- colnames(X[[i]])
    dimnames(ss$S[[i]]) <- list(nm, nm)
    if (!is.null(ss$mean)) names(ss$mean[[i]]) <- nm
  }
  out <- c(ss, list(
    X = X,
    ov_names = ov_by_group,
    group_var = group_var,
    group_labels = labels,
    scaling = scaling
  ))
  class(out) <- c("magmaan_data", "list")
  out
}

df_to_fiml_data <- function(x, model, group = NULL) {
  if (!is.data.frame(x)) {
    stop("df_to_fiml_data(): `x` must be a data.frame")
  }
  model <- as_magmaan_model_spec(model)
  group_var <- if (is.null(group)) model$group_var else as.character(group)[1L]
  if (is.null(group_var) || !nzchar(group_var)) group_var <- ""

  if (nzchar(group_var)) {
    if (!group_var %in% names(x)) {
      stop("df_to_fiml_data(): grouping column not found: ", group_var)
    }
    g <- x[[group_var]]
    if (anyNA(g)) stop("df_to_fiml_data(): grouping column contains missing values")
    labels <- model$group_labels
    if (is.null(labels) || !length(labels)) {
      labels <- if (is.factor(g)) levels(g) else unique(as.character(g))
    }
    labels <- as.character(labels)
    if (!all(as.character(g) %in% labels)) {
      stop("df_to_fiml_data(): data contains groups outside `group_labels`: ",
           paste(setdiff(unique(as.character(g)), labels), collapse = ", "))
    }
    if (!is.null(model$syntax)) {
      model <- do.call(
        model_spec,
        c(list(syntax = model$syntax,
               group = group_var,
               group_labels = labels),
          model$options)
      )
    }
  } else {
    labels <- character()
  }

  rep <- model_matrix_rep(model$partable)
  ov_by_group <- rep$ov_names
  if (!is.list(ov_by_group)) ov_by_group <- list(ov_by_group)

  make_block <- function(rows, ov, label = NULL) {
    miss_cols <- setdiff(ov, names(x))
    if (length(miss_cols)) {
      stop("df_to_fiml_data(): data is missing observed variables: ",
           paste(miss_cols, collapse = ", "))
    }
    block <- x[rows, ov, drop = FALSE]
    bad_type <- names(block)[!vapply(block, is.numeric, logical(1))]
    if (length(bad_type)) {
      stop("df_to_fiml_data(): observed variables must be numeric: ",
           paste(bad_type, collapse = ", "))
    }
    if (!nrow(block)) {
      suffix <- if (is.null(label)) "" else paste0(" in group '", label, "'")
      stop("df_to_fiml_data(): no rows", suffix)
    }
    mat <- as.matrix(block)
    bad_finite <- !is.na(mat) & !is.finite(mat)
    if (any(bad_finite)) {
      suffix <- if (is.null(label)) "" else paste0(" in group '", label, "'")
      stop("df_to_fiml_data(): non-finite observed value", suffix)
    }
    mask <- !is.na(mat)
    dimnames(mask) <- dimnames(mat)
    list(X = mat, mask = mask)
  }

  if (nzchar(group_var)) {
    if (length(ov_by_group) != length(labels)) {
      stop("df_to_fiml_data(): model has ", length(ov_by_group),
           " group block(s), but data has ", length(labels), " group label(s)")
    }
    g_chr <- as.character(x[[group_var]])
    blocks <- Map(function(label, ov) {
      make_block(g_chr == label, ov, label)
    }, labels, ov_by_group)
    names(blocks) <- labels
  } else {
    blocks <- list(make_block(rep(TRUE, nrow(x)), ov_by_group[[1L]]))
  }

  X <- lapply(blocks, `[[`, "X")
  mask <- lapply(blocks, `[[`, "mask")
  if (length(labels)) {
    names(X) <- labels
    names(mask) <- labels
  }
  out <- list(
    X = X,
    mask = mask,
    ov_names = ov_by_group,
    group_var = group_var,
    group_labels = labels,
    nobs = vapply(X, nrow, integer(1))
  )
  class(out) <- c("magmaan_fiml_data", "list")
  out
}

fit_ml <- function(model, data, lbfgs = NULL) {
  fit_ml_impl(partable_arg(model), sample_stats_arg(data), lbfgs = lbfgs)
}

fit_fiml <- function(model, data, lbfgs = NULL) {
  if (is.data.frame(data)) data <- df_to_fiml_data(data, model)
  fit_fiml_impl(partable_arg(model), fiml_data_arg(data), lbfgs = lbfgs)
}

fit_uls <- function(model, data, lbfgsb = NULL, bounds = NULL) {
  fit_uls_impl(partable_arg(model), sample_stats_arg(data),
               lbfgsb = lbfgsb, bounds = bounds)
}

fit_gls <- function(model, data, lbfgsb = NULL, bounds = NULL) {
  fit_gls_impl(partable_arg(model), sample_stats_arg(data),
               lbfgsb = lbfgsb, bounds = bounds)
}

fit_wls <- function(model, data, W, lbfgsb = NULL, bounds = NULL) {
  fit_wls_impl(partable_arg(model), sample_stats_arg(data), W = W,
               lbfgsb = lbfgsb, bounds = bounds)
}

fit_dwls_ordinal <- function(model, data, lbfgsb = NULL, bounds = NULL) {
  pt <- augment_ordinal_partable(model, data)
  fit_dwls_ordinal_impl(pt, data, lbfgsb = lbfgsb, bounds = bounds)
}

fit_wls_ordinal <- function(model, data, lbfgsb = NULL, bounds = NULL) {
  pt <- augment_ordinal_partable(model, data)
  fit_wls_ordinal_impl(pt, data, lbfgsb = lbfgsb, bounds = bounds)
}

fit_dwls_mixed_ordinal <- function(model, data, lbfgsb = NULL, bounds = NULL) {
  pt <- augment_mixed_ordinal_partable(model, data)
  fit_dwls_mixed_ordinal_impl(pt, data, lbfgsb = lbfgsb, bounds = bounds)
}

fit_wls_mixed_ordinal <- function(model, data, lbfgsb = NULL, bounds = NULL) {
  pt <- augment_mixed_ordinal_partable(model, data)
  fit_wls_mixed_ordinal_impl(pt, data, lbfgsb = lbfgsb, bounds = bounds)
}

magmaan <- function(model, data, estimator = "ML", groups = NULL, ...,
                    ordered = NULL, parameterization = "delta",
                    missing = c("listwise", "error"),
                    se = "none", test = "none",
                    W = NULL, lbfgs = NULL, lbfgsb = NULL, bounds = NULL) {
  missing <- match.arg(missing)
  require_none_arg(se, "se", "standard errors")
  require_none_arg(test, "test", "test statistics")
  estimator <- toupper(as.character(estimator)[1L])
  if (!length(estimator) || is.na(estimator)) {
    stop("magmaan(): `estimator` must be a non-missing string")
  }
  if (estimator %in% c("MLM", "MLR")) {
    stop("magmaan(): `estimator` is estimate-only; robust corrections remain explicit post-fit calls")
  }
  allowed <- c("ML", "FIML", "ULS", "GLS", "WLS", "DWLS")
  if (!estimator %in% allowed) {
    stop("magmaan(): unsupported estimator '", estimator, "'")
  }

  group_var <- if (is.null(groups)) NULL else as.character(groups)[1L]
  group_labels <- NULL
  if (is.data.frame(data) && !is.null(group_var) && nzchar(group_var)) {
    if (!group_var %in% names(data)) {
      stop("magmaan(): grouping column not found: ", group_var)
    }
    g <- data[[group_var]]
    if (anyNA(g)) stop("magmaan(): grouping column contains missing values")
    group_labels <- if (is.factor(g)) levels(g) else unique(as.character(g))
    group_labels <- as.character(group_labels)
  }

  dots <- list(...)
  if (inherits(model, "magmaan_model_spec")) {
    if (length(dots)) {
      stop("magmaan(): model option arguments are only accepted when `model` is a syntax string")
    }
    spec <- model
    if (!is.null(ordered)) {
      spec$ordered <- as.character(ordered)
      attr(spec$partable, "magmaan.ordered") <- spec$ordered
    }
  } else if (is.character(model) && length(model) == 1L) {
    spec <- do.call(
      model_spec,
      c(list(syntax = model,
             group = group_var,
             group_labels = group_labels,
             ordered = ordered,
             parameterization = parameterization),
        dots)
    )
  } else {
    if (length(dots) || !is.null(ordered) || !is.null(groups)) {
      stop("magmaan(): model options require a syntax string or magmaan_model_spec")
    }
    spec <- as_magmaan_model_spec(model)
  }

  spec_group_labels <- spec$group_labels %||% character()
  needs_group_rebuild <- !is.null(group_var) && !identical(group_var, "") &&
    !is.null(spec$syntax) &&
    (!identical(spec$group_var, group_var) ||
       (!is.null(group_labels) && !identical(as.character(spec_group_labels), group_labels)))
  if (needs_group_rebuild) {
    spec <- do.call(
      model_spec,
      c(list(syntax = spec$syntax,
             group = group_var,
             group_labels = group_labels %||% spec$group_labels),
        spec$options)
    )
  }

  ordinal_requested <- length(spec$ordered) > 0L || inherits(data, "magmaan_ordinal_data") ||
    inherits(data, "magmaan_mixed_ordinal_data")

  if (identical(estimator, "FIML")) {
    if (is.data.frame(data)) data <- df_to_fiml_data(data, spec, group = group_var)
    return(fit_fiml(spec, data, lbfgs = lbfgs))
  }

  if (ordinal_requested && estimator %in% c("DWLS", "WLS")) {
    if (is.data.frame(data)) {
      ov_by_group <- model_matrix_rep(spec$partable)$ov_names
      if (!is.list(ov_by_group)) ov_by_group <- list(ov_by_group)
      all_ordinal <- all(vapply(ov_by_group, function(ov) setequal(spec$ordered, ov), logical(1)))
      data <- if (all_ordinal) {
        data_ordinal_stats_from_df(data, spec, group = group_var, missing = missing)
      } else {
        data_mixed_ordinal_stats_from_df(data, spec, group = group_var, missing = missing)
      }
    }
    if (inherits(data, "magmaan_ordinal_data")) {
      if (identical(estimator, "DWLS")) return(fit_dwls_ordinal(spec, data, lbfgsb = lbfgsb, bounds = bounds))
      return(fit_wls_ordinal(spec, data, lbfgsb = lbfgsb, bounds = bounds))
    }
    if (inherits(data, "magmaan_mixed_ordinal_data")) {
      if (identical(estimator, "DWLS")) return(fit_dwls_mixed_ordinal(spec, data, lbfgsb = lbfgsb, bounds = bounds))
      return(fit_wls_mixed_ordinal(spec, data, lbfgsb = lbfgsb, bounds = bounds))
    }
    stop("magmaan(): ordinal estimators require a data.frame, magmaan_ordinal_data, or magmaan_mixed_ordinal_data")
  }

  if (identical(estimator, "DWLS")) {
    stop("magmaan(): DWLS requires ordered variables; pass `ordered =` or a categorical data object")
  }

  if (is.data.frame(data)) data <- df_to_data(data, spec, group = group_var, missing = missing)
  switch(estimator,
         ML = fit_ml(spec, data, lbfgs = lbfgs),
         ULS = fit_uls(spec, data, lbfgsb = lbfgsb, bounds = bounds),
         GLS = fit_gls(spec, data, lbfgsb = lbfgsb, bounds = bounds),
         WLS = {
           if (is.null(W)) {
             stop("magmaan(): continuous WLS requires explicit `W`; categorical WLS requires `ordered =`")
           }
           fit_wls(spec, data, W = W, lbfgsb = lbfgsb, bounds = bounds)
         })
}

compute_defined <- function(model, fit, vcov) {
  compute_defined_impl(model_syntax_arg(model), fit, vcov)
}

fit_uls_snlls <- function(model, data, lbfgsb = NULL, bounds = NULL) {
  fit_uls_snlls_impl(partable_arg(model), sample_stats_arg(data),
                     lbfgsb = lbfgsb, bounds = bounds)
}

fit_gls_snlls <- function(model, data, lbfgsb = NULL, bounds = NULL) {
  fit_gls_snlls_impl(partable_arg(model), sample_stats_arg(data),
                     lbfgsb = lbfgsb, bounds = bounds)
}

fit_wls_snlls <- function(model, data, W, lbfgsb = NULL, bounds = NULL) {
  fit_wls_snlls_impl(partable_arg(model), sample_stats_arg(data), W = W,
                     lbfgsb = lbfgsb, bounds = bounds)
}

fit_uls_ceres <- function(model, data, ceres = NULL, bounds = NULL) {
  fit_uls_ceres_impl(partable_arg(model), sample_stats_arg(data),
                     ceres = ceres, bounds = bounds)
}

fit_gls_ceres <- function(model, data, ceres = NULL, bounds = NULL) {
  fit_gls_ceres_impl(partable_arg(model), sample_stats_arg(data),
                     ceres = ceres, bounds = bounds)
}

fit_wls_ceres <- function(model, data, W, ceres = NULL, bounds = NULL) {
  fit_wls_ceres_impl(partable_arg(model), sample_stats_arg(data), W = W,
                     ceres = ceres, bounds = bounds)
}

fit_uls_snlls_ceres <- function(model, data, ceres = NULL, bounds = NULL) {
  fit_uls_snlls_ceres_impl(partable_arg(model), sample_stats_arg(data),
                           ceres = ceres, bounds = bounds)
}

fit_gls_snlls_ceres <- function(model, data, ceres = NULL, bounds = NULL) {
  fit_gls_snlls_ceres_impl(partable_arg(model), sample_stats_arg(data),
                           ceres = ceres, bounds = bounds)
}

fit_wls_snlls_ceres <- function(model, data, W, ceres = NULL, bounds = NULL) {
  fit_wls_snlls_ceres_impl(partable_arg(model), sample_stats_arg(data), W = W,
                           ceres = ceres, bounds = bounds)
}

partable_arg <- function(model) {
  if (inherits(model, "magmaan_model_spec")) return(model$partable)
  model
}

sample_stats_arg <- function(data) {
  if (inherits(data, "magmaan_data")) {
    return(list(S = data$S, mean = data$mean, nobs = data$nobs))
  }
  data
}

fiml_data_arg <- function(data) {
  if (inherits(data, "magmaan_fiml_data")) {
    return(list(X = data$X, mask = data$mask))
  }
  data
}

model_syntax_arg <- function(model) {
  if (inherits(model, "magmaan_model_spec")) {
    if (!is.null(model$syntax)) return(model$syntax)
    stop("compute_defined(): model spec does not carry source syntax")
  }
  if (is.character(model) && length(model) == 1L) return(model)
  stop("compute_defined(): `model` must be a syntax string or magmaan_model_spec")
}

require_none_arg <- function(value, arg, what) {
  if (is.null(value)) value <- "none"
  value <- tolower(as.character(value))
  if (length(value) != 1L || is.na(value) || !identical(value, "none")) {
    stop("magmaan(): `", arg, "` is estimate-only and currently accepts only \"none\"; ",
         what, " remain explicit post-fit calls")
  }
  invisible(NULL)
}

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}
