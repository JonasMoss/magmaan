# Chart invariance on sample data: one model under four identifications.
# The sphere solution must not depend on which identification the user wrote,
# and translating a fit from one identification to another must reproduce the
# ordinary fit there.

# "f =~ a + b + c" -> "f =~ NA*a + 1*b + c" on every measurement line.
marker2_syntax <- function(syntax) {
  lines <- strsplit(syntax, "\n", fixed = TRUE)[[1L]]
  for (i in grep("=~", lines, fixed = TRUE)) {
    parts <- strsplit(lines[i], "=~", fixed = TRUE)[[1L]]
    rhs <- trimws(strsplit(parts[2L], "+", fixed = TRUE)[[1L]])
    rhs[1L] <- paste0("NA*", rhs[1L])
    rhs[2L] <- paste0("1*", rhs[2L])
    lines[i] <- paste0(trimws(parts[1L]), " =~ ", paste(rhs, collapse = " + "))
  }
  paste(lines, collapse = "\n")
}

invariance_cases <- function() {
  list(
    ernst_sem = list(data = "ernst", syntax = ernst_sem, groups = NULL, opts = list(),
                     label = "Ernst two-factor SEM"),
    hs3 = list(data = "hs", syntax = hs3, groups = NULL, opts = list(),
               label = "HS 3-factor CFA"),
    pd_free = list(data = "pd", syntax = gsub("[abc]\\*", "", pd_labels), groups = NULL,
                   opts = list(), label = "PoliticalDemocracy SEM"),
    hs_metric = list(data = "hs", syntax = hs3, groups = "school",
                     opts = list(group_equal = "loadings"),
                     label = "HS metric invariance (2 groups)"))
}

inv_spec <- function(ic, chart, d) {
  opts <- ic$opts
  opts$syntax <- if (identical(chart, "marker2")) marker2_syntax(ic$syntax) else ic$syntax
  if (identical(chart, "std_lv")) opts$std_lv <- TRUE
  if (identical(chart, "effect")) opts$effect_coding <- TRUE
  if (!is.null(ic$groups)) {
    opts$group <- ic$groups
    opts$group_labels <- unique(as.character(d[[ic$groups]]))
  }
  do.call(model_spec, opts)
}

sphere_abs <- function(fit) {
  pt <- fit$gauge$sphere_partable
  sort(abs(pt$est[pt$free > 0 & !pt$op %in% c("==", ":=")]))
}

run_invariance_case <- function(id, ic, data) {
  d <- data[[ic$data]]
  specs <- lapply(setNames(charts, charts), function(ch) inv_spec(ic, ch, d))
  dd <- lapply(specs, function(sp) df_to_data(d, sp, group = ic$groups))
  ord <- lapply(charts, function(ch) timed(fit_model(specs[[ch]], dd[[ch]])))
  sph <- lapply(charts, function(ch) timed(frontier_fit_sphere(specs[[ch]], dd[[ch]])))
  names(ord) <- names(sph) <- charts
  ref <- sph$marker$value
  fits <- do.call(rbind, lapply(charts, function(ch) {
    o <- ord[[ch]]$value
    s <- sph[[ch]]$value
    data.frame(
      case = id, label = ic$label, chart = ch,
      ord_ok = is_fit(o) && isTRUE(o$converged),
      sph_ok = is_fit(s) && isTRUE(s$converged),
      fmin_ord = if (is_fit(o)) o$fmin else NA_real_,
      fmin_sph = if (is_fit(s)) s$fmin else NA_real_,
      d_fmin_vs_marker = if (is_fit(s) && is_fit(ref)) abs(s$fmin - ref$fmin) else NA_real_,
      d_sphere_point_vs_marker = if (is_fit(s) && is_fit(ref)) {
        max_abs(sphere_abs(s), sphere_abs(ref))
      } else NA_real_,
      d_est_vs_ordinary = if (is_fit(s) && is_fit(o)) {
        max_abs(s$theta, o$theta)
      } else NA_real_,
      time_ord = ord[[ch]]$time, time_sph = sph[[ch]]$time,
      stringsAsFactors = FALSE)
  }))
  reid <- list()
  for (from in charts) for (to in setdiff(charts, from)) {
    s <- sph[[from]]$value
    o <- ord[[to]]$value
    r <- if (is_fit(s)) safe(frontier_reidentify(s, specs[[to]])) else NULL
    ok <- !is.null(r) && !inherits(r, "condition")
    reid[[length(reid) + 1L]] <- data.frame(
      case = id, from_chart = from, to_chart = to, translated = ok,
      d_vs_ordinary = if (ok && is_fit(o)) max_abs(r$theta, o$theta) else NA_real_,
      error = if (inherits(r, "condition")) conditionMessage(r) else NA_character_,
      stringsAsFactors = FALSE)
  }
  list(fits = fits, reidentify = do.call(rbind, reid))
}
