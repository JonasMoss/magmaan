# Foldnes, Gronneberg and Moss (2026), Study 2: weak-invariance Type-I
# design. Constants are transcribed from the article and its OSF supplement.

study2_distributions <- c(
  "normal", "vm1", "ig1", "pl1", "vm2", "ig2", "pl2")

.study2_null_loadings <- c(
  .8, .9, 1.1, 1.4, .7, 1.4, 1.4, 1.2, 1.1, .6,
  .7, .7, 1.2, .9, 1.3, 1, 1.2, 1.5, .9, 1.3)

study2_full_design <- function() {
  out <- expand.grid(
    p = c(5L, 10L, 20L),
    groups = c(2L, 4L, 8L),
    n_group = c(400L, 800L, 2000L),
    distribution = study2_distributions,
    stringsAsFactors = FALSE
  )
  out$df <- with(out, as.integer((groups - 1L) * (p - 1L)))
  out$n_total <- out$groups * out$n_group
  out$cell_id <- seq_len(nrow(out))
  out
}

study2_design <- function(profile = c("smoke", "pilot", "full")) {
  profile <- match.arg(profile)
  out <- study2_full_design()
  if (profile == "smoke") {
    keep <- with(out,
      (p == 5L  & groups == 2L & n_group == 400L  & distribution == "normal") |
      (p == 10L & groups == 4L & n_group == 800L  & distribution == "vm1") |
      (p == 20L & groups == 2L & n_group == 2000L & distribution == "ig2") |
      (p == 20L & groups == 8L & n_group == 400L  & distribution == "pl2"))
    out <- out[keep, , drop = FALSE]
  } else if (profile == "pilot") {
    keep <- with(out,
      (p == 5L  & groups == 2L & n_group == 400L) |
      (p == 10L & groups == 4L & n_group == 800L) |
      (p == 20L & groups == 8L & n_group == 2000L))
    out <- out[keep, , drop = FALSE]
  }
  row.names(out) <- NULL
  out
}

study2_population <- function(p) {
  stopifnot(length(p) == 1L, p %in% c(5L, 10L, 20L))
  lambda <- .study2_null_loadings[seq_len(p)]
  list(
    p = as.integer(p),
    lambda = lambda,
    Sigma = tcrossprod(lambda) + diag(p)
  )
}

study2_syntax <- function(p) {
  paste0("f =~ ", paste0("x", seq_len(p), collapse = " + "))
}

study2_model_specs <- function(p, groups) {
  labels <- paste0("g", seq_len(groups))
  args <- list(
    syntax = study2_syntax(p),
    std_lv = FALSE,
    meanstructure = FALSE,
    group = "group",
    group_labels = labels
  )
  list(
    H1 = do.call(magmaan::model_spec, args),
    H0 = do.call(
      magmaan::model_spec,
      c(args, list(group_equal = "loadings"))
    )
  )
}

study2_method_table <- function() {
  robust <- data.frame(
    family = c("sb", "ss", "sf", "all", "peba2", "peba4", "peba6",
               "pebad", "pols"),
    paper_family = c("SB", "SS", "SF", "ALL", "pEBA2", "pEBA4",
                     "pEBA6", "pEBAdf", "pOLS"),
    stringsAsFactors = FALSE
  )
  rows <- list(
    data.frame(
      method_id = c("std_ml", "std_rls"),
      family = "std",
      paper_family = c("ML", "RLS"),
      gamma = "none",
      base = c("ml", "rls"),
      paper_label = c("ML", "RLS"),
      stringsAsFactors = FALSE
    )
  )
  for (k in seq_len(nrow(robust))) {
    for (gamma in c("biased", "unbiased")) {
      for (base in c("ml", "rls")) {
        suffix <- paste(if (gamma == "unbiased") "ug" else "b", base,
                        sep = "_")
        paper_label <- paste0(
          robust$paper_family[[k]],
          if (gamma == "unbiased") "UG" else "",
          if (base == "rls") "RLS" else ""
        )
        rows[[length(rows) + 1L]] <- data.frame(
          method_id = paste(robust$family[[k]], suffix, sep = "_"),
          family = robust$family[[k]],
          paper_family = robust$paper_family[[k]],
          gamma = gamma,
          base = base,
          paper_label = paper_label,
          stringsAsFactors = FALSE
        )
      }
    }
  }
  do.call(rbind, rows)
}

study2_test_plan <- function(df) {
  methods <- study2_method_table()
  input <- character(nrow(methods))
  input[methods$method_id == "std_ml"] <- "STD_ML"
  input[methods$method_id == "std_rls"] <- "STD_RLS"
  robust <- methods$family != "std"
  type <- methods$paper_family
  type[methods$family == "pebad"] <- paste0("pEBA", df)
  type[methods$family == "pols"] <- "pOLS"
  input[robust] <- paste0(
    type[robust],
    ifelse(methods$gamma[robust] == "unbiased", "_UG", ""),
    "_", toupper(methods$base[robust])
  )
  cbind(methods, input = input, stringsAsFactors = FALSE)
}

study2_validate_design <- function() {
  grid <- study2_full_design()
  methods <- study2_method_table()
  stopifnot(
    length(.study2_null_loadings) == 20L,
    nrow(grid) == 189L,
    identical(sort(unique(grid$df)),
              c(4L, 9L, 12L, 19L, 27L, 28L, 57L, 63L, 133L)),
    nrow(methods) == 38L,
    !anyDuplicated(methods$method_id)
  )
  expected <- data.frame(
    p = c(5L, 5L, 5L, 10L, 10L, 10L, 20L, 20L, 20L),
    groups = rep(c(2L, 4L, 8L), 3L),
    df = c(4L, 12L, 28L, 9L, 27L, 63L, 19L, 57L, 133L)
  )
  actual <- unique(grid[c("p", "groups", "df")])
  actual <- actual[order(actual$p, actual$groups), ]
  row.names(actual) <- NULL
  stopifnot(identical(actual, expected))
  for (p in c(5L, 10L, 20L)) {
    pop <- study2_population(p)
    stopifnot(
      length(pop$lambda) == p,
      identical(dim(pop$Sigma), c(p, p)),
      min(eigen(pop$Sigma, symmetric = TRUE, only.values = TRUE)$values) >
        .999
    )
  }
  invisible(TRUE)
}
