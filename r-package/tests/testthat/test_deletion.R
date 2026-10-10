.deletion_test_data <- function() {
  d <- lavaan::HolzingerSwineford1939
  d$x1[c(1, 160)] <- NA_real_
  d$x2[1] <- NA_real_ # One deleted row, even with two missing variables.
  d
}

test_that("staged and convenience fits retain per-group listwise provenance", {
  d <- .deletion_test_data()
  for (group in c("", "school")) {
    spec <- model_spec("f =~ x1 + x2 + x3", group = group,
      group_labels = if (nzchar(group)) unique(as.character(d$school)) else NULL)
    m <- prepare_model(spec)
    prepared <- prepare_data(m, d, missing = "listwise")
    staged <- suppressWarnings(estimate(m, prepared))
    fresh <- suppressWarnings(fit_model(spec, d))
    expect_equal(staged$deletion$per_group, fresh$deletion$per_group)
    expect_equal(staged$deletion$policy, "listwise")
    counts <- fresh$deletion$per_group
    expect_equal(sum(counts$rows_supplied), nrow(d))
    expect_equal(sum(counts$rows_deleted), 2)
    expect_equal(counts$rows_deleted, counts$missing_model_values)
    expect_equal(counts$rows_supplied, counts$rows_used + counts$rows_deleted)
    expect_equal(as.integer(counts$rows_used), as.integer(fresh$nobs))
    helper <- df_to_data(d, spec)
    expect_equal(helper$deletion$per_group, counts)
    lower <- fit_ml(spec, helper)
    expect_equal(lower$deletion$per_group, counts)
    expect_null(lower$raw_data$deletion)
    expect_error(fresh$deletion$policy <- "changed", "read-only")
    expect_error(fresh$deletion[["policy"]] <- "changed", "read-only")
    expect_error(fresh$deletion["policy"] <- "changed", "read-only")
    expect_error(fresh$deletion$per_group$rows_deleted <- 0L, "read-only")
    from_summary <- estimate(m, prepare_data(m, helper))
    expect_equal(from_summary$deletion$per_group, counts)
  }
})

test_that("categorical preparation and low-level fits retain deleted rows", {
  d <- .deletion_test_data()
  for (ordered in list(c("x1", "x2", "x3"), "x1")) {
    for (v in ordered) d[[v]] <- ordered(cut(d[[v]], 3), levels = levels(cut(d[[v]], 3)))
    spec <- model_spec("f =~ x1 + x2 + x3", ordered = ordered)
    m <- prepare_model(spec, prototype = d)
    staged <- suppressWarnings(estimate(m, prepare_data(m, d, missing = "listwise")))
    fresh <- suppressWarnings(fit_model(spec, d, estimator = "DWLS"))
    expect_equal(staged$deletion$per_group, fresh$deletion$per_group)
    expect_equal(sum(fresh$deletion$per_group$rows_deleted), 2)
    helper <- if (length(ordered) == 3L) data_ordinal_stats_from_df(d, spec) else
      data_mixed_ordinal_stats_from_df(d, spec)
    lower <- if (length(ordered) == 3L) fit_dwls_ordinal(spec, helper) else
      fit_dwls_mixed_ordinal(spec, helper)
    expect_equal(lower$deletion$per_group, fresh$deletion$per_group)
    expect_null(lower$ordinal_stats$deletion)
    expect_null(lower$mixed_ordinal_stats$deletion)
    d <- .deletion_test_data()
  }
})

test_that("raw missing-data routes record no listwise deletion", {
  d <- .deletion_test_data()
  spec <- model_spec("f =~ x1 + x2 + x3", meanstructure = TRUE)
  raw <- df_to_fiml_data(d, spec)
  m <- prepare_model(spec)
  staged <- suppressWarnings(estimate(m, prepare_data(m, d, kind = "raw")))
  for (estimator in c("FIML", "ML2S")) {
    fit <- suppressWarnings(fit_model(spec, d, estimator = estimator))
    expect_equal(fit$deletion$per_group, raw$deletion$per_group)
    expect_equal(fit$deletion$policy, "fiml")
    expect_equal(fit$deletion$per_group$rows_deleted, 0)
    expect_equal(fit$deletion$per_group$rows_used, nrow(d))
  }
  expect_equal(staged$deletion$per_group, raw$deletion$per_group)
  expect_error(estimate(m, prepare_data(m, d, kind = "raw"), estimator = "ML2S"),
               class = "magmaan_unsupported_estimator")
  expect_error(prepare_data(m, d), "missing observations")
})

test_that("group refusals and two-level deletion reasons are preserved", {
  d <- .deletion_test_data()
  spec <- model_spec("f =~ x1 + x2 + x3", group = "school",
                     group_labels = unique(as.character(d$school)))
  d$school <- as.character(d$school)
  d$school[2] <- NA_character_
  expect_error(fit_model(spec, d), "grouping column contains missing values")
  expect_error(df_to_data(d, spec), "grouping column contains missing values")
  expect_error(prepare_data(prepare_model(spec), d, missing = "listwise"), "group schema changed")
  d$school[2] <- "unknown"
  expect_error(df_to_data(d, spec), "outside `group_labels`")
  # Exercise the exact input projection used by fit_twolevel; fitting itself
  # is covered by the existing lavaan two-level gates.
  d$school[2] <- NA_character_
  d$cluster <- rep(seq_len(ceiling(nrow(d) / 5)), each = 5)[seq_len(nrow(d))]
  d$cluster[3] <- NA_integer_
  parts <- .twolevel_data_arg(d, spec, "cluster", "school", "listwise")
  counts <- parts$deletion$per_group
  expect_equal(sum(counts$rows_deleted), 4)
  expect_equal(sum(counts$missing_group_label), 1)
  expect_equal(sum(counts$missing_cluster_id), 1)
  expect_equal(sum(counts$missing_model_values), 2)
  expect_equal(sum(counts$rows_used), nrow(parts$X))
  expect_true(is.na(tail(counts$group, 1)))
})

test_that("summary-only input does not invent historical deletion counts", {
  spec <- model_spec("f =~ x1 + x2 + x3")
  ss <- df_to_data(.deletion_test_data(), spec)
  fit <- suppressWarnings(fit_model(spec, list(S = ss$S, nobs = ss$nobs)))
  expect_false(fit$deletion$known)
  expect_true(all(is.na(fit$deletion$per_group$rows_supplied)))
})

test_that("two-level fits expose the preparation record with unassigned rows", {
  e <- new.env()
  utils::data("Demo.twolevel", package = "lavaan", envir = e)
  d <- e$Demo.twolevel
  d$g <- ifelse(d$cluster %% 2L == 0L, "A", "B")
  syntax <- "level: 1\n fw =~ y1 + y2 + y3\nlevel: 2\n fb =~ y1 + y2 + y3"
  spec <- model_spec(syntax, group = "g", group_labels = c("A", "B"))
  d$g[1] <- NA_character_
  d$cluster[2] <- NA_integer_
  d$y1[3] <- NA_real_
  lower <- suppressWarnings(fit_twolevel(spec, d, cluster = "cluster"))
  fresh <- suppressWarnings(fit_model(spec, d, cluster = "cluster"))
  counts <- lower$deletion$per_group
  expect_equal(fresh$deletion$per_group, counts)
  expect_equal(sum(counts$rows_deleted), 3)
  expect_equal(sum(counts$missing_group_label), 1)
  expect_equal(sum(counts$missing_cluster_id), 1)
  expect_equal(sum(counts$missing_model_values), 1)
  expect_equal(sum(counts$rows_supplied), nrow(d))
})
