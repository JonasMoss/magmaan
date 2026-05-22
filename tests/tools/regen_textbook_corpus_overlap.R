#!/usr/bin/env Rscript

# Regenerate the advisory overlap graph for magmaan_textbook_corpus_v1.
#
# The graph preserves source cases as separate records. It only adds derived
# fingerprints, clusters, and typed edges that help identify duplicates,
# variants, and same-shape examples across textbook corpora.

suppressPackageStartupMessages({
  library(jsonlite)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", ".."))
fixtures <- file.path(repo_root, "tests", "fixtures")
out_dir <- file.path(fixtures, "textbook_corpus")
manifest_path <- file.path(out_dir, "manifest.json")
overrides_path <- file.path(out_dir, "overlap_overrides.json")

`%||%` <- function(x, y) if (is.null(x)) y else x

read_fixture <- function(...) {
  jsonlite::fromJSON(file.path(fixtures, ...), simplifyVector = FALSE)
}

read_json_file <- function(path) {
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

read_text <- function(path) {
  paste(readLines(path, warn = FALSE), collapse = "\n")
}

is_blank <- function(x) {
  is.null(x) || length(x) == 0L || (length(x) == 1L && (
    is.na(x) || !nzchar(trimws(as.character(x)))))
}

field <- function(x, name, default = NULL) {
  value <- x[[name]]
  if (is.null(value)) default else value
}

sort_named <- function(x) {
  if (is.list(x)) {
    nms <- names(x)
    if (!is.null(nms)) {
      x <- x[order(nms)]
    }
    return(lapply(x, sort_named))
  }
  x
}

canonical_json <- function(x) {
  jsonlite::toJSON(sort_named(x), auto_unbox = TRUE, null = "null",
                   na = "null", digits = 16)
}

hash_text <- function(text) {
  tmp <- tempfile("magmaan-textbook-overlap-")
  on.exit(unlink(tmp), add = TRUE)
  writeChar(text, tmp, eos = NULL, useBytes = TRUE)
  paste0("md5:", unname(tools::md5sum(tmp)))
}

hash_object <- function(x) {
  hash_text(canonical_json(x))
}

merge_missing <- function(old, new) {
  if (is.null(old)) return(new)
  if (is.null(new)) return(old)
  if (is.list(old) && is.list(new)) {
    out <- old
    for (nm in names(new)) {
      if (is.null(out[[nm]]) || is_blank(out[[nm]])) {
        out[[nm]] <- new[[nm]]
      } else if (is.list(out[[nm]]) && is.list(new[[nm]])) {
        out[[nm]] <- merge_missing(out[[nm]], new[[nm]])
      }
    }
    return(out)
  }
  old
}

case_key <- function(source, id) paste(source, id, sep = "::")

detail_map <- new.env(parent = emptyenv())

add_detail <- function(source, id, payload) {
  if (is_blank(id)) return(invisible(NULL))
  key <- case_key(source, id)
  old <- if (exists(key, envir = detail_map, inherits = FALSE)) {
    get(key, envir = detail_map, inherits = FALSE)
  } else {
    NULL
  }
  assign(key, merge_missing(old, payload), envir = detail_map)
}

add_cases <- function(source, fixture, container = "cases", id_field = "id") {
  rows <- fixture[[container]]
  if (is.null(rows) || length(rows) == 0L) return(invisible(NULL))
  for (row in rows) add_detail(source, row[[id_field]], row)
}

load_source_details <- function() {
  add_cases("geiser", read_fixture("geiser", "gls_reference.json"))
  add_cases("geiser", read_fixture("geiser", "uls_reference.json"))

  mplus_manifest <- read_fixture("mplus_sem", "manifest.json")
  add_cases("mplus_sem", mplus_manifest, id_field = "case_id")
  add_cases("mplus_sem", read_fixture("mplus_sem", "continuous_reference.json"))
  add_cases("mplus_sem", read_fixture("mplus_sem", "ordinal_reference.json"),
            container = "retained_not_tested", id_field = "case_id")
  add_cases("mplus_sem", read_fixture("mplus_sem", "mixed_reference.json"),
            container = "retained_not_tested", id_field = "case_id")

  for (source in c("little", "newsom")) {
    add_cases(source, read_fixture(source, "manifest.json"))
    for (name in c("continuous_reference.json", "ordinal_reference.json",
                   "mixed_reference.json", "observed_reference.json")) {
      fixture <- read_fixture(source, name)
      add_cases(source, fixture)
      add_cases(source, fixture, container = "retained_not_tested")
      add_cases(source, fixture, container = "skipped")
    }
  }
}

normalize_syntax <- function(model) {
  if (is_blank(model)) return(NULL)
  text <- gsub("\r\n?", "\n", as.character(model))
  lines <- unlist(strsplit(text, "\n", fixed = TRUE), use.names = FALSE)
  lines <- gsub("#.*$", "", lines)
  lines <- trimws(gsub("[[:space:]]+", " ", lines))
  lines <- lines[nzchar(lines)]
  if (!length(lines)) return(NULL)
  paste(sort(lines), collapse = "\n")
}

regex_escape <- function(x) {
  gsub("([][{}()+*^$|\\\\?.])", "\\\\\\1", x, perl = TRUE)
}

anonymize_tokens <- function(text) {
  if (is_blank(text)) return(NULL)
  pattern <- "(?<![A-Za-z0-9_.])[.A-Za-z][.A-Za-z0-9_]*(?![A-Za-z0-9_.])"
  hits <- regmatches(text, gregexpr(pattern, text, perl = TRUE))[[1]]
  hits <- hits[!hits %in% c("NA", "TRUE", "FALSE")]
  if (!length(hits)) return(text)
  seen <- unique(hits)
  repl <- setNames(sprintf("v%03d", seq_along(seen)), seen)
  for (token in seen[order(nchar(seen), decreasing = TRUE)]) {
    pat <- paste0("(?<![A-Za-z0-9_.])", regex_escape(token),
                  "(?![A-Za-z0-9_.])")
    text <- gsub(pat, repl[[token]], text, perl = TRUE)
  }
  text
}

first_fit <- function(fits) {
  if (is.null(fits) || !length(fits)) return(NULL)
  preferred <- c("ML", "ULS", "GLS", "WLS", names(fits))
  for (nm in unique(preferred)) {
    if (!is.null(fits[[nm]])) return(fits[[nm]])
  }
  NULL
}

sample_stats_payload <- function(c) {
  sample_cov <- field(c, "sample_cov")
  if (!is.null(sample_cov)) {
    return(list(n_obs = field(c, "n_obs") %||% NULL,
                sample_cov = sample_cov,
                sample_mean = field(c, "sample_mean") %||% NULL))
  }
  fit <- first_fit(field(c, "fits"))
  if (!is.null(fit) && !is.null(field(fit, "sample_cov"))) {
    return(list(n_obs = field(fit, "n_obs") %||% field(c, "n_obs") %||% NULL,
                sample_cov = field(fit, "sample_cov"),
                sample_mean = field(fit, "sample_mean") %||% NULL))
  }
  NULL
}

row_value <- function(row, name) {
  value <- row[[name]]
  if (is.null(value) || length(value) == 0L || is.na(value)) "" else value
}

canonical_rows <- function(rows, anonymize = FALSE) {
  if (is.null(rows) || !length(rows)) return(NULL)
  out <- lapply(rows, function(row) {
    lhs <- as.character(row_value(row, "lhs"))
    rhs <- as.character(row_value(row, "rhs"))
    op <- as.character(row_value(row, "op"))
    if (identical(op, "~~") && nzchar(rhs) && rhs < lhs) {
      tmp <- lhs
      lhs <- rhs
      rhs <- tmp
    }
    list(lhs = lhs,
         op = op,
         rhs = rhs,
         group = field(row, "group") %||% 1L,
         free = if (is.null(field(row, "free"))) {
           "row"
         } else if (field(row, "free") > 0) {
           "free"
         } else {
           "fixed"
         })
  })
  if (anonymize) {
    tokens <- unique(unlist(lapply(out, function(row) c(row$lhs, row$rhs)),
                            use.names = FALSE))
    tokens <- tokens[nzchar(tokens)]
    repl <- setNames(sprintf("v%03d", seq_along(tokens)), tokens)
    out <- lapply(out, function(row) {
      if (nzchar(row$lhs)) row$lhs <- repl[[row$lhs]]
      if (nzchar(row$rhs)) row$rhs <- repl[[row$rhs]]
      row
    })
  }
  keys <- vapply(out, function(row) paste(row$group, row$op, row$lhs, row$rhs,
                                          row$free, sep = "\r"), character(1L))
  out[order(keys)]
}

oracle_rows <- function(c) {
  lavaan <- field(c, "lavaan")
  if (is.list(lavaan)) {
    if (!is.null(field(lavaan, "free_rows"))) return(field(lavaan, "free_rows"))
    theta <- field(lavaan, "theta")
    if (!is.null(theta) && is.list(theta) &&
        length(theta) > 0L && is.list(theta[[1]])) {
      return(theta)
    }
  }
  fit <- first_fit(field(c, "fits"))
  if (!is.null(field(fit, "free_rows"))) return(field(fit, "free_rows"))
  NULL
}

fingerprint_value <- function(x) {
  if (is.null(x)) NULL else hash_object(x)
}

with_prefix <- function(prefix, value) {
  if (is_blank(value)) NULL else paste(prefix, as.character(value), sep = ":")
}

case_fingerprints <- function(case, detail) {
  c <- merge_missing(case, detail)
  source <- field(c, "source")
  source_input <- field(c, "source_input", "")
  data_path <- field(c, "source_data") %||% field(c, "generated_data") %||% ""
  n_obs <- field(c, "n_obs") %||% {
    fit <- first_fit(field(c, "fits"))
    if (is.null(fit)) NULL else field(fit, "n_obs")
  }
  syntax <- normalize_syntax(field(c, "model"))
  rows_named <- canonical_rows(oracle_rows(c), anonymize = FALSE)
  rows_shape <- canonical_rows(oracle_rows(c), anonymize = TRUE)

  data_shape <- list(
    data_kind = field(c, "data_kind") %||% "",
    measurement_kind = field(c, "measurement_kind") %||% field(c, "data_kind") %||% "",
    n_obs = n_obs %||% NULL,
    ordered = field(c, "ordered") %||% "",
    group_var = field(c, "group_var") %||% "",
    meanstructure = field(c, "meanstructure") %||% NULL,
    fixed_x = field(c, "fixed_x") %||% NULL,
    observed_only = field(c, "observed_only") %||% FALSE
  )
  family_shape <- c(data_shape, list(
    family = field(c, "family") %||% field(c, "model_kind") %||% "",
    lavaan_function = field(c, "lavaan_function") %||% ""
  ))

  list(
    case_id = field(case, "id"),
    source = source,
    dimensions = list(
      family = field(c, "family") %||% field(c, "model_kind") %||% "",
      data_kind = field(c, "data_kind") %||% "",
      measurement_kind = field(c, "measurement_kind") %||% field(c, "data_kind") %||% "",
      observed_only = field(c, "observed_only") %||% FALSE,
      strict_parity = field(c, "strict_parity") %||% FALSE,
      n_obs = n_obs %||% NULL
    ),
    values = list(
      source_artifact = fingerprint_value(with_prefix(source, source_input)),
      data_artifact = fingerprint_value(with_prefix(source, data_path)),
      data_shape = fingerprint_value(data_shape),
      sample_stats = fingerprint_value(sample_stats_payload(c)),
      syntax_named = fingerprint_value(syntax),
      syntax_shape = fingerprint_value(anonymize_tokens(syntax)),
      oracle_named = fingerprint_value(rows_named),
      oracle_shape = fingerprint_value(rows_shape),
      family_shape_hint = fingerprint_value(family_shape)
    )
  )
}

make_clusters <- function(fingerprints) {
  kinds <- c("source_artifact", "data_artifact", "data_shape", "sample_stats",
             "syntax_named", "syntax_shape", "oracle_named", "oracle_shape",
             "family_shape_hint")
  clusters <- list()
  idx <- 1L
  for (kind in kinds) {
    buckets <- list()
    for (fp in fingerprints) {
      value <- fp$values[[kind]]
      if (is.null(value)) next
      buckets[[value]] <- c(buckets[[value]], fp$case_id)
    }
    for (value in sort(names(buckets))) {
      case_ids <- sort(unique(buckets[[value]]))
      if (length(case_ids) < 2L) next
      clusters[[idx]] <- list(
        id = sprintf("cluster_%04d", idx),
        kind = kind,
        fingerprint = value,
        case_ids = case_ids
      )
      idx <- idx + 1L
    }
  }
  clusters
}

edge_policy <- function(kind) {
  switch(kind,
         source_artifact = list(kind = "same_source_artifact",
                                confidence = "exact", max_size = Inf),
         data_artifact = list(kind = "same_data_artifact",
                              confidence = "exact", max_size = Inf),
         sample_stats = list(kind = "same_sample_stats",
                             confidence = "exact", max_size = Inf),
         syntax_named = list(kind = "same_named_syntax",
                             confidence = "canonical", max_size = Inf),
         syntax_shape = list(kind = "same_model_shape",
                             confidence = "shape", max_size = Inf),
         oracle_named = list(kind = "same_oracle_structure",
                             confidence = "canonical", max_size = Inf),
         oracle_shape = list(kind = "same_oracle_structure",
                             confidence = "shape", max_size = Inf),
         family_shape_hint = list(kind = "same_family_shape_hint",
                                  confidence = "heuristic", max_size = 12L),
         NULL)
}

pair_key <- function(ids) paste(sort(ids), collapse = "\r")

cluster_edges <- function(clusters) {
  edges <- list()
  idx <- 1L
  for (cluster in clusters) {
    policy <- edge_policy(cluster$kind)
    if (is.null(policy)) next
    ids <- cluster$case_ids
    if (length(ids) > policy$max_size) next
    pairs <- utils::combn(ids, 2L, simplify = FALSE)
    for (pair in pairs) {
      edges[[idx]] <- list(
        id = sprintf("edge_%05d", idx),
        kind = policy$kind,
        confidence = policy$confidence,
        case_ids = pair,
        evidence = list(
          cluster_id = cluster$id,
          fingerprint_kind = cluster$kind,
          fingerprint = cluster$fingerprint
        )
      )
      idx <- idx + 1L
    }
  }
  edges
}

default_overrides <- function() {
  list(
    `_meta` = list(
      format_version = 1L,
      fixture_kind = "textbook_corpus.overlap_overrides",
      corpus_id = "magmaan_textbook_corpus_v1",
      note = "Manual curation hook; empty by default."
    ),
    known_equivalent = list(),
    known_distinct = list(),
    aliases = list(),
    notes = list()
  )
}

load_overrides <- function() {
  if (file.exists(overrides_path)) read_json_file(overrides_path) else default_overrides()
}

apply_overrides <- function(edges, overrides) {
  distinct <- overrides$known_distinct %||% list()
  distinct_keys <- unique(vapply(distinct, function(x) {
    ids <- x$case_ids %||% x
    if (length(ids) < 2L) return("")
    pair_key(ids[1:2])
  }, character(1L)))
  distinct_keys <- distinct_keys[nzchar(distinct_keys)]
  if (length(distinct_keys)) {
    edges <- edges[!vapply(edges, function(edge) {
      pair_key(edge$case_ids) %in% distinct_keys
    }, logical(1L))]
  }

  start <- length(edges) + 1L
  known <- overrides$known_equivalent %||% list()
  for (item in known) {
    ids <- sort(unique(item$case_ids %||% item))
    if (length(ids) < 2L) next
    for (pair in utils::combn(ids, 2L, simplify = FALSE)) {
      edges[[length(edges) + 1L]] <- list(
        id = sprintf("edge_%05d", start),
        kind = item$kind %||% "manual_equivalence",
        confidence = "manual",
        case_ids = pair,
        evidence = list(note = item$note %||% "")
      )
      start <- start + 1L
    }
  }
  edges
}

count_edges <- function(edges, field_name) {
  values <- sort(unique(vapply(edges, function(x) x[[field_name]], character(1L))))
  stats <- lapply(values, function(value) {
    sum(vapply(edges, function(x) identical(x[[field_name]], value), logical(1L)))
  })
  names(stats) <- values
  stats
}

load_source_details()
manifest <- read_json_file(manifest_path)
overrides <- load_overrides()
cases <- manifest$cases
fingerprints <- lapply(cases, function(case) {
  key <- case$id
  detail <- if (exists(key, envir = detail_map, inherits = FALSE)) {
    get(key, envir = detail_map, inherits = FALSE)
  } else {
    NULL
  }
  case_fingerprints(case, detail)
})
clusters <- make_clusters(fingerprints)
edges <- apply_overrides(cluster_edges(clusters), overrides)

payload <- list(
  `_meta` = list(
    format_version = 1L,
    fixture_kind = "textbook_corpus.overlap",
    corpus_id = "magmaan_textbook_corpus_v1",
    tool = "tests/tools/regen_textbook_corpus_overlap.R",
    generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
    source_manifest_hash = hash_text(read_text(manifest_path)),
    overrides_hash = if (file.exists(overrides_path)) {
      hash_text(read_text(overrides_path))
    } else {
      NULL
    }
  ),
  counts = list(
    cases = length(cases),
    fingerprints = length(fingerprints),
    clusters = length(clusters),
    edges = length(edges),
    edge_kinds = count_edges(edges, "kind"),
    edge_confidence = count_edges(edges, "confidence")
  ),
  fingerprints = fingerprints,
  clusters = clusters,
  edges = edges,
  curation = list(
    overrides_file = "textbook_corpus/overlap_overrides.json",
    known_equivalent = length(overrides$known_equivalent %||% list()),
    known_distinct = length(overrides$known_distinct %||% list()),
    aliases = length(overrides$aliases %||% list()),
    notes = length(overrides$notes %||% list())
  )
)

jsonlite::write_json(payload, file.path(out_dir, "overlap.json"),
                     pretty = TRUE, auto_unbox = TRUE, null = "null",
                     na = "null", digits = NA)
cat("Wrote ", file.path(out_dir, "overlap.json"), "\n", sep = "")
