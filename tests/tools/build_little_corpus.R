#!/usr/bin/env Rscript

# Build the ignored external/little corpus from Little's RAR archives.
#
# The source material is LISREL 8 syntax plus LISREL output. We preserve every
# .LS8 model, extract embedded summary statistics when present, and emit a
# lavaan translation for simple LY measurement models whose mapping is
# unambiguous. More complex LISREL models stay catalogued as source-only.

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", ".."))

root <- Sys.getenv("LITTLE_ROOT", unset = "")
if (!nzchar(root)) root <- file.path(repo_root, "external", "little")
root <- normalizePath(root, mustWork = TRUE)

source_dir <- file.path(root, "source")
for (d in c(source_dir, file.path(root, "models"), file.path(root, "models_lisrel"),
            file.path(root, "data"), file.path(root, "scripts"),
            file.path(root, "goldens"), file.path(root, "results"))) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

archives <- list.files(root, "\\.rar$", full.names = TRUE, ignore.case = TRUE)
for (rar in archives) {
  status <- system2("unar", c("-q", "-force-overwrite", "-o", source_dir, rar))
  if (!identical(status, 0L)) stop("unar failed for ", rar, call. = FALSE)
}

sanitize_id <- function(x) {
  x <- tolower(gsub("\\.[Ll][Ss]8$", "", basename(x)))
  x <- gsub("[^a-z0-9]+", "_", x)
  gsub("^_|_$", "", x)
}

strip_comments <- function(lines) sub("!.*$", "", lines)

numbers_in <- function(x) {
  vals <- regmatches(paste(x, collapse = " "),
                     gregexpr("-?[0-9]+(?:[.][0-9]+)?(?:[Ee][+-]?[0-9]+)?",
                              paste(x, collapse = " "), perl = TRUE))[[1L]]
  as.numeric(vals)
}

param_value <- function(txt, key) {
  m <- regexec(paste0("\\b", key, "\\s*=\\s*([^\\s]+)"), txt,
               ignore.case = TRUE, perl = TRUE)
  mm <- regmatches(txt, m)[[1L]]
  if (length(mm)) mm[[2L]] else ""
}

command <- function(line) {
  x <- toupper(trimws(line))
  sub("\\s.*$", "", x)
}

section_lines <- function(lines, start_cmd) {
  starts <- which(command(lines) == start_cmd)
  if (!length(starts)) return(character())
  i <- starts[[1L]] + 1L
  known <- c("DA", "ME", "SD", "KM", "CM", "LA", "MO", "FR", "FI", "VA",
             "CO", "LE", "OU", "EQ", "PA", "SE")
  out <- character()
  while (i <= length(lines)) {
    if (command(lines[[i]]) %in% known) break
    out <- c(out, lines[[i]])
    i <- i + 1L
  }
  out
}

names_section <- function(lines, start_cmd) {
  toks <- unlist(strsplit(paste(section_lines(lines, start_cmd), collapse = " "),
                          "\\s+"))
  toks[nzchar(toks)]
}

lower_matrix <- function(vals, n) {
  if (length(vals) < n * (n + 1L) / 2L) return(NULL)
  m <- matrix(0, n, n)
  k <- 1L
  for (i in seq_len(n)) {
    for (j in seq_len(i)) {
      m[i, j] <- vals[[k]]
      m[j, i] <- vals[[k]]
      k <- k + 1L
    }
  }
  m
}

matrix_refs <- function(txt, mat) {
  pat <- paste0("\\b", mat, "\\(([0-9]+),([0-9]+)\\)")
  mm <- gregexpr(pat, txt, ignore.case = TRUE, perl = TRUE)
  hits <- regmatches(txt, mm)[[1L]]
  if (!length(hits)) return(matrix(integer(), ncol = 2L))
  do.call(rbind, lapply(hits, function(h) {
    m <- regexec(pat, h, ignore.case = TRUE, perl = TRUE)
    as.integer(regmatches(h, m)[[1L]][2:3])
  }))
}

fixed_values <- function(txt, mat) {
  pat <- paste0("\\bVA\\s+(-?[0-9]+(?:[.][0-9]+)?)\\s+", mat,
                "\\(([0-9]+),([0-9]+)\\)")
  mm <- gregexpr(pat, txt, ignore.case = TRUE, perl = TRUE)
  starts <- regmatches(txt, mm)[[1L]]
  if (!length(starts)) return(data.frame(row = integer(), col = integer(),
                                         value = numeric()))
  out <- lapply(starts, function(h) {
    m <- regexec(pat, h, ignore.case = TRUE, perl = TRUE)
    g <- regmatches(h, m)[[1L]]
    data.frame(row = as.integer(g[[3L]]), col = as.integer(g[[4L]]),
               value = as.numeric(g[[2L]]))
  })
  do.call(rbind, out)
}

write_named_vector <- function(x, path) {
  utils::write.csv(data.frame(name = names(x), value = as.numeric(x)),
                   path, row.names = FALSE, quote = TRUE)
}

write_matrix <- function(m, names, path) {
  rownames(m) <- names
  colnames(m) <- names
  utils::write.csv(m, path, quote = FALSE)
}

lavaan_from_lisrel <- function(lines, id, obs, lat) {
  txt <- paste(strip_comments(lines), collapse = "\n")
  mo <- paste(grep("^\\s*MO\\b", lines, ignore.case = TRUE, value = TRUE),
              collapse = " ")
  if (grepl("(?mi)^\\s*SE\\b", txt, perl = TRUE)) {
    return(list(model = "", status = "source_only",
                note = "LISREL selection commands require manual conversion"))
  }
  if (!grepl("\\bLY\\b", mo, ignore.case = TRUE)) {
    return(list(model = "", status = "source_only",
                note = "no LY measurement matrix found"))
  }
  if (grepl("\\b(BE|GA|PH)\\b", mo, ignore.case = TRUE) ||
      grepl("^\\s*CO\\b", txt, ignore.case = TRUE, perl = TRUE)) {
    return(list(model = "", status = "source_only",
                note = "complex LISREL matrices or constraints require manual conversion"))
  }
  fr_ly <- matrix_refs(paste(grep("^\\s*FR\\b", lines, ignore.case = TRUE,
                                  value = TRUE), collapse = "\n"), "LY")
  va_ly <- fixed_values(txt, "LY")
  if (!nrow(fr_ly) && !nrow(va_ly)) {
    return(list(model = "", status = "source_only",
                note = "no free or fixed LY entries found"))
  }
  rows <- character()
  for (j in seq_along(lat)) {
    fixed <- va_ly[va_ly$col == j, , drop = FALSE]
    free <- if (nrow(fr_ly)) {
      fr_ly[fr_ly[, 2L] == j, , drop = FALSE]
    } else {
      matrix(integer(), ncol = 2L)
    }
    idx <- sort(unique(c(fixed$row, if (length(free)) free[, 1L] else integer())))
    idx <- idx[idx >= 1L & idx <= length(obs)]
    if (!length(idx)) next
    terms <- character()
    for (r in idx) {
      fv <- fixed[fixed$row == r, , drop = FALSE]
      if (nrow(fv)) {
        terms <- c(terms, paste0(format(fv$value[[1L]], scientific = FALSE),
                                 "*", obs[[r]]))
      } else {
        terms <- c(terms, obs[[r]])
      }
    }
    rows <- c(rows, paste(lat[[j]], "=~", paste(terms, collapse = " + ")))
  }
  va_te <- fixed_values(txt, "TE")
  if (nrow(va_te)) {
    diag_te <- va_te[va_te$row == va_te$col & va_te$row <= length(obs), ,
                     drop = FALSE]
    rows <- c(rows, sprintf("%s ~~ %s*%s", obs[diag_te$row],
                            format(diag_te$value, scientific = FALSE),
                            obs[diag_te$row]))
  }
  va_ps <- fixed_values(txt, "PS")
  if (nrow(va_ps)) {
    diag_ps <- va_ps[va_ps$row == va_ps$col & va_ps$row <= length(lat), ,
                     drop = FALSE]
    rows <- c(rows, sprintf("%s ~~ %s*%s", lat[diag_ps$row],
                            format(diag_ps$value, scientific = FALSE),
                            lat[diag_ps$row]))
  }
  fr_te <- matrix_refs(paste(grep("^\\s*FR\\b", lines, ignore.case = TRUE,
                                  value = TRUE), collapse = "\n"), "TE")
  if (nrow(fr_te)) {
    off <- fr_te[fr_te[, 1L] != fr_te[, 2L] &
                   fr_te[, 1L] <= length(obs) & fr_te[, 2L] <= length(obs),
                 , drop = FALSE]
    if (nrow(off)) {
      rows <- c(rows, sprintf("%s ~~ %s", obs[off[, 1L]], obs[off[, 2L]]))
    }
  }
  if (!length(rows)) {
    return(list(model = "", status = "source_only",
                note = "no lavaan rows generated"))
  }
  list(model = paste(rows, collapse = "\n"), status = "retained",
       note = "auto-converted simple LISREL LY measurement model")
}

ls8_paths <- list.files(source_dir, "\\.[Ll][Ss]8$", recursive = TRUE,
                        full.names = TRUE)
rows <- list()
for (path in ls8_paths) {
  message("little: ", basename(path))
  raw <- readLines(path, warn = FALSE)
  lines <- strip_comments(raw)
  da <- paste(grep("^\\s*DA\\b", lines, ignore.case = TRUE, value = TRUE),
              collapse = " ")
  ni <- suppressWarnings(as.integer(param_value(da, "NI")))
  no <- suppressWarnings(as.integer(param_value(da, "NO")))
  if (is.na(ni)) ni <- length(names_section(lines, "LA"))
  id <- sanitize_id(path)
  obs <- names_section(lines, "LA")
  if (!length(obs) && !is.na(ni)) obs <- paste0("y", seq_len(ni))
  lat <- names_section(lines, "LE")
  if (!length(lat)) {
    ne <- suppressWarnings(as.integer(param_value(paste(lines, collapse = " "), "NE")))
    if (!is.na(ne) && ne > 0L) lat <- paste0("eta", seq_len(ne))
  }
  cov <- lower_matrix(numbers_in(section_lines(lines, "KM")), length(obs))
  mean <- numbers_in(section_lines(lines, "ME"))
  if (length(mean) < length(obs)) mean <- rep(NA_real_, length(obs))
  names(mean) <- obs
  lisrel_rel <- file.path("models_lisrel", paste0(id, ".LS8"))
  writeLines(raw, file.path(root, lisrel_rel), useBytes = TRUE)
  data_rel <- ""
  if (!is.null(cov)) {
    data_rel <- file.path("data", paste0(id, "_cov.csv"))
    write_matrix(cov, obs, file.path(root, data_rel))
    if (!all(is.na(mean))) {
      write_named_vector(mean, file.path(root, "data", paste0(id, "_mean.csv")))
    }
  }
  conv <- lavaan_from_lisrel(raw, id, obs, lat)
  model_rel <- ""
  if (nzchar(conv$model)) {
    model_rel <- file.path("models", paste0(id, ".lav"))
    writeLines(conv$model, file.path(root, model_rel), useBytes = TRUE)
  }
  rows[[length(rows) + 1L]] <- data.frame(
    id = id,
    name = gsub("_", " ", id),
    family = if (length(lat)) "latent measurement" else "unknown",
    provenance = "Little LISREL examples",
    source_input = sub(paste0("^", root, .Platform$file.sep), "", path),
    source_data = "",
    data_kind = if (nzchar(data_rel)) "summary" else "source",
    measurement_kind = "continuous",
    observed_only = FALSE,
    generated_data = data_rel,
    generated_model = model_rel,
    generated_lisrel_model = lisrel_rel,
    generated_script = "",
    group_var = "",
    ordered = "",
    lavaan_function = "sem",
    estimator = "ML",
    meanstructure = !all(is.na(mean)),
    fixed_x = FALSE,
    nobs = ifelse(is.na(no), NA_integer_, no),
    strict_parity = identical(conv$status, "retained") && nzchar(data_rel) &&
      length(obs) >= 2L,
    status = conv$status,
    note = conv$note,
    stringsAsFactors = FALSE)
}

catalogue <- if (length(rows)) do.call(rbind, rows) else data.frame()
utils::write.csv(catalogue, file.path(root, "catalogue.csv"), row.names = FALSE,
                 na = "")
message("Wrote ", nrow(catalogue), " Little catalogue rows to ",
        file.path(root, "catalogue.csv"))
