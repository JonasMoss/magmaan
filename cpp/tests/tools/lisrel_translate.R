# LISREL 8 all-Y interpreter, lavaan translator and output verifier.
#
# Sourced by build_little_corpus.R. Little (2013) distributes LISREL 8.80
# .LS8 inputs together with the .OUT files LISREL produced from them. The
# interpreter replays an input's command stream the way LISREL does:
#
#   * DA/LA/SE/KM/CM/SD/ME/RA read the data. FI= files are read as one
#     sequential stream per file, so ME, SD and KM may share a file.
#   * MO sets each matrix's form and mode. Groups after the first inherit
#     via IN (the same parameters), PS (same pattern and values) or
#     SP (same pattern). Matrices a later group does not mention are IN.
#   * FR/FI change status; VA, ST and MA set values (the fixed value of a
#     fixed element, the start value of a free one); EQ ties a list to its
#     first element (a fixed leader fixes the whole list); CO and IR add
#     equality and inequality constraints.
#
# The result is written as explicit lavaan syntax that is invariant to
# sem()'s auto.* defaults. lis_translate() runs the whole chain for one
# input; lis_verify() compares the translation with LISREL's own output: the
# Parameter Specifications (free/fixed/equal/constrained pattern), every
# printed estimate (fixed values included), the input moments, df, and the
# minimum-fit-function chi-square. Only the all-Y submodel (LY, BE, PS, TE,
# TY, AL) is supported, which covers every Little input.

`%||%` <- function(a, b) if (is.null(a) || !length(a)) b else a

lis_commands <- c("DA", "LA", "LE", "LK", "SE", "KM", "CM", "SD", "ME", "MM",
                  "AM", "RA", "MO", "FR", "FI", "VA", "ST", "EQ", "CO", "MA",
                  "OU", "PD", "IR", "PA")
lis_mats <- c("LY", "BE", "PS", "TE", "TY", "AL")

lis_fail <- function(...) stop(paste0(...), call. = FALSE)

lis_keyword <- function(line) {
  m <- regmatches(line, regexec("^\\s*([A-Za-z]+)", line))[[1L]]
  if (!length(m)) return("")
  kw <- toupper(substr(m[[2L]], 1L, 2L))
  if (kw %in% lis_commands) kw else ""
}

lis_rest <- function(line) sub("^\\s*[A-Za-z]+", "", line)

lis_num <- function(x) as.numeric(gsub("[Dd]", "E", x))

lis_numbers <- function(txt) {
  txt <- paste(txt, collapse = " ")
  txt <- gsub("\\([^)]*\\)", " ", txt)          # Fortran format specs
  toks <- unlist(strsplit(trimws(txt), "[[:space:],]+"))
  toks <- toks[nzchar(toks) & toks != "*"]
  v <- suppressWarnings(lis_num(toks))
  if (anyNA(v)) lis_fail("non-numeric data token: ",
                         paste(toks[is.na(v)][1:min(3, sum(is.na(v)))],
                               collapse = " "))
  v
}

# KEY=VALUE options from a DA/MO/OU/RA/KM line; bare tokens become flags.
lis_options <- function(txt) {
  txt <- gsub("\\s*=\\s*", "=", trimws(txt))
  toks <- unlist(strsplit(txt, "[[:space:]]+"))
  toks <- toks[nzchar(toks)]
  out <- list()
  for (t in toks) {
    if (grepl("=", t, fixed = TRUE)) {
      k <- toupper(sub("=.*$", "", t))
      out[[k]] <- sub("^[^=]*=", "", t)
    } else {
      out[[toupper(t)]] <- TRUE
    }
  }
  out
}

lis_find_file <- function(name, dir) {
  hits <- list.files(dir, full.names = TRUE)
  hit <- hits[tolower(basename(hits)) == tolower(basename(name))]
  if (!length(hit)) lis_fail("missing input file ", name)
  hit[[1L]]
}

# One cursor per file, shared by all groups of a run.
lis_stream_take <- function(ctx, name, n, rewind = FALSE) {
  path <- lis_find_file(name, ctx$dir)
  key <- tolower(basename(path))
  if (is.null(ctx$streams[[key]])) {
    ctx$streams[[key]] <- list(v = lis_numbers(readLines(path, warn = FALSE)),
                               pos = 0L)
  }
  s <- ctx$streams[[key]]
  if (s$pos + n > length(s$v)) {
    lis_fail("file ", basename(path), " holds ", length(s$v),
             " numbers; need ", s$pos + n)
  }
  out <- s$v[s$pos + seq_len(n)]
  s$pos <- if (rewind) 0L else s$pos + n
  ctx$streams[[key]] <- s
  out
}

# PRELIS system file: a 64-byte banner, int32 N and NI, NI 268-byte variable
# records (name at offset 56), 56 bytes, then N x NI doubles by row. The
# global missing-value code is the double at file offset 112.
read_psf <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))
  sz <- file.info(path)$size
  banner <- rawToChar(readBin(con, "raw", 64L)[1:23])
  if (!identical(banner, "PRELIS DATA SYSTEM FILE")) {
    lis_fail("not a PRELIS system file: ", basename(path))
  }
  dims <- readBin(con, "integer", 2L, size = 4L, endian = "little")
  n <- dims[[1L]]
  ni <- dims[[2L]]
  if (sz != 72 + ni * 268 + 56 + 8 * n * ni) {
    lis_fail("unexpected PRELIS layout in ", basename(path))
  }
  nm <- character(ni)
  miss <- NA_real_
  for (k in seq_len(ni)) {
    rec <- readBin(con, "raw", 268L)
    if (k == 1L) miss <- readBin(rec[41:48], "double", 1L, size = 8L,
                                 endian = "little")
    nm[[k]] <- rawToChar(rec[57:88][rec[57:88] != as.raw(0)])
  }
  readBin(con, "raw", 56L)
  v <- readBin(con, "double", n * ni, size = 8L, endian = "little")
  x <- matrix(v, n, ni, byrow = TRUE)
  x[x == miss | x < -1e300] <- NA
  colnames(x) <- nm
  x
}

lis_raw <- function(ctx, spec, ni) {
  path <- lis_find_file(spec, ctx$dir)
  if (grepl("\\.psf$", path, ignore.case = TRUE)) {
    x <- read_psf(path)
    if (ncol(x) != ni) lis_fail("PSF has ", ncol(x), " variables; NI=", ni)
    return(x)
  }
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nzchar(trimws(lines))]
  per <- lapply(lines, lis_numbers)
  len <- lengths(per)
  if (all(len >= ni)) {
    # each case starts on a new record; LISREL reads NI values from it
    return(do.call(rbind, lapply(per, `[`, seq_len(ni))))
  }
  v <- unlist(per)
  if (length(v) %% ni) lis_fail("raw file ", basename(path),
                                " is not a multiple of NI=", ni)
  matrix(v, ncol = ni, byrow = TRUE)
}

lis_sym_from_lower <- function(v, n) {
  if (length(v) != n * (n + 1L) / 2L) lis_fail("bad triangle length")
  m <- matrix(0, n, n)
  m[upper.tri(m, diag = TRUE)] <- v   # row-wise lower = column-wise upper
  m[lower.tri(m)] <- t(m)[lower.tri(m)]
  m
}

# Split an input into groups: group g spans from its DA line to the line
# before the next group's title. Title lines are everything between the
# previous group's OU line and the next DA line.
lis_split_groups <- function(lines) {
  kw <- vapply(lines, lis_keyword, character(1L), USE.NAMES = FALSE)
  da <- which(kw == "DA")
  if (!length(da)) lis_fail("no DA line")
  ou <- which(kw == "OU")
  groups <- vector("list", length(da))
  for (g in seq_along(da)) {
    stop_at <- if (g < length(da)) da[[g + 1L]] - 1L else length(lines)
    title_from <- if (g == 1L) 1L else {
      prev_ou <- ou[ou > da[[g - 1L]] & ou < da[[g]]]
      if (!length(prev_ou)) lis_fail("group ", g, " not preceded by OU")
      max(prev_ou) + 1L
    }
    if (g < length(da)) {
      this_ou <- ou[ou > da[[g]] & ou < da[[g + 1L]]]
      if (!length(this_ou)) lis_fail("group ", g, " has no OU line")
      stop_at <- max(this_ou)
    }
    title <- trimws(lines[title_from:(da[[g]] - 1L)])
    title <- title[nzchar(title)]
    groups[[g]] <- list(title = title, lines = lines[da[[g]]:stop_at])
  }
  groups
}

# Parse one group's statements. Label and data sections are consumed by
# count, so a label such as "Seventh" is never mistaken for a command.
lis_parse_group <- function(glines, inherit, ctx) {
  kw <- vapply(glines, lis_keyword, character(1L), USE.NAMES = FALSE)
  n <- length(glines)
  g <- list(da = list(), la = inherit$la, le = NULL, se = inherit$se,
            data = list(), ra = NULL, mo = list(), cmds = list(),
            ou = list())
  follow <- function(i) {
    j <- i + 1L
    while (j <= n && !nzchar(kw[[j]])) j <- j + 1L
    j
  }
  take_tokens <- function(i, first, need, until_slash = FALSE) {
    toks <- character()
    add <- function(s) {
      t <- unlist(strsplit(trimws(gsub(",", " ", s)), "[[:space:]]+"))
      t[nzchar(t) & t != "*"]
    }
    toks <- add(first)
    j <- i
    done <- function() {
      if (until_slash) any(grepl("/", toks, fixed = TRUE))
      else length(toks) >= need
    }
    while (!done()) {
      j <- j + 1L
      if (j > n) lis_fail("ran out of lines reading labels")
      toks <- c(toks, add(glines[[j]]))
    }
    if (until_slash) {
      s <- paste(toks, collapse = " ")
      s <- sub("/.*$", "", s)
      toks <- unlist(strsplit(trimws(s), "[[:space:]]+"))
      toks <- toks[nzchar(toks)]
    } else if (length(toks) != need) {
      lis_fail("expected ", need, " labels, found ", length(toks))
    }
    list(toks = toks, next_i = j + 1L)
  }
  ni <- function() {
    v <- suppressWarnings(as.integer(g$da$NI %||% inherit$ni))
    if (!length(v) || is.na(v)) lis_fail("NI unknown")
    v
  }
  i <- 1L
  while (i <= n) {
    line <- glines[[i]]
    k <- kw[[i]]
    rest <- lis_rest(line)
    if (!nzchar(trimws(line))) { i <- i + 1L; next }
    if (!nzchar(k) && grepl("^\\s*([A-Za-z]{2}\\s*\\([0-9 ,]+\\)\\s*)+$", line)) {
      # A bare element list without a command (left over from a commented
      # command); LISREL skips it without a message.
      g$ignored <- c(g$ignored, trimws(line))
      i <- i + 1L
      next
    }
    if (!nzchar(k)) lis_fail("unrecognised line: ", trimws(line))
    if (k == "DA") {
      g$da <- lis_options(rest)
      g$da_line <- line
      i <- i + 1L
    } else if (k == "LA") {
      opt <- lis_options(rest)
      if (!is.null(opt$FI)) lis_fail("LA FI= not supported")
      r <- take_tokens(i, rest, ni())
      g$la <- r$toks
      i <- r$next_i
    } else if (k == "LE") {
      ne <- suppressWarnings(as.integer(g$mo$NE %||% inherit$ne))
      if (!length(ne) || is.na(ne)) lis_fail("LE before NE is known")
      r <- take_tokens(i, rest, ne)
      g$le <- r$toks
      i <- r$next_i
    } else if (k == "SE") {
      r <- take_tokens(i, rest, 0L, until_slash = TRUE)
      g$se <- r$toks
      i <- r$next_i
    } else if (k %in% c("KM", "CM", "SD", "ME")) {
      opt <- lis_options(sub("^\\s*=", "FI=", rest))
      need <- if (k %in% c("KM", "CM")) {
        if (isTRUE(opt$FU)) ni()^2 else ni() * (ni() + 1L) / 2L
      } else ni()
      if (!is.null(opt$FI)) {
        g$files <- unique(c(g$files, basename(opt$FI)))
        v <- lis_stream_take(ctx, opt$FI, need, rewind = isTRUE(opt$REWIND))
        i <- i + 1L
      } else {
        j <- follow(i)
        v <- lis_numbers(glines[seq_len(j - i - 1L) + i])
        if (length(v) != need) lis_fail(k, " has ", length(v),
                                        " values; need ", need)
        i <- j
      }
      if (k %in% c("KM", "CM")) {
        v <- if (isTRUE(opt$FU)) matrix(v, ni(), ni(), byrow = TRUE)
             else lis_sym_from_lower(v, ni())
      }
      g$data[[k]] <- v
    } else if (k == "RA") {
      opt <- lis_options(sub("^\\s*=", "FI=", rest))
      if (is.null(opt$FI)) lis_fail("RA without a file")
      g$ra <- opt$FI
      g$files <- unique(c(g$files, basename(opt$FI)))
      i <- i + 1L
    } else if (k == "MO") {
      g$mo <- lis_options(rest)
      i <- i + 1L
    } else if (k == "OU") {
      g$ou <- lis_options(rest)
      i <- i + 1L
    } else if (k == "PD") {
      i <- i + 1L
    } else if (k == "PA") {
      lis_fail("PA pattern matrices are not supported")
    } else {
      # MA may carry its values inline; they follow on data lines.
      j <- if (k == "MA") follow(i) else i + 1L
      g$cmds[[length(g$cmds) + 1L]] <-
        list(kw = k, text = trimws(rest),
             inline = if (j > i + 1L) glines[(i + 1L):(j - 1L)] else character())
      i <- j
    }
  }
  g
}

lis_parse <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "latin1")
  lines <- sub("\r$", "", lines)
  lines <- sub("!.*$", "", lines)
  ctx <- new.env()
  ctx$dir <- dirname(path)
  ctx$streams <- list()
  chunks <- lis_split_groups(lines)
  groups <- vector("list", length(chunks))
  inherit <- list()
  for (gi in seq_along(chunks)) {
    grp <- lis_parse_group(chunks[[gi]]$lines, inherit, ctx)
    grp$title <- chunks[[gi]]$title
    if (gi == 1L) {
      ng <- suppressWarnings(as.integer(grp$da$NG %||% "1"))
      if (!identical(ng, length(chunks))) {
        lis_fail("NG=", ng, " but ", length(chunks), " DA blocks")
      }
    }
    inherit <- list(ni = grp$da$NI %||% inherit$ni, la = grp$la,
                    se = grp$se, ne = grp$mo$NE %||% inherit$ne,
                    ny = grp$mo$NY %||% inherit$ny)
    grp$ni <- as.integer(inherit$ni)
    grp$ma <- toupper(grp$da$MA %||% groups[[1L]]$ma %||% "CM")
    groups[[gi]] <- grp
  }
  list(path = path, groups = groups, ctx = ctx)
}

# --- data ------------------------------------------------------------------

lis_select <- function(grp) {
  la <- grp$la %||% paste0("VAR", seq_len(grp$ni))
  if (length(la) != grp$ni) lis_fail("LA has ", length(la), " labels, NI=",
                                     grp$ni)
  if (is.null(grp$se)) return(seq_len(grp$ni))
  key8 <- function(x) toupper(substr(x, 1L, 8L))
  idx <- vapply(grp$se, function(t) {
    if (grepl("^[0-9]+$", t)) as.integer(t) else {
      m <- match(key8(t), key8(la))
      if (is.na(m)) lis_fail("SE names unknown variable ", t)
      m
    }
  }, integer(1L), USE.NAMES = FALSE)
  idx
}

# Per-group analysed moments in LISREL's convention: S with divisor N-1,
# rescaled to a correlation matrix when MA=KM.
lis_group_data <- function(grp, ctx) {
  sel <- lis_select(grp)
  la <- grp$la %||% paste0("VAR", seq_len(grp$ni))
  out <- list(names = la[sel], raw = NULL)
  if (!is.null(grp$ra)) {
    x <- lis_raw(ctx, grp$ra, grp$ni)
    x <- x[, sel, drop = FALSE]
    if (anyNA(x)) lis_fail("raw data contain missing values")
    no <- suppressWarnings(as.integer(grp$da$NO))
    if (length(no) && !is.na(no) && no != nrow(x)) {
      lis_fail("NO=", no, " but the raw file has ", nrow(x), " rows")
    }
    colnames(x) <- out$names
    out$raw <- x
    out$n <- nrow(x)
    out$cov <- stats::cov(x)
    out$mean <- colMeans(x)
    out$kind <- "raw"
  } else {
    d <- grp$data
    no <- suppressWarnings(as.integer(grp$da$NO))
    if (!length(no) || is.na(no)) lis_fail("summary data without NO")
    out$n <- no
    if (!is.null(d$CM)) {
      cm <- d$CM
    } else if (!is.null(d$KM)) {
      cm <- if (!is.null(d$SD)) diag(d$SD) %*% d$KM %*% diag(d$SD) else d$KM
    } else lis_fail("no moment matrix")
    cm <- cm[sel, sel, drop = FALSE]
    out$cov <- cm
    out$mean <- if (!is.null(d$ME)) d$ME[sel] else NULL
    out$kind <- "summary"
  }
  if (grp$ma == "KM") {
    out$cov <- stats::cov2cor(out$cov)
    if (out$kind == "raw") {
      out$kind <- "summary"   # LISREL analyses the correlation matrix
      out$raw <- NULL
    }
  } else if (grp$ma != "CM") {
    lis_fail("MA=", grp$ma, " not supported")
  }
  dimnames(out$cov) <- list(out$names, out$names)
  if (!is.null(out$mean)) names(out$mean) <- out$names
  out
}

# --- model -----------------------------------------------------------------

lis_dims <- function(m, ny, ne) {
  switch(m, LY = c(ny, ne), BE = c(ne, ne), PS = c(ne, ne), TE = c(ny, ny),
         TY = c(ny, 1L), AL = c(ne, 1L))
}
lis_is_sym <- function(m) m %in% c("PS", "TE")
lis_default_form <- c(LY = "FU", BE = "FU", PS = "SY", TE = "DI", TY = "FU",
                      AL = "FU")
lis_default_mode <- c(LY = "FI", BE = "FI", PS = "FR", TE = "FR", TY = "FR",
                      AL = "FR")

lis_new_matrix <- function(m, d, form, mode) {
  free <- matrix(FALSE, d[[1L]], d[[2L]])
  val <- matrix(0, d[[1L]], d[[2L]])
  fr <- identical(mode, "FR")
  if (m %in% c("TY", "AL")) {
    if (form != "ZE") free[, 1L] <- fr
  } else {
    switch(form,
      ZE = NULL,
      ID = diag(val) <- 1,
      DI = diag(free) <- fr,
      SD = free[lower.tri(free)] <- TRUE,
      SY = free[lower.tri(free, diag = TRUE)] <- fr,
      ST = { diag(val) <- 1; free[lower.tri(free)] <- fr },
      FU = free[, ] <- fr,
      lis_fail("unsupported form ", form, " for ", m))
  }
  list(free = free, val = val,
       lead = matrix(NA_character_, d[[1L]], d[[2L]]),
       present = TRUE)
}

lis_key <- function(g, m, i, j) {
  if (lis_is_sym(m) && j > i) { t <- i; i <- j; j <- t }
  sprintf("%d:%s:%d:%d", g, m, i, j)
}
lis_unkey <- function(k) {
  p <- strsplit(k, ":", fixed = TRUE)[[1L]]
  list(g = as.integer(p[[1L]]), m = p[[2L]], i = as.integer(p[[3L]]),
       j = as.integer(p[[4L]]))
}

# Vector ranges such as AL(1)-AL(10) (FR/FI/VA/ST/EQ only; in CO a minus
# is a subtraction).
lis_expand_ranges <- function(txt) {
  pat <- "([A-Za-z]{2})\\s*\\(\\s*([0-9]+)\\s*\\)\\s*-\\s*([A-Za-z]{2})\\s*\\(\\s*([0-9]+)\\s*\\)"
  repeat {
    m <- regmatches(txt, regexec(pat, txt, perl = TRUE))[[1L]]
    if (!length(m)) break
    if (toupper(m[[2L]]) != toupper(m[[4L]])) lis_fail("bad range ", m[[1L]])
    a <- as.integer(m[[3L]])
    b <- as.integer(m[[5L]])
    txt <- sub(m[[1L]], paste0(m[[2L]], "(", a:b, ")", collapse = " "), txt,
               fixed = TRUE)
  }
  if (grepl("\\)\\s*-\\s*[A-Za-z]{2}\\s*\\(", txt)) lis_fail("matrix ranges unsupported")
  txt
}

# Matrix element references: LY(i,j), LY(g,i,j), TY(i), TY(g,i).
lis_refs <- function(txt, g) {
  pat <- "([A-Za-z]{2})\\s*\\(\\s*([0-9]+)\\s*(?:,\\s*([0-9]+)\\s*)?(?:,\\s*([0-9]+)\\s*)?\\)"
  mm <- regmatches(txt, gregexpr(pat, txt, perl = TRUE))[[1L]]
  lapply(mm, function(h) {
    p <- regmatches(h, regexec(pat, h, perl = TRUE))[[1L]]
    m <- toupper(p[[2L]])
    if (!m %in% lis_mats) lis_fail("unsupported matrix reference ", h)
    ix <- suppressWarnings(as.integer(p[3:5]))
    ix <- ix[!is.na(ix)]
    vec <- m %in% c("TY", "AL")
    if (vec) {
      if (length(ix) == 1L) list(g = g, m = m, i = ix[[1L]], j = 1L)
      else if (length(ix) == 2L) list(g = ix[[1L]], m = m, i = ix[[2L]], j = 1L)
      else lis_fail("bad vector reference ", h)
    } else {
      if (length(ix) == 1L && m %in% c("PS", "TE")) {
        list(g = g, m = m, i = ix[[1L]], j = ix[[1L]])   # diagonal element
      } else if (length(ix) == 2L) list(g = g, m = m, i = ix[[1L]], j = ix[[2L]])
      else if (length(ix) == 3L) list(g = ix[[1L]], m = m, i = ix[[2L]],
                                      j = ix[[3L]])
      else lis_fail("bad matrix reference ", h)
    }
  })
}

lis_build_model <- function(parsed) {
  groups <- parsed$groups
  ctx <- parsed$ctx
  G <- length(groups)
  mo1 <- groups[[1L]]$mo
  ny <- as.integer(mo1$NY %||% lis_fail("MO without NY"))
  ne <- as.integer(mo1$NE %||% lis_fail("MO without NE"))
  if (!is.null(mo1$NX) || !is.null(mo1$NK)) lis_fail("X-side models unsupported")
  st <- vector("list", G)          # st[[g]][[m]] element state
  meanstructure <- any(c("TY", "AL") %in% names(mo1))
  co <- list()
  ir <- list()
  skipped_ma <- character()

  spec_of <- function(val) {
    toks <- toupper(unlist(strsplit(val, "[,.]")))
    toks <- toks[nzchar(toks)]
    list(form = toks[toks %in% c("ZE", "ID", "DI", "SD", "SY", "ST", "FU",
                                 "IZ", "ZI")][1L],
         mode = toks[toks %in% c("FI", "FR")][1L],
         inh = toks[toks %in% c("IN", "PS", "SP", "SS")][1L])
  }
  loc <- function(r) {
    k <- lis_unkey(lis_key(r$g, r$m, r$i, r$j))
    mat <- st[[k$g]][[k$m]]
    if (is.null(mat) || !isTRUE(mat$present)) {
      lis_fail("reference to absent matrix ", k$m)
    }
    d <- dim(mat$free)
    if (k$i > d[[1L]] || k$j > d[[2L]]) {
      lis_fail("reference out of range: ", lis_key(r$g, r$m, r$i, r$j))
    }
    k
  }
  set_el <- function(r, ...) {
    k <- loc(r)
    upd <- list(...)
    for (f in names(upd)) st[[k$g]][[k$m]][[f]][k$i, k$j] <<- upd[[f]]
  }
  get_el <- function(r, field) {
    k <- loc(r)
    st[[k$g]][[k$m]][[field]][k$i, k$j]
  }
  read_ma <- function(m, g, cmd) {
    opt <- lis_options(sub(paste0("^(?i)", m), "", cmd$text, perl = TRUE))
    mat <- st[[g]][[m]]
    d <- dim(mat$free)
    form <- st[[g]]$forms[[m]]
    n_need <- if (m %in% c("TY", "AL")) d[[1L]]
      else if (lis_is_sym(m) && form == "DI") d[[1L]]
      else if (lis_is_sym(m)) d[[1L]] * (d[[1L]] + 1L) / 2L
      else prod(d)
    if (!is.null(opt$FI) &&
        inherits(try(lis_find_file(opt$FI, ctx$dir), silent = TRUE),
                 "try-error")) {
      # Start-value files missing from the archive: only fixed elements
      # would depend on them, and verification against the output decides.
      skipped_ma <<- c(skipped_ma, paste(m, basename(opt$FI)))
      return(invisible())
    }
    v <- if (!is.null(opt$FI)) lis_stream_take(ctx, opt$FI, n_need)
         else lis_numbers(cmd$inline)
    if (length(v) != n_need) lis_fail("MA ", m, " needs ", n_need,
                                      " values; got ", length(v))
    full <- if (m %in% c("TY", "AL")) matrix(v, ncol = 1L)
      else if (lis_is_sym(m) && form == "DI") diag(v, d[[1L]])
      else if (lis_is_sym(m)) lis_sym_from_lower(v, d[[1L]])
      else matrix(v, d[[1L]], d[[2L]], byrow = TRUE)
    # MA fills this group's own copy of every element, including elements
    # inherited by IN: an element later detached by FI keeps this value
    # (Little's Ch7 HomogC input relies on it). Elements that stay
    # invariant take the previous group's value at the end.
    mat$val[] <- full
    st[[g]][[m]] <<- mat
  }

  for (g in seq_len(G)) {
    mo <- groups[[g]]$mo
    st[[g]] <- list(forms = list())
    for (m in lis_mats) {
      d <- lis_dims(m, ny, ne)
      given <- mo[[m]]
      if (m %in% c("TY", "AL") && !meanstructure) {
        st[[g]][[m]] <- list(present = FALSE)
        next
      }
      if (g == 1L || (!is.null(given) && is.na(spec_of(given)$inh))) {
        sp <- if (is.null(given)) list(form = NA, mode = NA)
              else spec_of(given)
        if (is.null(given) && m == "BE") sp$form <- "ZE"
        if (is.null(given) && m %in% c("TY", "AL")) {
          # Means requested for one of TY/AL only: TY free, AL zero.
          sp$form <- if (m == "AL") "ZE" else "FU"
        }
        form <- if (is.na(sp$form)) lis_default_form[[m]] else sp$form
        mode <- if (is.na(sp$mode)) lis_default_mode[[m]] else sp$mode
        mat <- lis_new_matrix(m, d, form, mode)
        mat$inv <- matrix(FALSE, d[[1L]], d[[2L]])
        st[[g]][[m]] <- mat
        st[[g]]$forms[[m]] <- form
      } else {
        inh <- if (is.null(given)) "IN" else spec_of(given)$inh
        prev <- st[[g - 1L]][[m]]
        mat <- prev
        mat$lead[] <- NA_character_
        mat$inv[] <- FALSE
        st[[g]]$forms[[m]] <- st[[g - 1L]]$forms[[m]]
        if (inh == "IN") {
          mat$inv[] <- TRUE
          for (i in seq_len(d[[1L]])) for (j in seq_len(d[[2L]])) {
            if (prev$free[i, j]) mat$lead[i, j] <- lis_key(g - 1L, m, i, j)
          }
        } else {
          # "Same pattern" includes the previous group's own EQ ties, which
          # are replicated within this group.
          for (i in seq_len(d[[1L]])) for (j in seq_len(d[[2L]])) {
            ld <- prev$lead[i, j]
            if (is.na(ld) || isTRUE(prev$inv[i, j])) next
            lk <- lis_unkey(ld)
            if (lk$g == g - 1L) {
              mat$lead[i, j] <- lis_key(g, lk$m, lk$i, lk$j)
            }
          }
        }
        if (inh == "SP") {
          mat$val[] <- 0
          if (st[[g]]$forms[[m]] %in% c("ID", "ST")) diag(mat$val) <- 1
        } else if (inh == "SS") {
          lis_fail("SS inheritance not supported")
        }
        st[[g]][[m]] <- mat
      }
    }
    for (cmd in groups[[g]]$cmds) {
      k <- cmd$kw
      if (k %in% c("FR", "FI")) {
        for (r in lis_refs(lis_expand_ranges(cmd$text), g)) {
          set_el(r, free = (k == "FR"), lead = NA_character_, inv = FALSE)
          if (k == "FI") co[[lis_key(r$g, r$m, r$i, r$j)]] <- NULL
        }
      } else if (k %in% c("VA", "ST")) {
        v <- lis_num(sub("^\\s*([-+0-9.EeDd]+).*$", "\\1", cmd$text))
        if (is.na(v)) lis_fail(k, " without a value: ", cmd$text)
        rest <- lis_expand_ranges(sub("^\\s*[-+0-9.EeDd]+", "", cmd$text))
        if (grepl("^\\s*ALL\\s*$", rest, ignore.case = TRUE)) {
          for (m in lis_mats) {
            mat <- st[[g]][[m]]
            if (!isTRUE(mat$present)) next
            mat$val[mat$free & !mat$inv] <- v
            st[[g]][[m]] <- mat
          }
        } else {
          refs <- lis_refs(rest, g)
          if (!length(refs)) lis_fail(k, " without references: ", cmd$text)
          for (r in refs) {
            # A value for an element inherited by IN detaches and fixes it.
            if (get_el(r, "inv")) {
              set_el(r, free = FALSE, lead = NA_character_, inv = FALSE)
            }
            set_el(r, val = v)
          }
        }
      } else if (k == "MA") {
        m <- toupper(sub("^\\s*([A-Za-z]{2}).*$", "\\1", cmd$text))
        if (!m %in% lis_mats) lis_fail("MA for unsupported matrix ", m)
        read_ma(m, g, cmd)
      } else if (k == "EQ") {
        refs <- lis_refs(lis_expand_ranges(cmd$text), g)
        if (length(refs) < 2L) {
          # an EQ line naming fewer than two elements constrains nothing
          next
        }
        lead <- refs[[1L]]
        lead_key <- lis_key(lead$g, lead$m, lead$i, lead$j)
        for (r in refs[-1L]) {
          if (get_el(lead, "free")) {
            set_el(r, free = TRUE, lead = lead_key, inv = FALSE)
          } else {
            set_el(r, free = FALSE, lead = NA_character_, inv = FALSE,
                   val = get_el(lead, "val"))
          }
        }
      } else if (k == "CO") {
        lhs <- sub("=.*$", "", cmd$text)
        rhs <- sub("^[^=]*=", "", cmd$text)
        lr <- lis_refs(lhs, g)
        if (length(lr) != 1L) lis_fail("CO needs one left-hand element")
        pat <- "[A-Za-z]{2}\\s*\\([0-9 ,]+\\)"
        expr <- gsub("\\*\\*", "^", rhs)
        rr <- lis_refs(rhs, g)
        pieces <- regmatches(expr, gregexpr(pat, expr, perl = TRUE))[[1L]]
        if (length(pieces) != length(rr)) lis_fail("bad CO: ", cmd$text)
        for (q in seq_along(pieces)) {
          r <- rr[[q]]
          expr <- sub(pieces[[q]],
                      paste0("{", lis_key(r$g, r$m, r$i, r$j), "}"),
                      expr, fixed = TRUE)
        }
        if (grepl("[A-Za-z]", gsub("\\{[^}]*\\}", "", expr))) {
          lis_fail("unsupported CO expression: ", cmd$text)
        }
        co[[lis_key(lr[[1L]]$g, lr[[1L]]$m, lr[[1L]]$i, lr[[1L]]$j)]] <-
          trimws(gsub("\\s+", " ", expr))
      } else if (k == "IR") {
        m <- regmatches(cmd$text,
                        regexec("^(.*?)(<|>)\\s*([-+0-9.EeDd]+)\\s*$",
                                cmd$text, perl = TRUE))[[1L]]
        if (!length(m)) lis_fail("unsupported IR: ", cmd$text)
        r <- lis_refs(m[[2L]], g)
        if (length(r) != 1L) lis_fail("IR needs one element")
        ir[[length(ir) + 1L]] <- list(
          key = lis_key(r[[1L]]$g, r[[1L]]$m, r[[1L]]$i, r[[1L]]$j),
          op = m[[3L]], value = lis_num(m[[4L]]))
      } else {
        lis_fail("unsupported command ", k)
      }
    }
  }
  # Fixed elements inherited by IN take the (final) previous-group value.
  for (g in seq_len(G)[-1L]) for (m in lis_mats) {
    mat <- st[[g]][[m]]
    if (!isTRUE(mat$present)) next
    fx <- mat$inv & !mat$free
    mat$val[fx] <- st[[g - 1L]][[m]]$val[fx]
    st[[g]][[m]] <- mat
  }
  list(G = G, ny = ny, ne = ne, st = st, co = co, ir = ir,
       meanstructure = meanstructure, skipped_ma = skipped_ma)
}

# Resolve EQ/IN ties into parameter classes and record every element.
lis_elements <- function(model) {
  rows <- list()
  for (g in seq_len(model$G)) for (m in lis_mats) {
    mat <- model$st[[g]][[m]]
    if (!isTRUE(mat$present)) next
    d <- dim(mat$free)
    for (i in seq_len(d[[1L]])) for (j in seq_len(d[[2L]])) {
      if (lis_is_sym(m) && j > i) next
      rows[[length(rows) + 1L]] <- data.frame(
        key = lis_key(g, m, i, j), g = g, m = m, i = i, j = j,
        free = mat$free[i, j], val = mat$val[i, j], lead = mat$lead[i, j],
        stringsAsFactors = FALSE)
    }
  }
  el <- do.call(rbind, rows)
  rownames(el) <- el$key
  rep_of <- function(k) {
    seen <- character()
    while (!is.na(el[k, "lead"])) {
      if (k %in% seen) lis_fail("cyclic EQ chain at ", k)
      seen <- c(seen, k)
      k <- el[k, "lead"]
    }
    k
  }
  el$rep <- vapply(el$key, rep_of, character(1L))
  bad <- el$free & !el[el$rep, "free"]
  if (any(bad)) lis_fail("free element tied to a fixed leader: ",
                         el$key[bad][[1L]])
  el$rep[!el$free] <- NA_character_
  co_reps <- vapply(names(model$co), function(k) el[k, "rep"] %||% k,
                    character(1L))
  el$constrained <- el$free & el$rep %in% co_reps
  el
}

# --- lavaan syntax ---------------------------------------------------------

lis_names <- function(obs, lat) {
  clean <- function(x) {
    x <- gsub("[^A-Za-z0-9_.]", "_", x)
    ifelse(grepl("^[A-Za-z.]", x), x, paste0("v", x))
  }
  obs <- clean(obs)
  if (anyDuplicated(obs)) lis_fail("duplicate observed labels")
  # LE labels are free text (Little uses "==>" for unnamed constructs);
  # anything that is not a usable name becomes ETA<k>.
  bad <- !grepl("[A-Za-z0-9]", lat)
  lat <- clean(lat)
  bad <- bad | duplicated(lat) | duplicated(lat, fromLast = TRUE)
  lat[bad] <- paste0("ETA", which(bad))
  clash <- lat %in% obs
  lat[clash] <- paste0(lat[clash], "_lv")
  if (anyDuplicated(lat)) lis_fail("duplicate latent labels")
  list(obs = obs, lat = lat)
}

lis_fmt <- function(v) trimws(formatC(v, digits = 15, format = "g"))

lis_label <- function(key, G) {
  k <- lis_unkey(key)
  base <- if (k$m %in% c("TY", "AL")) sprintf("%s_%d", tolower(k$m), k$i)
          else sprintf("%s_%d_%d", tolower(k$m), k$i, k$j)
  if (G > 1L) paste0(base, "_g", k$g) else base
}

lis_to_lavaan <- function(model, obs, lat) {
  G <- model$G
  el <- lis_elements(model)
  nm <- lis_names(obs, lat)
  obs <- nm$obs
  lat <- nm$lat
  cls_size <- table(el$rep[el$free])
  co_keys <- names(model$co)
  co_refs <- unlist(lapply(model$co, function(e) {
    regmatches(e, gregexpr("\\{[^}]*\\}", e))[[1L]]
  }))
  co_refs <- gsub("[{}]", "", co_refs)
  needs <- unique(el[c(co_keys, co_refs), "rep"])
  needs <- c(needs, names(cls_size)[cls_size > 1L])
  el$label <- NA_character_
  lab_rep <- el$free & el$rep %in% needs
  el$label[lab_rep] <- vapply(el$rep[lab_rep], lis_label, "", G = G)

  nameof <- function(m, idx) if (m %in% c("LY", "TE", "TY")) obs[idx]
                             else lat[idx]
  pos_rows <- function(m, i, j) el[el$m == m & el$i == i & el$j == j, ,
                                   drop = FALSE][order(el$g[el$m == m &
                                   el$i == i & el$j == j]), , drop = FALSE]
  # LISREL IR on a single element against a constant is a box bound.
  bound_of <- function(r) {
    hit <- Filter(function(x) x$key %in% r$key, model$ir)
    if (!length(hit)) return("")
    if (G > 1L) lis_fail("IR in a multi-group model is not supported")
    x <- hit[[1L]]
    paste0(if (x$op == ">") "lower(" else "upper(", lis_fmt(x$value), ")*")
  }
  modifier <- function(r, loading) {
    any_fixed <- any(!r$free)
    any_lab <- any(!is.na(r$label))
    if (G == 1L) {
      if (!r$free) return(paste0(lis_fmt(r$val), "*"))
      paste0(if (loading) "NA*" else "", bound_of(r),
             if (!is.na(r$label)) paste0(r$label, "*") else "")
    } else {
      vals <- if (any_fixed || loading) {
        paste0("c(", paste(ifelse(r$free, "NA", lis_fmt(r$val)),
                           collapse = ","), ")*")
      } else ""
      labs <- if (any_lab) {
        l <- ifelse(is.na(r$label), vapply(r$key, lis_label, "", G = G),
                    r$label)
        paste0("c(", paste(l, collapse = ","), ")*")
      } else ""
      paste0(vals, labs)
    }
  }
  active <- function(r) any(r$free) || any(r$val != 0)

  lines <- character()
  lat_has_ind <- logical(length(lat))
  for (j in seq_along(lat)) {
    terms <- character()
    for (i in seq_along(obs)) {
      r <- pos_rows("LY", i, j)
      if (active(r)) terms <- c(terms, paste0(modifier(r, TRUE), obs[[i]]))
    }
    if (length(terms)) {
      lat_has_ind[[j]] <- TRUE
      lines <- c(lines, paste(lat[[j]], "=~", paste(terms, collapse = " + ")))
    }
  }
  # A latent with no loadings (a phantom) is declared with a zero loading.
  for (j in which(!lat_has_ind)) {
    z <- if (G == 1L) "0*" else paste0("c(", paste(rep("0", G), collapse = ","),
                                       ")*")
    lines <- c(lines, paste0(lat[[j]], " =~ ", z, obs[[1L]]))
  }
  for (i in seq_along(lat)) {
    terms <- character()
    for (j in seq_along(lat)) {
      r <- pos_rows("BE", i, j)
      if (active(r)) terms <- c(terms, paste0(modifier(r, FALSE), lat[[j]]))
    }
    if (length(terms)) lines <- c(lines, paste(lat[[i]], "~",
                                                paste(terms, collapse = " + ")))
  }
  for (m in c("PS", "TE")) {
    n <- if (m == "PS") length(lat) else length(obs)
    for (i in seq_len(n)) for (j in seq_len(i)) {
      r <- pos_rows(m, i, j)
      if (i == j || active(r)) {
        lines <- c(lines, paste0(nameof(m, i), " ~~ ", modifier(r, FALSE),
                                 nameof(m, j)))
      }
    }
  }
  if (model$meanstructure) {
    for (m in c("TY", "AL")) {
      n <- if (m == "TY") length(obs) else length(lat)
      for (i in seq_len(n)) {
        r <- pos_rows(m, i, 1L)
        lines <- c(lines, paste0(nameof(m, i), " ~ ", modifier(r, FALSE), "1"))
      }
    }
  }
  lab_of_key <- function(k) {
    r <- el[k, ]
    if (!r$free) lis_fmt(r$val) else el[r$rep, "label"] %||% r$label
  }
  cons <- character()
  for (k in co_keys) {
    e <- model$co[[k]]
    refs <- gsub("[{}]", "", regmatches(e, gregexpr("\\{[^}]*\\}", e))[[1L]])
    for (q in refs) e <- sub(paste0("{", q, "}"), lab_of_key(q), e,
                             fixed = TRUE)
    cons <- c(cons, paste(lab_of_key(k), "==", e))
  }
  cons <- unique(cons)
  list(lines = lines, constraints = cons, elements = el, obs = obs, lat = lat)
}

# Intended free/fixed pattern as lavaan rows, for checking what sem() builds.
lis_intended <- function(tr) {
  el <- tr$elements
  nm <- function(m, idx) if (m %in% c("LY", "TE", "TY")) tr$obs[idx]
                         else tr$lat[idx]
  lhs <- ifelse(el$m == "LY", nm("PS", el$j),
          ifelse(el$m == "BE", nm("PS", el$i),
          ifelse(el$m %in% c("PS", "AL"), nm("PS", el$i), nm("TE", el$i))))
  rhs <- ifelse(el$m == "LY", nm("TE", el$i),
          ifelse(el$m %in% c("BE", "PS"), nm("PS", el$j),
          ifelse(el$m == "TE", nm("TE", el$j), "")))
  op <- c(LY = "=~", BE = "~", PS = "~~", TE = "~~", TY = "~1",
          AL = "~1")[el$m]
  data.frame(lhs = lhs, op = unname(op), rhs = rhs, group = el$g,
             free = el$free, val = el$val, key = el$key, rep = el$rep,
             stringsAsFactors = FALSE)
}

lis_match_rows <- function(pt, want) {
  a <- paste(pt$lhs, pt$op, pt$rhs, pt$group)
  b <- paste(pt$rhs, pt$op, pt$lhs, pt$group)
  k <- paste(want$lhs, want$op, want$rhs, want$group)
  m <- match(k, a)
  sym <- is.na(m) & want$op == "~~"
  m[sym] <- match(k[sym], b)
  m
}

# Assemble syntax; add explicit zero rows for anything sem() would free
# that LISREL fixes, and confirm the final partable is exactly LISREL's.
lis_finalize <- function(tr, fit_fun) {
  extra <- character()
  G <- max(tr$elements$g)
  for (iter in 1:5) {
    model <- paste(c(tr$lines, extra, tr$constraints), collapse = "\n")
    pt <- lavaan::parTable(fit_fun(model))
    want <- lis_intended(tr)
    idx <- lis_match_rows(pt, want)
    wanted_free <- rep(FALSE, nrow(pt))
    wanted_free[idx[!is.na(idx) & want$free]] <- TRUE
    stray <- pt[!wanted_free & pt$free > 0L &
                  pt$op %in% c("=~", "~", "~~", "~1"), , drop = FALSE]
    if (!nrow(stray)) break
    for (s in split(stray, paste(stray$lhs, stray$op, stray$rhs))) {
      z <- if (G == 1L) "0*" else paste0("c(", paste(rep("0", G),
                                                     collapse = ","), ")*")
      extra <- c(extra, if (s$op[[1L]] == "~1") paste0(s$lhs[[1L]], " ~ ", z, "1")
                        else paste0(s$lhs[[1L]], " ", s$op[[1L]], " ", z,
                                    s$rhs[[1L]]))
    }
  }
  if (nrow(stray)) lis_fail("could not block auto-added parameters")
  # Every LISREL element must now be represented with the right status.
  ok <- !is.na(idx) | (!want$free & want$val == 0)
  if (!all(ok)) lis_fail("element missing from partable: ", want$key[!ok][[1L]])
  got <- pt[idx[!is.na(idx)], , drop = FALSE]
  w <- want[!is.na(idx), , drop = FALSE]
  if (any((got$free > 0L) != w$free)) {
    lis_fail("free/fixed mismatch at ", w$key[(got$free > 0L) != w$free][[1L]])
  }
  fx <- !w$free
  if (any(abs(got$ustart[fx] - w$val[fx]) > 1e-12)) {
    lis_fail("fixed value mismatch at ", w$key[fx][abs(got$ustart[fx] -
                                                      w$val[fx]) > 1e-12][[1L]])
  }
  list(model = model, partable = pt, match = idx)
}


# --- LISREL output ---------------------------------------------------------

out_block_names <- c("LAMBDA-Y" = "LY", "BETA" = "BE", "PSI" = "PS",
                     "THETA-EPS" = "TE", "TAU-Y" = "TY", "ALPHA" = "AL")

# Values sit in 11-column fields after a 9-column row label.
out_fields <- function(line, ncol) {
  vapply(seq_len(ncol), function(k) {
    trimws(substr(line, 10L + (k - 1L) * 11L, 9L + k * 11L))
  }, character(1L))
}

out_is_data <- function(line) {
  rest <- substring(line, 10L)
  rest <- gsub("- -", " NA ", rest, fixed = TRUE)
  rest <- gsub("Constr'd", " NA ", rest, fixed = TRUE)
  toks <- unlist(strsplit(trimws(rest), "\\s+"))
  length(toks) > 0L && nzchar(toks[[1L]]) &&
    all(grepl("^(NA|\\(?-?[0-9]*\\.?[0-9]+(E[-+][0-9]+)?\\)?)$", toks))
}

# Panels of the recognised matrix blocks in one output section.
out_panels <- function(sec) {
  res <- list()
  cur <- NULL
  k <- 1L
  n <- length(sec)
  while (k <= n) {
    tl <- trimws(sec[[k]])
    if (!nzchar(tl) || grepl("^Note:", tl)) { k <- k + 1L; next }
    if (tl %in% names(out_block_names)) {
      cur <- out_block_names[[tl]]
      # a wide matrix repeats its name before each continuation panel
      if (!is.list(res[[cur]])) res[[cur]] <- list()
      k <- k + 1L
      next
    }
    eqm <- regmatches(tl, regexec("^([A-Z-]+) EQUALS \\1 IN THE FOLLOWING GROUP",
                                  tl))[[1L]]
    if (length(eqm)) {
      if (eqm[[2L]] %in% names(out_block_names)) {
        res[[out_block_names[[eqm[[2L]]]]]] <- "EQUALS"
      }
      cur <- NULL
      k <- k + 1L
      next
    }
    if (k < n && grepl("^\\s*-{4,}", sec[[k + 1L]])) {
      # one dash run per column; header labels can be blank
      cols <- unlist(strsplit(trimws(sec[[k + 1L]]), "\\s+"))
      j <- k + 2L
      rows <- character()
      while (j <= n) {
        if (nzchar(trimws(sec[[j]]))) {
          if (!out_is_data(sec[[j]]) && !grepl("^\\s*\\(", sec[[j]])) break
          rows <- c(rows, sec[[j]])
          j <- j + 1L
        } else {
          # a blank line ends the panel unless data rows resume after it
          nb <- j + 1L
          while (nb <= n && !nzchar(trimws(sec[[nb]]))) nb <- nb + 1L
          if (nb <= n && nzchar(trimws(substr(sec[[nb]], 1L, 9L))) &&
              out_is_data(sec[[nb]]) &&
              !(nb < n && grepl("^\\s*-{4,}", sec[[nb + 1L]]))) {
            j <- nb
          } else break
        }
      }
      if (!is.null(cur)) {
        res[[cur]][[length(res[[cur]]) + 1L]] <- list(cols = cols, rows = rows)
      }
      k <- j
      next
    }
    cur <- NULL          # any other heading ends the current block
    k <- k + 1L
  }
  res
}

# Map panels to an element matrix of printed fields (NA = not printed).
out_matrix <- function(panels, m, d) {
  out <- matrix(NA_character_, d[[1L]], d[[2L]])
  col0 <- 0L
  for (p in panels) {
    ncol <- length(p$cols)
    if (!length(p$rows)) next
    vec <- !nzchar(trimws(substr(p$rows[[1L]], 1L, 9L)))
    idx <- col0 + seq_len(ncol)
    if (vec) {
      f <- out_fields(p$rows[[1L]], ncol)
      f[!nzchar(f)] <- NA_character_
      if (m %in% c("TY", "AL")) out[idx, 1L] <- f
      else if (lis_is_sym(m)) out[cbind(idx, idx)] <- f
      else lis_fail("vector panel for ", m)
    } else {
      rows <- p$rows[nzchar(trimws(substr(p$rows, 1L, 9L))) &
                       !grepl("^\\s*\\(", p$rows)]
      r0 <- if (lis_is_sym(m)) col0 else 0L
      for (q in seq_along(rows)) {
        f <- out_fields(rows[[q]], ncol)
        i <- r0 + q
        if (i > d[[1L]]) lis_fail("too many rows in ", m, " panel")
        for (c in seq_len(ncol)) {
          j <- idx[[c]]
          if (!lis_is_sym(m) || j <= i) {
            out[i, j] <- if (nzchar(f[[c]])) f[[c]] else NA_character_
          }
        }
      }
    }
    col0 <- col0 + ncol
  }
  out
}

# Moments printed after the input echo (3 decimals), per group.
# Moments printed after the input echo (at ND precision), group after group.
# Wide matrices print as several panels, each under its own heading; panels
# are consumed in order until a group's p columns are complete.
out_moments <- function(x, p) {
  heads <- grep("^\\s*(Covariance|Correlation) Matrix\\s*$", x)
  ends <- grep("^\\s*(Means|Parameter Specifications|Standard Deviations|Total Variance|Number of)|Matrix\\s*$",
               x)
  panels <- list()
  for (h in heads) {
    e <- ends[ends > h]
    e <- if (length(e)) e[[1L]] - 1L else length(x)
    panels <- c(panels, out_panels(c("PSI", x[(h + 1L):e]))$PS)
  }
  res <- list()
  cur <- list()
  width <- 0L
  for (pn in panels) {
    cur[[length(cur) + 1L]] <- pn
    width <- width + length(pn$cols)
    if (width >= p) {
      v <- out_matrix(cur, "PS", c(p, p))
      res[[length(res) + 1L]] <- suppressWarnings(matrix(as.numeric(v), p, p))
      cur <- list()
      width <- 0L
    }
  }
  res
}

lis_read_out <- function(path, titles, ny, ne) {
  x <- readLines(path, warn = FALSE, encoding = "latin1")
  x <- sub("\r$", "", x)
  norm <- function(v) gsub("\\s+", " ", trimws(v))
  tt <- unique(norm(unlist(titles)))
  x <- x[!(norm(x) %in% tt)]
  spec <- grep("^\\s*Parameter Specifications\\s*$", x)
  est <- grep("^\\s*LISREL Estimates", x)
  nobs <- as.integer(sub(".*Number of Observations\\s+", "",
                         grep("Number of Observations", x, value = TRUE)))
  df <- grep("^\\s*Degrees of Freedom =", x, value = TRUE)
  df <- if (length(df)) as.integer(sub(".*=\\s*", "", df[[length(df)]])) else NA
  chi <- grep("Minimum Fit Function Chi-Square =", x, value = TRUE)
  chi <- if (length(chi)) {
    as.numeric(sub("^.*Chi-Square =\\s*([-0-9.]+).*$", "\\1",
                   chi[[length(chi)]]))
  } else NA
  if (is.na(df) && any(grepl("Model is Saturated", x))) { df <- 0L; chi <- 0 }
  stops <- sort(c(spec, est, grep("Number of Iterations|Goodness of Fit|Standardized Solution|Total and Indirect|Modification Indices",
                                  x)))
  section <- function(from) {
    to <- stops[stops > from]
    to <- if (length(to)) to[[1L]] - 1L else length(x)
    x[from:to]
  }
  dims <- list(LY = c(ny, ne), BE = c(ne, ne), PS = c(ne, ne),
               TE = c(ny, ny), TY = c(ny, 1L), AL = c(ne, 1L))
  to_mats <- function(starts) {
    res <- lapply(starts, function(s) {
      pn <- out_panels(section(s))
      lapply(names(pn), function(m) {
        if (identical(pn[[m]], "EQUALS")) "EQUALS"
        else out_matrix(pn[[m]], m, dims[[m]])
      }) -> mats
      names(mats) <- names(pn)
      mats
    })
    # "X EQUALS X IN THE FOLLOWING GROUP" in group g: copy from group g+1.
    for (g in rev(seq_along(res))) for (m in names(res[[g]])) {
      if (identical(res[[g]][[m]], "EQUALS")) {
        if (g == length(res)) lis_fail("unresolvable EQUALS for ", m)
        # the next group may not print an all-fixed matrix at all
        res[[g]][m] <- list(res[[g + 1L]][[m]])
      }
    }
    res
  }
  fatal <- grep("F_A_T_A_L E_R_R_O_R", x, value = TRUE)
  list(nobs = nobs, df = df, chisq = chi, fatal = trimws(fatal),
       spec = to_mats(spec), est = to_mats(est),
       moments = out_moments(x, ny),
       warnings = unique(trimws(grep("W_A_R_N_I_N_G|Serious problems|not converge|not positive definite",
                                     x, value = TRUE))))
}

# --- verification ----------------------------------------------------------

# Compare a fitted translation with LISREL's own output. `fit` must be the
# Wishart (N-1) lavaan fit on LISREL's analysed moments.
lis_verify <- function(tr, fin, fit, out, data, nd) {
  el <- tr$elements
  want <- lis_intended(tr)
  pe <- lavaan::parTable(fit)
  G <- max(el$g)
  # printed rounding plus a small allowance for convergence differences
  tol_est <- 0.5 * 10^(-nd) + 2e-4
  issues <- character()
  note <- function(...) issues <<- c(issues, paste0(...))

  if (length(out$fatal)) {
    note("LISREL output is a failed run: ", out$fatal[[1L]])
    return(list(ok = FALSE, issues = issues, df = NA, chisq = NA,
                lisrel_df = NA, lisrel_chisq = NA, max_est_diff = NA,
                max_moment_diff = NA, n_free = NA, unprinted = NA,
                printed = NULL, lisrel_warnings = out$warnings))
  }
  if (length(out$spec) != G) note("spec sections: ", length(out$spec),
                                  " for ", G, " groups")
  if (length(out$est) != G) note("estimate sections: ", length(out$est),
                                 " for ", G, " groups")
  if (length(out$nobs) >= G && any(out$nobs[seq_len(G)] != data$n)) {
    note("N: LISREL ", paste(out$nobs, collapse = "/"), " vs ",
         paste(data$n, collapse = "/"))
  }

  # Pattern: fixed <-> 0, CO <-> Constr'd, free classes <-> LISREL numbers.
  ours <- character(nrow(el))
  theirs <- character(nrow(el))
  cls <- match(el$rep, unique(el$rep[el$free]))
  for (r in seq_len(nrow(el))) {
    e <- el[r, ]
    ours[[r]] <- if (!e$free) "0" else if (e$constrained) "C"
                 else as.character(cls[[r]])
    s <- if (e$g <= length(out$spec)) out$spec[[e$g]][[e$m]] else NULL
    v <- if (is.null(s)) "0" else s[e$i, e$j]
    theirs[[r]] <- if (is.na(v)) NA_character_ else if (v == "Constr'd") "C"
                   else v
  }
  unprinted <- sum(is.na(theirs))
  map <- unique(data.frame(o = ours, t = theirs,
                           stringsAsFactors = FALSE)[!is.na(theirs), ])
  bad_fix <- map[(map$o == "0") != (map$t == "0") |
                   (map$o == "C") != (map$t == "C"), , drop = FALSE]
  if (nrow(bad_fix)) {
    r <- which(ours == bad_fix$o[[1L]] & theirs == bad_fix$t[[1L]])[[1L]]
    note("pattern differs at ", el$key[[r]], ": ours ", ours[[r]],
         ", LISREL ", theirs[[r]])
  }
  fr <- map[!map$o %in% c("0", "C"), , drop = FALSE]
  if (anyDuplicated(fr$o) || anyDuplicated(fr$t)) {
    d <- if (anyDuplicated(fr$o)) fr$o[duplicated(fr$o)][[1L]]
         else fr$t[duplicated(fr$t)][[1L]]
    note("equality classes differ (class ", d, ")")
  }

  # Estimates, fixed values included.
  idx <- fin$match
  est <- numeric(nrow(el))
  est[!is.na(idx)] <- pe$est[idx[!is.na(idx)]]
  est[is.na(idx)] <- 0
  worst <- 0
  worst_key <- ""
  printed <- rep(NA_real_, nrow(el))
  for (r in seq_len(nrow(el))) {
    e <- el[r, ]
    s <- if (e$g <= length(out$est)) out$est[[e$g]][[e$m]] else NULL
    if (is.null(s)) {
      # LISREL omits all-fixed zero and identity matrices from the estimates
      blk <- el[el$g == e$g & el$m == e$m, , drop = FALSE]
      ident <- blk$i == blk$j & !(e$m %in% c("TY", "AL"))
      if (!any(blk$free) && (all(blk$val == 0) ||
                             all(blk$val == as.numeric(ident)))) next
    }
    v <- if (is.null(s)) "- -" else s[e$i, e$j]
    if (is.na(v)) next
    lv <- if (v == "- -" || v == "NA") 0 else as.numeric(v)
    if (is.na(lv)) next
    printed[[r]] <- lv
    dlt <- abs(est[[r]] - lv)
    if (dlt > worst) { worst <- dlt; worst_key <- e$key }
  }
  if (worst > tol_est) note("estimate ", worst_key, " differs by ",
                            signif(worst, 3))

  fm <- lavaan::fitMeasures(fit, c("chisq", "df"))
  if (!is.na(out$df) && fm[["df"]] != out$df) {
    note("df ", fm[["df"]], " vs LISREL ", out$df)
  }
  chi_tol <- max(0.0015, 2e-5 * abs(out$chisq))
  if (!is.na(out$chisq) && abs(fm[["chisq"]] - out$chisq) > chi_tol) {
    note("chisq ", signif(fm[["chisq"]], 8), " vs LISREL ", out$chisq)
  }
  mom_diff <- NA_real_
  if (length(out$moments) == G) {
    mom_diff <- max(vapply(seq_len(G), function(g) {
      m <- out$moments[[g]]
      s <- data$cov[[g]]
      if (is.null(m) || any(dim(m) != dim(s))) return(Inf)
      max(abs(m[lower.tri(m, diag = TRUE)] - s[lower.tri(s, diag = TRUE)]),
          na.rm = TRUE)
    }, numeric(1L)))
    if (mom_diff > 0.5 * 10^(-nd) + 1e-4) {
      note("input moments differ by ", signif(mom_diff, 3))
    }
  }
  list(ok = !length(issues), issues = issues, df = unname(fm[["df"]]),
       chisq = unname(fm[["chisq"]]), lisrel_df = out$df,
       lisrel_chisq = out$chisq, max_est_diff = worst,
       max_moment_diff = mom_diff,
       n_free = length(unique(el$rep[el$free & !el$constrained])),
       unprinted = unprinted,
       printed = data.frame(lhs = want$lhs, op = want$op, rhs = want$rhs,
                            group = want$group, free = want$free,
                            lisrel = printed, stringsAsFactors = FALSE)[
                              !is.na(printed) & (want$free | printed != 0), ],
       lisrel_warnings = out$warnings)
}

# --- one input end to end --------------------------------------------------

# Groups may select different variables for the same positions (Little's
# MTMM example selects wave 1, 2 and 3 indicators). LISREL is positional, so
# the lavaan names come from the longest label suffix common to all groups.
lis_group_names <- function(dat) {
  nms <- do.call(rbind, lapply(dat, `[[`, "names"))
  if (nrow(nms) == 1L || all(apply(nms, 2L, function(x) length(unique(x)) == 1L))) {
    return(nms[1L, ])
  }
  common <- apply(nms, 2L, function(x) {
    n <- min(nchar(x))
    k <- 0L
    while (k < n && length(unique(substring(x, nchar(x) - k))) == 1L) k <- k + 1L
    substring(x[[1L]], nchar(x[[1L]]) - k + 1L)
  })
  if (any(!nzchar(common)) || anyDuplicated(common)) {
    common <- paste0("Y", seq_len(ncol(nms)))
  }
  common
}

lis_translate <- function(ls8, out_path = NULL, labels = NULL) {
  parsed <- lis_parse(ls8)
  if (!is.null(labels)) {
    for (g in seq_along(parsed$groups)) {
      la <- parsed$groups[[g]]$la
      la[as.integer(names(labels))] <- unname(labels)
      parsed$groups[[g]]$la <- la
    }
  }
  model <- lis_build_model(parsed)
  G <- model$G
  dat <- lapply(parsed$groups, lis_group_data, ctx = parsed$ctx)
  obs <- lis_group_names(dat)
  if (length(obs) != model$ny) lis_fail("NY=", model$ny, " but ", length(obs),
                                        " selected variables")
  lat <- parsed$groups[[1L]]$le %||% paste0("ETA", seq_len(model$ne))
  tr <- lis_to_lavaan(model, obs, lat)
  covs <- lapply(dat, function(d) {
    s <- d$cov
    dimnames(s) <- list(tr$obs, tr$obs)
    s
  })
  means <- NULL
  if (model$meanstructure) {
    means <- lapply(dat, function(d) {
      if (is.null(d$mean)) lis_fail("the model has means but the data do not")
      stats::setNames(d$mean, tr$obs)
    })
  }
  ns <- vapply(dat, function(d) as.integer(d$n), integer(1L))
  one <- function(x) if (G == 1L) x[[1L]] else x
  # ceq.simple reparameterizes label equalities instead of running lavaan's
  # nested constrained optimizer, which is very slow once a bound is present
  # (the Ch6 cognition simplex: minutes versus half a second, same optimum).
  # The iteration cap keeps an ill-conditioned default-start fit from
  # stalling the build; a non-converged fit falls through to LISREL's starts.
  fit_fun <- function(txt, do.fit = FALSE, start = "default") {
    lavaan::sem(txt, sample.cov = one(covs), sample.mean = if (is.null(means)) NULL else one(means),
                sample.nobs = ns, meanstructure = model$meanstructure,
                likelihood = "wishart", fixed.x = FALSE, do.fit = do.fit,
                start = start, warn = FALSE, se = "none", ceq.simple = TRUE,
                control = list(iter.max = 3000L, eval.max = 6000L))
  }
  fin <- lis_finalize(tr, fit_fun)
  try_fit <- function(start = "default") {
    f <- tryCatch(suppressWarnings(fit_fun(fin$model, do.fit = TRUE,
                                           start = start)),
                  error = function(e) NULL)
    if (!is.null(f) && !isTRUE(lavaan::lavInspect(f, "converged"))) NULL else f
  }
  fit <- try_fit()
  nd <- 2L
  for (g in parsed$groups) if (!is.null(g$ou$ND)) nd <- as.integer(g$ou$ND)
  titles <- lapply(parsed$groups, function(g) {
    c(g$title, trimws(g$da_line))
  })
  res <- list(parsed = parsed, model = model, data = dat, tr = tr,
              syntax = fin$model, fit = fit, covs = covs, means = means,
              ns = ns, nd = nd, out = NULL, verify = NULL)
  if (is.null(out_path)) {
    cand <- sub("\\.[Ll][Ss]8$", ".OUT", ls8)
    hits <- list.files(dirname(ls8), full.names = TRUE)
    hit <- hits[tolower(basename(hits)) == tolower(basename(cand))]
    out_path <- if (length(hit)) hit[[1L]] else NA_character_
  }
  if (is.na(out_path)) {
    if (is.null(fit)) lis_fail("lavaan could not fit the translation")
    return(res)
  }
  res$out <- lis_read_out(out_path, titles, model$ny, model$ne)
  verify <- function(f) lis_verify(tr, fin, f, res$out,
                                   list(n = ns, cov = covs), nd)
  v <- if (is.null(fit)) NULL else verify(fit)
  start_used <- "default"
  # Default starts can stop at a worse local optimum than LISREL's, or
  # diverge. The model is still LISREL's if the input's own start values, or
  # LISREL's printed solution, reach the printed optimum.
  fit_issue <- is.null(v) ||
    (any(grepl("^(chisq|estimate)", v$issues)) &&
       !any(grepl("^(pattern|equality|df|N:|spec|input)", v$issues)))
  if (fit_issue) {
    for (how in c("lisrel_input", "lisrel_solution")) {
      f2 <- try_fit(lis_start_table(fin, tr, res$out, how))
      if (is.null(f2)) next
      v2 <- verify(f2)
      if (v2$ok || is.null(v)) {
        fit <- f2
        v <- v2
        start_used <- how
        if (v2$ok) break
      }
    }
  }
  if (is.null(v)) lis_fail("lavaan could not fit the translation")
  v$start <- start_used
  res$fit <- fit
  res$verify <- v
  res
}

# Start values for a verification refit: the values the LISREL input set
# (ST/VA/MA) or the estimates LISREL printed.
lis_start_table <- function(fin, tr, out, how) {
  pt <- fin$partable
  el <- tr$elements
  st <- pt$start
  idx <- fin$match
  for (r in which(!is.na(idx) & el$free)) {
    e <- el[r, ]
    v <- if (how == "lisrel_input") e$val else {
      s <- out$est[[e$g]][[e$m]]
      x <- if (is.null(s)) NA else s[e$i, e$j]
      if (is.na(x) || x %in% c("- -", "NA")) 0 else as.numeric(x)
    }
    if (how == "lisrel_input" && v == 0 && e$m %in% c("PS", "TE") &&
        e$i == e$j) next
    st[[idx[[r]]]] <- v
  }
  pt$est <- st
  pt
}
