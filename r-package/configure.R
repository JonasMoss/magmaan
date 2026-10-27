# Shared portable configuration for configure and configure.win.
flags <- Sys.getenv("NLOPT_CFLAGS")
libs <- Sys.getenv("NLOPT_LIBS")
probe <- function(option) {
  if (!nzchar(Sys.which("pkg-config"))) return("")
  result <- suppressWarnings(system2("pkg-config", c(option, "nlopt"),
                                    stdout = TRUE, stderr = FALSE))
  if (!is.null(attr(result, "status"))) return("")
  paste(result, collapse = " ")
}
if (!nzchar(flags)) flags <- probe("--cflags")
if (!nzchar(libs)) libs <- probe("--libs")
if (!nzchar(libs) && .Platform$OS.type == "windows")
  stop("NLopt was not found: install Rtools with its NLopt library, or set NLOPT_CFLAGS and NLOPT_LIBS. The nloptr DLL does not export the NLopt C API.")
if (!nzchar(libs)) {
  dll <- list.files(system.file("libs", package = "nloptr"),
                    pattern = "nloptr[.]so$", full.names = TRUE, recursive = TRUE)
  if (length(dll)) {
    directory <- normalizePath(dirname(dll[1]), winslash = "/")
    if (identical(Sys.info()[["sysname"]], "Linux")) {
      # Linking the absolute .so path embeds the build machine in DT_NEEDED.
      # A sibling-library runpath also supports relocated binary installs.
      libs <- paste0("-L", shQuote(directory), " -l:nloptr.so ",
                     "-Wl,-rpath,'$$ORIGIN/../../nloptr/libs' ",
                     "-Wl,-rpath,", shQuote(directory))
    } else {
      libs <- paste(shQuote(normalizePath(dll[1], winslash = "/")),
                    paste0("-Wl,-rpath,", shQuote(directory)))
    }
  } else {
    libs <- "-lnlopt"
  }
}
sources <- list.files("src", pattern = "[.](cpp|c)$", recursive = TRUE)
objects <- paste(sub("[.](cpp|c)$", ".o", sort(sources)), collapse = " \\\n  ")
text <- paste0(paste(readLines("src/Makevars.in"), collapse = "\n"), "\n")
for (pair in list(c("@NLOPT_CFLAGS@", flags), c("@NLOPT_LIBS@", libs),
                 c("@OBJECTS@", objects))) {
  text <- paste(strsplit(text, pair[1], fixed = TRUE)[[1]], collapse = pair[2])
}
output <- commandArgs(trailingOnly = TRUE)
stopifnot(length(output) == 1L)
writeLines(text, output)
