# Read populations/<id>.json into the in-memory form the generators expect.
read_population <- function(path) {
  p <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  p$groups <- lapply(p$groups, function(g) {
    ov <- unlist(g$ov)
    g$sigma <- matrix(unlist(g$sigma), length(ov), byrow = TRUE, dimnames = list(ov, ov))
    g$mu <- unlist(g$mu)
    g$ov <- ov
    g
  })
  p
}

read_populations <- function(dir) {
  files <- list.files(dir, pattern = "[.]json$", full.names = TRUE)
  pops <- lapply(files, read_population)
  names(pops) <- vapply(pops, `[[`, character(1), "id")
  pops[case_order[case_order %in% names(pops)]]
}

case_order <- c("cfa_ptsd8", "cfa_mtmm9", "cfa_second12", "cfa_long18",
                "sem_worland11", "growth6", "mg_path5")
sample_sizes <- c(100L, 300L, 1000L)

# The canonical grid fixes each cell's id, and so its seeds, independently of
# which cells a particular invocation selects.
full_grid <- function() {
  g <- expand.grid(n = sample_sizes, dgp = dgps, case = case_order, stringsAsFactors = FALSE)
  g <- g[, c("case", "dgp", "n")]
  g$cell_id <- seq_len(nrow(g))
  g$key <- sprintf("%s:%s:%d", g$case, g$dgp, g$n)
  g
}
