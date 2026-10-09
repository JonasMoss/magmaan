# Source helpers.R, then this file, as for the other repo-local support helpers.
# Keep nested records intact: groups and population values are plain input data.
basis_ids <- function() {
  path <- support_path("basis", "sem_score")
  if (!dir.exists(path)) stop("Tier-1 basis directory is missing: ", path,
                              call. = FALSE)
  sort(sub("\\.json$", "", list.files(path, pattern = "\\.json$")))
}

basis_model <- function(id) {
  if (!is.character(id) || length(id) != 1L || is.na(id) ||
      !id %in% basis_ids()) {
    stop("Unknown tier-1 basis id: ", paste(id, collapse = ", "), call. = FALSE)
  }
  require_pkg("jsonlite")
  record <- jsonlite::fromJSON(support_path("basis", "sem_score",
                                          paste0(id, ".json")),
                               simplifyVector = FALSE)
  required <- c("id", "corpus", "type", "h1", "h0", "groups",
                "meanstructure", "source_n")
  if (!all(required %in% names(record)) || !identical(record$id, id)) {
    stop("Invalid tier-1 basis record: ", id, call. = FALSE)
  }
  record
}
