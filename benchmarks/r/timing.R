# Shared batched wall timer. Callers own validation and result serialization.
# Comparable arms rotate within a workload; unrelated stages are timed separately.
time_paired <- function(arms, batches = 7L, target_ms = 100, warmups = 2L,
                        session = 1L, max_calls = 10000L) {
  stopifnot(length(arms) > 0L, !is.null(names(arms)), batches > 0,
            target_ms > 0, warmups >= 0, max_calls > 0)
  elapsed <- function(fun, calls) {
    start <- Sys.time()
    for (i in seq_len(calls)) value <- fun()
    # R eagerly evaluates the call; retain the last value through the clock read.
    ms <- as.numeric(difftime(Sys.time(), start, units = 'secs')) * 1000
    invisible(value)
    ms
  }
  calls <- vapply(arms, function(fun) {
    for (i in seq_len(warmups)) invisible(fun())
    k <- 1L
    repeat {
      ms <- elapsed(fun, k)
      if (ms >= target_ms / 4 || k >= max_calls) break
      k <- min(max_calls, k * 4L)
    }
    as.integer(min(max_calls, max(1, ceiling(k * target_ms / max(ms, .001)))))
  }, integer(1))
  rows <- list()
  for (b in seq_len(batches)) {
    order <- ((seq_along(arms) + b + session - 3L) %% length(arms)) + 1L
    for (j in seq_along(order)) {
      a <- order[j]
      gc(FALSE) # no forced collection inside a batch; natural GC is included
      ms <- elapsed(arms[[a]], calls[a])
      rows[[length(rows) + 1L]] <- data.frame(session = session, batch = b,
        order = j, engine = names(arms)[a], calls = unname(calls[a]),
        elapsed_ms = ms, ms = ms / calls[a])
    }
  }
  do.call(rbind, rows)
}
