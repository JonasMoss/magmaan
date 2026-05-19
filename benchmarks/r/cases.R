benchmark_cases <- list(
  hs_3factor_cfa = list(
    id = "hs_3factor_cfa",
    tier = 0L,
    status = "active",
    estimator = "ML",
    source_type = "lavaan_dataset",
    package = "lavaan",
    dataset = "HolzingerSwineford1939",
    variables = paste0("x", 1:9),
    lavaan_function = "cfa",
    meanstructure = FALSE,
    supported_now = TRUE
  ),
  hs_3factor_cfa_fiml_masked = list(
    id = "hs_3factor_cfa_fiml_masked",
    tier = 0L,
    status = "active",
    estimator = "FIML",
    magmaan_estimator = "FIML",
    lavaan_estimator = "ML",
    lavaan_missing = "fiml",
    source_type = "lavaan_dataset",
    package = "lavaan",
    dataset = "HolzingerSwineford1939",
    variables = paste0("x", 1:9),
    data_transform = "hs_controlled_missing",
    lavaan_function = "cfa",
    meanstructure = TRUE,
    supported_now = TRUE
  ),
  hs_3factor_cfa_uls = list(
    id = "hs_3factor_cfa_uls",
    tier = 0L,
    status = "active",
    estimator = "ULS",
    source_type = "lavaan_dataset",
    package = "lavaan",
    dataset = "HolzingerSwineford1939",
    variables = paste0("x", 1:9),
    lavaan_function = "cfa",
    meanstructure = FALSE,
    est_tolerance = 5e-3,
    supported_now = TRUE
  ),
  hs_3factor_cfa_gls = list(
    id = "hs_3factor_cfa_gls",
    tier = 0L,
    status = "active",
    estimator = "GLS",
    source_type = "lavaan_dataset",
    package = "lavaan",
    dataset = "HolzingerSwineford1939",
    variables = paste0("x", 1:9),
    lavaan_function = "cfa",
    meanstructure = FALSE,
    est_tolerance = 5e-3,
    supported_now = TRUE
  ),
  bollen_democracy_sem = list(
    id = "bollen_democracy_sem",
    tier = 1L,
    status = "active",
    estimator = "ML",
    source_type = "lavaan_dataset",
    package = "lavaan",
    dataset = "PoliticalDemocracy",
    variables = c(paste0("y", 1:8), paste0("x", 1:3)),
    lavaan_function = "sem",
    meanstructure = FALSE,
    supported_now = TRUE
  ),
  demo_growth_linear = list(
    id = "demo_growth_linear",
    tier = 1L,
    status = "active",
    estimator = "ML",
    source_type = "lavaan_dataset",
    package = "lavaan",
    dataset = "Demo.growth",
    variables = c(paste0("t", 1:4), paste0("x", 1:2), paste0("c", 1:4)),
    lavaan_function = "growth",
    meanstructure = TRUE,
    supported_now = TRUE
  ),
  bfi_5factor = list(
    id = "bfi_5factor",
    tier = 1L,
    status = "active",
    estimator = "ML",
    source_type = "r_package_dataset",
    package = "psychTools",
    dataset = "bfi",
    variables = c(paste0("A", 1:5), paste0("C", 1:5), paste0("E", 1:5),
                  paste0("N", 1:5), paste0("O", 1:5)),
    lavaan_function = "cfa",
    meanstructure = FALSE,
    supported_now = TRUE
  ),
  openmx_mimic = list(
    id = "openmx_mimic",
    tier = 1L,
    status = "dormant",
    estimator = "ML",
    source_type = "manual_summary",
    source_url = "https://openmx.ssri.psu.edu/wiki/example-models",
    supported_now = FALSE
  ),
  stata_higher_order = list(
    id = "stata_higher_order",
    tier = 1L,
    status = "candidate",
    estimator = "ML",
    source_type = "stata_dta",
    source_url = "https://www.stata-press.com/data/r15/sem_hcfa1.dta",
    raw_file = "sem_hcfa1.dta",
    lavaan_function = "sem",
    meanstructure = FALSE,
    supported_now = FALSE
  ),
  stata_correlated_uniqueness = list(
    id = "stata_correlated_uniqueness",
    tier = 1L,
    status = "candidate",
    estimator = "ML",
    source_type = "stata_dta",
    source_url = "https://www.stata-press.com/data/r15/sem_cu1.dta",
    raw_file = "sem_cu1.dta",
    lavaan_function = "cfa",
    meanstructure = FALSE,
    supported_now = FALSE
  ),
  stata_growth = list(
    id = "stata_growth",
    tier = 1L,
    status = "candidate",
    estimator = "ML",
    source_type = "stata_dta",
    source_url = "https://www.stata-press.com/data/r15/sem_lcm.dta",
    raw_file = "sem_lcm.dta",
    lavaan_function = "growth",
    meanstructure = TRUE,
    supported_now = FALSE
  ),
  mplus_ex5_1 = list(
    id = "mplus_ex5_1",
    tier = 1L,
    status = "active",
    estimator = "ML",
    source_type = "mplus_dat",
    source_url = "https://www.statmodel.com/usersguide/chap5/ex5.1.dat",
    raw_file = "ex5.1.dat",
    variables = paste0("y", 1:6),
    lavaan_function = "cfa",
    meanstructure = FALSE,
    supported_now = TRUE
  )
)

case_ids <- function() names(benchmark_cases)

get_case <- function(case_id) {
  if (!case_id %in% names(benchmark_cases)) {
    stop("unknown benchmark case: ", case_id, call. = FALSE)
  }
  benchmark_cases[[case_id]]
}

apply_case_data_transform <- function(case, data) {
  transform <- case$data_transform %||% ""
  if (!nzchar(transform)) return(data)

  if (identical(transform, "hs_controlled_missing")) {
    out <- data
    out$x2[seq(7L, nrow(out), by = 11L)] <- NA_real_
    out$x5[seq(5L, nrow(out), by = 13L)] <- NA_real_
    out$x8[seq(3L, nrow(out), by = 17L)] <- NA_real_
    return(out)
  }

  stop("unknown benchmark data transform: ", transform, call. = FALSE)
}

load_case_source_data <- function(case) {
  if (identical(case$source_type, "lavaan_dataset")) {
    require_pkg("lavaan")
    env <- new.env(parent = emptyenv())
    utils::data(list = case$dataset, package = case$package, envir = env)
    data <- as.data.frame(get(case$dataset, envir = env))
    return(apply_case_data_transform(case, data[, case$variables, drop = FALSE]))
  }

  if (identical(case$source_type, "r_package_dataset")) {
    require_pkg(case$package)
    env <- new.env(parent = emptyenv())
    utils::data(list = case$dataset, package = case$package, envir = env)
    data <- as.data.frame(get(case$dataset, envir = env))
    return(apply_case_data_transform(case, data[, case$variables, drop = FALSE]))
  }

  if (identical(case$source_type, "stata_dta")) {
    require_pkg("haven")
    raw_path <- file.path(raw_dir(case$id), case$raw_file)
    if (!file.exists(raw_path)) {
      stop("raw Stata data not found; run fetch_data.R or download: ", raw_path, call. = FALSE)
    }
    return(apply_case_data_transform(case, as.data.frame(haven::read_dta(raw_path))))
  }

  if (identical(case$source_type, "mplus_dat")) {
    raw_path <- file.path(raw_dir(case$id), case$raw_file)
    if (!file.exists(raw_path)) {
      stop("raw Mplus data not found; run fetch_data.R or download: ", raw_path, call. = FALSE)
    }
    data <- utils::read.table(raw_path, header = FALSE, col.names = case$variables)
    return(apply_case_data_transform(case, as.data.frame(data)))
  }

  stop("case does not have an automatic data loader yet: ", case$id, call. = FALSE)
}
