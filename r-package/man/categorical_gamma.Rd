\name{categorical_gamma}
\alias{categorical_gamma}
\title{Categorical first-stage Gamma sources}
\description{The lab's categorical raw and data-frame builders accept
\code{gamma = "opg"} (default) or \code{gamma = "exact"}. Exact Gamma requires
complete observations and retains its exact sampling rows and the complete
per-case influence of its diagonal. These statistics define RDWLS weights
\code{diag(Gamma_exact)^-1}; DWLS retains conventional OPG weights.}
\details{The builders are \code{magmaan_core$data_ordinal_stats_from_raw},
\code{magmaan_core$data_mixed_ordinal_stats_from_raw},
\code{data_ordinal_stats_from_df}, and \code{data_mixed_ordinal_stats_from_df}.
On exact-Gamma stats, both first-stage inference selectors use exact rows.
RDWLS has no lavaan fitting preset or compatibility bundle. This definition of
RDWLS differs from LISREL's DWLS-with-robust-inference use of the name.}
\seealso{\code{\link{prepare_weight}}}
