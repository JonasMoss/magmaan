#pragma once
#include "magmaan/robust/prepared_ntml.hpp"
namespace magmaanr {
inline std::shared_ptr<magmaan::robust::frontier::NTMLFit> ntml_snapshot(Rcpp::List fit) {
  SEXP ptr=fit.attr("magmaan_ntml");
  if (TYPEOF(ptr)!=EXTPTRSXP || !R_ExternalPtrAddr(ptr) ||
      R_ExternalPtrTag(ptr)!=Rf_install("magmaan_ntml_fit")) return {};
  Rcpp::List keys(R_ExternalPtrProtected(ptr));
  int i=0;
  for (const char* name : {"partable","S","nobs","sample_mean","theta","fmin","raw_data"}) {
    if (!fit.containsElementNamed(name) || SEXP(keys[i++])!=SEXP(fit[name])) return {};
  }
  return *Rcpp::XPtr<std::shared_ptr<magmaan::robust::frontier::NTMLFit>>(ptr);
}
inline void validate_ntml_raw(const magmaan::robust::frontier::NTMLFit& fit,
                              const magmaan::data::RawData& raw) {
  if (raw.X.size()!=fit.data->raw.X.size()) Rcpp::stop("prepared inference data layout differs");
  for (std::size_t b=0;b<raw.X.size();++b)
    if (raw.X[b].rows()!=fit.data->raw.X[b].rows() || raw.X[b].cols()!=fit.data->raw.X[b].cols() ||
        !(raw.X[b].array()==fit.data->raw.X[b].array()).all())
      Rcpp::stop("prepared inference data observations differ");
}

}
