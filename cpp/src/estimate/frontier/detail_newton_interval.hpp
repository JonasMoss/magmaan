#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <iterator>
#include <optional>
#include <vector>
#include <Eigen/Core>

namespace magmaan::estimate::frontier::detail {
using Wide = long double;
inline Wide below(Wide x) { return std::nextafter(x, -std::numeric_limits<Wide>::infinity()); }
inline Wide above(Wide x) { return std::nextafter(x, std::numeric_limits<Wide>::infinity()); }
struct Interval {
  Wide lower = 0, upper = 0;
  Interval() = default;
  explicit Interval(Wide x) : lower(x), upper(x) {}
  Interval(Wide lo, Wide hi) : lower(lo), upper(hi) {}
  bool zero() const { return lower == 0 && upper == 0; }
  bool one() const { return lower == 1 && upper == 1; }
  bool finite() const { return std::isfinite(lower) && std::isfinite(upper) && lower <= upper; }
};
inline Interval operator+(Interval a, Interval b) {
  if (a.zero()) return b;
  if (b.zero()) return a;
  return {below(a.lower + b.lower), above(a.upper + b.upper)};
}
inline Interval operator-(Interval a) { return {-a.upper, -a.lower}; }
inline Interval operator-(Interval a, Interval b) { return a + (-b); }
inline Interval operator*(Interval a, Interval b) {
  if (a.zero() || b.zero()) return {};
  if (a.one()) return b;
  if (b.one()) return a;
  const Wide v[] = {a.lower*b.lower,a.lower*b.upper,a.upper*b.lower,a.upper*b.upper};
  if (std::any_of(std::begin(v),std::end(v),[](Wide x){ return !std::isfinite(x); }))
    return {-std::numeric_limits<Wide>::infinity(),std::numeric_limits<Wide>::infinity()};
  return {below(*std::min_element(std::begin(v),std::end(v))),
          above(*std::max_element(std::begin(v),std::end(v)))};
}
inline Interval operator/(Interval a, Interval b) {
  if (b.lower <= 0 && b.upper >= 0)
    return {-std::numeric_limits<Wide>::infinity(),std::numeric_limits<Wide>::infinity()};
  if (b.one()) return a;
  return a * Interval(below(1/b.upper),above(1/b.lower));
}
inline Interval square_root(Interval a) {
  if (a.lower < 0 || !a.finite())
    return {-std::numeric_limits<Wide>::infinity(),std::numeric_limits<Wide>::infinity()};
  if (a.zero() || a.one()) return a;
  return {std::max(Wide(0),below(std::sqrt(a.lower))),above(std::sqrt(a.upper))};
}
struct IntervalMatrix {
  Eigen::Index rows = 0, cols = 0;
  std::vector<Interval> values;
  IntervalMatrix() = default;
  IntervalMatrix(Eigen::Index r, Eigen::Index c) : rows(r), cols(c), values(static_cast<std::size_t>(r*c)) {}
  explicit IntervalMatrix(const Eigen::MatrixXd& x) : IntervalMatrix(x.rows(),x.cols()) {
    for (Eigen::Index j=0;j<cols;++j) for (Eigen::Index i=0;i<rows;++i)
      (*this)(i,j)=Interval(static_cast<Wide>(x(i,j)));
  }
  Interval& operator()(Eigen::Index i,Eigen::Index j) { return values[static_cast<std::size_t>(i+rows*j)]; }
  const Interval& operator()(Eigen::Index i,Eigen::Index j) const { return values[static_cast<std::size_t>(i+rows*j)]; }
  IntervalMatrix transpose() const {
    IntervalMatrix out(cols,rows);
    for(Eigen::Index j=0;j<cols;++j) for(Eigen::Index i=0;i<rows;++i) out(j,i)=(*this)(i,j);
    return out;
  }
  bool finite() const { return std::all_of(values.begin(),values.end(),[](Interval x){return x.finite();}); }
};
inline IntervalMatrix identity(Eigen::Index n) {
  IntervalMatrix out(n,n); for(Eigen::Index i=0;i<n;++i) out(i,i)=Interval(1); return out;
}
inline IntervalMatrix operator+(const IntervalMatrix& a,const IntervalMatrix& b) {
  IntervalMatrix out(a.rows,a.cols);
  for(std::size_t i=0;i<a.values.size();++i) out.values[i]=a.values[i]+b.values[i];
  return out;
}
inline IntervalMatrix operator-(const IntervalMatrix& a,const IntervalMatrix& b) {
  IntervalMatrix out(a.rows,a.cols);
  for(std::size_t i=0;i<a.values.size();++i) out.values[i]=a.values[i]-b.values[i];
  return out;
}
inline IntervalMatrix operator*(const IntervalMatrix& a,const IntervalMatrix& b) {
  IntervalMatrix out(a.rows,b.cols);
  for(Eigen::Index j=0;j<b.cols;++j) for(Eigen::Index k=0;k<a.cols;++k)
    for(Eigen::Index i=0;i<a.rows;++i) out(i,j)=out(i,j)+a(i,k)*b(k,j);
  return out;
}
inline IntervalMatrix operator*(Interval a,IntervalMatrix b) {
  for(auto& x:b.values) x=a*x; return b;
}
inline Interval trace_product(const IntervalMatrix& a,const IntervalMatrix& b) {
  Interval out;
  for(Eigen::Index j=0;j<a.cols;++j) for(Eigen::Index i=0;i<a.rows;++i) out=out+a(i,j)*b(j,i);
  return out;
}
inline std::optional<IntervalMatrix> inverse(const IntervalMatrix& x) {
  if (x.rows != x.cols || !x.finite()) return std::nullopt;
  const Eigen::Index n=x.rows;
  IntervalMatrix a=x, b=identity(n);
  for(Eigen::Index k=0;k<n;++k) {
    Eigen::Index pivot=-1; Wide strength=0;
    for(Eigen::Index i=k;i<n;++i) {
      const auto v=a(i,k);
      if(v.lower<=0 && v.upper>=0) continue;
      const Wide size=std::min(std::abs(v.lower),std::abs(v.upper));
      if(size>strength) {pivot=i; strength=size;}
    }
    if(pivot<0) return std::nullopt;
    for(Eigen::Index j=0;j<n;++j) {std::swap(a(k,j),a(pivot,j)); std::swap(b(k,j),b(pivot,j));}
    const Interval divisor=a(k,k);
    for(Eigen::Index j=0;j<n;++j) {
      if(j!=k) a(k,j)=a(k,j)/divisor;
      b(k,j)=b(k,j)/divisor;
    }
    a(k,k)=Interval(1); // exact algebraic identities avoid interval dependency
    for(Eigen::Index i=0;i<n;++i) if(i!=k) {
      const Interval multiplier=a(i,k);
      for(Eigen::Index j=0;j<n;++j) {
        if(j!=k) a(i,j)=a(i,j)-multiplier*a(k,j);
        b(i,j)=b(i,j)-multiplier*b(k,j);
      }
      a(i,k)=Interval();
    }
  }
  return b.finite() ? std::optional<IntervalMatrix>(std::move(b)) : std::nullopt;
}
inline std::optional<IntervalMatrix> lower_inverse(const IntervalMatrix& L) {
  const Eigen::Index n=L.rows; IntervalMatrix out(n,n);
  for(Eigen::Index j=0;j<n;++j) {
    if(L(j,j).lower<=0 && L(j,j).upper>=0) return std::nullopt;
    out(j,j)=Interval(1)/L(j,j);
    for(Eigen::Index i=j+1;i<n;++i) {
      Interval sum;
      for(Eigen::Index k=j;k<i;++k) sum=sum+L(i,k)*out(k,j);
      out(i,j)=(-sum)/L(i,i);
    }
  }
  return out.finite() ? std::optional<IntervalMatrix>(std::move(out)) : std::nullopt;
}
inline std::optional<IntervalMatrix> cholesky(const IntervalMatrix& a) {
  const Eigen::Index n=a.rows; IntervalMatrix L(n,n);
  for(Eigen::Index j=0;j<n;++j) {
    Interval d=a(j,j);
    for(Eigen::Index k=0;k<j;++k) d=d-L(j,k)*L(j,k);
    if(!(d.lower>0)) return std::nullopt;
    L(j,j)=square_root(d);
    for(Eigen::Index i=j+1;i<n;++i) {
      Interval v=a(i,j);
      for(Eigen::Index k=0;k<j;++k) v=v-L(i,k)*L(j,k);
      L(i,j)=v/L(j,j);
    }
  }
  return L.finite() ? std::optional<IntervalMatrix>(std::move(L)) : std::nullopt;
}
// Frobenius enclosure of exact target minus the supplied rounded artifact.
inline double error_upper(const IntervalMatrix& exact,const Eigen::MatrixXd& rounded) {
  if(exact.rows!=rounded.rows() || exact.cols!=rounded.cols() || !exact.finite() || !rounded.allFinite())
    return std::numeric_limits<double>::infinity();
  Interval sum;
  for(Eigen::Index j=0;j<exact.cols;++j) for(Eigen::Index i=0;i<exact.rows;++i) {
    const Interval d=exact(i,j)-Interval(static_cast<Wide>(rounded(i,j)));
    const Wide e=std::max(std::abs(d.lower),std::abs(d.upper));
    sum=sum+Interval(e)*Interval(e);
  }
  sum.lower=std::max(Wide(0),sum.lower);
  const Wide norm=square_root(sum).upper;
  return std::nextafter(static_cast<double>(norm),std::numeric_limits<double>::infinity());
}
}  // namespace magmaan::estimate::frontier::detail
