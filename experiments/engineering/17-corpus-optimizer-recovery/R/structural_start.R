# Experiment-only prototype. No package/default changes. Affine constraints are
# interpreted here solely to test the proposed moment-based construction.
affine_start_system <- function(p,n) {
  unit<-function(i){z<-numeric(n+1L);z[i+1L]<-1;z}
  symbols<-list();extra<-list()
  for(i in which(p$op %in% c('=~','~','~~','~1'))) {
    z<-if(p$free[i]>0) unit(p$free[i]) else c(p$ustart[i],numeric(n))
    for(k in c(p$plabel[i],p$label[i])) if(!is.na(k) && nzchar(k)) {
      if(!is.null(symbols[[k]])) extra[[length(extra)+1L]]<-z-symbols[[k]] else symbols[[k]]<-z
    }
  }
  expr<-function(e) {
    if(is.numeric(e)) return(c(as.numeric(e),numeric(n)))
    if(is.symbol(e)) {v<-symbols[[as.character(e)]];if(is.null(v)) stop('unknown constraint symbol');return(v)}
    op<-as.character(e[[1]]);if(!op %in% c('(','+','-','*','/')) stop('nonlinear/unsupported constraint')
    a<-expr(e[[2]]);if(op=='(')return(a)
    if(length(e)==2) return(if(op=='-') -a else a)
    b<-expr(e[[3]])
    if(op=='+') return(a+b)
    if(op=='-') return(a-b)
    if(op=='*' && all(a[-1]==0)) return(a[1]*b)
    if(op=='*' && all(b[-1]==0)) return(b[1]*a)
    if(op=='/' && all(b[-1]==0) && b[1]!=0) return(a/b[1])
    stop('nonlinear/unsupported constraint')
  }
  if(any(p$op %in% c('<','>'))) stop('inequality constraint')
  for(i in which(p$op=='==')) extra[[length(extra)+1L]]<-expr(parse(text=p$lhs[i])[[1]])-expr(parse(text=p$rhs[i])[[1]])
  for(i in which(p$free>0 & is.finite(p$ustart))) {z<-unit(p$free[i]);z[1]<--p$ustart[i];extra[[length(extra)+1L]]<-z}
  if(!length(extra)) return(list(A=matrix(0,0,n),b=numeric()))
  e<-do.call(rbind,extra);list(A=e[,-1,drop=FALSE],b=-e[,1])
}
least_correction <- function(A,b) {
  if(!nrow(A)) return(numeric(ncol(A)))
  s<-svd(A);keep<-s$d>max(dim(A))*max(s$d,1)*1e-12
  if(!any(keep)) return(numeric(ncol(A)))
  as.numeric(s$v[,keep,drop=FALSE] %*% (crossprod(s$u[,keep,drop=FALSE],b)/s$d[keep]))
}
structural_start <- function(case,x) {
  p<-case$model$partable;n<-length(x);original<-as.numeric(x)
  result<-function(reason,theta=original,paths=0L) list(theta=theta,reason=reason,paths=paths)
  tryCatch({
    # Canonicalize higher-order loadings into the same directed latent edges.
    lv<-unique(p$lhs[p$op=='=~']);edge<-p$op=='=~' & p$rhs %in% lv
    left<-p$lhs[edge];p$lhs[edge]<-p$rhs[edge];p$rhs[edge]<-left;p$op[edge]<-'~'
    active<-p$free>0 | (!is.na(p$ustart) & p$ustart!=0)
    candidates<-integer()
    for(i in which(p$op=='~' & p$lhs %in% lv & p$rhs %in% lv & p$free>0 & !is.finite(p$ustart))) {
      g<-p$group[i];child<-p$lhs[i]
      incoming<-which(p$op=='~' & p$lhs==child & p$group==g & active)
      v<-which(p$op=='~~' & p$lhs==child & p$rhs==child & p$group==g)
      # A deterministic, single-parent child identifies a scale path. Ordinary
      # regressions and noisy multiple-parent equations are outside this pilot.
      if(length(incoming)!=1 || length(v)!=1 || p$free[v]!=0 || p$ustart[v]!=0) next
      cov<-which(p$op=='~~' & p$group==g & xor(p$lhs==child,p$rhs==child))
      if(any(active[cov])) next
      l<-which(p$op=='=~' & p$lhs==child & p$group==g & active)
      if(length(l)<3 || any(!p$rhs[l] %in% colnames(case$sample$S[[g]]))) next
      cross<-vapply(p$rhs[l],function(ov)sum(p$op=='=~' & p$rhs==ov & p$group==g & active)>1,logical(1))
      if(any(cross)) next
      candidates<-c(candidates,i)
    }
    if(!length(candidates)) return(result('no supported scale paths'))
    con<-affine_start_system(p,n)
    x<-original+least_correction(con$A,con$b-as.numeric(con$A%*%original))
    if(nrow(con$A) && max(abs(con$A%*%x-con$b))>1e-8) return(result('infeasible constraints/hints'))
    val<-function(i) ifelse(p$free[i]>0,x[pmax(p$free[i],1L)],p$ustart[i])
    proposal<-list();idx<-integer()
    for(g in unique(p$group[candidates])) {
      factors<-unique(p$lhs[p$op=='=~' & p$group==g]);L<-length(factors)
      if(any(p$op=='~' & p$lhs %in% factors & !p$rhs %in% factors & p$group==g & active))
        stop('observed regressors unsupported')
      B<-Psi<-matrix(0,L,L,dimnames=list(factors,factors))
      er<-which(p$op=='~' & p$lhs %in% factors & p$rhs %in% factors & p$group==g)
      B[cbind(match(p$lhs[er],factors),match(p$rhs[er],factors))]<-val(er)
      graph<-matrix(FALSE,L,L);graph[cbind(match(p$lhs[er[active[er]]],factors),match(p$rhs[er[active[er]]],factors))]<-TRUE
      left<-seq_len(L);order<-integer()
      while(length(left)) {roots<-left[rowSums(graph[left,left,drop=FALSE])==0];if(!length(roots))stop('cyclic latent structure');order<-c(order,roots);left<-setdiff(left,roots)}
      vr<-which(p$op=='~~' & p$lhs %in% factors & p$rhs %in% factors & p$group==g)
      for(j in vr) Psi[p$lhs[j],p$rhs[j]]<-Psi[p$rhs[j],p$lhs[j]]<-val(j)
      for(child in factors[order]) for(i in candidates[p$group[candidates]==g & p$lhs[candidates]==child]) {
        T<-solve(diag(L)-B);Phi<-T%*%Psi%*%t(T);parent<-p$rhs[i];pv<-Phi[parent,parent]
        if(!is.finite(pv) || pv<=0) next
        l<-which(p$op=='=~' & p$lhs==child & p$group==g & active);lam<-val(l);ov<-p$rhs[l]
        pairs<-combn(seq_along(l),2);numer<-denom<-0
        for(k in seq_len(ncol(pairs))) {
          a<-pairs[1,k];b<-pairs[2,k]
          r<-which(p$op=='~~' & p$group==g & ((p$lhs==ov[a]&p$rhs==ov[b]) | (p$lhs==ov[b]&p$rhs==ov[a])))
          if(any(p$free[r]>0)) next
          residual<-if(length(r))val(r[1]) else 0
          w<-lam[a]*lam[b];numer<-numer+w*(case$sample$S[[g]][ov[a],ov[b]]-residual);denom<-denom+w*w
        }
        v<-numer/denom;if(!is.finite(v)||v<=0)next
        sign<-if(x[p$free[i]]<0) -1 else 1
        a<-sign*sqrt(v/pv);idx<-c(idx,p$free[i]);proposal[[length(proposal)+1L]]<-a
        B[child,parent]<-a
      }
    }
    if(!length(idx)) return(result('no usable moment estimates'))
    ids<-sort(unique(idx));target<-vapply(ids,function(k)mean(unlist(proposal)[idx==k]),numeric(1))
    delta<-target-x[ids];A<-con$A[,ids,drop=FALSE]
    delta<-delta-least_correction(A,as.numeric(A%*%delta));candidate<-x;candidate[ids]<-x[ids]+delta
    if(any(!is.finite(candidate)) || any(abs(candidate[ids])<1e-10)) return(result('degenerate constrained proposal'))
    if(nrow(con$A) && max(abs(con$A%*%candidate-con$b))>1e-8) return(result('constraint residual'))
    # The caller checks the original objective; retain a failed proposal's
    # reason rather than silently changing the estimator/feasible set.
    result('moment scale paths',candidate,length(ids))
  },error=function(e)result(conditionMessage(e)))
}
