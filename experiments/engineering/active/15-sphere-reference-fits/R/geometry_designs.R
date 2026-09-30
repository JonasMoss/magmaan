# Deterministic covariance witnesses; mathematical classes precede optimization.
geometry_designs <- function() {
  one <- function(load, residual, kind) {
    S <- tcrossprod(load) + diag(residual)
    ov <- paste0('x', seq_along(load)); dimnames(S) <- list(ov, ov)
    list(sigma=S, blocks=list(X=ov), structural=character(),
      syntax=paste('X =~', paste(ov, collapse=' + ')), kind=kind, std_lv=FALSE)
  }
  regular <- one(c(.7,.8,.6,.5), rep(1,4), 'finite_interior')
  pole <- one(c(0,.8,.6,.5), rep(1,4), 'marker_pole')
  face <- one(c(.7,.8,.6,.5), c(0,1,1,1), 'psd_residual_face')
  # Y=.6X with zero disturbance. Positive indicator residuals keep Sigma PD.
  L <- matrix(0,6,2); L[1:3,1] <- c(.7,.8,.6); L[4:6,2] <- c(.9,.7,.5)
  S <- L %*% matrix(c(1,.6,.6,.36),2) %*% t(L) + diag(6)
  ov <- c(paste0('x',1:3),paste0('y',1:3)); dimnames(S) <- list(ov,ov)
  stdlv <- list(sigma=S, blocks=list(X=ov[1:3],Y=ov[4:6]), structural='Y ~ X',
    syntax='X =~ x1 + x2 + x3\nY =~ y1 + y2 + y3\nY ~ X',
    kind='stdlv_disturbance_pole', std_lv=TRUE)
  # One factor gives s12*s13/s23=lambda1^2*phi whenever s23 != 0.
  # Here s12=.3, s13=.4, s23=0: no finite solution, even with signed phi.
  # lambda=(t,.3/t,.4/t), phi=1, theta=1-lambda^2 approaches S.
  # The first residual becomes negative, so this sequence is ML-only.
  closure <- one(c(1,1,1), rep(1,3), 'ml_nonattainment')
  closure$sigma <- matrix(c(1,.3,.4,.3,1,0,.4,0,1),3,
    dimnames=list(paste0('x',1:3),paste0('x',1:3)))
  list(interior=regular, marker_pole=pole, residual_face=face,
       stdlv_pole=stdlv, nonattainment=closure)
}

geometry_marker <- function(d) {
  # Fixed second indicator, selected before fits; never reference-informed.
  lines <- vapply(names(d$blocks), function(f) {
    obs <- d$blocks[[f]]
    paste(f, '=~', paste(paste0(ifelse(seq_along(obs)==2,'1*','NA*'),obs),collapse=' + '))
  }, '')
  magmaanlab::model_spec(paste(c(lines,d$structural),collapse='\n'))
}

exact_data <- function(S, n=100L) {
  # Deterministic centered orthonormal columns give ML covariance exactly S.
  x <- outer(seq_len(n), seq_len(ncol(S)), function(i,j) cos(pi*(i-.5)*j/n))
  x <- sweep(x,2,colMeans(x)); Q <- qr.Q(qr(x))
  data <- as.data.frame(sqrt(n)*Q %*% chol(S)); names(data) <- colnames(S)
  stopifnot(max(abs(sample_moments(data)$S[[1]]-S)) < 1e-12)
  data
}

geometry_endpoint <- function(pt, d, sample, domain) {
  missing <- data.frame(covariance_gap=NA_real_, min_scaled_primitive=NA_real_,
    psd_position='unavailable', chart_position='unavailable')
  if (is.null(pt)) return(missing)
  tryCatch({
    target <- geometry_marker(d)
    tr <- magmaanlab::frontier_reidentify(pt,target,pole_tol=0)
    ev <- magmaanlab::magmaan_core$evaluate_at(target$partable,sample,tr$theta,estimator='ML')
    S <- magmaanlab::magmaan_core$model_implied(ev)$sigma[[1]]
    p <- ev$partable; lv <- names(d$blocks); k <- length(lv)
    Psi <- matrix(0,k,k,dimnames=list(lv,lv))
    for(i in which(p$op=='~~' & p$lhs %in% lv)) {
      Psi[p$lhs[i],p$rhs[i]] <- p$est[i]; Psi[p$rhs[i],p$lhs[i]] <- p$est[i]
    }
    energy <- vapply(lv,function(f) {
      z<-p[p$lhs==f & p$op=='=~',]; sum(z$est^2/diag(sample$S[[1]])[z$rhs])
    },0)
    residual <- p[p$op=='~~' & p$lhs==p$rhs & !p$lhs %in% lv,]
    mineig <- min(eigen(Psi*sqrt(outer(energy,energy)),symmetric=TRUE,only.values=TRUE)$values,
      residual$est/diag(sample$S[[1]])[residual$lhs])
    original <- magmaanlab::model_spec(d$syntax,std_lv=d$std_lv)
    chart <- tryCatch({magmaanlab::frontier_reidentify(pt,original,pole_tol=1e-6); 'representable'},
                      error=function(e) 'translation_failed_or_near_boundary')
    data.frame(covariance_gap=max(abs(S-d$sigma)/sqrt(outer(diag(d$sigma),diag(d$sigma)))),
      min_scaled_primitive=mineig,
      psd_position=if(mineig < -1e-6) 'outside_PSD' else if(mineig<=1e-7) 'near_PSD_face' else 'PSD_interior',
      chart_position=chart)
  },error=function(e) missing)
}
