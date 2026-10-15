# Populations specific to the approved psd-factor lane. F1 remains unchanged.
psd_factor_populations <- function() {
  f2 <- lapply(list(c(.95,.5,.5), c(.99,.5,.5), c(1,.5,.5), c(.99,.99,.99)), function(r) {
    phi <- diag(3); phi[lower.tri(phi)] <- r; phi <- phi + t(phi) - diag(3)
    nm <- paste0("x", 1:18)
    lambda <- kronecker(diag(3), matrix(.7, 6, 1))
    p <- population("F2", paste0("collinear_", paste(r, collapse = "_")), "test",
      "psd-factor criteria F2", named(lambda %*% phi %*% t(lambda) + diag(.51,18), nm),
      list(fitted_model("three_factor", cfa_syntax(split(1:18, rep(1:3,each=6)),nm))))
    p$ns <- c(50L,100L); p
  })
  lambda <- matrix(.7,8,1); theta <- diag(.51,8)
  v <- c(.6,.6,.3); w <- c(0,.3,.6)
  theta[1:3,1:3] <- tcrossprod(v) + tcrossprod(w)
  f3 <- population("F3", "singular_residual", "test", "psd-factor criteria F3",
    named(tcrossprod(lambda)+theta,paste0("x",1:8)),
    list(fitted_model("one_factor_residual", paste("f1 =~",paste0("x",1:8,collapse=" + "),
      "\nx1 ~~ x2 + x3\nx2 ~~ x3"))))
  f3$ns <- c(100L,200L)
  control <- population("control_weak_marker", "weak_marker_three_factor", "control", "development disclosure",
    named(kronecker(diag(3), tcrossprod(c(.5,rep(.3,5)))) +
      diag(rep(1-c(.5,rep(.3,5))^2,3)),paste0("x",1:18)),
    list(fitted_model("three_factor",cfa_syntax(split(1:18,rep(1:3,each=6)),paste0("x",1:18)))))
  control$ns <- 100L
  f1 <- all_populations()
  f1 <- lapply(f1,function(p) {p$ns <- population_ns(p); p$family <- paste0("F1_",p$family); p})
  c(f1,f2,list(f3,control))
}
