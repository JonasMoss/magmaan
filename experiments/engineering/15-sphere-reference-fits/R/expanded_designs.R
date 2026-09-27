# Broader simple-structure families, fixed before the expansion run.
expanded_designs <- function() {
  make <- function(loadings, structural, B, Psi) {
    k <- length(loadings); factors <- LETTERS[seq_len(k)]
    blocks <- setNames(lapply(seq_len(k), function(j) paste0(tolower(factors[j]), seq_along(loadings[[j]]))), factors)
    observed <- unlist(blocks, use.names = FALSE)
    L <- matrix(0, length(observed), k); offset <- 0L
    for (j in seq_len(k)) { ii <- offset + seq_along(loadings[[j]]); L[ii,j] <- loadings[[j]]; offset <- max(ii) }
    A <- solve(diag(k) - B); Phi <- A %*% Psi %*% t(A)
    Sigma <- L %*% Phi %*% t(L) + diag(rep(c(.7, 1, 1.2), length.out = length(observed)))
    dimnames(Sigma) <- list(observed, observed)
    syntax <- paste(c(vapply(factors, function(f) paste(f, '=~', paste(blocks[[f]], collapse=' + ')), ''), structural), collapse='\n')
    list(blocks=blocks, structural=structural, sigma=Sigma, syntax=syntax)
  }
  B <- matrix(0,3,3); B[2,1] <- .6; B[3,2] <- .5
  list(one_factor_five = make(list(c(.15,.8,.6,-.7,.4)), character(), matrix(0,1,1), matrix(1,1,1)),
       correlated_two = make(list(c(.1,.8,-.6,.5),c(.15,.9,.7,-.5)), 'A ~~ B', matrix(0,2,2), matrix(c(1,.5,.5,1),2)),
       three_factor_chain = make(rep(list(c(.15,.8,.6)),3), c('B ~ A','C ~ B'), B, diag(c(1,.64,.75))))
}

expanded_recipes <- function(spec, sample) {
  lv <- unique(spec$partable$lhs[spec$partable$op == '=~'])
  stopifnot(length(lv) <= 3L)
  recipes <- list(canonical = NULL, spectral_positive = spectral_start(spec, sample))
  for (mask in seq_len(2^length(lv)-1L)) {
    negative <- lv[as.logical(intToBits(mask)[seq_along(lv)])]
    recipes[[paste0('negative_',paste(negative,collapse=''))]] <- spectral_start(spec, sample, negative)
  }
  recipes
}

expanded_marker <- function(design) function(pt, sample) {
  sd <- sqrt(diag(sample$S[[1]]))
  lines <- vapply(names(design$blocks), function(f) {
    z <- pt[pt$lhs == f & pt$op == '=~',]
    marker <- which.max(abs(z$est)/sd[z$rhs])
    paste(f, '=~', paste(paste0(ifelse(seq_len(nrow(z)) == marker,'1*','NA*'),z$rhs),collapse=' + '))
  }, '')
  magmaanlab::model_spec(paste(c(lines,design$structural),collapse='\n'))
}

# Generalizes the original extent heuristic; this is not an admissibility test.
expanded_extent <- function(pt, sigma) {
  lv <- unique(pt$lhs[pt$op == '=~']); ov <- unique(pt$rhs[pt$op == '=~']); k <- length(lv)
  B <- Psi <- matrix(0,k,k,dimnames=list(lv,lv))
  for(i in which(pt$op == '~')) B[pt$lhs[i],pt$rhs[i]] <- pt$est[i]
  for(i in which(pt$op == '~~' & pt$lhs %in% lv)) {
    Psi[pt$lhs[i],pt$rhs[i]] <- pt$est[i]; Psi[pt$rhs[i],pt$lhs[i]] <- pt$est[i]
  }
  A <- solve(diag(k)-B); Phi <- A %*% Psi %*% t(A)
  v <- setNames(abs(diag(Phi)),lv); obs <- setNames(abs(diag(sigma)),ov)
  l <- pt[pt$op == '=~',]; th <- pt[pt$op == '~~' & pt$lhs %in% ov,]
  values <- c(abs(l$est)*sqrt(v[l$lhs]/obs[l$rhs]),
    abs(th$est)/sqrt(obs[th$lhs]*obs[th$rhs]),
    abs(B)*sqrt(outer(1/v,v)), abs(Phi)/sqrt(outer(v,v)), abs(diag(Psi))/v)
  if(any(!is.finite(values))) Inf else max(values)
}
