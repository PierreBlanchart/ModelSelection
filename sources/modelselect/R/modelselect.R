# modelselect.R
#
# Copyright (c) 2018 <pierre.blanchart>
# Author : Pierre Blanchart
#
# This file is part of the "modelselect" package distribution.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
# You should have received a copy of the GNU General Public License
# ---------------------------------------------------------------------------


#' generates all K-permutations of "nb.models" elements
#' @export
genKperm <- function(nb.models, K) {
  
  div.factor <- nb.models^(0:(K-1))
  
  nb.Kperm <- nb.models^K
  Kperm <- matrix(NA, nb.Kperm, K)
  residu <- 0:(nb.Kperm-1)
  for (k in seq(K, 1, -1)) {
    Kperm[, k] <- floor(residu/div.factor[k])
    residu <- residu - Kperm[, k]*div.factor[k]
  }
  
  return(Kperm+1)
  
}



#' @export
initTrans <- function(nb.models, K) {
  
  perm <- genKperm(nb.models, K) # nb.perm x K
  nb.perm <- nrow(perm)
  
  # compute possible transitions between u and u+1
  if (K > 1) {
    
    vec.basis <- nb.models^(seq(K-1, 0, -1))
    index.perm <- as.numeric((perm-1)%*%vec.basis)
    mapping <- rep(NA, max(index.perm)+1) # there can be a 0, hence "+1"
    mapping[index.perm+1] <- 1:nb.perm # mapping between sequences and perm index
    
    # for dynamic programming : possible next states
    state.next <- perm[, 2:K, drop=FALSE]
    state.next <- as.numeric((state.next-1)%*%vec.basis[1:(K-1)])
    ind.next.temp <- matrix(rep(state.next, nb.models), nb.perm, nb.models) + matrix(rep(0:(nb.models-1), nb.perm), nb.perm, nb.models, byrow = TRUE)
    ind.next <- apply(ind.next.temp, 2, function(v) mapping[v+1])
    
  } else {
    if (K > 0) {
      ind.next <- matrix(rep(1:nb.models, each=nb.perm), nb.perm)
    } else {
      stop("K has to be strictly positive")
    }
  }
  
  ind.selected <- perm[, K]
  
  obj_Kmodel <- list()
  # obj_Kmodel$vec.basis <- vec.basis
  # obj_Kmodel$mapping <- mapping
  obj_Kmodel$perm <- perm
  obj_Kmodel$ind.next <- ind.next
  obj_Kmodel$ind.selected <- ind.selected
  
  return(obj_Kmodel)
  
}



#' @export
madymos <- function(nb.models, seq.len, mat.feat, K=2, mu.LBI=0.8, gamma.DP=0.95, lambda=1e-1, thresh.switch.off=1e-1,
                    lambda.mu=0.9, pct.fst=2/3, clamp=NULL) {
  
  nb.states <- nb.models^K
  obj_Kmodel <- initTrans(nb.models, K)
  
  if (is.null(clamp)) clamp <- seq.len
  
  # compute used features over [t-K+1, t-1]
  num.feat <- matrix(NA, nb.states, fs)
  for (p in 1:nb.states) {
    num.feat[p, ] <- colSums(mat.feat[obj_Kmodel$perm[p, 1:(K-1)], , drop=FALSE])
  }
  
  # cosine similarity with ind.next models
  cos.sim <- matrix(NA, nb.states, nb.models)
  temp <- rowSums(mat.feat^2)
  for (p in 1:nb.states) {
    cos.sim[p, ] <- (mat.feat %*% num.feat[p, ]) / sqrt( sum(num.feat[p, ]^2) * temp )
  }
  
  # prune useless transitions
  ind.keep <- cos.sim > thresh.switch.off
  nb.keep.per.row <- rowSums(ind.keep)
  N.next <- max(nb.keep.per.row)
  
  ind.next <- matrix(NA, nb.states, nb.models)
  for (p in 1:nb.states) {
    temp.next.p <- obj_Kmodel$ind.next[p, ]
    ind.next[p, ] <- c(temp.next.p[ind.keep[p, ]], temp.next.p[!ind.keep[p, ]])
    cos.sim[p, ] <- -1e8
    cos.sim[p, 1:sum(ind.keep[p, ])] <- 1
  }
  obj_Kmodel$ind.next <- ind.next[, 1:N.next]
  obj_Kmodel$mat.trans <- cos.sim[, 1:N.next]
  
  obj_Kmodel$perm <- as.vector(obj_Kmodel$perm)
  obj_Kmodel$ind.next <- as.vector(obj_Kmodel$ind.next)
  
  return(list(nb.models = nb.models,
              seq.len = seq.len,
              K = K,
              obj_Kmodel = obj_Kmodel,
              nb.states = nb.states,
              mu.LBI = mu.LBI,
              gamma.DP = gamma.DP,
              clamp = clamp,
              LBI.profile = mu.LBI^(0:(seq.len-1)),
              R.cur = matrix(1, nb.models, seq.len),
              lambda = lambda,
              lambda.mu = lambda.mu,
              pct.fst = pct.fst,
              E.estimate = matrix(0, seq.len, nb.models),
              R.estimate = matrix(0, nb.models, seq.len),
              N.E.estimate = 0,
              mat.LBI = matrix(0, seq.len, nb.models),
              cnt.LBI = matrix(0, seq.len, nb.models)
  )
  )
  
}



estimateLBI <- function(pred, obs, pct.fst=2/3) {
  
  nb.models <- ncol(pred)
  seq.len <- nrow(pred)
  error <- abs(pred-obs)
  
  ind.sorted <- t(apply(error, MARGIN=c(1), FUN=function(row) {
    obj.sorted <- sort(row, index.return=TRUE)
    obj.sorted$ix
  }))
  
  for (t in 1:seq.len) {
    
    ind.best.t <- which.min(error[t, ])
    ind.t <- match(TRUE, rowSums(ind.sorted[t:seq.len, 1:max(1, round(pct.fst*nb.models)), drop=FALSE] == ind.best.t) == 0)
    
    if (!is.na(ind.t)) {
      obj.madymos$mat.LBI[t, ind.best.t[1]] <<- obj.madymos$mat.LBI[t, ind.best.t[1]] + (ind.t-1)
      obj.madymos$cnt.LBI[t, ind.best.t[1]] <<- obj.madymos$cnt.LBI[t, ind.best.t[1]] + 1
    }
  }
  
  duration.mean <- sum( obj.madymos$mat.LBI/sum(obj.madymos$cnt.LBI, na.rm=TRUE) , na.rm=TRUE )
  
  return(1 - 1/duration.mean)
  
}



#' @export
updateStrategy <- function(newPred, newObs) {
  
  nb.models <- obj.madymos$nb.models
  seq.len <- obj.madymos$seq.len
  nb.newPred <- nrow(newPred)
  temp.pred <- rbind(newPred, matrix(NA, seq.len-nb.newPred, nb.models))
  
  obj.pred <- modelSelection_trans_mat(mat_pred=temp.pred, newObs, obj.madymos$R.cur, seq.len,
                                       obj.madymos$K, obj.madymos$obj_Kmodel, obj.madymos$gamma.DP,
                                       obj.madymos$LBI.profile, clamp=obj.madymos$clamp)
  
  return(list(
    mat_pred=obj.pred$mat_pred[1:nb.newPred, , drop=FALSE],
    mat_index=obj.pred$mat_index[1:nb.newPred, , drop=FALSE])
  )
  
}



#' @export
#' simulates a run over a whole period
#' outputs seq.len successive model selection strategies with corresponding predictions
runModelSelection <- function(pred, obs, online=FALSE, compute.LBI=FALSE) {
  
  nb.models <- ncol(pred)
  seq.len <- nrow(pred)
  error <- abs(pred-obs)
  
  if (online) {
    R.cur <- 1 + obj.madymos$R.estimate
  } else {
    R.cur <- matrix(1, nb.models, seq.len)
  }
  obj.pred <- modelSelection_trans_mat(mat_pred=pred, obs, R.cur, seq.len,
                                       obj.madymos$K, obj.madymos$obj_Kmodel, obj.madymos$gamma.DP,
                                       obj.madymos$LBI.profile, clamp=obj.madymos$clamp)
  
  obj.madymos$E.estimate <<- (obj.madymos$E.estimate*obj.madymos$N.E.estimate + error)/(obj.madymos$N.E.estimate + 1)
  obj.madymos$N.E.estimate <<- obj.madymos$N.E.estimate + 1
  if (online) {
    obj.madymos$R.estimate <<- matrix(obj.madymos$lambda, nb.models, seq.len)
    computeR(obj.madymos$E.estimate, obj.madymos$R.estimate, obj.madymos$LBI.profile, clamp=obj.madymos$clamp)
  }
  
  if (compute.LBI) {
    mu.LBI.update <- estimateLBI(pred, obs, obj.madymos$pct.fst)
    obj.madymos$mu.LBI <<- obj.madymos$lambda.mu*obj.madymos$mu.LBI + (1-obj.madymos$lambda.mu)*mu.LBI.update
    obj.madymos$LBI.profile <<- obj.madymos$mu.LBI^(0:(seq.len-1))
  }
  
  return(obj.pred)
  
}




###########################################################################################################################################
###################################################### plot functions #####################################################################
###########################################################################################################################################

#' generic function for plotting several curves on the same graph
#' @param curves : seq_len x nb.curves matrix. Each column of "curves" contains a "seq_len" vector to plot.
#' @param subS : subsampling factor associated with results in "curves" : "subS=30" means that values in "curves" are the result of a 30mn-granularity sampling
#' @param step.plot : subsampling factor in hours applied to the x-labels for clarity of display : "step.plot=1" means one label displayed every hour on the x-axis
#' @export
plot_curves <- function(subS=30, step.plot=1, curves, lwd=NULL, colors=NULL, legend=NULL, x.name=NA, y.name=NA) {
  
  if (is.vector(curves)) seq_len <- length(curves) else seq_len <- nrow(curves)
  if (is.vector(curves)) nb.curves <- 1 else nb.curves <- ncol(curves)
  x <- seq(0, 24, length.out = seq_len) # assuming 24H periodicity
  
  if (is.null(colors) || length(colors) < nb.curves) {
    col.choice <- c('green', 'blue', 'red', 'black', 'magenta', 'cyan', 'orange')
    colors <- col.choice[sample(length(col.choice), nb.curves, replace=TRUE)]
  }
  if (is.null(lwd)) lwd = 1
  
  if (is.vector(curves)) curve1 <- curves else curve1 <- curves[, 1]
  plot(x, curve1, type='l', col=colors[1], lwd=lwd, xlim=c(0, 24), ylim=c(min(curves, na.rm=TRUE), max(curves, na.rm=TRUE)), xaxt="n", cex.axis=1.5, xlab=x.name, ylab=y.name, cex.lab=1.5)
  pos.labels <- unique(round(seq(0, 24, by=step.plot)))
  axis(1, at=pos.labels, tck=1, lty="dotted", col="lightgray", labels=paste0(pos.labels, 'H'), lwd=1, cex.axis=1.5)
  # Adding horizontal grid
  grid(NA, NULL, col = "lightgray", lty = "dotted", lwd = 1)
  
  if (nb.curves > 1) {
    for (i in 2:nb.curves) {
      lines(x, curves[, i], col=colors[i], lwd=lwd)
    }
  }
  
  # Adding legend
  if (!is.null(legend)) legend("topright", inset=-1e-2, legend=legend, col=colors, lty=rep(1, nb.curves), cex=1.1, lwd=3)
  
  box()
}



#' plots a 2D matrix as heatmap
#' @export
plotMatrix <- function(a, labels.rows=NULL, pos.rows=NULL, labels.cols=NULL, pos.cols=NULL, colormap=NULL, labels.rows.centered=FALSE, labels.cols.centered= FALSE) {
  
  nb.rows <- nrow(a)
  nb.cols <- ncol(a)
  if (is.null(labels.rows)) labels.rows <- rownames(a)
  if (is.null(labels.cols)) labels.cols <- colnames(a)
  if (is.null(colormap)) colormap <- heat.colors(64)
  
  par(las=1, mar=c(5,6,3,1))
  image(t(a[nb.rows:1, ]), yaxt='n', xaxt='n', col=colormap)
  
  if (is.null(pos.cols)) {
    if (!labels.cols.centered) {
      pos.cols <- seq(-(1/(2*(nb.cols-1))), 1+1/(2*(nb.cols-1)), length.out=nb.cols)
    } else {
      pos.cols <- seq(0, 1, length.out=nb.cols)
    }
  } else { # re-spread between -(1/(2*(nb.cols-1))) and 1+1/(2*(nb.cols-1)), since pos.cols are provided between 0 and 1
    if (!labels.cols.centered) pos.cols <- -(1/(2*(nb.cols-1))) + (pos.cols*(1 + (1/(nb.cols-1))))
  }
  axis(3, at=pos.cols, labels=labels.cols, srt=5, cex.axis=1.3, tick=FALSE)
  
  if (is.null(pos.rows)) {
    if (!labels.rows.centered) {
      pos.rows <- seq(-(1/(2*(nb.rows-1))), 1+1/(2*(nb.rows-1)), length.out=nb.rows)
    } else {
      pos.rows <- seq(0, 1, length.out=nb.rows)
    }
  } else {
    if (!labels.rows.centered) pos.rows <- -(1/(2*(nb.rows-1))) + (pos.rows*(1 + (1/(nb.rows-1))))
  }
  axis(2, at=1-pos.rows, labels=labels.rows, srt=5, cex.axis=1.3, tick=FALSE)
  
  w <- 1/(nb.cols-1)
  axis(3, at=seq(0, nb.cols*w, length.out=nb.cols+1) - w/2, tck=1, lty="solid", col="black", lwd=3, labels=FALSE)
  w <- 1/(nb.rows-1)
  axis(2, at=seq(0, nb.rows*w, length.out=nb.rows+1) - w/2, tck=1, lty="solid", col="black", lwd=3, labels=FALSE)
  box()
  
}


