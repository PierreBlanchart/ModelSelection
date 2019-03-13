# utils_DNN.R
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


adam <- function(x, dfdx, wd, lr, lrd, state_t, state_m, state_v, eps=1e-8, beta1=0.9, beta2=0.999) {

  clr <- lr/(1 + state_t*lrd) # learning rate decay (annealing)
  state_t <- state_t+1

  dfdx <- dfdx + wd*x

  biasCorrection1 <- 1 - beta1^state_t
  biasCorrection2 <- 1 - beta2^state_t
  step_size <- clr*sqrt(biasCorrection2)/biasCorrection1
  OneMinusBeta1 <- 1 - beta1
  OneMinusBeta2 <- 1 - beta2

  # Decay the first and second moment running average coefficient
  state_m <- state_m*beta1 + OneMinusBeta1*dfdx
  state_v <- state_v*beta2 + OneMinusBeta2*(dfdx^2)
  denom <- sqrt(state_v) + eps
  dx <- step_size*(state_m/denom)

  return(list(dx=dx, state_t=state_t, state_m=state_m, state_v=state_v))

}


sigmoid <- function(x) {
  return(1/(1+exp(-x)))
}
sigmoid.deriv <- function(y) {
  return(y*(1-y))
}

ReLU <- function(x) {
  x[x<0] <- 0
  return(x)
}
ReLU.deriv <- function(y) {
  y[y<0] <- 0
  y[y>0] <- 1
  return(y)
}

idFun <- function(x) {
  return(x)
}
idFun.deriv <- function(x) {
  return(1)
}

transfer <- list(sigmoid=sigmoid, ReLU=ReLU, idFun=idFun)
transfer.deriv <- list(sigmoid=sigmoid.deriv, ReLU=ReLU.deriv, idFun=idFun.deriv)


initWeights <- function(nbh, init.type='Uniform', init.param=1) {

  N.layers <- length(nbh)

  W <- list()
  b <- list()
  Nparams <- 0
  for (l in 2:N.layers) {

    switch(init.type,

           Uniform={
             # random uniform weights
             W[[l]] <- init.param*(2*matrix(runif(nbh[l]*nbh[l-1]), nbh[l]) - 1)
           },

           Xavier={
             # Xavier initialization
             W[[l]] <- matrix(sqrt(1/nbh[l-1])*rnorm(nbh[l]*nbh[l-1]), nrow=nbh[l])
           },

           Glorot={
             # Glorot and Bengio initialization
             W[[l]] <- matrix(sqrt(2/(nbh[l-1]+nbh[l]))*rnorm(nbh[l]*nbh[l-1]), nrow=nbh[l])
           },

           ReLU={
             # ReLU initialization
             W[[l]] <- matrix(sqrt(2/nbh[l-1])*rnorm(nbh[l]*nbh[l-1]), nrow=nbh[l])
           },

           {
             # default
             stop('Unknown initialization type : implemented types are Uniform, Xavier, Glorot and ReLU !')
           }

    )

    # bias initialization
    b[[l]] <- rep(0, nbh[l])

    # updates number of parameters
    nb.n <- nrow(W[[l]])*ncol(W[[l]]) + nrow(W[[l]])
    Nparams <- Nparams + nb.n

  }

  return(list(W=W, b=b, Nparams=Nparams))

}


param2vector <- function(W, b) {

  N.layers <- length(W)
  N.tot <- 0
  nb.n <- rep(NA, N.layers)
  for (n in 2:N.layers) {
    nb.n[n] <- nrow(W[[n]])*ncol(W[[n]]) + nrow(W[[n]])
    N.tot <- N.tot + nb.n[n]
  }

  params <- rep(0, N.tot)
  ind.n <- 1
  for (n in 2:N.layers) {
    params[ind.n:(ind.n+nb.n[n]-1)] <- c(W[[n]][,], b[[n]])
    ind.n <- ind.n+nb.n[n]
  }

  return(params)

}


vector2param <- function(params, nbh) {

  N.layers <- length(nbh)

  W <- list()
  b <- list()
  ind.n <- 1
  for (n in 2:N.layers) {
    nb.n <- nbh[n-1]*nbh[n]
    W[[n]] <- matrix(params[ind.n:(ind.n+nb.n-1)], nbh[n], nbh[n-1])
    ind.n <- ind.n+nb.n
    b[[n]] <- params[ind.n:(ind.n+nbh[n]-1)]
    ind.n <- ind.n+nbh[n]
  }

  return(list(W=W, b=b))

}


forward.NN <- function(X, W, b, Z=NULL, fun=NULL) {

  N.layers <- length(W)
  if (is.null(fun)) fun <- rep('sigmoid', N.layers)

  A <- list()
  A[[1]] <- X
  for (n in 2:N.layers) {
    A[[n]] <- transfer[[fun[n]]](W[[n]]%*%A[[n-1]] + b[[n]])
  }

  if (!is.null(Z)) {
    err <- sum((A[[N.layers]]-Z)^2)
  } else {
    err <- NA
  }

  return(list(err=err, A=A))

}


forward.siameese <- function(X1, X2, W, b, fun=NULL, sigma.sim=1, measure='gaussian') {

  N.layers <- length(W)
  if (is.null(fun)) fun <- rep('sigmoid', N.layers)

  pred1 <- forward.NN(X1, W, b, Z=NULL, fun=fun)
  pred2 <- forward.NN(X2, W, b, Z=NULL, fun=fun)

  if (measure == 'gaussian') {

    d12 <- as.numeric(colSums((pred1$A[[N.layers]]-pred2$A[[N.layers]])^2))
    s12 <- exp(-d12/sigma.sim)
    return(list(A1=pred1$A, A2=pred2$A, s12=s12, d12=d12))

  } else {
    if (measure == 'cosine') {

      out.sz <- nrow(pred1$A[[N.layers]])
      dot.prod <- rep(1, out.sz)%*%(pred1$A[[N.layers]]*pred2$A[[N.layers]])
      norm1.sq <- rep(1, out.sz)%*%(pred1$A[[N.layers]]^2)
      norm2.sq <- rep(1, out.sz)%*%(pred2$A[[N.layers]]^2)
      norm1 <- as.numeric(sqrt(norm1.sq))
      norm2 <- as.numeric(sqrt(norm2.sq))
      s12 <- (as.numeric(dot.prod/(norm1*norm2)) + 1)/2 # cosine similarity is between -1 and 1: so we apply affine transform (x+1)/2 to be between 0 and 1
      return(list(A1=pred1$A, A2=pred2$A, s12=s12, norm1=norm1, norm2=norm2))

    } else {
      stop("Unknown similarity measure")
    }
  }

}


backward.NN <- function(X, W, b, A, Z=NULL, fun=NULL, bm=NULL) {

  N.layers <- length(W)
  if (is.null(fun)) fun <- rep('sigmoid', N.layers)

  batch_size <- ncol(A[[1]])

  grad.W <- list()
  grad.b <- list()

  if (!is.null(Z)) {
    bm.prev <- 2*(A[[N.layers]]-Z)
  } else { # if Z is null, backward message should be provided
    bm.prev <- bm
  }

  for (n in N.layers:2) {
    bm.n <- bm.prev*transfer.deriv[[fun[n]]](A[[n]])
    grad.W[[n]] <- bm.n%*%t(A[[n-1]])
    grad.b[[n]] <- as.numeric(bm.n%*%rep(1, batch_size))
    bm.prev <- t(W[[n]])%*%bm.n
  }

  return(list(grad.W=grad.W, grad.b=grad.b, bm=bm.prev))

}


backward.siameese <- function(X1, X2, W, b, obj.fwd, bmLoss=1, fun=NULL, sigma.sim=1, measure='gaussian') {

  N.layers <- length(W)
  if (is.null(fun)) fun <- rep('sigmoid', N.layers)

  out.sz <- nrow(obj.fwd$A1[[N.layers]])

  if (measure=='gaussian') {

    bm1 <- matrix(rep((bmLoss*obj.fwd$s12), each=out.sz), nrow=out.sz)*((-2/sigma.sim)*(obj.fwd$A1[[N.layers]]-obj.fwd$A2[[N.layers]]))
    bm2 <- -bm1

  } else {

    if (measure=='cosine') {
      prod12 <- obj.fwd$norm1*obj.fwd$norm2
      A1 <- t(obj.fwd$A1[[N.layers]])
      A2 <- t(obj.fwd$A2[[N.layers]])
      bm1 <- (A2/prod12) - (obj.fwd$s12/obj.fwd$norm1^2)*A1
      bm2 <- (A1/prod12) - (obj.fwd$s12/obj.fwd$norm2^2)*A2
      bm1 <- t(as.numeric(bmLoss)*bm1)/2 # we divide by 2 due to affine transformation used to keep cosine similarity between 0 and 1
      bm2 <- t(as.numeric(bmLoss)*bm2)/2

    } else {
      stop('Unknown similarity measure')
    }

  }

  obj.grad1 <- backward.NN(X1, W, b, obj.fwd$A1, Z=NULL, fun=fun, bm=bm1)
  obj.grad2 <- backward.NN(X2, W, b, obj.fwd$A2, Z=NULL, fun=fun, bm=bm2)

  grad.W <- sapply(1:N.layers,
                   function(i) { if (i==1) {return(NULL)} else {return(obj.grad1$grad.W[[i]]+obj.grad2$grad.W[[i]])} },
                   simplify=FALSE)
  grad.b <- sapply(1:N.layers,
                   function(i) { if (i==1) {return(NULL)} else {return(obj.grad1$grad.b[[i]]+obj.grad2$grad.b[[i]])} },
                   simplify=FALSE)

  return(list(grad.W=grad.W, grad.b=grad.b))

}


predict.simple <- function(ind.train, mat.pred, obj.pred, net, funLayers, sigma, pct=0, measure='gaussian') {

  nb.train <- sum(ind.train)
  net.pred <- forward.siameese(t(mat.train[ind.train, ]), t(matrix(rep(mat.pred, each=nb.train), nrow=nb.train)),
                               net$W, net$b, fun=funLayers, sigma.sim=sigma, measure=measure)
  sim.train <- net.pred$s12

  # obj.sorted <- sort.int(sim.train, decreasing = TRUE, index.return=TRUE)
  # max.sim <- obj.sorted$x[1]
  # ind.lst <- match(TRUE, obj.sorted$x < pct*max.sim)
  # if (is.na(ind.lst)) ind.lst <- nb.train+1
  # ind.sim <- obj.sorted$ix[1:(ind.lst-1)]
  # ind.sim.train <- which(ind.train)[ind.sim]
  #
  # sim.train <- sim.train[ind.sim]
  # sim.sum <- sum(sim.train)
  # pred <- as.numeric(sim.train%*%obj.train[ind.sim.train, ])/sim.sum

  sim.sum <- sum(sim.train)
  pred <- as.numeric(sim.train%*%obj.train[ind.train, ])/sim.sum
  ind.sim.train <- NA

  LOSS.MAE <- sum(abs(pred-obj.pred))

  return(list(obj.fwd=net.pred, pred=pred, LOSS=LOSS.MAE, ind.sim.train=ind.sim.train, sim.sum=sim.sum))

}


predictTest.simple <- function(net, funLayers, sigma, pct=0, measure='gaussian', plot.results=FALSE, return.pred=FALSE, sim.return=FALSE, verbose=FALSE) {

  LOSS.test <- 0

  # pb <- txtProgressBar(min=0, max=1)
  seq.len <- ncol(obj.test)
  mat.predicted <- matrix(NA, nrow(obj.test), seq.len)
  mat.sim <- matrix(NA, nrow(mat.train), nrow(obj.test))
  for (n in 1:nrow(mat.test)) {
    ind.train.n <- rep(TRUE, nrow(mat.train))
    obj.n <- predict.simple(ind.train.n, mat.test[n, ], obj.test[n, ], net, funLayers, sigma, pct=pct, measure=measure)
    mat.predicted[n, ] <- obj.n$pred
    mat.sim[ind.train.n, n] <- obj.n$obj.fwd$s12
    LOSS.test <- LOSS.test+obj.n$LOSS
    # setTxtProgressBar(pb, n/nrow(mat.test))
  }
  # close(pb)

  LOSS.test <- LOSS.test/nrow(mat.test)
  if (verbose) print(paste0('LOSS.test = ', LOSS.test))

  if (plot.results) {
    labels.plot <- rep(NA, nrow(obj.test)*seq.len)
    preds.plot <- rep(NA, nrow(obj.test)*seq.len)
    ind.n <- 1
    for (n in 1:nrow(obj.test)) {
      labels.plot[ind.n:(ind.n+seq.len-1)] <- obj.test[n, ]
      preds.plot[ind.n:(ind.n+seq.len-1)] <- mat.predicted[n, ]
      ind.n <- ind.n + seq.len
    }

    plot(labels.plot, type='l', lwd=2, xlim=c(0, ind.n),
         ylim=c(min(c(labels.plot, preds.plot)), max(c(labels.plot, preds.plot))), col='darkorange')
    lines(preds.plot, type='l', lwd=2, col='cyan')
  }

  if (return.pred) {
    obj.predict <- list()
    obj.predict$LOSS.test <- LOSS.test
    obj.predict$predictions <- mat.predicted
    if (sim.return) {
      obj.predict$mat.sim <- mat.sim
    }
    return(obj.predict)
  }

  return(LOSS.test)

}


bp.simple <- function(ind.train, mat.pred, obj.pred, pred, sim.sum, obj.fwd, net, funLayers, sigma, measure='gaussian') {

  # nb.train <- length(ind.train)
  nb.train <- sum(ind.train)
  diff <- abs(mat.train[ind.train, ]-matrix(rep(mat.pred, each=nb.train), nrow=nb.train))

  dLoss_dpred <- sign(pred-obj.pred) # MAE Loss
  dLoss_dsi <- as.numeric((dLoss_dpred/sim.sum)%*%(t(obj.train[ind.train, ])-pred)) # 1 x nb.train

  grad.siam <- backward.siameese(t(mat.train[ind.train, ]), t(matrix(rep(mat.pred, each=nb.train), nrow=nb.train)),
                                 net$W, net$b, obj.fwd, bmLoss=dLoss_dsi, fun=funLayers, sigma.sim=sigma, measure=measure)

  return(param2vector(grad.siam$grad.W, grad.siam$grad.b))

}


bp.simple.sigma <- function(ind.train, mat.pred, obj.pred, pred, sim.sum, obj.fwd, net, funLayers, sigma, measure='gaussian') {

  # nb.train <- length(ind.train)
  nb.train <- sum(ind.train)
  diff <- abs(mat.train[ind.train, ]-matrix(rep(mat.pred, each=nb.train), nrow=nb.train))

  dLoss_dpred <- sign(pred-obj.pred) # MAE Loss
  dLoss_dsi <- as.numeric((dLoss_dpred/sim.sum)%*%(t(obj.train[ind.train, ])-pred)) # 1 x nb.train

  # bm1 <- matrix(rep((bmLoss*obj.fwd$s12), each=out.sz), nrow=out.sz)*((-2/sigma.sim)*(obj.fwd$A1[[N.layers]]-obj.fwd$A2[[N.layers]]))
  grad.sigma <- dLoss_dsi%*%((obj.fwd$d12/(sigma^2))*obj.fwd$s12)

  return(grad.sigma)

}


SGD <- function(ind.batch, net, funLayers, index.train, sanity, sigma, pct=0, measure='gaussian') {

  grad.params <- rep(0, net$Nparams)

  ind.n <- 1
  # pb <- txtProgressBar(min=0, max=1)
  for (n in ind.batch) {

    ind.train.n <- (abs(index.train-index.train[n]) >= sanity)
    obj.n <- predict.simple(ind.train.n, mat.train[n, ], obj.train[n, ], net, funLayers, sigma, pct=pct, measure=measure)
    grad.n <- bp.simple(ind.train.n, mat.train[n, ], obj.train[n, ], obj.n$pred, obj.n$sim.sum, obj.n$obj.fwd, net, funLayers, sigma, measure=measure)

    grad.params <- grad.params + grad.n

    # setTxtProgressBar(pb, ind.n/length(ind.batch))
    ind.n <- ind.n + 1

  }
  # close(pb)

  return(grad.params/length(ind.batch))

}


SGD.sigma <- function(ind.batch, net, funLayers, index.train, sanity, sigma, pct=0, measure='gaussian') {

  grad.sigma <- 0

  ind.n <- 1
  # pb <- txtProgressBar(min=0, max=1)
  for (n in ind.batch) {

    ind.train.n <- (abs(index.train-index.train[n]) >= sanity)
    obj.n <- predict.simple(ind.train.n, mat.train[n, ], obj.train[n, ], net, funLayers, sigma, pct=pct, measure=measure)
    grad.n <- bp.simple.sigma(ind.train.n, mat.train[n, ], obj.train[n, ], obj.n$pred, obj.n$sim.sum, obj.n$obj.fwd, net, funLayers, sigma, measure=measure)

    grad.sigma <- grad.sigma + grad.n

    # setTxtProgressBar(pb, ind.n/length(ind.batch))
    ind.n <- ind.n + 1

  }
  # close(pb)

  return(grad.sigma/length(ind.batch))

}


saveModel <- function(net, funLayers, sigma) {

  model <- list()
  model$net <- net
  model$funLayers <- funLayers
  model$sigma <- sigma
  model$mat.train <- mat.train
  model$obj.train <- obj.train

  modelId <- round(as.numeric(as.POSIXct(Sys.time())))
  saveRDS(file=paste0('model_', modelId, '.rds'), model)

  return(modelId)

}

