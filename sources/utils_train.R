library(mgcv)
library(xgboost)
library(foreach)
source('utils_DNN.R')
source('utils_GAMs.R')



# trains CF model on test data
closure_train_CF <- function(num.proc, dataset, ind.feat, feat.names, target, verbose=FALSE) {
  
  set.seed(seed=num.proc*1e4 + round(as.numeric(as.POSIXlt(Sys.time()))))
  
  if (!exists('param.optim')) {
    param.optim <- list(
      # sigma initialization
      sigma.init=1e0,
      
      # NN weights update
      batch.size=64, # batch size for NN weights update
      wd=1e-4, # weight decay (L2 regularization)
      lr=1e-3, # learning rate
      lrd=1e-7, # learning rate decay
      nIter=1e1, # number of NN weights update iterations
      
      # layers parameters
      nbh=c(NA, 64, 32), # layer sizes
      funLayers=c(NA, 'ReLU', 'ReLU'),
      init.type='Glorot',
      
      # sigma update
      batch.size.sig=64, # batch size for sigma update
      wd.sig=1e-5,
      lr.sig=1e-3,
      lrd.sig=1e-7,
      nIter.sig=1e1,
      
      # alternate optimization
      sanity.train=3,
      nIter.optim=15, # number of alternate optimization (NN weights / sigma) iterations
      sigma.update=TRUE
    )
  }
  
  obj_train <- format_CF(obj.split$ind.train, obj.split$ind.test, ind.feat=ind.feat)
  mat.train <<- obj_train$mat.train; mat.test <<- obj_train$mat.test; obj.train <<- obj_train$obj.train; obj.test <<- obj_train$obj.test
  N.train <- nrow(mat.train)
  
  # load architecture and init weights
  nbh <- param.optim$nbh; nbh[1] <- sum(ind.feat)*seq.len # input layer size
  net <- initWeights(nbh, init.type=param.optim$init.type)
  funLayers <- param.optim$funLayers
  sigma <- param.optim$sigma.init
  
  # load optim parameters
  wd <- param.optim$wd # weight decay (L2 regularization)
  lr <- param.optim$lr # learning rate
  lrd <- param.optim$lrd # learning rate decay
  wd.sig <- param.optim$wd.sig
  lr.sig <- param.optim$lr.sig
  lrd.sig <- param.optim$lrd.sig
  
  # adam optim structures
  state_t <- 0
  state_m <- rep(0, net$Nparams)
  state_v <- rep(0, net$Nparams)
  params <- param2vector(net$W, net$b)
  
  state_t.sig <- 0
  state_m.sig <- c(0)
  state_v.sig <- c(0)
  
  if (verbose) {
    predictTest.simple(net, funLayers, sigma, pct=NA, measure='gaussian', verbose=verbose)
  }
  
  for (iter.optim in 1:param.optim$nIter.optim) {
    
    if (verbose) {
      print(' ')
      print(paste0('Global iter ', iter.optim, ' : '))
      print(' ')
      
      print("Iter NN ...")
    }
    
    for (n in 1:param.optim$nIter) {
      
      ind.batch <- sample.int(N.train, size=param.optim$batch.size, replace=FALSE)
      grad.params <- SGD(ind.batch, net, funLayers, obj_train$index.train, param.optim$sanity.train, sigma, pct=NA, measure='gaussian')
      
      # optim
      obj_optim <- adam(params, grad.params, wd, lr, lrd, state_t, state_m, state_v)
      params <- params - obj_optim$dx
      
      obj.param <- vector2param(params, nbh)
      net$W <- obj.param$W
      net$b <- obj.param$b
      
      state_t <- obj_optim$state_t
      state_m <- obj_optim$state_m
      state_v <- obj_optim$state_v
      if (verbose) print(paste0('grad iter ', n))
      
    }
    
    # optimize sigma
    if (param.optim$sigma.update) {
      if (verbose) print("Iter Sigma ...")
      for (n in 1:param.optim$nIter.sig) {
        
        ind.batch <- sample.int(N.train, size=param.optim$batch.size.sig, replace=FALSE)
        grad.sigma <- SGD.sigma(ind.batch, net, funLayers, obj_train$index.train, param.optim$sanity.train, sigma, pct=NA, measure='gaussian')
        
        # optim
        obj_optim <- adam(sigma, grad.sigma, wd.sig, lr.sig, lrd.sig, state_t.sig, state_m.sig, state_v.sig)
        sigma <- sigma - obj_optim$dx[1, 1]
        
        state_t.sig <- obj_optim$state_t
        state_m.sig <- obj_optim$state_m
        state_v.sig <- obj_optim$state_v
        if (verbose) print(paste0('grad iter ', n))
        
      }
    }
    
    if (verbose) {
      print(paste0("sigma value  = ", sigma))
      predictTest.simple(net, funLayers, sigma, pct=NA, measure='gaussian', verbose=verbose)
    }
    
  }
  
  LOSS.test <- predictTest.simple(net, funLayers, sigma, pct=NA, measure='gaussian', verbose=verbose) # LOSS on test data
  
  # saves model and other info
  model <- list()
  model$net <- net
  model$nbh <- nbh
  model$funLayers <- funLayers
  model$sigma <- sigma
  model$ind.feat <- ind.feat
  model$target.max <- obj_train$target.max
  model$LOSS.test <- LOSS.test
  saveRDS(file=paste0(loc.models, '/', genModelName('model')), model)
  
  return(LOSS.test)
  
}



# trains XGBoost model on test data
closure_train_xgb <- function(num.proc, dataset, ind.feat, feat.names, target, verbose=FALSE) {
  
  set.seed(seed=num.proc*1e4 + round(as.numeric(as.POSIXlt(Sys.time()))))
  
  # data formatting
  ind.train.seq <- rep((obj.split$ind.train-1)*seq.len, each=seq.len) + (1:seq.len)
  ind.test.seq <- rep((obj.split$ind.test-1)*seq.len, each=seq.len) + (1:seq.len)
  
  mat.train <- featmat[ind.train.seq, c(ind.feat, FALSE)]
  obj.train <- featmat[ind.train.seq, fs+1]
  target.max <- max(obj.train)
  obj.train <- obj.train/target.max
  
  mat.test <- featmat[ind.test.seq, c(ind.feat, FALSE)]
  obj.test <- featmat[ind.test.seq, fs+1]
  obj.test <- obj.test/target.max
  
  # xgboost training
  if (!exists('param.optim')) {
    param.optim <- list(
      objective = "reg:linear",
      eval_metric="mae",
      eta = 0.05,
      max_depth = 8, # 10
      colsample_bytree = 0.9,
      subsample = 0.9,
      min_child_weight = 4,
      lambda = 1,
      maximize=FALSE,
      nrounds=5e2, 
      verbosity=1
    )
  }
  
  dtrain <- xgb.DMatrix(data=mat.train, label=obj.train, missing=NA)
  model.xgb <- xgb.train(param.optim, dtrain, nrounds=param.optim$nrounds, watchlist=list(train=dtrain), early_stopping_round=3)
  
  # testing
  dtest <- xgb.DMatrix(data=mat.test, label=obj.test, missing=NA)
  pred <- predict(model.xgb, dtest)
  LOSS.test <- sum(abs(pred-obj.test))/length(obj.split$ind.test)
  
  # saves model and other info
  model <- list(xgb_model=model.xgb, ind.feat=ind.feat, target.max=target.max, LOSS.test=LOSS.test)
  saveRDS(file=paste0(loc.models, '/', genModelName('model')), object=model)
  
  return(LOSS.test)
  
}



# trains GAM model on test data
closure_train_GAM <- function(num.proc, dataset, ind.feat, feat.names, target, verbose=FALSE) {
  
  set.seed(seed=num.proc*1e4 + round(as.numeric(as.POSIXlt(Sys.time()))))
  
  # data formatting
  ind.train.seq <- rep((obj.split$ind.train-1)*seq.len, each=seq.len) + (1:seq.len)
  ind.test.seq <- rep((obj.split$ind.test-1)*seq.len, each=seq.len) + (1:seq.len)
  
  feat.data.train <- as.data.frame(featmat[ind.train.seq, ])
  feat.data.test <- as.data.frame(featmat[ind.test.seq, ])

  obj.GAM <- generateFormula(dataset, ind.feat, feat.names, feat.data.train)
  
  # training GAM model
  model.gam <- gam(obj.GAM$formula, data=feat.data.train, family = gaussian)
  gam.forecast <- pmax(0, as.vector(predict(model.gam, newdata=feat.data.test)))
  LOSS.test <- sum(abs(feat.data.test[[target]]-gam.forecast))/N.test
  
  ind.feat[!obj.GAM$ind.keep] <- FALSE
  model <- list(model=model.gam, ind.feat=ind.feat, target.max=1, LOSS.test=LOSS.test)
  saveRDS(file=paste0(loc.models, '/', genModelName('model')), object=model)
  
  return(LOSS.test)
  
}



train_fun <- list(CF=closure_train_CF, xgb=closure_train_xgb, GAM=closure_train_GAM)


