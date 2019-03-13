library(opera)
library(modelselect)
source('utils_generic.R')
# set.seed(seed = 1e3)



computeR.debug <- function(start.t, end.t, E.obs, R.val, mu.LBI) {
  
  E.max <- apply(E.obs, MARGIN=1, FUN=max)
  rwd <- t(E.max-E.obs)
  for (t in start.t:end.t) {
    R.val[, t:end.t] <- R.val[, t:end.t]*(1 + rwd[, t, drop=FALSE]%*%mu.LBI[1:(end.t-t+1)])
  }
  
  return(R.val)
  
}



modelSelection_simple <- function(start.t, end.t, pred, E.obs, R.val, mu.LBI) {
  
  E.max <- apply(E.obs, MARGIN=1, FUN=max)
  rwd <- t(E.max-E.obs)
  
  mat.pred <- matrix(NA, seq.len, seq.len)
  for (t in start.t:end.t) {
    R.val[, t:end.t] <- R.val[, t:end.t]*(1 + rwd[, t, drop=FALSE]%*%mu.LBI[1:(end.t-t+1)])
    ind.max.t <- apply(R.val[, t:end.t, drop=FALSE], MARGIN=c(2), FUN=which.max)
    mat.pred[t:end.t, t] <- pred[(ind.max.t-1)*seq.len + (t:end.t)]
  }
  
  return(list(pred=mat.pred, rwd=R.val))
  
}



########################################################################################################################
loc.pred <- './allpreds/'

model.types <- c('CF', 'xgb', 'GAM')
dataset.names <- c('bike', 'CO2', 'Irradiance', 'Electricity', 'Traffic')
dataset.run <- 'bike'

num.modelPerDataset <- cntModelsPerDataset(model.types, dataset.names)



########################################################################################################################
load(paste0('data_', dataset.run, '.RData'))
rm(featmat); gc()

nb.models <- 32 # number of models to select from



########################################################################################################################
# R-opera baselines
baselines.op <- c('BOA', 'FS', 'MLpol', 'MLewa', 'EWA', 'OGD', 'Ridge')
baselines.loss <- c('absolute', 'absolute', 'absolute', 'absolute', 'absolute', 'absolute', 'square')
ind.base <- 6

coeffs.aggr <- rep(1/nb.models, nb.models) # simple aggregate rule



########################################################################################################################
N.run <- 2



library(doMC)
registerDoMC(min(N.run, max(1, detectCores()-1))) # number of CPU cores
res.run <- foreach(n = 1:N.run) %do% ({
  
  print(paste0('Performing run ', n, ' ...'))
  
  # samples models to use for model selection
  obj.sampleModels <- sampleAndLoadPred(dataset.run, nb.models, num.modelPerDataset, loc.pred)
  E.per.model <- obj.sampleModels$E.per.model
  names(E.per.model) <- model.types[obj.sampleModels$ind.type]
  allpreds <- obj.sampleModels$pred
  obj.test <- obj.sampleModels$obj
  mat.feat <- obj.sampleModels$mat.feat
  ind.best.model <- which.min(E.per.model)
  
  
  #################################################################################################
  ######################################## model selection ########################################
  #################################################################################################
  
  obj.madymos <- madymos(nb.models, seq.len, mat.feat, K=2, mu.LBI=0.8, gamma.DP=0.95, lambda=1e-1, thresh.switch.off=0.1)
  
  # runs model selection
  N.test <- dim(allpreds)[1]
  array.pred.ms <- array(NA, c(N.test, seq.len, seq.len))
  array.pred.aggr <- array(NA, c(N.test, seq.len, seq.len))
  array.pred.bm <- array(NA, c(N.test, seq.len, seq.len))
  array.pred.op <- array(NA, c(N.test, seq.len, seq.len))
  
  pb <- txtProgressBar()
  for (jj in 1:N.test) {
    
    pred.jj <- allpreds[jj, , ]
    obj.select.jj <- runModelSelection(pred=pred.jj, obs=obj.test[jj, ], online=FALSE)
    array.pred.ms[jj, , ] <- obj.select.jj$mat_pred

    # aggregate model, opera
    model.op <- mixture(model = baselines.op[ind.base], loss.type = baselines.loss[ind.base])
    for (tt in 1:seq.len) {
      # aggregate
      array.pred.aggr[jj,tt:seq.len,tt] <- pred.jj[tt:seq.len, , drop=FALSE]%*%coeffs.aggr
      
      # best model
      array.pred.bm[jj,tt:seq.len,tt] <- pred.jj[tt:seq.len, ind.best.model]
      
      # opera
      model.op <- predict(model.op, newexperts=matrix(pred.jj[tt, ], nrow=1), newY = obj.test[jj, tt], online=TRUE, type="model")
      array.pred.op[jj,tt:seq.len,tt] <- predict(model.op, newexperts=pred.jj[tt:seq.len, , drop=FALSE], online=FALSE, type="response")
    }
    
    setTxtProgressBar(pb, jj/N.test)
    
  }
  close(pb)
  
  
  # scores model selection
  cae.ms <- scoreRun(array.pred.ms, obj.test) # madymos
  cae.aggr <- scoreRun(array.pred.aggr, obj.test) # simple aggregation rule
  cae.bm <- scoreRun(array.pred.bm, obj.test) # best model
  cae.op <- scoreRun(array.pred.op, obj.test) # opera baseline "ind.base"
  
  # print(paste0('Aggr. score ms = ',  sum(cae.ms, na.rm=TRUE)))
  # plot_curves(
  #   subS=subS,
  #   step.plot=1,
  #   curves=cbind(cae.ms, cae.aggr, cae.bm, cae.op),
  #   colors=c('skyblue3', 'black', 'pink', 'darkorange'),
  #   legend=c('ms', 'aggr.', paste0('bm.: ', model.types[[obj.sampleModels$ind.type[ind.best.model]]]), 'op.'),
  #   x.name='time of day', y.name='cumulative AE'
  # )
  
  # format res
  obj.run <- list(
    cae.ms=cae.ms,
    cae.aggr=cae.aggr,
    cae.bm=cae.bm,
    cae.op=cae.op, 
    ind.model=obj.sampleModels$ind.model
  )
  
})


loc.results <- './resMultirun/'
dir.create(file.path('./', loc.pred), showWarnings=FALSE)

saveRDS(file=paste0(loc.results, '/results_', dataset.run, '_', nb.models, '.rds'), res.run)



