library(xgboost)
library(mgcv)
library(modelselect)
library(opera)
library(tsensembler2)
library(fastmatch)
source('../utils_generic.R')
loc.models <- './models_DETS/'
loc.res <- './results_DETS/'

dataset.run <- 'CO2' # dataset
nb.models <- 64 # number of models to select from
N.run <- 8 # number of test runs

# load data
load(paste0('../data_', dataset.run, '.RData'))
data.array <- array(t(featmat), dim=c(fs+1, seq.len, nrow(featmat)/seq.len))
obj.split <- formSplit(sanity=7)
obj.test <- getObj(obj.split$ind.test, target.max=1)
N.test <-nrow(obj.test)



########################################################################################################################
# R-opera baselines
baselines.op <- c('BOA', 'FS', 'MLpol', 'MLewa', 'EWA', 'OGD', 'Ridge', 'MLprod')
baselines.loss <- c('absolute', 'absolute', 'absolute', 'absolute', 'absolute', 'absolute', 'square', 'absolute'); names(baselines.loss) <- baselines.op
baselines.op <- c('EWA', 'MLewa', 'MLpol')
baselines.loss <- baselines.loss[baselines.op]


# load meta model
ind.seq <- rep((obj.split$ind.test-1)*seq.len, each=seq.len) + rep(1:seq.len, length(obj.split$ind.test))
test <- featmat[ind.seq, ]
test <- as.data.frame(cbind(test[, fs+1, drop=FALSE], test[, 1:fs]))


# run tests : retrieve recorded tsensembler test results, and perform comparison with MaDyMos and opera baselines
PLOT <- FALSE
res.run <- list()
for (idrun in 1:N.run) {
  
  DETS.model <- readRDS(file=paste0(loc.models, "/DETS_model_", dataset.run, '_', nb.models, "_r", idrun, ".rds"))
  
  
  # allocate structures to store results
  array.pred.ts <- array(NA, c(N.test, seq.len, seq.len))
  array.pred.ms <- array(NA, c(N.test, seq.len, seq.len))
  lst.array.pred.op <- list()
  for (base in baselines.op) lst.array.pred.op[[base]] <- array(NA, c(N.test, seq.len, seq.len))
  
  
  # madymos
  model.names <- names(DETS.model$model@base_ensemble@base_models)
  perm.models <- fmatch(model.names, rownames(DETS.model$mat.feat))
  obj.madymos <- madymos(nb.models, seq.len, DETS.model$mat.feat[perm.models, ], K=2, mu.LBI=0.8, gamma.DP=0.95, lambda=1e-1, thresh.switch.off=0.01, 
                         lambda.mu=0.9, pct.fst=2/3)
  
  
  pb <- txtProgressBar()
  for (n in 1:N.test) {
    
    preds.n <- as.matrix(predict(DETS.model$model@base_ensemble, test[((n-1)*seq.len+1):(n*seq.len), ], use_all=TRUE))
    
    # tsensembler
    array.pred.ts[n, , ] <- readRDS(paste0(loc.res, "/res_", dataset.run, '_', nb.models, '_r', idrun, '_d', n, '.rds'))
    
    # madymos
    obj.select.n <- runModelSelection(pred=preds.n, obs=obj.test[n, ], online=FALSE, compute.LBI=FALSE)
    array.pred.ms[n, , ] <- obj.select.n$mat_pred
    
    # opera
    lst.model.op <- list()
    for (base in baselines.op) lst.model.op[[base]] <- mixture(model = base, loss.type = baselines.loss[base])
    for (tt in 1:seq.len) {
      for (base in baselines.op) {
        lst.model.op[[base]] <- predict(lst.model.op[[base]], newexperts=matrix(preds.n[tt, ], nrow=1), newY = obj.test[n, tt], online=TRUE, type="model")
        lst.array.pred.op[[base]][n, tt:seq.len, tt] <- predict(lst.model.op[[base]], newexperts=preds.n[tt:seq.len, , drop=FALSE], online=FALSE, type="response")
      }
    }
    
    setTxtProgressBar(pb, n/N.test)
    
  }
  close(pb)
  
  
  # score results
  cae.ts <- scoreRun(array.pred.ts, obj.test)
  cae.ms <- scoreRun(array.pred.ms, obj.test)
  cae.op <- matrix(NA, length(baselines.op), seq.len); rownames(cae.op) <- baselines.op
  for (base in baselines.op)  cae.op[base, ] <- scoreRun(lst.array.pred.op[[base]], obj.test)
  
  
  # plot results
  if (PLOT) {
    print(paste0('Aggr. score ts = ',  sum(cae.ts, na.rm=TRUE)))
    print(paste0('Aggr. score ms = ',  sum(cae.ms, na.rm=TRUE)))
    N.methods <- 2 + length(baselines.op)
    colors <- glasbey(); N.colors <- length(colors)
    plot_curves(
      subS=subS,
      step.plot=1,
      curves=cbind(cae.ts, cae.ms, t(cae.op)),
      colors= colors[(0:(N.methods-1))%%N.colors + 1],
      legend=c('ts', 'ms', rownames(cae.op)),
      x.name='time of day', y.name='cumulative AE'
    )
  }
  
  # format res
  res.run[[idrun]] <- list(
    cae.ts=cae.ts,
    cae.ms=cae.ms,
    cae.op=cae.op
  )
  
}


# save results
loc.results <- './resMultirun/'
dir.create(file.path('./', loc.results), showWarnings=FALSE)
saveRDS(file=paste0(loc.results, '/results_', dataset.run, '_', nb.models, '.rds'), res.run)





