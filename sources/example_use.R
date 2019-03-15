library(modelselect)
source('utils_generic.R')
source('utils_train.R')
source('utils_predict.R')


########################################################################################################################
################################################ loads dataset #########################################################
########################################################################################################################
dataset <- 'CO2'
load(paste0('../data_', dataset, '.RData')) # loads dataset

nb.models <- 8
model.types <- c('xgb', 'GAM') # "nb.models" models of each category will be trained

do.par <- TRUE # set this to FALSE on Windows
if (do.par) library(doMC);


########################################################################################################################
######################################## splits between train and test #################################################
########################################################################################################################
data.array <- array(t(featmat), dim=c(fs+1, seq.len, nrow(featmat)/seq.len))
obj.split <- formSplit(pct.test=3e-1, sanity=7) # keeps last 30 percents for test, and removes 7 days between last day of train and first day of test
N.test <- length(obj.split$ind.test) # number of test days
feat.names <- colnames(featmat)[1:(ncol(featmat)-1)] # feature names
target <- colnames(featmat)[ncol(featmat)] # variable to predict


########################################################################################################################
########################################### trains prediction models ###################################################
########################################################################################################################
for (model.type in model.types) {
  
  loc.models <- paste0('./predModels_', dataset, '_', model.type, '/') # trained models will be saved in this directory
  dir.create(file.path('./', loc.models), showWarnings=FALSE)
  
  # samples feature space
  mat.feat <- matrix(runif(nb.models*length(feat.names)), nb.models) > 0.5
  for (r in 1:nb.models) mat.feat[r, ind.analytical] <- TRUE
  
  if (model.type != 'xgb' && do.par) { # XGBoost is already parallelized
    registerDoMC(min(nb.models, max(1, detectCores()-1))) # number of CPU cores (keep one for OS)
    LOSS <- foreach(it=1:nb.models, .combine='c') %dopar% (train_fun[[model.type]](it, dataset, mat.feat[it, ], feat.names, target, verbose=FALSE))
  } else {
    LOSS <- foreach(it=1:nb.models, .combine='c') %do% (train_fun[[model.type]](it, dataset, mat.feat[it, ], feat.names, target, verbose=FALSE))
  }
  
}


########################################################################################################################
###################################### predicts test days with trained models ##########################################
########################################################################################################################
loc.pred <- './allpreds/'
dir.create(file.path('./', loc.pred), showWarnings=FALSE)
for (model.type in model.types) {
  
  loc.models <- paste0('./predModels_', dataset, '_', model.type, '/')
  
  # loads base learners
  obj.learners <- loadLearners(loc.models)
  nb.models <- length(obj.learners$predictors)
  
  # predicts base learners on test set
  print(paste0("predicting ",  nb.models, " ", model.type, " models on dataset ", dataset, ' ...'))
  
  if (model.type != 'xgb' && do.par) {
    registerDoMC(min(nb.models, max(1, detectCores()-1))) # number of CPU cores
    allpred <- foreach(it=1:nb.models, .combine='acomb', .multicombine=TRUE) %dopar% (predict_fun[[model.type]](it)) # N.test x seq.len x nb.models
  } else {
    allpred <- foreach(it=1:nb.models, .combine='acomb', .multicombine=TRUE) %do% (predict_fun[[model.type]](it)) # N.test x seq.len x nb.models
  }
  
  # saves prediction and ground truth objective
  obj.test <- getObj(obj.split$ind.test, target.max=1)
  saveRDS(file=paste0(loc.pred, '/pred_', dataset, '_', model.type, '_all.rds'), 
          object=list(pred=allpred, obj=obj.test, mat.feat=obj.learners$mat.feat))
  
}


########################################################################################################################
####################################### runs model selection on test days ##############################################
########################################################################################################################
num.modelPerDataset <- cntModelsPerDataset(model.types, dataset)
nb.select <- sum(num.modelPerDataset) # number of models to select from

# samples models to use for model selection (or loads models predictions if the number of models to select from is that same as the number of trained models)
obj.sampleModels <- sampleAndLoadPred(dataset, nb.select, num.modelPerDataset, loc.pred)
E.per.model <- obj.sampleModels$E.per.model # prediction errors per model on test dataset
names(E.per.model) <- model.types[obj.sampleModels$ind.type]
ind.best.model <- which.min(E.per.model) # best model such as determined by an oracle

mat.feat <- obj.sampleModels$mat.feat # feature spaces of models to select from
allpreds <- obj.sampleModels$pred # predictions of models to select from
obj.test <- obj.sampleModels$obj # ground truth objective

array.pred.ms <- array(NA, c(N.test, seq.len, seq.len))
array.pred.bm <- array(NA, c(N.test, seq.len, seq.len))
array.pred.aggr <- array(NA, c(N.test, seq.len, seq.len))


obj.madymos <- madymos(nb.select, seq.len, mat.feat, K=2, mu.LBI=0.8, gamma.DP=0.95, thresh.switch.off=0.1)

pb <- txtProgressBar()
for (jj in 1:N.test) { # loops over test days
  
  pred.jj <- allpreds[jj, , ] # predictions of all models for day "jj"
  
  obj.madymos$R.cur[, ] <- 1
  for (tt in 1:seq.len) { # performs "seq.len" strategy readjustments as new observations get available
    
    # gets new observation at time "tt"
    newObs.tt <- obj.test[jj, tt]
    
    # updates model selection strategy
    obj.pred <- updateStrategy(pred.jj[tt:seq.len, , drop=FALSE], newObs=newObs.tt)
    array.pred.ms[jj, tt:seq.len, tt] <- obj.pred$mat_pred # record predictions associated with current model selection strategy on the interval [tt, seq.len]
    
    # best model (oracle)
    array.pred.bm[jj, tt:seq.len, tt] <- pred.jj[tt:seq.len, ind.best.model]
    
    # simple aggregate model
    array.pred.aggr[jj,tt:seq.len,tt] <- rowMeans(pred.jj[tt:seq.len, , drop=FALSE])
    
  }
  
  setTxtProgressBar(pb, jj/N.test)
  
}
close(pb)



########################################################################################################################
############################################ scores and plots results ##################################################
########################################################################################################################
cae.ms <- scoreRun(array.pred.ms, obj.test)
cae.bm <- scoreRun(array.pred.bm, obj.test)
cae.aggr <- scoreRun(array.pred.aggr, obj.test)

plot_curves(
  subS=subS,
  step.plot=1,
  curves=cbind(cae.ms, cae.bm, cae.aggr),
  colors=c('skyblue3', 'pink', 'darkorange'),
  legend=c('ms', paste0('bm.: ', model.types[[obj.sampleModels$ind.type[ind.best.model]]]), 'aggr.'),
  x.name='time of day', y.name='cumulative AE'
)


