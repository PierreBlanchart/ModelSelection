source('utils_generic.R')
source('utils_predict.R')



########################################################################################################################
model.types <- c('CF', 'xgb', 'GAM')
dataset.names <- c('bike', 'CO2', 'Irradiance', 'Electricity', 'Traffic')

do.par <- TRUE # set this to FALSE on Windows
if (do.par) library(doMC);



########################################################################################################################
library(doMC)


loc.pred <- './allpreds/'
dir.create(file.path('./', loc.pred), showWarnings=FALSE)

for (dataset in dataset.names) {
  
  # loads dataset
  load(paste0('data_', dataset, '.RData'))
  data.array <- array(t(featmat), dim=c(fs+1, seq.len, nrow(featmat)/seq.len))
  obj.split <- formSplit(sanity=7); N.test <- length(obj.split$ind.test)
  
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
  
}

