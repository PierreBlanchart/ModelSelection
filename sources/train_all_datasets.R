source('utils_generic.R')
source('utils_train.R')



########################################################################################################################
model.types <- c('CF', 'xgb', 'GAM')
dataset.names <- c('Electricity', 'CO2', 'Irradiance', 'Electricity', 'Traffic')

nb.models <- 256 # number of models to train per type of model and per dataset

do.par <- TRUE # set this to FALSE on Windows
if (do.par) library(doMC);



########################################################################################################################
library(doMC)

for (dataset in dataset.names) {
  
  # loads dataset
  load(paste0('data_', dataset, '.RData'))
  data.array <- array(t(featmat), dim=c(fs+1, seq.len, nrow(featmat)/seq.len))
  obj.split <- formSplit(sanity=7); N.test <- length(obj.split$ind.test)
  
  feat.names <- colnames(featmat)[1:(ncol(featmat)-1)]
  target <- colnames(featmat)[ncol(featmat)]
  
  for (model.type in model.types) {
    
    loc.models <- paste0('./predModels_', dataset, '_', model.type, '/')
    dir.create(file.path('./', loc.models), showWarnings=FALSE)
    
    # samples feature spaces
    mat.feat <- matrix(runif(nb.models*length(feat.names)), nb.models) > 0.5
    for (r in 1:nb.models) mat.feat[r, ind.analytical] <- TRUE
    
    if (model.type != 'xgb' && do.par) {
      registerDoMC(min(nb.models, max(1, detectCores()-0))) # number of CPU cores
      LOSS <- foreach(it=1:nb.models, .combine='c') %dopar% (train_fun[[model.type]](it, dataset, mat.feat[it, ], feat.names, target, verbose=FALSE))
    } else {
      LOSS <- foreach(it=1:nb.models, .combine='c') %do% (train_fun[[model.type]](it, dataset, mat.feat[it, ], feat.names, target, verbose=FALSE))
    }
    
  }
  
}


