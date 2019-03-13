source('utils_generic.R')
source('utils_train.R')



########################################################################################################################
model.types <- c('CF', 'xgb', 'GAM')
dataset.names <- c('Electricity', 'CO2', 'Irradiance', 'Electricity', 'Traffic')

sanitydays <- 7
nb.models <- 256 # number of models to train per type of model and per dataset

do.par <- TRUE



########################################################################################################################
library(doMC)

for (dataset in dataset.names) {
  
  # loads dataset
  load(paste0('data_', dataset, '.RData'))
  data.array <- array(t(featmat), dim=c(fs+1, seq.len, nrow(featmat)/seq.len))
  obj.split <- formSplit(sanity=sanitydays); N.test <- length(obj.split$ind.test)
  
  feat.names <- colnames(featmat)[1:(ncol(featmat)-1)]
  target <- colnames(featmat)[ncol(featmat)]
  
  for (model.type in model.types) {
    
    loc.models <- paste0('./predModels_', dataset, '_', model.type, '/')
    dir.create(file.path('./', loc.models), showWarnings=FALSE)
    
    # sample feature space
    mat.feat <- matrix(runif(nb.models*length(feat.names)), nb.models) > 0.5
    for (r in 1:nb.models) mat.feat[r, ind.analytical] <- TRUE # hard-encoded inside the formula
    
    if (model.type != 'xgb' && do.par) {
      registerDoMC(min(nb.models, max(1, detectCores()-0))) # number of CPU cores
      LOSS <- foreach(it=1:nb.models, .combine='c') %dopar% (train_fun[[model.type]](it, dataset, mat.feat[it, ], feat.names, target, verbose=FALSE))
    } else {
      LOSS <- foreach(it=1:nb.models, .combine='c') %do% (train_fun[[model.type]](it, dataset, mat.feat[it, ], feat.names, target, verbose=FALSE))
    }
    
  }
  
}


