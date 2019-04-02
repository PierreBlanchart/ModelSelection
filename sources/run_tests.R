source('utils_multirun.R')



########################################################################################################################
loc.pred <- './allpreds/'

model.types <- c('CF', 'xgb', 'GAM')
dataset.names <- c('bike', 'CO2', 'Irradiance', 'Electricity', 'Traffic')
dataset.run <- 'bike'

num.modelPerDataset <- cntModelsPerDataset(model.types, dataset.names)
nb.models <- c(16, 64, 128) # number of models to select from

if (nb.models > sum(num.modelPerDataset[, dataset.run])) {
  stop(paste0("Number of models to select from should be less than total number of models trained for dataset ", dataset.run))
}


########################################################################################################################
# R-opera baselines
all.baselines <- c('BOA', 'FS', 'MLpol', 'MLewa', 'EWA', 'OGD', 'Ridge')
baselines.loss <- c('absolute', 'absolute', 'absolute', 'absolute', 'absolute', 'absolute', 'square'); names(baselines.loss) <- all.baselines

baselines.op <- c('FS', 'MLpol', 'MLewa', 'EWA')
baselines.loss <- baselines.loss[baselines.op]

# performs runs
load(paste0('data_', dataset.run, '.RData')); rm(featmat); gc()


for (n in nb.models) {
  
  print(paste0('Running tests with ', n, ' models ...'))
  
  res.run <- multirun(dataset.run, nb.models=n, baselines.op, baselines.loss, N.run=128, PLOT=FALSE)
  
  # saves results
  loc.results <- './resMultirun/'
  dir.create(file.path('./', loc.results), showWarnings=FALSE)
  saveRDS(file=paste0(loc.results, '/results_', dataset.run, '_', n, '.rds'), res.run)

}
