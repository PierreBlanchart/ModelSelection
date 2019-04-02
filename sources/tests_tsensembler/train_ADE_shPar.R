library(tsensembler2)
library(modelselect)
library(fastmatch)
library(mgcv)
library(xgboost)
source('../utils_generic.R')
source('../utils_GAMs.R')
source('../utils_predict.R')

parseCmdLine <- function(flag, cmd_args) {
  ind.flag <- regexpr(paste0('^', flag, '=[0-9]+$'), cmd_args)
  val.flag <- as.numeric(gsub(paste0('^', flag, '=([0-9]+)$'), "\\1", cmd_args[match(TRUE, ind.flag==1)]))
  return(val.flag)
}
parseCmdLine_str <- function(flag, cmd_args) {
  ind.flag <- regexpr(paste0('^', flag, '=[A-Za-z0-9_]+$'), cmd_args)
  val.flag <- gsub(paste0('^', flag, '=([A-Za-z0-9_]+)$'), "\\1", cmd_args[match(TRUE, ind.flag==1)])
  return(val.flag)
}
cmd_args <- commandArgs()
dataset.run <- parseCmdLine_str('dataset', cmd_args) # dataset name
nb.models <- parseCmdLine('nbmodels', cmd_args) # number of models to select from
ind.run <- parseCmdLine('indrun', cmd_args) # index of test run

set.seed(seed=ind.run*1e4 + + round(as.numeric(as.POSIXlt(Sys.time()))))



########################################################################################################################
model.types <- c('xgb', 'GAM')



########################################################################################################################
# load data
load(paste0('../data_', dataset.run, '.RData'))
data.array <- array(t(featmat), dim=c(fs+1, seq.len, nrow(featmat)/seq.len))
feat.names <- colnames(featmat)[1:(ncol(featmat)-1)]
target <- colnames(featmat)[ncol(featmat)]

obj.split <- formSplit(pct.test=3e-1, sanity=7) # keeps last 30 percents for test, and removes 7 days between last day of train and first day of test
obj.test <- getObj(obj.split$ind.test, target.max=1)
N.test <- length(obj.split$ind.test) # number of test days



########################################################################################################################
# building the ensemble
ind.seq <- rep((obj.split$ind.train-1)*seq.len, each=seq.len) + rep(1:seq.len, length(obj.split$ind.train))
train <- featmat[ind.seq, ]
train <- as.data.frame(cbind(train[, fs+1, drop=FALSE], train[, 1:fs]))


# setting up base model parameters
while (TRUE) {
  type.models <- model.types[ceiling(length(model.types)*runif(nb.models))]
  if (sum(type.models=='xgb') > 0 && sum(type.models=='GAM') > 0) break;
}
name.models <- paste0(type.models, '_', 1:nb.models)

# sampling model feature spaces
mat.feat <- matrix(runif(nb.models*fs), nb.models) > 0.5
for (n in 1:nb.models) mat.feat[n, ind.analytical] <- TRUE
rownames(mat.feat) <- name.models
colnames(mat.feat) <- feat.names

# generating GAM models formulas
formula.GAM <- c()
for (n in 1:nb.models) {
	if (type.models[n]=='GAM') {
	  obj.GAM <- generateFormula(dataset.run, mat.feat[n, ], feat.names, train)
		formula.GAM <- c(formula.GAM, obj.GAM$formula)
	}
}

# XGBoost model parameters
param.xgb <- list(
	objective = "reg:linear",
	eval_metric="mae",
	eta = 0.05,
	max_depth = 8,
	colsample_bytree = 0.9,
	subsample = 0.9,
	min_child_weight = 4,
	lambda = 1,
	maximize=FALSE,
	nrounds=5e2,
	verbosity=1
)

specs <- model_specs(
	learner = paste0("bm_", model.types),
	learner_pars = list(
		bm_xgb = list(model_names=name.models[type.models=='xgb'], mat_feat=mat.feat[type.models=='xgb', , drop=FALSE], param_xgb=param.xgb), 
		bm_GAM = list(model_names=name.models[type.models=='GAM'], formula=formula.GAM)
	))

model.init <- ADE(as.formula(paste0(target, ' ~.')), train, specs)
print(model.init@meta_model[[1]]$forest$independent.variable.names) # check variable names on which meta-model relies


# saves model
loc.models <- './models_ADE/'
dir.create(file.path('./', loc.models), showWarnings=FALSE)
saveRDS(file=paste0(loc.models, "/ADE_model_", dataset.run, "_", nb.models, "_r", ind.run, ".rds"), 
        object=list(model=model.init, specs=specs, train=train, mat.feat=mat.feat))


