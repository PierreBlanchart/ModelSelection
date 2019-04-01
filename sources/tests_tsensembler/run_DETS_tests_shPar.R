library(tsensembler2)
library(modelselect)
library(fastmatch)
library(mgcv)
library(xgboost)
source('../utils_generic.R')
source('../utils_predict.R')

parseCmdLine <- function(flag, cmd_args) {
  ind.flag <- regexpr(paste0('^', flag, '=[0-9]+$'), cmd_args)
  val.flag <- as.numeric(gsub(paste0('^', flag, '=([0-9]+)$'), "\\1", cmd_args[match(TRUE, ind.flag==1)]))
  return(as.numeric(val.flag))
}
parseCmdLine_str <- function(flag, cmd_args) {
  ind.flag <- regexpr(paste0('^', flag, '=[A-Za-z0-9_]+$'), cmd_args)
  val.flag <- gsub(paste0('^', flag, '=([A-Za-z0-9_]+)$'), "\\1", cmd_args[match(TRUE, ind.flag==1)])
  return(val.flag)
}
cmd_args <- commandArgs()
ind.day <- parseCmdLine('indday', cmd_args) # index of test day
nb.models <- parseCmdLine('nbmodels', cmd_args) # number of models to select from
ind.run <- parseCmdLine('indrun', cmd_args) # index of test run
dataset.run <- parseCmdLine_str('dataset', cmd_args) # dataset name

set.seed(seed=(ind.run*1e4 + ind.day))



########################################################################################################################
# load data
load(paste0('../data_', dataset.run, '.RData'))
data.array <- array(t(featmat), dim=c(fs+1, seq.len, nrow(featmat)/seq.len))
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
loc.models <- './models_DETS/'
DETS.model <- readRDS(file=paste0(loc.models, "/DETS_model_", dataset.run, "_", nb.models, "_r", ind.run, ".rds"))
specs <- DETS.model$specs
model.init <- DETS.model$model



########################################################################################################################
ind.seq <- rep((obj.split$ind.test-1)*seq.len, each=seq.len) + rep(1:seq.len, length(obj.split$ind.test))
test <- featmat[ind.seq, ]
test <- as.data.frame(cbind(test[, fs+1, drop=FALSE], test[, 1:fs]))


# compute predictions
print(paste0("Predicting test day ", ind.day, " ..."))
pred.n <- predict(model.init@base_ensemble, test[((ind.day-1)*seq.len+1):(ind.day*seq.len), ], use_all=TRUE)


# running model selection on test days
print(paste0("Doing model selection on test day ", ind.day, " ..."))
loc.res <- './results_DETS/'
dir.create(file.path('./', loc.res), showWarnings=FALSE)

fun_pred <- function(n) {
  
  model.n <- model.init
  
  ind.seq.n <- (n-1)*seq.len + 1
  model.n <- update_weights(model.n, test[1:(ind.seq.n-1), ])
  
  mat.pred.ts <- matrix(NA, seq.len, seq.len)
  
  # pb <- txtProgressBar()
  for (t in 1:(seq.len-1)) {
    
    # tsensembler
    model.n <- update_weights(model.n, test[ind.seq.n, ]) # updates meta model with current observation
    model.t <- model.n
    
    test.t <- test[ind.seq.n:(ind.seq.n+seq.len-t+1-1), ]
    test.t[2:(seq.len-t+1), target] <- NA
    pred.t <- pred.n[t:seq.len, ]
    
    # forecasts seq.len-t+1 steps ahead (including current time step)
    h <- seq.len-t+1
    forecasts.n <- rep(NA, h)
    for (i in 1:h) {
      
      data_o <- test.t[i, ]
      Y_hat <- pred.t[i, ]
      
      Y_hat_recent <- predict(model.t@base_ensemble, model.t@recent_series)
      Y_recent <- get_y(model.t@recent_series, model.t@form)
      
      scores <- model_recent_performance(Y_hat_recent,
                                         Y_recent,
                                         model.t@lambda,
                                         model.t@omega,
                                         model.t@base_ensemble@pre_weights)
      
      model_scores <- scores$model_scores
      top_models <- scores$top_models
      W <- as.vector(model_scores[nrow(model_scores), ])
      C <- top_models[[length(top_models)]]
      Y_hat_j <- Y_hat[, C]
      W_j <- proportion(W[C])
      p_forecast <- sum(Y_hat_j * W_j)
      forecasts.n[i] <- p_forecast
      
      data_o[, 1] <- p_forecast
      model.t@recent_series <- rbind.data.frame(model.t@recent_series, data_o)
      
    }
    
    mat.pred.ts[t:seq.len, t] <- forecasts.n[1:h]
    ind.seq.n <- ind.seq.n+1
    
    # setTxtProgressBar(pb, t/seq.len)
    
  }
  # close(pb)
  
  saveRDS(file=paste0(loc.res, '/res_', dataset.run, '_', nb.models, '_r', ind.run, '_d', n, '.rds'), mat.pred.ts)
  
}


# run and save
fun_pred(ind.day)



