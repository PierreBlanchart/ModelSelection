library(foreach)
library(abind)
acomb <- function(...) abind(..., along=3)



# ind.train : index of train days
# ind.test : index of test days
format_CF <- function(ind.train, ind.test, ind.feat=NULL, target.max=NA, melt=FALSE) {
  
  seq.len <- dim(data.array)[2]
  fs <- dim(data.array)[1] - 1
  
  if (is.null(ind.feat)) {
    ind.feat <- c(rep(TRUE, fs), FALSE)
  } else {
    ind.feat <- c(ind.feat, FALSE)
  }
  
  if (!melt) {
    temp.size <- sum(ind.feat)*seq.len
    
    if (!is.na(ind.train[1])) {
      mat.train <- t(matrix(data.array[ind.feat, , ind.train], temp.size))
      obj.train <- t(matrix(data.array[fs+1, , ind.train], seq.len))
      if (is.na(target.max)) target.max <- max(obj.train)
      obj.train <- obj.train/target.max
    } else {
      mat.train <- NA
      obj.train <- NA
    }
    mat.test <- t(matrix(data.array[ind.feat, , ind.test], temp.size))
    obj.test <- t(matrix(data.array[fs+1, , ind.test], seq.len))/target.max
    
  } else {
    temp.size <- sum(ind.feat)
    
    if (!is.na(ind.train[1])) {
      mat.train <- t(matrix(data.array[ind.feat, , ind.train], temp.size))
      obj.train <- as.vector(data.array[fs+1, , ind.train])
      if (is.na(target.max)) target.max <- max(obj.train)
      obj.train <- obj.train/target.max
    } else {
      mat.train <- NA
      obj.train <- NA
    }
    mat.test <- t(matrix(data.array[ind.feat, , ind.test], temp.size))
    obj.test <- as.vector(data.array[fs+1, , ind.test])/target.max
  }
  
  res <- list()
  res$mat.train <- mat.train
  res$mat.test <- mat.test
  res$obj.train <- obj.train
  res$obj.test <- obj.test
  res$index.train <- ind.train
  res$index.test <- ind.test
  res$target.max <- target.max
  
  return(res)
  
}


# get objective
getObj <- function(ind.obj, target.max) {
  seq.len <- dim(data.array)[2]
  fs <- dim(data.array)[1] - 1
  return(t(matrix(data.array[fs+1, , ind.obj], seq.len))/target.max)
}


# generates random model name
genModelName <- function(model.name='model', N.units=3, max.key=1e3) {
  for (i in 1:N.units) {
    model.name <- paste0(model.name, '_', ceiling(runif(1)*max.key))
  }
  return(paste0(model.name, '.rds'))
}


# split dataset between train and test set, with a "sanity" interval between end of train and beginning of test
formSplit <- function(pct.test=3e-1, sanity=7) {
  
  if (!exists("data.array")) data.array <- array(t(featmat), dim=c(fs+1, seq.len, nrow(featmat)/seq.len))
  seq.len <- dim(data.array)[2]
  fs <- dim(data.array)[1] - 1
  
  # establishing an index of available days
  find.nna <- apply(data.array, MARGIN=c(3), FUN=function(mat) sum(is.na(mat)))
  find.nna <- which(find.nna < 1)
  
  # separating between train and test
  pct.test <- 3e-1
  N.tot <- length(find.nna)
  N.test <- round(pct.test*N.tot)
  ind.train <- find.nna[1:(N.tot-(N.test+sanity))]
  ind.test <- find.nna[(N.tot-N.test+1):N.tot]
  
  obj.split <- list()
  obj.split$seq.len <- seq.len
  obj.split$ind.train <- ind.train
  obj.split$ind.test <- ind.test
  obj.split$target.max.train <- max(data.array[fs+1, , ind.train])
  
  return(obj.split)
  
}


# load models located in "loc.models"
loadLearners <- function(loc.models='./predModels/') {
  
  models <- list.files(path=loc.models, pattern = "^model_[0-9_]+.rds$")
  N.models <- length(models)
  if (N.models < 1) return(list(N.predictors=0))
  
  predictors <- list()
  for (n in 1:N.models) {
    predictors[[n]] <- readRDS(paste0(loc.models, '/', models[n]))
    if (n==1) {
      fs <- length(predictors[[n]]$ind.feat)
      mat.feat <- matrix(NA, N.models, fs)
    }
    mat.feat[n, ] <- predictors[[n]]$ind.feat
  }
  
  # format info
  obj.info <- list(predictors=predictors, mat.feat=mat.feat, model.names=models)
  
  return(obj.info)
  
}


scoreRun <- function(array.pred, obj.gt) {
  N.test <- dim(array.pred)[1]
  seq.len <- ncol(obj.gt)
  mat.score <- matrix(NA, N.test, seq.len)
  for (n in 1:N.test) {
    for (t in 1:(seq.len-1)) {
      mat.score[n, t] <- sum(abs(array.pred[n, (t+1):seq.len, t] - obj.gt[n, (t+1):seq.len]))
    }
  }
  return(colMeans(mat.score))
}


cntModelsPerDataset <- function(model.types, dataset.names) {
  
  nb.types <- length(model.types)
  nb.datasets <- length(dataset.names)
  
  num.modelPerDataset <- matrix(NA, nb.types, nb.datasets)
  colnames(num.modelPerDataset) <- dataset.names; rownames(num.modelPerDataset) <- model.types
  for (model.type in model.types) {
    for (dataset in dataset.names) {
      obj.preds <- readRDS(file=paste0(loc.pred, '/pred_', dataset, '_', model.type, '_all.rds'))
      num.modelPerDataset[model.type, dataset] <- dim(obj.preds$pred)[3]
    }
  }
  
  return(num.modelPerDataset)
  
}


# samples nb.models models from the three categories of learned models and loads corresponding predictions and feature spaces
sampleAndLoadPred <- function(dataset.run, nb.models, num.modelPerDataset, loc.pred) {
  
  temp.cum <- c(0, cumsum(num.modelPerDataset[, dataset.run]))
  names(temp.cum)[1:nrow(num.modelPerDataset)] <- model.types
  
  # samples models to use for model selection
  N.dataset <- sum(num.modelPerDataset[, dataset.run])
  ind.sampled <- sample.int(N.dataset, nb.models, replace=FALSE)
  ind.type.n <- findInterval(ind.sampled, temp.cum, left.open=TRUE, all.inside=TRUE) # sampled model type
  ind.model.n <- ind.sampled - temp.cum[ind.type.n] # index of sampled model inside chosen type
  
  # loads predictions
  ind.type.prev <- 0
  obj.sorted <- sort(ind.type.n, index.return=TRUE) # orders by model type not to load several times the same prediction files
  
  allpreds <- NULL
  mat.feat <- matrix(NA, nb.models, fs)
  for (m in 1:nb.models) {
    
    if (obj.sorted$x[m] != ind.type.prev) {
      obj.preds <- readRDS(file=paste0(loc.pred, '/pred_', dataset.run, '_', model.types[obj.sorted$x[m]], '_all.rds'))
      if (is.null(allpreds)) {
        dim.preds <- dim(obj.preds$pred)
        N.test <- dim.preds[1]
        seq.len <- dim.preds[2]
        allpreds <- array(NA, dim=c(N.test, seq.len, nb.models))
        obj.test <- obj.preds$obj
      }
      ind.type.prev <- obj.sorted$x[m]
    }
    
    allpreds[, , obj.sorted$ix[m]] <- obj.preds$pred[, , ind.model.n[obj.sorted$ix[m]]]
    mat.feat[obj.sorted$ix[m], ] <- obj.preds$mat.feat[ind.model.n[obj.sorted$ix[m]], ]
    
  }
  
  E.MAE <- foreach(it=1:nb.models, .combine='acomb', .multicombine=TRUE) %do% (abs(allpreds[, , it]-obj.test))
  E.per.model <- apply(E.MAE, MARGIN=c(3), FUN=sum) # error per model
  
  return(list(pred=allpreds, obj=obj.test, mat.feat=mat.feat, E.per.model=E.per.model, ind.type=ind.type.n, ind.model=ind.model.n))
  
}


