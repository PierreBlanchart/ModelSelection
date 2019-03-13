library(mgcv)
library(foreach)
library(abind)
acomb <- function(...) abind(..., along=3)



# predicts CF models on test data
closure_pred_CF <- function(n) {
  
  # print(paste0("Predicting model ", n, " on test data ..."))
  model.n <- obj.learners$predictors[[n]]
  obj_pred <- format_CF(obj.split$ind.train, obj.split$ind.test, model.n$ind.feat, target.max=model.n$target.max)
  mat.train <<- obj_pred$mat.train; obj.train <<- obj_pred$obj.train; mat.test <<- obj_pred$mat.test; obj.test <<- obj_pred$obj.test
  
  obj.pred.n <- rmadymos2:::predictTest.simple(model.n$net, model.n$funLayers, model.n$sigma,
                                               measure='gaussian', return.pred=TRUE, verbose=TRUE)
  
  # return(matrix(n, N.test, seq.len))
  return(obj.pred.n$predictions*model.n$target.max)
  
}



# predicts xgb models on test data
closure_pred_xgb <- function(n) {
  
  # print(paste0("Predicting model ", n, " on test data ..."))
  model.n <- obj.learners$predictors[[n]]
  obj_test <- format_CF(NA, obj.split$ind.test, ind.feat=model.n$ind.feat, target.max=model.n$target.max, melt=TRUE)
  pred.test <- t(matrix(predict(model.n$xgb_model, obj_test$mat.test), seq.len))
  
  return(pred.test*model.n$target.max)
  
}



# predicts GAM models on test data
closure_pred_GAM <- function(n) {
  
  # print(paste0("Predicting model ", n, " on test data ..."))
  model.n <- obj.learners$predictors[[n]]
  ind.test.seq <- rep((obj.split$ind.test-1)*seq.len, each=seq.len) + (1:seq.len)
  feat.data.test <- as.data.frame(featmat[ind.test.seq, ])
  gam.forecast <- pmax(0, as.vector(predict(model.n$model, newdata=feat.data.test)))
  
  return(t(matrix(gam.forecast, seq.len)))
  
}



predict_fun <- list(CF=closure_pred_CF, xgb=closure_pred_xgb, GAM=closure_pred_GAM)


