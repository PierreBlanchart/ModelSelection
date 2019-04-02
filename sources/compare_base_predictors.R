library(modelselect)
source('utils_generic.R')



###########################################################################################################
###########################################################################################################
loc.pred <- './allpreds/'
dataset.names <- c('bike', 'CO2', 'SolarEnergy', 'ElecReduc')
model.types <- c('CF', 'xgb', 'GAM')


###########################################################################################################
###########################################################################################################
results <- list()
for (dataset in dataset.names) {
  
  results[[dataset]] <- list()
  avg.res <- c()
  for (model.type in model.types) {
    
    print(paste0("Scoring ", model.type, " models on dataset ", dataset, " ..."))
    
    obj.preds <- readRDS(file=paste0(loc.pred, '/pred_', dataset, '_', model.type, '_all.rds'))
    N.test <- dim(obj.preds$pred)[1]
    seq.len <- dim(obj.preds$pred)[2]
    nb.models <- dim(obj.preds$pred)[3] # number of trained models of type "model.type" for dataset "dataset"
    
    start.t <- 1; end.t <- seq.len-1
    if (dataset == 'SolarEnergy') start.t <- 11 # 5H30 <==> earliest sunrise (no observations are done at night time)
    
    mat.cae <- matrix(NA, nb.models, seq.len)
    array.pred <- array(NA, c(N.test, seq.len, seq.len))
    for (n in 1:nb.models) {
      for (jj in 1:N.test) {
        pred.n.jj <- obj.preds$pred[jj, , n]
        for (tt in 1:seq.len) {
          array.pred[jj, tt:seq.len, tt] <- pred.n.jj[tt:seq.len] 
        }
      }
      mat.cae[n, ] <- scoreRun(array.pred, obj.preds$obj)
      
    }
    
    results[[dataset]][[model.type]] <- computeMCAE(mat.cae, start.t, end.t)
    avg.res <- cbind(avg.res, colMeans(mat.cae))
    
  }
  
  # plot averaged results over all predictors
  colors <- glasbey(); N.colors <- length(colors)
  plot_curves(
    subS=subS,
    step.plot=1,
    curves=avg.res,
    colors= colors[(0:(length(model.types)-1))%%N.colors + 1],
    legend=c(model.types),
    x.name='time of day', y.name='cumulative AE'
  )
  
}



###########################################################################################################
###########################################################################################################
# dump results in a latex table
library(kableExtra)

N.datasets <- length(dataset.names)
N.models <- length(model.types)

res.tab <- matrix(NA, N.models, N.datasets)
colnames(res.tab) <- dataset.names
rownames(res.tab) <- model.types

for (dataset in dataset.names) {
  
  for (model.type in model.types) {
    
    txt <- paste0('$', round(results[[dataset]][[model.type]]$MCAE, 1), 
                  ' \\pm ', round(results[[dataset]][[model.type]]$sd_CAE, 1), '$')
    
    res.tab[model.type, dataset] <- txt
    
  }
  
}

code.latex <- kable(res.tab, "latex", booktabs=TRUE, caption="Perf. of base models", escape=FALSE) %>%
  column_spec(1, bold=TRUE) %>%
  row_spec(0, bold=TRUE) %>%
  kable_styling(latex_options = c("striped", "scale_down"))



