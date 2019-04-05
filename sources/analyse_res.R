library(modelselect)
source('utils_generic.R')



###########################################################################################################
###########################################################################################################
loc.results <- './resMultirun/'
dataset.names <- c('bike', 'CO2', 'Irradiance', 'Electricity')
nb.models <- c(16, 64, 128)



###########################################################################################################
###########################################################################################################
results <- list()
for (dataset in dataset.names) {
  
  results[[dataset]] <- list()
  for (nb in 1:length(nb.models)) {
    
    results[[dataset]][[nb]] <- list()
    if (!file.exists(paste0(loc.results, '/results_', dataset, '_', nb.models[nb], '.rds'))) next;
    res.run <- readRDS(file=paste0(loc.results, '/results_', dataset, '_', nb.models[nb], '.rds'))
    
    N.run <- length(res.run)
    seq.len <- length(res.run[[1]]$cae.ms)
    start.t <- 1; end.t <- seq.len-1
    if (dataset == 'SolarEnergy') start.t <- 11 # 5H30 <==> earliest sunrise (no observations are done at night time)
    
    baselines.op <- rownames(res.run[[1]]$cae.op)
    
    mat.cae.ms <- matrix(0, N.run, seq.len)
    mat.cae.aggr <- matrix(0, N.run, seq.len)
    mat.cae.bm <- matrix(0, N.run, seq.len)
    mat.cae.op <- array(0, c(length(baselines.op), N.run, seq.len))
    for (n in 1:N.run) {
      mat.cae.ms[n, ] <- res.run[[n]]$cae.ms
      mat.cae.aggr[n, ] <- res.run[[n]]$cae.aggr
      mat.cae.bm[n, ] <- res.run[[n]]$cae.bm
      mat.cae.op[, n, ] <- res.run[[n]]$cae.op
    }
    
    results[[dataset]][[nb]]$ms <- computeMCAE(mat.cae.ms, start.t, end.t)
    results[[dataset]][[nb]]$aggr <- computeMCAE(mat.cae.aggr, start.t, end.t)
    results[[dataset]][[nb]]$bm <- computeMCAE(mat.cae.bm, start.t, end.t)
    ind.b <- 1
    for (b in baselines.op) {
      results[[dataset]][[nb]][[b]] <- computeMCAE(mat.cae.op[ind.b, , ], start.t, end.t)
      ind.b <- ind.b+1
    }
    
    
    ###########################################################################################################
    ###########################################################################################################
    # plot averaged results of over all runs
    cae.ms <- colMeans(mat.cae.ms)
    cae.aggr <- colMeans(mat.cae.aggr)
    cae.bm <- colMeans(mat.cae.bm)
    cae.op <- apply(mat.cae.op, MARGIN=c(1, 3), FUN=mean)
    print(paste0('Aggr. score ms = ',  sum(cae.ms, na.rm=TRUE)))
    
    N.methods <- 3 + length(baselines.op)
    colors <- glasbey(); N.colors <- length(colors)
    plot_curves(
      subS=subS,
      step.plot=1,
      curves=cbind(cae.ms, cae.aggr, cae.bm, t(cae.op)),
      colors= colors[(0:(N.methods-1))%%N.colors + 1],
      legend=c('ms', 'aggr.', 'bm', baselines.op),
      x.name='time of day', y.name='cumulative AE'
    )
    
  }
  
}



###########################################################################################################
###########################################################################################################
# dump in latex table
library(kableExtra)

N.models <- length(nb.models)
N.datasets <- length(dataset.names)

baselines.names <- c('ms', 'aggr', 'bm', baselines.op)
aliases <- c('MaDyMos', 'Simple aggr.', 'Best Model (oracle)', baselines.op)
names(aliases) <- baselines.names
N.baselines <- length(baselines.names)

res.tab <- matrix(NA, N.baselines, N.models*N.datasets)
rownames(res.tab) <- aliases
colnames.tab <- rep(NA, length(nb.models)*N.datasets)
for (n in 1:length(dataset.names)) colnames.tab[((n-1)*N.models + 1):(n*N.models)] <- paste0(dataset.names[n], " (Sel.", nb.models, ")")
colnames(res.tab) <- colnames.tab


for (dataset in dataset.names) {
  
  for (n in 1:N.models) {
    
    for (baseline in baselines.names) {
      
      if (length(results[[dataset]][[n]]) > 0) {
        txt <- paste0('$', round(results[[dataset]][[n]][[baseline]]$MCAE, 1), 
                      ' \\pm ', round(results[[dataset]][[n]][[baseline]]$sd_CAE, 1), '$')
      } else {
        txt <- '$NA \\pm NA$'
      }
      res.tab[aliases[baseline], paste0(dataset, " (Sel.", nb.models[n], ")")] <- txt
      
    }
    
  }
  
}

header <- rep(N.models, N.datasets)
names(header) <- dataset.names

code.latex <- kable(res.tab, "latex", booktabs=TRUE, caption="Perf.", escape=FALSE) %>%
  column_spec(1, bold=TRUE) %>%
  row_spec(0, bold=TRUE) %>%
  kable_styling(latex_options = c("striped", "scale_down")) %>%
  add_header_above(c(" " = 1, header))




