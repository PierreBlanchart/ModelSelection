# generates formula for training GAM model for dataset "dataset" using only features in "feat.names[ind.feat]"
library(foreach)
generateFormula <- function(dataset, ind.feat, feat.names, feat.data.train, verbose=FALSE) {
  
  switch(dataset, 
         
         "Traffic" = {
           tv1 <- c("Temperature.y", "Shortwaveirradiation.y", "Rainfall.y", "RelativeHumidity.y")
           tv2 <- "Shortwaveirradiation.y"
           
           ind.keep <- rep(FALSE, length(feat.names))
           names(ind.keep) <- feat.names
           ind.keep[c("hourminuteofday", "dayofweek", "dayofyear", "month.int", tv1, tv2, "isbizday", "isweekend")] <- TRUE
           
           tv1 <- intersect(tv1, feat.names[ind.feat])
           tv2 <- intersect(tv2, feat.names[ind.feat])
           
           myformula <- c(
             "value ~ te(hourminuteofday, dayofweek, k=c(24, 7))", 
             "s(dayofyear, k=20)", 
             "s(month.int, k=12)", 
             if (length(tv1) > 0) paste0("s(",  paste0(tv1, collapse=", "), ", k=20)"), 
             if (length(tv2) > 0) paste0("s(", tv2, ", k=20)"), 
             "isbizday + isweekend"
           )
         }, 
         
         "CO2" = {
           tv1 <- c("temp.forecast", "neb.forecast")
           
           ind.keep <- rep(FALSE, length(feat.names))
           names(ind.keep) <- feat.names
           ind.keep[c("hourofday", "dayofweek", tv1, "dayofyear.rel", "vacances", "ferie", "pont")] <- TRUE
           
           tv1 <- intersect(tv1, feat.names[ind.feat])
           if (length(tv1) > 0) n.degree.tv1 <- max(foreach(it=1:length(tv1), .combine=c) %do% (length(unique(feat.data.train[, tv1[it]]))))
           
           myformula <- c(
             'CO2 ~ te(hourofday, dayofweek, k=c(24, 7), bs=c("cr", "ps"))', 
             if (length(tv1) > 0) paste0("s(",  paste0(tv1, collapse=", "), ", k=", min(n.degree.tv1, 20), ")"), 
             "s(dayofyear.rel, k=20)", 
             "vacances + ferie + pont"
           )
         }, 
         
         "bike" = {
           tv1 <- c("temp", "atemp", "hum", "windspeed")
           
           ind.keep <- rep(FALSE, length(feat.names))
           names(ind.keep) <- feat.names
           ind.keep[c("hr", "weekday", tv1, "dayofyear")] <- TRUE
           
           tv1 <- intersect(tv1, feat.names[ind.feat])
           
           myformula <- c(
             'cnt ~ te(hr, weekday, k=c(24, 7))', 
             if (length(tv1) > 0) paste0("s(",  paste0(tv1, collapse=", "), ", k=20)"), 
             "s(dayofyear, k=20)"
           )
         }, 
         
         "Irradiance" = {
           tv1 <- c("lwir.1dah", "swir.1dah", "temp.1dah", "heat_gr.1dah", "water.1dah", "wind.u.1dah", "clcover.1dah")
           tv2 <- "swir.1dah"
           
           ind.keep <- rep(FALSE, length(feat.names))
           names(ind.keep) <- feat.names
           ind.keep[c("solarelevation", "daynight", tv1, tv2, "timeofyear")] <- TRUE
           
           tv1 <- intersect(tv1, feat.names[ind.feat])
           tv2 <- intersect(tv2, feat.names[ind.feat])
           if (length(tv1) == 1 && length(tv2) == 1 && tv1[1] == "swir.1dah") include_tv2 <- FALSE else include_tv2 <- TRUE
           
           if (length(tv1) > 0) n.degree.tv1 <- max(foreach(it=1:length(tv1), .combine=c) %do% (length(unique(feat.data.train[, tv1[it]]))))
           
           myformula <- c(
             "ghi ~ s(solarelevation, daynight, k=30)", 
             if (length(tv1) > 0) paste0("s(",  paste0(tv1, collapse=", "), ", k=", min(n.degree.tv1, 20), ")"), 
             if (length(tv2) > 0 && include_tv2) paste0("s(", tv2, ", k=12)"), 
             "s(timeofyear, k=20)"
           )
         }, 
         
         "Electricity" = {
           temperature.vars <- grep("^temperature_[0-9]+$", feat.names)
           temperature.vars <- intersect(feat.names[temperature.vars], feat.names[ind.feat])
           formula.temperatures <- ( paste0("s(", temperature.vars, ", k=12)", collapse=" + ") )
           
           nebul.vars <- grep("^nebulosite_[0-9]+$", feat.names)
           nebul.vars <- intersect(feat.names[nebul.vars], feat.names[ind.feat])
           formula.nebul <- ( paste0("s(", nebul.vars, ", k=6)", collapse=" + ") )
           
           tv.mean <- c("mean.per.month.temperature_mean", "temperature_mean", "d2mean.month.temperature_mean")
           
           ind.keep <- rep(FALSE, length(feat.names))
           names(ind.keep) <- feat.names
           ind.keep[c("dayh", "dayofweek", "solarelevation", "dayofyear.rel", 
                      temperature.vars, nebul.vars, tv.mean, 
                      "ferie", "dayofweek", "vacances.ete", "vacances.printemps", 
                      "vacances.toussaint", "vancances.noel", "vacances.hiver", "France.vacances"
           )] <- TRUE
           
           tv.mean <- intersect(tv.mean, feat.names[ind.feat])
           
           myformula <- c(
             'Consommation ~ te(dayh, dayofweek, k=c(24, 7)) ', 
             's(solarelevation, k=24) ', 
             's(dayofyear.rel, k=20) ', 
             if (length(temperature.vars) > 0) formula.temperatures,
             if (length(nebul.vars) > 0) formula.nebul, 
             if (length(tv.mean) > 0) paste0("s(",  paste0(tv.mean, collapse=", "), ", k=20)"), 
             'ferie + dayofweek + vacances.ete + vacances.printemps ', 
             'vacances.toussaint + vancances.noel + vacances.hiver + France.vacances'
           )
         }
         
  )
  
  myformula <- as.formula(paste0(myformula, collapse=" + "))
  if (verbose) print(myformula)
  
  return(list(formula=myformula, ind.keep=ind.keep))
  
}



