## Forecasting using Dynamic Model Selection

This package optimizes muti-step model selection strategies in the context of multiple-step-ahead multivariate pseudo-periodic time series forecasting. The core method, MaDyMos (Multiple-step-ahead Dynamic Model Selection), uses observations about the past local performance of each model to issue a model selection strategy running on the whole prediction horizon. 
As new observations become available, the model is able to update the current model selection strategy in an online way, and as such, is designed for fine-grained and reactive management of complex dynamical systems.

The typical application scenario is the following:

* We have at our disposal several predictors operating on exogenous variables.

* Exogenous variables are a mix of non-noisy analytical features (such as the time of day, the solar elevation ...), and of noisy/erroneous features (such as weather forecasts).

* Noisy features are updated at the beginning of each period, for a defined time horizon running up to several periods ahead, as shown in the figure below.

In this context, our goal is to produce a model selection strategy, that runs from the current time till the end of the prediction horizon (or the end of the current period, depending on what is needed).

![Alt text](./figures/applicationScenario.png?raw=true "Typical application scenario")

## Datasets

To support the experiments of the paper "Multiple-step-ahead Dynamic Model Selection", we provide 4 datasets belonging to 4 different application domains. These datasets are provided so that the reproducibility of the approach described in the paper can be assessed. They cannot be used for any other purpose while the paper isn't published.

### Bike sharing dataset :

A dataset containing 2 years of demand in a bike sharing network with a sampling rate of 1 hour. The value to predict is the total number of bike rentals. There are 14 exogenous variables corresponding to weather forecasts and calendar data.

### Irradiance dataset :

A solar irradiance dataset containing 3 years of data with a sampling rate of 30 minutes. The value to predict is the Global Horizontal Irradiance (Watts per square meter). There are 10 exogenous features which are a mix of calendar data, and day-ahead weather forecasts.

### Electrical load dataset :

A dataset containing 4 years of electrical load over the territory of France with a sampling rate of 30 minutes. The value to predict is the electrical load (in MegaWatts). There are 21 exogenous features which are a mix of calendar data, day-ahead weather forecasts, and engineered features.

### CO2 pollution dataset :

A dataset containing 3.5 years of air quality measures in the metro with a sampling rate of 1 hour. The value to predict is the air concentration in CO2 (parts-per-million -- ppm). There are 8 exogenous features corresponding to weather forecasts and calendar data.

## Using the package

To use the package, go to the source folder, and, first, install the package "modelselect" implementing the MaDyMos method described in the paper (requires RcppArmadillo).
Other packages are required as well to train the predictors, and perform tests.

```r
install.packages(pkgs=c("xgboost", "mgcv", "foreach", "abind"))
install.packages("RcppArmadillo") # the MaDyMos procedure is coded in RcppArmadillo, a templated C++ linear algebra library by Conrad Sanderson.
install.packages(pkgs="./modelselect/", repos=NULL)
```

Linux users can install the package "doMC" which allows to parellize the training and testing of models.

```r
install.packages("doMC")
```

An example script can be run that summarizes all the above tests - from model training to model prediction and selection. This example showcases the gain in performance brought by using our model selection algorithm, compared to simple ensembling/model selection approaches (such as prediction averaging and cross validation) commonly used to aggregate several models:

```r
source('example_use.R')
```
