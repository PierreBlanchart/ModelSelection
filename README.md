## Forecasting using Dynamic Model Selection

This package optimizes muti-step model selection strategies in the context of multiple-step-ahead multivariate pseudo-periodic time series forecasting. The core method, MaDyMos (Multiple-step-ahead Dynamic Model Selection), uses observations about the past local performance of each model to issue a model selection strategy running on the whole prediction horizon. 
As new observations become available, the model is able to update the current model selection strategy in an online way, and as such, is designed for fine-grained and reactive management of complex dynamical systems.

The typical application scenario is the following:

* We have at our disposal several predictors operating on exogenous variables.

* Exogenous variables are a mix of non-noisy analytical features (such as the time of day, the solar elevation ...), and of noisy/erroneous features (such as weather forecasts).

* Noisy features are updated at the beginning of each period, for the period to come.

In this context, our goal is to produce a model selection strategy, that runs from the current time till the end of the prediction horizon (or the end of the current period, depending on what is needed).

![Alt text](./figures/applicationScenario.png?raw=true "Typical application scenario")

## Datasets

To support the experiments of the paper "Multiple-step-ahead Dynamic Model Selection", we provide 4 datasets belonging to 4 different application domains. These datasets are provided so that the reproducibility of the approach described in the paper can be assessed. They cannot be used for any other purpose while the paper isn't published.

### Bike sharing dataset :



### Irradiance dataset :

A solar irradiance dataset containing 3 years of data with a sampling rate of 30 minutes. The value to predict is the Global Horizontal Irradiance (Watts per square meter). There are 10 exogenous features which are a mix of calendar data, and day-ahead weather forecasts.

### Electricity consumption dataset :

A dataset containing 4 years of electricity consumption over the territory of France with a sampling rate of 30 minutes. The value to predict is the electrical consumption (in MegaWatts). There are 119 exogenous features which are a mix of calendar data, day-ahead weather forecasts, and engineered features.

### CO2 pollution dataset :

A dataset containing 3.5 years of air quality measures in the metro with a sampling rate of 1 hour. The value to predict is the air concentration in CO2 (parts-per-million -- ppm). There are 8 exogenous features corresponding to weather forecasts and calendar data.

