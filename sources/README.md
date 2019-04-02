## Running tests

### Quick try

The script "example_run.R" can be run to have a quick overview of the potential of our model selection approach compared to basic ensembling selection techniques such as model averaging and cross validation.
The script performs model training, model prediction and selection, and plots the resulting performance scores, as well, as an example of a model selection strategy re-updated at each instant, as new observations get available.

```r
source('example_use.R')
```

The script takes less than 1 minute to run on an Intel Core I7, with a Linux distribution.

The paragraphs below describe the way to perform more thorough tests, and, in particular, explain how to reproduce the results of the paper.

### Training models

Models are trained by running the following command from the R command line:

```r
source('train_all_datasets.R')
```

The datasets, the number and types of models to train are set directly in the script:

Example : training 256 models of 3 different types - CF, xgb and GAM - on 5 different datasets, i.e. a total of 3 x 256 x 5 models :
```r
model.types <- c('CF', 'xgb', 'GAM')
dataset.names <- c('bikeShare', 'CO2', 'Irradiance', 'Electricity', 'Traffic')
nb.models <- 256 # number of models to train per type of model and per dataset
```

Three types of models are implemented: Siamese-network based Collaborative Filtering (CF), XGBoost regression trees (xgb) and Generalized additive models (GAMs).
CF models are pretty long to train since it is a custom implementation we did in R. We will release later a more efficient package implementing these models.

Trained models are stored on the disk in a local folder of the type: "./predModels_#DATASET_#MODELTYPE/".

### Predicting models

Each trained model is used to predict the test set of each dataset by running the command:

```r
source('test_all_datasets.R')
```

Prediction are stored on the disk in a local folder: "./allpreds/".

### Running model selection

Model selection is performed on each dataset by running the command:

```r
source('run_tests.R')
```

The datasets, the number and types of models to select from, as well as the number of run to perform and the baselines to compare to are set directly in the script:

Example : performing model selection among 3 pools of models of respective sizes 8, 16 and 32 containing models of 2 different types - xgb and GAM - trained on the 'CO2' dataset, and comparing each time with the baselines FS, MLpol, MLewa and EWA :
```r
model.types <- c('xgb', 'GAM')
dataset.run <- 'CO2'
nb.models <- c(8, 16, 32) # number of models to aggregate / select from
baselines.op <- c('FS', 'MLpol', 'MLewa', 'EWA')
N.run <- 1
```

The performance scores associated with each run are stored on the disk in a local folder: "./resMultirun/".

The results from each run are plotted, and assembled in a table (such as the one in the paper) by running the command:

```r
source('analyse_res.R')
```


### End-to-end example run:
An example script can be run that summarizes all the above tests - from model training to model prediction and selection. This example showcases the gain in performance brought by using our model selection algorithm, compared to simple ensembling/model selection approaches (such as prediction averaging and cross validation) commonly used to aggregate several models:

```r
source('example_use.R')
```

The script takes less than 1 minute to run on an Intel Core I7, with a Linux distribution.

The figures below shows the output of the script using the CO2 dataset. 
* The first figure shows an example run of the MaDyMoS procedure on a single test day. A colored square symbolizes the choice of one model among 16 possible models, with a 1 hour time granularity. At each instant, a strategy is issued that goes from the current time to the end of the prediction horizon (end of the day in this case). Thus, each line of the color matrix below corresponds to a readjustment of the previous model selection strategy given the observations at the current time step.
* The second figure represents the Cumulative Mean Average Error (CMAE) corresponding to successive strategy re-adjustements between 0H and 24H. For this dataset, a re-adjustment is performed every hour. The CMAE is averaged here over all test days.

![Alt text](../figures/strategy_CO2.png?raw=true "Model Selection Strategy Readjustment")

![Alt text](../figures/CAE_CO2.png?raw=true "Cumulative Mean Average Error")


