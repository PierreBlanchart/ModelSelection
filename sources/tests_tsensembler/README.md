The tests below allow to reproduce the performance scores obtained by the baselines, DETS and ADE, implemented in the [Tsensembler](https://github.com/vcerqueira/tsensembler) package.
To run the tests, the user needs to install our forked implementation of the Tsensembler package.
```r
install.packages(pkgs="./tsensembler2/", repos=NULL)
```
The tests requires the shell tool "[GNU parallel](https://www.gnu.org/software/parallel/)". The tests below are computationally intensive : a single train and test run of a DETS model with 16 predictors takes about 7 hours on a 48 cores cluster node.

The following steps then need to be followed in the right order :

## Training DETS models :

```sh
bash train_DETS_shPar.sh nRuns nModels datasetName nPar
```
The above command launches the parallel training of "nRuns" DETS models containing "nModels" predictors on the dataset "datasetName", using "nPar" parallel processes.
Models are saved on the disk in a local folder "./models_DETS/".


## Predicting DETS models :

```sh
bash run_DETS_tests_shPar.sh nRuns nDays nbModels datasetName nPar
```
The above command launches the parallel testing of the previously trained "nRuns" DETS models on the dataset "datasetName" containing "nDays" test days, using "nPar" parallel processes.
Results from each run are stored on the disk in a local folder "./results_DETS/".

## Gathering results from test runs :

Results recorded in the folder "./results_DETS/" are finally put together to obtain aggregated results over all test runs. A comparison is done with MaDyMos and Opera baselines on the same predictors that were trained in the tsensembler models.
```r
source('gather_results_DETS.R')
```
In the header of the script, it should be checked that the initializations of the variables "dataset.run", "nb.models" and "N.run" match the values entered in the previous steps (i.e. dataset.run <--> datasetName, nb.models <--> nModels, and, N.run <--> nRuns).

## Using Tsensembler/ADE :

The same steps as above can be repeated to perform model ensembling with Tsensembler/ADE method. Run the same R and bash scripts by replacing DETS with ADE in the names of scripts.
The method is much more computationally demanding than DETS though, and we couldn't perform ensembling using more than 16 predictors with our ressources. It remains interesting to test this method though, which has scaling problems but which performs well on a small number of predictors.

## References :

* Cerqueira, Vitor; Torgo, Luis; Oliveira, Mariana, and Bernhard Pfahringer. "Dynamic and Heterogeneous Ensembles for Time Series Forecasting." Data Science and Advanced Analytics (DSAA), 2017 IEEE International Conference on. IEEE, 2017.
* Cerqueira, Vitor; Torgo, Luis; Pinto, Fabio; and Soares, Carlos. "Arbitrated Ensemble for Time Series Forecasting". Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer International Publishing, 2017.
