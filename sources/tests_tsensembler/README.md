To launch the following tests, the user needs to install our forked implementation of the tsensembler package.
```r
install.packages(pkgs="./tsensembler2/", repos=NULL)
```
It requires the shell tool "GNU parallel", and a big cluster. A single train and test run of a DETS model with 16 predictors takes about 6 hours on a 48 cores cluster node.

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
source('gather_results.R')
```
In the header of the script, it should be checked that the initializations of the variables "dataset.run", "nb.models" and "N.run" match the values entered in the previous steps (i.e. dataset.run <--> datasetName, nb.models <--> nModels, and, N.run <--> nRuns).

