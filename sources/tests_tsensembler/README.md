## Training DETS models :


```sh
bash train_DETS_shPar.sh nRuns nModels datasetName nPar
```
The above command launches the parallel training of "Nruns" DETS models containing "nModels" predictors on the dataset "datasetName", using "nPar" parallel processes.
Models are saved on the disk in a local folder "./models_DETS/".


## Predicting DETS models :

```sh
bash run_DETS_tests_shPar.sh nRuns nDays nbModels datasetName nPar
```
The above command launches the parallel testing of "Nruns" DETS models containing "nbModels" predictors on the dataset "datasetName" containing "nDays" test days, using "nPar" parallel processes.

