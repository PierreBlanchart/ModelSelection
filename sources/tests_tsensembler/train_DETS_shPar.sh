#!/bin/bash

cwd=$(pwd);
nModels=$1;
nRuns=$2;
datasetName=$3;
nbPar=$4;

fun1 () {
  echo "Starting : run=$1 | nbModels=$2 | dataset=$3";
  R --slave -f train_DETS_shPar.R --args nbmodels=$2 indrun=$1 dataset=$3;
  echo "Done : run=$1 | nbModels=$2 | dataset=$3";
}
export -f fun1; # export function so that it's inherited by subshells

parallel --jobs $nbPar --wd $cwd --joblog log_parallel_DETS fun1 {1} $nModels $datasetName ::: `seq 1 $nRuns`;
