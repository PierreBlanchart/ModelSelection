#!/bin/bash

cwd=$(pwd);
nDays=$1;
nModels=$2;
nRuns=$3;
datasetName=$4;
nbPar=$5;

fun1 () {
  echo "Starting : run=$1 | day=$2 | nbModels=$3 | dataset=$4";
  #R --slave -f run_DETS_tests_shPar.R --args indday=$2 nbmodels=$3 indrun=$1 dataset=$4;
  echo "Done : run=$1 | day=$2 | nbModels=$3 | dataset=$4";
}
export -f fun1; # export function so that it's inherited by subshells

parallel --jobs $nbPar --wd $cwd --joblog log_parallel_DETS fun1 {1} {2} $nModels $datasetName ::: `seq 1 $nRuns` ::: `seq 1 $nDays`;
