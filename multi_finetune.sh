#!/bin/sh

models="../AE.1/"
data="../cifar-100"

for each in $models/*
do
  for B in "b-residual" "residual"
  do
    echo $each
    echo $(basename $each)
    current_model_folder="$models/CLF.AUG.10k.FR.$B.$(basename $each)/"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae_clf.py --model $current_model_folder -df $data -ae $each -n 10000 -aug all -freeze True -bb $B
      python test_clf.py -df $data --model $current_model_folder
    fi

    current_model_folder="$models/CLF.AUG.10k.NFR.$B.$(basename $each)/"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae_clf.py --model $current_model_folder -df $data -ae $each -n 10000 -aug all -bb $B
      python test_clf.py -df $data --model $current_model_folder
    fi
  done
done