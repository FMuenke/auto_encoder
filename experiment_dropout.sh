#!/bin/sh

model="../AE.1/"
data="../cifar-100"


for DROP in 0.10 0.25 0.50 0.75 0.90 0.95
do
  for DSTRUCTURE in general spatial local
  do
    current_model_folder="$model/AE-R2-16-EMB256-$DSTRUCTURE-$DROP"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae.py -df $data/train --model $current_model_folder -bb residual -drop $DROP -drops $DSTRUCTURE
      python semi_supervised_classification.py -df $data --model $current_model_folder
    fi

    current_model_folder="$model/AE-SK-R2-16-EMB256-$DSTRUCTURE-$DROP"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae.py -df $data/train --model $current_model_folder -bb residual --use_skip True  -drop $DROP -drops $DSTRUCTURE
      python semi_supervised_classification.py -df $data --model $current_model_folder
    fi

  done
done
