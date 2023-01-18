#!/bin/sh

model="../AE.PATCH"
data="../TS-DATA-GROUPED"


for DROP in 0.10 0.25 0.50 0.75 0.90 0.95
do
  for DSTRUCTURE in patch_general patch_spatial patch_local general spatial local
  do
    current_model_folder="$model/AE-PR2.2-16-EMB256-$DSTRUCTURE-$DROP"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae.py -df $data/train --model $current_model_folder -s 2 -bb patch-residual -drop $DROP -drops $DSTRUCTURE
      python semi_supervised_classification.py -df $data --model $current_model_folder
    fi

  done
done


for DROP in 0.10 0.25 0.50 0.75 0.90 0.95
do
  for DSTRUCTURE in patch_general patch_spatial patch_local general spatial local
  do
    current_model_folder="$model/AE-PR2-16-EMB256-$DSTRUCTURE-$DROP"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae.py -df $data/train --model $current_model_folder -bb patch-residual -drop $DROP -drops $DSTRUCTURE
      python semi_supervised_classification.py -df $data --model $current_model_folder
    fi

  done
done