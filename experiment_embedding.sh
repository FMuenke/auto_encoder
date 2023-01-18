#!/bin/sh

model="../AE.1"
data="../TS-DATA-GROUPED"



for ETYPE in glob_avg flatten glob_max
do
  for EACT in leaky_relu relu softmax sigmoid
  do
    current_model_folder="$model/AE-R2-16-EMB256-$ETYPE-$EACT"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae.py -df $data/train --model $current_model_folder -bb residual -etype $ETYPE -eact $EACT
      python semi_supervised_classification.py -df $data --model $current_model_folder
    fi
  done
done