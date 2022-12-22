#!/bin/sh

model="../AE.1/"
data="../cifar-100"


for NOISE in 0.05 0.10 0.15 0.20 0.25 0.50 0.75 0.90
do
  current_model_folder="$model/AE-R2-16-EMB256-N$NOISE"
  if [ -d $current_model_folder ]
  then
    echo "Directory $current_model_folder"
  else
    python train_ae.py -df $data/train --model $current_model_folder -bb residual -noise $NOISE
    python semi_supervised_classification.py -df $data --model $current_model_folder
  fi
done
