#!/bin/sh

model="../AE-RESIDUAL.9/"

data="../cifar-100"


for D in 2
do
  for R in 16
  do
    for EMB in 256
    do
      for NOISE in 0.10 0.25 0.50 0.75 0.90 0.95
      do
        current_model_folder="$model/AE-R$D-$R-EMB$EMB-N$NOISE"
        if [ -d $current_model_folder ]
        then
          echo "Directory $current_model_folder"
        else
          python train_ae.py -df $data/train --model $current_model_folder --depth $D --resolution $R --embedding_size $EMB -bb residual -noise $NOISE
          python semi_supervised_classification.py -df $data --model $current_model_folder
        fi
      done
    done
  done
done