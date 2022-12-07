#!/bin/sh

model="../AE-RESIDUAL.9/"

data="../cifar-100"


for D in 1 2 4
do
  for R in 4 8 16
  do
    for EMB in 128 256 512
    do
      current_model_folder="$model/AE-V$D-$R-EMB$EMB"
      if [ -d $current_model_folder ]
      then
        echo "Directory $current_model_folder"
      else
        python train_ae.py -df $data/train --model $current_model_folder --depth $D --resolution $R --embedding_size $EMB -bb vit
        python semi_supervised_classification.py -df $data --model $current_model_folder
      fi
    done
  done
done


for D in 1 2 4
do
  for R in 4 8 16
  do
    for EMB in 128 256 512
    do
      current_model_folder="$model/AE-AV$D-$R-EMB$EMB"
      if [ -d $current_model_folder ]
      then
        echo "Directory $current_model_folder"
      else
        python train_ae.py -df $data/train --model $current_model_folder -aysm True --depth $D --resolution $R --embedding_size $EMB -bb vit
        python semi_supervised_classification.py -df $data --model $current_model_folder
      fi
    done
  done
done