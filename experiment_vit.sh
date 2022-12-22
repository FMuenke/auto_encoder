#!/bin/sh

model="../AE.1/"
data="../cifar-100"


for R in 4 8 16
do
  for EMB in 128 256
  do
    current_model_folder="$model/AE-V2-$R-EMB$EMB"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae.py -df $data/train --model $current_model_folder -bb vit --resolution $R -size $EMB
      python semi_supervised_classification.py -df $data --model $current_model_folder
    fi

    current_model_folder="$model/AE-V2-$R-EMB$EMB-D0.75-local"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae.py -df $data/train --model $current_model_folder -bb vit --resolution $R -size $EMB -drop 0.75 -drops local
      python semi_supervised_classification.py -df $data --model $current_model_folder
    fi

    current_model_folder="$model/AE-AV2-$R-EMB$EMB-D0.75-local"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae.py -df $data/train --model $current_model_folder -asym True -bb vit --resolution $R -size $EMB -drop 0.75 -drops local
      python semi_supervised_classification.py -df $data --model $current_model_folder
    fi
  done
done