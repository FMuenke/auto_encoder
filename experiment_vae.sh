#!/bin/sh

model="../AE.1/"
data="../cifar-100"



for R in 8 16 32
do
  for EMB in 128 256
  do
    current_model_folder="$model/VAE-R2-$R-EMB$EMB"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae.py -df $data/train --model $current_model_folder -bb residual --resolution $R --type variational-autoencoder -size $EMB
      python semi_supervised_classification.py -df $data --model $current_model_folder
    fi

    current_model_folder="$model/VAE-L2-$R-EMB$EMB"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae.py -df $data/train --model $current_model_folder -bb linear --resolution $R --type variational-autoencoder -size $EMB
      python semi_supervised_classification.py -df $data --model $current_model_folder
    fi
  done
done