#!/bin/sh

model="../AE-RESIDUAL.7/"

data="../cifar-100"


for D in 1 2 4
do
  for R in 2 4 8 16
  do
    for EMB in 32 64 128 256
    do
      current_model_folder="$model/AE-R$D-$R-EMB$EMB"
      if [ -d $current_model_folder ]
      then
        echo "Directory $current_model_folder"
      else
        python train_ae.py -df $data/train --model $current_model_folder --depth $D --resolution $R --embedding_size $EMB -bb residual --type autoencoder
        python semi_supervised_classification.py -df $data --model $current_model_folder
      fi
    done
  done
done


for D in 1 2 4
do
  for R in 2 4 8 16
  do
    for EMB in 32 64 128 256
    do
      current_model_folder="$model/AE-L$D-$R-EMB$EMB"
      if [ -d $current_model_folder ]
      then
        echo "Directory $current_model_folder"
      else
        python train_ae.py -df $data/train --model $current_model_folder --depth $D --resolution $R --embedding_size $EMB -bb linear --type autoencoder
        python semi_supervised_classification.py -df $data --model $current_model_folder
      fi
    done
  done
done


for D in 1 2 4
do
  for R in 2 4 8 16
  do
    for EMB in 32 64 128 256
    do
      current_model_folder="$model/VAE-R$D-$R-EMB$EMB"
      if [ -d $current_model_folder ]
      then
        echo "Directory $current_model_folder"
      else
        python train_ae.py -df $data/train --model $current_model_folder --depth $D --resolution $R --embedding_size $EMB -bb residual --type variational-autoencoder
        python semi_supervised_classification.py -df $data --model $current_model_folder
      fi
    done
  done
done