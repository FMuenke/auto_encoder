#!/bin/sh

model="../AE.1/"
data="../cifar-100"



current_model_folder="$model/AE-R2-16-EMB256-D0.90-local"
if [ -d $current_model_folder ]
then
  echo "Directory $current_model_folder"
else
  python train_ae.py -df $data/train --model $current_model_folder -d 4 -drop 0.90 -drops local --task denoise --task_difficulty 0.90
  python semi_supervised_classification.py -df $data --model $current_model_folder
fi

current_model_folder="$model/AE-R4-32-EMB512-D0.90-local"
if [ -d $current_model_folder ]
then
  echo "Directory $current_model_folder"
else
  python train_ae.py -df $data/train --model $current_model_folder -d 4 -r 32 -size 512 -drop 0.90 -drops local --task denoise --task_difficulty 0.90
  python semi_supervised_classification.py -df $data --model $current_model_folder
fi