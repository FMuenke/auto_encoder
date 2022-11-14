#!/bin/sh

model="../AE-RESIDUAL.6/"

data="../cifar-100"


for TASK in blurring denoise completion_cross_cut reconstruction_shuffled completion_blackhole completion_masking reconstruction_rotation
do
  for TASKDIFF in 0.10 0.25 0.50 0.75 0.90
  do
    current_model_folder="$model/AE-R2-16-EMB256-$TASK-$TASKDIFF"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae.py -df $data/train --model $current_model_folder --task $TASK  --task_difficulty $TASKDIFF --depth 2 --resolution 16 --embedding_size 256
      python semi_supervised_classification.py -df $data --model $current_model_folder
    fi
  done
done


for TASK in blurring denoise completion_cross_cut reconstruction_shuffled completion_blackhole completion_masking reconstruction_rotation
do
  for TASKDIFF in 0.10 0.25 0.50 0.75 0.90
  do
    current_model_folder="$model/AE-R4-4-EMB256-$TASK-$TASKDIFF"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae.py -df $data/train --model $current_model_folder --task $TASK  --task_difficulty $TASKDIFF --depth 4 --resolution 4 --embedding_size 256
      python semi_supervised_classification.py -df $data --model $current_model_folder
    fi
  done
done