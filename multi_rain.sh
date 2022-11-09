#!/bin/sh

model="../AE-RESIDUAL.4/"

data="../cifar-100"


for TASK in reconstruction_shuffled completion_blackhole completion_masking completion_cross_cut
do
  for TASKDIFF in 0.10 0.25 0.50 0.75 0.9
  do
    current_model_folder="$model/VAL-AE-R2-16-EMB256-$TASK-$TASKDIFF"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae.py -df $data/train --model $current_model_folder --task $TASK  --task_difficulty $TASKDIFF --depth 2 --resolution 16 --embedding_size 256
      python semi_supervised_classification.py -df $data --model $current_model_folder
    fi
  done
done


for TASK in completion_blackhole completion_cross_cut reconstruction_shuffled
do
  for TASKDIFF in 0.10 0.25 0.50 0.75 0.9
  do
    current_model_folder="$model/VAL-AE-R4-16-EMB32-$TASK-$TASKDIFF"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae.py -df $data/train --model $current_model_folder --task $TASK  --task_difficulty $TASKDIFF --depth 4 --resolution 16 --embedding_size 256
      python semi_supervised_classification.py -df $data --model $current_model_folder
    fi
  done
done