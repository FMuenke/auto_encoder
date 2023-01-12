#!/bin/sh

model="../AE.TASK/"
data="../cifar-100"

# completion_cross_cut denoise completion_masking reconstruction_shuffled reconstruction_rotated warp
for TASK in imagine_patches imagine_mask
do
  for TDIFF in 0.10 0.25 0.50 0.75 0.90
  do
    current_model_folder="$model/AE-R2-16-EMB256-$TASK-$TDIFF"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae.py -df $data/train --model $current_model_folder -bb residual --task $TASK --task_difficulty $TDIFF
      python semi_supervised_classification.py -df $data --model $current_model_folder
    fi
  done
done


for TASK in imagine_patches imagine_mask
do
  for TDIFF in 0.10 0.25 0.50 0.75 0.90
  do
    current_model_folder="$model/AE-R4-16-EMB256-$TASK-$TDIFF"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      python train_ae.py -df $data/train --model $current_model_folder -d 4 -bb residual --task $TASK --task_difficulty $TDIFF
      python semi_supervised_classification.py -df $data --model $current_model_folder
    fi
  done
done