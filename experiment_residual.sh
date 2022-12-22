#!/bin/sh

model="../AE.1/"
data="../cifar-100"


for ETYPE in flatten glob_avg glob_max
do
  for R in 8 16 32
  do
    for EMB in 128 256 512
    do
      current_model_folder="$model/AE-R2-$R-EMB$EMB-$ETYPE"
      if [ -d $current_model_folder ]
      then
        echo "Directory $current_model_folder"
      else
        python train_ae.py -df $data/train --model $current_model_folder -bb residual --resolution $R -type $ETYPE -size $EMB
        python semi_supervised_classification.py -df $data --model $current_model_folder
      fi

      current_model_folder="$model/AE-AR2-$R-EMB$EMB-$ETYPE"
      if [ -d $current_model_folder ]
      then
        echo "Directory $current_model_folder"
      else
        python train_ae.py -df $data/train --model $current_model_folder -bb residual  --resolution $R -type $ETYPE -size $EMB -asym True
        python semi_supervised_classification.py -df $data --model $current_model_folder
      fi
    done
  done
done