#!/bin/sh

model="../AE.3/"
data="../cifar-100"


for ETYPE in glob_avg
do
  for R in 8 16 32
  do
    for EMB in 128 256 512
    do
      current_model_folder="$model/AE-R4-$R-EMB$EMB-$ETYPE"
      if [ -d $current_model_folder ]
      then
        echo "Directory $current_model_folder"
      else
        python train_ae.py -df $data/train --model $current_model_folder -bb residual -d 4 --resolution $R -type $ETYPE -size $EMB
        python semi_supervised_classification.py -df $data --model $current_model_folder
      fi

      current_model_folder="$model/AE-AR4-$R-EMB$EMB-$ETYPE"
      if [ -d $current_model_folder ]
      then
        echo "Directory $current_model_folder"
      else
        python train_ae.py -df $data/train --model $current_model_folder -bb residual -d 4  --resolution $R -type $ETYPE -size $EMB -asym True
        python semi_supervised_classification.py -df $data --model $current_model_folder
      fi
    done
  done
done