#!/bin/sh

model="../AE.1"
data="../TS-DATA-GROUPED"


for ETYPE in glob_avg
do
  for R in 8 16 32
  do
    for EMB in 128 256 512
    do
      current_model_folder="$model/AE-L2-$R-EMB$EMB-$ETYPE"
      if [ -d $current_model_folder ]
      then
        echo "Directory $current_model_folder"
      else
        python train_ae.py -df $data/train --model $current_model_folder -bb linear -r $R -etype $ETYPE -esize $EMB
        python semi_supervised_classification.py -df $data --model $current_model_folder
      fi

      current_model_folder="$model/AE-AL2-$R-EMB$EMB-$ETYPE"
      if [ -d $current_model_folder ]
      then
        echo "Directory $current_model_folder"
      else
        python train_ae.py -df $data/train --model $current_model_folder -bb linear  -r $R -etype $ETYPE -esize $EMB -asym True
        python semi_supervised_classification.py -df $data --model $current_model_folder
      fi
    done
  done
done