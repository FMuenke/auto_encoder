#!/bin/sh

model="../AE.1"
data="../TS-DATA-GROUPED"


for ETYPE in glob_avg
do
  for D in 2
  do
    for R in 1 8 16 32
    do
      for EMB in 128 256 512
      do
        current_model_folder="$model/AE-R$D-$R-EMB$EMB-$ETYPE"
        if [ -d $current_model_folder ]
        then
          echo "Directory $current_model_folder"
        else
          echo "GO"
          python train_ae.py -df $data/train --model $current_model_folder -bb residual -d $D -r $R -etype $ETYPE -esize $EMB
          python semi_supervised_classification.py -df $data --model $current_model_folder
        fi

        current_model_folder="$model/AE-AR$D-$R-EMB$EMB-$ETYPE"
        if [ -d $current_model_folder ]
        then
          echo "Directory $current_model_folder"
        else
          echo "GO"
          python train_ae.py -df $data/train --model $current_model_folder -bb residual -d $D  -r $R -etype $ETYPE -esize $EMB -asym True
          python semi_supervised_classification.py -df $data --model $current_model_folder
        fi
      done
    done
  done
done


for ETYPE in glob_avg
do
  for D in 4
  do
    for R in 1 2 4 8
    do
      for EMB in 128 256 512
      do
        current_model_folder="$model/AE-R$D-$R-EMB$EMB-$ETYPE"
        if [ -d $current_model_folder ]
        then
          echo "Directory $current_model_folder"
        else
          echo "GO"
          python train_ae.py -df $data/train --model $current_model_folder -bb residual -d $D --resolution $R -type $ETYPE -size $EMB
          python semi_supervised_classification.py -df $data --model $current_model_folder
        fi

        current_model_folder="$model/AE-AR$D-$R-EMB$EMB-$ETYPE"
        if [ -d $current_model_folder ]
        then
          echo "Directory $current_model_folder"
        else
          echo "GO"
          python train_ae.py -df $data/train --model $current_model_folder -bb residual -d $D  --resolution $R -type $ETYPE -size $EMB -asym True
          python semi_supervised_classification.py -df $data --model $current_model_folder
        fi
      done
    done
  done
done