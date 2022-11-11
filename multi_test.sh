#!/bin/sh

models="../../ai_models/AE-RESIDUAL.4/"
data="../../datasets/cifar-100"

for each in $models/*
do
  echo $each
  python semi_supervised_classification.py --model $each -df $data
done