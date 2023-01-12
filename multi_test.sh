#!/bin/sh

models="../AE.1/"
data="../cifar-100"

for each in $models/*
do
  echo $each
  python semi_supervised_classification.py --model $each -df $data
done