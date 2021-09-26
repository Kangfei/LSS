#!/bin/bash
dt=yeast

python ../active_train.py \
  --dataset ${dt} \
  --mode ensemble \
  --ensemble_num 5 \
  --active_iters 5 \
  --epochs 50 \
  --active_epochs 50 > ../${dt}_ensemble.txt

