#!/bin/bash
dt=aids
uncertainties=( entropy margin confident consist random )

for uncertainty in ${uncertainties[@]}; do
  python ../active_train.py \
  --dataset ${dt} \
  --mode train \
  --embed_type freq \
  --active_iters 2 \
  --active_epochs 50 \
  --uncertainty ${uncertainty} > ../${dt}_none_${uncertainty}.txt
done
