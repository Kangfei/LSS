#!/bin/bash
dt=eu2005-syn
embed_types=( prone prone_concat )

for e_type in ${embed_types[@]}; do
  python ../active_train.py \
  --mode pretrain \
  --dataset ${dt} \
  --embed_type ${e_type} \
  --epochs 80
done