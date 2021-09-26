#!/bin/bash
dt=aids
runmode=cross_val
match_type=homo
embed_type=freq
pool_type=att
model_type=GIN


python ../active_train.py \
  --dataset ${dt} \
  --mode ${runmode} \
  --matching ${match_type} \
  --embed_type ${embed_type} \
  --pool_type ${pool_type} \
  --model_type ${model_type} > ../${dt}_${embed_type}_att.txt


python ../active_train.py \
  --dataset ${dt} \
  --mode ${runmode} \
  --matching ${match_type} \
  --embed_type ${embed_type} \
  --pool_type ${pool_type} \
  --model_type GAT  > ../${dt}_${embed_type}_GAT.txt

python ../active_train.py \
  --dataset ${dt} \
  --mode ${runmode} \
  --matching ${match_type} \
  --embed_type ${embed_type} \
  --pool_type ${pool_type} \
  --model_type SAGE  > ../${dt}_${embed_type}_SAGE.txt


python ../active_train.py \
  --dataset ${dt} \
  --mode ${runmode} \
  --matching ${match_type} \
  --embed_type ${embed_type} \
  --pool_type ${pool_type} \
  --model_type GCN > ../${dt}_${embed_type}_GCN.txt



python ../active_train.py \
  --dataset ${dt} \
  --mode ${runmode} \
  --matching ${match_type} \
  --embed_type ${embed_type} \
  --pool_type mean \
  --model_type ${model_type}  > ../${dt}_${embed_type}_mean.txt


python ../active_train.py \
  --dataset ${dt} \
  --mode ${runmode} \
  --matching ${match_type} \
  --embed_type ${embed_type} \
  --pool_type sum \
  --model_type ${model_type}  > ../${dt}_${embed_type}_sum.txt


python ../active_train.py \
  --dataset ${dt} \
  --mode ${runmode} \
  --matching ${match_type} \
  --embed_type ${embed_type} \
  --pool_type max \
  --model_type ${model_type}  > ../${dt}_${embed_type}_max.txt


