#!/bin/bash

# Define parameters to tweak
batch_sizes=(16 32 64)
n_tokens=(12 32 64 128 256)
n_layers=(2 4 6 8 10)
n_heads=(2 4 6 8 10)
d_models=(32 64 128 256 512)

# Define other parameters
use_lr_decay=True
dataset_path='./datasets/shakespear_corpus.txt'
max_iter=1
val_int=25
cross_val=True
k_fold=20
save=True
save_int=50

# Iterate over combinations
for batch_size in "${batch_sizes[@]}"; do
  for n_token in "${n_tokens[@]}"; do
    for n_layer in "${n_layers[@]}"; do
      for n_head in "${n_heads[@]}"; do
        for d_model in "${d_models[@]}"; do
          name="Shakespear_b${batch_size}_t${n_token}_l${n_layer}_h${n_head}_d${d_model}"
          python3 ./train.py --batch_size="$batch_size" \
                             --n_tokens="$n_token" \
                             --n_layers="$n_layer" \
                             --n_heads="$n_head" \
                             --d_model="$d_model" \
                             --use_lr_decay="$use_lr_decay" \
                             --dataset_path="$dataset_path" \
                             --max_iter="$max_iter" \
                             --val_int="$val_int" \
                             --cross_val="$cross_val" \
                             --k_fold="$k_fold" \
                             --save="$save" \
                             --save_int="$save_int" \
                             --name="$name"
        done
      done
    done
  done
done