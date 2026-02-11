#!/usr/bin/env bash

python main.py \
    --dataset endovis2017 \
    --sam2_version 'med' \
    --name_exp endovis2017_med \
    --num_frames 8 \
    --max_skip 1 \
    --epochs 5 \
    --batch_size 2  \
    --HSA \
    --lr 1e-5
