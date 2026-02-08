#!/usr/bin/env bash

python main.py \
    --dataset endovis2017 \
    --name_exp endovis2017_cme \
    --num_frames 8 \
    --max_skip 1 \
    --epochs 5 \
    --batch_size 2  \
    --HSA \
    --use_cme_head
