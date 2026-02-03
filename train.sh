#!/usr/bin/env bash

python main.py \
    --dataset endovis2017 \
    --num_frames 5 \
    --max_skip 1 \
    --all True \
    --epochs 5 \
    --batch_size 2 \