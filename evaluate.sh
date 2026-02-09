#!/usr/bin/env bash

python evaluation.py \
    --sam2_version 'med' \
    --model output/endovis2017_med/checkpoint0002.pth \
    --file_path data/endovis2017/val1 \
    --num_frames 8 \
    --use_cme_head \
    --HSA 
