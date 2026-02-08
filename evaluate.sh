#!/usr/bin/env bash

python evaluation.py \
    --sam2_version 'med' \
    --model output/demo/model.pth \
    --file_path data/endovis2017/val1 \
    --num_frames 8 \ 
    --HSA \
    --use_cme_head
