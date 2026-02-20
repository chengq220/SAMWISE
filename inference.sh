#!/usr/bin/env bash

python inference.py \
    --sam2_version 'med' \
    --input_path data/endovis2018/valid/JPEGImages/seq1 \
    --mask_input data/endovis2018/valid/VOS/seq1 \
    --resume output/endovis2017_med/checkpoint0002.pth \
    --text_prompts 'all' \
    --num_frames 8 \
    --threshold 0.5 \
    --HSA \
    --use_cme_head 
