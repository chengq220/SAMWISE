#!/usr/bin/env bash

python inference.py \
    --sam2_version 'med' \
    --input_path data/endovis2017/val1/image \
    --resume output/endovis2017_med/checkpoint0004.pth \
    --num_frames 8 \
    --threshold 0.5 \
    --HSA \
    --use_cme_head \
    --create_video \
    --video_name 'all'
