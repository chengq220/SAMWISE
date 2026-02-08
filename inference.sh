#!/usr/bin/env bash

python inference.py \
    --sam2_version 'med' \
    --input_path data/endovis2017/val1/image \
    --resume output/endovis2017_cme/checkpoint0004.pth \
    --text_prompts "Bipolar Forceps" \
    --threshold 0.7 \
    --HSA \
    --use_cme_head
