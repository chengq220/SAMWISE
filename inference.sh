#!/usr/bin/env bash

python inference.py \
    --sam2_version 'med' \
    --input_path data/endovis2017/val1/image \
    --resume output/endovis2017_med/checkpoint0004.pth \
    --text_prompts "Bipolar Forceps" \
    --threshold 0.8 \
    --HSA \
    --multi_class
