#!/usr/bin/env bash

python evaluation.py \
    --pred_root ./output/all/pred/ \
    --gt_root ./data/endovis2018/valid/Annotations
