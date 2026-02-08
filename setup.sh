#!/usr/bin/env bash

#Install requirements
pip install -r requirements.txt

#Download data
mkdir data
cd data
gdown --fuzzy "https://drive.google.com/file/d/1XG_lqhQZHBBHb1WM9s9TTu9Gbd4JbND7/view?usp=drive_link"
tar -xvf endovis2017.tar
rm endovis2017.tar
cd ..

#Download pretrain-models
mkdir pretrain
cd pretrain
wget "https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-fairseq.tar.gz"
tar -xzvf RoBERTa-base-PM-fairseq.tar.gz --no-same-owner
rm RoBERTa-base-PM-fairseq.tar.gz
