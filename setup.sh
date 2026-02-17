#!/usr/bin/env bash

# Update system
apt update
apt install tmux

#Install requirements
pip install -r requirements.txt

#Download data
mkdir data
cd data
gdown --fuzzy "https://drive.google.com/file/d/1vkJbJ7JSQrq--Qze-E2_WL1xJeaLWsZx/view?usp=sharing"
tar -xvf VOS-Endovis17.tar
rm VOS-Endovis17.tar
cd ..

#Download pretrain-models
mkdir pretrain
cd pretrain
wget "https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-fairseq.tar.gz"
tar -xzvf RoBERTa-base-PM-fairseq.tar.gz --no-same-owner
rm RoBERTa-base-PM-fairseq.tar.gz

# sam2.1_hiera_s_endo18.pth
# gdown --fuzzy "https://drive.google.com/file/d/1DyrrLKst1ZQwkgKM7BWCCwLxSXAgOcMI/view?usp=sharing"

# MedSAM2 
wget "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
