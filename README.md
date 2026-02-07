# Surgical Video Segmentation

## Install Dependencies
```
pip install -r requirements.txt
```

## Dataset
```
mkdir data
cd data
gdown --fuzzy <google drive link>
tar -xvf <filename>
```

## Download Pretrain Weights
```
mkdir pretrain
cd pretrain
wget "https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-fairseq.tar.gz"
tar -xzvf <filename> --no-same-owner
```

## Running scipts
###Training Command
```
python main.py \
    --dataset endovis2017 \
    --name_exp endovis2017_cme \
    --num_frames 8 \
    --max_skip 1 \
    --epochs 5 \
    --batch_size 2  \
    --HSA \
    --use_cme_head
```

### Evaluation Command 
```
python evaluatation.py \
    --model output/demo/model.pth \
    --file_path data/endovis2017/val1 \
    --num_frames 8 \ 
    --HSA \
    --use_cme_head
```

### Inference Command
```
python inference.py \
    --input_path data/endovis2017/val1/image \
    --resume output/endovis2017_all/checkpoint0004.pth \
    --text_prompts "<Prompt>" \
    --HSA \
    --use_cme_head
```
