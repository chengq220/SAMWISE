import py3_wget
import os
import sys


SAM2_WEIGHTS_URL = {
    'tiny':  'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt', 
    'base':  'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt', 
    'large': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt'
}
SAM2_PATHS_CONFIG = {
    'tiny':  ('pretrain/sam2_hiera_tiny.pt', 'sam2_configs/sam2_hiera_t.yaml'),
    'base':  ('pretrain/sam2_hiera_base_plus.pt', 'sam2_configs/sam2_hiera_b+.yaml'),
    'large': ('pretrain/sam2_hiera_large.pt', 'sam2_configs/sam2_hiera_l.yaml')
}

# ROBERTA_WEIGHTS_URL = 'https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz' # General text encoder 
# ROBERTA_WEIGHTS_PATH = 'pretrain/roberta.base'

ROBERTA_WEIGHTS_URL = 'https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-fairseq.tar.gz' # Medical text encoder
ROBERTA_WEIGHTS_PATH = 'RoBERTa-base-PM/RoBERTa-base-PM-fairseq'

def get_roberta_weights():
    print(f"Downloading Roberta Base..")
    py3_wget.download_file(ROBERTA_WEIGHTS_URL, ROBERTA_WEIGHTS_PATH+'.tar.gz')
    print(f"Extracting Roberta Base weights...")
    cmd = 'cd pretrain && tar -xzvf pretrain/RoBERTa-base-PM.tar.gz --no-same-owner'
    ret = os.system(cmd)
    if ret != 0:
        print('Something went wrong untarring Roberta weights, exiting...')
        sys.exit(ret)
