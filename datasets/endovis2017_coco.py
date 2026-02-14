"""
Endovis2017 data loader
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset
import datasets.transforms_video as T
import torchvision.transforms as TF
import os
from PIL import Image
import json
import numpy as np
import random
from collections import defaultdict

from datasets.categories import endovis2017_category_rev_dict as rev_category_dict, \
    endovis2017_category_descriptor_dict as descriptor

class EndoVis2017Dataset(Dataset):
    def __init__(self, img_folder: Path, ann_file: Path,
                num_frames: int, max_skip: int):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._img_transforms = make_transforms()
        self._mask_transforms = make_transforms(isMask=True)
        self.num_frames = num_frames
        self.max_skip = max_skip
        self.available_classes = list(rev_category_dict.keys())
        # create video meta data
        self.prepare_metas()

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))
        print('\n')

    def prepare_metas(self):
            self.videos = defaultdict(list)
            with open(str(self.ann_file), 'r') as f:
                coco_data = json.load(f)
            self.coco_img = [img['file_name'] for img in coco_data['images']]
            coco_id = [img['id'] for img in coco_data['images']]
            for img, cc_id in zip(self.coco_img, coco_id):
                vid_id = str(img).split('_')[0][-1]
                vid = (cc_id, img)
                self.videos[vid_id].append(vid)
            self.annoation_dict = defaultdict(list)
            annotations = [ann for ann in coco_data['annotations']]
            for ann in annotations:
                img_id = ann['image_id']
                self.annoation_dict[img_id].append(ann)
            self.metas = []
            for vid_key in self.videos.keys():
                vid_frames = sorted(list(self.videos[vid_key]), key=lambda x: x[1] )
                vid_len = len(vid_frames)
                for frame_id in range(0, vid_len, self.num_frames):
                    id, frame_name = vid_frames[frame_id]
                    for inst in self.annoation_dict[id]: 
                        meta = {}
                        meta['video'] = vid_key
                        meta['frames'] = vid_frames
                        meta['frame_id'] = frame_id
                        meta['bbox'] = inst['bbox']
                        meta['category'] = inst['category_id']
                        self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]  # dict

            video, frames, frame_id, cls = \
                        meta['video'], meta['frames'], meta['frame_id'], meta['category']

            vid_len = len(frames)
            num_frames = self.num_frames
            # random sparse sample
            sample_indx = [frame_id]
            # local sample
            sample_id_before = random.randint(1, 3)
            sample_id_after = random.randint(1, 3)
            local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
            sample_indx.extend(local_indx)

            # global sampling
            if num_frames > 3:
                all_inds = list(range(vid_len))
                global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                global_n = num_frames - len(sample_indx)
                if len(global_inds) > global_n:
                    select_id = random.sample(range(len(global_inds)), global_n)
                    for s_id in select_id:
                        sample_indx.append(global_inds[s_id])
                elif vid_len >=global_n:  # sample long range global frames
                    select_id = random.sample(range(vid_len), global_n)
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
                else:
                    select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
            sample_indx.sort()

            # read frames and masks
            imgs, masks, valid = [], [], []
            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                cc_id, frame_name = frames[frame_indx]

                img_path = os.path.join(str(self.img_folder), 'images', frame_name)
                mask_path = os.path.join(str(self.img_folder), 'annotations', 'images', frame_name)

                img = Image.open(img_path).convert('RGB')
                img = self._img_transforms(img)

                mask = Image.open(mask_path).convert('P')
                mask = self._mask_transforms(mask)

                # create the target
                mask = (mask==cls).float()
                
                if (mask > 0).any():
                    valid.append(1)
                else:
                    valid.append(0)

                # append
                imgs.append(img)
                masks.append(mask)

            # transform

            imgs = torch.stack(imgs, dim=0) # [T, 3, H, W]s
            t, c, w, h = imgs.shape
            masks = torch.stack(masks, dim=0)
 
            descriptions = list(descriptor.get(cls, []))
            cap = f"{rev_category_dict.get(cls, 'other')} with {random.choice(descriptions)}"

            target = {
                'frames_idx': torch.tensor(sample_indx), # [T,]
                'masks': masks,                          # [T, H, W]
                'valid': torch.tensor(valid),            # [T,]
                'orig_size': torch.as_tensor([int(h), int(w)]),
                'size': torch.as_tensor([int(h), int(w)]),
                'caption': cap,
            }

            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return imgs, target

def make_transforms(isMask = False, max_size= 1024):
    if isMask:
        return TF.Compose([
            TF.CenterCrop(max_size),
            TF.ToTensor()
        ])
    
    return TF.Compose([
        TF.CenterCrop(max_size),
        TF.ToTensor(),
        TF.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def build(image_set, args):
    root = Path(args.endovis2017)
    assert root.exists(), f'provided path {root} does not exist'
    PATHS = {
        "train": (root),
        "val": (root),
    }
    img_folder = PATHS[image_set]
    dataset = EndoVis2017Dataset(img_folder, os.path.join(img_folder, "annotations/Fold0/train.json"),
                                num_frames=args.num_frames, max_skip=args.max_skip)
    
    return dataset
