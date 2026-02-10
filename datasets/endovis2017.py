"""
Endovis2017 data loader
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset
import datasets.transforms_video as T
import os
from PIL import Image

import numpy as np
import random
from collections import defaultdict

from datasets.categories import endovis2017_category_rev_dict as rev_category_dict, \
    endovis2017_category_descriptor_dict as descriptor

class EndoVis2017Dataset(Dataset):
    def __init__(self, img_folder: Path, transforms,
                num_frames: int, max_skip: int):
        self.img_folder = img_folder
        self._transforms = transforms
        self.num_frames = num_frames
        self.max_skip = max_skip
        self.available_classes = list(rev_category_dict.keys())
        # create video meta data
        self.prepare_metas()

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))
        print('\n')

    def prepare_metas(self):
        self.videos = defaultdict(list)
        video_list = list(Path(os.path.join(self.img_folder, "image")).glob("*"))

        for vid in video_list:
            vid_name = vid.name
            vid_num = vid_name.split("_")[1]
            self.videos[vid_num].append(vid)
        self.metas = []
        for vid_key in self.videos.keys():
            vid_frames = sorted(list(self.videos[vid_key])) # gets the files in video idx and sort them in order
            vid_len = len(vid_frames)
            for frame_id in range(0, vid_len, self.num_frames): # Get each frame
                cur_frame = vid_frames[frame_id]
                frame_name = str(cur_frame).split("/")[-1]
                mask_path = os.path.join(str(self.img_folder), 'label', frame_name)
                mask = Image.open(mask_path).convert('P')
                mask = np.array(mask) 
                category = np.unique(mask)
                category = category[category > 0] # positive classes
                for cls in category:
                    meta = {}
                    meta['video'] = vid_key
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    meta['cls'] = (cls, 1)
                    self.metas.append(meta)
                negative = [cls for cls in self.available_classes if cls not in set(category)]
                neg_cls = random.sample(negative, 1)
                for cls in neg_cls: # negative classes
                    meta = {}
                    meta['video'] = vid_key
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    meta['cls'] = (cls, 0)
                    self.metas.append(meta)
                    
    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]  # dict

            video, frames, frame_id, cls_tuple = \
                        meta['video'], meta['frames'], meta['frame_id'], meta['cls']

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
            imgs, boxes, masks, valid = [], [], [], []
            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                frame_path = frames[frame_indx]
                frame_name = str(frame_path).split('/')[-1]
                img_path = os.path.join(str(self.img_folder), 'image', frame_name)
                mask_path = os.path.join(str(self.img_folder), 'label', frame_name)
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('P')
                # create the target
                mask = np.array(mask)
                
                cls, presence = cls_tuple

                mask = (mask==cls).astype(np.float32) 
                if (mask > 0).any():
                    y1, y2, x1, x2 = self.bounding_box(mask)
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valid.append(1)
                else: # some frame didn't contain the instance
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                    valid.append(0)
                mask = torch.from_numpy(mask)

                # append
                imgs.append(img)
                masks.append(mask)
                boxes.append(box)

            # transform
            w, h = img.size
            boxes = torch.stack(boxes, dim=0)
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            masks = torch.stack(masks, dim=0)
 
            if presence == 0: # negative sample
                cap = f"no instance of {rev_category_dict.get(cls, 'other')}"
            else:
                descriptions = list(descriptor.get(cls, []))
                cap = f"{rev_category_dict.get(cls, 'other')} with {random.choice(descriptions)}"

            target = {
                'frames_idx': torch.tensor(sample_indx), # [T,]
                'boxes': boxes,                          # [T, 4], xyxy
                'masks': masks,                          # [T, H, W]
                'valid': torch.tensor(valid),            # [T,]
                'orig_size': torch.as_tensor([int(h), int(w)]),
                'size': torch.as_tensor([int(h), int(w)]),
                'caption': cap,
            }

            # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
            imgs, target = self._transforms(imgs, target)
            imgs = torch.stack(imgs, dim=0) # [T, 3, H, W]

            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return imgs, target

def make_transforms():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # we need augumentation for training or do we
    return normalize

def build(image_set, args):
    root = Path(args.endovis2017)
    assert root.exists(), f'provided path {root} does not exist'
    PATHS = {
        "train": (root / "train"),
        "val": (root / args.split),
    }
    img_folder = PATHS[image_set]
    dataset = EndoVis2017Dataset(img_folder, transforms=make_transforms(),
                                num_frames=args.num_frames, max_skip=args.max_skip)
    
    return dataset
