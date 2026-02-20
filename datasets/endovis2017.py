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

from datasets.categories import endovis2017_category_rev_dict as rev_category_dict, \
    endovis2017_category_descriptor_dict as descriptor, endovis2018_category_verb_dict as verb_dict

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
        self.videos = []
        video_list = list(Path(os.path.join(self.img_folder, "JPEGImages")).glob("*"))
        for vid in video_list:
            vid_name = vid.name
            self.videos.append(vid_name)
        self.metas = []
        for vid_name in self.videos:
            vid_frames = sorted(Path(os.path.join(self.img_folder, "JPEGImages", vid_name)).glob("*"))
            vid_len = len(vid_frames)
            for frame_id in range(0, vid_len, self.num_frames): # Get each frame
                cur_frame = vid_frames[frame_id]
                frame_name = os.path.basename(cur_frame)
                mask_path = os.path.join(str(self.img_folder), 'Annotations', vid_name, frame_name)
                mask = Image.open(mask_path).convert('P')
                mask = np.array(mask) 
                category = np.unique(mask)
                category = category[category > 0] # positive classes
                for cls in category:
                    meta = {}
                    meta['video'] = vid_name
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    meta['category'] = (cls,1)
                    self.metas.append(meta)
                # negative = [cls for cls in self.available_classes if cls not in set(category)]
                # neg_cls = random.sample(negative, 1)
                # for cls in neg_cls: # negative classes
                #     meta = {}
                #     meta['video'] = vid_name
                #     meta['frames'] = vid_frames
                #     meta['frame_id'] = frame_id
                #     meta['category'] = (cls, 0)
                #     self.metas.append(meta)
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

                video, frames, frame_id, (cls, is_instance) = \
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
                imgs, boxes, masks, valid = [], [], [], []
                for j in range(self.num_frames):
                    frame_indx = sample_indx[j]
                    frame_path = frames[frame_indx]
                    frame_name = os.path.basename(frame_path)
                    img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name)
                    mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name)
                    img = Image.open(img_path).convert('RGB')
                    mask = Image.open(mask_path).convert('P')

                    # create the target
                    mask = np.array(mask)
                    mask = (mask==cls).astype(np.float64) 
                    
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

                if(is_instance == 1):
                    # descriptions = list(descriptor.get(cls, []))
                    # cap = f"{rev_category_dict.get(cls, 'other')} with {random.choice(descriptions)}"
                    cap = verb_dict.get(cls, 'Ultrasound Probe scanning and imaging')
                else:
                    cap = f"No {rev_category_dict.get(cls, 'other')}"

                target = {
                    'frames_idx': torch.tensor(sample_indx), # [T,]
                    'masks': masks,                          # [T, H, W]
                    'valid': torch.tensor(valid),            # [T,]
                    'orig_size': torch.as_tensor([int(h), int(w)]),
                    'boxes': boxes,
                    'size': torch.as_tensor([int(h), int(w)]),
                    'caption': cap,
                }

                imgs, target = self._transforms(imgs, target)
                imgs = torch.stack(imgs, dim=0) # [T, 3, H, W]

                if torch.any(target['valid'] == 1):  # at leatst one instance
                    instance_check = True
                else:
                    idx = random.randint(0, self.__len__() - 1)

            return imgs, target

def make_transforms(max_size=1024):
    return T.Compose([
        T.CenterCrop((max_size, max_size)),
        # T.Resize(max_size-4, max_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def build(image_set, args):
    if(args.dataset_file == 'endovis2017'):
        root = Path(args.endovis2017)
    elif(args.dataset_file == 'endovis2018'):
        root = Path(args.endovis2018)
    else:
        raise ValueError(f"dataset {args.dataset_file} not supported.")
    PATHS = {
        "train": (root / "train"),
        "val": (root / args.split),
    }
    img_folder = PATHS[image_set]
    dataset = EndoVis2017Dataset(img_folder, transforms=make_transforms(args.max_size),
                                num_frames=args.num_frames, max_skip=args.max_skip)
    
    return dataset
