import torch
import os
import argparse
import torch
import os
from PIL import Image
from pathlib import Path 
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.samwise import build_samwise
from util.misc import on_load_checkpoint
from torch.utils.data import Dataset
import torchvision.transforms as TF
import time 
import numpy as np
from datasets.categories import endovis2017_category_rev_dict as category_dict, endovis2017_category_descriptor_dict
from tools.metrics import db_eval_iou
import random 


class EvalDataset(Dataset):
    def __init__(self, vid_folder, max_size=1024):
        super().__init__()
        self.vid_folder = vid_folder
        self.frames = sorted(list(Path(os.path.join(vid_folder, "image")).glob('*')))
        self.vid_len = len(self.frames)
        self.origin_w, self.origin_h = Image.open(self.frames[0]).size
        self.img_transform = TF.Compose([
            TF.Resize(max_size-4, max_size=max_size), #T.Resize(360),
            TF.ToTensor(),
            TF.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
            
    def __len__(self):
        return self.vid_len
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        img = Image.open(frame).convert('RGB')
        mask_path = os.path.join(self.vid_folder, "label", os.path.basename(frame))
        mask = Image.open(mask_path).convert('P')
        
        return self.img_transform(img), np.array(mask).astype('uint8'), idx


def evaluate(args):
    # load data
    if not os.path.exists(args.file_path):
         raise FileNotFoundError(f"Dataset directory not found: {args.file_path}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    # load model
    device = torch.device(args.device)
    model = build_samwise(args)
    model.to(device)
    model_without_ddp = model

    checkpoint = torch.load(args.model, map_location='cpu')
    checkpoint = on_load_checkpoint(model_without_ddp, checkpoint)
    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        
    model.eval()
    start_time = time.time()
    print(f'Begin Evaluation')
    mIou = []

    for cls in range(1, 8):
        text_prompt = category_dict.get(cls, "Other")

        all_pred_masks = []
        all_gt_masks = []
        vd = EvalDataset(args.file_path)
        dl = DataLoader(vd, batch_size=args.num_frames, num_workers=args.num_workers, shuffle=False)
        origin_w, origin_h = vd.origin_w, vd.origin_h
        
        for imgs, masks, clip_frames_ids in tqdm(dl):
            clip_frames_ids = clip_frames_ids.tolist()
            imgs = imgs.to(args.device)  # [eval_clip_window, 3, h, w]
            img_h, img_w = imgs.shape[-2:]
            size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
            target = {"size": size, 'frame_ids': clip_frames_ids}

            descriptions = list(endovis2017_category_descriptor_dict.get(cls, []))
            aug_prompt = f"{text_prompt} with {random.choice(descriptions)}"

            with torch.no_grad():
                outputs = model([imgs], [aug_prompt], [target])

            pred_masks = outputs["pred_masks"]  # [t, q, h, w]
            pred_masks_logits = pred_masks
            pred_masks = pred_masks.unsqueeze(0)
            pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False) 
            pred_masks = (pred_masks.sigmoid() > args.threshold)[0].cpu() 
            all_pred_masks.append(pred_masks)

            masks_cls = (masks==cls).bool()
            all_gt_masks.append(masks_cls)
        
        all_pred_masks = torch.cat(all_pred_masks, dim=0).numpy()
        all_gt_masks = torch.cat(all_gt_masks, dim=0).squeeze().numpy()

        iou = db_eval_iou(all_gt_masks, all_pred_masks)
        print(f"Raw logits range: [{pred_masks_logits.min():.4f}, {pred_masks_logits.max():.4f}]")
        avg_iou = np.mean(iou)
        mIou.append(avg_iou)
        print(f"Evaluation IoU for class {text_prompt}: {avg_iou:.4f}")
    mIou = np.mean(np.array(mIou))
    print(f"Evaluation mIoU: {mIou:.4f}")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Evaluation time: {total_time:.2f} s")


if __name__ == '__main__':
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam2_version', default='tiny', type=str, choices=['tiny', 'base', 'large', 'med'],
                        help="Version of SAM2 image encoder to use")
    parser.add_argument('--disable_pred_obj_score', default=False, action='store_true',
                        help="Disable predicted object score")
    parser.add_argument('--motion_prompt', default=False, action='store_true',
                        help="Enable motion-based prompting")

    # Cross Modal Temporal Adapter settings
    parser.add_argument('--HSA', action='store_true', default=False,
                        help="Use Hierarchical Selective Attention (HSA)")
    parser.add_argument('--HSA_patch_size', default=[8, 4, 2], type=int, nargs='+',
                        help="Patch sizes used in HSA")
    parser.add_argument('--adapter_dim', default=256, type=int,
                        help="Dimensionality of adapter layers")
    parser.add_argument('--fusion_stages_txt', default=[4,8,12], type=int,
                        help="Text encoder fusion stages")
    parser.add_argument('--fusion_stages', default=[1,2,3], nargs='+', type=int,
                        help="Fusion stages")


    # Conditional Memory Encoder (CME) settings
    parser.add_argument('--use_cme_head', default=False, action='store_true',
                        help="Use Conditional Memory Encoder (CME)")
    parser.add_argument('--switch_mem', default='reweight', type=str, choices=['all_mask', 'reweight', 'avg'],
                        help="Memory switch strategy")
    parser.add_argument('--cme_decision_window', default=4, type=int,
                        help="Minimum number of frames considered between consecutive CME decisions")
    
    # Evaluation settings
    parser.add_argument('--device', default='cuda', type=str,
                        help="Device for computation ('cuda' or 'cpu')")
    parser.add_argument('--gpu', default=0, type=int,
                        help="GPUs")
    parser.add_argument('--threshold', default=0.5, type=float,
                    help="Threshold for binary mask predictions")
    parser.add_argument("--model", type = str, default = None, required=True, help="The save file for the trained model")
    parser.add_argument("--file_path", type = str, default = None, required=True, help="The path to the evaluation images")
    parser.add_argument("--save_path", type = str, default = "output/default", help="directory to save the results/logs")
    parser.add_argument("--num_frames", type = int, default = 6, help="BNumber of frames to use")
    parser.add_argument("--num_workers", type = int, default = 1, help="Number of workers")

    args = parser.parse_args()
    evaluate(args)
