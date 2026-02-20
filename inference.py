'''
Inference code for SAMWISE
Modified from DETR (https://github.com/facebookresearch/detr)
'''

import argparse
from collections import defaultdict
import random
import time
import numpy as np
import torch
import os
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import sys
from tools.colormap import colormap
import opts
from models.samwise import build_samwise
from util.misc import on_load_checkpoint
from datasets.transform_utils import VideoEvalDataset
from torch.utils.data import DataLoader
from os.path import join
from datasets.transform_utils import vis_add_mask, vis_add_mask_multiclass
from datasets.categories import endovis2017_category_dict, endovis2018_category_verb_dict
import glob
import torchvision.transforms as TF


# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()


def main(args):    
    # fix the seed for reproducibility
    seed = 0
    #utils.init_distributed_mode(args)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # save path
    input_path = args.input_path
    save_path_prefix = args.output_dir
    os.makedirs(save_path_prefix, exist_ok=True)

    start_time = time.time()
    # model
    model = build_samwise(args)
    device = torch.device(args.device)
    model.to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if list(checkpoint['model'].keys())[0].startswith('module'):
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}        
        checkpoint = on_load_checkpoint(model, checkpoint)
        model.load_state_dict(checkpoint['model'], strict=False)

    print('Start inference')
    inference(args, model, save_path_prefix, input_path, args.text_prompts)

    end_time = time.time()
    total_time = end_time - start_time
    print("Total inference time: %.4f s" %(total_time))

def extract_frames_from_mp4(video_path):
    extract_folder = 'frames_' + os.path.basename(video_path).split('.')[0]
    print(f'Extracting frames from .mp4 in {extract_folder} with ffmpeg...')
    if os.path.isdir(extract_folder):
        print(f'{extract_folder} already exists, using frames in that folder')
    else:
        os.makedirs(extract_folder)
        extract_cmd = "ffmpeg -i {in_path} -loglevel error -vf fps=10 {folder}/frame_%05d.png"
        ret = os.system(extract_cmd.format(in_path=video_path, folder=extract_folder))
        if ret != 0:
            print('Something went wrong extracting frames with ffmpeg')
            sys.exit(ret)
    frames_list=os.listdir(extract_folder)
    frames_list = sorted([os.path.splitext(frame)[0] for frame in frames_list])

    return extract_folder, frames_list, '.png'

def apply_non_overlapping_constraints(pred_masks):
    """
    Apply non-overlapping constraints to the object scores in pred_masks. Here we
    keep only the highest scoring object at each spatial location in pred_masks.
    """
    batch_size = pred_masks.size(0)
    if batch_size == 1:
        return pred_masks

    device = pred_masks.device
    # "max_obj_inds": object index of the object with the highest score at each location
    max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
    # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
    batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
    keep = max_obj_inds == batch_obj_inds
    # suppress overlapping regions' scores below -10.0 so that the foreground regions
    # don't overlap (here sigmoid(-10.0)=4.5398e-05)
    pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
    return pred_masks

def compute_masks(model, text_prompt, frames_folder, frames_list, ext):
    all_pred_masks = []
    all_pred_logits = []
    vd = VideoEvalDataset(frames_folder, frames_list, ext=ext)
    dl = DataLoader(vd, batch_size=args.eval_clip_window, num_workers=args.num_workers, shuffle=False)
    origin_w, origin_h = vd.origin_w, vd.origin_h
    # 3. for each clip
    for imgs, clip_frames_ids in tqdm(dl):
        clip_frames_ids = clip_frames_ids.tolist()
        imgs = imgs.to(args.device)  # [eval_clip_window, 3, h, w]
        img_h, img_w = imgs.shape[-2:]
        size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
        target = {"size": size, 'frame_ids': clip_frames_ids}

        with torch.no_grad():
            outputs = model([imgs], [text_prompt], [target])
        pred_masks = outputs["pred_masks"]  # [t, h, w]
        pred_masks = pred_masks.unsqueeze(0)
        pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear',
                    align_corners=False)
        all_pred_logits.append(pred_masks.squeeze(0).cpu())
        pred_masks = pred_masks.sigmoid()[0] > args.threshold # [t, h, w]
        all_pred_masks.append(pred_masks)
            
    # store the video results
    all_pred_masks = torch.cat(all_pred_masks, dim=0).cpu().numpy()  # (video_len, h, w)
    all_pred_logits = torch.cat(all_pred_logits, dim=0).unsqueeze(1) # (video_len, 1, h, w)
    print(f"Raw logits range: [{all_pred_masks.min():.4f}, {all_pred_masks.max():.4f}]")

    return all_pred_masks, all_pred_logits

def get_masks_id(masks_path, transform):
    vos_path = list(glob.glob(join(masks_path, '*.png')))
    obj_id_list = []
    for mask in vos_path:
        obj_img = Image.open(mask).convert('L')
        obj_img = transform(obj_img)
        obj_id = np.unique(obj_img)
        for id in obj_id:
            if id not in obj_id_list and id != 0:
                obj_id_list.append(id)
    return obj_id_list

    
def inference(args, model, save_path_prefix, in_path, text_prompts):
    # load data
    if os.path.isfile(in_path) and not args.image_level:
        frames_folder, frames_list, ext = extract_frames_from_mp4(in_path)
    elif os.path.isfile(in_path) and args.image_level:
        fname, ext = os.path.splitext(in_path)
        frames_list = [os.path.basename(fname)]
        frames_folder = os.path.dirname(in_path)
    else:
        frames_folder = in_path
        frames_list = sorted(os.listdir(frames_folder))
        ext = os.path.splitext(frames_list[0])[1]
        frames_list = [os.path.splitext(frame)[0] for frame in frames_list if os.path.splitext(frame)[1] == ext]
        
    model.eval()
    print(f'Begin inference on {len(frames_list)} frames')

    transform = TF.Compose([TF.CenterCrop(args.max_size)])
    obj_id_list = get_masks_id(args.mask_input, transform)
    print(f"Object IDs found in VOS masks: {obj_id_list}")
    in_path_folder = os.path.basename(in_path)
    obj_logits = defaultdict(torch.Tensor)
    name = args.text_prompts[0]
    # For each expression
    for id in obj_id_list:
        text_prompt = endovis2018_category_verb_dict.get(id, "Ultrasound Probe scanning and visualizing internal structures")

        all_pred_masks, all_pred_logits = compute_masks(model, text_prompt, frames_folder, frames_list, ext)
        obj_logits[id] = all_pred_logits
            
        save_visualize_path_dir = join(save_path_prefix, name, 'viz', in_path_folder, str(id))
        save_mask_path_dir = join(save_path_prefix, name, "pred", in_path_folder, str(id))
        
        os.makedirs(save_visualize_path_dir, exist_ok=True)
        os.makedirs(save_mask_path_dir, exist_ok=True)

        print(f'Saving output to disk in {save_visualize_path_dir}')
        for t, frame in enumerate(frames_list):
            cur_mask = all_pred_masks[t]
            cur_mask = cur_mask.reshape(cur_mask.shape[0], cur_mask.shape[1]).astype('uint8')
            pil_mask = Image.fromarray(cur_mask)
            save_mask_path = join(save_mask_path_dir, frame + '.png')
            pil_mask.save(save_mask_path)

            img_path = join(frames_folder, frame + ext)
            source_img = Image.open(img_path).convert('RGBA') # PIL image
            source_img = transform(source_img)

            source_img = vis_add_mask(source_img, all_pred_masks[t], color_list[id%len(color_list)])
     
            save_visualize_path = join(save_visualize_path_dir, frame + '.png')
            source_img.save(save_visualize_path)

    first_logits = next(iter(obj_logits.values()))
    num_frames, _, h, w = first_logits.shape  # num_frames, 1, height, width
    object_ids = sorted(list(obj_logits.keys()))
    num_objects = len(object_ids)

    video_segments = {} 
    multiclass_masks = {} 

    for frame_idx in range(num_frames):
        scores = torch.full(
            size=(num_objects, 1, h, w),
            fill_value=-1024.0,
            dtype=torch.float32,
        )
    
        for i, object_id in enumerate(object_ids):
            scores[i] = obj_logits[object_id][frame_idx]

        scores = apply_non_overlapping_constraints(scores)
        per_obj_output_mask = {
            object_id: (scores[i] > args.threshold).cpu()
            for i, object_id in enumerate(object_ids)
        }
        video_segments[frame_idx] = per_obj_output_mask
        
        multiclass_mask = np.zeros((h, w), dtype=np.uint8)
        for object_id in sorted(object_ids, reverse=True):
            object_mask = per_obj_output_mask[object_id].numpy().squeeze()
            multiclass_mask[object_mask] = object_id
    
        multiclass_masks[frame_idx] = multiclass_mask

    save_mask_multi_path = join(save_path_prefix, name, 'pred', in_path_folder, 'all')
    save_visualize_multi_path = join(save_path_prefix, name, 'viz', in_path_folder, 'all')
    os.makedirs(save_mask_multi_path, exist_ok=True)
    os.makedirs(save_visualize_multi_path, exist_ok=True)
    for mask, frame_idx in zip(multiclass_masks.items(), frames_list):
        img_path = join(frames_folder, frame_idx + ext)
        source_img = Image.open(img_path).convert('RGBA') # PIL image
        source_img = transform(source_img)
        save_path_all = join(save_mask_multi_path, f"{frame_idx}.png")
        Image.fromarray(mask[1]).save(save_path_all)

        viz_multi = vis_add_mask_multiclass(source_img, mask[1])
        viz_multi_save = join(save_visualize_multi_path, f"{frame_idx}.png")
        viz_multi.save(viz_multi_save)

    print(f'Output masks and videos can be found in {save_path_prefix}')
    return 


def check_args(args):
    assert os.path.isfile(args.input_path) or os.path.isdir(args.input_path), f'The provided path {args.input_path} does not exist'
    args.image_level = False
    if os.path.isfile(args.input_path):
        ext = os.path.splitext(args.input_path)[1]
        assert ext in ['.jpg', '.png', '.mp4', '.jpeg', '.bmp'], f"Provided file extension should be one of ['.jpg', '.png', '.mp4', '.jpeg', '.bmp']"
        if ext in ['.jpg', '.png', '.jpeg', '.bmp']: 
            args.image_level = True
            pretrained_model = 'pretrain/pretrained_model.pth'
            pretrained_model_link = 'https://drive.google.com/file/d/1gRGzARDjIisZ3PnCW77Y9TMM_SbV8aaa/view?usp=drive_link'
            print(f'Specified path is an image, using image-level configuration')

    if not args.image_level: # it's video inference
        # set default args
        args.HSA = True
        args.use_cme_head = False
        pretrained_model = 'pretrain/final_model_mevis.pth'
        pretrained_model_link = 'https://drive.google.com/file/d/1Molt2up2bP41ekeczXWQU-LWTskKJOV2/view?usp=sharing'
        print(f'Specified path is a video or folder with frames, using video-level configuration')
        
    if args.resume == '':
        args.resume = pretrained_model

    assert os.path.isfile(args.resume), f"You should download the model checkpoint first. Run 'cd pretrain &&  gdown --fuzzy {pretrained_model_link}"


if __name__ == '__main__':
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser('SAMWISE evaluation script', parents=[opts.get_args_parser()])
    parser.add_argument('--input_path', default=None, type=str, required=True, help='path to mp4 video or frames folder')
    parser.add_argument('--mask_input', default=None, type=str, required=True, help='path to mask inputs')
    parser.add_argument('--text_prompts', default=[''], type=str, required=True, nargs='+', help="List of referring expressions, separated by whitespace")

    args = parser.parse_args()
    check_args(args)
    main(args)
