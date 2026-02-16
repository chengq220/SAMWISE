'''
Inference code for SAMWISE, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
'''

import argparse
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
from datasets.categories import endovis2017_category_dict, endovis2017_category_descriptor_dict
from multi_class import multiclass_segmentation


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
    inference(args, model, save_path_prefix, input_path)

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


def compute_masks(model, frames_folder, frames_list, ext):
    all_pred_masks = []
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

        # cls = endovis2017_category_dict.get(text_prompt, 7)
        # descriptions = list(endovis2017_category_descriptor_dict.get(cls, []))
        # aug_prompt = f"{text_prompt} with {random.choice(descriptions)}"
        # if not multi:
        #     with torch.no_grad():
        #         outputs = model([imgs], [aug_prompt], [target])
        #     pred_masks = outputs["pred_masks"]  # [t, h, w]
        #     pred_masks = pred_masks.unsqueeze(0)
        #     pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False) 
        #     pred_masks = (pred_masks.sigmoid() > args.threshold)[0].cpu() 
        #     all_pred_masks.append(pred_masks)
        # else:
        outputs = multiclass_segmentation(model, imgs, target, threshold=args.threshold, size = (origin_h, origin_w)).long()
        all_pred_masks.append(outputs.cpu())
            
    # store the video results
    all_pred_masks = torch.cat(all_pred_masks, dim=0).numpy()  # (video_len, h, w)
    print(f"Raw logits range: [{all_pred_masks.min():.4f}, {all_pred_masks.max():.4f}]")

    return all_pred_masks

    
def inference(args, model, save_path_prefix, in_path):
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
    # For each expression
    # for i in range(len(text_prompts)):
    #     text_prompt = text_prompts[i]
    
    name = args.video_name

    all_pred_masks = compute_masks(model, frames_folder, frames_list, ext, args.multi_class)
        
    save_visualize_path_dir = join(save_path_prefix, name.replace(' ', '_'))
    os.makedirs(save_visualize_path_dir, exist_ok=True)
    print(f'Saving output to disk in {save_visualize_path_dir}')
    out_files_w_mask = []
    # crop_size = args.max_size
    for t, frame in enumerate(frames_list):
        # original
        img_path = join(frames_folder, frame + ext)
        source_img = Image.open(img_path).convert('RGBA') # PIL image

        # source_img = source_img.crop((
        #     (source_img.width - crop_size) // 2,
        #     (source_img.height - crop_size) // 2,
        #     ((source_img.width - crop_size) // 2) + crop_size,
        #     ((source_img.height - crop_size) // 2) + crop_size
        # ))
        # draw mask
        # if not args.multi_class:
        #     source_img = vis_add_mask(source_img, all_pred_masks[t], color_list[i%len(color_list)])
        # else:
        source_img = vis_add_mask_multiclass(source_img, all_pred_masks[t])
        # save
        save_visualize_path = join(save_visualize_path_dir, frame + '.png')
        source_img.save(save_visualize_path)
        out_files_w_mask.append(save_visualize_path)

    if not args.image_level and args.create_video:
        # Create the video clip from images
        from moviepy import ImageSequenceClip
        clip = ImageSequenceClip(out_files_w_mask, fps=10)
        # Write the video file
        clip.write_videofile(join(save_path_prefix, name.replace(' ', '_')+'.mp4'), codec='libx264')

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
    parser.add_argument('--create_video', action='store_true', help='whether to create video from output frames')

    args = parser.parse_args()
    check_args(args)
    main(args)
