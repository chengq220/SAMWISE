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
import torchvision.transforms as TF
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

        outputs = multiclass_segmentation(model, imgs, target, threshold=args.threshold, size = (origin_h, origin_w)).long()
        all_pred_masks.append(outputs.cpu())
            
    # store the video results
    all_pred_masks = torch.cat(all_pred_masks, dim=0).numpy()  # (video_len, h, w)
    print(f"Unique classes: {np.unique(all_pred_masks)}")

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
    
    name = args.text_prompts[0]

    all_pred_masks = compute_masks(model, frames_folder, frames_list, ext)
        
    save_visualize_path_dir = join(save_path_prefix, name.replace(' ', '_'))
    os.makedirs(save_visualize_path_dir, exist_ok=True)
    print(f'Saving output to disk in {save_visualize_path_dir}')
    out_files_w_mask = []
    mask_path = join(save_visualize_path_dir, "pred", os.path.basename(frames_folder))
    gt_save_path = join(save_visualize_path_dir, "gt", os.path.basename(frames_folder))
    overlay_path = join(save_visualize_path_dir, "viz")
    if not os.path.isdir(mask_path):
        os.makedirs(mask_path)
    if not os.path.isdir(gt_save_path):
        os.makedirs(gt_save_path)
    if not os.path.isdir(overlay_path):
        os.makedirs(overlay_path)
    transform = TF.Compose([
        # TF.Resize(args.max_size-4, max_size=args.max_size),
        TF.CenterCrop(args.max_size),
    ])
    # crop_size = args.max_size
    for t, frame in enumerate(frames_list):
        # original
        img_path = join(frames_folder, frame + ext)
        mask_path_open = img_path.replace('JPEGImages','Annotations')
        
        source_img = Image.open(img_path).convert('RGBA') # PIL image
        source_img = transform(source_img)

        source_mask = Image.open(mask_path_open).convert('L')
        source_mask = transform(source_mask)
        
        source_mask.save(os.path.join(gt_save_path, frame + '.png'))
        
        source_img = vis_add_mask_multiclass(source_img, all_pred_masks[t])
        # save
        save_visualize_path = join(overlay_path, frame + '.png')
        source_img.save(save_visualize_path)
        out_files_w_mask.append(save_visualize_path)

        mask_multi = all_pred_masks[t]
        mask_np = mask_multi.squeeze().astype('uint8')
        Image.fromarray(mask_np).save(join(mask_path, frame+'.png'))

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
    parser.add_argument('--text_prompts', default=[''], type=str, required=True, nargs='+', help="List of referring expressions, separated by whitespace")

    args = parser.parse_args()
    check_args(args)
    main(args)
