"""
Train and eval functions used in main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
# The evaluate function need to be changed 

import math
import os
import sys
from typing import Iterable
from time import time
import numpy as np
import random
import torch
import Path 
import util.misc as utils
from torch.nn import functional as F
from models.segmentation import loss_masks
from datasets import build_dataset

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    lr_scheduler=None, args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    step=0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        step+=1
        model.train()
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        outputs = model(samples, captions, targets)
        losses = {}
        seg_loss = loss_masks(torch.cat(outputs["masks"]), targets, num_frames=samples.tensors.shape[1])
        losses.update(seg_loss)
        if args.use_cme_head and "pred_cme_logits" in outputs:
            weight = torch.tensor([1., 2.]).to(device)
            CME_loss = F.cross_entropy(torch.cat(outputs["pred_cme_logits"]), ignore_index=-1,
                                        target=torch.tensor(outputs["cme_label"]).long().to(device),
                                        weight=weight)
            losses.update({"CME_loss": CME_loss if not CME_loss.isnan() else torch.tensor(0).to(device)})

        loss_dict = losses
        losses = sum(loss_dict[k] for k in loss_dict.keys())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v for k, v in loss_dict_reduced.items()}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        optimizer.step()
        lr_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def eval_endovis2017(args,
                    model: torch.nn.Module,
                    data_loader: Iterable,
                    save_path_prefix: str):
    model.eval()
    print("Evaluation only supports for batch size = 1")
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # save path
    os.makedirs(save_path_prefix, exist_ok=True)

    # load data
    start_time = time.time()
    print('Start Evaluation')

    # Build the evaluation dataset
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        step+=1
        model.train()
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        outputs = model(samples, captions, targets)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total inference time: {total_time:.2f} s")
