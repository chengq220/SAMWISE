"""
Train and eval functions used in main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
# The evaluate function need to be changed 

import math
import os
import sys
from typing import Iterable
import time
import torch
import util.misc as utils
from torch.nn import functional as F
from models.segmentation import loss_masks
from torchmetrics.classification import BinaryJaccardIndex

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
                     device: torch.device,
                     data_loader: Iterable):
    model.eval()
    metric = BinaryJaccardIndex()

    # load data
    start_time = time.time()
    print('Start Evaluation')

    all_pred_masks = []
    all_gt = []
    for samples, targets in data_loader:
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]

        targets = utils.targets_to(targets, device)
        gt = torch.stack([t["masks"] for t in targets])
        print(targets)

        outputs = model(samples, captions, targets)
        pred_masks = torch.cat(outputs["masks"])
        pred_masks = (pred_masks.sigmoid() > args.threshold)
        
        all_pred_masks.append(pred_masks)
        all_gt.append(gt)

    # store the video results
    all_pred_masks = torch.cat(all_pred_masks, dim=0)
    all_gt = torch.cat(all_gt, dim=0)

    iou = metric(all_pred_masks, all_gt)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Evaluation IoU: {iou:.4f}")
    print(f"Total Evaluation time: {total_time:.2f} s")
