import torch
from datasets.categories import endovis2017_category_rev_dict, endovis2017_category_descriptor_dict
import torch.nn.functional as F

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

def multiclass_segmentation(model, images, targets, threshold, size):
    (origin_h, origin_w) = size
    t, q, h, w = images.shape
    num_classes = 7

    frames = []
    
    all_class_tensor = torch.full(
        (t, num_classes, h, w), 
        fill_value=-1024.0,
        dtype=torch.float32
    )
    
    for class_idx in range(num_classes):
        class_name = endovis2017_category_rev_dict.get(class_idx+1, "Other")
        prompt = f"{class_name}"
        with torch.no_grad():
            outputs = model([images], [prompt], [targets])
        logits = outputs["pred_masks"]  # [t, h, w] raw logits where t are the number of frames
        all_class_tensor[:,class_idx,:,:] = logits #[t, 7, h, w]

    for clip_idx in range(all_class_tensor.shape[0]):
        cur_frame = all_class_tensor[clip_idx]
        scores = cur_frame.unsqueeze(1)
        scores = apply_non_overlapping_constraints(scores)
        scores = scores.squeeze(1)
        max_scores, max_classes = torch.max(scores, dim=0)  # (H, W), (H, W)
        final_mask = torch.zeros((h, w), dtype=torch.long)
        
        valid_pixels = max_scores > threshold
        final_mask[valid_pixels] = max_classes[valid_pixels] + 1 
    
        if (h, w) != (origin_h, origin_w):
            final_mask = final_mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            final_mask = F.interpolate(
                final_mask,
                size=(origin_h, origin_w),
                mode='nearest'
            ).squeeze().long()
        
        frames.append(final_mask)
    frames = torch.stack(frames)
    return frames