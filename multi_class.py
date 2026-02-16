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
    t,q, h, w = images.shape
    num_classes = 7
    scores = torch.full(
        size=(num_classes, 1, h, w),
        fill_value=-1024.0,
        dtype=torch.float32,
    )
    for class_idx in range(num_classes):
        class_name = endovis2017_category_rev_dict.get(class_idx+1, "Other")
        descriptors = list(endovis2017_category_descriptor_dict.get(class_idx+1, []))
        # prompt = f"{class_name} with {random.choice(descriptors)}"
        prompt = f"{class_name}"
        with torch.no_grad():
            outputs = model([images], [prompt], [targets])
        logits = outputs["pred_masks"]  # [1, h, w] raw logits
        scores[class_idx] = logits.unsqueeze(1)  # [1, 1, h, w]
    scores = apply_non_overlapping_constraints(scores)
    scores = scores.squeeze(1)
    max_scores, max_classes = torch.max(scores, dim=0)  # (H, W), (H, W)
    final_mask = torch.zeros((h, w), dtype=torch.uint8)
    
    valid_pixels = max_scores > threshold
    final_mask[valid_pixels] = max_classes[valid_pixels] + 1 
    
    if (h, w) != (origin_h, origin_w):
        final_mask = final_mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        final_mask = F.interpolate(
            final_mask,
            size=(origin_h, origin_w),
            mode='nearest'
        ).squeeze().long()
    
    return final_mask


# def multiclass_segmentation(model, images, targets, threshold, size):
#     t, q, h, w = images.shape
#     (origin_h, origin_w) = size
#     num_classes = 7
#     final_seg = torch.zeros((t, h, w), device='cpu', dtype=torch.long)
#     confidence_map = torch.zeros((t, h, w), device='cpu')
    
#     for class_idx in range(num_classes):
#         class_name = endovis2017_category_rev_dict.get(class_idx+1, "Other")
#         descriptors = list(endovis2017_category_descriptor_dict.get(class_idx+1, []))
#         prompt = f"{class_name} with {random.choice(descriptors)}"

#         with torch.no_grad():
#             outputs = model([images], [prompt], [targets])
        
#         pred_masks = outputs["pred_masks"]  # [t, h, w] raw logits
#         probs = torch.sigmoid(pred_masks).cpu()  # [t, h, w]
#         binary_mask = probs > threshold  # [t, h, w] 1 and 0s
#         current_confidence_obj = probs[binary_mask == 1].mean() 
#         if(current_confidence_obj < threshold): # skip if not confident the tool exists
#             continue
#         current_confidence_map = torch.zeros_like(confidence_map)
#         current_confidence_map[binary_mask == 1] = current_confidence_obj
#         update_mask = (current_confidence_map > confidence_map) & binary_mask
#         final_seg[update_mask] = class_idx + 1
#         confidence_map = torch.max(confidence_map, current_confidence_map)

#     # binary_final = final_seg.cpu().numpy() > 0
#     # eroded = ndimage.binary_erosion(binary_final, iterations=1)
#     # erode_mask = torch.from_numpy(eroded > 0)
#     # final_seg = final_seg[erode_mask]
#     return final_seg
