import torch
from datasets.categories import endovis2017_category_rev_dict, endovis2017_category_descriptor_dict
import random 
import torch.nn.functional as F
from scipy import ndimage


def multiclass_segmentation(model, images, targets, threshold, size):
    t, q, h, w = images.shape
    (origin_h, origin_w) = size
    num_classes = 7
    final_seg = torch.zeros((t, h, w), device='cpu', dtype=torch.long)
    confidence_map = torch.zeros((t, h, w), device='cpu')
    
    for class_idx in range(num_classes):
        class_name = endovis2017_category_rev_dict.get(class_idx+1, "Other")
        descriptors = list(endovis2017_category_descriptor_dict.get(class_idx+1, []))
        prompt = f"{class_name} with {random.choice(descriptors)}"

        with torch.no_grad():
            outputs = model([images], [prompt], [targets])
        
        pred_masks = outputs["pred_masks"]  # [t, h, w] raw logits
        probs = torch.sigmoid(pred_masks).cpu()  # [t, h, w]
        binary_mask = probs > threshold  # [t, h, w] 1 and 0s
        current_confidence_obj = probs[binary_mask == 1].mean() 
        if(current_confidence_obj < threshold): # skip if not confident the tool exists
            continue
        current_confidence_map = torch.zeros_like(confidence_map)
        current_confidence_map[binary_mask == 1] = current_confidence_obj
        update_mask = (current_confidence_map > confidence_map) & binary_mask
        final_seg[update_mask] = class_idx + 1
        confidence_map = torch.max(confidence_map, current_confidence_map)

    # binary_final = final_seg.cpu().numpy() > 0
    # eroded = ndimage.binary_erosion(binary_final, iterations=1)
    # erode_mask = torch.from_numpy(eroded > 0)
    # final_seg = final_seg[erode_mask]
    return final_seg
