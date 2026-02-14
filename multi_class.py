import torch
from datasets.categories import endovis2017_category_rev_dict, endovis2017_category_descriptor_dict
import random 
import torch.nn.functional as F


def multiclass_segmentation(model, images, targets, size ):
    """
    Multiclass segmentation function
    :param images: tensor [txqxwxh]
    """
    num_classes = 7 
    origin_h, origin_w = size
    t, q, h, w = images.shape
    all_logits = torch.zeros((t, num_classes + 1, origin_h, origin_w), 
                           device=images.device, dtype=images.dtype)
    for class_idx in range(7):
        class_name = endovis2017_category_rev_dict.get(class_idx+1, "Other")
        descriptors = list(endovis2017_category_descriptor_dict.get(class_idx+1, []))
        prompt = f"{class_name} with {random.choice(descriptors)}"

        with torch.no_grad():
            outputs = model([images], [prompt], [targets])
        pred_masks = outputs["pred_masks"]  # [t, h, w]
        pred_masks = pred_masks.unsqueeze(0)
        logits = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False) 
        all_logits[:, class_idx + 1, :, :] = logits
    probabilities = torch.softmax(all_logits, dim=1)
    seg = torch.argmax(probabilities, dim=1)
    return seg
