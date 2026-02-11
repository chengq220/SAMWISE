import torch
from datasets.categories import endovis2017_category_rev_dict, endovis2017_category_descriptor_dict
import random 

def multiclass_segmentation(model, images, targets):
    batch_size = images.shape[0]
    num_classes = 7 
    h, w = images.shape[-2], images.shape[-1]
    all_logits = torch.zeros((batch_size, num_classes + 1, h, w), 
                           device=images.device, dtype=images.dtype)
    for class_idx in range(7):
        class_name = endovis2017_category_rev_dict.get(class_idx, "Other")
        descriptors = list(endovis2017_category_descriptor_dict.get(class_idx, []))
        prompt = f"{class_name} with {random.choice(descriptors)}"

        with torch.no_grad():
            outputs = model([images], [prompt], [targets])
            logits = outputs["pred_masks"]  # [t, q, h, w]
            # logits.reshape((t*q, 1, h, w))
            logits = logits.unsqueeze(0)
        all_logits[:, class_idx + 1, :, :] = logits
    probabilities = torch.softmax(all_logits, dim=1)
    return probabilities
