import numpy as np
import torch

def calculate_iou(pred, target, num_classes, return_per_class=False):
    pred = pred.view(-1)
    target = target.view(-1)
    
    per_class_iou = np.full(num_classes, np.nan, dtype=float)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = int((pred_cls & target_cls).sum().item())
        union = int((pred_cls | target_cls).sum().item())
        
        if union > 0:
            per_class_iou[cls] = intersection / union
    
    # Compute mean over classes that have data (nan-safe)
    valid_ious = per_class_iou[~np.isnan(per_class_iou)]
    mean_iou = float(np.mean(valid_ious)) if valid_ious.size > 0 else 0.0
    
    if return_per_class:
        return mean_iou, per_class_iou
    else:
        return mean_iou