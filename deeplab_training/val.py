import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import sys

# Add shared directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

from iou import calculate_iou

def validate_epoch(model, dataloader, criterion, device, num_classes, logger=None, unique_categories=None):
    model.eval()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            predictions = torch.argmax(outputs, dim=1)
            
            # Accumulate for epoch-level IoU
            all_preds.append(predictions.cpu())
            all_targets.append(masks.cpu())
            
            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()
            running_loss += loss.item()
            
            accuracy = correct_pixels / total_pixels
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.4f}'
            })
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute per-class IoU
    mean_iou, per_class_iou = calculate_iou(all_preds, all_targets, num_classes, return_per_class=True)
    
    avg_loss = running_loss / len(dataloader)
    pixel_accuracy = correct_pixels / total_pixels
    
    # Log per-class IoU
    if logger:
        logger.log(f"Val IoU (mean): {mean_iou:.4f}")
        logger.log("Per-class IoU:")
        for cls_idx, iou_val in enumerate(per_class_iou):
            if not np.isnan(iou_val):
                logger.log(f"  Class {cls_idx}: {iou_val:.4f}")
    
    return avg_loss, pixel_accuracy, mean_iou

    