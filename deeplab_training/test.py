import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt

# Add shared directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

from iou import calculate_iou

# Hard-coded label mapping
sparse_to_dense = {0: 0, 1: 1, 2: 2, 3: 3, 6: 4, 7: 5, 16: 6, 17: 7, 18: 8, 20: 9}
dense_to_sparse = {v: k for k, v in sparse_to_dense.items()}

def test_and_save_predictions(model, test_loader, criterion, device, CONFIG, output_dir, logger=None, unique_categories=None):
    """Test model and save predictions with overlays and category IDs"""
    model.eval()
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    all_preds = []
    all_targets = []
    
    # Use hard-coded mapping instead of dynamic one
    dense_to_category = dense_to_sparse
    
    # Define colors for categories
    np.random.seed(42)
    if unique_categories is not None:
        category_colors = {idx: tuple(np.random.randint(50, 255, 3).tolist()) for idx in range(len(unique_categories))}
    else:
        category_colors = {i: tuple(np.random.randint(50, 255, 3).tolist()) for i in range(CONFIG['num_classes'])}
    category_colors[0] = (0, 0, 0)  # Background is black
    
    with torch.no_grad():
        total_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            predictions = torch.argmax(outputs, dim=1)
            
            # Accumulate for epoch-level IoU
            all_preds.append(predictions.cpu())
            all_targets.append(masks.cpu())
            
            # Calculate metrics
            total_loss += loss.item()
            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()
            
            # Save first 20 images with overlays
            if i < 20:
                for j in range(images.shape[0]):
                    # Denormalize image
                    img_tensor = images[j].cpu()
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_tensor = img_tensor * std + mean
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    
                    # Get masks
                    gt_mask = masks[j].cpu().numpy()
                    pred_mask = predictions[j].cpu().numpy()
                    
                    # Calculate IoU for this image
                    intersection = ((pred_mask == gt_mask) & (gt_mask != 0)).sum()
                    union_pred = (pred_mask != 0).sum()
                    union_gt = (gt_mask != 0).sum()
                    union = union_pred + union_gt - intersection
                    iou = intersection / union if union > 0 else 0.0
                    
                    # Create figure with 2 subplots side by side
                    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                    
                    # Ground Truth Overlay
                    axes[0].imshow(img_np)
                    gt_overlay = np.zeros((*gt_mask.shape, 4), dtype=np.uint8)
                    for dense_id in np.unique(gt_mask):
                        if dense_id == 0:
                            continue
                        mask_area = (gt_mask == dense_id)
                        color = category_colors.get(int(dense_id), (255, 0, 0))
                        gt_overlay[mask_area] = (*color, 128)  # 50% opacity
                    axes[0].imshow(gt_overlay)
                    
                    # Add category ID labels for GT (show actual category ID, not dense index)
                    for dense_id in np.unique(gt_mask):
                        if dense_id == 0:
                            continue
                        y_coords, x_coords = np.where(gt_mask == dense_id)
                        if len(y_coords) > 0:
                            center_y, center_x = int(y_coords.mean()), int(x_coords.mean())
                            # Map dense index to actual category ID for display
                            display_id = dense_to_category[int(dense_id)] if dense_to_category else int(dense_id)
                            axes[0].text(center_x, center_y, f'{display_id}', 
                                       fontsize=16, fontweight='bold', color='white',
                                       bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8),
                                       ha='center', va='center')
                    
                    # Get category IDs for title
                    gt_category_ids = [dense_to_category[int(d)] if dense_to_category else int(d) 
                                      for d in np.unique(gt_mask) if d != 0]
                    axes[0].set_title(f'Ground Truth\nCategories: {gt_category_ids}', 
                                    fontsize=13, fontweight='bold')
                    axes[0].axis('off')
                    
                    # Prediction Overlay
                    axes[1].imshow(img_np)
                    pred_overlay = np.zeros((*pred_mask.shape, 4), dtype=np.uint8)
                    for dense_id in np.unique(pred_mask):
                        if dense_id == 0:
                            continue
                        mask_area = (pred_mask == dense_id)
                        color = category_colors.get(int(dense_id), (255, 0, 0))
                        pred_overlay[mask_area] = (*color, 128)  # 50% opacity
                    axes[1].imshow(pred_overlay)
                    
                    # Add category ID labels for predictions (show actual category ID, not dense index)
                    for dense_id in np.unique(pred_mask):
                        if dense_id == 0:
                            continue
                        y_coords, x_coords = np.where(pred_mask == dense_id)
                        if len(y_coords) > 0:
                            center_y, center_x = int(y_coords.mean()), int(x_coords.mean())
                            # Map dense index to actual category ID for display
                            display_id = dense_to_category[int(dense_id)] if dense_to_category else int(dense_id)
                            axes[1].text(center_x, center_y, f'{display_id}', 
                                       fontsize=16, fontweight='bold', color='white',
                                       bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8),
                                       ha='center', va='center')
                    
                    # Get category IDs for title
                    pred_category_ids = [dense_to_category[int(d)] if dense_to_category else int(d) 
                                        for d in np.unique(pred_mask) if d != 0]
                    axes[1].set_title(f'Prediction (IoU: {iou:.3f})\nCategories: {pred_category_ids}', 
                                    fontsize=13, fontweight='bold')
                    axes[1].axis('off')
                    
                    plt.tight_layout()
                    pred_path = os.path.join(predictions_dir, f"prediction_{i}_{j}.png")
                    plt.savefig(pred_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute test metrics
        iou_result, per_class_iou = calculate_iou(all_preds, all_targets, CONFIG['num_classes'], return_per_class=True)
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct_pixels / total_pixels
        
        if logger:
            logger.log(f"\n{'='*50}")
            logger.log("TEST RESULTS")
            logger.log(f"{'='*50}")
            logger.log(f"Test Loss: {avg_loss:.4f}")
            logger.log(f"Test Accuracy: {accuracy:.4f}")
            logger.log(f"Test Mean IoU: {iou_result:.4f}")
            
            if unique_categories is not None:
                logger.log(f"\nPer-category IoU:")
                for dense_idx, sparse_cat in dense_to_sparse.items():
                    if not np.isnan(per_class_iou[dense_idx]):
                        logger.log(f"  Category {sparse_cat} (dense {dense_idx}): {per_class_iou[dense_idx]:.4f}")
            
            logger.log(f"\nPredictions saved to: {predictions_dir}")
            logger.log(f"{'='*50}\n")
        
        # Save test metrics
        test_metrics = {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'test_mean_iou': iou_result,
            'per_class_iou': per_class_iou.tolist() if isinstance(per_class_iou, np.ndarray) else per_class_iou
        }
        
        with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        return avg_loss, accuracy, iou_result
