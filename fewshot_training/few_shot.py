import os
import sys
import json
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from segment_anything import sam_model_registry, SamPredictor
from scipy.spatial.distance import directed_hausdorff, cdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from iou import calculate_iou


def load_data(bbox_json_path, labels_json_path, patches_dir, masks_dir):
    with open(bbox_json_path, 'r') as f:
        bbox_data = json.load(f)
    
    with open(labels_json_path, 'r') as f:
        labels_data = json.load(f)
    
    label_1_data = []
    for bbox_item, label_item in zip(bbox_data['patches'], labels_data['patches']):
        if bbox_item['label'] == 1:
            label_1_data.append({
                'file_name': bbox_item['file_name'],
                'bbox': bbox_item['bbox'],
                'label': bbox_item['label'],
                'category_id': bbox_item['category_id'], 
                'image_path': os.path.join(patches_dir, bbox_item['file_name']),
                'mask_path': os.path.join(masks_dir, bbox_item['file_name'])
            })
    
    return label_1_data


def convert_bbox_format(bbox):
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h])


def load_ground_truth_mask(mask_path, category_id):
    mask = Image.open(mask_path).convert('L')
    mask_array = np.array(mask)
    binary_mask = (mask_array == category_id).astype(np.uint8)
    return binary_mask


def compute_boundary_iou(pred_mask, gt_mask, dilation=2):
    kernel = np.ones((3, 3), np.uint8)
    pred_boundary = cv2.dilate(pred_mask.astype(np.uint8), kernel, iterations=dilation) - \
                   cv2.erode(pred_mask.astype(np.uint8), kernel, iterations=dilation)
    gt_boundary = cv2.dilate(gt_mask.astype(np.uint8), kernel, iterations=dilation) - \
                 cv2.erode(gt_mask.astype(np.uint8), kernel, iterations=dilation)
    
    intersection = (pred_boundary & gt_boundary).sum()
    union = (pred_boundary | gt_boundary).sum()
    
    return intersection / union if union > 0 else 0.0


def compute_hausdorff_distance(pred_mask, gt_mask):
    pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), 
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(pred_contours) == 0 or len(gt_contours) == 0:
        return float('inf')
    
    pred_points = pred_contours[0].reshape(-1, 2)
    gt_points = gt_contours[0].reshape(-1, 2)
    
    # Symmetric Hausdorff distance
    d1 = directed_hausdorff(pred_points, gt_points)[0]
    d2 = directed_hausdorff(gt_points, pred_points)[0]
    
    return max(d1, d2)


def compute_boundary_mae(pred_mask, gt_mask):
    pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), 
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(pred_contours) == 0 or len(gt_contours) == 0:
        return float('inf')
    
    pred_points = pred_contours[0].reshape(-1, 2)
    gt_points = gt_contours[0].reshape(-1, 2)
    
    # Distance from each predicted point to nearest GT point
    distances = cdist(pred_points, gt_points)
    min_distances = distances.min(axis=1)
    
    return min_distances.mean()


def evaluate_prediction(pred_mask, gt_mask):
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)
    
    # Standard IoU
    intersection = (pred_bool & gt_bool).sum()
    union = (pred_bool | gt_bool).sum()
    iou = intersection / union if union > 0 else 0.0
    
    # Dice coefficient
    dice = 2 * intersection / (pred_bool.sum() + gt_bool.sum()) if (pred_bool.sum() + gt_bool.sum()) > 0 else 0.0
    
    # Precision and Recall
    precision = intersection / pred_bool.sum() if pred_bool.sum() > 0 else 0.0
    recall = intersection / gt_bool.sum() if gt_bool.sum() > 0 else 0.0
    
    # F1-Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Boundary IoU (more relevant for polygons)
    boundary_iou = compute_boundary_iou(pred_mask, gt_mask)
    
    # Hausdorff distance (boundary alignment)
    hausdorff = compute_hausdorff_distance(pred_mask, gt_mask)
    
    # Boundary MAE
    boundary_mae = compute_boundary_mae(pred_mask, gt_mask)
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'boundary_iou': boundary_iou,
        'hausdorff_distance': hausdorff,
        'boundary_mae': boundary_mae
    }


def augment_image_and_mask(image, mask):
    import random    
    # Random brightness adjustment
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
    
    # Random contrast adjustment
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        mean = image.mean()
        image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    return image, mask


def finetune_sam(sam, train_data, device, num_epochs=10):
    print(f"\nFine-tuning SAM...")
    
    # Freeze image encoder, unfreeze mask decoder
    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    predictor = SamPredictor(sam)
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_samples = 0
        
        for item in tqdm(train_data, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Load image and ground truth
            image = cv2.imread(item['image_path'])
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gt_mask = load_ground_truth_mask(item['mask_path'], item['category_id'])
            
            # Train on BOTH original and augmented version
            versions = [
                (image_rgb.copy(), gt_mask.copy()),  # Original
                augment_image_and_mask(image_rgb.copy(), gt_mask.copy())  # Augmented
            ]
            
            for img, mask in versions:
                # Convert to boolean then float tensor
                gt_mask_bool = mask.astype(bool)
                gt_mask_tensor = torch.from_numpy(gt_mask_bool).float().to(device)
                
                # Set image
                predictor.set_image(img)
                
                # Get bbox
                bbox = convert_bbox_format(item['bbox'])
                
                # Get features from predictor (need to access internal state)
                coords_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)
                coords_torch = coords_torch[None, None, :]  # (1, 1, 4)
                
                # Transform box coordinates
                coords_torch = predictor.transform.apply_boxes_torch(coords_torch, image_rgb.shape[:2])
                
                # Get sparse and dense embeddings
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=None,
                    boxes=coords_torch,
                    masks=None,
                )
                
                # Predict mask
                low_res_masks, _ = sam.mask_decoder(
                    image_embeddings=predictor.features,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                # Upsample predictions to original size
                masks = torch.nn.functional.interpolate(
                    low_res_masks,
                    size=mask.shape,
                    mode='bilinear',
                    align_corners=False
                )
                masks = masks.squeeze(1)  # Remove channel dimension
                
                # Compute loss
                loss = criterion(masks[0], gt_mask_tensor)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_samples += 1
        
        avg_loss = total_loss / num_samples
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
    
    print("Fine-tuning complete!")
    return sam


def main():
    DATA_DIR = '/home/3182217/segmentation_cv/EcoEyeSpy/segmentation_datasets'
    SAM_CHECKPOINT = '/home/3182217/segmentation_cv/EcoEyeSpy/fewshot_training/sam_vit_b_01ec64.pth'
    MODEL_TYPE = 'vit_b'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_TRAIN_EXAMPLES = 64
    
    # Create timestamped output directory in fewshot_training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = f'/home/3182217/segmentation_cv/EcoEyeSpy/fewshot_training/sam_training_{timestamp}'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'visualizations'), exist_ok=True)
    
    print("="*60)
    print("SAM FEW-SHOT FINE-TUNING & PREDICTION")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"SAM Model: {MODEL_TYPE}")
    print(f"Training examples: {NUM_TRAIN_EXAMPLES}")
    print(f"Output: {OUTPUT_DIR}")
    
    print("\nLoading SAM model...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    print("SAM model loaded successfully!")
    
    print("\nLoading dataset...")
    bbox_json = os.path.join(DATA_DIR, 'patches_bbox.json')
    labels_json = os.path.join(DATA_DIR, 'patches_labels.json')
    patches_dir = os.path.join(DATA_DIR, 'patches')
    masks_dir = os.path.join(DATA_DIR, 'mask')
    
    data = load_data(bbox_json, labels_json, patches_dir, masks_dir)
    print(f"Found {len(data)} images with label 1")
    
    # Split into train and test
    train_data = data[:NUM_TRAIN_EXAMPLES]
    test_data = data[NUM_TRAIN_EXAMPLES:]
        
    # Fine-tune SAM
    sam = finetune_sam(sam, train_data, DEVICE)
    
    # Create predictor with fine-tuned model
    predictor = SamPredictor(sam)
    print(f"\nPredicting on {len(test_data)} test images...")
    
    # Process images
    results = []
    metrics_summary = {
        'total': len(test_data),
        'ious': [],
        'dices': [],
        'precisions': [],
        'recalls': [],
        'f1_scores': [],
        'boundary_ious': [],
        'hausdorff_distances': [],
        'boundary_maes': []
    }
    
    print("\nProcessing test images...")
    for item in tqdm(test_data):
        try:
            # Load image
            image = cv2.imread(item['image_path'])
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Set image for predictor
            predictor.set_image(image_rgb)
            
            # Convert bbox format
            bbox = convert_bbox_format(item['bbox'])
            
            # Predict masks
            masks, scores, logits = predictor.predict(
                box=bbox,
                multimask_output=True
            )
            
            # Select best mask (highest score)
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]
            
            # Load ground truth mask using the actual category_id
            gt_mask = load_ground_truth_mask(item['mask_path'], item['category_id'])
            
            # Evaluate prediction
            metrics = evaluate_prediction(best_mask, gt_mask)
            
            # Store results
            result = {
                'file_name': item['file_name'],
                'bbox': item['bbox'],
                'sam_score': float(best_score),
                'iou': metrics['iou'],
                'dice': metrics['dice'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'boundary_iou': metrics['boundary_iou'],           # ✅ Now saved
                'hausdorff_distance': metrics['hausdorff_distance'], # ✅ Now saved
                'boundary_mae': metrics['boundary_mae']            # ✅ Now saved
            }
            results.append(result)
            
            # Update metrics summary
            metrics_summary['ious'].append(metrics['iou'])
            metrics_summary['dices'].append(metrics['dice'])
            metrics_summary['precisions'].append(metrics['precision'])
            metrics_summary['recalls'].append(metrics['recall'])
            metrics_summary['f1_scores'].append(metrics['f1_score'])
            metrics_summary['boundary_ious'].append(metrics['boundary_iou'])
            metrics_summary['hausdorff_distances'].append(metrics['hausdorff_distance'])
            metrics_summary['boundary_maes'].append(metrics['boundary_mae'])
            
            # Save predicted mask
            mask_save_path = os.path.join(OUTPUT_DIR, 'masks', item['file_name'])
            cv2.imwrite(mask_save_path, (best_mask * 255).astype(np.uint8))
            
            # Create visualization (overlay mask on image)
            vis_image = image_rgb.copy()
            # Create colored mask overlay (green)
            mask_overlay = np.zeros_like(vis_image)
            mask_overlay[best_mask] = [0, 255, 0]
            vis_image = cv2.addWeighted(vis_image, 0.7, mask_overlay, 0.3, 0)
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Add metrics text
            text = f"IoU: {metrics['iou']:.3f} | Score: {best_score:.3f}"
            cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            # Save visualization
            vis_save_path = os.path.join(OUTPUT_DIR, 'visualizations', item['file_name'])
            cv2.imwrite(vis_save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f"\nError processing {item['file_name']}: {e}")
            continue
    
    # Calculate average metrics
    avg_iou = np.mean(metrics_summary['ious'])
    avg_dice = np.mean(metrics_summary['dices'])
    avg_precision = np.mean(metrics_summary['precisions'])
    avg_recall = np.mean(metrics_summary['recalls'])
    avg_f1 = np.mean(metrics_summary['f1_scores'])
    avg_boundary_iou = np.mean(metrics_summary['boundary_ious'])
    avg_hausdorff = np.mean(metrics_summary['hausdorff_distances'])
    avg_boundary_mae = np.mean(metrics_summary['boundary_maes'])
    std_iou = np.std(metrics_summary['ious'])
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION METRICS SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(results)}")
    print(f"\nAverage IoU: {avg_iou:.4f} (±{std_iou:.4f})")
    print(f"Average Dice: {avg_dice:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")
    print(f"\nAverage Boundary IoU: {avg_boundary_iou:.4f}")
    print(f"Average Hausdorff Distance: {avg_hausdorff:.4f}")
    print(f"Average Boundary MAE: {avg_boundary_mae:.4f}")
    print(f"\nMedian IoU: {np.median(metrics_summary['ious']):.4f}")
    print(f"Min IoU: {np.min(metrics_summary['ious']):.4f}")
    print(f"Max IoU: {np.max(metrics_summary['ious']):.4f}")
    
    # Save results to JSON
    results_path = os.path.join(OUTPUT_DIR, 'predictions_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'summary': {
                'total_images': len(results),
                'avg_iou': avg_iou,
                'avg_dice': avg_dice,
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f1_score': avg_f1,
                'avg_boundary_iou': avg_boundary_iou,
                'avg_hausdorff_distance': avg_hausdorff,
                'avg_boundary_mae': avg_boundary_mae,
                'std_iou': std_iou,
                'median_iou': float(np.median(metrics_summary['ious'])),
                'min_iou': float(np.min(metrics_summary['ious'])),
                'max_iou': float(np.max(metrics_summary['ious']))
            },
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Masks saved to: {os.path.join(OUTPUT_DIR, 'masks')}")
    print(f"Visualizations saved to: {os.path.join(OUTPUT_DIR, 'visualizations')}")
    print("="*60)


if __name__ == "__main__":
    main()