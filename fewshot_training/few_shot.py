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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))


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


def evaluate_prediction(pred_mask, gt_mask):
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)
    intersection = (pred_bool & gt_bool).sum()
    union = (pred_bool | gt_bool).sum()
    iou = intersection / union if union > 0 else 0.0
    precision = intersection / pred_bool.sum() if pred_bool.sum() > 0 else 0.0
    recall = intersection / gt_bool.sum() if gt_bool.sum() > 0 else 0.0
    return {'iou': iou, 'precision': precision, 'recall': recall}


def augment_image_and_mask(image, mask):
    import random
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        mean = image.mean()
        image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    return image, mask


def finetune_sam(sam, train_data, device, num_epochs=10):
    print(f"\nFine-tuning SAM...")
    
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
            image = cv2.imread(item['image_path'])
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gt_mask = load_ground_truth_mask(item['mask_path'], item['category_id'])
            
            versions = [
                (image_rgb.copy(), gt_mask.copy()),
                augment_image_and_mask(image_rgb.copy(), gt_mask.copy())
            ]
            
            for img, mask in versions:
                gt_mask_tensor = torch.from_numpy(mask.astype(bool)).float().to(device)
                predictor.set_image(img)
                bbox = convert_bbox_format(item['bbox'])
                coords_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)[None, None, :]
                coords_torch = predictor.transform.apply_boxes_torch(coords_torch, image_rgb.shape[:2])
                
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=None,
                    boxes=coords_torch,
                    masks=None,
                )
                
                low_res_masks, _ = sam.mask_decoder(
                    image_embeddings=predictor.features,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                masks = torch.nn.functional.interpolate(
                    low_res_masks,
                    size=mask.shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                
                loss = criterion(masks[0], gt_mask_tensor)
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
    
    results = []
    ious = []
    precisions = []
    recalls = []
    
    print("\nProcessing test images...")
    for item in tqdm(test_data):
        image = cv2.imread(item['image_path'])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        bbox = convert_bbox_format(item['bbox'])
        
        masks, scores, logits = predictor.predict(
            box=bbox,
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        gt_mask = load_ground_truth_mask(item['mask_path'], item['category_id'])
        metrics = evaluate_prediction(best_mask, gt_mask)
        
        result = {
            'file_name': item['file_name'],
            'iou': metrics['iou'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        }
        results.append(result)
        ious.append(metrics['iou'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        
        mask_save_path = os.path.join(OUTPUT_DIR, 'masks', item['file_name'])
        cv2.imwrite(mask_save_path, (best_mask * 255).astype(np.uint8))
        
        vis_image = image_rgb.copy()
        mask_overlay = np.zeros_like(vis_image)
        mask_overlay[best_mask] = [0, 255, 0]
        vis_image = cv2.addWeighted(vis_image, 0.7, mask_overlay, 0.3, 0)
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = f"IoU: {metrics['iou']:.3f}"
        cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        vis_save_path = os.path.join(OUTPUT_DIR, 'visualizations', item['file_name'])
        cv2.imwrite(vis_save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    avg_iou = np.mean(ious)
    std_iou = np.std(ious)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"total_images: {len(results)}")
    print(f"avg_iou: {avg_iou:.10f}")
    print(f"std_iou: {std_iou:.10f}")
    print(f"avg_precision: {avg_precision:.10f}")
    print(f"avg_recall: {avg_recall:.10f}")
    print(f"median_iou: {np.median(ious):.10f}")
    print(f"min_iou: {np.min(ious):.10f}")
    print(f"max_iou: {np.max(ious):.10f}")
    print("="*70)
    
    results_path = os.path.join(OUTPUT_DIR, 'predictions_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'summary': {
                'total_images': len(results),
                'avg_iou': avg_iou,
                'std_iou': std_iou,
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'median_iou': float(np.median(ious)),
                'min_iou': float(np.min(ious)),
                'max_iou': float(np.max(ious))
            },
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Masks saved to: {os.path.join(OUTPUT_DIR, 'masks')}")
    print(f"Visualizations saved to: {os.path.join(OUTPUT_DIR, 'visualizations')}")


if __name__ == "__main__":
    main()