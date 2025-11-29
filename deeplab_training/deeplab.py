import os
import sys
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.models.segmentation import deeplabv3_resnet50

# Add parent and shared directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

from make_dataset import SegmentationDataset
from logger import Logger
from train import train_epoch
from val import validate_epoch
from test import test_and_save_predictions
from metrics import save_metrics_to_csv
from plots import plot_training_results


def setup_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_category_ids(base_dir):
    labels_path = os.path.join(base_dir, 'segmentation_datasets', 'patches_labels.json')
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    # Get all patches with their category IDs
    all_patches = labels_data['patches']
    
    # Filter out patches with category_id == 0
    filtered_patches = [p for p in all_patches if p['category_id'] != 0]
    
    # Extract category IDs and image names
    category_ids = np.array([p['category_id'] for p in filtered_patches])
    image_names = [p['file_name'] for p in filtered_patches]
    
    return category_ids, image_names


def create_stratified_splits(full_dataset, category_ids, train_frac, test_frac):
    val_frac = 1.0 - train_frac - test_frac
    indices = np.arange(len(full_dataset))
    
    # Train+val vs test split
    train_val_idx, test_idx = train_test_split(indices, test_size=test_frac, stratify=category_ids, random_state=42)
    
    # Train vs val split
    train_val_cats = category_ids[train_val_idx]
    val_size = val_frac / (train_frac + val_frac)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size, stratify=train_val_cats, random_state=42)
    
    return (Subset(full_dataset, train_idx), Subset(full_dataset, val_idx), Subset(full_dataset, test_idx))


def setup_model(num_classes, device, atrous_rates=None):
    """Setup DeepLabv3 with optional custom ASPP dilation rates"""
    from torchvision.models.segmentation.deeplabv3 import ASPP
    
    # Load pretrained model with original 21 classes
    model = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
    
    if atrous_rates is not None:
        # Replace ASPP with custom dilation rates
        model.classifier = nn.Sequential(
            ASPP(in_channels=2048, atrous_rates=atrous_rates, out_channels=256),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
    else:
        # Keep pretrained ASPP, just change final layer
        in_channels = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    
    # Replace aux_classifier if it exists
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        aux_in_channels = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)
    
    model.to(device)
    
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier heads
    for param in model.classifier.parameters():
        param.requires_grad = True
    if hasattr(model, 'aux_classifier'):
        for param in model.aux_classifier.parameters():
            param.requires_grad = True
    
    return model


def find_best_dilations(train_loader, val_loader, device, num_classes, logger):
    """Grid search to find optimal ASPP dilation rates"""
    
    dilation_configs = [
        [3, 6, 12],   # Optimized for 256x256
        [6, 12, 18],  # Default DeepLab
        [2, 4, 8],    # Small objects
        [4, 8, 16],   # Alternative
    ]
    
    best_iou = 0
    best_config = None
    results = []
    
    for rates in dilation_configs:
        logger.log(f"\nTesting dilation rates: {rates}")
        
        model = setup_model(num_classes, device, atrous_rates=rates)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(3):
            train_epoch(model, train_loader, criterion, optimizer, device)
        
        _, _, val_iou = validate_epoch(model, val_loader, criterion, device, num_classes, logger)
        
        results.append({'rates': rates, 'val_iou': val_iou})
        logger.log(f"  Validation IoU: {val_iou:.4f}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            best_config = rates
        
        del model, optimizer, criterion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.log(f"\nBest dilation config: {best_config} (IoU: {best_iou:.4f})")
    return best_config or [6, 12, 18]


def unfreeze_backbone_layer(model, layer_name):
    layer = getattr(model.backbone, layer_name)
    for param in layer.parameters():
        param.requires_grad = True


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                config, output_dir, logger, unique_categories):
    train_losses, val_losses, val_accuracies, val_ious = [], [], [], []
    best_val_iou = 0.0
    
    layers_to_unfreeze = ['layer4', 'layer3', 'layer2', 'layer1']
    logger.log("Starting training...")
    
    for epoch in range(config['num_epochs']):
        # Progressive unfreezing every 10 epochs
        if (epoch + 1) % 10 == 0 and (epoch + 1) // 10 <= len(layers_to_unfreeze):
            layer_idx = (epoch + 1) // 10 - 1
            layer_name = layers_to_unfreeze[layer_idx]
            unfreeze_backbone_layer(model, layer_name)
            logger.log(f"Epoch {epoch+1}: Unfroze {layer_name}")
            
            backbone_lr = config['learning_rate'] * 0.1
            optimizer.param_groups.clear()
            optimizer.add_param_group({
                'params': filter(lambda p: p.requires_grad, model.backbone.parameters()),
                'lr': backbone_lr
            })
            optimizer.add_param_group({
                'params': model.classifier.parameters(),
                'lr': config['learning_rate']
            })
            if hasattr(model, 'aux_classifier'):
                optimizer.add_param_group({
                    'params': model.aux_classifier.parameters(),
                    'lr': config['learning_rate']
                })
        
        # Train and validate
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config['device'])
        val_loss, val_acc, val_iou = validate_epoch(model, val_loader, criterion, config['device'], config['num_classes'], logger, unique_categories)
        
        scheduler.step(val_loss)
        
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_ious.append(val_iou)
        
        logger.log(f"Epoch {epoch+1}/{config['num_epochs']} - Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'config': config
            }, os.path.join(output_dir, "models", "best_model.pth"))
            logger.log(f"  New best IoU: {val_iou:.4f}")
        
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config
            }, os.path.join(output_dir, "models", f"checkpoint_epoch_{epoch+1}.pth"))
    
    return train_losses, val_losses, val_accuracies, val_ious, best_val_iou


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, f"training_results_{timestamp}")
    for subdir in ["plots", "models", "logs"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = setup_device()
    logger = Logger(os.path.join(output_dir, "logs", "training.log"))
    
    # Load data - go up one level to access segmentation_datasets
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(base_dir, 'segmentation_datasets', 'patches')
    masks_dir = os.path.join(base_dir, 'segmentation_datasets', 'mask')
    
    category_ids, image_names = load_category_ids(base_dir)
    
    unique_categories = sorted(np.unique(category_ids))
    num_classes = len(unique_categories)
    sparse_to_dense = {cat_id: idx for idx, cat_id in enumerate(unique_categories)}
    
    logger.log(f"Loaded {len(category_ids)} images, {num_classes} categories")
    
    # Configuration
    config = {
        'num_epochs': 50,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'train_split': 0.8,
        'test_split': 0.1,
        'num_classes': num_classes,
        'device': device,
        'timestamp': timestamp
    }
    
    logger.log(f"Config: {config['num_epochs']} epochs, batch_size={config['batch_size']}, lr={config['learning_rate']}")
    
    # Create filtered dataset WITH label mapping
    full_dataset = SegmentationDataset(images_dir, masks_dir, 
                                       label_mapping=sparse_to_dense, 
                                       augment=False,
                                       image_list=image_names)
    
    # Create stratified splits using raw category IDs
    train_dataset_base, val_dataset, test_dataset = create_stratified_splits(
        full_dataset, category_ids, config['train_split'], config['test_split']
    )
    
    # Get train indices
    train_indices = train_dataset_base.indices
    
    # Create val/test loaders (for grid search and evaluation - no augmentation)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create temporary train loader for grid search (original images only)
    train_loader_grid_search = DataLoader(train_dataset_base, batch_size=config['batch_size'], shuffle=True)
    
    logger.log("Auto-tuning ASPP dilation rates...")
    best_dilations = find_best_dilations(train_loader_grid_search, val_loader, device, config['num_classes'], logger)
    config['atrous_rates'] = best_dilations
    
    # NOW create augmented training set for final training
    train_dataset_original = Subset(full_dataset, train_indices)
    
    train_dataset_aug_full = SegmentationDataset(images_dir, masks_dir,
                                                  label_mapping=sparse_to_dense,
                                                  augment=True,
                                                  image_list=image_names)
    train_dataset_augmented = Subset(train_dataset_aug_full, train_indices)
    
    # Combine original + augmented (doubles training set size)
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset([train_dataset_original, train_dataset_augmented])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    logger.log(f"Stratified split: Train={len(train_dataset)} (original+augmented), Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Save config
    config['unique_categories'] = [int(c) for c in unique_categories]
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump({k: str(v) if isinstance(v, torch.device) else v for k, v in config.items()}, f, indent=2)
    
    model = setup_model(config['num_classes'], device, atrous_rates=best_dilations)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    train_losses, val_losses, val_accuracies, val_ious, best_val_iou = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, config, output_dir, logger, unique_categories
    )
    
    logger.log(f"\nTraining completed! Best IoU: {best_val_iou:.4f}")
    
    # Save results
    save_metrics_to_csv(train_losses, val_losses, val_accuracies, val_ious, output_dir, logger)
    plot_training_results(train_losses, val_losses, val_accuracies, val_ious, output_dir, logger)
    test_and_save_predictions(model, test_loader, criterion, device, config, output_dir, logger, unique_categories)
    
    # Summary
    summary = {
        'training_completed': True,
        'best_val_iou': best_val_iou,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_val_accuracy': val_accuracies[-1],
        'total_epochs': len(train_losses),
        'config': {k: str(v) if isinstance(v, torch.device) else v for k, v in config.items()}
    }
    
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.log(f"\nAll results saved to: {output_dir}")
    logger.log("Training complete!")


if __name__ == "__main__":
    main()
