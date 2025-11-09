import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from make_dataset import SegmentationDataset
from label_map import label_map
# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# Configuration
CONFIG = {
    'num_epochs': 50,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'train_split': 0.8,
    'num_classes': 21
}

# Dataset and DataLoader
full_dataset = SegmentationDataset(
    '/Users/omar/Downloads/segmentation_datasets/patches', 
    '/Users/omar/Downloads/segmentation_datasets/mask'
)

# Split dataset
train_size = int(CONFIG['train_split'] * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# Model setup
model = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1', num_classes=CONFIG['num_classes'])
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Metrics tracking
train_losses = []
val_losses = []
val_accuracies = []
val_ious = []

def calculate_iou(pred, target, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection = (pred_cls & target_cls).sum().float().item()
        union = (pred_cls | target_cls).sum().float().item()
        
        if union == 0:
            ious.append(float('nan'))  # Ignore classes not present
        else:
            ious.append(intersection / union)
    
    # Return mean IoU (ignoring nan values)
    ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(ious) if ious else 0.0

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)['out']  
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device, num_classes):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    total_iou = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.permute(0, 3, 1, 2).float().to(device)
            masks = masks.long().to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            predictions = torch.argmax(outputs, dim=1)
            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()
            
            # Calculate IoU
            iou = calculate_iou(predictions, masks, num_classes)
            total_iou += iou
            
            running_loss += loss.item()
            
            accuracy = correct_pixels / total_pixels
            avg_iou = total_iou / (len(pbar.iterable) if hasattr(pbar, 'iterable') else 1)
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.4f}',
                'IoU': f'{avg_iou:.4f}'
            })
    
    avg_loss = running_loss / len(dataloader)
    pixel_accuracy = correct_pixels / total_pixels
    mean_iou = total_iou / len(dataloader)
    
    return avg_loss, pixel_accuracy, mean_iou

# Training loop
print("Starting training...")
best_val_iou = 0.0

for epoch in range(CONFIG['num_epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
    print("-" * 50)
    
    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_acc, val_iou = validate_epoch(model, val_loader, criterion, device, CONFIG['num_classes'])
    
    # Update scheduler
    scheduler.step(val_loss)
    
    # Save metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    val_ious.append(val_iou)
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Val IoU: {val_iou:.4f}")
    
    # Save best model
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_iou': val_iou,
            'val_accuracy': val_acc,
            'config': CONFIG
        }, 'best_deeplab_model.pth')
        print(f"🎉 New best model saved! IoU: {val_iou:.4f}")

print(f"\nTraining completed! Best IoU: {best_val_iou:.4f}")

# Plot results
def plot_training_results():
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Val Loss')
    axes[0, 0].set_title('Loss Over Time')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(val_accuracies)
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Pixel Accuracy')
    
    # IoU
    axes[1, 0].plot(val_ious)
    axes[1, 0].set_title('Validation IoU')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Mean IoU')
    
    # Combined
    ax = axes[1, 1]
    ax2 = ax.twinx()
    
    l1 = ax.plot(val_losses, 'b-', label='Val Loss')
    l2 = ax2.plot(val_ious, 'r-', label='Val IoU')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color='b')
    ax2.set_ylabel('IoU', color='r')
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300)
    plt.show()

# Plot results
plot_training_results()

# Test model performance
def test_model():
    """Test the trained model"""
    model.eval()
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            if i >= 5:  # Show first 5 examples
                break
                
            image = image.permute(0, 3, 1, 2).float().to(device)
            mask = mask.long().to(device)
            
            output = model(image)['out']
            prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
            
            # Visualize
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
            plt.title('Input Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(mask[0].cpu().numpy(), cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(prediction, cmap='gray')
            plt.title('Prediction')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()

# Run tests
print("\nTesting model...")
test_model()