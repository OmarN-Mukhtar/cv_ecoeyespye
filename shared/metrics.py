import os
import csv
import json
import numpy as np

def save_metrics_to_csv(train_losses, val_losses, val_accuracies, val_ious, output_dir, logger=None):
    """Save metrics to CSV for easy analysis"""
    try:
        import pandas as pd
        metrics_df = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies,
            'val_iou': val_ious
        })
        csv_path = os.path.join(output_dir, "training_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)
    except ImportError:
        # Fallback to manual CSV writing if pandas not available
        csv_path = os.path.join(output_dir, "training_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_iou'])
            for i in range(len(train_losses)):
                writer.writerow([i+1, train_losses[i], val_losses[i], val_accuracies[i], val_ious[i]])
    
    if logger:
        logger.log(f"Metrics saved to {csv_path}")
    return csv_path