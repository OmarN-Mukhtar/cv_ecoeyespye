import os
import numpy as np
import matplotlib
matplotlib.use('Agg')            # non-interactive backend for HPC
import matplotlib.pyplot as plt

def plot_training_results(train_losses, val_losses, val_accuracies, val_ious, output_dir, logger=None):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, train_losses, label='Train Loss')
    axes[0, 0].plot(epochs, val_losses, label='Val Loss')
    axes[0, 0].set_title('Loss Over Time')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(epochs, val_accuracies)
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Pixel Accuracy')
    axes[0, 1].grid(True)
    
    # IoU
    axes[1, 0].plot(epochs, val_ious)
    axes[1, 0].set_title('Validation IoU')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Mean IoU')
    axes[1, 0].grid(True)
    
    # Combined
    ax = axes[1, 1]
    ax2 = ax.twinx()
    
    l1 = ax.plot(epochs, val_losses, 'b-', label='Val Loss')
    l2 = ax2.plot(epochs, val_ious, 'r-', label='Val IoU')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color='b')
    ax2.set_ylabel('IoU', color='r')
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')
    ax.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "plots", "training_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to save memory
    if logger:
        logger.log(f"Training plots saved to {plot_path}")
    return plot_path
