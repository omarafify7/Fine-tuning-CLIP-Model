import torch
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# Device Configuration
# ==============================================================================

def get_device(verbose=True):
    """
    Get the available device, prioritizing MPS (Apple Silicon) > CUDA > CPU.
    
    Args:
        verbose (bool): If True, prints the selected device.
        
    Returns:
        torch.device: The selected device.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("Success: MPS (Metal Performance Shaders) is available! Using Apple Silicon GPU.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"Success: CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("MPS/CUDA not available. Using CPU.")
            
    return device

# ==============================================================================
# Checkpointing Functions
# ==============================================================================

def save_checkpoint(state, checkpoint_dir, is_best=False, filename='checkpoint.pth', best_filename='model_best.pth'):
    """
    Save a training checkpoint.

    Args:
        state (dict): The state dictionary to save. 
                      Recommended structure:
                      {
                          'epoch': epoch,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict(), # Optional
                          'best_metric': best_metric,          # Optional
                      }
        checkpoint_dir (str): Directory to save the checkpoint in.
        is_best (bool): If True, copies the saved checkpoint to a 'best' file.
        filename (str): Name of the checkpoint file.
        best_filename (str): Name of the best checkpoint file.
    
    NOTE: You may want to add logic to save only the last N checkpoints to save space.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, best_filename)
        shutil.copyfile(filepath, best_path)
        print(f"Saved new best model to {best_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Load a checkpoint and resume the model/optimizer state.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        device (str or torch.device): Device to map the checkpoint to.

    Returns:
        dict: The full checkpoint dictionary (useful for extracting epoch, best_metric, etc.)
    
    NOTE: Ensure that the model architecture exactly matches the checkpoint. 
          If you changed the model, you might need strict=False in load_state_dict.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
        
    print(f"Loading checkpoint from '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    # NOTE: If your checkpoint saves the model under a different key (e.g. 'model_state_dict'), change this.
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Fallback: try loading the whole dict if it's just the state dict
        try:
            model.load_state_dict(checkpoint)
        except:
            print("Warning: Could not find 'state_dict' or 'model_state_dict' key. Inspect checkpoint structure.")

    # Load optimizer state
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    elif optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    # Load scheduler state
    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    elif scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint

# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_training_metrics(train_losses, val_losses, train_metrics=None, val_metrics=None, 
                          metric_label='Accuracy', save_dir='.', filename='training_plot.png'):
    """
    Plot training and validation loss and a secondary metric (e.g., Accuracy, mIoU).

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        train_metrics (list, optional): List of training metric values per epoch.
        val_metrics (list, optional): List of validation metric values per epoch.
        metric_label (str): Label for the secondary metric (y-axis label).
        save_dir (str): Directory to save the plot.
        filename (str): Filename for the plot.
    
    NOTE: This function assumes one value per epoch. If you have batch-level data, 
          average it before passing here.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    has_metrics = train_metrics is not None and val_metrics is not None
    
    if has_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        loss_ax = axes[0]
        metric_ax = axes[1]
    else:
        fig, loss_ax = plt.subplots(1, 1, figsize=(8, 5))
        metric_ax = None

    # Plot Loss
    epochs = range(1, len(train_losses) + 1)
    loss_ax.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
    loss_ax.plot(epochs, val_losses, label='Val Loss', marker='s', markersize=3)
    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylabel('Loss')
    loss_ax.set_title('Training and Validation Loss')
    loss_ax.legend()
    loss_ax.grid(True, alpha=0.3)

    # Plot Metric
    if has_metrics:
        metric_ax.plot(epochs, train_metrics, label=f'Train {metric_label}', marker='o', markersize=3)
        metric_ax.plot(epochs, val_metrics, label=f'Val {metric_label}', marker='s', markersize=3)
        metric_ax.set_xlabel('Epoch')
        metric_ax.set_ylabel(metric_label)
        metric_ax.set_title(f'Training and Validation {metric_label}')
        metric_ax.legend()
        metric_ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training plot to {save_path}")
