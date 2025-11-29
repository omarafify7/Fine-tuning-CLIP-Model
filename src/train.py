import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
from tqdm import tqdm
import os
import json

from dataset import CocoDataset, get_transforms
from model import CLIPModel
import util

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for images, text_inputs in pbar:
        images = images.to(device)
        text_inputs = text_inputs.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        image_features, text_features = model(images, text_inputs)
        
        # InfoNCE Loss
        # Logits: (batch_size, batch_size)
        # We need to scale logits by temperature (usually learnable, but outline doesn't specify)
        # Standard CLIP uses a learnable temperature. I'll add a fixed one or learnable if needed.
        # For simplicity, let's use a fixed temperature of 0.07 (common default)
        logit_scale = torch.tensor(1/0.07).to(device) 
        
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        
        batch_size = images.shape[0]
        labels = torch.arange(batch_size).to(device)
        
        loss_i = nn.functional.cross_entropy(logits_per_image, labels)
        loss_t = nn.functional.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch} Train Loss: {epoch_loss:.4f} | Time: {time.time() - start_time:.2f}s")
    return epoch_loss

def validate(model, dataloader, device, epoch):
    model.eval()
    running_loss = 0.0
    start_time = time.time()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for images, text_inputs in pbar:
            images = images.to(device)
            text_inputs = text_inputs.to(device)
            
            image_features, text_features = model(images, text_inputs)
            
            logit_scale = torch.tensor(1/0.07).to(device)
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()
            
            batch_size = images.shape[0]
            labels = torch.arange(batch_size).to(device)
            
            loss_i = nn.functional.cross_entropy(logits_per_image, labels)
            loss_t = nn.functional.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch} Val Loss: {epoch_loss:.4f} | Time: {time.time() - start_time:.2f}s")
    return epoch_loss

def main():
    parser = argparse.ArgumentParser(description="Train CLIP Model")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--output_dir', type=str, default='../output', help='Directory to save outputs (plots, metrics)')
    parser.add_argument('--data_dir', type=str, default='../coco2014', help='Root directory of COCO dataset')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (subset of data)')
    args = parser.parse_args()
    
    device = util.get_device()
    print(f"Using device: {device}")
    
    # Dataset Paths
    train_dir = os.path.join(args.data_dir, 'images', 'train2014')
    val_dir = os.path.join(args.data_dir, 'images', 'val2014')
    train_ann = os.path.join(args.data_dir, 'annotations', 'captions_train2014.json')
    val_ann = os.path.join(args.data_dir, 'annotations', 'captions_val2014.json')
    
    # Datasets
    print("Initializing Datasets...")
    train_dataset = CocoDataset(train_dir, train_ann, transform=get_transforms(is_train=True))
    val_dataset = CocoDataset(val_dir, val_ann, transform=get_transforms(is_train=False))
    
    if args.debug:
        print("Debug mode: Using subset of data (100 samples)")
        indices = torch.arange(100)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    # Model
    print("Initializing Model...")
    model = CLIPModel().to(device)
    
    # Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Resume Logic
    start_epoch = 1
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Load metrics if exist
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        print(f"Loading metrics from {metrics_path}...")
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            train_losses = metrics.get('train_losses', [])
            val_losses = metrics.get('val_losses', [])
            best_val_loss = metrics.get('best_val_loss', float('inf'))
            start_epoch = len(train_losses) + 1
            print(f"Resuming from epoch {start_epoch}")
    
    # Load Checkpoint if resuming
    # Try to load the latest checkpoint if it exists
    latest_checkpoint = os.path.join(args.checkpoint_dir, 'checkpoint_last.pth')
    if os.path.exists(latest_checkpoint):
        print(f"Loading latest checkpoint from {latest_checkpoint}...")
        checkpoint = util.load_checkpoint(latest_checkpoint, model, optimizer, device=device)
        # We can also load start_epoch from checkpoint if we want to be safe
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Checkpoint says next epoch is {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device, epoch)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Checkpointing
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            
        # Save latest checkpoint
        util.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, args.checkpoint_dir, is_best=is_best, filename='checkpoint_last.pth')
        
        # Save every 5 epochs to subfolder
        if epoch % 5 == 0:
            epoch_dir = os.path.join(args.checkpoint_dir, f'epoch_{epoch}')
            util.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, epoch_dir, is_best=False, filename=f'checkpoint_epoch_{epoch}.pth')

        # Save Metrics
        os.makedirs(args.output_dir, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }, f)
        
        # Plotting
        util.plot_training_metrics(train_losses, val_losses, save_dir=args.output_dir)
        
    print("Training Complete!")

if __name__ == "__main__":
    main()
